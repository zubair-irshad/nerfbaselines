# pylint: disable=import-outside-toplevel
# pylint: disable=protected-access
import os
from functools import partial
import logging
from dataclasses import fields
from pathlib import Path
import copy
import tempfile
from collections import defaultdict
from typing import Iterable, Optional
import numpy as np
from ..types import Method, ProgressCallback, CurrentProgress, MethodInfo
from ..types import Dataset
from ..registry import MethodSpec
from ..distortion import Distortions, CameraModel
from ..backends.docker import DockerMethod
from ..backends.conda import CondaMethod
from ..utils import cached_property


# Hack to add progress to existing models
def _hacked_get_outputs_for_camera_ray_bundle(self, camera_ray_bundle, update_callback: Optional[callable] = None):
    import torch

    with torch.no_grad():
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                outputs_lists[output_name].append(output.to(input_device))
            if update_callback:
                update_callback(min(num_rays, i + num_rays_per_chunk), num_rays)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs


class NerfStudio(Method):
    nerfstudio_name: Optional[str] = None
    _dataparser_transform = None
    _dataparser_scale = None

    def __init__(self, nerfstudio_name: Optional[str] = None, checkpoint: str = None):
        self.checkpoint = checkpoint
        self.nerfstudio_name = nerfstudio_name or self.nerfstudio_name
        if checkpoint is not None:
            import yaml
            # Load nerfstudio checkpoint
            with open(os.path.join(checkpoint, "config.yml"), "r", encoding="utf8") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            self._original_config = copy.deepcopy(config)
            config.get_base_dir = lambda *_: Path(checkpoint)
            config.load_dir = config.get_checkpoint_dir()
        elif self.nerfstudio_name is not None:
            from nerfstudio.configs.method_configs import method_configs
            config = method_configs[self.nerfstudio_name]
            self._original_config = copy.deepcopy(config)
        else:
            raise ValueError("Either checkpoint or name must be provided")
        self.config = copy.deepcopy(config)
        super().__init__(batch_size=self.config.pipeline.datamanager.train_num_rays_per_batch)
        self._trainer = None
        self._dm = None
        self.step = 0
        self._tmpdir = tempfile.TemporaryDirectory()
        self._mode = None

    @property
    def batch_size(self):
        return self.config.pipeline.datamanager.train_num_rays_per_batch

    @cached_property
    def info(self) -> MethodInfo:
        info = MethodInfo(
            loaded_step=None,
            num_iterations=self.config.max_num_iterations,
            required_features=["images"],
            supports_undistortion=True,
            batch_size=self.config.pipeline.datamanager.train_num_rays_per_batch,
            eval_batch_size=self.config.pipeline.model.eval_num_rays_per_chunk)
        if self.checkpoint is not None:
            model_path = os.path.join(self.checkpoint, self.config.relative_model_dir)
            if not os.path.exists(model_path):
                raise RuntimeError(f"Model directory {model_path} does not exist")
            info.loaded_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(model_path))[-1]
        return info

    def render(self,
               poses: np.ndarray,
               intrinsics: np.ndarray,
               sizes: np.ndarray,
               nears_fars: np.ndarray,
               distortions: Optional[Distortions] = None,
               progress_callback: Optional[ProgressCallback] = None) -> Iterable[np.ndarray]:
        if self._mode is None:
            self._setup_eval()
        from nerfstudio.cameras.cameras import Cameras, CameraType as NPCameraType
        from nerfstudio.models.base_model import Model
        import torch
        poses = torch.from_numpy(poses)
        assert poses.dim() == 3
        poses = self._transform_poses(poses)
        intrinsics = torch.from_numpy(intrinsics)
        camera_types = [NPCameraType.PERSPECTIVE for _ in range(len(poses))]
        if distortions is not None:
            npmap = {x.name.lower(): x.value for x in NPCameraType.__members__.values()}
            npmap["pinhole"] = npmap["perspective"]
            npmap["opencv"] = npmap["perspective"]
            npmap["opencv_fisheye"] = npmap["fisheye"]
            camera_types = [npmap[CameraModel(distortions.camera_types[i]).name.lower()] for i in range(len(poses))]
        cameras = Cameras(
            camera_to_worlds=poses.contiguous(),
            fx=intrinsics[..., 0].contiguous(),
            fy=intrinsics[..., 1].contiguous(),
            cx=intrinsics[..., 2].contiguous(),
            cy=intrinsics[..., 3].contiguous(),
            distortion_params=torch.from_numpy(distortions.distortion_params).contiguous() if distortions is not None else None,
            width=torch.from_numpy(sizes[..., 0]).long().contiguous(),
            height=torch.from_numpy(sizes[..., 1]).long().contiguous(),
            camera_type=torch.tensor(camera_types, dtype=torch.long))
        self._trainer.pipeline.eval()
        global_total = int(sizes.prod(-1).sum())
        global_i = 0
        if progress_callback:
            progress_callback(CurrentProgress(global_i, global_total, 0, len(poses)))
        for i in range(len(poses)):
            ray_bundle = cameras.generate_rays(camera_indices=i, keep_shape=True)
            get_outputs = self._trainer.pipeline.model.get_outputs_for_camera_ray_bundle
            if progress_callback and self._trainer.pipeline.model.__class__.get_outputs_for_camera_ray_bundle == Model.get_outputs_for_camera_ray_bundle:
                def local_progress(i, num_rays):
                    progress_callback(CurrentProgress(global_i + i, global_total, i, len(poses)))
                get_outputs = partial(_hacked_get_outputs_for_camera_ray_bundle, self._trainer.pipeline.model,  update_callback=local_progress)
            outputs = get_outputs(ray_bundle)
            global_i += int(sizes[i].prod(-1))
            if progress_callback:
                progress_callback(CurrentProgress(global_i, global_total, i+1, len(poses)))
            yield {
                "color": outputs["rgb"].detach().cpu().numpy(),
            }
        self._trainer.pipeline.train()

    def _transform_poses(self, poses):
        import torch
        assert poses.dim() == 3
        poses = (self._dataparser_transform @ torch.cat(
            [poses, torch.tensor([[[0, 0, 0, 1]]], dtype=self._dataparser_transform.dtype).expand((len(poses), 1, 4))], -2
        ))[:, :3, :].contiguous()
        poses[:, :3, 3] *= self._dataparser_scale
        return poses

    def _get_pose_transform(self, poses):
        import torch
        from nerfstudio.cameras import camera_utils
        poses = np.copy(poses)
        lastrow = np.array([[[0, 0, 0, 1]]] * len(poses), dtype=poses.dtype)
        poses = np.concatenate([poses, lastrow], axis=-2)
        poses = poses[..., np.array([1, 0, 2, 3]), :]
        poses[..., 2, :] *= -1

        applied_transform = np.eye(4)[:3, :]
        applied_transform = applied_transform[np.array([1, 0, 2]), :]
        applied_transform[2, :] *= -1

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(poses, method="up", center_method="poses")

        scale_factor = 1.0
        scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        poses[:, :3, 3] *= scale_factor

        applied_transform = torch.tensor(applied_transform, dtype=transform_matrix.dtype)
        transform_matrix = transform_matrix @ torch.cat(
            [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
        )
        transform_matrix_extended = torch.cat(
            [transform_matrix, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], -2
        )
        return transform_matrix_extended, scale_factor

    def setup_train(self, train_dataset: Dataset, *, num_iterations: int):
        import torch
        method = self
        if self.checkpoint is not None:
            self._dataparser_transform, self._dataparser_scale = torch.load(
                os.path.join(self.checkpoint, "dataparser_transform.pth"),
                map_location="cpu")
        self.config = copy.deepcopy(self._original_config)
        # We use this hack to release the memory after the data was copied to cached dataloader
        images_holder = [train_dataset.images]
        del train_dataset.images
        from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataparserOutputs
        from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager, InputDataset
        from nerfstudio.cameras.cameras import Cameras, CameraType as NPCameraType
        from nerfstudio.data.scene_box import SceneBox

        class CustomDataParser(DataParser):
            def __init__(self, config, *args, **kwargs):
                super().__init__(config)
                method._dp = self

            def _generate_dataparser_outputs(self,
                                             split: str = "train",
                                             **kwargs) -> DataparserOutputs:
                if split != "train":
                    return DataparserOutputs([], Cameras(
                        torch.zeros((1, 3, 4), dtype=torch.float32),
                        torch.zeros((1,), dtype=torch.float32),
                        torch.zeros((1,), dtype=torch.float32),
                        torch.zeros((1,), dtype=torch.float32),
                        torch.zeros((1,), dtype=torch.float32),
                        torch.zeros((1,), dtype=torch.long),
                        torch.zeros((1,), dtype=torch.long),
                        torch.zeros((1,), dtype=torch.float32),
                        torch.zeros((1,), dtype=torch.long),
                        ), None, None, [], {})
                image_names = [f"{i:06d}.png" for i in range(len(train_dataset.camera_poses))]
                camera_types = [NPCameraType.PERSPECTIVE for _ in range(len(train_dataset.camera_poses))]
                if train_dataset.camera_distortions is not None:
                    npmap = {x.name.lower(): x.value for x in NPCameraType.__members__.values()}
                    npmap["pinhole"] = npmap["perspective"]
                    npmap["opencv"] = npmap["perspective"]
                    npmap["opencv_fisheye"] = npmap["fisheye"]
                    camera_types = [npmap[CameraModel(train_dataset.camera_distortions.camera_types[i]).name.lower()] for i in range(len(train_dataset.camera_poses))]

                # in x,y,z order
                # assumes that the scene is centered at the origin
                aabb_scale = 1
                scene_box = SceneBox(
                    aabb=torch.tensor(
                        [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
                    )
                )

                if method.checkpoint is None:
                    method._dataparser_transform, method._dataparser_scale = method._get_pose_transform(train_dataset.camera_poses)
                else:
                    assert method._dataparser_transform is not None
                th_poses = method._transform_poses(torch.from_numpy(train_dataset.camera_poses).float())
                cameras = Cameras(camera_to_worlds=th_poses,
                                  fx=torch.from_numpy(train_dataset.camera_intrinsics[..., 0]).contiguous(),
                                  fy=torch.from_numpy(train_dataset.camera_intrinsics[..., 1]).contiguous(),
                                  cx=torch.from_numpy(train_dataset.camera_intrinsics[..., 2]).contiguous(),
                                  cy=torch.from_numpy(train_dataset.camera_intrinsics[..., 3]).contiguous(),
                                  distortion_params=(
                                      torch.from_numpy(train_dataset.camera_distortions.distortion_params).contiguous()
                                      if train_dataset.camera_distortions is not None else None
                                  ),
                                  width=torch.from_numpy(train_dataset.image_sizes[..., 0]).long().contiguous(),
                                  height=torch.from_numpy(train_dataset.image_sizes[..., 1]).long().contiguous(),
                                  camera_type=torch.tensor(camera_types, dtype=torch.long))
                return DataparserOutputs(
                    image_names,
                    cameras,
                    None, scene_box, image_names if train_dataset.sampling_masks else None, {},
                    dataparser_transform=method._dataparser_transform[..., :3, :].contiguous(), # pylint: disable=protected-access
                    dataparser_scale=method._dataparser_scale) # pylint: disable=protected-access
        self.config.pipeline.datamanager.dataparser._target = CustomDataParser # pylint: disable=protected-access
        self.config.max_num_iterations = num_iterations

        dm = self.config.pipeline.datamanager
        if dm.__class__.__name__ == "ParallelDataManagerConfig":
            dm = VanillaDataManagerConfig(**{
                k.name: getattr(dm, k.name) for k in fields(VanillaDataManagerConfig)
            })
            dm._target = VanillaDataManager # pylint: disable=protected-access
            self.config.pipeline.datamanager = dm
        class DM(dm._target): # pylint: disable=protected-access
            @property
            def dataset_type(self):
                class DatasetL(getattr(self, '_idataset_type', InputDataset)):
                    def get_image(self, image_idx: int):
                        img = images_holder[0][image_idx]
                        if img.dtype == np.uint8:
                            img = img.astype(np.float32) / 255.0
                        image = torch.from_numpy(img)
                        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
                            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
                        return image
                return DatasetL

            @dataset_type.setter
            def dataset_type(self, value):
                self._idataset_type = value

        self.config.output_dir = Path(self._tmpdir.name)
        self.config.pipeline.datamanager._target = DM  # pylint: disable=protected-access
        self.config.set_timestamp()
        self.config.vis = None
        self.config.machine.device_type = "cuda"
        self.config.load_step = None
        self._trainer = self.config.setup()
        self._trainer.setup()
        if self.checkpoint is not None:
            self.config.load_dir = Path(os.path.join(self.checkpoint, self.config.relative_model_dir))
            self._trainer._load_checkpoint()
        if getattr(self.config.pipeline.datamanager, "train_num_times_to_repeat_images", None) == -1:
            logging.debug("NerfStudio will cache all images, we will release the memory now")
            images_holder[0] = None
        self._mode = "train"

    def _setup_eval(self):
        if self.checkpoint is None:
            raise RuntimeError("Checkpoint must be provided to setup_eval")
        import torch
        self.config = copy.deepcopy(self._original_config)
        self.config.output_dir = Path(self._tmpdir.name)
        class DM(self.config.pipeline.datamanager):
            def __init__(self, *args, **kwargs):
                pass
        self.config.pipeline.datamanager._target = DM  # pylint: disable=protected-access
        # Set eval batch size
        self.config.pipeline.model.eval_num_rays_per_chunk = 4096
        self.config.set_timestamp()
        self.config.vis = None
        self.config.machine.device_type = "cuda"
        self.config.load_step = None
        self.config.load_dir = Path(os.path.join(self.checkpoint, self.config.relative_model_dir))
        self._trainer = self.config.setup()
        self._trainer.setup()
        if self.checkpoint is not None:
            self._trainer._load_checkpoint()
        self._dataparser_transform, self._dataparser_scale = torch.load(
            os.path.join(self.checkpoint, "dataparser_transform.pth"),
            map_location="cpu")
        self._mode = "eval"

    def _load_checkpoint(self):
        import torch
        if self.checkpoint is not None:
            load_path = os.path.join(self.checkpoint, self.config.relative_model_dir, f"step-{self.info.loaded_step:09d}.ckpt")
            loaded_state = torch.load(load_path, map_location="cpu")
            self._trainer.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            print(loaded_state)

    def train_iteration(self, step: int):
        if self._mode != "train":
            raise RuntimeError("Method is not in train mode. Call setup_train first.")
        import torch
        from nerfstudio.engine.trainer import TrainingCallbackLocation
        self.step = step

        self._trainer.pipeline.train()

        # training callbacks before the training iteration
        for callback in self._trainer.callbacks:
            callback.run_callback_at_location(
                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
            )

        # time the forward pass
        loss, loss_dict, metrics_dict = self._trainer.train_iteration(step)

        # training callbacks after the training iteration
        for callback in self._trainer.callbacks:
            callback.run_callback_at_location(
                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
            )

        metrics = metrics_dict
        metrics.update(loss_dict)
        metrics.update({"loss": loss})
        metrics.update({"num_rays": self.config.pipeline.datamanager.train_num_rays_per_batch})
        def detach(v):
            if torch.is_tensor(v):
                return v.detach().cpu().item()
            elif isinstance(v, np.ndarray):
                return v.item()
            assert isinstance(v, (str, float, int))
            return v
        self.step = step + 1
        return {k: detach(v) for k, v in metrics.items()}

    def save(self, path: str):
        """
        Save model.

        Args:
            path: Path to save.
        """
        if self._mode is None:
            self._setup_eval()
        from nerfstudio.engine.trainer import Trainer
        import yaml
        import torch
        assert isinstance(self._trainer, Trainer)
        bckp = self._trainer.checkpoint_dir
        self._trainer.checkpoint_dir = Path(path)
        config_yaml_path = Path(path) / "config.yml"
        config_yaml_path.write_text(yaml.dump(self._original_config), "utf8")
        self._trainer.checkpoint_dir = Path(os.path.join(path, self._original_config.relative_model_dir))
        self._trainer.save_checkpoint(self.step)
        self._trainer.checkpoint_dir = bckp
        torch.save((
            self._dataparser_transform.cpu(),
            float(self._dataparser_scale)
        ), os.path.join(path, "dataparser_transform.pth"))

    def close(self):
        self._tmpdir.cleanup()
        self._tmpdir = None


NerfStudioSpec = MethodSpec(
    method=NerfStudio,
    conda=CondaMethod.wrap(
        NerfStudio,
        conda_name="nerfstudio",
        python_version="3.10",
        install_script = """
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio==0.3.4
"""),
    docker=DockerMethod.wrap(
        NerfStudio,
        image="dromni/nerfstudio:0.3.4",
        python_path="python3",
        home_path="/home/user",
        mounts=[(os.path.expanduser("~/.cache/torch"), "/home/user/.cache/torch")]))

# Register supported methods
NerfStudioSpec.register("nerfacto", nerfstudio_name="nerfacto")
NerfStudioSpec.register("nerfacto:big", nerfstudio_name="nerfacto-big")
NerfStudioSpec.register("nerfacto:huge", nerfstudio_name="nerfacto-huge")
