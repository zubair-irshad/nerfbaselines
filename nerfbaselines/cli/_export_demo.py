import logging
import click
from pathlib import Path
import json
from nerfbaselines import backends
from nerfbaselines import (
    get_method_spec, build_method_class,
)
from nerfbaselines.io import open_any_directory, deserialize_nb_info
from nerfbaselines.datasets import load_dataset
from ._common import click_backend_option
from ._common import SetParamOptionType, NerfBaselinesCliCommand


@click.command("export-demo", cls=NerfBaselinesCliCommand)
@click.option("--checkpoint", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
@click.option("--data", required=False, default=None)
@click.option("--train-embedding", type=int, default=None, help="Select the train embedding index to use for the demo.")
@click.option("--set", "options", help="Set a parameter for demo export.", type=SetParamOptionType(), multiple=True, default=None)
@click_backend_option()
def main(*, checkpoint: str, output: str, backend_name, data=None, train_embedding=None, options):
    checkpoint = str(checkpoint)
    output = str(output)
    options = dict(options or [])

    # Read method nb-info
    with open_any_directory(checkpoint, mode="r") as _checkpoint_path:
        checkpoint_path = Path(_checkpoint_path)
        assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
        assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
        with (checkpoint_path / "nb-info.json").open("r") as f:
            nb_info = json.load(f)
        nb_info = deserialize_nb_info(nb_info)

        method_name = nb_info["method"]
        backends.mount(checkpoint_path, checkpoint_path)
        method_spec = get_method_spec(method_name)
        with build_method_class(method_spec, backend=backend_name) as method_cls:
            method = method_cls(checkpoint=str(checkpoint_path))
            dataset_metadata = nb_info.get("dataset_metadata")
            if data is not None:
                dataset = load_dataset(data, split="train", load_features=False)
                if dataset_metadata is not None:
                    logging.warning("Overwriting dataset metadata stored in the checkpoint")
                dataset_metadata = dataset["metadata"]
            if dataset_metadata is None:
                logging.warning("No dataset metadata found in the checkpoint and no dataset provided as input. Some methods may require dataset metadata to export a demo. Please provide a dataset using the --data option.")
            try:
                method_export_demo = method.export_demo  # type: ignore
            except AttributeError:
                raise NotImplementedError(f"Method {method_name} does not support export_demo")

            # If train embedding is enabled, select train_embedding
            embedding = None
            if train_embedding is not None:
                embedding = method.get_train_embedding(train_embedding)
                if train_embedding is None:
                    logging.error(f"Train embedding {train_embedding} not found or not supported by the method.")
            method_export_demo(
                path=output,
                options=dict(
                    **options,
                    embedding=embedding,
                    dataset_metadata=dataset_metadata)
            )


if __name__ == "__main__":
    main()
