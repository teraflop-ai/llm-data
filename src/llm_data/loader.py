import daft
from daft import CheckpointStore, CheckpointConfig
from typing import Optional
from pathlib import Path

daft.set_runner_ray()


def checkpoint_uri(path: str) -> str:
    if "://" in path:
        return path
    return Path(path).resolve().as_uri()


class DataLoader:
    def __init__(
        self,
        loader_type: str = "parquet",
        checkpoint_path: Optional[str] = "data_checkpoints",
        file_path_column_name: Optional[str] = "source_path",
        checkpoint_on: Optional[str] = "source_path",
        num_workers: Optional[int] = None,
        cpus_per_worker: Optional[float] = None,
    ):
        self.loader_type = loader_type
        self.file_path_column_name = file_path_column_name

        if checkpoint_path and checkpoint_on:
            self.checkpoint_path = checkpoint_uri(checkpoint_path)
            self.config = CheckpointConfig(
                store=CheckpointStore(self.checkpoint_path),
                on=checkpoint_on,
                settings=daft.KeyFilteringSettings(
                    num_workers=num_workers,
                    cpus_per_worker=cpus_per_worker,
                ),
            )

    def read_data(self, input_path: str):
        if self.loader_type == "parquet":
            df = daft.read_parquet(
                input_path,
                checkpoint=self.config,
                file_path_column=self.file_path_column_name,
            )
        elif self.loader_type == "json":
            df = daft.read_json(
                input_path,
                checkpoint=self.config,
                file_path_column=self.file_path_column_name,
            )
        elif self.loader_type == "csv":
            df = daft.read_csv(
                input_path,
                checkpoint=self.config,
                file_path_column=self.file_path_column_name,
            )
        elif self.loader_type == "huggingface":
            df = daft.read_huggingface(input_path)
        elif self.loader_type == "lance":
            df = daft.read_lance(input_path, checkpoint=self.config)
        elif self.loader_type == "warc":
            df = daft.read_warc(
                input_path,
                checkpoint=self.config,
                file_path_column=self.file_path_column_name,
            )
        else:
            raise ValueError("DataLoader not supported.")
        return df
