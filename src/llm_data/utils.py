from daft import DataType
from pathlib import Path


def checkpoint_uri(path: str) -> str:
    if "://" in path:
        return path
    return Path(path).resolve().as_uri()


def get_daft_dtype(precision: str):
    if precision == "float32":
        daft_dtype = DataType.float32()
    else:
        raise ValueError("Daft Datatype not supported")
    return daft_dtype
