import math
from daft import DataType
from pathlib import Path
from typing import Literal


def checkpoint_uri(path: str) -> str:
    if "://" in path:
        return path
    return Path(path).resolve().as_uri()


def daft_dtype(
    precision: Literal["float32", "binary", "int8"] = "float32",
):
    if precision == "float32":
        daft_dtype = DataType.float32()
    elif precision == "binary":
        daft_dtype = DataType.binary()
    elif precision == "int8":
        daft_dtype = DataType.int8()
    else:
        raise ValueError(f"Daft datatype not supported: {precision}")
    return daft_dtype
