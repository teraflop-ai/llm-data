import daft
from daft import col
from daft.functions import length
from daft import DataFrame


class LengthFilter:
    def __init__(
        self, input_column: str = "text", max_len: int = 10000, min_len: int = 0
    ):
        self.input_column = input_column
        self.max_len = max_len
        self.min_len = min_len

    def __call__(self, df: DataFrame) -> DataFrame:
        text_len = length(col(self.input_column))
        return df.where((text_len < self.max_len) & (text_len > self.min_len))
