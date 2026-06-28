import daft
from daft import DataFrame, col
from ftfy import fix_text


@daft.func
def ftfy_text(text: str) -> str:
    return fix_text(text)


class FixEncoding:
    def __init__(self, input_column: str = "text", output_column: str = "text"):
        self.input_column = input_column
        self.output_column = output_column

    def __call__(self, df: DataFrame) -> DataFrame:
        df = df.with_column(self.output_column, ftfy_text(col(self.input_column)))
        return df
