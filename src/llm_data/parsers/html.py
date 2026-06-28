import daft
from daft import DataFrame, col
from magic_html import GeneralExtractor
from trafilatura import extract
from resiliparse.extract.html2text import extract_plain_text


@daft.func
def magic_parse(html: str) -> str:
    return GeneralExtractor().extract(html)["html"]


@daft.func
def trafilatura_extract(html: str) -> str:
    return extract(filecontent=html)


@daft.func
def resiliparse_extract(html: str) -> str:
    return extract_plain_text(html)


class ParseHtml:
    def __init__(
        self,
        input_column: str = "html",
        output_column: str = "text",
        parser_type="resiliparse",
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.parser_type = parser_type

    def __call__(self, df: DataFrame) -> DataFrame:
        if self.parser_type == "resiliparse":
            df = df.with_column("cleaned_html", magic_parse(col(self.input_column)))
            df = df.with_column(
                self.output_column, resiliparse_extract(col("cleaned_html"))
            ).exclude("cleaned_html")
        elif self.parser_type == "trafilatura":
            df = df.with_column(
                self.output_column, trafilatura_extract(col(self.input_column))
            )
        else:
            return ValueError("Selected parser is not available")
        return df
