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
    def __init__(self, parser_type="resiliparse"):
        self.parser_type = parser_type

    def __call__(self, df: DataFrame) -> DataFrame:
        if self.parser_type == "resiliparse":
            df = df.with_column("cleaned_html", magic_parse(col("html")))
            df = df.with_column("text", resiliparse_extract(col("cleaned_html")))
        elif self.parser_type == "trafilatura":
            df = df.with_column("text", trafilatura_extract(col("html")))
        else:
            return ValueError("Selected parser is not available")
        return df
