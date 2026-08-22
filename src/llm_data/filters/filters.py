import csv
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from daft import DataFrame, col
from daft.functions import length


class LengthFilter:
    def __init__(
        self,
        input_column: str = "text",
        max_len: int = 10000,
        min_len: int = 0,
        name: str = "LengthFilter",
    ):
        self.input_column = input_column
        self.max_len = max_len
        self.min_len = min_len
        self.name = name

    def __call__(self, df: DataFrame) -> DataFrame:
        text_len = length(col(self.input_column))
        return df.where((text_len < self.max_len) & (text_len > self.min_len))


def train_jigsaw_classifiers(dataset_path: str, output_dir: str) -> dict[str, Path]:
    import fasttext

    targets = {
        "hate": ("toxic", "severe_toxic", "threat", "insult", "identity_hate"),
        "nsfw": ("obscene",),
    }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_paths = {name: output_dir / f"{name}.bin" for name in targets}

    with TemporaryDirectory() as directory:
        train_paths = {name: Path(directory) / f"{name}.txt" for name in targets}
        with ExitStack() as stack:
            outputs = {
                name: stack.enter_context(path.open("w", encoding="utf-8"))
                for name, path in train_paths.items()
            }
            source = stack.enter_context(
                Path(dataset_path).open(encoding="utf-8", newline="")
            )
            for row in csv.DictReader(source):
                text = " ".join(row["comment_text"].split())
                if text:
                    for name, columns in targets.items():
                        positive = any(row[column] == "1" for column in columns)
                        label = name if positive else f"non_{name}"
                        outputs[name].write(f"__label__{label} {text}\n")

        for name, path in train_paths.items():
            model = fasttext.train_supervised(input=str(path), wordNgrams=2)
            model.save_model(str(model_paths[name]))

    return model_paths
