from dataclasses import dataclass
from typing import Optional
from daft import DataFrame, col

def embedding_factory(
    *,
    model_name: str = "lightonai/DenseOn",
    batch_size: int = 32,
    embedding_dim: int = 768,
    max_seq_len: Optional[int] = None,
    normalize_embeddings: Optional[bool] = True,
    precision: str = "float32",
    gpus: int | float = 1,
    cpus: Optional[float] = None,
):
    import daft
    from daft import DataType, Series
    from llm_data.utils import daft_dtype

    dtype = daft_dtype(precision=precision)
    return_dtype = DataType.embedding(dtype, embedding_dim)

    @daft.cls(gpus=gpus, cpus=cpus)
    class TextEmbedding:
        def __init__(self):
            self.model = None

        @daft.method.batch(return_dtype=return_dtype, batch_size=batch_size)
        def embed_batch(self, text: Series):
            inputs = text.to_pylist()
            model = self.lazy_load()
            embeddings = model.encode(
                inputs,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize_embeddings,
                truncate_dim=embedding_dim,
                precision=precision,
                convert_to_numpy=True,
            )
            return embeddings

        def lazy_load(self):
            if self.model is None:
                import torch
                from sentence_transformers import SentenceTransformer

                assert torch.cuda.is_available()

                self.model = SentenceTransformer(
                    model_name,
                    model_kwargs={
                        "attn_implementation": "flash_attention_2",
                        "torch_dtype": "bfloat16",
                    },
                    device="cuda",
                )

                if max_seq_len is not None:
                    self.model.max_seq_length = max_seq_len

            return self.model

    return TextEmbedding().embed_batch


@dataclass
class EmbedText:
    input_column: str = "text"
    output_column: str = "text_embedding"
    model_name: str = "lightonai/DenseOn"
    batch_size: int = 32
    embedding_dim: int = 768
    max_seq_len: Optional[int] = None
    normalize_embeddings: Optional[bool] = True
    precision: str = "float32"
    gpus: int | float = 1
    cpus: Optional[float] = None

    name: str = "EmbedText"

    def __post_init__(self):
        self.embed_batch = embedding_factory(
            model_name=self.model_name,
            batch_size=self.batch_size,
            embedding_dim=self.embedding_dim,
            max_seq_len=self.max_seq_len,
            normalize_embeddings=self.normalize_embeddings,
            precision=self.precision,
            gpus=self.gpus,
            cpus=self.cpus,
        )

    def __call__(self, df: DataFrame) -> DataFrame:
        return df.with_column(
            self.output_column,
            self.embed_batch(col(self.input_column)),
        )
