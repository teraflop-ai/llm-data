from typing import Optional


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


def embed_text(
    df,
    *,
    input_column: str = "text",
    output_column: str = "text_embedding",
    model_name: str = "lightonai/DenseOn",
    batch_size: int = 32,
    embedding_dim: int = 768,
    max_seq_len: Optional[int] = None,
    normalize_embeddings: Optional[bool] = True,
    precision: str = "float32",
    gpus: int | float = 1,
    cpus: Optional[float] = None,
):
    from daft import col

    embed_batch = embedding_factory(
        model_name=model_name,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        max_seq_len=max_seq_len,
        normalize_embeddings=normalize_embeddings,
        precision=precision,
        gpus=gpus,
        cpus=cpus,
    )

    return df.with_column(
        output_column,
        embed_batch(col(input_column)),
    )
