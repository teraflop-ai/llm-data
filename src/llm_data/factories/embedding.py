from typing import Optional

def embedding_factory(
    *,
    model_name: str = "lightonai/DenseOn",
    batch_size: int = 32,
    embedding_dim: int = 768,
    gpus: float = 1,
    cpus: Optional[float] = None, 
):
    import daft
    from daft import DataType, Series

    return_dtype = DataType.embedding(DataType.float32(), embedding_dim)

    @daft.cls(gpus=gpus, cpus=cpus)
    class TextEmbedding:
        def __init__(self):
            self.model_name = model_name
            self.batch_size = batch_size
            self.model = None

        @daft.method.batch(return_dtype=return_dtype, batch_size=batch_size)
        def embed_batch(self, text: Series):
            inputs = text.to_pylist()
            model = self.lazy_load()
            embeddings = model.encode(inputs)
            return embeddings.astype("float32")

        def lazy_load(self):
            if self.model is None:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(self.model_name)
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
    gpus: int | float = 1,
    cpus: Optional[float] = None,
):
    from daft import col

    embed_batch = embedding_factory(
        model_name=model_name,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        gpus=gpus,
        cpus=cpus,
    )

    return df.with_column(
        output_column,
        embed_batch(col(input_column)),
    )
