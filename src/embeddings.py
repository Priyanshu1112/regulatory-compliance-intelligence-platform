import os
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

class LocalEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function generating vector representations locally using sentence-transformers.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        # sentence-transformers encode takes a list of strings and returns a list of vectors
        embeddings = self.model.encode(input, convert_to_numpy=True).tolist()
        return embeddings

class OpenAIEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function generating vector representations using OpenAI's API.
    """
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = None, base_url: str = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.embeddings.create(
            input=input,
            model=self.model_name
        )
        return [data.embedding for data in response.data]

class EmbeddingFactory:
    """
    Factory to retrieve configured embedding functions.
    """
    @staticmethod
    def get_embedding_function(
        provider: str = "local",
        model_name: str = "all-MiniLM-L6-v2",
        api_key: str = None,
        base_url: str = None
    ) -> EmbeddingFunction:
        if provider == "local":
            return LocalEmbeddingFunction(model_name=model_name)
        elif provider == "openai":
            return OpenAIEmbeddingFunction(model_name=model_name, api_key=api_key, base_url=base_url)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
