import os
import shutil
import chromadb
from chromadb.api.types import EmbeddingFunction

class ChromaVectorStore:
    """
    Manages vector storage and retrieval using ChromaDB.
    """
    def __init__(self, persist_directory: str = "vectorstore", collection_name: str = "regulatory_documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_directory)

    def get_or_create_collection(self, embedding_function: EmbeddingFunction):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_function
        )

    def add_chunks(self, chunks: list[dict], embedding_function: EmbeddingFunction):
        """
        Adds text chunks to the Chroma DB collection.
        """
        if not chunks:
            return

        collection = self.get_or_create_collection(embedding_function)

        ids = []
        documents = []
        metadatas = []

        for c in chunks:
            filename = c["metadata"]["filename"]
            page = c["metadata"]["page"]
            chunk_idx = c["metadata"]["chunk_index"]
            # Unique identifier for the chunk
            chunk_id = f"chunk_{filename}_p{page}_c{chunk_idx}"
            
            ids.append(chunk_id)
            documents.append(c["text"])
            
            # ChromaDB requires all metadata fields to be str, int, float, or bool.
            # Our loader and chunker metadata meets this requirement.
            metadatas.append(c["metadata"])

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def delete_collection(self):
        """
        Deletes the Chroma collection.
        """
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

    def rebuild_index(self):
        """
        Clears the database directory to completely rebuild.
        """
        self.delete_collection()
        # Clean up files inside directory to clear locks or state issues.
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
            except Exception as e:
                # Fallback to delete individual files if locked
                pass
        self.client = chromadb.PersistentClient(path=self.persist_directory)

    def query(self, query_text: str, embedding_function: EmbeddingFunction, top_k: int = 5, where: dict = None) -> list[dict]:
        """
        Queries Chroma for top_k results.
        
        Returns:
            A list of dicts:
            {
                "text": str,
                "metadata": dict,
                "distance": float,
                "similarity_score": float
            }
        """
        collection = self.get_or_create_collection(embedding_function)
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where
        )

        formatted_results = []
        if not results or not results["documents"] or len(results["documents"][0]) == 0:
            return formatted_results

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, distances):
            # Chroma distances can be L2 distance (euclidean) by default.
            # If distance is small, similarity is high. Let's map it:
            # similarity = 1 - (dist / 2.0) for normalized embeddings.
            # Clip between 0 and 1.
            similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            formatted_results.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "similarity_score": similarity
            })

        return formatted_results

    def get_all_documents_metadata(self, embedding_function: EmbeddingFunction) -> list[dict]:
        """
        Returns all chunk metadata inside the store.
        """
        collection = self.get_or_create_collection(embedding_function)
        results = collection.get()
        if not results or not results["metadatas"]:
            return []
        return results["metadatas"]
