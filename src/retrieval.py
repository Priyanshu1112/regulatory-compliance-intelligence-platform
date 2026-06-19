import re
from src.vector_store import ChromaVectorStore

def compute_keyword_overlap(query: str, doc_text: str) -> float:
    """
    Computes a keyword overlap score between 0.0 and 1.0.
    Calculated as (number of query terms in document) / (total query terms).
    """
    # Simple word tokenization
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    doc_words = set(re.findall(r'\b\w+\b', doc_text.lower()))

    # Basic stop words filtering
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
        "for", "with", "by", "of", "is", "was", "were", "are", "be", 
        "been", "that", "this", "these", "those", "which", "who", "whom"
    }
    query_words = query_words - STOP_WORDS

    if not query_words:
        # Fallback if query only contains stop words
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        if not query_words:
            return 0.0

    overlap = query_words.intersection(doc_words)
    return len(overlap) / len(query_words)


class ComplianceRetriever:
    """
    Retrieves compliance documents using Semantic and Hybrid (Semantic + Keyword) search.
    """
    def __init__(self, vector_store: ChromaVectorStore, embedding_function):
        self.vector_store = vector_store
        self.embedding_function = embedding_function

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        relevance_threshold: float = 0.0,
        where: dict = None,
        hybrid: bool = True
    ) -> list[dict]:
        """
        Retrieves top-k relevant document chunks using the specified search strategy.

        Args:
            query: The user question/compliance search term.
            top_k: Number of chunks to retrieve.
            relevance_threshold: Filter out any matches with scores below this value.
            where: Metadata filtering dictionary passed to Chroma DB.
            hybrid: If True, executes hybrid search (80% Semantic + 20% Keyword Overlap).
            
        Returns:
            List of dictionaries representing relevant chunks:
            {
                "text": str,
                "metadata": dict,
                "distance": float,
                "similarity_score": float, (semantic only)
                "semantic_score": float,
                "keyword_score": float,
                "final_score": float
            }
        """
        if hybrid:
            # Query a larger candidate set semantic-wise to allow keyword overlap to rerank them
            candidate_k = max(25, top_k * 3)
            candidates = self.vector_store.query(
                query_text=query,
                embedding_function=self.embedding_function,
                top_k=candidate_k,
                where=where
            )

            scored_candidates = []
            for item in candidates:
                semantic_score = item["similarity_score"]
                keyword_score = compute_keyword_overlap(query, item["text"])
                
                # Hybrid weighting formula: 80% Semantic + 20% Keyword Overlap
                final_score = (semantic_score * 0.8) + (keyword_score * 0.2)
                
                scored_item = item.copy()
                scored_item["semantic_score"] = semantic_score
                scored_item["keyword_score"] = keyword_score
                scored_item["final_score"] = final_score
                scored_candidates.append(scored_item)

            # Rerank by hybrid final score
            scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
            
            # Apply relevance threshold
            filtered = [c for c in scored_candidates if c["final_score"] >= relevance_threshold]
            return filtered[:top_k]
        else:
            # Semantic search only
            candidates = self.vector_store.query(
                query_text=query,
                embedding_function=self.embedding_function,
                top_k=top_k,
                where=where
            )
            
            scored_candidates = []
            for item in candidates:
                scored_item = item.copy()
                scored_item["semantic_score"] = item["similarity_score"]
                scored_item["keyword_score"] = 0.0
                scored_item["final_score"] = item["similarity_score"]
                scored_candidates.append(scored_item)
                
            filtered = [c for c in scored_candidates if c["final_score"] >= relevance_threshold]
            return filtered
