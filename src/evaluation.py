class RetrievalEvaluator:
    """
    Evaluates RAG retrieval quality, search metrics, and similarity score metrics.
    """
    
    @staticmethod
    def evaluate_retrieval(retrieved_chunks: list[dict], ground_truth_filenames: list[str] = None) -> dict:
        """
        Calculates retrieval metrics: Precision, Recall, and Hit Rate.
        If no ground-truth is provided, it estimates relevance using similarity score thresholds.
        """
        if not retrieved_chunks:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "hit_rate": 0.0,
                "average_similarity": 0.0
            }

        retrieved_files = [c["metadata"]["filename"] for c in retrieved_chunks]
        avg_sim = sum(c.get("similarity_score", 0.0) for c in retrieved_chunks) / len(retrieved_chunks)

        if not ground_truth_filenames:
            # Heuristic-based evaluation when live: treat a chunk as "relevant" if its similarity is >= 0.65
            relevant_count = sum(1 for c in retrieved_chunks if c.get("similarity_score", 0.0) >= 0.65)
            precision = relevant_count / len(retrieved_chunks)
            # Live query recall is estimated based on the retrieval count
            recall = 1.0 if relevant_count > 0 else 0.0
            hit_rate = 1.0 if relevant_count > 0 else 0.0
        else:
            # Evaluation against a known ground truth list of documents
            gt_set = set(ground_truth_filenames)
            retrieved_set = set(retrieved_files)
            
            overlap = retrieved_set.intersection(gt_set)
            precision = len(overlap) / len(retrieved_set) if retrieved_set else 0.0
            recall = len(overlap) / len(gt_set) if gt_set else 0.0
            hit_rate = 1.0 if len(overlap) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "hit_rate": hit_rate,
            "average_similarity": avg_sim
        }

    @staticmethod
    def calculate_similarity_metrics(retrieved_chunks: list[dict]) -> dict:
        """
        Analyzes statistics of the similarity scores from vector store retrieval.
        """
        if not retrieved_chunks:
            return {
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "mean_similarity": 0.0,
                "mean_final_hybrid_score": 0.0
            }

        sims = [c.get("similarity_score", 0.0) for c in retrieved_chunks]
        hybrid_scores = [c.get("final_score", 0.0) for c in retrieved_chunks]

        return {
            "min_similarity": min(sims),
            "max_similarity": max(sims),
            "mean_similarity": sum(sims) / len(sims),
            "mean_final_hybrid_score": sum(hybrid_scores) / len(hybrid_scores)
        }

    @classmethod
    def generate_evaluation_report(cls, query: str, retrieved_chunks: list[dict], ground_truth_filenames: list[str] = None) -> dict:
        """
        Assembles all retrieval quality statistics into a single report.
        """
        retrieval_stats = cls.evaluate_retrieval(retrieved_chunks, ground_truth_filenames)
        similarity_stats = cls.calculate_similarity_metrics(retrieved_chunks)
        
        report = {
            "query": query,
            "retrieved_count": len(retrieved_chunks),
            "metrics": {
                "precision": retrieval_stats["precision"],
                "recall": retrieval_stats["recall"],
                "hit_rate": retrieval_stats["hit_rate"],
                "average_similarity_score": retrieval_stats["average_similarity"]
            },
            "similarity_stats": similarity_stats
        }
        return report
