import os
from openai import OpenAI
from src.embeddings import EmbeddingFactory
from src.vector_store import ChromaVectorStore
from src.retrieval import ComplianceRetriever
from src.risk_scoring import RiskScorer
from src.evaluation import RetrievalEvaluator
from src.report_generator import ReportGenerator

class RAGPipeline:
    """
    Coordinates the entire Retrieval Augmented Generation workflow.
    """
    def __init__(self, config: dict):
        """
        Initializes the pipeline with configuration settings.

        Args:
            config: A dictionary containing database and LLM parameter settings:
                - provider: 'openai' | 'ollama' | 'lm_studio' | 'custom'
                - api_key: str
                - base_url: str
                - model_name: str
                - temperature: float
                - embedding_provider: 'local' | 'openai'
                - embedding_model: str
                - persist_directory: str
                - collection_name: str
        """
        self.config = config
        
        # Initialize Embeddings
        self.embedding_function = EmbeddingFactory.get_embedding_function(
            provider=config.get("embedding_provider", "local"),
            model_name=config.get("embedding_model", "all-MiniLM-L6-v2"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )
        
        # Initialize Database
        self.vector_store = ChromaVectorStore(
            persist_directory=config.get("persist_directory", "vectorstore"),
            collection_name=config.get("collection_name", "regulatory_documents")
        )
        
        # Initialize Retriever
        self.retriever = ComplianceRetriever(
            vector_store=self.vector_store,
            embedding_function=self.embedding_function
        )

        # Setup LLM client credentials
        provider = config.get("provider", "openai")
        api_key = config.get("api_key")
        base_url = config.get("base_url")

        # Resolve local providers endpoints
        if provider == "ollama":
            base_url = base_url or "http://localhost:11434/v1"
            api_key = api_key or "ollama"
        elif provider == "lm_studio":
            base_url = base_url or "http://localhost:1234/v1"
            api_key = api_key or "lm-studio"
        
        # Initialize native OpenAI Client (OpenAI-compatible)
        self.llm_client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy_key"),
            base_url=base_url
        )

    def generate_response(
        self,
        query: str,
        top_k: int = 5,
        relevance_threshold: float = 0.0,
        hybrid: bool = True,
        ground_truth_filenames: list[str] = None,
        stream: bool = False
    ) -> dict:
        """
        Executes search, constructs prompts, invokes LLM, scores risk, and structures report context.
        """
        # 1. Retrieve context
        retrieved_chunks = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            where=None,
            hybrid=hybrid
        )

        # 2. Evaluate retrieval precision, recall and similarity score distributions
        eval_report = RetrievalEvaluator.generate_evaluation_report(
            query=query,
            retrieved_chunks=retrieved_chunks,
            ground_truth_filenames=ground_truth_filenames
        )

        # 3. Construct prompt with source citations
        context_str = ""
        for idx, chunk in enumerate(retrieved_chunks):
            filename = chunk["metadata"].get("filename", "Unknown Document")
            page = chunk["metadata"].get("page", "?")
            context_str += f"[{idx+1}] File: {filename} | Page: {page}\nExcerpt: {chunk['text'].strip()}\n\n"

        prompt = f"""You are an enterprise regulatory compliance analyst assistant.
Analyze the following context from policies, compliance manuals, and regulation reports, and answer the user query based strictly on the provided documents.

Context:
{context_str if context_str else "No relevant documents found in the database. Rely only on general advice and state clearly that no document sources match."}

User Query: {query}

Instructions:
1. Provide a direct, factual, and audit-ready compliance analysis.
2. You MUST cite source files and page numbers in brackets (e.g. [1]) for every finding or policy rule cited.
3. If the context does not contain sufficient details to answer, state clearly that the reference documents do not address the query. Do not fabricate rules or numbers.
4. Keep the tone professional, objective, and structured.

Response:"""

        model_name = self.config.get("model_name", "gpt-3.5-turbo")
        temperature = self.config.get("temperature", 0.0)

        if stream:
            # Return stream generator alongside retrieval objects
            def response_generator():
                chat_stream = self.llm_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    stream=True
                )
                for chunk in chat_stream:
                    content = chunk.choices[0].delta.content or ""
                    yield content

            return {
                "stream": response_generator(),
                "retrieved_chunks": retrieved_chunks,
                "eval_report": eval_report
            }
        else:
            # Synchronous response execution
            completion = self.llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            response_text = completion.choices[0].message.content or ""
            
            # Risk assessment
            risk_data = RiskScorer.score_content(query, response_text)

            # Build report structure
            report_data = ReportGenerator.generate_report(
                query=query,
                response=response_text,
                risk_data=risk_data,
                retrieved_chunks=retrieved_chunks,
                eval_data=eval_report,
                model_name=model_name
            )

            return {
                "answer": response_text,
                "risk_data": risk_data,
                "retrieved_chunks": retrieved_chunks,
                "eval_report": eval_report,
                "report_data": report_data
            }
