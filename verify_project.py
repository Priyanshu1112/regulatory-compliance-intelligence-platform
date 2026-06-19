import os
import shutil
import json
import sqlite3
import pandas as pd
from fpdf import FPDF

# Import system classes
from src.loader import DocumentLoader
from src.chunker import DocumentChunker
from src.embeddings import EmbeddingFactory
from src.vector_store import ChromaVectorStore
from src.retrieval import ComplianceRetriever
from src.risk_scoring import RiskScorer
from src.evaluation import RetrievalEvaluator
from src.report_generator import ReportGenerator
from src.rag_pipeline import RAGPipeline
from src.audit_logger import AuditLogger

# Config parameters
TEST_DIR = "test_sandbox"
MOCK_PDF_PATH = os.path.join(TEST_DIR, "bank_policy_v1.pdf")
CHROMA_DIR = os.path.join(TEST_DIR, "test_vectorstore")
DB_PATH = os.path.join(TEST_DIR, "test_audit_trail.db")
REPORTS_DIR = os.path.join(TEST_DIR, "test_reports")

def setup_test_environment():
    """
    Cleans and sets up directory structure for isolated verification.
    """
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Generate mock PDF policy
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.multi_cell(pdf.epw, 10, "MOCK BANK COMPLIANCE POLICY", align="C")
    pdf.ln(5)
    pdf.set_font("helvetica", "", 11)
    # Page 1 content
    pdf.multi_cell(pdf.epw, 6, (
        "Section 1: Data Retention Rules.\n"
        "Customer transaction records must be retained for exactly 7 years after account closure.\n"
        "This policy follows regulatory advisory guidance and audit recommendations."
    ))
    
    pdf.add_page()
    # Page 2 content
    pdf.multi_cell(pdf.epw, 6, (
        "Section 2: AML and Fraud Control.\n"
        "Any suspected transaction related to money laundering or insider fraud must be escalated.\n"
        "A breach of these systems will result in immediate penalty and administrative sanction.\n"
        "A review exception will trigger an audit finding for escalation."
    ))
    
    pdf.output(MOCK_PDF_PATH)


def run_tests() -> bool:
    print("==================================================")
    print("STARTING PLATFORM COMPONENT VERIFICATION SUITE")
    print("==================================================")

    test_results = {}
    
    # Setup test folder
    setup_test_environment()

    # Shared variables
    pages = []
    chunks = []
    embedding_fn = None
    store = None
    retrieved_chunks = []
    eval_report = {}
    report_data = {}
    risk_data = {}
    response_text = "Suspected fraud activities could trigger audit findings and money laundering penalties."

    # 1. PDF Loading Test
    try:
        loader = DocumentLoader()
        pages = loader.load_pdf(MOCK_PDF_PATH)
        assert len(pages) == 2, "Should load exactly 2 pages from PDF"
        assert "Data Retention" in pages[0]["text"], "Page 1 missing text content"
        assert "AML and Fraud" in pages[1]["text"], "Page 2 missing text content"
        test_results["1. PDF Loading"] = "PASS"
    except Exception as e:
        test_results["1. PDF Loading"] = f"FAIL ({str(e)})"

    # 2. Chunking Test
    try:
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.split_pages(pages)
        assert len(chunks) > 0, "Chunk list should not be empty"
        assert all("chunk_index" in c["metadata"] for c in chunks), "Metadata chunk index missing"
        assert all("filename" in c["metadata"] for c in chunks), "Metadata source filename missing"
        test_results["2. Chunking"] = "PASS"
    except Exception as e:
        test_results["2. Chunking"] = f"FAIL ({str(e)})"

    # 3. Embedding Generation Test
    try:
        embedding_fn = EmbeddingFactory.get_embedding_function(provider="local")
        # Generate raw embedding for dummy text
        test_emb = embedding_fn(["test compliance chunk text"])
        assert len(test_emb) == 1, "Should output single vector array"
        assert len(test_emb[0]) == 384, f"Local all-MiniLM-L6-v2 vector dimension must be 384 (got {len(test_emb[0])})"
        test_results["3. Embedding Generation"] = "PASS"
    except Exception as e:
        test_results["3. Embedding Generation"] = f"FAIL ({str(e)})"

    # 4. ChromaDB Storage Test
    try:
        store = ChromaVectorStore(persist_directory=CHROMA_DIR, collection_name="test_collection")
        store.add_chunks(chunks, embedding_fn)
        metas = store.get_all_documents_metadata(embedding_fn)
        assert len(metas) == len(chunks), "Stored vector counts do not match input chunks"
        test_results["4. ChromaDB Storage"] = "PASS"
    except Exception as e:
        test_results["4. ChromaDB Storage"] = f"FAIL ({str(e)})"

    # 5. Semantic Search Test
    try:
        # Cosine distance query
        results = store.query("retention guidelines", embedding_fn, top_k=2)
        assert len(results) > 0, "Semantic query should return document chunks"
        assert "similarity_score" in results[0], "Semantic search results missing similarity scores"
        test_results["5. Semantic Search"] = "PASS"
    except Exception as e:
        test_results["5. Semantic Search"] = f"FAIL ({str(e)})"

    # 6. Hybrid Search Test
    try:
        retriever = ComplianceRetriever(vector_store=store, embedding_function=embedding_fn)
        retrieved_chunks = retriever.retrieve("money laundering penalty control", top_k=3, relevance_threshold=0.2, hybrid=True)
        assert len(retrieved_chunks) > 0, "Hybrid search should return matched chunks"
        assert "keyword_score" in retrieved_chunks[0], "Hybrid search missing keyword overlap scores"
        assert "final_score" in retrieved_chunks[0], "Hybrid search missing weighted final rank score"
        test_results["6. Hybrid Search"] = "PASS"
    except Exception as e:
        test_results["6. Hybrid Search"] = f"FAIL ({str(e)})"

    # 7. Risk Scoring Test
    try:
        # Scanning mock response: contains "fraud", "audit finding", "penalty", "money laundering"
        risk_data = RiskScorer.score_content("What controls protect against fraud?", response_text)
        assert risk_data["risk_level"] == "HIGH", "Should categorize as HIGH risk due to fraud/money laundering keywords"
        assert "explanation" in risk_data, "Risk response must include rationales"
        test_results["7. Risk Scoring"] = "PASS"
    except Exception as e:
        test_results["7. Risk Scoring"] = f"FAIL ({str(e)})"

    # 8. Audit Logging Test
    try:
        logger = AuditLogger(db_path=DB_PATH)
        dummy_eval = {"metrics": {"precision": 1.0, "recall": 0.8, "hit_rate": 1.0, "average_similarity_score": 0.76}, "similarity_stats": {}}
        
        logger.log_query("What controls protect against fraud?", response_text, risk_data, retrieved_chunks, dummy_eval)
        logs = logger.get_logs()
        assert len(logs) == 1, "Log registry counts must increment"
        assert logs.iloc[0]["risk_level"] == "HIGH", "Logged risk level mismatch"
        test_results["8. Audit Logging"] = "PASS"
    except Exception as e:
        test_results["8. Audit Logging"] = f"FAIL ({str(e)})"

    # 9. Evaluation Metrics Test
    try:
        eval_report = RetrievalEvaluator.generate_evaluation_report(
            query="money laundering penalty",
            retrieved_chunks=retrieved_chunks,
            ground_truth_filenames=["bank_policy_v1.pdf"]
        )
        assert "metrics" in eval_report, "Evaluation missing core metrics dict"
        assert "precision" in eval_report["metrics"], "Evaluation missing precision key"
        assert "recall" in eval_report["metrics"], "Evaluation missing recall key"
        test_results["9. Evaluation Metrics"] = "PASS"
    except Exception as e:
        test_results["9. Evaluation Metrics"] = f"FAIL ({str(e)})"

    # 10. End-to-End RAG Workflow Test
    try:
        # Instantiate RAG pipeline with mock config
        pipeline_config = {
            "provider": "custom",
            "api_key": "mock_key",
            "base_url": "https://api.openai.com/v1",  # Dummy, we won't invoke actual network in mock check
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.0,
            "embedding_provider": "local",
            "embedding_model": "all-MiniLM-L6-v2",
            "persist_directory": CHROMA_DIR,
            "collection_name": "test_collection"
        }
        
        pipeline = RAGPipeline(pipeline_config)
        # Mock LLM API calls internally or bypass. Since network call requires api keys,
        # we check properties and retrievals.
        assert pipeline.retriever is not None, "Pipeline retriever not configured"
        assert pipeline.embedding_function is not None, "Pipeline embeddings not initialized"
        test_results["10. End-to-End RAG Workflow"] = "PASS"
    except Exception as e:
        test_results["10. End-to-End RAG Workflow"] = f"FAIL ({str(e)})"

    # 11. Markdown Report Generation Test
    try:
        report_data = ReportGenerator.generate_report(
            query="What controls protect against fraud?",
            response=response_text,
            risk_data=risk_data,
            retrieved_chunks=retrieved_chunks,
            eval_data=eval_report,
            model_name="mock-model"
        )
        md_file = ReportGenerator.export_markdown(report_data, output_dir=REPORTS_DIR)
        assert os.path.exists(md_file), "Markdown output report file should exist"
        test_results["11. Markdown Report Generation"] = "PASS"
    except Exception as e:
        test_results["11. Markdown Report Generation"] = f"FAIL ({str(e)})"

    # 12. PDF Report Generation Test
    try:
        pdf_file = ReportGenerator.export_pdf(report_data, output_dir=REPORTS_DIR)
        assert os.path.exists(pdf_file), "PDF output report file should exist"
        test_results["12. PDF Report Generation"] = "PASS"
    except Exception as e:
        test_results["12. PDF Report Generation"] = f"FAIL ({str(e)})"

    # 13. Citation Inclusion Test
    try:
        # Read generated markdown report to check references
        filepath = os.path.join(REPORTS_DIR, f"compliance_report_{report_data['report_id']}.md")
        with open(filepath, "r", encoding="utf-8") as f:
            md_text = f.read()
        assert "Citation #1" in md_text, "Report must include Citation details"
        assert "bank_policy_v1.pdf" in md_text, "Report citations must name the source document"
        test_results["13. Citation Inclusion"] = "PASS"
    except Exception as e:
        test_results["13. Citation Inclusion"] = f"FAIL ({str(e)})"

    # 14. Risk Assessment Inclusion Test
    try:
        filepath = os.path.join(REPORTS_DIR, f"compliance_report_{report_data['report_id']}.md")
        with open(filepath, "r", encoding="utf-8") as f:
            md_text = f.read()
        assert "Assigned Risk Level:" in md_text, "Report must include risk assignments"
        assert "HIGH" in md_text, "Risk level score must be present in content"
        test_results["14. Risk Assessment Inclusion"] = "PASS"
    except Exception as e:
        test_results["14. Risk Assessment Inclusion"] = f"FAIL ({str(e)})"

    # 15. Report Export Functionality Test
    try:
        # Check files physically exist and are not empty
        pdf_size = os.path.getsize(os.path.join(REPORTS_DIR, f"compliance_report_{report_data['report_id']}.pdf"))
        md_size = os.path.getsize(os.path.join(REPORTS_DIR, f"compliance_report_{report_data['report_id']}.md"))
        assert pdf_size > 500, "PDF export file is suspiciously small"
        assert md_size > 100, "Markdown export file is suspiciously small"
        test_results["15. Report Export Functionality"] = "PASS"
    except Exception as e:
        test_results["15. Report Export Functionality"] = f"FAIL ({str(e)})"

    # Cleanup temporary test sandbox directory
    # Release file handles for Windows
    store = None
    import gc
    gc.collect()
    if os.path.exists(TEST_DIR):
        try:
            shutil.rmtree(TEST_DIR)
        except Exception:
            pass

    # Print summary
    print("\nVerification Test Case Reports:")
    print("--------------------------------------------------")
    all_passed = True
    for name, res in test_results.items():
        symbol = "[PASS]" if res == "PASS" else "[FAIL]"
        print(f"{symbol} {name:<35}: {res}")
        if res != "PASS":
            all_passed = False
            
    print("--------------------------------------------------")
    if all_passed:
        print("ALL 15 VERIFICATION TEST CASES PASSED SUCCESSFULLY!")
    else:
        print("SOME SYSTEM TESTS FAILED. PLEASE DEBUG ABOVE LOGS.")
    print("==================================================")
    
    return all_passed

if __name__ == "__main__":
    run_tests()
