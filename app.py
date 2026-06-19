import os
import json
import streamlit as st
import pandas as pd
from datetime import datetime

# Import platform backend modules
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
from src.dashboard import DashboardVisualizer

# Set Streamlit Page Config
st.set_page_config(
    page_title="Regulatory Compliance Intelligence Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "report_paths" not in st.session_state:
    st.session_state.report_paths = None

# Custom CSS for Premium Design
st.markdown("""
<style>
    .main {
        background-color: #F8FAFC;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 4px 4px 0px 0px;
        padding-left: 16px;
        padding-right: 16px;
        border: 1px solid #E2E8F0;
        font-weight: 600;
        font-size: 14px;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border-color: #3B82F6 !important;
    }
    /* Risk Badges */
    .risk-badge {
        padding: 6px 12px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
        margin-bottom: 10px;
    }
    .risk-high {
        background-color: #FEE2E2;
        color: #EF4444;
        border: 1px solid #FCA5A5;
    }
    .risk-medium {
        background-color: #FEF3C7;
        color: #D97706;
        border: 1px solid #FDE68A;
    }
    .risk-low {
        background-color: #D1FAE5;
        color: #059669;
        border: 1px solid #6EE7B7;
    }
    /* Citation Card */
    .citation-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .citation-title {
        font-weight: 700;
        color: #1E293B;
        font-size: 13px;
        margin-bottom: 4px;
    }
    .citation-body {
        font-size: 12px;
        color: #475569;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.title("⚖️ Compliance Control")
st.sidebar.markdown("Configure LLM Providers and database index actions below.")

# Provider Configuration
st.sidebar.subheader("LLM Configuration")
provider = st.sidebar.selectbox(
    "Provider",
    options=["openai", "ollama", "lm_studio", "custom"],
    index=0,
    help="Select the inference hosting provider."
)

api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    placeholder="OpenAI API Key (if using OpenAI)",
    value=os.environ.get("OPENAI_API_KEY", "")
)

base_url = st.sidebar.text_input(
    "Base URL Override",
    placeholder="http://localhost:11434/v1",
    help="Set for local/custom endpoints (Ollama/LM Studio)."
)

model_name = st.sidebar.text_input(
    "Model Name",
    value="gpt-3.5-turbo" if provider == "openai" else "llama2",
    help="Name of the model deployed on the host."
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    help="Keep at 0.0 for deterministic compliance outputs."
)

# Text Splitter Settings
st.sidebar.subheader("Chunking Configuration")
chunk_size = st.sidebar.slider("Chunk Size (Chars)", min_value=200, max_value=2000, value=1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap (Chars)", min_value=50, max_value=500, value=200, step=50)

# Build Core Config Dictionary
pipeline_config = {
    "provider": provider,
    "api_key": api_key,
    "base_url": base_url,
    "model_name": model_name,
    "temperature": temperature,
    "embedding_provider": "local",  # Hardcoded default local-first
    "embedding_model": "all-MiniLM-L6-v2",
    "persist_directory": "vectorstore",
    "collection_name": "regulatory_documents"
}

# Database Actions Section
st.sidebar.subheader("Database Maintenance")

# Create local directories
os.makedirs("data", exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)
os.makedirs("audit_logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Helper function to get database instances
def get_db_and_embeddings():
    embedding_fn = EmbeddingFactory.get_embedding_function(
        provider=pipeline_config["embedding_provider"],
        model_name=pipeline_config["embedding_model"]
    )
    store = ChromaVectorStore(
        persist_directory=pipeline_config["persist_directory"],
        collection_name=pipeline_config["collection_name"]
    )
    return store, embedding_fn

# Database actions buttons
store, embedding_fn = get_db_and_embeddings()
logger = AuditLogger()

if st.sidebar.button("Rebuild Index"):
    with st.spinner("Rebuilding database vector index..."):
        store.rebuild_index()
        st.sidebar.success("Database rebuild complete!")
        st.rerun()

if st.sidebar.button("Clear Database"):
    if st.sidebar.checkbox("Are you sure?"):
        store.rebuild_index()
        logger.clear_logs()
        # Clean uploads folder
        for f in os.listdir("data"):
            os.remove(os.path.join("data", f))
        st.sidebar.success("Database and uploaded files cleared!")
        st.rerun()

# ----------------- MAIN INTERFACE -----------------
st.title("Regulatory Compliance Intelligence Platform")
st.markdown("Enterprise-grade compliance auditing, risk scoring, and automated reporting powered by Hybrid RAG.")

# Initialize the Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Document Management", 
    "💬 Compliance Assistant", 
    "📜 Audit Center", 
    "📊 Compliance Intelligence Dashboard"
])

# ----------------- TAB 1: DOCUMENT MANAGEMENT -----------------
with tab1:
    st.header("Upload and Process Compliance Documents")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Ingest New Documents")
        uploaded_files = st.file_uploader(
            "Select regulatory PDF policies or manuals",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process & Index Documents", type="primary"):
                loader = DocumentLoader()
                chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                
                total_pages = 0
                total_chunks = 0
                
                with st.spinner("Parsing PDFs and generating embedding vectors..."):
                    for uploaded_file in uploaded_files:
                        # Save file to local data folder
                        file_path = os.path.join("data", uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                        # Load and extract text
                        pages = loader.load_pdf(file_path)
                        total_pages += len(pages)
                        
                        # Chunk pages
                        chunks = chunker.split_pages(pages)
                        total_chunks += len(chunks)
                        
                        # Store in ChromaDB
                        store.add_chunks(chunks, embedding_fn)
                        
                st.success(f"Successfully processed {len(uploaded_files)} files! Loaded {total_pages} pages and created {total_chunks} vectors.")
                st.rerun()

    with col2:
        st.subheader("Indexed Documents Registry")
        
        # Load all metadata directly from ChromaDB
        metadatas = store.get_all_documents_metadata(embedding_fn)
        
        if metadatas:
            # Group metadata by filename
            doc_summary = {}
            for meta in metadatas:
                fname = meta.get("filename", "Unknown")
                if fname not in doc_summary:
                    doc_summary[fname] = {
                        "Chunks Count": 0,
                        "Upload Date": meta.get("upload_date", datetime.now().isoformat())[:10],
                        "File Size": f"{meta.get('file_size', 0) / 1024:.1f} KB"
                    }
                doc_summary[fname]["Chunks Count"] += 1
                
            # Render to dataframe
            doc_df = pd.DataFrame.from_dict(doc_summary, orient="index").reset_index()
            doc_df.columns = ["Document Name", "Chunks Count", "Upload Date", "File Size"]
            st.dataframe(doc_df, use_container_width=True, hide_index=True)
        else:
            st.info("No documents currently indexed in ChromaDB. Please upload a PDF to get started.")

# ----------------- TAB 2: COMPLIANCE ASSISTANT -----------------
with tab2:
    st.header("Compliance RAG Search & Assistant")
    st.markdown("Ask regulatory policy questions. Chunks are retrieved using hybrid search (Cosine similarity + keyword overlap).")
    
    # Initialize RAG Pipeline
    pipeline = RAGPipeline(pipeline_config)
    
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display risk badge and metadata if RAG output
            if "risk_data" in message:
                r_lvl = message["risk_data"]["risk_level"]
                r_exp = message["risk_data"]["explanation"]
                
                badge_class = "risk-high" if r_lvl == "HIGH" else ("risk-medium" if r_lvl == "MEDIUM" else "risk-low")
                st.markdown(f'<div class="risk-badge {badge_class}">Compliance Risk Score: {r_lvl}</div>', unsafe_allow_html=True)
                st.markdown(f"**Risk Evaluation Summary:** *{r_exp}*")
                
            if "citations" in message:
                with st.expander("Show Retrieved Compliance Excerpts (Citations)"):
                    for idx, cit in enumerate(message["citations"]):
                        st.markdown(f"""
                        <div class="citation-card">
                            <div class="citation-title">[{idx+1}] File: {cit['metadata'].get('filename')} | Page: {cit['metadata'].get('page')} | Search Match: {cit.get('final_score', 0.0):.2%}</div>
                            <div class="citation-body">"{cit['text'].strip()}"</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
    # Chat Input
    query = st.chat_input("What are the data retention requirements for compliance?")
    
    if query:
        # User input display
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        # AI Output streaming
        with st.chat_message("assistant"):
            # Set top_k to 5, threshold to 0.4 by default
            with st.spinner("Retrieving sources and synthesizing compliance audit response..."):
                response = pipeline.generate_response(
                    query=query,
                    top_k=5,
                    relevance_threshold=0.4,
                    hybrid=True,
                    stream=False # Stream is false to easily allow post-process scoring & report generation metadata caching
                )
                
            st.markdown(response["answer"])
            
            # Risk Rating Badge
            risk_data = response["risk_data"]
            r_lvl = risk_data["risk_level"]
            r_exp = risk_data["explanation"]
            
            badge_class = "risk-high" if r_lvl == "HIGH" else ("risk-medium" if r_lvl == "MEDIUM" else "risk-low")
            st.markdown(f'<div class="risk-badge {badge_class}">Compliance Risk Score: {r_lvl}</div>', unsafe_allow_html=True)
            st.markdown(f"**Risk Evaluation Summary:** *{r_exp}*")
            
            # Citations List
            citations = response["retrieved_chunks"]
            with st.expander("Show Retrieved Compliance Excerpts (Citations)"):
                for idx, cit in enumerate(citations):
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-title">[{idx+1}] File: {cit['metadata'].get('filename')} | Page: {cit['metadata'].get('page')} | Search Match: {cit.get('final_score', 0.0):.2%}</div>
                        <div class="citation-body">"{cit['text'].strip()}"</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            # Cache evaluation and metadata for reports
            eval_report = response["eval_report"]
            report_data = response["report_data"]
            
            # Save RAG Message inside session state history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "risk_data": risk_data,
                "citations": citations
            })
            
            # Log this query search activity to sqlite
            logger.log_query(
                query=query,
                response=response["answer"],
                risk_data=risk_data,
                retrieved_chunks=citations,
                eval_data=eval_report,
                report_generated=False,
                report_id=None
            )
            
            # Enable reporting options
            st.session_state.last_report = report_data
            st.session_state.report_paths = None
            st.rerun()

    # Report Generator UI Actions (If we have a recent query/response in state)
    if st.session_state.last_report:
        st.markdown("---")
        st.subheader("📄 Executive Compliance Reporting")
        st.info("Synthesize the RAG findings above into an audit-ready compliance report.")
        
        rep_col1, rep_col2 = st.columns([1, 2])
        
        with rep_col1:
            if st.button("Generate Executive Investigation Report", type="primary"):
                with st.spinner("Assembling report templates and generating PDF..."):
                    # Generate files
                    report_data = st.session_state.last_report
                    md_path = ReportGenerator.export_markdown(report_data)
                    pdf_path = ReportGenerator.export_pdf(report_data)
                    
                    st.session_state.report_paths = {
                        "md": md_path,
                        "pdf": pdf_path,
                        "report_id": report_data["report_id"]
                    }
                    
                    # Update SQLite log with report metadata (for last inserted record)
                    conn = logger.get_logs()
                    if not conn.empty:
                        # Log report generation details
                        logger.log_query(
                            query=report_data["query"],
                            response=report_data["response"],
                            risk_data=report_data["risk_data"],
                            retrieved_chunks=report_data["retrieved_chunks"],
                            eval_data=report_data["eval_data"],
                            report_generated=True,
                            report_id=report_data["report_id"]
                        )
                        
                    st.success(f"Report {report_data['report_id']} generated successfully!")
                    st.rerun()

            if st.session_state.report_paths:
                # Add downloads
                with open(st.session_state.report_paths["md"], "r", encoding="utf-8") as f:
                    md_bytes = f.read()
                with open(st.session_state.report_paths["pdf"], "rb") as f:
                    pdf_bytes = f.read()
                    
                st.download_button(
                    label="Download Markdown Report (.md)",
                    data=md_bytes,
                    file_name=os.path.basename(st.session_state.report_paths["md"]),
                    mime="text/markdown",
                    use_container_width=True
                )
                
                st.download_button(
                    label="Download PDF Report (.pdf)",
                    data=pdf_bytes,
                    file_name=os.path.basename(st.session_state.report_paths["pdf"]),
                    mime="application/pdf",
                    use_container_width=True
                )
                
        with rep_col2:
            if st.session_state.report_paths:
                st.markdown("### Report Live Preview")
                with st.container(border=True, height=250):
                    # Show markdown preview
                    with open(st.session_state.report_paths["md"], "r", encoding="utf-8") as f:
                        st.markdown(f.read())

# ----------------- TAB 3: AUDIT CENTER -----------------
with tab3:
    st.header("Compliance Audit Logs & Trail")
    st.markdown("All queries, LLM answers, document search hits, risk results, and quality metrics are logged in SQLite.")
    
    logs_df = logger.get_logs()
    
    if not logs_df.empty:
        # Search parameters
        search_col1, search_col2 = st.columns([2, 1])
        with search_col1:
            search_query = st.text_input("Search query or response logs keyword")
        with search_col2:
            filter_risk = st.selectbox("Filter by Risk Level", options=["ALL", "HIGH", "MEDIUM", "LOW"])
            
        # Filters application
        filtered_df = logs_df.copy()
        if search_query:
            filtered_df = filtered_df[
                filtered_df["query"].str.contains(search_query, case=False) |
                filtered_df["response"].str.contains(search_query, case=False)
            ]
        if filter_risk != "ALL":
            filtered_df = filtered_df[filtered_df["risk_level"] == filter_risk]
            
        # Export Actions
        exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])
        with exp_col1:
            csv_path = logger.export_logs(export_format="csv")
            with open(csv_path, "r", encoding="utf-8") as f:
                csv_data = f.read()
            st.download_button(
                "Export Log as CSV",
                data=csv_data,
                file_name=os.path.basename(csv_path),
                mime="text/csv",
                use_container_width=True
            )
        with exp_col2:
            json_path = logger.export_logs(export_format="json")
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = f.read()
            st.download_button(
                "Export Log as JSON",
                data=json_data,
                file_name=os.path.basename(json_path),
                mime="application/json",
                use_container_width=True
            )
            
        st.markdown("### Search Results")
        # Columns formatting
        render_cols = ["timestamp", "query", "risk_level", "report_generated", "report_id"]
        st.dataframe(
            filtered_df[render_cols], 
            use_container_width=True, 
            hide_index=True
        )
        
        # Detail expansion card
        st.markdown("### Log Record Inspector")
        inspect_id = st.number_input("Enter log ID row to inspect detailed payload:", min_value=1, max_value=int(logs_df["id"].max()) if not logs_df.empty else 1, step=1)
        record = logs_df[logs_df["id"] == inspect_id]
        if not record.empty:
            rec_row = record.iloc[0]
            st.write(f"**Timestamp:** {rec_row['timestamp']} | **Risk:** {rec_row['risk_level']} | **Report ID:** {rec_row['report_id']}")
            st.write(f"**User Inquiry:** \"{rec_row['query']}\"")
            st.markdown(f"**Platform Analysis Output:** \n{rec_row['response']}")
            
            # Show sources
            try:
                sources_list = json.loads(rec_row["retrieved_sources"])
                st.write("**Retrieved Documents Citations:**")
                for s in sources_list:
                    st.write(f"- File: `{s['filename']}` | Page: {s['page']} | Score: {s['score']:.4f}")
            except Exception:
                pass
    else:
        st.info("No compliance query records found in the audit trail database yet.")

# ----------------- TAB 4: COMPLIANCE INTELLIGENCE DASHBOARD -----------------
with tab4:
    st.header("Compliance Intelligence Dashboard")
    st.markdown("Visual analytics tracking operational risks, document usage frequency, and RAG retrieval search precision.")
    
    # Reload logs for analytics
    analytics_df = logger.get_logs()
    
    # Calculate stats
    total_docs = len(doc_df) if 'doc_df' in locals() and not doc_df.empty else 0
    total_chunks = len(metadatas) if metadatas else 0
    total_queries = len(analytics_df)
    
    high_queries = len(analytics_df[analytics_df["risk_level"] == "HIGH"])
    med_queries = len(analytics_df[analytics_df["risk_level"] == "MEDIUM"])
    low_queries = len(analytics_df[analytics_df["risk_level"] == "LOW"])
    
    # Evaluation stats (Average Retrieval Score)
    avg_precision = 0.0
    if not analytics_df.empty:
        precisions = []
        for em in analytics_df["eval_metrics"]:
            try:
                precisions.append(json.loads(em)["metrics"]["average_similarity_score"])
            except Exception:
                pass
        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0

    # Report metrics
    total_reports = int(analytics_df["report_generated"].sum()) if not analytics_df.empty else 0
    # Reports generated today
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_reports = int(analytics_df[(analytics_df["report_generated"] == 1) & (analytics_df["timestamp"].str.contains(today_str))]["report_generated"].sum()) if not analytics_df.empty else 0
    # High risk reports
    high_risk_reports = len(analytics_df[(analytics_df["report_generated"] == 1) & (analytics_df["risk_level"] == "HIGH")]) if not analytics_df.empty else 0

    # Document reference stats
    most_ref_doc = "None"
    if not analytics_df.empty:
        ref_counter = Counter()
        for src_str in analytics_df["retrieved_sources"].dropna():
            try:
                for s in json.loads(src_str):
                    if s.get("filename"):
                        ref_counter[s["filename"]] += 1
            except Exception:
                pass
        if ref_counter:
            most_ref_doc = ref_counter.most_common(1)[0][0]

    # Render KPI Cards
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Total Indexed Policies", total_docs)
        st.metric("Total Text Vectors", total_chunks)
    with col_b:
        st.metric("Total User Inquiries", total_queries)
        st.metric("Average Retrieval Confidence", f"{avg_precision:.2%}")
    with col_c:
        st.metric("Compliance Reports Generated", total_reports)
        st.metric("Reports Generated Today", today_reports)
    with col_d:
        st.metric("High Risk Investigations", high_queries)
        st.metric("Most Referenced Document", most_ref_doc if len(most_ref_doc) <= 22 else most_ref_doc[:20] + "...")

    # Render Visualizations using Altair
    if not analytics_df.empty:
        st.markdown("---")
        st.subheader("Operational Risk & Search Analytics")
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            risk_chart = DashboardVisualizer.plot_risk_distribution(analytics_df)
            if risk_chart:
                st.altair_chart(risk_chart, use_container_width=True)
            
            trend_chart = DashboardVisualizer.plot_query_trends(analytics_df)
            if trend_chart:
                st.altair_chart(trend_chart, use_container_width=True)
                
        with col_chart2:
            doc_chart = DashboardVisualizer.plot_referenced_documents(analytics_df)
            if doc_chart:
                st.altair_chart(doc_chart, use_container_width=True)
                
            quality_chart = DashboardVisualizer.plot_retrieval_quality_metrics(analytics_df)
            if quality_chart:
                st.altair_chart(quality_chart, use_container_width=True)
    else:
        st.info("Visual charts will populate automatically once user query audit logs begin recording.")
