import os
import uuid
from datetime import datetime
from fpdf import FPDF

class ReportGenerator:
    """
    Enterprise reporting engine generating markdown and PDF audit-ready compliance reports.
    """

    @staticmethod
    def generate_report(
        query: str,
        response: str,
        risk_data: dict,
        retrieved_chunks: list[dict],
        eval_data: dict,
        model_name: str
    ) -> dict:
        """
        Synthesizes RAG session data into a structured report payload.
        """
        report_id = f"REP-{str(uuid.uuid4())[:8].upper()}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        referenced_docs = list(set(c["metadata"]["filename"] for c in retrieved_chunks))

        # Executive summary from first two paragraphs of LLM response
        paragraphs = [p.strip() for p in response.split("\n") if p.strip()]
        if paragraphs:
            summary = " ".join(paragraphs[:2])
            if len(summary) > 250:
                summary = summary[:247] + "..."
        else:
            summary = "No response content to summarize."

        # Risk-based action recommendation
        risk_level = risk_data.get("risk_level", "LOW")
        if risk_level == "HIGH":
            recommended_actions = [
                "1. Escalate directly to the Board Compliance Committee and Chief Compliance Officer.",
                "2. Initiate a targeted operational audit on the flagged controls immediately.",
                "3. Prepare an incident reporting file for regulatory disclosure submissions.",
                "4. Restructure access policies or suspend activities identified as violating rules."
            ]
        elif risk_level == "MEDIUM":
            recommended_actions = [
                "1. Refer this policy gap or exception to business unit heads for procedural review.",
                "2. Deploy updated control guidelines to relevant personnel.",
                "3. Perform a targeted sample review of activities over the past 30 days.",
                "4. Monitor control indicators at the next operational committee meeting."
            ]
        else:
            recommended_actions = [
                "1. Document findings in the annual compliance self-assessment file.",
                "2. Standardize current operations as per current guidelines.",
                "3. Maintain monitoring schedules as per standard operating procedures."
            ]

        return {
            "report_id": report_id,
            "timestamp": timestamp,
            "query": query,
            "response": response,
            "risk_data": risk_data,
            "retrieved_chunks": retrieved_chunks,
            "eval_data": eval_data,
            "model_name": model_name,
            "referenced_docs": referenced_docs,
            "executive_summary": summary,
            "recommended_actions": recommended_actions
        }

    @staticmethod
    def export_markdown(report_data: dict, output_dir: str = "reports") -> str:
        """
        Exports the report to reports/compliance_report_REP-XXXX.md.
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"compliance_report_{report_data['report_id']}.md")

        referenced_str = ", ".join(report_data["referenced_docs"]) if report_data["referenced_docs"] else "None"
        
        md = []
        md.append(f"# REGULATORY COMPLIANCE REPORT")
        md.append(f"**Report ID:** {report_data['report_id']} | **Generated:** {report_data['timestamp']}\n")
        md.append("---")
        md.append(f"## 1. Executive Summary")
        md.append(report_data["executive_summary"])
        md.append("\n---")
        md.append(f"## 2. Investigation Parameters")
        md.append(f"- **User Query:** \"{report_data['query']}\"")
        md.append(f"- **AI Model:** {report_data['model_name']}")
        md.append(f"- **Referenced Policies:** {referenced_str}")
        md.append("\n---")
        md.append(f"## 3. Compliance Risk Assessment")
        md.append(f"- **Assigned Risk Level:** **{report_data['risk_data']['risk_level']}**")
        md.append(f"- **Rationale:** {report_data['risk_data']['explanation']}")
        md.append("\n---")
        md.append(f"## 4. Key Findings")
        md.append(report_data["response"])
        md.append("\n---")
        md.append(f"## 5. Supporting Evidence & Citations")
        
        for i, c in enumerate(report_data["retrieved_chunks"]):
            m = c["metadata"]
            md.append(f"### Citation #{i+1}")
            md.append(f"- **Document:** `{m.get('filename')}`")
            md.append(f"- **Location:** Page {m.get('page')}")
            md.append(f"- **Hybrid Search Relevance Score:** {c.get('final_score', 0.0):.2%}")
            md.append(f"- **Excerpt:**")
            md.append(f"  > {c['text'].strip()}")
            md.append("")

        md.append("---")
        md.append("## 6. Recommended Actions")
        for action in report_data["recommended_actions"]:
            md.append(action)
        
        md.append("\n---")
        md.append("## 7. RAG System Metrics & Audit Trail")
        metrics = report_data["eval_data"]["metrics"]
        sims = report_data["eval_data"]["similarity_stats"]
        md.append(f"- **Search Precision:** {metrics['precision']:.2f}")
        md.append(f"- **Search Recall:** {metrics['recall']:.2f}")
        md.append(f"- **Search Hit Rate:** {metrics['hit_rate']:.2f}")
        md.append(f"- **Average Retrieval Similarity Score:** {metrics['average_similarity_score']:.4f}")
        md.append(f"- **Mean Hybrid Search Score:** {sims.get('mean_final_hybrid_score', 0.0):.4f}")

        content = "\n".join(md)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath

    @staticmethod
    def export_pdf(report_data: dict, output_dir: str = "reports") -> str:
        """
        Exports the report to reports/compliance_report_REP-XXXX.pdf.
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"compliance_report_{report_data['report_id']}.pdf")

        # Clean string from special characters to avoid PDF generation errors (Latin-1 fallback)
        def clean_txt(t):
            return str(t).encode('latin-1', 'replace').decode('latin-1')

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Header Title
        pdf.set_font("helvetica", "B", 16)
        pdf.multi_cell(pdf.epw, 10, clean_txt("REGULATORY COMPLIANCE REPORT"), align="C")
        pdf.set_font("helvetica", "I", 10)
        pdf.multi_cell(pdf.epw, 6, clean_txt(f"Report ID: {report_data['report_id']} | Generated: {report_data['timestamp']}"), align="C")
        pdf.ln(5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # Section 1: Executive Summary
        pdf.set_font("helvetica", "B", 12)
        pdf.multi_cell(pdf.epw, 8, clean_txt("1. Executive Summary"))
        pdf.set_font("helvetica", "", 10)
        pdf.multi_cell(pdf.epw, 5, clean_txt(report_data["executive_summary"]))
        pdf.ln(4)

        # Section 2: Investigation Parameters
        pdf.set_font("helvetica", "B", 12)
        pdf.multi_cell(pdf.epw, 8, clean_txt("2. Investigation Parameters"))
        pdf.set_font("helvetica", "", 10)
        pdf.multi_cell(pdf.epw, 5, clean_txt(f"Compliance Query: \"{report_data['query']}\""))
        pdf.multi_cell(pdf.epw, 5, clean_txt(f"AI Model Instance: {report_data['model_name']}"))
        referenced_str = ", ".join(report_data["referenced_docs"]) if report_data["referenced_docs"] else "None"
        pdf.multi_cell(pdf.epw, 5, clean_txt(f"Referenced Policies: {referenced_str}"))
        pdf.ln(4)

        # Section 3: Risk Assessment
        pdf.set_font("helvetica", "B", 12)
        pdf.multi_cell(pdf.epw, 8, clean_txt("3. Compliance Risk Assessment"))
        pdf.set_font("helvetica", "B", 10)
        pdf.multi_cell(pdf.epw, 5, clean_txt(f"Assigned Risk Level: {report_data['risk_data']['risk_level']}"))
        pdf.set_font("helvetica", "", 10)
        pdf.multi_cell(pdf.epw, 5, clean_txt(f"Rationale: {report_data['risk_data']['explanation']}"))
        pdf.ln(4)

        # Section 4: Key Findings
        pdf.set_font("helvetica", "B", 12)
        pdf.multi_cell(pdf.epw, 8, clean_txt("4. Key Findings"))
        pdf.set_font("helvetica", "", 10)
        pdf.multi_cell(pdf.epw, 5, clean_txt(report_data["response"]))
        pdf.ln(4)

        # Section 5: Citations
        pdf.set_font("helvetica", "B", 12)
        pdf.multi_cell(pdf.epw, 8, clean_txt("5. Supporting Evidence & Citations"))
        for i, c in enumerate(report_data["retrieved_chunks"]):
            m = c["metadata"]
            pdf.set_font("helvetica", "B", 10)
            pdf.multi_cell(pdf.epw, 5, clean_txt(f"Citation #{i+1}: {m.get('filename')} (Page {m.get('page')})"))
            pdf.set_font("helvetica", "I", 9)
            pdf.multi_cell(pdf.epw, 5, clean_txt(f"Relevance: {c.get('final_score', 0.0):.2%} (Semantic: {c.get('semantic_score', 0.0):.2%}, Keyword: {c.get('keyword_score', 0.0):.2%})"))
            pdf.set_font("helvetica", "", 9)
            pdf.multi_cell(pdf.epw, 4, clean_txt(f"Excerpt: {c['text'].strip()}"))
            pdf.ln(3)

        pdf.ln(2)

        # Section 6: Recommended Actions
        pdf.set_font("helvetica", "B", 12)
        pdf.multi_cell(pdf.epw, 8, clean_txt("6. Recommended Actions"))
        pdf.set_font("helvetica", "", 10)
        for action in report_data["recommended_actions"]:
            pdf.multi_cell(pdf.epw, 5, clean_txt(action))
        pdf.ln(4)

        # Section 7: Audit Info
        pdf.set_font("helvetica", "B", 12)
        pdf.multi_cell(pdf.epw, 8, clean_txt("7. System Audit Information"))
        pdf.set_font("helvetica", "", 9)
        metrics = report_data["eval_data"]["metrics"]
        sims = report_data["eval_data"]["similarity_stats"]
        pdf.multi_cell(pdf.epw, 5, clean_txt(f"RAG Precision: {metrics['precision']:.2f}"))
        pdf.multi_cell(pdf.epw, 5, clean_txt(f"RAG Recall: {metrics['recall']:.2f}"))
        pdf.multi_cell(pdf.epw, 5, clean_txt(f"RAG Hit Rate: {metrics['hit_rate']:.2f}"))
        pdf.multi_cell(pdf.epw, 5, clean_txt(f"Average Similarity Score: {metrics['average_similarity_score']:.4f}"))
        pdf.multi_cell(pdf.epw, 5, clean_txt(f"Mean Hybrid Search Score: {sims.get('mean_final_hybrid_score', 0.0):.4f}"))

        pdf.output(filepath)
        return filepath
