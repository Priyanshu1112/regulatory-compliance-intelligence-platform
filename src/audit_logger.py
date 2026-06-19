import os
import json
import sqlite3
import pandas as pd
from datetime import datetime

class AuditLogger:
    """
    Manages SQLite-based audit logging for compliance tracking and monitoring.
    """
    def __init__(self, db_path: str = "audit_logs/audit_trail.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """
        Creates the database tables if they do not exist.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Primary audit log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                retrieved_sources TEXT,
                response TEXT,
                risk_level TEXT,
                risk_explanation TEXT,
                eval_metrics TEXT,
                report_generated INTEGER DEFAULT 0,
                report_id TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_query(
        self,
        query: str,
        response: str,
        risk_data: dict,
        retrieved_chunks: list[dict],
        eval_data: dict,
        report_generated: bool = False,
        report_id: str = None
    ):
        """
        Logs a user query and RAG pipeline outputs into SQLite.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Simplify retrieved sources for logging
        sources_list = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            sources_list.append({
                "filename": meta.get("filename", "Unknown"),
                "page": meta.get("page", 0),
                "score": chunk.get("final_score", 0.0),
                "similarity": chunk.get("similarity_score", 0.0),
                "keyword": chunk.get("keyword_score", 0.0)
            })

        sources_json = json.dumps(sources_list)
        eval_json = json.dumps(eval_data)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO audit_logs (
                timestamp, query, retrieved_sources, response, 
                risk_level, risk_explanation, eval_metrics, 
                report_generated, report_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            query,
            sources_json,
            response,
            risk_data.get("risk_level", "LOW"),
            risk_data.get("explanation", ""),
            eval_json,
            1 if report_generated else 0,
            report_id
        ))

        conn.commit()
        conn.close()

    def get_logs(self) -> pd.DataFrame:
        """
        Retrieves all query logs from the database.
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM audit_logs ORDER BY id DESC", conn)
        conn.close()
        return df

    def export_logs(self, export_format: str = "csv", output_dir: str = "audit_logs") -> str:
        """
        Exports audit trails to CSV or JSON formats.
        """
        os.makedirs(output_dir, exist_ok=True)
        df = self.get_logs()
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if export_format.lower() == "csv":
            file_path = os.path.join(output_dir, f"audit_trail_{timestamp_str}.csv")
            df.to_csv(file_path, index=False)
        else:
            file_path = os.path.join(output_dir, f"audit_trail_{timestamp_str}.json")
            df.to_json(file_path, orient="records", indent=4)

        return file_path
        
    def clear_logs(self):
        """
        Resets the database table.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM audit_logs")
        conn.commit()
        conn.close()
