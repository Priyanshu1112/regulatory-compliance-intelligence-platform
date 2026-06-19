import json
import pandas as pd
import altair as alt
from collections import Counter

class DashboardVisualizer:
    """
    Utility module that constructs Altair chart objects for the compliance intelligence dashboard.
    """

    @staticmethod
    def plot_risk_distribution(df: pd.DataFrame) -> alt.Chart:
        """
        Creates a bar chart of Risk Level distribution.
        """
        if df.empty:
            return None

        # Standardize and count risk levels
        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]

        # Color mapping configuration
        color_scale = alt.Scale(
            domain=["HIGH", "MEDIUM", "LOW"],
            range=["#EF4444", "#F59E0B", "#10B981"]  # Tailwind red, amber, emerald
        )

        chart = alt.Chart(risk_counts).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Risk Level:O", sort=["HIGH", "MEDIUM", "LOW"], axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Count:Q", title="Number of Queries"),
            color=alt.Color("Risk Level:N", scale=color_scale, legend=None),
            tooltip=["Risk Level", "Count"]
        ).properties(
            title="Risk Distribution of Inquiries",
            height=300
        )
        return chart

    @staticmethod
    def plot_query_trends(df: pd.DataFrame) -> alt.Chart:
        """
        Plots the volume of compliance queries over time.
        """
        if df.empty:
            return None

        # Convert to datetime and aggregate by date
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["timestamp"]).dt.date
        trend_df = df_copy.groupby("date").size().reset_index(name="Volume")
        trend_df["date"] = pd.to_datetime(trend_df["date"])

        chart = alt.Chart(trend_df).mark_line(point=True, color="#3B82F6").encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d")),
            y=alt.Y("Volume:Q", title="Query Count"),
            tooltip=["date:T", "Volume:Q"]
        ).properties(
            title="Query Volume Trends",
            height=300
        )
        return chart

    @staticmethod
    def plot_referenced_documents(df: pd.DataFrame) -> alt.Chart:
        """
        Displays the most frequently referenced documents in RAG search outputs.
        """
        if df.empty:
            return None

        doc_counter = Counter()
        for retrieved_str in df["retrieved_sources"].dropna():
            try:
                sources = json.loads(retrieved_str)
                for s in sources:
                    filename = s.get("filename")
                    if filename:
                        doc_counter[filename] += 1
            except Exception:
                pass

        if not doc_counter:
            return None

        doc_df = pd.DataFrame(doc_counter.items(), columns=["Document", "References"])
        doc_df = doc_df.sort_values(by="References", ascending=False).head(10)

        chart = alt.Chart(doc_df).mark_bar(color="#6366F1", cornerRadiusBottomRight=3, cornerRadiusTopRight=3).encode(
            x=alt.X("References:Q", title="Reference Count"),
            y=alt.Y("Document:N", sort="-x", title=None),
            tooltip=["Document", "References"]
        ).properties(
            title="Top 10 Referenced Compliance Documents",
            height=300
        )
        return chart

    @staticmethod
    def plot_retrieval_quality_metrics(df: pd.DataFrame) -> alt.Chart:
        """
        Plots Search Precision, Recall, and Similarity Scores over time.
        """
        if df.empty:
            return None

        records = []
        for _, row in df.iterrows():
            try:
                eval_data = json.loads(row["eval_metrics"])
                metrics = eval_data.get("metrics", {})
                records.append({
                    "timestamp": pd.to_datetime(row["timestamp"]),
                    "Precision": metrics.get("precision", 0.0),
                    "Recall": metrics.get("recall", 0.0),
                    "Similarity": metrics.get("average_similarity_score", 0.0)
                })
            except Exception:
                pass

        if not records:
            return None

        metrics_df = pd.DataFrame(records)
        melted_df = metrics_df.melt(
            id_vars=["timestamp"],
            value_vars=["Precision", "Recall", "Similarity"],
            var_name="Metric",
            value_name="Value"
        )

        chart = alt.Chart(melted_df).mark_line(strokeWidth=2).encode(
            x=alt.X("timestamp:T", title="Timeline"),
            y=alt.Y("Value:Q", title="Metric Value", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metric:N", scale=alt.Scale(
                domain=["Precision", "Recall", "Similarity"],
                range=["#3B82F6", "#10B981", "#8B5CF6"]  # Blue, Emerald, Purple
            )),
            tooltip=["timestamp:T", "Metric", "Value"]
        ).properties(
            title="Retrieval Performance & Quality Analysis",
            height=300
        )
        return chart
