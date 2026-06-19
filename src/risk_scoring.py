class RiskScorer:
    """
    Evaluates compliance queries and responses to assign a risk category.
    """
    # Specific compliance keywords mapped to levels
    HIGH_KEYWORDS = ["violation", "fraud", "breach", "penalty", "sanction", "money laundering"]
    MEDIUM_KEYWORDS = ["review", "exception", "audit finding", "escalation"]
    LOW_KEYWORDS = ["recommendation", "guidance", "advisory"]

    @classmethod
    def score_content(cls, query: str, response: str) -> dict:
        """
        Analyzes the combined text of the query and the response to assign a risk score.

        Returns:
            A dictionary:
            {
                "risk_level": "LOW" | "MEDIUM" | "HIGH",
                "matched_keywords": list[str],
                "explanation": str
            }
        """
        combined = (query + " " + response).lower()

        matched_high = [kw for kw in cls.HIGH_KEYWORDS if kw in combined]
        matched_medium = [kw for kw in cls.MEDIUM_KEYWORDS if kw in combined]
        matched_low = [kw for kw in cls.LOW_KEYWORDS if kw in combined]

        if matched_high:
            risk_level = "HIGH"
            explanation = (
                f"Critical compliance risk flags detected: {', '.join(matched_high)}. "
                "This indicates potential regulatory violations, financial crime, sanctions, or penalties "
                "that require immediate action and senior officer sign-off."
            )
        elif matched_medium:
            risk_level = "MEDIUM"
            explanation = (
                f"Compliance monitoring review terms detected: {', '.join(matched_medium)}. "
                "This indicates items requiring escalation, operational audits, exceptions, or procedural reviews "
                "within standard review intervals."
            )
        else:
            risk_level = "LOW"
            if matched_low:
                explanation = (
                    f"Standard compliance terms detected: {', '.join(matched_low)}. "
                    "This context suggests advisory guidelines or recommended best practices with minimal operational risk."
                )
            else:
                explanation = (
                    "No immediate compliance risk flags detected in the conversation content. "
                    "Assessed as standard operational context."
                )

        return {
            "risk_level": risk_level,
            "matched_keywords": matched_high + matched_medium + matched_low,
            "explanation": explanation
        }
