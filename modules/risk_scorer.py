"""
risk_scorer.py
──────────────
Calculates a weighted risk score (0–20) from cross-document validation results.

Updated Rubric:
  +10 — Ownership Match Failure (Title Break) [CRITICAL]
  +10 — Survey Number Mismatch [CRITICAL]
  +5  — Active mortgage (no release)
  +5  — Court order / legal attachment
  +4  — Power of Attorney transaction
  +3  — Suspicious frequent transfers
  +1  — Minor formatting / metadata issues
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ─── Scoring Rules (Updated for Multi-Doc Verification) ──────────────────────

RISK_RULES = [
    {
        "flag": "TITLE BREAK: Ownership Mismatch",
        "check_name": "Cross-Document Ownership Match",
        "points": 10,
        "reason": "Owner names do not match across Sale Deed, Khata, and EC."
    },
    {
        "flag": "CRITICAL: Survey Number Mismatch",
        "check_name": "Survey Number Consistency",
        "points": 10,
        "reason": "Survey/Plot numbers vary between uploaded documents."
    },
    {
        "flag": "Active Mortgage",
        "check_name": "Mortgage Without Release",
        "points": 5,
        "reason": "Active mortgage found without corresponding discharge."
    },
    {
        "flag": "Court Order / Legal Dispute",
        "check_name": "Court Orders / Legal Disputes",
        "points": 5,
        "reason": "Legal attachments or disputes found on property."
    },
    {
        "flag": "Power of Attorney Risk",
        "check_name": "Power of Attorney Sales",
        "points": 4,
        "reason": "PoA-based transfers detected (High Fraud Risk)."
    },
    {
        "flag": "Suspicious Frequent Transfers",
        "check_name": "Suspicious Frequent Transfers",
        "points": 3,
        "reason": "Frequent ownership changes in short window."
    }
]

# -----------changed on 20th Feb 2026 by Susanna ------------------
def calculate_risk_score(validation_results, ec_data):
    """Calculates 0-100 risk based on weighted legal gravity."""
    score = 0
    flags = []
    
    for check in validation_results:
        if not check.get('passed'):
            # Weighted Critical Failures
            if check['risk_level'] == "critical":
                score += 40 
                flags.append(check['check_name'])
            elif check['risk_level'] == "high":
                score += 20
    
    # Cap and Scale
    final_score = min(100, score)
    
    if final_score >= 80: rating = "CRITICAL - DO NOT BUY"
    elif final_score >= 40: rating = "HIGH RISK - PROCEED WITH CAUTION"
    else: rating = "LOW RISK - PROCEED"
    
    return {"score": final_score, "rating": rating, "flags": flags}


def get_risk_color(risk_level: str) -> str:
    colors = {"LOW": "#22c55e", "MODERATE": "#f59e0b", "HIGH": "#f97316", "CRITICAL": "#ef4444"}
    return colors.get(risk_level, "#6b7280")