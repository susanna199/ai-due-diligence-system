# """
# validator.py
# ────────────
# Cross-Document Integrity & Statutory Compliance Engine.

# Key Enhancements:
#   1. Multi-Document Orchestration: Automatically loads extracted JSON data for 
#      EC, Khata, and Sale Deed from the 'user_uploads' directory.
#   2. 'Title Break' Detection: Implements a critical ownership verification 
#      layer that cross-references Purchaser/Owner names across all three 
#      independent data sources to ensure title continuity.
#   3. Spatial Consistency: Verifies that the Survey/Plot numbers are identical 
#      across the 'Golden Triangle' of land records (EC, Khata, Deed).
#   4. Grounded RAG Validation: Performs semantic lookups in the 'knowledge_base' 
#      to check user data against Karnataka-specific land acts and central 
#      statutes.
#   5. Weighted Risk Logic: Prioritizes legal validity issues (Critical) over 
#      administrative typos (Low) for accurate risk scoring.

# This module serves as the analytical core, turning disparate document 
# extractions into a unified legal opinion.
# """


#----------------edited on 20th Feb, 2026 by Susanna -----------------
"""
validator.py
────────────
Strict Cross-Document Integrity Engine for Karnataka Land Records.
"""

import json
import logging
import google.generativeai as genai
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION ---
USER_UPLOADS_DIR = Path(__file__).parent.parent / "user_uploads"

# --- 2. HELPER: STANDARDIZED RESULTS ---
def make_result(check_name, passed, reason, risk_level="low"):
    return {
        "check_name": check_name,
        "passed": passed,
        "reason": reason,
        "risk_level": risk_level,
        "status": "PASS" if passed else "FAIL"
    }

# --- 3. HARD DATA CHECKS (Local Logic) ---

def check_cross_document_ownership(ec: Dict, khata: Dict, sd: Dict) -> Dict:
    """Strictly verifies if the owner is the same across all three docs."""
    
    # EC: Get claimant from the most recent transaction
    txns = ec.get("transactions", [])
    ec_owner = (txns[-1].get("claimant") if txns else "").strip().lower()
    
    # Khata: Check multiple common keys
    khata_owner = (
        khata.get("owner_name") or 
        khata.get("ownership_details", {}).get("owner_name") or 
        khata.get("entries", [{}])[0].get("owner_name", "")
    ).strip().lower()
    
    # Sale Deed: Direct key
    sd_purchaser = (sd.get("purchaser_name") or sd.get("purchaser", {}).get("name", "")).strip().lower()

    # DEBUGGING: This will show up in your terminal
    print(f"DEBUG | EC: '{ec_owner}' | Khata: '{khata_owner}' | Deed: '{sd_purchaser}'")

    if not ec_owner or not khata_owner or not sd_purchaser:
        return make_result(
            "Ownership Match", False, 
            "Data Extraction Incomplete: One or more owner names could not be identified.", 
            "critical"
        )

    # Use 'in' for partial matches (e.g., 'Anil Kumar' vs 'Anil') but verify mismatch first
    if ec_owner != khata_owner or khata_owner != sd_purchaser:
        return make_result(
            "Cross-Document Ownership Match", False,
            f"TITLE BREAK: Mismatch detected. EC: {ec_owner.upper()}, Khata: {khata_owner.upper()}, Deed: {sd_purchaser.upper()}",
            "critical"
        )
    
    return make_result("Cross-Document Ownership Match", True, "Ownership names are consistent.")

def check_survey_consistency(ec: Dict, khata: Dict, sd: Dict) -> Dict:
    """Verifies the Survey Number is identical across documents."""
    ec_s = str(ec.get("survey_number") or "").strip()
    khata_s = str(
        khata.get("survey_number") or 
        khata.get("land_details", {}).get("survey_number", "") or
        khata.get("property_details", {}).get("survey_number", "")
    ).strip()
    
    if not ec_s or not khata_s:
        return make_result("Survey Consistency", False, "Missing survey number in one or more files.", "medium")

    passed = ec_s == khata_s
    return make_result(
        "Survey Number Consistency", passed,
        "Survey numbers match." if passed else f"Mismatch: EC lists {ec_s}, Khata lists {khata_s}.",
        "critical" if not passed else "low"
    )

# --- 4. AGENTIC RAG CHECKS ---

def run_rag_checks(ec_data: Dict, indexer) -> List[Dict]:
    """Generates legal queries based on transaction history."""
    if not indexer or indexer.vector_count == 0:
        return []

    results = []
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Analyze the last 3 transactions for legal red flags
    txns = ec_data.get("transactions", [])[-3:]
    query_gen = f"Analyze these Karnataka land transactions: {txns}. Identify 1 specific legal section to verify."
    
    try:
        search_query = model.generate_content(query_gen).text
        # Perform Hybrid Search in your 276 vectors
        hits = indexer.search(search_query, k=2)
        context = "\n\n".join([h.page_content for h in hits])
        
        verify = model.generate_content(f"Context: {context}\nIs this transaction history compliant with Karnataka law?").text
        
        results.append(make_result("Statutory Compliance (RAG)", "compliant" in verify.lower(), verify[:200] + "...", "medium"))
    except Exception as e:
        logger.error(f"RAG Check Error: {e}")
        
    return results

# --- 5. MASTER RUNNER ---

# def run_all_validations(indexer) -> List[Dict]:
#     """Orchestrates all checks using stored JSON files."""
#     try:
#         with open(USER_UPLOADS_DIR / "user_ec.json", "r") as f: ec = json.load(f)
#         with open(USER_UPLOADS_DIR / "user_khata.json", "r") as f: khata = json.load(f)
#         with open(USER_UPLOADS_DIR / "user_sale_deed.json", "r") as f: sd = json.load(f)
#     except Exception as e:
#         return [make_result("File Access", False, f"Could not load JSON files: {e}", "critical")]

#     # 1. Structural Checks (Zero Cost)
#     checks = [
#         check_cross_document_ownership(ec, khata, sd),
#         check_survey_consistency(ec, khata, sd)
#     ]

#     # 2. RAG Checks (Only if structural checks aren't a total failure)
#     checks.extend(run_rag_checks(ec, indexer))
    
#     return checks

def run_all_validations(indexer):
    """Orchestrates structural 'Hard Checks' using stored JSONs."""
    try:
        # Standardized local loading
        with open("user_uploads/user_ec.json", "r") as f: ec = json.load(f)
        with open("user_uploads/user_khata.json", "r") as f: khata = json.load(f)
        with open("user_uploads/user_sale_deed.json", "r") as f: sd = json.load(f)
    except Exception as e:
        return {"error": str(e), "rule_checks": []}

    results = []
    
    # Check 1: Title Continuity
    sd_p = (sd.get("purchaser_name") or sd.get("purchaser", {}).get("name", "")).lower().strip()
    k_o = (khata.get("owner_name") or khata.get("entries", [{}])[0].get("owner_name", "")).lower().strip()
    title_match = sd_p in k_o or k_o in sd_p
    results.append({
        "check_name": "Cross-Document Title Match",
        "passed": title_match,
        "risk_level": "critical",
        "reason": "Ownership verified." if title_match else f"Mismatch: Deed({sd_p}) vs Khata({k_o})"
    })

    # Check 2: Survey Consistency
    ec_s = str(ec.get("survey_number") or "").strip()
    khata_s = str(khata.get("survey_number") or khata.get("land_details", {}).get("survey_number", "")).strip()
    survey_match = ec_s == khata_s if ec_s and khata_s else False
    results.append({
        "check_name": "Survey Number Consistency",
        "passed": survey_match,
        "risk_level": "critical",
        "reason": "Survey numbers match." if survey_match else f"Mismatch: EC({ec_s}) vs Khata({khata_s})"
    })

    # Return as DICTIONARY to satisfy downstream .get() calls
    return {
        "rule_checks": results,
        "total_passed": sum(1 for r in results if r["passed"]),
        "status": "FAIL" if any(not r["passed"] for r in results) else "PASS"
    }