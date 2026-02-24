# --------------updated on 20th Feb, 2026 by Susanna-----------------

"""
rag_engine.py
─────────────
Synthesizes a legal advisory report by combining:
  - Multi-document extracted data (EC, Khata, Sale Deed)
  - Cross-document validation results (Structural Failures)
  - Contextual RAG-retrieved legal provisions via Hybrid Search
"""

import os
import logging
import google.generativeai as genai
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# --- 1. AGENTIC QUERY GENERATION ---

def generate_smart_legal_query(failure_details: str) -> str:
    """Uses Gemini to translate a technical error into a legal research query."""
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""
    Transform this property document discrepancy into a formal legal research query 
    for the Karnataka Land Revenue and Registration Acts:
    Error: "{failure_details}"
    
    Example Output: "Legal validity of sale deeds with survey number mismatches in Karnataka"
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Query generation failed: {e}")
        return failure_details # Fallback to raw error

# --- 2. THE RAG ADVISORY GENERATOR ---

logger = logging.getLogger(__name__)

def generate_advisory_report(ec_data, khata_data, sd_data, validation_results, risk_data, indexer):
    """
    Synthesizes a legal advisory by batching local retrieval and a single LLM call.
    Resolves 'AttributeError' by safely handling both Dict and Object metadata.
    """
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Filter for failed structural checks to drive the research
    failures = [f for f in validation_results if not f.get('passed')]
    
    # --- STAGE 1: LOCAL HYBRID RETRIEVAL (Zero API Cost) ---
    combined_context = ""
    print("\n" + "="*50)
    print("📋 BATCH PROCESSING: LOCAL LEGAL RETRIEVAL")
    
    for fail in failures:
        # Use the specific failure to query your 276 indexed vectors
        search_term = f"{fail['check_name']} {fail['reason']}"
        print(f"🔍 Searching Knowledge Base for: {search_term}")
        
        # Local search doesn't hit Gemini quota
        hits = indexer.search(search_term, k=2) 
        
        # SAFE DATA EXTRACTION: Handles 'dict' vs 'LangChain Document'
        for h in hits:
            if isinstance(h, dict):
                # If hit is a dictionary, use bracket notation
                content = h.get('text') or h.get('page_content') or str(h)
            else:
                # If hit is an object (LangChain), use getattr to prevent crashes
                content = getattr(h, 'page_content', str(h))
            
            combined_context += content + "\n\n"
    
    print("="*50 + "\n")

    # --- STAGE 2: SINGLE BATCH API CALL (Prevents 429 Errors) ---
    # We consolidate data, failures, and retrieved laws into ONE prompt
    batch_prompt = f"""
    You are a Senior Karnataka Land Law Expert. 
    Review the following discrepancies and provide a formal legal opinion.

    === DISCREPANCIES DETECTED ===
    {failures if failures else "No structural errors found."}

    === RELEVANT LEGAL STATUTES (Retrieved via RAG) ===
    {combined_context if combined_context else "No specific statutes found in knowledge base."}

    === PROPERTY METADATA ===
    - Survey Number: {ec_data.get('survey_number', 'N/A')}
    - Village/District: {ec_data.get('village', 'N/A')}, {ec_data.get('district', 'N/A')}
    - Risk Score: {risk_data.get('score', 0)}/100

    === TASK ===
    1. Explain the legal consequences of the discrepancies (e.g., Title Breaks, Section 17 violations).
    2. Ground your advice in the 'Relevant Legal Statutes' provided.
    3. Conclude with a definitive 'Buy' or 'Abort' recommendation.
    4. Maintain a formal, authoritative tone. No markdown code blocks.
    """

    try:
        # This is the ONLY API call made in this function, staying under RPM limits
        response = model.generate_content(batch_prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"RAG Synthesis failed: {e}")
        if "429" in str(e):
            return "⚠️ Quota Exhausted: Please wait 60 seconds and try again."
        return f"Error: The AI could not generate the report. Details: {e}"

def perform_agentic_research(validation_results, indexer):
    """Translates structural failures into legal queries & retrieves context."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Filter for failed checks only
    failures = [f for f in validation_results if not f.get('passed')]
    if not failures:
        return "No structural discrepancies found; legal research skipped."

    # Step A: Agentic Query Generation
    query_prompt = f"Identify the specific Karnataka land laws relevant to these errors: {failures}. Provide 2 precise research queries."
    queries = model.generate_content(query_prompt).text.split('\n')

    # Step B: Hybrid Retrieval
    retrieved_laws = []
    for q in queries:
        if q.strip():
            # Uses your indexer to find Section numbers like 'Section 17'
            hits = indexer.search(q, k=2) 
            retrieved_laws.extend([h.page_content for h in hits])
            
    return "\n\n".join(list(set(retrieved_laws)))