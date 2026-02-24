"""
app.py
──────
AI Due-Diligence System for Karnataka Land Records
"""

import os
import sys
import json
import time
import logging
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# ── Environment & Path Setup ──────────────────────────────────────────────────
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

# ── Module Imports ─────────────────────────────────────────────────────────────
from modules.document_extractor_new import extract_document_data 
from modules.validator import run_all_validations
from modules.risk_scorer import calculate_risk_score, get_risk_color
from modules.rag_engine import generate_advisory_report
from modules.legal_indexer import LegalIndexer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LegalDoc AI — Land Due Diligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
    h1, h2, h3 { font-family: 'Lora', Georgia, serif !important; letter-spacing: -0.02em; }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f2d4a 100%);
        padding: 2.5rem 2rem; border-radius: 12px; margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 4px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #f0f4ff !important; font-size: 2rem !important; margin: 0 !important; font-weight: 700 !important; }
    .header-badge {
        display: inline-block; background: rgba(96, 165, 250, 0.15); color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.3); padding: 0.25rem 0.75rem;
        border-radius: 999px; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.75rem;
    }
    .advisory-report {
        background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #3b82f6;
        border-radius: 0 10px 10px 0; padding: 1.5rem; white-space: pre-wrap;
        font-size: 0.9rem; line-height: 1.7; color: #334155; max-height: 600px; overflow-y: auto;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1e40af, #2563eb) !important; color: white !important;
        border-radius: 8px !important; font-weight: 600 !important; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ──────────────────────────────────────────────
def init_session_state():
    defaults = {
        "indexer": None,
        "processing_done": False,
        "ec_data": None,
        "khata_data": None,
        "sale_deed_data": None,
        "advisory_report": None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

@st.cache_resource
def get_indexer():
    return LegalIndexer()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ LegalDoc AI")
    st.markdown("**🔑 LLM Configuration**")
    api_key = st.text_input("Google API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
    if api_key: os.environ["GOOGLE_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("**📚 Legal Knowledge Base**")
    indexer = get_indexer()
    if indexer and indexer.vector_count > 0:
        st.success(f"✓ {indexer.vector_count} vectors indexed")

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="header-badge">AI-Powered Due Diligence</div>
    <h1>⚖️ Legal Property Document Analysis System</h1>
    <p>Upload your documents for automated extraction and cross-verification.</p>
</div>
""", unsafe_allow_html=True)

# ── SECTION 1: Multi-Document Upload ──────────────────────────────────────────
st.markdown("---")
st.markdown("## 📜 Step 1 — Upload Property Documents")

col_ec, col_khata, col_sale = st.columns(3)

with col_ec:
    ec_file = st.file_uploader("Upload Encumbrance Certificate (EC)", type=["pdf"])
    if st.session_state.ec_data: st.success("✅ EC Data Cached")

with col_khata:
    khata_file = st.file_uploader("Upload Khata Certificate", type=["pdf"])
    if st.session_state.khata_data: st.success("✅ Khata Data Cached")

with col_sale:
    sale_deed_file = st.file_uploader("Upload Sale Deed", type=["pdf"])
    if st.session_state.sale_deed_data: st.success("✅ Sale Deed Data Cached")

# ── SECTION 2: Process & Analyze ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## ⚙️ Step 2 — Run Due-Diligence Analysis")

all_ready = ec_file and khata_file and sale_deed_file

if st.button("🔍 Perform Full Verification", type="primary", disabled=not all_ready):
    if not api_key:
        st.error("Please provide a Google API Key in the sidebar.")
        st.stop()
    
    status = st.empty()
    progress_bar = st.progress(0)
    
    # Critical: Ensure target directory exists for validator
    upload_dir = Path("user_uploads")
    upload_dir.mkdir(exist_ok=True)
    
    try:
        # Sequential Extraction with Session State "Checkpoints"
        
        # 1. EC Extraction
        if st.session_state.ec_data is None:
            status.info("Extracting EC... (Stage 1/3)")
            data = extract_document_data(ec_file, "ec")
            st.session_state.ec_data = data
            # Sync to local disk for run_all_validations
            with open(upload_dir / "user_ec.json", "w") as f:
                json.dump(data, f, indent=4)
            status.success("✓ EC Cached. Cooling down API for 12s...")
            time.sleep(12) 
        progress_bar.progress(30)

        # 2. Khata Extraction
        if st.session_state.khata_data is None:
            status.info("Extracting Khata... (Stage 2/3)")
            data = extract_document_data(khata_file, "khata")
            st.session_state.khata_data = data
            # Sync to local disk
            with open(upload_dir / "user_khata.json", "w") as f:
                json.dump(data, f, indent=4)
            status.success("✓ Khata Cached. Cooling down...")
            time.sleep(12)
        progress_bar.progress(60)

        # 3. Sale Deed Extraction
        if st.session_state.sale_deed_data is None:
            status.info("Extracting Sale Deed... (Stage 3/3)")
            data = extract_document_data(sale_deed_file, "sale_deed")
            st.session_state.sale_deed_data = data
            # Sync to local disk
            with open(upload_dir / "user_sale_deed.json", "w") as f:
                json.dump(data, f, indent=4)
            status.success("✓ All documents extracted.")
        progress_bar.progress(85)

        # 4. Final Analysis (Sequential Hierarchical Flow)
        status.info("Stage 1: Performing Structural 'Hard' Checks...")
        # Note: validator.run_all_validations now returns a dictionary
        validation_output = run_all_validations(indexer)
        
        # Save to session state for the UI display below
        st.session_state.validation_results = validation_output.get("rule_checks", [])
        
        # Stage 2: Agentic RAG (Only if structural failures exist)
        failures = [f for f in st.session_state.validation_results if not f.get('passed')]
        
        if failures:
            status.info("Stage 2: Structural Failures Detected. Triggering Agentic Legal Research...")
            advisory_report = generate_advisory_report(
                st.session_state.ec_data, 
                st.session_state.khata_data, 
                st.session_state.sale_deed_data, 
                st.session_state.validation_results, 
                {}, # Pass empty risk for now
                indexer
            )
        else:
            status.success("Stage 2: Structure is sound. Applying standard clearance.")
            advisory_report = "The document structure is consistent across the Golden Triangle. No critical legal discrepancies found."

        # Stage 3: Final Risk Scoring
        status.info("Stage 3: Calculating Final Risk Score...")
        st.session_state.risk_data = calculate_risk_score(st.session_state.validation_results, st.session_state.ec_data)
        
        st.session_state.advisory_report = advisory_report
        st.session_state.processing_done = True
        progress_bar.progress(100)
        status.success("Verification Complete!")
        st.rerun()

    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            st.warning("⚠️ Quota hit. Wait 60s and click 'Run' again. Progress is SAVED.")
        elif "'NoneType' object has no attribute 'getvalue'" in str(e):
            st.error("File pointer lost. Please re-upload your PDFs.")
        else:
            st.error(f"Analysis failed: {str(e)}")
            logger.error("Full verification failed", exc_info=True)

# ── RESULTS DISPLAY ───────────────────────────────────────────────────────────
if st.session_state.processing_done:
    st.markdown("---")
    st.markdown("## 📊 Verification & Legal Advisory")

    # A. Display Structural Integrity results (The PASS/FAIL boxes)
    st.markdown("### 🔍 Structural Integrity Results")
    v_results = st.session_state.get("validation_results", [])
    if v_results:
        cols = st.columns(len(v_results))
        for i, check in enumerate(v_results):
            with cols[i]:
                if check.get('passed'):
                    st.success(f"**{check['check_name']}**\n\n✓ PASS")
                else:
                    st.error(f"**{check['check_name']}**\n\n✗ FAIL")

    st.markdown("---")

    # B. Risk & Advisory Report
    r_col1, r_col2 = st.columns([1, 2.5])
    
    with r_col1:
        risk = st.session_state.get("risk_data", {})
        st.markdown("### ⚖️ Risk Assessment")
        st.metric("Total Risk Score", f"{risk.get('score', 0)}/100")
        
        # Color-coded Rating Display
        rating = risk.get('rating', 'UNKNOWN')
        st.subheader(f"Rating: {rating}")
        
        st.info("The score is weighted based on the legal gravity of detected structural failures.")

    with r_col2:
        st.markdown("### ⚖️ Final Legal Advisory & Recommendation")
        st.markdown(f"""<div class='advisory-report'>{st.session_state.advisory_report}</div>""", unsafe_allow_html=True)
    
    # Reset Button
    if st.button("Clear Results & Restart"):
        for key in ["ec_data", "khata_data", "sale_deed_data", "advisory_report", 
                    "processing_done", "validation_results", "risk_data"]:
            st.session_state[key] = None if key != "processing_done" else False
        st.rerun()