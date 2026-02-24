# ⚖️ LegalDoc AI: Property Due-Diligence System
LegalDoc AI is an advanced Retrieval-Augmented Generation (RAG) platform designed to automate the verification of Indian property documents. By cross-referencing Encumbrance Certificates (EC), Khata Certificates, and Sale Deeds against a localized legal knowledge base, the system identifies "Title Breaks," legal encumbrances, and statutory non-compliance.

## 🏗️ System Architecture
The system operates through a modular four-layer pipeline:

**1. Document Intelligence Layer**
Schema-Guided Extraction: Instead of generic OCR, the system uses "Gold Standard" JSON templates to extract specific attributes (Owner names, Survey numbers, Transaction tables) from user uploads.

Template Matching: Ensures that extracted data is consistent with the standard formats used in Karnataka.

**2. Validation Layer**
Cross-Document Verification: Performs the "Golden Triangle" check, ensuring identity and property identifiers match across the EC, Khata, and Sale Deed.

Rule-Based Auditing: Automatically flags active mortgages without releases, suspicious frequent transfers, and Power of Attorney (PoA) risks.

**3. Retrieval Layer (RAG)**
Logical Chunking: Legal acts are indexed by their actual sections and paragraphs to preserve the integrity of legal clauses.

Hybrid Search: Combines semantic vector similarity with exact keyword matching to pinpoint specific statutes in the Registration Act or Stamp Act.

**4. Reasoning & Advisory Layer**
Weighted Risk Scoring: Assigns a risk score (0–20) where legal validity issues (Title Breaks) are weighted significantly higher than administrative typos.

Agentic Reporting: Generates a professional advisory report grounded in retrieved legal provisions and cited statutes.

## 📂 Directory Structure

```
files/
├── app.py                     # Streamlit UI Entry Point
├── user_uploads/              # Extracted user JSONs (auto-generated)
├── knowledge_base/            # Legal Ground Truth
│   ├── central/               # Central Indian Laws (PDFs)
│   ├── karnataka/             # Karnataka State Laws (PDFs)
│   ├── templates/             # JSON Schemas for EC, Khata, Sale Deed
│   └── vector_db/             # FAISS Index and Metadata
└── modules/                   # System Logic
    ├── document_extractor.py  # Template-guided extraction
    ├── legal_indexer.py       # FAISS indexing and search
    ├── validator.py           # Cross-document verification logic
    ├── risk_scorer.py         # Weighted scoring algorithms
    └── rag_engine.py          # AI advisory generation
```

### **Setup & Installation**
1. Environment Configuration
Create a .env file in the root directory and add your API keys:

```
Code snippet:

ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
LLM_PROVIDER=anthropic
```

2. Knowledge Base Initialization
Place your legal PDFs in knowledge_base/central/ and knowledge_base/karnataka/. Run the python notebooks to obtain the vector data files

3. Launching the UI
Run the Streamlit application:

```
Bash
streamlit run app.py
```

**Security & Privacy**
Stateless Processing: User documents are processed in memory and structured JSONs are stored locally in user_uploads/ for the duration of the session.

No Training Data: The system uses Enterprise-tier APIs, ensuring that sensitive land records are never used to train global AI models.

Grounded Accuracy: By using a RAG architecture, the system eliminates hallucinations by forcing the AI to cite actual legal statutes for every claim made.

**Disclaimer**
This system is an AI-assisted tool intended for informational purposes. It is not a substitute for professional legal counsel. Users should conduct independent due diligence before committing to property transactions.
