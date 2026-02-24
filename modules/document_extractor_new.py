import os
import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Environment & Path Setup ──────────────────────────────────────────────────
# Automatically find and load the .env file
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Gemini Client Configuration ───────────────────────────────────────────────
# Uses the newer google-genai SDK for modern features like Gemini 2.5 Flash
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please check your configuration.")

client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemini-2.5-flash"  # As requested

# ── Extraction Logic ──────────────────────────────────────────────────────────

def extract_document_data(pdf_file, doc_type: str) -> dict:
    """
    Extracts structured data from a PDF file using the Gemini File API.
    Handles scanned documents natively using multimodal vision.
    """
    
    # Save a temporary file for the Google File API
    temp_path = BASE_DIR / f"temp_{int(time.time())}_{doc_type}.pdf"
    
    # Check if the input is a Streamlit UploadedFile object or a local path
    if hasattr(pdf_file, 'getvalue'):
        with open(temp_path, "wb") as f:
            f.write(pdf_file.getvalue())
    else:
        # If it's a string path, copy it
        with open(pdf_file, "rb") as src, open(temp_path, "wb") as dst:
            dst.write(src.read())

    try:
        # 1. Upload to Gemini Cloud Storage (Files are stored for 48 hours)
        logger.info(f"Uploading {doc_type} to Google File API...")
        uploaded_file = client.files.upload(file=str(temp_path))
        
        # 2. Define the System Prompt
        # This tells the model how to act and what format to return
        system_instruction = (
            f"You are a professional legal document extractor for Karnataka Land Records. "
            f"Extract all relevant information from the provided {doc_type} document."
        )

        # 3. Perform Generation with Structured Output (JSON Mode)
        logger.info(f"Generating content for {doc_type} using {MODEL_ID}...")
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[uploaded_file, f"Extract data from this {doc_type} into a clear JSON structure."],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json", # Forces valid JSON output
                temperature=0.1,  # Lower temperature for higher accuracy
            )
        )

        # 4. Parse and return JSON data
        return json.loads(response.text)

    except Exception as e:
        logger.error(f"Extraction failed for {doc_type}: {str(e)}")
        raise e

    finally:
        # 5. Local Cleanup of the temporary file
        if temp_path.exists():
            os.remove(temp_path)

# Example Usage
if __name__ == "__main__":
    # Test path - replace with your actual sample path
    test_pdf = BASE_DIR / "knowledge_base" / "templates" / "ec" / "EC_sample.pdf"
    if test_pdf.exists():
        data = extract_document_data(test_pdf, "Encumbrance Certificate")
        print(json.dumps(data, indent=2))
    else:
        print(f"Sample file not found at {test_pdf}")