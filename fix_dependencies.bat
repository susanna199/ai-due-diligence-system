@echo off
echo ============================================
echo  FRESH INSTALL - Latest Compatible Versions
echo ============================================

echo Step 1: Removing old conflicting packages...
pip uninstall sentence-transformers transformers torch torchvision torchaudio tokenizers huggingface-hub accelerate -y 2>nul

echo Step 2: Installing latest PyTorch (CPU)...
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo Step 3: Installing latest transformers stack...
pip install transformers tokenizers huggingface-hub

echo Step 4: Installing project packages...
pip install streamlit faiss-cpu pdfplumber PyPDF2 anthropic openai python-dotenv numpy pandas

echo Step 5: Verifying...
python -c "import torch; print('torch', torch.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import faiss; print('faiss OK')"
python -c "import streamlit; print('streamlit', streamlit.__version__)"
python -c "import anthropic; print('anthropic OK')"
python -c "from transformers import AutoTokenizer, AutoModel; print('transformers load OK')"

echo ============================================
echo  All done! Run: streamlit run app.py
echo ============================================
pause
