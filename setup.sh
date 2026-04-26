#!/bin/bash
# Agentic AI Phishing Detector — Setup Script
# Reference: Loo, Galindo, Romero et al. (2025)

echo "========================================="
echo "  Agentic AI Phishing Detector - Setup"
echo "========================================="

echo ""
echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/3] Downloading DistilBERT model (one-time download)..."
python -c "
from transformers import DistilBertModel, DistilBertTokenizer
print('Downloading DistilBERT tokenizer...')
DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print('Downloading DistilBERT model...')
DistilBertModel.from_pretrained('distilbert-base-uncased')
print('Model downloaded successfully.')
"

echo ""
echo "[3/3] Verifying installation..."
python -c "
import fastapi, uvicorn, torch, transformers, sklearn, tldextract, numpy
print('All core dependencies verified.')
print(f'  FastAPI:      {fastapi.__version__}')
print(f'  PyTorch:      {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  scikit-learn: {sklearn.__version__}')
print(f'  NumPy:        {numpy.__version__}')
"

echo ""
echo "========================================="
echo "  Setup complete!"
echo ""
echo "  Run the application with:"
echo "    uvicorn app.main:app --reload --port 8000"
echo ""
echo "  Then open: http://localhost:8000"
echo "========================================="
