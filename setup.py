"""
One-command setup script for the Agentic AI Phishing Detector.

Usage: python setup.py
"""

import subprocess
import sys


def run(cmd, desc):
    print(f"\n[*] {desc}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[!] Warning: {desc} encountered an issue (exit code {result.returncode})")
    return result.returncode


def main():
    print("=" * 50)
    print("  Agentic AI Phishing Detector — Setup")
    print("=" * 50)

    # Install dependencies
    run(f"{sys.executable} -m pip install -r requirements.txt", "Installing dependencies")

    # Download DistilBERT model
    print("\n[*] Downloading DistilBERT model...")
    try:
        from transformers import DistilBertModel, DistilBertTokenizer
        DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        DistilBertModel.from_pretrained("distilbert-base-uncased")
        print("[+] DistilBERT model downloaded successfully")
    except Exception as e:
        print(f"[!] Could not download model: {e}")
        print("    The app will use heuristic fallback mode")

    # Verify
    print("\n[*] Verifying installation...")
    try:
        import fastapi, torch, transformers, sklearn, numpy
        print(f"    FastAPI:      {fastapi.__version__}")
        print(f"    PyTorch:      {torch.__version__}")
        print(f"    Transformers: {transformers.__version__}")
        print(f"    scikit-learn: {sklearn.__version__}")
        print(f"    NumPy:        {numpy.__version__}")
    except ImportError as e:
        print(f"[!] Missing dependency: {e}")

    print("\n" + "=" * 50)
    print("  Setup complete!")
    print()
    print("  Run: uvicorn app.main:app --reload --port 8000")
    print("  Open: http://localhost:8000")
    print("=" * 50)


if __name__ == "__main__":
    main()
