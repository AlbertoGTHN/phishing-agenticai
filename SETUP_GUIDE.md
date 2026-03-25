# Agentic AI Phishing Detector — Complete Setup Guide

## Prerequisites

| Software | Version | Download |
|----------|---------|----------|
| Python | 3.10 - 3.12 (recommended) | https://www.python.org/downloads/ |
| Tesseract OCR | 5.x | https://github.com/UB-Mannheim/tesseract/wiki |
| Git | latest | https://git-scm.com/downloads |

> **Note:** Python 3.14 works but 3.12 is recommended for maximum compatibility with torch/transformers.

---

## Option A: Setup from GitHub (Recommended)

### Step 1: Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/phishing-detector.git
cd phishing-detector
```

### Step 2: Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

**Full install (with ML model support ~2GB):**
```bash
pip install -r requirements.txt
```

**Lightweight install (no torch/transformers, uses heuristic classifier):**
```bash
pip install -r requirements-cloud.txt
```

### Step 4: Install Tesseract OCR

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (default path: `C:\Program Files\Tesseract-OCR`)
3. Add to PATH: System Properties > Environment Variables > Path > Add `C:\Program Files\Tesseract-OCR`
4. Verify: open new terminal and run `tesseract --version`

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

### Step 5: Create data directory
```bash
mkdir -p data models
```
The `data/brand_domains.json` file should already exist from the repo. The SQLite database (`data/phishing_detector.db`) is auto-created on first run.

### Step 6: (Optional) Set Anthropic API Key
The Security Advisor chatbot requires an Anthropic API key.

**Option 1 — Environment variable (recommended for servers):**
```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-your-key-here

# macOS/Linux
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Option 2 — Enter via the UI:**
Click the gear icon in the Security Advisor panel and paste your key.

Get a key at: https://console.anthropic.com/settings/keys

### Step 7: Run the application
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open your browser at: **http://127.0.0.1:8000**

---

## Option B: Setup from ZIP (No Git Required)

### Step 1: Transfer the project
Copy the entire `phishing-detector/` folder to the new computer via USB, cloud drive, or ZIP.

### Step 2: Follow Steps 2-7 from Option A above.

---

## Option C: Deploy to Cloud (Railway)

### Step 1: Push to GitHub
```bash
cd phishing-detector
git init
git add .
git commit -m "Initial commit: Agentic AI Phishing Detector"
git remote add origin https://github.com/YOUR_USERNAME/phishing-detector.git
git push -u origin main
```

### Step 2: Deploy on Railway
1. Go to https://railway.com and sign in with GitHub
2. Click "New Project" > "Deploy from GitHub Repo"
3. Select `phishing-detector`
4. In Settings > Variables, add:
   - `REQUIREMENTS_FILE` = `requirements-cloud.txt`
   - `ANTHROPIC_API_KEY` = `sk-ant-your-key-here` (optional)
5. In Settings > Networking, generate a public domain
6. Deploy

Railway auto-detects the `Procfile` and starts the server.

> **Note:** OCR (screenshot analysis) may not work on Railway without additional Tesseract setup via a Dockerfile. Text and .eml analysis work fine.

---

## Option D: Docker Deployment (Any Cloud)

### Step 1: Create Dockerfile (already included or create):
```dockerfile
FROM python:3.12-slim

# Install Tesseract OCR
RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 2: Build and run
```bash
docker build -t phishing-detector .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-your-key phishing-detector
```

### Step 3: Deploy to any cloud
```bash
# Google Cloud Run
gcloud run deploy phishing-detector --source .

# AWS (with Docker image pushed to ECR)
# Azure (with Docker image pushed to ACR)
# DigitalOcean App Platform (connect GitHub repo)
```

---

## Project Structure

```
phishing-detector/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app + API endpoints
│   ├── database.py              # SQLite scan history + training data
│   ├── models/
│   │   ├── classifier.py        # Random Forest + heuristic classifier
│   │   ├── explainer.py         # Human-readable threat explanations
│   │   ├── semantic_engine.py   # Engine A: NLP/semantic analysis
│   │   └── structural_engine.py # Engine B: structural feature extraction
│   ├── parsers/
│   │   ├── eml_parser.py        # .eml file parser
│   │   ├── text_parser.py       # Raw text parser
│   │   └── ocr_parser.py        # Screenshot OCR parser
│   ├── utils/
│   │   ├── header_analyzer.py   # Email header analysis (SPF/DKIM/DMARC)
│   │   ├── html_analyzer.py     # HTML content analysis
│   │   └── url_analyzer.py      # URL/domain analysis
│   ├── adversarial/
│   │   ├── evaluator.py         # Adversarial robustness testing
│   │   ├── obfuscator.py        # Text obfuscation techniques
│   │   └── paraphraser.py       # Paraphrase-based evasion
│   └── static/
│       └── index.html           # Single-page frontend (Tailwind CSS)
├── data/
│   ├── brand_domains.json       # 50 brand domains for typosquatting detection
│   ├── phishing_detector.db     # SQLite database (auto-created)
│   └── sample_emails/           # Test .eml files
├── models/                      # Trained ML models (auto-created)
├── tessdata/
│   └── eng.traineddata          # Tesseract English language data
├── requirements.txt             # Full dependencies (with torch)
├── requirements-cloud.txt       # Lightweight dependencies (no torch)
├── Procfile                     # Railway/Heroku process definition
├── railway.json                 # Railway deployment config
├── runtime.txt                  # Python version specification
└── SETUP_GUIDE.md               # This file
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Frontend SPA |
| GET | `/api/health` | Health check |
| POST | `/api/analyze/eml` | Analyze .eml file upload |
| POST | `/api/analyze/text` | Analyze raw email text |
| POST | `/api/analyze/screenshot` | Analyze email screenshot (OCR) |
| GET | `/api/dashboard/stats` | KPI dashboard statistics |
| GET | `/api/history` | Scan history log |
| POST | `/api/chat` | Security Advisor chatbot |
| POST | `/api/feedback` | Submit classification feedback |
| POST | `/api/train` | Train ML model from feedback data |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `tesseract is not installed` | Install Tesseract OCR and add to PATH. Restart terminal. |
| `'float' object has no attribute 'get'` | Update to latest code — this bug was fixed. |
| `name 'text_lower' is not defined` | Update to latest code — this bug was fixed. |
| `Port 8000 already in use` | Kill the process: `taskkill /F /PID <pid>` (Windows) or `kill <pid>` (Linux/Mac). Find PID with `netstat -ano \| findstr :8000` |
| `Credit balance too low` (Anthropic API) | Buy API credits at console.anthropic.com. Ensure API key matches the workspace with credits. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in your virtual environment. |
| Slow first startup | Normal — transformers downloads DistilBERT model (~260MB) on first run. |

---

## Quick Start (Copy-Paste for New Computer)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/phishing-detector.git
cd phishing-detector

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements-cloud.txt

# 4. Run
uvicorn app.main:app --reload --port 8000

# 5. Open browser
# http://127.0.0.1:8000
```
