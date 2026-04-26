# CLAUDE CODE PROMPT — Agentic AI Phishing Detector

## OBJECTIVE

Build a complete, production-ready **Agentic AI Phishing Detection** web application. This is a dual-engine system based on the MCO (Monitoring, Classification, Optimization) architecture from the research paper "Agentic AI for Phishing Detection and Prevention" by Loo, Galindo, Romero et al. (Universidad Tecnologica de Honduras, 2025).

The app analyzes emails for phishing indicators using two engines running in parallel, fuses their outputs, classifies the email, and provides explainable results with an AI-powered security advisor chatbot.

---

## ARCHITECTURE OVERVIEW

```
Email Input (.eml / Text / Screenshot-OCR)
    │
    ├──► ENGINE A: Semantic Understanding
    │     • DistilBERT transformer → 768-dim [CLS] embedding
    │     • Rule-based pattern scoring (urgency, authority, pressure, credentials, rewards, grammar)
    │     • Brand mention detection (50+ brands)
    │     • Combined weighted semantic score
    │
    ├──► ENGINE B: Structural Analysis
    │     • URL Analysis: IP-based, shorteners, suspicious TLDs, typosquatting (Levenshtein distance vs 45+ brands)
    │     • Header Analysis: SPF/DKIM/DMARC validation, sender-brand mismatch (loads from brand_domains.json)
    │     • HTML Analysis: hidden elements, external forms, tracking pixels, obfuscated JS, iframes, Base64
    │     • ~50 structural features extracted
    │
    └──► FEATURE FUSION → CLASSIFIER
          • Random Forest (100 estimators) on 768+50 feature vector (when trained)
          • Heuristic fallback: weighted scoring with multi-signal boost and non-linear scaling
          • Thresholds: ≥0.40 = phishing, ≥0.22 = suspicious, <0.22 = legitimate
          │
          └──► OUTPUT
                • Verdict (phishing / suspicious / legitimate)
                • Confidence score (0-100%)
                • Recommended action (quarantine / alert / pass)
                • Threat indicators breakdown (5 categories with scores + details)
                • Human-readable explanation
                • Educational security note
                • Raw features JSON for technical users
```

---

## TECH STACK

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| NLP Model | DistilBERT (HuggingFace Transformers) — optional, falls back to rule-based |
| Deep Learning | PyTorch — optional |
| ML Classifier | scikit-learn RandomForestClassifier (100 estimators, balanced class weights) |
| Numerical | NumPy |
| Email Parsing | Python `email` stdlib with `policy.default` |
| URL Analysis | tldextract, python-Levenshtein |
| OCR | Tesseract-OCR + pytesseract + Pillow |
| AI Chatbot | Anthropic Claude API (claude-sonnet-4-20250514) |
| Database | SQLite (scan history + training samples) |
| Frontend | Single HTML file, Tailwind CSS (CDN), Vanilla JavaScript |
| Data Validation | Pydantic |

---

## PROJECT STRUCTURE

```
phishing-detector/
├── app/
│   ├── __init__.py                  # Empty
│   ├── main.py                      # FastAPI app, all API endpoints, analysis pipeline orchestration
│   ├── database.py                  # SQLite: scan history, training samples, stats queries
│   ├── models/
│   │   ├── __init__.py              # Empty
│   │   ├── semantic_engine.py       # Engine A: DistilBERT + rule-based NLP scoring
│   │   ├── structural_engine.py     # Engine B: combines URL, header, HTML analysis
│   │   ├── classifier.py           # Random Forest + heuristic fallback classifier
│   │   └── explainer.py            # Human-readable explanations + education notes + threat indicators
│   ├── parsers/
│   │   ├── __init__.py              # Empty
│   │   ├── eml_parser.py           # .eml file parser (headers, body text/html, attachments)
│   │   ├── text_parser.py          # Raw text input parser
│   │   └── ocr_parser.py           # Screenshot OCR parser (Tesseract)
│   ├── utils/
│   │   ├── __init__.py              # Empty
│   │   ├── url_analyzer.py         # URL extraction + per-URL risk scoring
│   │   ├── header_analyzer.py      # Email header analysis (SPF/DKIM/DMARC, brand mismatch)
│   │   └── html_analyzer.py        # HTML content analysis (hidden elements, forms, JS, pixels)
│   ├── adversarial/
│   │   ├── __init__.py              # Empty
│   │   ├── evaluator.py            # Phase 2 stub: baseline + adversarial evaluation metrics
│   │   ├── obfuscator.py           # Phase 2 stub: URL/header/HTML obfuscation techniques
│   │   └── paraphraser.py          # Phase 2 stub: semantic paraphrasing attacks
│   └── static/
│       └── index.html              # Single-page frontend (~1290 lines)
├── data/
│   ├── brand_domains.json          # 50 brand entries with legitimate domains
│   ├── phishing_detector.db        # SQLite database (auto-created on first run)
│   └── sample_emails/              # Test .eml files
│       ├── phishing_urgent.eml
│       ├── phishing_brand_spoof.eml
│       ├── phishing_bec.eml
│       ├── legitimate_newsletter.eml
│       └── legitimate_receipt.eml
├── models/                         # Trained ML models (auto-created after training)
│   └── rf_classifier.pkl           # Serialized Random Forest (created via /api/train/run)
├── tessdata/
│   └── eng.traineddata             # Tesseract English language data
├── requirements.txt                # Full dependencies (with torch/transformers)
├── requirements-cloud.txt          # Lightweight (no torch — uses heuristic fallback)
├── Procfile                        # Railway/Heroku: web: uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
├── railway.json                    # Railway deployment config
├── runtime.txt                     # python-3.12.0
├── .gitignore                      # __pycache__, *.pyc, .env, *.db, dist/, build/, venv/
└── SETUP_GUIDE.md                  # Setup instructions for new machines
```

---

## API ENDPOINTS

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve frontend SPA (index.html) |
| GET | `/api/health` | Health check → `{status, service, version, timestamp}` |
| POST | `/api/analyze/eml` | Upload .eml file → full analysis pipeline result |
| POST | `/api/analyze/text` | JSON `{subject, body}` → full analysis pipeline result |
| POST | `/api/analyze/screenshot` | Upload PNG/JPG → OCR + analysis pipeline result |
| GET | `/api/dashboard/stats` | KPI stats: totals, rates, threat breakdown, recent analyses |
| GET | `/api/scans?limit=N&offset=M` | Paginated scan history |
| GET | `/api/scans/{id}` | Single scan detail |
| POST | `/api/chat` | JSON `{message, context?}` + header `X-Api-Key` → Claude AI response |
| POST | `/api/train/label` | JSON `{subject, body, label}` → label sample for training |
| POST | `/api/train/label/eml` | Upload .eml + query param `label=0/1` → label for training |
| POST | `/api/train/run` | Train Random Forest on accumulated labeled samples |
| GET | `/api/train/stats` | Training data statistics |

---

## ANALYSIS PIPELINE DETAIL

### 1. Parsing Layer
- **EML Parser**: Uses `email.message_from_bytes(content, policy=policy.default)`. Extracts Subject, From, To, Date, Reply-To, body_text, body_html, attachments. Walks multipart messages. Falls back to stripping HTML tags if only HTML body exists.
- **Text Parser**: Takes raw subject+body. Detects embedded HTML and strips tags. Returns standardized dict matching EML parser output.
- **OCR Parser**: Uses `pytesseract.image_to_string()` on Pillow Image. Computes OCR confidence from `image_to_data()`. Heuristically extracts subject line from first lines. Configures Tesseract path for Windows (`C:\Program Files\Tesseract-OCR\tesseract.exe`).

### 2. Engine A: Semantic Analysis (`semantic_engine.py`)

**Transformer Path** (when torch+transformers installed):
- Load DistilBERT (`distilbert-base-uncased`) lazily on first use
- Tokenize with max_length=512, truncation=True
- Extract [CLS] token embedding (768-dim) from `last_hidden_state[:,0,:]`

**Rule-Based Path** (always runs):
Six pattern categories, each scored independently:

| Category | # Patterns | Examples |
|----------|-----------|---------|
| Urgency | ~70 | "account suspended", "verify now", "subscription expir", "quota exceeded" |
| Authority | ~60 | "paypal", "microsoft", "verification team", "trust and safety", "cfo" |
| Pressure | ~40 | "within 24 hours", "legal action", "permanently deleted", "failure to comply" |
| Generic Greetings | ~20 | "dear customer", "dear valued member", "to whom it may concern" |
| Reward Lures | ~20 | "congratulations", "you have won", "claim your refund" |
| Credential Requests | ~50 | "enter your password", "for security purposes", "authentication code", "enable content" |

**Scoring per category**: 0 matches=0.0, 1=0.35, 2=0.60, 3=0.80, 4+=0.90+

**Combined semantic score**: Weighted average (urgency 20%, authority 15%, pressure 15%, greetings 10%, rewards 10%, credentials 20%, grammar 10%) with boosts:
- If top score ≥ 0.35 → floor at top_score * 0.75
- If 2+ categories ≥ 0.3 → floor at average of top 2 * 0.85
- If brand mentioned + urgency OR credentials → floor at 0.45
- If brand + urgency + credentials → floor at 0.60

**Brand mention detection**: 30+ brand names scanned in body text (paypal, microsoft, apple, google, amazon, netflix, chase, wells fargo, coinbase, fedex, ups, usps, etc.)

**Grammatical anomalies**: Excessive capitalization ratio >0.3, excessive punctuation (!!!), phishing misspellings ("recieve", "verifiy", "acount", "kindly revert", "do the needful")

### 3. Engine B: Structural Analysis

**URL Analyzer** (`url_analyzer.py`):
- Extracts URLs from text (regex) and HTML (href attributes)
- Per-URL features: is_ip_based, uses_shortener (18 domains), suspicious_tld (24 TLDs), long_domain (>30 chars), has_at_symbol, excessive_subdomains (>3), excessive_hyphens (>2), uses_https, suspicious_path (17 keywords like login/verify/password)
- **Typosquatting**: Levenshtein distance against 45+ top brand domains. Score = 1 - (distance / max_length) when 0 < distance ≤ 2
- Per-URL risk score: weighted sum of indicators (IP=0.25, @=0.20, typosquatting=0.25, shortener=0.15, suspicious_tld=0.15, etc.)
- Overall URL score: 0.7 * max_risk + 0.3 * avg_risk

**Header Analyzer** (`header_analyzer.py`):
- Loads `data/brand_domains.json` at module init: 50 brands with legitimate domains
- Builds keyword→domains mapping for brand mismatch detection
- Checks SPF/DKIM/DMARC from `Authentication-Results` header
- **Sender-brand mismatch**: If display name contains a brand keyword but sender domain doesn't match any legitimate domain for that brand → strong phishing signal (score +0.30)
- Reply-To domain mismatch detection (score +0.20)
- X-Mailer suspicious check (phpmailer, swiftmailer, mass, bulk, sendinblue)
- Missing Message-ID, missing Date, suspicious received chain (>8 or 0 hops)
- Score weights: SPF fail=0.20, DKIM fail=0.20, DMARC fail=0.15, brand mismatch=0.30, reply-to mismatch=0.20

**HTML Analyzer** (`html_analyzer.py`):
- Hidden elements: 7 CSS patterns (display:none, visibility:hidden, opacity:0, etc.)
- External forms: `<form action="http...">` pointing to external domains (score +0.30)
- Tracking pixels: `<img>` tags with width/height 0 or 1
- Iframes (score +0.15)
- Obfuscated JS: 7 patterns (eval, document.write, unescape, fromCharCode, atob, hex/unicode escapes) → score +0.25
- Base64 content blocks (>2 → score +0.10)

**Structural Feature Vector** (~50 features):
- 15 URL features (counts, scores, normalized)
- 15 header features (SPF/DKIM/DMARC mapped to numeric, boolean flags)
- 15 HTML features (normalized counts, scores)
- 5 combined features (cross-engine interactions)

**Combined structural score**: URL 40% + Header 35% + HTML 25%

### 4. Classifier (`classifier.py`)

**Random Forest path** (when `models/rf_classifier.pkl` exists):
- Loads pickled model
- Concatenates embedding (768-d or 20-d rule-based substitute) + structural features (50-d)
- `predict_proba()` → phishing probability
- Thresholds: ≥0.50 phishing, ≥0.30 suspicious, else legitimate

**Heuristic fallback** (default — no trained model):
- Weighted: semantic 35%, URL 30%, header 20%, HTML 15%
- Single-indicator boost: if max > 0.4 → floor at max * 0.70
- Multi-signal boost: 2+ engines > 0.2 → × 1.15; 3+ → × 1.10
- Non-linear scaling: above 0.2, stretch by 1.3×
- Thresholds: ≥0.40 phishing, ≥0.22 suspicious, else legitimate

**`_rule_scores_to_features()`**: Converts rule scores dict to 20-feature vector (score+count for each of 6 categories, grammar score, combined score, brand mention count, padding)

**Action mapping**:
- Phishing + confidence > 85% → quarantine
- Phishing/suspicious + confidence ≥ 50% → alert
- Otherwise → pass

**Training pipeline**:
- `add_training_sample()`: stores features + label in memory
- `train_model(min_samples=6)`: requires both classes, trains RF(100, balanced, max_depth=20), cross-validates if ≥20 samples, saves pickle

### 5. Explainer (`explainer.py`)

**`generate_explanation()`**: Natural language paragraph citing specific indicators detected (urgency phrases, authority references, suspicious URLs, SPF failures, etc.)

**`generate_education_note()`**: Attack-type-specific educational content (urgency tactics, authority impersonation, credential harvesting, typosquatting). Multiple notes joined with " | ".

**`build_threat_indicators()`**: Returns 5-category breakdown:
- suspicious_urls: URL overall score + details
- urgency_language: weighted urgency(40%)+authority(30%)+pressure(30%) + matched phrases
- header_anomalies: header score + details
- grammatical_anomalies: grammar score + issues
- html_suspicious: HTML score + details

### 6. Database (`database.py`)

SQLite with two tables:

**scan_history**: id, timestamp, input_type, subject(200), body_preview(500), verdict, confidence, recommended_action, explanation, education_note, threat_indicators(JSON), raw_features(JSON), analysis_time_seconds

**training_samples**: id, timestamp, subject(200), body_preview(500), label(0/1), features(JSON)

Indexes on timestamp, verdict, label. Auto-creates tables on first connection.

When running as PyInstaller `.exe`, uses `%APPDATA%\PhishingDetector\phishing_detector.db`. Otherwise uses `data/phishing_detector.db`.

---

## FRONTEND SPECIFICATION (`index.html`)

**Single HTML file (~1290 lines)** with Tailwind CSS (CDN) and vanilla JavaScript.

### Layout
- **Two-column layout**: Main content (left, flex:1) + Security Advisor panel (right, 380px fixed width)
- Side panel hidden on screens ≤1024px
- Dark theme with glassmorphism (rgba backgrounds, backdrop-filter:blur, subtle borders)

### Color Palette
```
Background: #0f0f1a
Cards: rgba(26, 26, 46, 0.8) with blur
Accent Blue: #3742fa
Accent Green: #00d26a
Accent Red: #ff4757
Accent Yellow: #ffa502
Text Primary: #e5e7eb (gray-200)
Text Muted: #6b7280 (gray-500)
```

### Navigation
Header bar with 4 nav buttons: **Analyzer** | **Dashboard** | **History** | **About**
Plus MCO badge (blue) and status badge (green/yellow/red)

### Pages

**1. Analyzer Page**
- Tab bar: Upload .eml | Paste Text | Upload Screenshot
- EML tab: drag-and-drop zone with dashed border, click-to-browse, file name display
- Text tab: Subject input + Body textarea
- Screenshot tab: drag-and-drop for images (PNG/JPG)
- "Analyze Email" button (full width, blue, disabled during analysis)
- Loading state: spinner + "Analyzing email through MCO pipeline..."
- Results section (hidden until analysis completes):
  - Verdict banner: gradient background (red/yellow/green) with large verdict text
  - 2-column grid: Confidence gauge (SVG circle, animated stroke-dashoffset) | Recommended action (icon + label + description)
  - Threat indicators: 5 progress bars with scores and detail bullets
  - Explanation paragraph
  - Education note (yellow left border, book icon, hidden for legitimate)
  - Collapsible raw features JSON

**2. Dashboard Page**
- 4 KPI cards: Total Analyzed, Phishing Detected, Suspicious, Legitimate
- 2 large stats: Phishing Detection Rate (%), Average Confidence (%)
- Average Threat Indicator Scores: 5 progress bars from database aggregation
- Recent Analyses table: Time, Subject, Verdict badge, Confidence, Action

**3. History Page**
- Filter dropdown: All / Phishing / Suspicious / Legitimate
- Paginated table (20 per page): ID, Date/Time, Type (emoji), Subject, Verdict badge, Confidence, Action, View link
- Scan detail panel (inline, shown on click):
  - Subject + Body preview
  - 4-column stats: Verdict, Confidence, Input Type, Analysis Time
  - Explanation + Education note
  - Collapsible raw features JSON
- Prev/Next pagination

**4. About Page**
- Hero section: shield icon, title, description, tech badges
- Research Background: paper citation, 4 stats cards (92.5% accuracy, 6.25% FNR, 1.25% FPR, 82.5K emails)
- **MCO Architecture SVG Diagram**: Email Input → Engine A (green) / Engine B (yellow) → Feature Fusion (purple) → Classification (red), with MCO phase labels at top and feedback loop at bottom
- Dual Engine Detail: side-by-side cards for Engine A (green border) and Engine B (yellow border) with bullet lists
- Classification Pipeline: 5-step horizontal flow (Ingest → NLP → Structural → Fusion → Decision)
- Technology Stack: 9-card grid
- Threat Indicator distribution bars from paper (Suspicious URLs 43%, Urgency 32%, Grammar 15%, Other 10%)
- Agentic Action Mapping: 3 cards (Quarantine >85% | Alert 50-85% | Pass <50%)
- **Data Flow SVG Diagram**: Input sources → Parsers → Dual engines → Classifier → Result + SQLite DB + Retraining loop + Claude Advisor
- Adversarial Attack Taxonomy: 4-card grid (Semantic Evasion, Structural Obfuscation, Content Manipulation, Social Engineering)
- Phase 2 Preview: adversarial evaluation roadmap
- Authors footer

### Security Advisor Panel (Right Side)
- **Always visible** (not a floating button)
- Panel header: chat icon, "Security Advisor" title, AI badge, gear icon for settings
- Collapsible API key input (stored in localStorage)
- Message area: system messages (green), bot messages (subtle bg), user messages (blue, right-aligned)
- Chat input with Send button
- **Proactive behavior**: After each analysis completes:
  1. Shows local verdict summary as system message
  2. Lists key triggers (indicators > 20%)
  3. If API key is set: sends context-aware prompt to Claude for educational feedback
  4. Also posts system messages on page navigation (Dashboard, History, About)

### JavaScript Logic
- `showPage(page)`: toggles page visibility, updates nav active states, loads data for Dashboard/History
- `switchTab(tab)`: toggles input panels, updates tab styles
- `analyze()`: constructs FormData or JSON based on tab, POSTs to appropriate endpoint, renders results, triggers proactive feedback
- `proactiveFeedback(data)`: local summary + Claude API call for educational response
- `renderResults(data)`: populates verdict banner, animated gauge, action card, threat indicator bars, explanation, education note, raw JSON
- `loadDashboard()`: fetches `/api/dashboard/stats`, populates KPIs and recent table
- `loadHistory()`: fetches paginated scans, renders table with filter, handles detail view
- `sendChat()`: sends user message + last analysis context to `/api/chat`
- Drag-and-drop handlers on both drop zones
- `escapeHtml()`: XSS prevention for user-generated content

---

## DATA FILES

### `data/brand_domains.json`
50 brand entries, each with name and array of legitimate domains:
```json
{"brands": [
  {"name": "Google", "domains": ["google.com", "gmail.com", "youtube.com", "googleapis.com"]},
  {"name": "Microsoft", "domains": ["microsoft.com", "outlook.com", "live.com", "office365.com", "office.com"]},
  {"name": "Apple", "domains": ["apple.com", "icloud.com", "me.com"]},
  {"name": "Amazon", "domains": ["amazon.com", "amazon.co.uk", "aws.amazon.com"]},
  {"name": "PayPal", "domains": ["paypal.com", "paypal.me"]},
  ...
]}
```
Include all 50: Google, Microsoft, Apple, Amazon, PayPal, Netflix, Facebook, Instagram, LinkedIn, Twitter/X, Chase, Wells Fargo, Bank of America, Citibank, Capital One, American Express, Dropbox, Adobe, Salesforce, Zoom, Slack, GitHub, Yahoo, WhatsApp, Telegram, Coinbase, Binance, Stripe, Shopify, eBay, Walmart, Target, Best Buy, Home Depot, Costco, FedEx, UPS, USPS, DHL, IRS, Social Security, Medicare, DocuSign, Uber, Lyft, Venmo, Cash App, T-Mobile, AT&T, Verizon.

### Sample .eml files
Create 5 test emails in `data/sample_emails/`:
- `phishing_urgent.eml`: Fake PayPal suspension with urgency language
- `phishing_brand_spoof.eml`: Microsoft brand spoofing with domain mismatch
- `phishing_bec.eml`: Business email compromise / CEO fraud
- `legitimate_newsletter.eml`: Real newsletter from a tech company
- `legitimate_receipt.eml`: Order confirmation from Amazon

---

## REQUIREMENTS FILES

### `requirements.txt` (Full — ~2GB with torch)
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
pydantic>=2.0.0
transformers>=4.36.0
torch>=2.1.0
scikit-learn>=1.3.0
numpy>=1.24.0
tldextract>=5.0.0
python-Levenshtein>=0.23.0
Pillow>=10.0.0
pytesseract>=0.3.10
anthropic>=0.86.0
```

### `requirements-cloud.txt` (Lightweight — no torch/transformers)
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
pydantic>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
tldextract>=5.0.0
python-Levenshtein>=0.23.0
Pillow>=10.0.0
pytesseract>=0.3.10
anthropic>=0.86.0
```

---

## CRITICAL IMPLEMENTATION NOTES

1. **The `if isinstance(v, dict)` guard**: In `main.py` line ~128, when iterating `rule_scores.items()`, filter out non-dict entries (like `combined_score` which is a float and `brand_mentions` which is a list):
   ```python
   for k, v in semantic_results.get("rule_scores", {}).items()
   if isinstance(v, dict)
   ```

2. **PyInstaller frozen mode**: Use `getattr(sys, 'frozen', False)` checks in `database.py` (use AppData for DB) and `header_analyzer.py` (use `sys._MEIPASS` for brand_domains.json).

3. **Tesseract path**: In `ocr_parser.py`, set `pytesseract.pytesseract.tesseract_cmd` to `C:\Program Files\Tesseract-OCR\tesseract.exe` if it exists. Also set `TESSDATA_PREFIX` to project's `tessdata/` directory.

4. **Lazy model loading**: DistilBERT is loaded only on first analysis, not at startup. If torch/transformers aren't installed, it falls back to rule-based silently.

5. **Thread safety**: SQLite connection uses `check_same_thread=False` since FastAPI is async.

6. **The `text_lower` variable**: In `compute_rule_based_scores()`, define `text_lower = text.lower()` at the top of the function BEFORE using it in brand mention detection and pattern scoring.

---

## HOW TO RUN

```bash
# Clone and setup
git clone <repo-url>
cd phishing-detector
python -m venv venv
venv/Scripts/activate  # Windows
pip install -r requirements-cloud.txt

# Run
uvicorn app.main:app --reload --port 8000
# Open http://127.0.0.1:8000
```

---

## ADVERSARIAL MODULE STUBS (Phase 2)

The `app/adversarial/` directory contains three stub modules with `NotImplementedError`. They document planned Phase 2 research:

- **evaluator.py**: Baseline vs adversarial evaluation metrics (accuracy, precision, recall, F1, FNR, FPR, ESR, DDI, ROC/AUC)
- **obfuscator.py**: URL obfuscation (percent encoding, homograph, redirects, IP encoding), header manipulation, HTML obfuscation
- **paraphraser.py**: T5/GPT paraphrasing, synonym substitution, sentence restructuring, tone shifting

These are NOT implemented — they serve as architectural documentation for the follow-up paper.
