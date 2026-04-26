"""Generate technical documentation PDF for the Agentic AI Phishing Detector."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.lib import colors
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "Agentic_AI_Phishing_Detector_Technical_Report.pdf")

# Colors
BLUE = HexColor("#3742fa")
GREEN = HexColor("#00d26a")
RED = HexColor("#ff4757")
YELLOW = HexColor("#ffa502")
PURPLE = HexColor("#9b59b6")
DARK = HexColor("#1a1a2e")
LIGHT_BLUE = HexColor("#e8eaff")
LIGHT_GRAY = HexColor("#f5f5f5")


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'DocTitle', parent=styles['Title'],
        fontSize=22, textColor=BLUE, spaceAfter=6, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'],
        fontSize=11, textColor=HexColor("#666666"), alignment=TA_CENTER, spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        'SectionHeader', parent=styles['Heading1'],
        fontSize=16, textColor=BLUE, spaceBefore=20, spaceAfter=10,
        borderWidth=0, borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        'SubSection', parent=styles['Heading2'],
        fontSize=13, textColor=HexColor("#333333"), spaceBefore=14, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        'SubSubSection', parent=styles['Heading3'],
        fontSize=11, textColor=HexColor("#555555"), spaceBefore=10, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        'BodyText2', parent=styles['Normal'],
        fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        'SmallText', parent=styles['Normal'],
        fontSize=8.5, leading=12, textColor=HexColor("#444444"),
    ))
    styles.add(ParagraphStyle(
        'CodeBlock', parent=styles['Normal'],
        fontName='Courier', fontSize=8, leading=10,
        backColor=LIGHT_GRAY, leftIndent=12, rightIndent=12,
        spaceBefore=4, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        'BulletItem', parent=styles['Normal'],
        fontSize=10, leading=13, leftIndent=20, bulletIndent=8,
        spaceAfter=3,
    ))
    return styles


def make_table(data, col_widths=None, header=True):
    """Create a styled table."""
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_GRAY]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]
    t.setStyle(TableStyle(style_cmds))
    return t


def build_pdf():
    styles = build_styles()
    doc = SimpleDocTemplate(
        OUTPUT_PATH, pagesize=letter,
        topMargin=0.7*inch, bottomMargin=0.7*inch,
        leftMargin=0.8*inch, rightMargin=0.8*inch,
    )
    story = []
    W = doc.width

    # ============================================================
    # TITLE PAGE
    # ============================================================
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("Agentic AI Phishing Detector", styles['DocTitle']))
    story.append(Paragraph("Technical Documentation &amp; System Architecture", styles['DocSubtitle']))
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="60%", thickness=2, color=BLUE, spaceAfter=12, spaceBefore=0))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        "Dual-Engine MCO Architecture for Automated Phishing Detection and Prevention",
        ParagraphStyle('CenterBody', parent=styles['BodyText2'], alignment=TA_CENTER, textColor=HexColor("#666666")),
    ))
    story.append(Spacer(1, 0.4*inch))
    story.append(Paragraph(
        "Based on: <b>\"Agentic AI for Phishing Detection and Prevention\"</b>",
        ParagraphStyle('Ref', parent=styles['BodyText2'], alignment=TA_CENTER),
    ))
    story.append(Paragraph(
        "Loo, Galindo, Romero et al. - Universidad Tecnologica de Honduras (UTH), 2025",
        ParagraphStyle('Ref2', parent=styles['SmallText'], alignment=TA_CENTER, textColor=HexColor("#888888")),
    ))
    story.append(Spacer(1, 0.5*inch))

    badges = [["DistilBERT NLP", "Random Forest", "Agentic AI", "Explainable", "Bilingual (EN/ES)"]]
    bt = Table(badges, colWidths=[W/5]*5)
    bt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, -1), BLUE),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0, white),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BOX', (0, 0), (-1, -1), 1, BLUE),
    ]))
    story.append(bt)

    story.append(PageBreak())

    # ============================================================
    # TABLE OF CONTENTS
    # ============================================================
    story.append(Paragraph("Table of Contents", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=12))
    toc_items = [
        "1. Architecture Overview",
        "2. Phase 1: Monitoring (Input Parsing)",
        "3. Phase 2: Classification (Dual-Engine Analysis)",
        "    3.1 Engine A - Semantic Understanding",
        "    3.2 Engine B - Structural Analysis",
        "4. Feature Fusion &amp; Classification",
        "5. Explainability Layer",
        "6. Phase 3: Optimization (Training Pipeline)",
        "    6.1 Training Data: SpaPhish Dataset",
        "    6.2 Training Process &amp; Results",
        "7. Security Advisor (Claude API)",
        "8. Data Persistence &amp; Dashboard",
        "9. Technology Stack",
    ]
    for item in toc_items:
        indent = 20 if item.startswith("    ") else 0
        story.append(Paragraph(
            item.strip(),
            ParagraphStyle('TOC', parent=styles['BodyText2'], leftIndent=indent, spaceAfter=4),
        ))
    story.append(PageBreak())

    # ============================================================
    # 1. ARCHITECTURE OVERVIEW
    # ============================================================
    story.append(Paragraph("1. Architecture Overview", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10))

    story.append(Paragraph(
        "The system follows a three-phase agentic loop: <b>Monitoring</b> (input parsing and normalization), "
        "<b>Classification</b> (dual-engine semantic + structural analysis fused into a Random Forest classifier), "
        "and <b>Optimization</b> (retraining with labeled samples for continuous improvement).",
        styles['BodyText2'],
    ))

    # Architecture flow diagram as table
    arch_data = [
        ["Phase", "Component", "Description"],
        ["MONITORING", "Input Parsers", "Parse .eml files, raw text, or screenshots (OCR) into normalized email dict"],
        ["CLASSIFICATION", "Engine A: Semantic", "DistilBERT embedding (768-d) + rule-based NLP scoring (urgency, authority, pressure, credentials)"],
        ["CLASSIFICATION", "Engine B: Structural", "URL analysis + typosquatting, SPF/DKIM/DMARC headers, HTML hidden elements (~50 features)"],
        ["CLASSIFICATION", "Feature Fusion", "Concatenate 20-d semantic + 50-d structural = 70-d feature vector"],
        ["CLASSIFICATION", "Classifier", "Random Forest (100 trees) or heuristic fallback with multi-signal boost"],
        ["OPTIMIZATION", "Explainer", "Human-readable explanations, education notes, threat indicator breakdown"],
        ["OPTIMIZATION", "Retraining", "Label samples via API, retrain RF, persist to rf_classifier.pkl"],
    ]
    story.append(make_table(arch_data, col_widths=[W*0.18, W*0.25, W*0.57]))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph(
        "<b>Data flow</b>: Email Input --> Parsers --> Engine A + Engine B (parallel) --> Feature Fusion --> "
        "Random Forest / Heuristic Classifier --> Verdict + Confidence + Action + Explanation --> "
        "SQLite DB (scan history) + Dashboard + Security Advisor chatbot.",
        styles['BodyText2'],
    ))

    story.append(PageBreak())

    # ============================================================
    # 2. PHASE 1: MONITORING
    # ============================================================
    story.append(Paragraph("2. Phase 1: Monitoring (Input Parsing)", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10))

    story.append(Paragraph(
        "Three input parsers normalize any email format into a standardized dictionary containing "
        "subject, body_text, body_html, and headers (if available from .eml).",
        styles['BodyText2'],
    ))

    parser_data = [
        ["Parser", "Input", "Method"],
        ["EML Parser", ".eml file bytes", "Python email.message_from_bytes() with policy.default. Walks multipart MIME, extracts Subject, From, To, Date, Reply-To, body_text, body_html, attachments, raw Message object"],
        ["Text Parser", "Raw subject + body", "Detects embedded HTML via regex, strips tags. Returns standardized dict matching EML parser output"],
        ["OCR Parser", "PNG/JPG screenshot", "Tesseract image_to_string(), computes confidence from image_to_data(), heuristically extracts subject from first lines"],
    ]
    story.append(make_table(parser_data, col_widths=[W*0.15, W*0.18, W*0.67]))

    story.append(PageBreak())

    # ============================================================
    # 3. PHASE 2: CLASSIFICATION
    # ============================================================
    story.append(Paragraph("3. Phase 2: Classification (Dual-Engine Analysis)", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10))

    # 3.1 Engine A
    story.append(Paragraph("3.1 Engine A - Semantic Understanding", styles['SubSection']))
    story.append(Paragraph(
        "Detects linguistic patterns indicative of social engineering attacks. Two paths run in parallel:",
        styles['BodyText2'],
    ))

    story.append(Paragraph("<b>Path 1: DistilBERT Transformer</b> (optional)", styles['SubSubSection']))
    story.append(Paragraph(
        "Model: distilbert-base-uncased (66M parameters). Tokenizes with max_length=512, extracts [CLS] token "
        "embedding producing a <b>768-dimensional</b> dense vector capturing deep semantic relationships.",
        styles['BodyText2'],
    ))

    story.append(Paragraph("<b>Path 2: Rule-Based Pattern Scoring</b> (always runs)", styles['SubSubSection']))

    pattern_data = [
        ["Category", "# Patterns", "Language", "Examples"],
        ["Urgency", "~90", "EN + ES", "\"account suspended\", \"cuenta cerrada\", \"verify now\", \"se cerrara su cuenta\""],
        ["Authority", "~70", "EN + ES", "\"paypal\", \"microsoft\", \"administrador\", \"office 365\""],
        ["Pressure", "~55", "EN + ES", "\"within 24 hours\", \"final notice\", \"sera eliminada\""],
        ["Credential Requests", "~65", "EN + ES", "\"enter your password\", \"haga clic aqui\", \"verify your account\""],
        ["Generic Greetings", "~27", "EN + ES", "\"dear customer\", \"estimado usuario\""],
        ["Reward Lures", "~20", "EN", "\"congratulations\", \"you have won\", \"claim your refund\""],
        ["Grammar Anomalies", "Heuristic", "Any", "Excessive caps (>30%), punctuation (!!!), known misspellings"],
        ["Brand Mentions", "30+", "Any", "PayPal, Microsoft, Amazon, Netflix, Chase, FedEx, etc."],
    ]
    story.append(make_table(pattern_data, col_widths=[W*0.18, W*0.10, W*0.10, W*0.62]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Scoring per category</b>: 0 matches = 0.0, 1 = 0.35, 2 = 0.60, 3 = 0.80, 4+ = 0.90+", styles['BodyText2']))

    story.append(Paragraph("<b>Combined semantic score</b> - weighted average with three boost mechanisms:", styles['BodyText2']))
    boosts = [
        ["Boost", "Condition", "Effect"],
        ["Single-indicator", "Top category score >= 0.35", "Floor at score x 0.75"],
        ["Multi-category", "2+ categories >= 0.3", "Floor at avg(top2) x 0.85"],
        ["Brand co-occurrence", "Brand + urgency OR credentials\nBrand + urgency + credentials", "Floor 0.45\nFloor 0.60"],
    ]
    story.append(make_table(boosts, col_widths=[W*0.20, W*0.40, W*0.40]))

    story.append(PageBreak())

    # 3.2 Engine B
    story.append(Paragraph("3.2 Engine B - Structural Analysis", styles['SubSection']))
    story.append(Paragraph(
        "Three sub-analyzers extract approximately 50 technical features from the email structure:",
        styles['BodyText2'],
    ))

    story.append(Paragraph("<b>URL Analyzer</b> (15 features)", styles['SubSubSection']))
    url_data = [
        ["Check", "Description", "Risk Score"],
        ["IP-based URL", "URL uses numeric IP instead of domain name", "+0.25"],
        ["URL Shortener", "18 known shortener domains (bit.ly, t.co, etc.)", "+0.15"],
        ["Suspicious TLD", "24 TLDs common in phishing (.xyz, .top, .buzz, etc.)", "+0.15"],
        ["Typosquatting", "Levenshtein distance <= 2 against 45+ brand domains", "+0.25"],
        ["Free Hosting", "30+ platforms (weebly.com, wix.com, wordpress.com, etc.)", "+0.30"],
        ["@ Symbol", "@ in URL domain section (credential theft technique)", "+0.20"],
        ["Excessive Subdomains", "More than 3 subdomain levels", "+0.10"],
        ["Suspicious Path", "Keywords: login, verify, password, account (EN + ES)", "+0.10"],
        ["No HTTPS", "URL uses HTTP instead of HTTPS", "+0.05"],
    ]
    story.append(make_table(url_data, col_widths=[W*0.20, W*0.60, W*0.20]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Header Analyzer</b> (15 features)", styles['SubSubSection']))
    header_data = [
        ["Check", "Description", "Risk Score"],
        ["SPF Fail", "SPF authentication check failed", "+0.20"],
        ["DKIM Fail", "DKIM signature verification failed", "+0.20"],
        ["DMARC Fail", "DMARC policy check failed", "+0.15"],
        ["Brand Mismatch", "Display name contains brand but domain doesn't match (50 brands from JSON)", "+0.30"],
        ["Reply-To Mismatch", "Reply-To domain differs from From domain", "+0.20"],
        ["Suspicious X-Mailer", "phpmailer, swiftmailer, mass/bulk mailer", "+0.05"],
        ["Missing Headers", "Missing Message-ID or Date headers", "+0.05 each"],
        ["Suspicious Chain", "Received hop count > 8 or = 0", "+0.10"],
    ]
    story.append(make_table(header_data, col_widths=[W*0.20, W*0.60, W*0.20]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>HTML Analyzer</b> (15 features)", styles['SubSubSection']))
    html_data = [
        ["Check", "Description", "Risk Score"],
        ["Hidden Elements", "7 CSS patterns: display:none, visibility:hidden, opacity:0, etc.", "+0.15"],
        ["External Forms", "form action pointing to external domain", "+0.30"],
        ["Tracking Pixels", "1x1 images used for tracking", "+0.10"],
        ["Obfuscated JS", "eval(), document.write, fromCharCode, atob, hex/unicode escapes", "+0.25"],
        ["Iframes", "Embedded iframe elements", "+0.15"],
        ["Base64 Content", "Encoded content blocks (>2 occurrences)", "+0.10"],
    ]
    story.append(make_table(html_data, col_widths=[W*0.20, W*0.60, W*0.20]))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Combined structural score</b>: URL 40% + Header 35% + HTML 25%",
        styles['BodyText2'],
    ))

    story.append(PageBreak())

    # ============================================================
    # 4. FEATURE FUSION & CLASSIFICATION
    # ============================================================
    story.append(Paragraph("4. Feature Fusion &amp; Classification", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10))

    story.append(Paragraph(
        "The two engines produce a <b>70-dimensional feature vector</b>:",
        styles['BodyText2'],
    ))
    feat_data = [
        ["Source", "Dimension", "Content"],
        ["Semantic (rule-based)", "20 features", "Score + count for each of 6 categories, grammar score, combined score, brand mention count, padding"],
        ["Structural", "50 features", "15 URL + 15 header + 15 HTML + 5 cross-engine interaction features"],
        ["Total", "70 features", "Concatenated into a single vector for classification"],
    ]
    story.append(make_table(feat_data, col_widths=[W*0.22, W*0.16, W*0.62]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Classification Methods</b>", styles['SubSubSection']))
    class_data = [
        ["Method", "When Used", "How It Works"],
        ["Random Forest", "When models/rf_classifier.pkl exists (after training)", "100 estimators, max_depth=20, balanced class weights. Uses predict_proba() for confidence score"],
        ["Heuristic Fallback", "No trained model available", "Weighted: semantic 35% + URL 30% + header 20% + HTML 15%. Multi-signal boost (2+ engines > 0.2 = x1.15). Non-linear scaling above 0.2"],
    ]
    story.append(make_table(class_data, col_widths=[W*0.18, W*0.32, W*0.50]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Decision Thresholds &amp; Action Mapping</b>", styles['SubSubSection']))
    thresh_data = [
        ["Confidence", "Verdict", "Action"],
        [">= 0.40", "PHISHING", "Quarantine (if > 75%) or Alert"],
        [">= 0.22", "SUSPICIOUS", "Alert - flag for manual review"],
        ["< 0.22", "LEGITIMATE", "Pass - deliver normally"],
    ]
    story.append(make_table(thresh_data, col_widths=[W*0.20, W*0.25, W*0.55]))

    story.append(PageBreak())

    # ============================================================
    # 5. EXPLAINABILITY
    # ============================================================
    story.append(Paragraph("5. Explainability Layer", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10))

    story.append(Paragraph(
        "Every classification includes three human-readable outputs designed to make the system's "
        "decisions transparent and educational:",
        styles['BodyText2'],
    ))

    explain_data = [
        ["Output", "Description"],
        ["Explanation", "Natural language paragraph citing specific indicators detected (e.g., urgency language, authority impersonation, suspicious URL on free hosting platform)"],
        ["Education Note", "Attack-type-specific teaching content: urgency tactics, credential harvesting, typosquatting, brand spoofing. Multiple notes joined for multi-vector attacks"],
        ["Threat Indicators", "5-category breakdown with scores and matched details:\n- Suspicious URLs (score + flagged URLs)\n- Urgency/Authority Language (weighted: urgency 40% + authority 30% + pressure 30%)\n- Header Anomalies (SPF/DKIM/DMARC + mismatch details)\n- Grammatical Anomalies (caps ratio, misspellings)\n- HTML Suspicious (hidden elements, external forms)"],
    ]
    story.append(make_table(explain_data, col_widths=[W*0.18, W*0.82]))

    story.append(PageBreak())

    # ============================================================
    # 6. PHASE 3: OPTIMIZATION
    # ============================================================
    story.append(Paragraph("6. Phase 3: Optimization (Training Pipeline)", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10))

    story.append(Paragraph("6.1 Training Data: SpaPhish Dataset", styles['SubSection']))
    story.append(Paragraph(
        "The model was trained on the <b>SpaPhish Dataset (DiB)</b>, a curated corpus of Spanish-language "
        "emails designed for phishing research:",
        styles['BodyText2'],
    ))

    train_data = [
        ["Metric", "Value"],
        ["Total Samples", "1,395"],
        ["Phishing (label=1)", "731 (52.4%)"],
        ["Legitimate (label=0)", "664 (47.6%)"],
        ["Language", "Spanish"],
        ["Source Columns", "hash, subject, body, date, url_count, urls, attachments_count, hops_count, Label"],
    ]
    story.append(make_table(train_data, col_widths=[W*0.30, W*0.70]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("6.2 Training Process &amp; Results", styles['SubSection']))
    story.append(Paragraph(
        "The training process runs each sample through the full analysis pipeline to generate feature vectors:",
        styles['BodyText2'],
    ))

    steps = [
        "1. <b>Load CSV</b>: Parse all 1,395 rows from the semicolon-delimited SpaPhish dataset",
        "2. <b>For each sample</b>: Run through semantic + structural engines:",
        "    - parse_text_input(subject, body) --> normalized email dict",
        "    - analyze_semantics(subject, body) --> rule scores + combined score",
        "    - analyze_structure(parsed) --> URL/header/HTML scores + 50-feature vector",
        "    - _rule_scores_to_features(rule_scores) --> 20-feature semantic vector",
        "3. <b>Concatenate</b>: 20 semantic + 50 structural = 70-feature vector per sample",
        "4. <b>Train RF</b>: RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced')",
        "5. <b>Cross-validate</b>: 5-fold CV with F1 scoring",
        "6. <b>Serialize</b>: Save to models/rf_classifier.pkl",
    ]
    for step in steps:
        story.append(Paragraph(step, styles['BulletItem']))
    story.append(Spacer(1, 10))

    results_data = [
        ["Metric", "Value"],
        ["Cross-validation F1 (mean)", "0.8574"],
        ["Cross-validation F1 (std)", "0.0207"],
        ["Feature Count", "70"],
        ["Model File", "models/rf_classifier.pkl"],
        ["Classifier", "RandomForestClassifier (scikit-learn)"],
        ["Estimators", "100"],
        ["Max Depth", "20"],
        ["Class Weight", "balanced"],
    ]
    story.append(make_table(results_data, col_widths=[W*0.35, W*0.65]))

    story.append(PageBreak())

    # ============================================================
    # 7. SECURITY ADVISOR
    # ============================================================
    story.append(Paragraph("7. Security Advisor (Claude API)", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10))

    story.append(Paragraph(
        "A right-side panel chatbot powered by <b>Claude Sonnet</b> (Anthropic API) that provides "
        "proactive educational feedback after each analysis:",
        styles['BodyText2'],
    ))

    advisor_items = [
        "<b>Proactive Analysis</b>: After each scan, automatically sends analysis context to Claude for educational feedback",
        "<b>Verdict Summary</b>: Posts local system messages with verdict, confidence, and key triggers",
        "<b>Educational Response</b>: Explains what specific indicators mean in simple terms and teaches users how to spot similar attacks",
        "<b>Interactive Q&amp;A</b>: Responds to freeform security questions with full analysis context",
        "<b>Model</b>: claude-sonnet-4-20250514 with max_tokens=1024",
        "<b>Authentication</b>: User provides their own Anthropic API key (stored in browser localStorage only)",
    ]
    for item in advisor_items:
        story.append(Paragraph("- " + item, styles['BulletItem']))

    story.append(Spacer(1, 0.2*inch))

    # ============================================================
    # 8. DATA PERSISTENCE
    # ============================================================
    story.append(Paragraph("8. Data Persistence &amp; Dashboard", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10))

    story.append(Paragraph(
        "SQLite database (data/phishing_detector.db) provides persistent storage across server restarts:",
        styles['BodyText2'],
    ))

    db_data = [
        ["Table", "Columns", "Purpose"],
        ["scan_history", "id, timestamp, input_type, subject, body_preview, verdict, confidence, recommended_action, explanation, education_note, threat_indicators (JSON), raw_features (JSON), analysis_time_seconds", "Every analysis result persisted for dashboard KPIs and scan history"],
        ["training_samples", "id, timestamp, subject, body_preview, label (0/1), features (JSON)", "Labeled samples for Random Forest retraining"],
    ]
    story.append(make_table(db_data, col_widths=[W*0.15, W*0.55, W*0.30]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "The <b>Dashboard</b> displays: total emails analyzed, phishing/suspicious/legitimate counts, "
        "phishing detection rate, average confidence, average threat indicator scores, and a recent analyses table. "
        "The <b>History</b> page provides paginated scan logs with filtering and full detail view.",
        styles['BodyText2'],
    ))

    story.append(PageBreak())

    # ============================================================
    # 9. TECHNOLOGY STACK
    # ============================================================
    story.append(Paragraph("9. Technology Stack", styles['SectionHeader']))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10))

    tech_data = [
        ["Layer", "Technology", "Purpose"],
        ["Backend", "Python 3.10+ / FastAPI / Uvicorn", "REST API web framework and ASGI server"],
        ["NLP Model", "DistilBERT (HuggingFace Transformers)", "768-dim semantic embeddings (optional)"],
        ["Deep Learning", "PyTorch", "Powers DistilBERT inference (optional)"],
        ["ML Classifier", "scikit-learn RandomForestClassifier", "100 estimators, balanced class weights"],
        ["Numerical", "NumPy", "Feature vector operations"],
        ["Email Parsing", "Python email stdlib", ".eml file parsing with policy.default"],
        ["URL Analysis", "tldextract + python-Levenshtein", "Domain extraction and typosquatting detection"],
        ["OCR", "Tesseract-OCR + pytesseract + Pillow", "Screenshot text extraction"],
        ["AI Chatbot", "Anthropic Claude API", "Security advisor (claude-sonnet-4-20250514)"],
        ["Database", "SQLite", "Persistent scan history and training samples"],
        ["Frontend", "Tailwind CSS (CDN) + Vanilla JavaScript", "Single-page application (~1290 lines)"],
        ["Data Validation", "Pydantic", "Request/response model validation"],
    ]
    story.append(make_table(tech_data, col_widths=[W*0.14, W*0.36, W*0.50]))
    story.append(Spacer(1, 0.3*inch))

    # API Endpoints
    story.append(Paragraph("<b>API Endpoints</b>", styles['SubSection']))
    api_data = [
        ["Method", "Path", "Description"],
        ["GET", "/", "Serve frontend SPA"],
        ["GET", "/api/health", "Health check"],
        ["POST", "/api/analyze/eml", "Upload .eml file for analysis"],
        ["POST", "/api/analyze/text", "Analyze raw text (JSON: subject + body)"],
        ["POST", "/api/analyze/screenshot", "Upload screenshot for OCR + analysis"],
        ["GET", "/api/dashboard/stats", "KPI statistics from database"],
        ["GET", "/api/scans", "Paginated scan history"],
        ["GET", "/api/scans/{id}", "Single scan detail"],
        ["POST", "/api/chat", "Claude AI security advisor"],
        ["POST", "/api/train/label", "Submit labeled sample for training"],
        ["POST", "/api/train/run", "Train Random Forest on accumulated samples"],
        ["GET", "/api/train/stats", "Training data statistics"],
    ]
    story.append(make_table(api_data, col_widths=[W*0.10, W*0.30, W*0.60]))

    story.append(Spacer(1, 0.5*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#cccccc"), spaceAfter=6))
    story.append(Paragraph(
        "Agentic AI Phishing Detector v1.0 -- Based on research by Loo, Galindo, Romero et al. -- "
        "Universidad Tecnologica de Honduras (UTH), 2025",
        ParagraphStyle('Footer', parent=styles['SmallText'], alignment=TA_CENTER, textColor=HexColor("#999999")),
    ))

    # Build
    doc.build(story)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build_pdf()
    print(f"PDF generated: {path}")
