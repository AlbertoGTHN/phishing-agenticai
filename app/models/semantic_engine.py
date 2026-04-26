"""
Semantic Understanding Engine (Engine A) for Phishing Detection.

Combines DistilBERT transformer embeddings with rule-based semantic
scoring to extract linguistic features indicative of phishing.

Architecture:
- DistilBERT: Extracts 768-dimensional [CLS] embedding
- Rule-based: Scores urgency, authority, pressure, and generic greeting patterns
- Output: Combined semantic feature vector

Reference: Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.
"""

import re
import logging
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Rule-based pattern categories
URGENCY_PHRASES = [
    "immediate action", "account suspended", "verify now",
    "click here immediately", "your account will be closed",
    "unauthorized access detected", "urgent", "act immediately",
    "response required", "action required", "verify your identity",
    "confirm your account", "your account has been compromised",
    "suspicious activity", "security alert", "unusual activity",
    "account will be locked", "account will be terminated",
    "failure to respond", "must be completed",
    "temporarily suspended", "temporarily locked", "restricted access",
    "unusual sign-in", "unrecognized device", "unrecognized login",
    "account at risk", "security breach", "detected unusual",
    "important update", "critical update", "mandatory update",
    "review your account", "secure your account", "protect your account",
    "we noticed", "we detected", "we have detected",
    "if this wasn't you", "if you did not", "if you didn't",
    "account verification", "identity verification",
    "your account was accessed", "someone tried to sign in",
    "payment declined", "payment failed", "transaction failed",
    "invoice attached", "invoice due", "overdue payment",
    "shipment notification", "delivery failed", "package held",
    "confirm delivery", "reschedule delivery", "update delivery",
    # Additional modern phishing patterns
    "requires your attention", "needs your attention",
    "pending verification", "incomplete verification",
    "account on hold", "account restricted", "access restricted",
    "will be disabled", "will be suspended", "will be restricted",
    "has been limited", "has been restricted", "has been flagged",
    "resolve this issue", "resolve immediately",
    "your order", "order confirmation", "order problem",
    "subscription expir", "membership expir", "trial expir",
    "renew your", "renewal notice", "auto-renewal",
    "unable to process", "could not process", "processing error",
    "sign-in attempt", "login attempt", "access attempt",
    "new device", "new location", "new browser",
    "take action", "requires action", "needs action",
    "contact us immediately", "call us immediately",
    "do not ignore", "do not disregard",
    "irregular activity", "fraudulent activity",
    "compromised", "breach", "leaked",
    "password expir", "credentials expir",
    "storage full", "mailbox full", "quota exceeded",
    "upgrade required", "update required",
    # Spanish phishing patterns (urgency)
    "cuenta suspendida", "cuenta cerrada", "se cerrará su cuenta",
    "se cerrara su cuenta", "cuenta será cancelada", "cuenta sera cancelada",
    "verificar su cuenta", "verifique su cuenta", "verificación requerida",
    "verificacion requerida", "acción inmediata", "accion inmediata",
    "actividad sospechosa", "actividad inusual", "actividad no autorizada",
    "acceso no autorizado", "acceso restringido", "acceso limitado",
    "cuenta en riesgo", "alerta de seguridad", "aviso de seguridad",
    "actualización obligatoria", "actualizacion obligatoria",
    "si no autorizó", "si no autorizo", "si usted no realizó",
    "si usted no realizo", "si no fue usted",
    "pago rechazado", "pago fallido", "transacción fallida",
    "factura adjunta", "pago pendiente", "entrega fallida",
    "cierre de cuenta", "cancelar su cuenta", "cancelar su correo",
    "suspender su cuenta", "bloquear su cuenta", "desactivar su cuenta",
    "confirmar su identidad", "confirme su identidad",
    "actualizar su información", "actualizar su informacion",
    "iniciar sesión", "iniciar sesion",
]

AUTHORITY_PHRASES = [
    "it department", "ceo", "security team", "paypal",
    "microsoft", "apple", "google", "amazon", "netflix",
    "internal revenue service", "irs", "social security",
    "human resources", "hr department", "compliance team",
    "legal department", "board of directors", "chief financial officer",
    "system administrator", "tech support",
    "fraud department", "account services",
    "wells fargo", "chase", "citibank", "bank of america",
    "usps", "fedex", "dhl", "postal service",
    "docusign", "dropbox", "sharepoint", "onedrive",
    "help desk", "service desk", "support team",
    "office 365", "webmail",
    "facebook", "instagram", "linkedin", "twitter",
    "whatsapp", "telegram", "coinbase", "binance",
    "geek squad", "norton", "mcafee",
    "account team", "billing department",
    # Additional authority impersonation
    "stripe", "shopify", "venmo", "cash app", "zelle",
    "capital one", "american express", "amex",
    "verification team", "trust and safety", "abuse team",
    "official notice", "official communication",
    "account recovery", "account protection",
    # Exact-match acronyms — only matched as whole words (see _score_patterns)
    "cfo", "cto", "coo", "managing director",
    "department of", "bureau of",
    "customs", "immigration", "tax authority",
    "your bank", "your provider", "your carrier",
    # Spanish authority impersonation
    "administrador", "equipo de seguridad", "soporte técnico", "soporte tecnico",
    "departamento de", "servicio al cliente", "servicio de atención",
    "recursos humanos", "departamento legal",
    "microsoft 365",
]

# Short acronyms that must be matched as whole words only (not substrings).
# e.g. "cto" should NOT fire on "director", "sector", "detector".
_WHOLE_WORD_PHRASES = {"ceo", "cfo", "cto", "coo", "irs", "ups", "dhl", "hr"}

PRESSURE_PHRASES = [
    "within 24 hours", "within 48 hours", "limited time",
    "act now", "don't delay", "expires today", "expires soon",
    "last chance", "final notice", "final warning",
    "deadline", "time sensitive", "time-sensitive",
    "offer expires", "respond immediately", "without delay",
    "account will be closed", "will be deactivated",
    "will be permanently", "no longer accessible",
    "must respond", "must verify", "must confirm",
    "hours remaining", "minutes remaining",
    "today only", "ending soon", "don't miss",
    "before it's too late", "risk losing", "avoid suspension",
    # Additional pressure tactics
    "only have", "running out of time", "clock is ticking",
    "won't be able to", "will lose access", "lose your",
    "permanently deleted", "permanently removed",
    "cannot be recovered", "cannot be restored",
    "no further notice", "without further notice",
    "grace period", "expiration date", "due date",
    "as soon as possible", "asap", "at your earliest",
    "we will have to", "forced to", "obligated to",
    "legal action", "legal consequences", "penalties",
    "your responsibility", "held responsible",
    "non-compliance", "failure to comply",
    "do not share this link", "this link is confidential",
    # Spanish pressure tactics
    "dentro de 24 horas", "dentro de 48 horas",
    "actúe ahora", "actue ahora", "responda inmediatamente",
    "última oportunidad", "ultima oportunidad",
    "aviso final", "advertencia final", "último aviso", "ultimo aviso",
    "plazo", "fecha límite", "fecha limite",
    "será eliminada", "sera eliminada", "permanentemente",
    "si no realiza", "si no completa", "de lo contrario",
    "perderá acceso", "perdera acceso", "perderá su cuenta", "perdera su cuenta",
]

GENERIC_GREETINGS = [
    "dear customer", "dear user", "dear valued member",
    "dear account holder", "dear client", "dear sir/madam",
    "dear valued customer", "dear member", "dear subscriber",
    "to whom it may concern", "attention",
    "dear sir", "dear madam", "dear friend",
    "dear email user", "dear webmail user",
    "dear beneficiary", "dear winner",
    "hello user", "hi customer", "greetings",
    # Spanish generic greetings
    "estimado usuario", "estimado cliente", "estimado miembro",
    "querido usuario", "querido cliente",
    "apreciado usuario", "apreciado cliente",
]

REWARD_LURE_PHRASES = [
    "congratulations", "you have won", "you've won", "prize",
    "lottery", "selected winner", "free gift", "claim your",
    "reward", "bonus", "inheritance", "million dollars",
    "unclaimed funds", "beneficiary",
    "gift card", "you've been selected", "exclusive offer",
    "you are eligible", "special promotion", "cash prize",
    "wire transfer", "money transfer", "compensation",
    "tax refund", "refund pending", "claim your refund",
]

CREDENTIAL_REQUEST_PHRASES = [
    "enter your password", "confirm your password",
    "update your credentials", "verify your login",
    "social security number", "credit card number",
    "bank account number", "routing number",
    "personal information", "sensitive information",
    "click the link below", "click here to verify",
    "log in to your account", "sign in to confirm",
    "update your payment", "update your billing",
    "confirm your identity", "verify your account",
    "verify your email", "verify your information",
    "click here to update", "click here to confirm",
    "click the button below", "click below",
    "open the attachment", "see attached",
    "download the file", "enable macros",
    "enter your details", "fill out the form",
    "reset your password", "change your password",
    "scan this qr code", "scan the code",
    # Additional credential harvesting patterns
    "provide your", "submit your", "input your",
    "we need your", "we require your",
    "for verification purposes", "for security purposes",
    "for your protection", "for your safety",
    "login credentials", "sign-in credentials",
    "access your account", "access the portal",
    "secure link", "secure portal", "secure form",
    "one-time password", "one-time code", "otp",
    "two-factor", "2fa", "authentication code",
    "temporary password", "temporary access",
    "re-enter your", "re-verify your", "reconfirm",
    "review attached", "view attachment", "open attached",
    "enable content", "enable editing",
    "follow this link", "use this link", "via the link",
    "copy and paste", "paste this url",
    "reply with your", "send us your", "forward this to",
    # Spanish credential requests
    "haga clic aquí", "haga clic aqui", "haga clic en el enlace",
    "haga clic para verificar", "clic aquí para", "clic aqui para",
    "pulse aquí", "pulse aqui", "presione aquí", "presione aqui",
    "ingrese su contraseña", "ingrese su contrasena",
    "confirme su contraseña", "confirme su contrasena",
    "proporcione su", "introduzca su", "envíe su", "envie su",
    "datos personales", "información personal", "informacion personal",
    "número de tarjeta", "numero de tarjeta",
    "número de cuenta", "numero de cuenta",
    "restablecer su contraseña", "restablecer su contrasena",
    "actualizar sus datos", "verificar su correo",
    "abra el archivo adjunto", "descargue el archivo",
]

# Transformer model (loaded lazily)
_transformer_model = None
_transformer_tokenizer = None
_model_load_attempted = False


def _load_transformer():
    """Lazy-load the DistilBERT model and tokenizer."""
    global _transformer_model, _transformer_tokenizer, _model_load_attempted

    if _model_load_attempted:
        return _transformer_model is not None

    _model_load_attempted = True
    try:
        from transformers import DistilBertModel, DistilBertTokenizer
        import torch

        logger.info("Loading DistilBERT model...")
        _transformer_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        _transformer_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        _transformer_model.eval()
        logger.info("DistilBERT model loaded successfully")
        return True
    except Exception as e:
        logger.warning(f"Could not load DistilBERT model: {e}. Using rule-based only.")
        return False


def get_transformer_embedding(text: str) -> Optional[np.ndarray]:
    """
    Extract 768-dimensional [CLS] embedding from DistilBERT.

    Args:
        text: Input text (subject + body concatenated)

    Returns:
        768-d numpy array or None if model unavailable.
    """
    if not _load_transformer():
        return None

    try:
        import torch

        inputs = _transformer_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            outputs = _transformer_model(**inputs)

        # [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return cls_embedding

    except Exception as e:
        logger.warning(f"Transformer embedding extraction failed: {e}")
        return None


def _score_patterns(text: str, patterns: List[str]) -> Dict[str, Any]:
    """
    Score text against a list of patterns, returning match count and matched phrases.

    Matching rules:
    - Short acronyms listed in _WHOLE_WORD_PHRASES (ceo, cto, cfo, coo, irs, …) are
      matched with \\b word boundaries so they do NOT fire inside longer words
      (e.g. "cto" must not match inside "director", "sector", "detector").
    - All other multi-word phrases use a word-boundary anchored regex too, preventing
      partial substring matches at word edges.
    """
    text_lower = text.lower()
    matches = []
    for phrase in patterns:
        phrase_lower = phrase.lower()
        if phrase_lower in _WHOLE_WORD_PHRASES or len(phrase_lower) <= 3:
            # Whole-word match only — prevents substring bleed
            pattern = r'\b' + re.escape(phrase_lower) + r'\b'
            if re.search(pattern, text_lower):
                matches.append(phrase)
        else:
            # Standard substring match for longer, unambiguous phrases
            if phrase_lower in text_lower:
                matches.append(phrase)
    count = len(matches)
    # Scoring: 1 match = 0.35, 2 = 0.60, 3 = 0.80, 4+ = 0.90+
    if count == 0:
        score = 0.0
    elif count == 1:
        score = 0.35
    elif count == 2:
        score = 0.60
    elif count == 3:
        score = 0.80
    else:
        score = min(0.90 + (count - 4) * 0.03, 1.0)
    return {"score": round(score, 4), "count": count, "matches": matches}


def compute_rule_based_scores(text: str) -> Dict[str, Any]:
    """
    Compute rule-based semantic scores for phishing indicators.

    Returns scores for urgency, authority, pressure, generic greetings,
    reward lures, and credential requests.
    """
    text_lower = text.lower()

    urgency = _score_patterns(text, URGENCY_PHRASES)
    authority = _score_patterns(text, AUTHORITY_PHRASES)
    pressure = _score_patterns(text, PRESSURE_PHRASES)
    greetings = _score_patterns(text, GENERIC_GREETINGS)
    rewards = _score_patterns(text, REWARD_LURE_PHRASES)
    credentials = _score_patterns(text, CREDENTIAL_REQUEST_PHRASES)

    # Brand mention detection (for body text analysis)
    brand_mentions_in_body = []
    brand_names = [
        "paypal", "microsoft", "apple", "google", "amazon", "netflix",
        "chase", "wells fargo", "citibank", "bank of america", "capital one",
        "facebook", "instagram", "linkedin", "coinbase", "binance",
        "docusign", "dropbox", "stripe", "shopify", "venmo", "cash app",
        "fedex", "ups", "usps", "dhl", "irs", "social security",
        "geek squad", "norton", "mcafee", "uber", "spotify",
    ]
    for brand in brand_names:
        if brand in text_lower:
            brand_mentions_in_body.append(brand)

    # Grammatical anomaly detection (simple heuristics)
    grammar_issues = []
    # Check for excessive capitalization
    words = text.split()
    if words:
        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / len(words)
        if caps_ratio > 0.3:
            grammar_issues.append("Excessive capitalization")

    # Check for multiple exclamation/question marks
    if text.count("!!!") > 0 or text.count("???") > 0:
        grammar_issues.append("Excessive punctuation")

    # Check for common phishing misspellings
    misspellings = ["recieve", "verifiy", "acount", "pasword", "informations",
                     "securty", "updation", "kindly revert", "do the needful"]
    found_misspellings = [m for m in misspellings if m in text.lower()]
    if found_misspellings:
        grammar_issues.append(f"Suspicious misspellings: {', '.join(found_misspellings)}")

    grammar_score = min(len(grammar_issues) / 3, 1.0)

    # Combined semantic score (weighted)
    # Use max-of-top-indicators approach to avoid dilution
    top_scores = sorted([
        urgency["score"], authority["score"], pressure["score"],
        credentials["score"], rewards["score"],
    ], reverse=True)

    # Weighted average emphasizing the strongest signals
    combined = (
        urgency["score"] * 0.20 +
        authority["score"] * 0.15 +
        pressure["score"] * 0.15 +
        greetings["score"] * 0.10 +
        rewards["score"] * 0.10 +
        credentials["score"] * 0.20 +
        grammar_score * 0.10
    )

    # Boost: if any single category scored high, raise the floor
    if top_scores[0] >= 0.35:
        combined = max(combined, top_scores[0] * 0.75)
    # Multi-category boost: if 2+ categories flagged, amplify
    flagged = sum(1 for s in top_scores if s >= 0.3)
    if flagged >= 2:
        combined = max(combined, (top_scores[0] + top_scores[1]) / 2 * 0.85)

    # Brand + urgency/credential co-occurrence boost
    if brand_mentions_in_body:
        if urgency["count"] > 0 or credentials["count"] > 0:
            combined = max(combined, 0.45)  # At minimum suspicious
        if urgency["count"] > 0 and credentials["count"] > 0:
            combined = max(combined, 0.60)  # Very likely phishing

    return {
        "urgency": urgency,
        "authority": authority,
        "pressure": pressure,
        "generic_greetings": greetings,
        "reward_lures": rewards,
        "credential_requests": credentials,
        "grammatical_anomalies": {
            "score": round(grammar_score, 4),
            "issues": grammar_issues,
        },
        "combined_score": round(min(combined, 1.0), 4),
        "brand_mentions": brand_mentions_in_body,
    }


def analyze_semantics(subject: str, body: str) -> Dict[str, Any]:
    """
    Full semantic analysis combining transformer embeddings and rule-based scoring.

    Args:
        subject: Email subject line
        body: Email body text

    Returns:
        Dictionary with semantic analysis results, embedding, and scores.
    """
    full_text = f"{subject} {body}".strip()

    # Rule-based scores
    rule_scores = compute_rule_based_scores(full_text)

    # Transformer embedding
    embedding = get_transformer_embedding(full_text)
    has_embedding = embedding is not None

    # Collect details for explainability
    details = []
    if rule_scores["urgency"]["matches"]:
        details.append(f"Urgency cues: {', '.join(rule_scores['urgency']['matches'][:5])}")
    if rule_scores["authority"]["matches"]:
        details.append(f"Authority references: {', '.join(rule_scores['authority']['matches'][:5])}")
    if rule_scores["pressure"]["matches"]:
        details.append(f"Pressure tactics: {', '.join(rule_scores['pressure']['matches'][:5])}")
    if rule_scores["generic_greetings"]["matches"]:
        details.append(f"Generic greetings: {', '.join(rule_scores['generic_greetings']['matches'][:3])}")
    if rule_scores["reward_lures"]["matches"]:
        details.append(f"Reward/lure language: {', '.join(rule_scores['reward_lures']['matches'][:3])}")
    if rule_scores["credential_requests"]["matches"]:
        details.append(f"Credential requests: {', '.join(rule_scores['credential_requests']['matches'][:3])}")
    if rule_scores["grammatical_anomalies"]["issues"]:
        details.append(f"Grammar issues: {', '.join(rule_scores['grammatical_anomalies']['issues'])}")

    if rule_scores["brand_mentions"]:
        details.append(f"Brand names mentioned: {', '.join(rule_scores['brand_mentions'][:5])}")
    if not details:
        details = ["No suspicious semantic patterns detected"]

    return {
        "embedding": embedding.tolist() if has_embedding else None,
        "has_embedding": has_embedding,
        "rule_scores": rule_scores,
        "overall_score": rule_scores["combined_score"],
        "details": details,
    }
