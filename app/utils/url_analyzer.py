"""
URL Analysis Module for Phishing Detection.

Extracts and analyzes URLs from email content to identify phishing indicators
such as IP-based URLs, suspicious TLDs, URL shorteners, typosquatting,
and other structural anomalies.

Reference: Loo, Galindo, Romero et al. (2025) — Agentic AI for Phishing Detection.
"""

import re
import logging
from urllib.parse import urlparse, unquote
from typing import List, Dict, Any

try:
    import tldextract
except ImportError:
    tldextract = None

try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Fallback Levenshtein distance implementation."""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

logger = logging.getLogger(__name__)

# URL shortener domains
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bl.ink", "lnkd.in", "db.tt", "qr.ae",
    "rebrand.ly", "short.io", "cutt.ly", "rb.gy", "v.gd", "clck.ru",
}

# Suspicious TLDs commonly used in phishing
SUSPICIOUS_TLDS = {
    "xyz", "top", "buzz", "club", "work", "click", "link", "info",
    "online", "site", "website", "space", "fun", "host", "icu",
    "cam", "monster", "rest", "beauty", "hair", "skin", "quest",
    "cfd", "sbs", "zip", "mov",
}

# Top brands for typosquatting detection
TOP_BRANDS = [
    "google", "microsoft", "apple", "amazon", "paypal", "netflix",
    "facebook", "instagram", "linkedin", "twitter", "chase", "wellsfargo",
    "bankofamerica", "citibank", "usbank", "capitalone", "americanexpress",
    "dropbox", "adobe", "salesforce", "zoom", "slack", "github",
    "yahoo", "outlook", "office365", "icloud", "whatsapp", "telegram",
    "coinbase", "binance", "stripe", "shopify", "ebay", "walmart",
    "target", "bestbuy", "homedepot", "costco", "fedex", "ups",
    "usps", "dhl", "irs", "socialsecurity", "medicare",
]

# Free hosting / website builder platforms (suspicious for verification links)
FREE_HOSTING_DOMAINS = {
    "weebly.com", "wix.com", "wixsite.com", "wordpress.com", "blogspot.com",
    "blogger.com", "sites.google.com", "github.io", "netlify.app",
    "vercel.app", "herokuapp.com", "firebaseapp.com", "web.app",
    "000webhostapp.com", "infinityfreeapp.com", "rf.gd",
    "godaddysites.com", "square.site", "carrd.co", "glitch.me",
    "repl.co", "surge.sh", "pages.dev", "fly.dev",
    "yolasite.com", "jimdo.com", "strikingly.com",
    "webflow.io", "framer.app", "notion.site",
    "docs.google.com", "forms.gle", "forms.google.com",
}

# Suspicious path keywords
SUSPICIOUS_PATH_KEYWORDS = {
    "login", "verify", "secure", "update", "confirm", "account",
    "password", "signin", "sign-in", "auth", "authenticate",
    "validate", "reset", "recover", "suspend", "unlock", "billing",
    "payment", "invoice", "refund", "reward", "prize", "winner",
    # Spanish path keywords
    "verificar", "confirmar", "iniciar", "sesion", "cuenta",
    "actualizar", "restablecer", "seguridad",
}

# URL extraction regex
URL_REGEX = re.compile(
    r'https?://[^\s<>"\')\]]+|'
    r'www\.[^\s<>"\')\]]+',
    re.IGNORECASE
)

# Also capture href attributes from HTML
HREF_REGEX = re.compile(
    r'href\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE
)


def extract_urls(text: str, html_content: str = "") -> List[str]:
    """Extract all URLs from plain text and HTML content."""
    urls = set()

    # From plain text
    for match in URL_REGEX.finditer(text):
        url = match.group(0).rstrip(".,;:!?)")
        if not url.startswith("http"):
            url = "http://" + url
        urls.add(url)

    # From HTML href attributes
    if html_content:
        for match in HREF_REGEX.finditer(html_content):
            href = match.group(1)
            if href.startswith(("http://", "https://", "www.")):
                if not href.startswith("http"):
                    href = "http://" + href
                urls.add(href)

    return list(urls)


def _extract_domain_parts(url: str) -> Dict[str, str]:
    """Extract domain, subdomain, suffix from a URL."""
    if tldextract:
        ext = tldextract.extract(url)
        return {
            "subdomain": ext.subdomain,
            "domain": ext.domain,
            "suffix": ext.suffix,
            "registered_domain": ext.registered_domain,
        }
    # Fallback
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    parts = hostname.split(".")
    if len(parts) >= 3:
        return {
            "subdomain": ".".join(parts[:-2]),
            "domain": parts[-2],
            "suffix": parts[-1],
            "registered_domain": ".".join(parts[-2:]),
        }
    elif len(parts) == 2:
        return {
            "subdomain": "",
            "domain": parts[0],
            "suffix": parts[1],
            "registered_domain": hostname,
        }
    return {"subdomain": "", "domain": hostname, "suffix": "", "registered_domain": hostname}


def analyze_single_url(url: str) -> Dict[str, Any]:
    """Analyze a single URL for phishing indicators."""
    features = {
        "url": url,
        "is_ip_based": False,
        "uses_shortener": False,
        "suspicious_tld": False,
        "long_domain": False,
        "has_at_symbol": False,
        "excessive_subdomains": False,
        "excessive_hyphens": False,
        "uses_https": False,
        "typosquatting_score": 0.0,
        "typosquatting_target": None,
        "suspicious_path": False,
        "suspicious_path_keywords": [],
        "uses_free_hosting": False,
        "free_hosting_platform": None,
        "risk_score": 0.0,
    }

    try:
        decoded_url = unquote(url)
        parsed = urlparse(decoded_url)
        hostname = parsed.hostname or ""
        path = (parsed.path or "").lower()

        # IP-based URL check
        ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
        features["is_ip_based"] = bool(ip_pattern.match(hostname))

        # URL shortener check
        domain_parts = _extract_domain_parts(decoded_url)
        registered = domain_parts["registered_domain"].lower()
        features["uses_shortener"] = registered in URL_SHORTENERS

        # Free hosting platform check
        for fh_domain in FREE_HOSTING_DOMAINS:
            if hostname.endswith(fh_domain) or registered == fh_domain:
                features["uses_free_hosting"] = True
                features["free_hosting_platform"] = fh_domain
                break

        # Suspicious TLD
        suffix = domain_parts["suffix"].lower()
        features["suspicious_tld"] = suffix in SUSPICIOUS_TLDS

        # Long domain
        features["long_domain"] = len(hostname) > 30

        # @ symbol in URL
        features["has_at_symbol"] = "@" in decoded_url.split("//", 1)[-1].split("/", 1)[0]

        # Excessive subdomains
        subdomain = domain_parts["subdomain"]
        subdomain_count = len(subdomain.split(".")) if subdomain else 0
        features["excessive_subdomains"] = subdomain_count > 3

        # Excessive hyphens
        domain = domain_parts["domain"]
        features["excessive_hyphens"] = domain.count("-") > 2

        # HTTPS check
        features["uses_https"] = parsed.scheme == "https"

        # Typosquatting detection
        min_dist = float("inf")
        closest_brand = None
        for brand in TOP_BRANDS:
            dist = levenshtein_distance(domain.lower(), brand)
            if dist < min_dist:
                min_dist = dist
                closest_brand = brand

        if 0 < min_dist <= 2 and closest_brand:
            features["typosquatting_score"] = 1.0 - (min_dist / max(len(domain), len(closest_brand)))
            features["typosquatting_target"] = closest_brand
        elif min_dist == 0:
            features["typosquatting_score"] = 0.0
            features["typosquatting_target"] = None

        # Suspicious path keywords
        found_keywords = [kw for kw in SUSPICIOUS_PATH_KEYWORDS if kw in path]
        features["suspicious_path"] = len(found_keywords) > 0
        features["suspicious_path_keywords"] = found_keywords

        # Compute risk score
        risk = 0.0
        if features["is_ip_based"]:
            risk += 0.25
        if features["uses_shortener"]:
            risk += 0.15
        if features["suspicious_tld"]:
            risk += 0.15
        if features["long_domain"]:
            risk += 0.05
        if features["has_at_symbol"]:
            risk += 0.20
        if features["excessive_subdomains"]:
            risk += 0.10
        if features["excessive_hyphens"]:
            risk += 0.05
        if not features["uses_https"]:
            risk += 0.05
        if features["typosquatting_score"] > 0.5:
            risk += 0.25
        if features["suspicious_path"]:
            risk += 0.10
        if features["uses_free_hosting"]:
            risk += 0.30  # Strong signal: verification links on free hosting

        features["risk_score"] = min(risk, 1.0)

    except Exception as e:
        logger.warning(f"Error analyzing URL {url}: {e}")
        features["risk_score"] = 0.5

    return features


def analyze_urls(text: str, html_content: str = "") -> Dict[str, Any]:
    """
    Extract and analyze all URLs from email content.

    Returns aggregate URL analysis including individual URL reports
    and an overall URL threat score.
    """
    urls = extract_urls(text, html_content)
    url_analyses = [analyze_single_url(url) for url in urls]

    if not url_analyses:
        return {
            "url_count": 0,
            "overall_score": 0.0,
            "urls": [],
            "details": ["No URLs found in email content"],
        }

    max_risk = max(a["risk_score"] for a in url_analyses)
    avg_risk = sum(a["risk_score"] for a in url_analyses) / len(url_analyses)
    overall = 0.7 * max_risk + 0.3 * avg_risk

    details = []
    for a in url_analyses:
        flags = []
        if a["is_ip_based"]:
            flags.append("IP-based URL")
        if a["uses_shortener"]:
            flags.append("URL shortener")
        if a["suspicious_tld"]:
            flags.append("Suspicious TLD")
        if a["typosquatting_score"] > 0.5:
            flags.append(f"Possible typosquatting of '{a['typosquatting_target']}'")
        if a["has_at_symbol"]:
            flags.append("Contains @ symbol")
        if a["suspicious_path"]:
            flags.append(f"Suspicious path keywords: {', '.join(a['suspicious_path_keywords'])}")
        if a.get("uses_free_hosting"):
            flags.append(f"Free hosting platform ({a['free_hosting_platform']})")
        if not a["uses_https"]:
            flags.append("No HTTPS")
        if flags:
            details.append(f"{a['url']}: {'; '.join(flags)}")

    return {
        "url_count": len(urls),
        "overall_score": round(overall, 4),
        "urls": url_analyses,
        "details": details if details else ["No suspicious URL indicators found"],
    }
