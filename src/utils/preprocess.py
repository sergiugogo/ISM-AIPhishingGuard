"""Email preprocessing utilities for PhishGuard."""
import re
from typing import List, Dict
from bs4 import BeautifulSoup

URGENCY_WORDS = {
    "urgent", "immediate", "immediately", "verify", "reset", "limited", 
    "expire", "expires", "expiring", "action required", "confirm", 
    "suspended", "locked", "unusual activity", "security alert",
    "update required", "click here", "act now"
}

SUSPICIOUS_DOMAINS = {
    "tk", "ml", "ga", "cf", "gq"  # Common free TLDs used in phishing
}


def html_to_text(html_content: str) -> str:
    """
    Convert HTML to plain text.
    
    Args:
        html_content: HTML string
        
    Returns:
        Plain text string
    """
    if not html_content:
        return ""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return html_content


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Input text
        
    Returns:
        List of URLs
    """
    if not text:
        return []
    # More robust URL pattern
    pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    return re.findall(pattern, text, flags=re.IGNORECASE)


def has_ip_url(url: str) -> bool:
    """
    Check if URL contains IP address.
    
    Args:
        url: URL string
        
    Returns:
        True if URL contains IP address
    """
    return bool(re.search(r'https?://\d{1,3}(?:\.\d{1,3}){3}', url))


def has_suspicious_tld(url: str) -> bool:
    """
    Check if URL has suspicious TLD.
    
    Args:
        url: URL string
        
    Returns:
        True if URL has suspicious TLD
    """
    return any(url.endswith(f".{tld}") for tld in SUSPICIOUS_DOMAINS)


def count_urgency(text: str) -> int:
    """
    Count urgency words in text.
    
    Args:
        text: Input text
        
    Returns:
        Count of urgency words
    """
    if not text:
        return 0
    text_lower = text.lower()
    return sum(1 for word in URGENCY_WORDS if word in text_lower)


def preprocess(subject: str, body: str, max_length: int = 10000) -> Dict:
    """
    Preprocess email for analysis.
    
    Args:
        subject: Email subject
        body: Email body (can be HTML or plain text)
        max_length: Maximum content length
        
    Returns:
        Dictionary with preprocessed features
    """
    # Sanitize inputs
    subject = (subject or "")[:max_length]
    body = (body or "")[:max_length]
    
    # Convert HTML to text
    body_text = html_to_text(body)
    
    # Extract features
    urls = extract_urls(body_text + " " + subject)
    has_ip = any(has_ip_url(url) for url in urls)
    has_suspicious = any(has_suspicious_tld(url) for url in urls)
    urgency_count = count_urgency(subject + " " + body_text)
    
    return {
        "subject": subject,
        "body": body_text,
        "urls": urls,
        "num_urls": len(urls),
        "has_ip_in_url": has_ip,
        "has_suspicious_tld": has_suspicious,
        "urgency_hits": urgency_count,
        "combined_text": f"{subject} . {body_text}"
    }


def generate_explanation(features: Dict, prediction: str, confidence: float) -> str:
    """
    Generate human-readable explanation for prediction.
    
    Args:
        features: Preprocessed features
        prediction: Model prediction
        confidence: Prediction confidence
        
    Returns:
        Explanation string
    """
    reasons = []
    
    # Check phishing indicators
    if features.get("has_ip_in_url"):
        reasons.append("URL contains IP address instead of domain name")
    
    if features.get("has_suspicious_tld"):
        reasons.append("URL uses suspicious top-level domain")
    
    if features.get("num_urls", 0) >= 3:
        reasons.append(f"Contains {features['num_urls']} URLs (unusually high)")
    
    if features.get("urgency_hits", 0) >= 2:
        reasons.append(f"Uses urgent language ({features['urgency_hits']} urgency indicators)")
    
    # Build explanation
    if prediction == "phishing":
        if reasons:
            return f"Classified as PHISHING ({confidence:.1%} confidence). Red flags: " + "; ".join(reasons)
        else:
            return f"Classified as PHISHING ({confidence:.1%} confidence) based on content patterns."
    else:
        if reasons:
            return f"Classified as BENIGN ({confidence:.1%} confidence), but note: " + "; ".join(reasons)
        else:
            return f"Classified as BENIGN ({confidence:.1%} confidence). No suspicious indicators found."
