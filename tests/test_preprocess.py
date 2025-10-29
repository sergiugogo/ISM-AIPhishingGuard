"""Tests for preprocessing utilities."""
import pytest
from src.utils.preprocess import (
    html_to_text,
    extract_urls,
    has_ip_url,
    has_suspicious_tld,
    count_urgency,
    preprocess,
    generate_explanation
)


def test_html_to_text():
    """Test HTML to text conversion."""
    html = "<p>Hello <b>World</b></p>"
    result = html_to_text(html)
    assert "Hello" in result
    assert "World" in result
    assert "<p>" not in result


def test_extract_urls():
    """Test URL extraction."""
    text = "Visit https://example.com and http://test.org"
    urls = extract_urls(text)
    assert len(urls) == 2
    assert "https://example.com" in urls
    assert "http://test.org" in urls


def test_has_ip_url():
    """Test IP URL detection."""
    assert has_ip_url("http://192.168.1.1/path") == True
    assert has_ip_url("https://example.com") == False


def test_has_suspicious_tld():
    """Test suspicious TLD detection."""
    assert has_suspicious_tld("http://example.tk") == True
    assert has_suspicious_tld("http://example.ml") == True
    assert has_suspicious_tld("http://example.com") == False


def test_count_urgency():
    """Test urgency word counting."""
    text = "URGENT: Please verify your account immediately"
    count = count_urgency(text)
    assert count >= 2  # "urgent" and "verify" and "immediately"


def test_preprocess():
    """Test email preprocessing."""
    subject = "Important Message"
    body = "Click here: http://192.168.1.1/verify"
    
    features = preprocess(subject, body)
    
    assert features["subject"] == subject
    assert "192.168.1.1" in features["body"]
    assert features["num_urls"] == 1
    assert features["has_ip_in_url"] == True
    assert "combined_text" in features


def test_generate_explanation_phishing():
    """Test explanation generation for phishing."""
    features = {
        "has_ip_in_url": True,
        "num_urls": 3,
        "urgency_hits": 2
    }
    
    explanation = generate_explanation(features, "phishing", 0.95)
    
    assert "PHISHING" in explanation
    assert "95" in explanation
    assert "IP address" in explanation


def test_generate_explanation_benign():
    """Test explanation generation for benign."""
    features = {
        "has_ip_in_url": False,
        "num_urls": 0,
        "urgency_hits": 0
    }
    
    explanation = generate_explanation(features, "benign", 0.92)
    
    assert "BENIGN" in explanation
    assert "92" in explanation
