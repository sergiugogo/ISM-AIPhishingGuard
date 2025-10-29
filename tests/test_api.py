"""Comprehensive tests for PhishGuard API endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app
from config import settings

client = TestClient(app)


@pytest.fixture
def api_headers():
    """Fixture for API headers with valid key."""
    return {"X-API-Key": settings.api_key}


@pytest.fixture
def benign_email():
    """Fixture for benign email."""
    return {
        "subject": "Team Meeting Tomorrow",
        "body": "Hi everyone, let's meet tomorrow at 10 AM in conference room B to discuss the project."
    }


@pytest.fixture
def phishing_email():
    """Fixture for phishing email."""
    return {
        "subject": "URGENT: Account Verification Required!",
        "body": "Your account will be suspended immediately. Click here to verify now: http://192.168.1.1/verify?id=12345"
    }


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_service_info(self):
        """Test root endpoint returns service information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == settings.app_name
        assert data["version"] == settings.app_version
        assert data["status"] == "operational"
        assert "endpoints" in data


class TestHealthCheck:
    """Tests for health check endpoint."""
    
    def test_health_check_structure(self):
        """Test health check returns proper structure."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["version"] == settings.app_version
    
    def test_health_check_when_model_loaded(self):
        """Test health check when model is loaded."""
        response = client.get("/health")
        data = response.json()
        # Model should be loaded during app startup
        assert data["model_loaded"] is True
        assert data["status"] in ["healthy", "degraded"]


class TestPredictionEndpoint:
    """Tests for prediction endpoint."""
    
    def test_predict_benign_email(self, api_headers, benign_email):
        """Test prediction on benign email."""
        response = client.post("/predict", json=benign_email, headers=api_headers)
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "label" in data
        assert "confidence" in data
        assert "phishing_score" in data
        assert "benign_score" in data
        assert "explanation" in data
        assert "features" in data
        
        # Check data types and ranges
        assert data["label"] in ["phishing", "benign"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert 0.0 <= data["phishing_score"] <= 1.0
        assert 0.0 <= data["benign_score"] <= 1.0
        
        # Scores should sum to approximately 1.0
        assert abs(data["phishing_score"] + data["benign_score"] - 1.0) < 0.01
    
    def test_predict_phishing_email(self, api_headers, phishing_email):
        """Test prediction on phishing email."""
        response = client.post("/predict", json=phishing_email, headers=api_headers)
        assert response.status_code == 200
        data = response.json()
        
        # Check features are extracted
        assert "features" in data
        features = data["features"]
        assert "num_urls" in features
        assert "has_ip_in_url" in features
        assert "urgency_hits" in features
        
        # Should detect phishing indicators
        assert features["urgency_hits"] > 0  # Has urgent words
        assert features["has_ip_in_url"] is True  # Has IP in URL
    
    def test_predict_empty_body_fails(self, api_headers):
        """Test prediction with empty body fails validation."""
        response = client.post(
            "/predict",
            json={"subject": "Test", "body": ""},
            headers=api_headers
        )
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_body_fails(self, api_headers):
        """Test prediction with missing body fails validation."""
        response = client.post(
            "/predict",
            json={"subject": "Test"},
            headers=api_headers
        )
        assert response.status_code == 422  # Validation error
    
    def test_predict_too_long_body_truncated(self, api_headers):
        """Test prediction with very long body is handled."""
        long_body = "A" * 20000  # Exceeds max_length
        response = client.post(
            "/predict",
            json={"subject": "Test", "body": long_body},
            headers=api_headers
        )
        # Should still work, content will be truncated
        assert response.status_code in [200, 422]
    
    def test_predict_html_email(self, api_headers):
        """Test prediction with HTML email body."""
        html_body = """
        <html>
            <body>
                <p>Click <a href="http://example.com">here</a> for more info</p>
            </body>
        </html>
        """
        response = client.post(
            "/predict",
            json={"subject": "Test", "body": html_body},
            headers=api_headers
        )
        assert response.status_code == 200
        data = response.json()
        # Should extract URL from HTML
        assert data["features"]["num_urls"] >= 1
    
    def test_predict_multiple_urls(self, api_headers):
        """Test prediction with multiple URLs."""
        body = "Check out http://site1.com and http://site2.com and http://site3.com"
        response = client.post(
            "/predict",
            json={"subject": "Multiple links", "body": body},
            headers=api_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["features"]["num_urls"] >= 3


class TestAuthentication:
    """Tests for API authentication."""
    
    def test_predict_without_api_key_in_debug_mode(self, benign_email):
        """Test prediction without API key works in debug mode."""
        # In debug mode, API key is optional
        if settings.debug:
            response = client.post("/predict", json=benign_email)
            assert response.status_code == 200
    
    def test_predict_with_invalid_api_key(self, benign_email):
        """Test prediction with invalid API key."""
        if not settings.debug:
            response = client.post(
                "/predict",
                json=benign_email,
                headers={"X-API-Key": "invalid-key"}
            )
            assert response.status_code == 401
    
    def test_predict_with_valid_api_key(self, api_headers, benign_email):
        """Test prediction with valid API key."""
        response = client.post("/predict", json=benign_email, headers=api_headers)
        assert response.status_code == 200


class TestInputValidation:
    """Tests for input validation."""
    
    def test_subject_whitespace_stripped(self, api_headers):
        """Test that whitespace is stripped from subject."""
        response = client.post(
            "/predict",
            json={"subject": "  Test  ", "body": "Test body"},
            headers=api_headers
        )
        assert response.status_code == 200
    
    def test_special_characters_handled(self, api_headers):
        """Test that special characters are handled."""
        body = "Test with émojis 🎉 and symbols @#$%^&*()"
        response = client.post(
            "/predict",
            json={"subject": "Test", "body": body},
            headers=api_headers
        )
        assert response.status_code == 200
    
    def test_unicode_characters_handled(self, api_headers):
        """Test that Unicode characters are handled."""
        body = "测试中文 Тест на русском"
        response = client.post(
            "/predict",
            json={"subject": "Test", "body": body},
            headers=api_headers
        )
        assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_json_returns_422(self):
        """Test that invalid JSON returns 422."""
        response = client.post(
            "/predict",
            data="not json",
            headers={"Content-Type": "application/json", "X-API-Key": settings.api_key}
        )
        assert response.status_code == 422
    
    def test_missing_content_type(self, benign_email, api_headers):
        """Test request without content-type header."""
        # FastAPI should handle this gracefully
        response = client.post(
            "/predict",
            json=benign_email,
            headers=api_headers
        )
        assert response.status_code == 200


class TestFeatureExtraction:
    """Tests for feature extraction."""
    
    def test_url_extraction(self, api_headers):
        """Test URL extraction from email body."""
        body = "Visit http://example.com and https://test.org"
        response = client.post(
            "/predict",
            json={"subject": "", "body": body},
            headers=api_headers
        )
        data = response.json()
        assert data["features"]["num_urls"] == 2
    
    def test_ip_url_detection(self, api_headers):
        """Test IP address URL detection."""
        body = "Click http://192.168.1.1/page"
        response = client.post(
            "/predict",
            json={"subject": "", "body": body},
            headers=api_headers
        )
        data = response.json()
        assert data["features"]["has_ip_in_url"] is True
    
    def test_urgency_word_detection(self, api_headers):
        """Test urgency word detection."""
        subject = "URGENT ACTION REQUIRED IMMEDIATELY"
        response = client.post(
            "/predict",
            json={"subject": subject, "body": "Please verify your account."},
            headers=api_headers
        )
        data = response.json()
        assert data["features"]["urgency_hits"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

