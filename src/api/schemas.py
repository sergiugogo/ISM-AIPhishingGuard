"""API schemas for PhishGuard."""
from pydantic import BaseModel, Field, validator
from typing import List, Optional


class EmailRequest(BaseModel):
    """Email prediction request."""
    
    subject: str = Field(
        default="",
        description="Email subject line",
        max_length=500
    )
    body: str = Field(
        ...,
        description="Email body content (can be HTML or plain text)",
        min_length=1,
        max_length=10000
    )
    
    @validator("subject", "body")
    def strip_whitespace(cls, v):
        """Strip whitespace from fields."""
        return v.strip() if v else v
    
    class Config:
        schema_extra = {
            "example": {
                "subject": "Urgent: Verify your account",
                "body": "Dear user, your account will be suspended. Click here to verify: http://192.168.1.1/verify"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response."""
    
    label: str = Field(
        ...,
        description="Prediction label: 'phishing' or 'benign'"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the prediction"
    )
    phishing_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of being phishing"
    )
    benign_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of being benign"
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of the prediction"
    )
    features: dict = Field(
        ...,
        description="Extracted email features"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "label": "phishing",
                "confidence": 0.95,
                "phishing_score": 0.95,
                "benign_score": 0.05,
                "explanation": "Classified as PHISHING (95.0% confidence). Red flags: URL contains IP address instead of domain name; Uses urgent language (3 urgency indicators)",
                "features": {
                    "num_urls": 1,
                    "has_ip_in_url": True,
                    "urgency_hits": 3
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    model_config = {"protected_namespaces": ()}
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
    model_dir: Optional[str] = Field(None, description="Model directory path")
    device: Optional[str] = Field(None, description="Device (CPU/CUDA)")


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional details")
