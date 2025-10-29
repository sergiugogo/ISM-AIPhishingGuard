"""
PhishGuard API - Production-ready phishing detection service.

This module provides a FastAPI-based REST API for detecting phishing emails
using a fine-tuned RoBERTa model.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from collections import defaultdict
from datetime import datetime, timedelta

from config import settings
from src.core.logger import setup_logger
from src.core.model_manager import model_manager
from src.core.exceptions import ModelLoadError, ValidationError, RateLimitExceeded
from src.utils.preprocess import preprocess, generate_explanation
from src.api.schemas import (
    EmailRequest,
    PredictionResponse,
    HealthResponse,
    ErrorResponse
)

# Setup logger
logger = setup_logger(__name__, level=settings.log_level, use_json=(settings.log_format == "json"))

# Rate limiting storage (in production, use Redis)
request_counts = defaultdict(list)

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting PhishGuard API...")
    try:
        model_manager.load_model(
            model_dir=settings.model_dir,
            max_length=settings.max_length
        )
        logger.info("PhishGuard API started successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise
    finally:
        # Shutdown
        logger.info("Shutting down PhishGuard API...")


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered phishing email detection service",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key."""
    if settings.debug:  # Skip in debug mode
        return "debug"
    
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    return api_key


def check_rate_limit(api_key: str) -> None:
    """Check rate limiting."""
    now = datetime.now()
    minute_ago = now - timedelta(minutes=1)
    
    # Clean old requests
    request_counts[api_key] = [
        req_time for req_time in request_counts[api_key]
        if req_time > minute_ago
    ]
    
    # Check limit
    if len(request_counts[api_key]) >= settings.rate_limit_per_minute:
        raise RateLimitExceeded(
            f"Rate limit exceeded. Maximum {settings.rate_limit_per_minute} requests per minute."
        )
    
    # Add current request
    request_counts[api_key].append(now)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.debug else None
        }
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    """Handle rate limit exceptions."""
    return JSONResponse(
        status_code=429,
        content={
            "error": "RateLimitExceeded",
            "message": str(exc)
        }
    )


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs" if settings.debug else "disabled"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = model_manager.is_ready()
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=settings.app_version,
        model_dir=settings.model_dir if model_loaded else None,
        device=str(model_manager.device) if model_loaded else None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_email(
    request: EmailRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Predict whether an email is phishing or benign.
    
    Args:
        request: Email content (subject and body)
        api_key: API key for authentication
        
    Returns:
        Prediction results with confidence scores and explanation
        
    Raises:
        HTTPException: If validation or prediction fails
    """
    start_time = time.time()
    
    try:
        # Rate limiting
        check_rate_limit(api_key)
        
        # Validate model is ready
        if not model_manager.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service unavailable."
            )
        
        # Preprocess email
        logger.info("Processing email prediction request")
        features = preprocess(
            request.subject,
            request.body,
            max_length=settings.max_content_length
        )
        
        # Make prediction
        prediction = model_manager.predict(features["combined_text"])
        
        # Generate explanation
        explanation = generate_explanation(
            features,
            prediction["label"],
            prediction["confidence"]
        )
        
        # Prepare response
        response = PredictionResponse(
            label=prediction["label"],
            confidence=prediction["confidence"],
            phishing_score=prediction["phishing_score"],
            benign_score=prediction["benign_score"],
            explanation=explanation,
            features={
                "num_urls": features["num_urls"],
                "has_ip_in_url": features["has_ip_in_url"],
                "has_suspicious_tld": features.get("has_suspicious_tld", False),
                "urgency_hits": features["urgency_hits"],
                "urls": features["urls"][:5]  # Limit to first 5 URLs
            }
        )
        
        # Log metrics
        elapsed = time.time() - start_time
        logger.info(
            f"Prediction completed",
            extra={
                "extra_data": {
                    "label": prediction["label"],
                    "confidence": prediction["confidence"],
                    "elapsed_ms": round(elapsed * 1000, 2)
                }
            }
        )
        
        return response
        
    except RateLimitExceeded:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
