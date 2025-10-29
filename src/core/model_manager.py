"""Model management for PhishGuard."""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from typing import Optional
import threading

from src.core.logger import setup_logger
from src.core.exceptions import ModelLoadError

logger = setup_logger(__name__)


class ModelManager:
    """Manages model loading and inference."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for model manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize model manager."""
        if not hasattr(self, "initialized"):
            self.tokenizer: Optional[AutoTokenizer] = None
            self.model: Optional[AutoModelForSequenceClassification] = None
            self.device: Optional[torch.device] = None
            self.model_dir: Optional[str] = None
            self.initialized = False
    
    def load_model(self, model_dir: str, max_length: int = 256):
        """
        Load model and tokenizer.
        
        Args:
            model_dir: Path to model directory
            max_length: Maximum sequence length
            
        Raises:
            ModelLoadError: If model fails to load
        """
        try:
            model_path = Path(model_dir)
            if not model_path.exists():
                raise ModelLoadError(f"Model directory not found: {model_dir}")
            
            logger.info(f"Loading model from {model_dir}")
            
            # Determine device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            logger.info("Tokenizer loaded successfully")
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            
            self.model_dir = model_dir
            self.max_length = max_length
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to load model: {str(e)}") from e
    
    def predict(self, text: str) -> dict:
        """
        Make prediction on text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction results
            
        Raises:
            ModelLoadError: If model is not loaded
        """
        if not self.initialized:
            raise ModelLoadError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).squeeze()
            
            # Get prediction
            label_idx = torch.argmax(logits, dim=1).item()
            label = "phishing" if label_idx == 1 else "benign"
            confidence = probs[label_idx].item()
            
            return {
                "label": label,
                "confidence": float(confidence),
                "phishing_score": float(probs[1]),
                "benign_score": float(probs[0])
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise
    
    def is_ready(self) -> bool:
        """Check if model is ready."""
        return self.initialized and self.model is not None


# Global instance
model_manager = ModelManager()
