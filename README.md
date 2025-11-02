# PhishGuard - AI-Powered Phishing Detection API

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

PhishGuard is a production-ready API service for detecting phishing emails using a fine-tuned RoBERTa transformer model. Built as a learning project to understand the complete ML pipeline from data collection to deployment, it achieves 99.7% accuracy on test samples and provides real-time email analysis with detailed explanations.

## 🎯 Project Goals

This project was created as a comprehensive learning experience to understand:
- End-to-end fine-tuning of transformer models
- Data collection and preparation for ML
- Building production-ready ML APIs
- Model evaluation and deployment

It demonstrates that you don't need massive models to build effective, focused ML applications.

## ✨ Features

- 🤖 **Fine-tuned RoBERTa-base**: 125M parameter transformer model
- 🎯 **High Accuracy**: 99.7% on 1,000 test samples (Precision: 99.8%, Recall: 99.6%)
- ⚡ **Fast Inference**: ~50ms on GPU, ~200ms on CPU
- 🔍 **Explainable AI**: Detailed reasoning for each prediction
- 📊 **Feature Extraction**: IP detection, urgency analysis, URL scanning
- 🐳 **Docker Ready**: Easy deployment with Docker Compose
- 📝 **Well Documented**: OpenAPI/Swagger documentation
- ✅ **Tested**: Comprehensive test suite

## Quick Start

### Prerequisites

- Python 3.11+
- 4GB+ RAM
- (Optional) NVIDIA GPU with CUDA support for faster inference
- (Optional) Docker and Docker Compose

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/sergiugogo/ISM-AIPhishingGuard.git
cd ISM-AIPhishingGuard
```

2. **Create virtual environment**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env and set your API_KEY
# Recommended: Generate a strong key
# python -c "import secrets; print(secrets.token_urlsafe(32))"
```

5. **Prepare training data**
```bash
# This downloads and combines datasets from HuggingFace and PhishTank
python scripts/prepare_training_data.py
```

6. **Train the model** (takes ~45-60 minutes on GPU)
```bash
python src/train.py
```

7. **Evaluate the model**
```bash
python scripts/evaluate_model.py
```

8. **Run the API**
```bash
# Development mode
python scripts/start_api.py --reload

# Production mode
python scripts/start_api.py --workers 4
```

Visit `http://localhost:8000/docs` for interactive API documentation (when DEBUG=true).

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

## API Usage

### Authentication

All requests require an API key in the header:
```
X-API-Key: your-secret-api-key-here
```

### Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Predict Email
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key-here" \
  -d '{
    "subject": "Urgent: Verify your account",
    "body": "Click here to verify: http://192.168.1.1/verify"
  }'
```

**Response:**
```json
{
  "label": "phishing",
  "confidence": 0.95,
  "phishing_score": 0.95,
  "benign_score": 0.05,
  "explanation": "Classified as PHISHING (95.0% confidence). Red flags: URL contains IP address instead of domain name; Uses urgent language (2 urgency indicators)",
  "features": {
    "num_urls": 1,
    "has_ip_in_url": true,
    "has_suspicious_tld": false,
    "urgency_hits": 2,
    "urls": ["http://192.168.1.1/verify"]
  }
}
```

## Configuration

Edit `.env` or set environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_DIR` | Path to trained model | `models/phishguard-model` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MAX_CONTENT_LENGTH` | Max email content length | `10000` |
| `RATE_LIMIT_PER_MINUTE` | API rate limit | `60` |
| `API_KEY` | API authentication key | `your-secret-api-key-here` |
| `DEBUG` | Enable debug mode | `false` |

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Structure
```
phishguard/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI application
│   │   └── schemas.py       # Pydantic models
│   ├── core/
│   │   ├── logger.py        # Logging configuration
│   │   ├── exceptions.py    # Custom exceptions
│   │   └── model_manager.py # Model loading/inference
│   ├── utils/
│   │   └── preprocess.py    # Email preprocessing
│   └── train.py             # Model training script
├── scripts/
│   ├── prepare_training_data.py  # Multi-source data collection
│   ├── evaluate_model.py         # Model evaluation
│   └── start_api.py              # Production startup script
├── models/
│   └── phishguard-model/    # Trained model files
├── data/                     # Training/validation data (gitignored)
├── tests/                    # Test suite
├── config.py                # Configuration management
├── Dockerfile               # Docker image
├── docker-compose.yml       # Docker Compose
├── .env.example             # Environment template
└── requirements.txt         # Python dependencies
```

## Model Training

PhishGuard uses a multi-source dataset approach to ensure quality and balance:

### Data Sources
1. **HuggingFace Datasets**:
   - zefang-liu/phishing-email-dataset (18,650 emails)
   - SetFit/enron_spam (31,716 emails)
2. **PhishTank**: Verified phishing URLs converted to email samples (5,000 emails)

### Training Process

```bash
# 1. Prepare data (downloads and combines all sources)
python scripts/prepare_training_data.py

# 2. Train the model
python src/train.py

# 3. Evaluate performance
python scripts/evaluate_model.py
```

**Training Configuration:**
- Base model: RoBERTa-base (125M parameters)
- Training samples: 41,094 emails (80%)
- Validation samples: 10,274 emails (20%)
- Epochs: 5
- Batch size: 8
- Learning rate: 2e-5
- Optimization: FP16 on GPU
- Training time: ~45-60 minutes (RTX 4060 GPU)

**Model Performance:**
- Test Accuracy: 99.7% (on 1,000 validation samples)
- Precision: 99.8%
- Recall: 99.6%
- F1 Score: 99.7%
- Only 3 misclassifications in 1,000 test cases

## Security Considerations

1. **API Key**: Always use strong, random API keys in production
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
2. **CORS**: Configure `ALLOWED_ORIGINS` in `.env` to restrict access
3. **Input Validation**: All inputs are validated and sanitized
4. **HTTPS**: Use HTTPS in production (configure reverse proxy)
5. **Container Security**: Docker runs as non-root user
6. **Environment Variables**: Never commit `.env` file to git

## Learning Resources

This project demonstrates:
- **Data Engineering**: Multi-source data collection and cleaning
- **Fine-tuning**: Transfer learning with transformer models
- **API Development**: Production-ready FastAPI implementation
- **Docker Deployment**: Containerization and orchestration
- **Testing**: Comprehensive test coverage

## Performance

- **Inference Time**: 
  - GPU (CUDA): ~50ms per email
  - CPU: ~200ms per email
- **Accuracy**: 99.7% on test samples
- **Memory**: ~500MB for model, ~1GB total with API
- **Throughput**: Up to 20 requests/second on GPU
- **GPU Support**: Automatic CUDA detection (RTX 4060 tested)

## Monitoring

The API provides:
- Health check endpoint (`/health`)
- Structured JSON logging
- Request/response metrics
- Error tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/sergiugogo/ISM-AIPhishingGuard/issues)
- Check the API documentation at `/docs` (when DEBUG=true)

## Acknowledgments

**Datasets:**
- [zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset)
- [SetFit/enron_spam](https://huggingface.co/datasets/SetFit/enron_spam)
- [PhishTank](https://phishtank.org/) - Verified phishing URLs

**Technologies:**
- [RoBERTa](https://huggingface.co/roberta-base) by Facebook AI
- [FastAPI](https://fastapi.tiangolo.com/)
- [Transformers](https://huggingface.co/transformers/) by HuggingFace
- [PyTorch](https://pytorch.org/)

## Roadmap

- [ ] Extended evaluation on larger test sets
- [ ] Additional data sources integration
- [ ] Real-time phishing URL database updates
- [ ] Email header analysis
- [ ] Multi-language support
- [ ] Performance benchmarking suite


