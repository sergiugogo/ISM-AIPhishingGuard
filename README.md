# PhishGuard - AI-Powered Phishing Detection API

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

PhishGuard is a production-ready API service for detecting phishing emails using a fine-tuned DistilRoBERTa transformer model. It provides real-time email analysis with confidence scores and detailed explanations.

## Features

- 🤖 **AI-Powered Detection**: Fine-tuned DistilRoBERTa model for accurate phishing detection
- 🚀 **Production Ready**: Comprehensive error handling, logging, and monitoring
- 🔒 **Secure**: API key authentication and rate limiting
- 📊 **Feature Extraction**: Automatic detection of suspicious URLs, urgency language, and more
- 🐳 **Containerized**: Docker and Docker Compose support
- 📝 **Well Documented**: OpenAPI/Swagger documentation
- ✅ **Tested**: Comprehensive test suite

## Quick Start

### Prerequisites

- Python 3.11+
- pip
- (Optional) Docker and Docker Compose

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd phishguard
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env and set your API_KEY and other settings
```

4. **Train the model** (if not already trained)
```bash
python src/train.py
```

5. **Run the API**
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

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
├── models/                   # Trained models
├── data/                     # Training data
├── tests/                    # Test suite
├── config.py                # Configuration
├── Dockerfile               # Docker image
├── docker-compose.yml       # Docker Compose
└── requirements.txt         # Python dependencies
```

## Model Training

The model is fine-tuned on phishing email datasets:

```bash
python src/train.py
```

**Training features:**
- Base model: DistilRoBERTa
- Automatic GPU detection
- Checkpoint saving
- Progress logging

## Security Considerations

1. **API Key**: Always use strong, random API keys in production
2. **Rate Limiting**: Configure appropriate rate limits for your use case
3. **Input Validation**: All inputs are validated and sanitized
4. **HTTPS**: Use HTTPS in production (configure reverse proxy)
5. **Container Security**: Run as non-root user in Docker

## Performance

- **Inference Time**: ~100-200ms per email (CPU)
- **GPU Support**: Automatic CUDA detection for faster inference
- **Memory**: ~1GB RAM for model + API
- **Throughput**: 60+ requests/minute (configurable)

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
- Create an issue on GitHub
- Check the API documentation at `/docs`

## Roadmap

- [ ] Redis-based rate limiting
- [ ] Prometheus metrics
- [ ] Batch prediction endpoint
- [ ] Email reputation checking
- [ ] Advanced feature extraction
- [ ] Multi-language support


