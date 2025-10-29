# PhishGuard API - Production Deployment Guide

## Quick Start

### 1. Prerequisites
- Python 3.11+
- 4GB+ RAM (for model)
- GPU (optional, for faster inference)

### 2. Installation

```bash
# Clone repository
git clone <your-repo>
cd phishguard

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and set your API key
# IMPORTANT: Generate a strong API key!
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 4. Run the API

#### Development Mode
```bash
python scripts/start_api.py --reload
```

#### Production Mode
```bash
python scripts/start_api.py --workers 4
```

#### Using Docker
```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Predict Email
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "subject": "Urgent: Verify your account",
    "body": "Click here: http://192.168.1.1/verify"
  }'
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## Performance

- **Model Size**: ~500 MB (RoBERTa-base)
- **Inference Time**: ~50-100ms per email (GPU), ~200-400ms (CPU)
- **Accuracy**: 99.7% on validation set
- **Recommended**: 4 workers for production on 4-core CPU

## Security Best Practices

1. **API Key**: Always use a strong, randomly generated API key
2. **HTTPS**: Enable SSL/TLS in production (configure in `start_api.py`)
3. **CORS**: Restrict `ALLOWED_ORIGINS` to your frontend domains
4. **Rate Limiting**: Default 60 requests/minute per API key
5. **Logging**: Monitor logs for suspicious activity

## Monitoring

### Health Check
The `/health` endpoint returns:
- Service status
- Model loading status
- API version
- Device (CPU/CUDA)

### Metrics
Enable metrics in `.env`:
```
ENABLE_METRICS=true
```

## Troubleshooting

### Model Not Found
```
ERROR: Model not found at models/phishguard-model
```
**Solution**: Train the model first:
```bash
python src/train.py
```

### Out of Memory
**Solution**: Reduce workers or use CPU:
```bash
python scripts/start_api.py --workers 1
```

### Import Errors
**Solution**: Ensure PYTHONPATH includes project root:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Production Deployment

### Using Gunicorn (Linux)
```bash
gunicorn src.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

### Using systemd (Linux)
Create `/etc/systemd/system/phishguard.service`:
```ini
[Unit]
Description=PhishGuard API
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/phishguard
Environment="PATH=/opt/phishguard/.venv/bin"
ExecStart=/opt/phishguard/.venv/bin/python scripts/start_api.py

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable phishguard
sudo systemctl start phishguard
sudo systemctl status phishguard
```

### Using Docker in Production
```bash
# Build
docker build -t phishguard-api .

# Run with proper resource limits
docker run -d \
  --name phishguard-api \
  -p 8000:8000 \
  -e API_KEY=your-strong-api-key \
  --memory=4g \
  --cpus=2 \
  -v $(pwd)/models:/app/models:ro \
  phishguard-api
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | (required) | API authentication key |
| `MODEL_DIR` | `models/phishguard-model` | Path to trained model |
| `DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | Log format (json/text) |
| `MAX_LENGTH` | `256` | Max token length |
| `RATE_LIMIT_PER_MINUTE` | `60` | Rate limit per API key |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `WORKERS` | auto | Number of workers |

## Model Information

- **Architecture**: RoBERTa-base (125M parameters)
- **Training Data**: 51,368 emails (3 sources combined)
- **Validation Accuracy**: 99.70%
- **Precision**: 99.80%
- **Recall**: 99.60%
- **F1 Score**: 99.70%

## Support

For issues or questions, please open an issue on GitHub.
