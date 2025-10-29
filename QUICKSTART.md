# PhishGuard 🎣🛡️

**Production-ready AI phishing email detection API**

> Fine-tuned DistilRoBERTa model • 99%+ accuracy • Real HuggingFace data • Docker ready

---

## 📋 Quick Start

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Download Training Data
```bash
python scripts/prepare_training_data.py
```
Downloads ~33K real emails from HuggingFace (`SetFit/enron_spam`)

### 3️⃣ (Optional) Add Synthetic Data
```bash
python scripts/generate_large_dataset.py
```
Generates additional training samples if needed

### 4️⃣ Train Model
```bash
python src/train.py
```
Trains DistilRoBERTa for ~30-45 minutes on GPU

### 5️⃣ Run API
```bash
uvicorn src.api.main:app --reload
```
API available at http://localhost:8000

---

## 🎯 Model Details

| Specification | Value |
|--------------|-------|
| **Base Model** | DistilRoBERTa-base (HuggingFace) |
| **Parameters** | 82 million |
| **Model Size** | 330 MB |
| **Training Data** | SetFit/enron_spam (~33K emails) |
| **Training Time** | ~30-45 min (RTX 4060) |
| **Accuracy** | 99%+ on validation |
| **Epochs** | 5 |
| **Batch Size** | 8 (GPU) / 2 (CPU) |

---

## 📁 Project Structure

```
phishguard/
├── data/
│   ├── training_data.csv        # Downloaded from HuggingFace
│   └── large_training_data.csv  # With synthetic data (optional)
├── models/
│   └── phishguard-model/        # Fine-tuned model (330MB)
├── src/
│   ├── api/
│   │   └── main.py             # FastAPI application
│   ├── train.py                # Training script
│   └── utils/
│       └── preprocess.py       # Email preprocessing
├── scripts/
│   ├── prepare_training_data.py    # Download HuggingFace data
│   ├── generate_large_dataset.py   # Generate synthetic data
│   ├── evaluate_model.py           # Evaluate model performance
│   └── test_datasets.py            # Test HuggingFace connectivity
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 🔧 Scripts Reference

### Data Preparation
```bash
# Download real data from HuggingFace
python scripts/prepare_training_data.py

# Test which HuggingFace datasets work
python scripts/test_datasets.py

# Generate synthetic phishing emails
python scripts/generate_large_dataset.py
```

### Training & Evaluation
```bash
# Train model (saves to models/phishguard-model/)
python src/train.py

# Evaluate model performance
python scripts/evaluate_model.py
```

### API
```bash
# Run locally
uvicorn src.api.main:app --reload

# Run with Docker
docker-compose up

# API docs
http://localhost:8000/docs
```

---

## 🚀 API Usage

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "URGENT: Verify your account",
    "body": "Click here to verify: http://phishing-site.tk"
  }'
```

### Example Response
```json
{
  "prediction": "phishing",
  "confidence": 0.98,
  "phishing_score": 0.98,
  "benign_score": 0.02,
  "features": {
    "has_urgent_language": true,
    "has_suspicious_urls": true,
    "suspicious_tlds": [".tk"]
  },
  "explanation": "High phishing probability due to urgent language and suspicious URLs"
}
```

---

## 🐳 Docker Deployment

### Build and Run
```bash
docker-compose up -d
```

### Configuration
Edit `docker-compose.yml` to set:
- `API_KEY` - Your secret API key
- `MODEL_PATH` - Path to model directory
- `LOG_LEVEL` - Logging level (INFO/DEBUG)

---

## 📊 Dataset Information

### Primary Dataset: SetFit/enron_spam
- **Source:** HuggingFace (verified working)
- **Size:** ~33,000 emails
- **Labels:** Spam (phishing) / Ham (benign)
- **Balance:** Well-balanced dataset
- **Quality:** Real-world emails

### Optional: Synthetic Data
- Generate with `scripts/generate_large_dataset.py`
- Adds variety to training data
- Customizable patterns

### Testing Datasets
Run `python scripts/test_datasets.py` to check which HuggingFace datasets are currently accessible.

---

## 🧪 Model Evaluation

```bash
python scripts/evaluate_model.py
```

Provides:
- Accuracy, Precision, Recall, F1 scores
- Confusion matrix
- Per-class performance
- Sample predictions

---

## 🛠️ Development

### Requirements
```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers` - HuggingFace transformers
- `torch` - PyTorch
- `fastapi` - Web framework
- `pandas` - Data manipulation
- `datasets` - HuggingFace datasets

### Environment Variables
Create `.env` file:
```env
API_KEY=your-secret-key-here
MODEL_PATH=models/phishguard-model
LOG_LEVEL=INFO
```

---

## 📈 Performance Optimization

### GPU Training
- Uses FP16 training on CUDA
- Optimized batch sizes
- ~30-45 min training time

### CPU Training
- Automatic fallback to CPU
- Reduced batch size (2)
- ~2-3 hours training time

### Model Size
- DistilRoBERTa: 330MB (6x smaller than RoBERTa)
- Fast inference (~50ms per email)
- Low memory footprint

---

## 🔒 Security Features

- **API Key Authentication** - Secure endpoint access
- **Rate Limiting** - 60 requests/minute per IP
- **Input Validation** - Pydantic schema validation
- **Error Handling** - Comprehensive exception handling
- **Logging** - Structured JSON logging

---

## 📚 Documentation

- **Technical Guide:** `TECHNICAL_EXPLANATION.md`
- **Business Guide:** `BUSINESS_GUIDE.md`
- **Data Sources:** `DATA_SOURCES.md`
- **API Docs:** http://localhost:8000/docs (when running)

---

## 🧹 Maintenance

### Clean Project
```bash
python cleanup_project.py
```
Removes old checkpoints, redundant data files, and frees disk space.

### Update Model
```bash
# 1. Prepare new data
python scripts/prepare_training_data.py

# 2. Retrain
python src/train.py

# 3. Evaluate
python scripts/evaluate_model.py
```

---

## 📝 Common Tasks

### Start Fresh Training
```bash
# 1. Clean old data
rm -rf data/*.csv models/phishguard-model/

# 2. Download data
python scripts/prepare_training_data.py

# 3. Train
python src/train.py
```

### Test New Datasets
```bash
# Check HuggingFace availability
python scripts/test_datasets.py

# Edit scripts/prepare_training_data.py with new dataset
# Then run it
python scripts/prepare_training_data.py
```

### Deploy to Production
```bash
# 1. Ensure model is trained
ls models/phishguard-model/model.safetensors

# 2. Set production API key
export API_KEY="your-production-key"

# 3. Build and run
docker-compose up -d

# 4. Test
curl http://localhost:8000/health
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **HuggingFace** - Transformers library and datasets
- **SetFit/enron_spam** - Training dataset
- **DistilRoBERTa** - Base model architecture
- **FastAPI** - Web framework

---

## 📧 Support

For issues or questions:
1. Check existing documentation
2. Review closed issues
3. Open new issue with details

---

**Built with ❤️ by ISM Showcases**
