# PhishGuard - AI-Powered Phishing Detection API# PhishGuard - AI-Powered Phishing Detection API# PhishGuard - AI-Powered Phishing Detection API



[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

PhishGuard is a production-ready API service for detecting phishing emails using a fine-tuned RoBERTa transformer model. Achieves **98.81% accuracy** with rigorous evaluation on clean, held-out test data. Built as a learning project to understand the complete ML pipeline from data collection to deployment while maintaining research-grade evaluation standards.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 🎯 Project Goals

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This project was created as a comprehensive learning experience covering:

- **End-to-end ML pipeline**: Data collection, cleaning, and quality validation

- **Rigorous evaluation**: Preventing data leakage and ensuring honest metrics

- **Transformer fine-tuning**: Training RoBERTa-base for focused classification tasksPhishGuard is a production-ready API service for detecting phishing emails using a fine-tuned RoBERTa transformer model. Achieves **98.81% accuracy** with rigorous evaluation on clean, held-out test data. Built as a learning project to understand the complete ML pipeline from data collection to deployment while maintaining research-grade evaluation standards.PhishGuard is a production-ready API service for detecting phishing emails using a fine-tuned RoBERTa transformer model. Built as a learning project to understand the complete ML pipeline from data collection to deployment, it achieves 99.7% accuracy on test samples and provides real-time email analysis with detailed explanations.

- **Production deployment**: Building scalable, documented APIs with FastAPI

- **Best practices**: Testing, monitoring, containerization, and security



**Key Learning**: 60% of effort went into data quality and preventing evaluation leakage—critical for real-world ML.## 🎯 Project Goals## 🎯 Project Goals



## ✨ Features



- 🤖 **Fine-tuned RoBERTa-base**: 125M parameter transformer modelThis project was created as a comprehensive learning experience covering:This project was created as a comprehensive learning experience to understand:

- 🎯 **High Accuracy**: **98.81%** on clean test set (Precision: 97.76%, Recall: 99.49%, ROC-AUC: 0.9991)

- 🔬 **Rigorous Evaluation**: Zero data leakage, stratified splits, held-out test set- **End-to-end ML pipeline**: Data collection, cleaning, and quality validation- End-to-end fine-tuning of transformer models

- ⚡ **Fast Inference**: ~50ms on GPU, ~200ms on CPU

- 🔍 **Explainable AI**: Detailed reasoning for each prediction- **Rigorous evaluation**: Preventing data leakage and ensuring honest metrics- Data collection and preparation for ML

- 📊 **Feature Extraction**: IP detection, urgency analysis, URL scanning

- 🐳 **Docker Ready**: Easy deployment with Docker Compose- **Transformer fine-tuning**: Training RoBERTa-base for focused classification tasks- Building production-ready ML APIs

- 📝 **Well Documented**: OpenAPI/Swagger documentation

- ✅ **Comprehensive Testing**: Full test suite with 17+ test cases- **Production deployment**: Building scalable, documented APIs with FastAPI- Model evaluation and deployment



## Quick Start- **Best practices**: Testing, monitoring, containerization, and security



### PrerequisitesIt demonstrates that you don't need massive models to build effective, focused ML applications.



- Python 3.11+**Key Learning**: 60% of effort went into data quality and preventing evaluation leakage—critical for real-world ML.

- 4GB+ RAM

- (Optional) NVIDIA GPU with CUDA support for faster training/inference## ✨ Features

- (Optional) Docker and Docker Compose

## ✨ Features

### Installation

- 🤖 **Fine-tuned RoBERTa-base**: 125M parameter transformer model

1. **Clone the repository**

```bash- 🤖 **Fine-tuned RoBERTa-base**: 125M parameter transformer model- 🎯 **High Accuracy**: 99.7% on 1,000 test samples (Precision: 99.8%, Recall: 99.6%)

git clone https://github.com/sergiugogo/ISM-AIPhishingGuard.git

cd ISM-AIPhishingGuard- 🎯 **High Accuracy**: **98.81%** on clean test set (Precision: 97.76%, Recall: 99.49%, ROC-AUC: 0.9991)- ⚡ **Fast Inference**: ~50ms on GPU, ~200ms on CPU

```

- 🔬 **Rigorous Evaluation**: Zero data leakage, stratified splits, held-out test set- 🔍 **Explainable AI**: Detailed reasoning for each prediction

2. **Create virtual environment**

```bash- ⚡ **Fast Inference**: ~50ms on GPU, ~200ms on CPU- 📊 **Feature Extraction**: IP detection, urgency analysis, URL scanning

python -m venv .venv

# On Windows:- 🔍 **Explainable AI**: Detailed reasoning for each prediction- 🐳 **Docker Ready**: Easy deployment with Docker Compose

.venv\Scripts\activate

# On Linux/Mac:- 📊 **Feature Extraction**: IP detection, urgency analysis, URL scanning- 📝 **Well Documented**: OpenAPI/Swagger documentation

source .venv/bin/activate

```- 🐳 **Docker Ready**: Easy deployment with Docker Compose- ✅ **Tested**: Comprehensive test suite



3. **Install dependencies**- 📝 **Well Documented**: OpenAPI/Swagger documentation

```bash

pip install -r requirements.txt- ✅ **Comprehensive Testing**: Full test suite with 17+ test cases## Quick Start

```



4. **Configure environment**

```bash## Quick Start### Prerequisites

cp .env.example .env

# Edit .env and set your API_KEY

# Generate a strong key:

python -c "import secrets; print(secrets.token_urlsafe(32))"### Prerequisites- Python 3.11+

```

- 4GB+ RAM

5. **Prepare training data**

```bash- Python 3.11+- (Optional) NVIDIA GPU with CUDA support for faster inference

# Downloads and combines datasets with proper splitting

# Removes synthetic templates from validation/test sets- 4GB+ RAM- (Optional) Docker and Docker Compose

python scripts/prepare_training_data.py

```- (Optional) NVIDIA GPU with CUDA support for faster training/inference



6. **Train the model** (takes ~45-60 minutes on GPU)- (Optional) Docker and Docker Compose### Installation

```bash

python src/train.py

```

### Installation1. **Clone the repository**

7. **Evaluate the model**

```bash```bash

python scripts/evaluate_model.py

```1. **Clone the repository**git clone https://github.com/sergiugogo/ISM-AIPhishingGuard.git



8. **Run the API**```bashcd ISM-AIPhishingGuard

```bash

# Development modegit clone https://github.com/sergiugogo/ISM-AIPhishingGuard.git```

python scripts/start_api.py --reload

cd ISM-AIPhishingGuard

# Production mode (4 workers)

python scripts/start_api.py --workers 4```2. **Create virtual environment**

```

```bash

Visit `http://localhost:8000/docs` for interactive API documentation (when DEBUG=true).

2. **Create virtual environment**python -m venv .venv

### Docker Deployment

```bash# On Windows:

```bash

# Build and run with Docker Composepython -m venv .venv.venv\Scripts\activate

docker-compose up -d

# On Windows:# On Linux/Mac:

# Check logs

docker-compose logs -f.venv\Scripts\activatesource .venv/bin/activate



# Stop# On Linux/Mac:```

docker-compose down

```source .venv/bin/activate



## API Usage```3. **Install dependencies**



### Authentication```bash



All requests require an API key in the header:3. **Install dependencies**pip install -r requirements.txt

```

X-API-Key: your-secret-api-key-here```bash```

```

pip install -r requirements.txt

### Endpoints

```4. **Configure environment**

#### Health Check

```bash```bash

curl http://localhost:8000/health

```4. **Configure environment**cp .env.example .env



#### Predict Email```bash# Edit .env and set your API_KEY

```bash

curl -X POST http://localhost:8000/predict \cp .env.example .env# Recommended: Generate a strong key

  -H "Content-Type: application/json" \

  -H "X-API-Key: your-secret-api-key-here" \# Edit .env and set your API_KEY# python -c "import secrets; print(secrets.token_urlsafe(32))"

  -d '{

    "subject": "Urgent: Verify your account",# Generate a strong key:```

    "body": "Click here to verify: http://192.168.1.1/verify"

  }'python -c "import secrets; print(secrets.token_urlsafe(32))"

```

```5. **Prepare training data**

**Response:**

```json```bash

{

  "label": "phishing",5. **Prepare training data**# This downloads and combines datasets from HuggingFace and PhishTank

  "confidence": 0.95,

  "phishing_score": 0.95,```bashpython scripts/prepare_training_data.py

  "benign_score": 0.05,

  "explanation": "Classified as PHISHING (95.0% confidence). Red flags: URL contains IP address instead of domain name; Uses urgent language (2 urgency indicators)",# Downloads and combines datasets with proper splitting```

  "features": {

    "num_urls": 1,# Removes synthetic templates from validation/test sets

    "has_ip_in_url": true,

    "has_suspicious_tld": false,python scripts/prepare_training_data.py6. **Train the model** (takes ~45-60 minutes on GPU)

    "urgency_hits": 2,

    "urls": ["http://192.168.1.1/verify"]``````bash

  }

}python src/train.py

```

6. **Train the model** (takes ~45-60 minutes on GPU)```

## Model Performance

```bash

### Honest Evaluation Metrics

python src/train.py7. **Evaluate the model**

Tested on **4,108 held-out emails** (57% benign, 43% phishing):

``````bash

| Metric | Score | Notes |

|--------|-------|-------|python scripts/evaluate_model.py

| **Accuracy** | **98.81%** | Overall classification accuracy |

| **Precision** | 97.76% | When flagging phishing, correct 97.76% of time |7. **Evaluate the model**```

| **Recall** | 99.49% | Catches 99.49% of actual phishing emails |

| **F1 Score** | 98.62% | Harmonic mean of precision and recall |```bash

| **ROC-AUC** | 0.9991 | Near-perfect discrimination capability |

python scripts/evaluate_model.py8. **Run the API**

### Error Analysis

``````bash

- **False Positives**: 40 out of 2,351 benign (1.70% FPR)

  - Legitimate emails incorrectly flagged as phishing# Development mode

  - Examples: Birth certificates, promotional emails, password resets

  8. **Run the API**python scripts/start_api.py --reload

- **False Negatives**: 9 out of 1,757 phishing (0.51% FNR)

  - Phishing emails that slipped through```bash

  - Mostly subtle, well-crafted phishing without obvious red flags

# Development mode# Production mode

### Per-Class Performance

python scripts/start_api.py --reloadpython scripts/start_api.py --workers 4

**Benign Emails:**

- Precision: 99.61%```

- Recall: 98.30%

- F1: 98.95%# Production mode (4 workers)



**Phishing Emails:**python scripts/start_api.py --workers 4Visit `http://localhost:8000/docs` for interactive API documentation (when DEBUG=true).

- Precision: 97.76%

- Recall: 99.49%```

- F1: 98.62%

### Docker Deployment

## Data Quality & Methodology

Visit `http://localhost:8000/docs` for interactive API documentation (when DEBUG=true).

### Dataset Sources

```bash

1. **HuggingFace Datasets**:

   - `zefang-liu/phishing-email-dataset` (18,650 emails)### Docker Deployment# Build and run with Docker Compose

   - `SetFit/enron_spam` (31,716 emails)

docker-compose up -d

**Total**: 51,368 emails after combining sources

```bash

### Data Preparation Pipeline

# Build and run with Docker Compose# Check logs

```bash

python scripts/prepare_training_data.pydocker-compose up -ddocker-compose logs -f

```



**Key Steps:**

1. ✅ **Subject deduplication**: Removes duplicate subjects across all sources# Check logs# Stop

2. ✅ **Synthetic template filtering**: Separates synthetic phishing templates from eval sets

3. ✅ **Generic spam removal**: Filters obvious spam (viagra, casino, etc.) from validation/testdocker-compose logs -fdocker-compose down

4. ✅ **Stratified splitting**: Ensures balanced classes in all splits

5. ✅ **Eval isolation**: Validation/test sets contain only real phishing vs real benign```



**Data Splits:**# Stop

- Training: 32,861 emails (80%) - includes synthetic templates for robustness

- Validation: 4,108 emails (10%) - clean, balanced, no syntheticdocker-compose down## API Usage

- Test: 4,108 emails (10%) - held-out, never seen during training

```

**Balance:**

- Training: 51.2% phishing / 48.8% benign### Authentication

- Validation: 42.8% phishing / 57.2% benign

- Test: 42.8% phishing / 57.2% benign## API Usage



### Preventing Data LeakageAll requests require an API key in the header:



❌ **Common pitfalls we avoided:**### Authentication```

1. Subject overlap between train/validation/test

2. Synthetic template memorization in eval setsX-API-Key: your-secret-api-key-here

3. Generic spam mislabeled as phishing

4. Using validation set for test metricsAll requests require an API key in the header:```



✅ **Our approach:**```

1. Deduplicate subjects before splitting

2. Remove templates from validation/test (keep in training for robustness)X-API-Key: your-secret-api-key-here### Endpoints

3. Filter generic spam patterns from eval sets

4. Maintain truly held-out test set```



## Training Configuration#### Health Check



- **Base Model**: `roberta-base` (125M parameters)### Endpoints```bash

- **Training Samples**: 32,861 emails

- **Validation Samples**: 4,108 emailscurl http://localhost:8000/health

- **Epochs**: 5

- **Batch Size**: 8#### Health Check```

- **Learning Rate**: 2e-5

- **Optimization**: FP16 mixed precision on GPU```bash

- **Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU

- **Training Time**: ~45-60 minutescurl http://localhost:8000/health#### Predict Email



## Configuration``````bash



Edit `.env` or set environment variables:curl -X POST http://localhost:8000/predict \



| Variable | Description | Default |#### Predict Email  -H "Content-Type: application/json" \

|----------|-------------|---------|

| `MODEL_DIR` | Path to trained model | `models/phishguard-model` |```bash  -H "X-API-Key: your-secret-api-key-here" \

| `LOG_LEVEL` | Logging level | `INFO` |

| `MAX_CONTENT_LENGTH` | Max email content length | `10000` |curl -X POST http://localhost:8000/predict \  -d '{

| `RATE_LIMIT_PER_MINUTE` | API rate limit | `60` |

| `API_KEY` | API authentication key | `your-secret-api-key-here` |  -H "Content-Type: application/json" \    "subject": "Urgent: Verify your account",

| `DEBUG` | Enable debug mode | `false` |

| `ALLOWED_ORIGINS` | CORS origins (JSON array) | `["*"]` |  -H "X-API-Key: your-secret-api-key-here" \    "body": "Click here to verify: http://192.168.1.1/verify"



## Development  -d '{  }'



### Project Structure    "subject": "Urgent: Verify your account",```

```

phishguard/    "body": "Click here to verify: http://192.168.1.1/verify"

├── src/

│   ├── api/  }'**Response:**

│   │   ├── main.py          # FastAPI application

│   │   └── schemas.py       # Pydantic models``````json

│   ├── core/

│   │   ├── logger.py        # Logging configuration{

│   │   ├── exceptions.py    # Custom exceptions

│   │   └── model_manager.py # Model loading/inference**Response:**  "label": "phishing",

│   ├── utils/

│   │   └── preprocess.py    # Email preprocessing```json  "confidence": 0.95,

│   └── train.py             # Model training script

├── scripts/{  "phishing_score": 0.95,

│   ├── prepare_training_data.py  # Clean data pipeline

│   ├── evaluate_model.py         # Comprehensive evaluation  "label": "phishing",  "benign_score": 0.05,

│   └── start_api.py              # Production startup

├── models/  "confidence": 0.95,  "explanation": "Classified as PHISHING (95.0% confidence). Red flags: URL contains IP address instead of domain name; Uses urgent language (2 urgency indicators)",

│   └── phishguard-model/    # Trained model (500MB)

├── data/  "phishing_score": 0.95,  "features": {

│   ├── train.csv            # Training set (32,861 emails)

│   ├── validation.csv       # Validation set (4,108 emails)  "benign_score": 0.05,    "num_urls": 1,

│   └── test.csv             # Test set (4,108 emails)

├── tests/                   # Test suite  "explanation": "Classified as PHISHING (95.0% confidence). Red flags: URL contains IP address instead of domain name; Uses urgent language (2 urgency indicators)",    "has_ip_in_url": true,

├── config.py               # Configuration management

├── Dockerfile              # Docker image  "features": {    "has_suspicious_tld": false,

├── docker-compose.yml      # Docker Compose

└── requirements.txt        # Dependencies    "num_urls": 1,    "urgency_hits": 2,

```

    "has_ip_in_url": true,    "urls": ["http://192.168.1.1/verify"]

### Running Tests

```bash    "has_suspicious_tld": false,  }

pytest tests/ -v

```    "urgency_hits": 2,}



### Code Quality    "urls": ["http://192.168.1.1/verify"]```

- Type hints throughout

- Pydantic for validation  }

- Structured logging (JSON format)

- Comprehensive error handling}## Configuration

- 17+ test cases covering all endpoints

```

## Performance

Edit `.env` or set environment variables:

- **Inference Time**: 

  - GPU (CUDA): ~50ms per email## Model Performance

  - CPU: ~200ms per email

- **Accuracy**: 98.81% on clean test set| Variable | Description | Default |

- **Memory**: ~500MB for model, ~1GB total with API

- **Throughput**: Up to 20 requests/second on GPU### Honest Evaluation Metrics|----------|-------------|---------|

- **GPU Support**: Automatic CUDA detection

| `MODEL_DIR` | Path to trained model | `models/phishguard-model` |

## Security Considerations

Tested on **4,108 held-out emails** (57% benign, 43% phishing):| `LOG_LEVEL` | Logging level | `INFO` |

1. **API Key**: Use strong, random keys in production

2. **CORS**: Configure `ALLOWED_ORIGINS` to restrict access| `MAX_CONTENT_LENGTH` | Max email content length | `10000` |

3. **Input Validation**: All inputs validated and sanitized

4. **HTTPS**: Use HTTPS in production (reverse proxy)| Metric | Score | Notes || `RATE_LIMIT_PER_MINUTE` | API rate limit | `60` |

5. **Container Security**: Docker runs as non-root user

6. **Environment Variables**: Never commit `.env` to git|--------|-------|-------|| `API_KEY` | API authentication key | `your-secret-api-key-here` |



## Monitoring & Observability| **Accuracy** | **98.81%** | Overall classification accuracy || `DEBUG` | Enable debug mode | `false` |



The API provides:| **Precision** | 97.76% | When flagging phishing, correct 97.76% of time |

- Health check endpoint (`/health`)

- Structured JSON logging| **Recall** | 99.49% | Catches 99.49% of actual phishing emails |## Development

- Request/response tracking

- Error monitoring| **F1 Score** | 98.62% | Harmonic mean of precision and recall |

- Model version info

| **ROC-AUC** | 0.9991 | Near-perfect discrimination capability |### Running Tests

## Contributing

```bash

1. Fork the repository

2. Create a feature branch (`git checkout -b feature/amazing-feature`)### Error Analysispytest tests/ -v

3. Make your changes

4. Add tests```

5. Commit changes (`git commit -m 'Add amazing feature'`)

6. Push to branch (`git push origin feature/amazing-feature`)- **False Positives**: 40 out of 2,351 benign (1.70% FPR)

7. Open a Pull Request

  - Legitimate emails incorrectly flagged as phishing### Code Structure

## License

  - Examples: Birth certificates, promotional emails, password resets```

MIT License - see LICENSE file for details

  phishguard/

## Support

- **False Negatives**: 9 out of 1,757 phishing (0.51% FNR)├── src/

For issues and questions:

- Open an issue on [GitHub](https://github.com/sergiugogo/ISM-AIPhishingGuard/issues)  - Phishing emails that slipped through│   ├── api/

- Check API docs at `/docs` (when DEBUG=true)

- See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment help  - Mostly subtle, well-crafted phishing without obvious red flags│   │   ├── main.py          # FastAPI application



## Acknowledgments│   │   └── schemas.py       # Pydantic models



**Datasets:**### Per-Class Performance│   ├── core/

- [zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset)

- [SetFit/enron_spam](https://huggingface.co/datasets/SetFit/enron_spam)│   │   ├── logger.py        # Logging configuration



**Technologies:****Benign Emails:**│   │   ├── exceptions.py    # Custom exceptions

- [RoBERTa](https://huggingface.co/roberta-base) by Facebook AI

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework- Precision: 99.61%│   │   └── model_manager.py # Model loading/inference

- [Transformers](https://huggingface.co/transformers/) by HuggingFace

- [PyTorch](https://pytorch.org/) - Deep learning framework- Recall: 98.30%│   ├── utils/



## Lessons Learned- F1: 98.95%│   │   └── preprocess.py    # Email preprocessing



1. **Data quality matters most**: Spent 60% of time on data collection, cleaning, and preventing leakage│   └── train.py             # Model training script

2. **Evaluation rigor is critical**: Easy to inflate metrics with data leakage

3. **Domain knowledge helps**: Understanding phishing patterns improved feature engineering**Phishing Emails:**├── scripts/

4. **Production readiness**: API development, testing, and deployment are as important as the model

5. **Documentation**: Clear docs make the project accessible and maintainable- Precision: 97.76%│   ├── prepare_training_data.py  # Multi-source data collection



## Roadmap- Recall: 99.49%│   ├── evaluate_model.py         # Model evaluation



- [x] Rigorous data cleaning and deduplication- F1: 98.62%│   └── start_api.py              # Production startup script

- [x] Prevent synthetic template leakage

- [x] Comprehensive evaluation with ROC-AUC├── models/

- [x] Production-ready API with authentication

- [x] Docker deployment## Data Quality & Methodology│   └── phishguard-model/    # Trained model files

- [ ] Real-time phishing URL database integration

- [ ] Email header analysis (SPF, DKIM, DMARC)├── data/                     # Training/validation data (gitignored)

- [ ] Multi-language support

- [ ] Model quantization for faster inference### Dataset Sources├── tests/                    # Test suite

- [ ] Continuous learning pipeline

├── config.py                # Configuration management

---

1. **HuggingFace Datasets**:├── Dockerfile               # Docker image

**Built with ❤️ as a learning project | Production-ready with research-grade evaluation**

   - `zefang-liu/phishing-email-dataset` (18,650 emails)├── docker-compose.yml       # Docker Compose

   - `SetFit/enron_spam` (31,716 emails)├── .env.example             # Environment template

└── requirements.txt         # Python dependencies

**Total**: 51,368 emails after combining sources```



### Data Preparation Pipeline## Model Training



```bashPhishGuard uses a multi-source dataset approach to ensure quality and balance:

python scripts/prepare_training_data.py

```### Data Sources

1. **HuggingFace Datasets**:

**Key Steps:**   - zefang-liu/phishing-email-dataset (18,650 emails)

1. ✅ **Subject deduplication**: Removes duplicate subjects across all sources   - SetFit/enron_spam (31,716 emails)

2. ✅ **Synthetic template filtering**: Separates synthetic phishing templates from eval sets2. **PhishTank**: Verified phishing URLs converted to email samples (5,000 emails)

3. ✅ **Generic spam removal**: Filters obvious spam (viagra, casino, etc.) from validation/test

4. ✅ **Stratified splitting**: Ensures balanced classes in all splits### Training Process

5. ✅ **Eval isolation**: Validation/test sets contain only real phishing vs real benign

```bash

**Data Splits:**# 1. Prepare data (downloads and combines all sources)

- Training: 32,861 emails (80%) - includes synthetic templates for robustnesspython scripts/prepare_training_data.py

- Validation: 4,108 emails (10%) - clean, balanced, no synthetic

- Test: 4,108 emails (10%) - held-out, never seen during training# 2. Train the model

python src/train.py

**Balance:**

- Training: 51.2% phishing / 48.8% benign# 3. Evaluate performance

- Validation: 42.8% phishing / 57.2% benignpython scripts/evaluate_model.py

- Test: 42.8% phishing / 57.2% benign```



### Preventing Data Leakage**Training Configuration:**

- Base model: RoBERTa-base (125M parameters)

❌ **Common pitfalls we avoided:**- Training samples: 41,094 emails (80%)

1. Subject overlap between train/validation/test- Validation samples: 10,274 emails (20%)

2. Synthetic template memorization in eval sets- Epochs: 5

3. Generic spam mislabeled as phishing- Batch size: 8

4. Using validation set for test metrics- Learning rate: 2e-5

- Optimization: FP16 on GPU

✅ **Our approach:**- Training time: ~45-60 minutes (RTX 4060 GPU)

1. Deduplicate subjects before splitting

2. Remove templates from validation/test (keep in training for robustness)**Model Performance:**

3. Filter generic spam patterns from eval sets- Test Accuracy: 99.7% (on 1,000 validation samples)

4. Maintain truly held-out test set- Precision: 99.8%

- Recall: 99.6%

## Training Configuration- F1 Score: 99.7%

- Only 3 misclassifications in 1,000 test cases

- **Base Model**: `roberta-base` (125M parameters)

- **Training Samples**: 32,861 emails## Security Considerations

- **Validation Samples**: 4,108 emails

- **Epochs**: 51. **API Key**: Always use strong, random API keys in production

- **Batch Size**: 8   ```bash

- **Learning Rate**: 2e-5   python -c "import secrets; print(secrets.token_urlsafe(32))"

- **Optimization**: FP16 mixed precision on GPU   ```

- **Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU2. **CORS**: Configure `ALLOWED_ORIGINS` in `.env` to restrict access

- **Training Time**: ~45-60 minutes3. **Input Validation**: All inputs are validated and sanitized

4. **HTTPS**: Use HTTPS in production (configure reverse proxy)

## Configuration5. **Container Security**: Docker runs as non-root user

6. **Environment Variables**: Never commit `.env` file to git

Edit `.env` or set environment variables:

## Learning Resources

| Variable | Description | Default |

|----------|-------------|---------|This project demonstrates:

| `MODEL_DIR` | Path to trained model | `models/phishguard-model` |- **Data Engineering**: Multi-source data collection and cleaning

| `LOG_LEVEL` | Logging level | `INFO` |- **Fine-tuning**: Transfer learning with transformer models

| `MAX_CONTENT_LENGTH` | Max email content length | `10000` |- **API Development**: Production-ready FastAPI implementation

| `RATE_LIMIT_PER_MINUTE` | API rate limit | `60` |- **Docker Deployment**: Containerization and orchestration

| `API_KEY` | API authentication key | `your-secret-api-key-here` |- **Testing**: Comprehensive test coverage

| `DEBUG` | Enable debug mode | `false` |

| `ALLOWED_ORIGINS` | CORS origins (JSON array) | `["*"]` |For detailed explanations, see:

- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment guide

## Development- [DATA_SOURCES.md](DATA_SOURCES.md) - Dataset information

- [TECHNICAL_EXPLANATION.md](TECHNICAL_EXPLANATION.md) - Technical deep dive

### Project Structure

```## Performance

phishguard/

├── src/- **Inference Time**: 

│   ├── api/  - GPU (CUDA): ~50ms per email

│   │   ├── main.py          # FastAPI application  - CPU: ~200ms per email

│   │   └── schemas.py       # Pydantic models- **Accuracy**: 99.7% on test samples

│   ├── core/- **Memory**: ~500MB for model, ~1GB total with API

│   │   ├── logger.py        # Logging configuration- **Throughput**: Up to 20 requests/second on GPU

│   │   ├── exceptions.py    # Custom exceptions- **GPU Support**: Automatic CUDA detection (RTX 4060 tested)

│   │   └── model_manager.py # Model loading/inference

│   ├── utils/## Monitoring

│   │   └── preprocess.py    # Email preprocessing

│   └── train.py             # Model training scriptThe API provides:

├── scripts/- Health check endpoint (`/health`)

│   ├── prepare_training_data.py  # Clean data pipeline- Structured JSON logging

│   ├── evaluate_model.py         # Comprehensive evaluation- Request/response metrics

│   └── start_api.py              # Production startup- Error tracking

├── models/

│   └── phishguard-model/    # Trained model (500MB)## Contributing

├── data/

│   ├── train.csv            # Training set (32,861 emails)1. Fork the repository

│   ├── validation.csv       # Validation set (4,108 emails)2. Create a feature branch

│   └── test.csv             # Test set (4,108 emails)3. Make your changes

├── tests/                   # Test suite4. Add tests

├── config.py               # Configuration management5. Submit a pull request

├── Dockerfile              # Docker image

├── docker-compose.yml      # Docker Compose## License

└── requirements.txt        # Dependencies

```MIT License - see LICENSE file for details



### Running Tests## Support

```bash

pytest tests/ -vFor issues and questions:

```- Open an issue on [GitHub](https://github.com/sergiugogo/ISM-AIPhishingGuard/issues)

- Check the API documentation at `/docs` (when DEBUG=true)

### Code Quality- See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment help

- Type hints throughout

- Pydantic for validation## Acknowledgments

- Structured logging (JSON format)

- Comprehensive error handling**Datasets:**

- 17+ test cases covering all endpoints- [zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset)

- [SetFit/enron_spam](https://huggingface.co/datasets/SetFit/enron_spam)

## Performance- [PhishTank](https://phishtank.org/) - Verified phishing URLs



- **Inference Time**: **Technologies:**

  - GPU (CUDA): ~50ms per email- [RoBERTa](https://huggingface.co/roberta-base) by Facebook AI

  - CPU: ~200ms per email- [FastAPI](https://fastapi.tiangolo.com/)

- **Accuracy**: 98.81% on clean test set- [Transformers](https://huggingface.co/transformers/) by HuggingFace

- **Memory**: ~500MB for model, ~1GB total with API- [PyTorch](https://pytorch.org/)

- **Throughput**: Up to 20 requests/second on GPU

- **GPU Support**: Automatic CUDA detection## Roadmap



## Security Considerations- [ ] Extended evaluation on larger test sets

- [ ] Additional data sources integration

1. **API Key**: Use strong, random keys in production- [ ] Real-time phishing URL database updates

2. **CORS**: Configure `ALLOWED_ORIGINS` to restrict access- [ ] Email header analysis

3. **Input Validation**: All inputs validated and sanitized- [ ] Multi-language support

4. **HTTPS**: Use HTTPS in production (reverse proxy)- [ ] Performance benchmarking suite

5. **Container Security**: Docker runs as non-root user

6. **Environment Variables**: Never commit `.env` to git


## Monitoring & Observability

The API provides:
- Health check endpoint (`/health`)
- Structured JSON logging
- Request/response tracking
- Error monitoring
- Model version info

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/sergiugogo/ISM-AIPhishingGuard/issues)
- Check API docs at `/docs` (when DEBUG=true)
- See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment help

## Acknowledgments

**Datasets:**
- [zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset)
- [SetFit/enron_spam](https://huggingface.co/datasets/SetFit/enron_spam)

**Technologies:**
- [RoBERTa](https://huggingface.co/roberta-base) by Facebook AI
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Transformers](https://huggingface.co/transformers/) by HuggingFace
- [PyTorch](https://pytorch.org/) - Deep learning framework

## Lessons Learned

1. **Data quality matters most**: Spent 60% of time on data collection, cleaning, and preventing leakage
2. **Evaluation rigor is critical**: Easy to inflate metrics with data leakage
3. **Domain knowledge helps**: Understanding phishing patterns improved feature engineering
4. **Production readiness**: API development, testing, and deployment are as important as the model
5. **Documentation**: Clear docs make the project accessible and maintainable

## Roadmap

- [x] Rigorous data cleaning and deduplication
- [x] Prevent synthetic template leakage
- [x] Comprehensive evaluation with ROC-AUC
- [x] Production-ready API with authentication
- [x] Docker deployment
- [ ] Real-time phishing URL database integration
- [ ] Email header analysis (SPF, DKIM, DMARC)
- [ ] Multi-language support
- [ ] Model quantization for faster inference
- [ ] Continuous learning pipeline

---

**Built with ❤️ as a learning project | Production-ready with research-grade evaluation**
