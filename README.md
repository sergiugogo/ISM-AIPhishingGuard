# 🛡️ PhishGuard
**AI-Powered Phishing Detection API (Fine-tuned RoBERTa)**

---

PhishGuard is a research-grade phishing detection system designed for real-world deployment.  
It delivers:

- ✅ **98.81% accuracy** on a clean, fully held-out test set
- ✅ **Zero data leakage** during evaluation
- ✅ **Real-time detection** via a secure FastAPI service

---

## 🎯 Key Features

- 🤖 **Fine-tuned RoBERTa-base transformer** (125M params)
- 🎯 **High Accuracy**: 98.81% — Recall 99.49%, Precision 97.76%
- 🔐 **Strict anti-leakage dataset pipeline** — honest metrics
- ⚡ **Fast inference**: ~50ms (GPU) / ~200ms (CPU)
- 🔍 **Explainability**: highlight phishing indicators
- 📊 **Additional features**: URL/IP detection, urgency analysis
- 🐳 **Docker-ready** production deployment
- 🧪 **Automated test suite** included

---

## 📈 Model Performance

| Metric      | Score  |
|-------------|--------|
| Accuracy    | 98.81% |
| Precision   | 97.76% |
| Recall      | 99.49% |
| F1 Score    | 98.62% |
| ROC-AUC     | 0.9991 |

- ❌ **False Positives**: 40 / 2,351 benign (1.70% FPR)
- ❌ **False Negatives**: 9 / 1,757 phishing (0.51% FNR)

> The model prioritizes safety → prefers catching phishing over missing attacks ✅

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- (Optional) CUDA-enabled GPU
- (Optional) Docker + Docker Compose

### Installation

```bash
git clone https://github.com/sergiugogo/ISM-AIPhishingGuard.git
cd ISM-AIPhishingGuard

python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### Configure API Key

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

➡ Paste output into `.env` → `API_KEY=...`

### 🧪 Run Evaluation & Train (optional)

```bash
python scripts/prepare_training_data.py
python src/train.py
python scripts/evaluate_model_v2.py
```

### 🌐 Run the API

```bash
python scripts/start_api.py --reload
```

📍 Visit Swagger UI:  
➡ http://localhost:8000/docs

---

## 🔌 API Usage

### Auth Header
```
X-API-Key: YOUR_KEY_HERE
```

### Predict Email Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY_HERE" \
  -d '{"subject": "Urgent: Verify your account", "body": "Click here: http://192.168.1.1/login"}'
```

---

## 📚 Dataset & Methodology

- ✔ Multiple high-quality sources
- ✔ Deduplication before splitting
- ✔ Spam templates removed from eval
- ✔ Clean, held-out test set
- ✔ Balanced phishing/benign in eval

📊 **Final dataset**:

- **Train**: 32,861 emails
- **Validation**: 4,108
- **Test**: 4,108 (fully isolated)

---

## 🧩 Tech Stack

- **RoBERTa-base** (HuggingFace Transformers)
- **PyTorch** for training/inference
- **FastAPI** + Uvicorn
- **Docker** deployment

---

## 🛠 Project Structure

```
phishguard/
├─ src/
│  ├─ api/              # FastAPI application
│  ├─ core/             # Logging/config/model management
│  ├─ utils/            # Preprocessing helpers
│  └─ train.py          # Model training
├─ scripts/             # Data prep / Eval / API runner
├─ data/                # Local datasets (gitignored)
├─ models/              # Trained model weights
├─ tests/               # Automated test suite
└─ docker-compose.yml
```

---

## 📌 Roadmap

- [ ] Multi-language phishing handling
- [ ] Header + SPF/DKIM/DMARC analysis
- [ ] Real-time phishing URL/IOC integration

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

⭐ **If you found this useful, consider starring the repo!**