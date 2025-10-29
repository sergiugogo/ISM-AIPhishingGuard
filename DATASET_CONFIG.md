# 📊 PhishGuard Final Dataset Configuration

**Date:** October 28, 2025  
**Status:** Ready to download and train

---

## ✅ Working HuggingFace Datasets

### 1. zefang-liu/phishing-email-dataset
- **Size:** 18,650 emails
- **Phishing:** 7,328 emails
- **Safe:** 11,322 emails
- **Quality:** Real phishing emails
- **Status:** ✅ Working

### 2. SetFit/enron_spam
- **Size:** 31,716 emails
- **Spam:** ~15,858 emails  
- **Ham:** ~15,858 emails
- **Quality:** Real Enron business emails + spam
- **Status:** ✅ Working

---

## 📈 Combined Dataset

**Total Emails:** ~50,366 emails  
**After deduplication:** ~45,000-48,000 emails (estimated)

**Split:**
- **Training:** 80% (~36,000-38,000 emails)
- **Validation:** 20% (~9,000-10,000 emails)

---

## ❌ Datasets That Don't Work

| Dataset | Issue |
|---------|-------|
| ealvaradob/phishing-dataset | Dataset scripts deprecated, trust_remote_code not supported |
| talby/phishing_email | Not found |
| swaggy/Phishing-email-detection | Not found |
| jonaschn/phishing_email | Not found |
| utkarshchugh/email-spam | Not found |

---

## 🚀 Ready to Run

```bash
# Step 1: Download and combine both datasets (50K+ emails)
python scripts/prepare_training_data.py

# Step 2: Train RoBERTa-base on combined data
python src/train.py

# Step 3: Evaluate on validation set
python scripts/evaluate_model.py
```

---

## 📊 Expected Results

With **~40,000 training emails** and **RoBERTa-base**:
- **Training time:** ~45-60 minutes on RTX 4060
- **Expected accuracy:** 95-99%+ on validation
- **Model size:** ~500 MB

---

## 🎯 Benefits of Combined Dataset

1. **More data = Better generalization**
   - 50K emails vs 18K single dataset
   - Diverse sources (phishing corpus + spam corpus)

2. **Balanced distribution**
   - Both datasets are reasonably balanced
   - ~50/50 phishing/benign split overall

3. **Real-world variety**
   - Business emails (Enron)
   - Phishing emails (various types)
   - Spam/scam emails

4. **Proper validation**
   - 10K unseen validation emails
   - True test of model performance

---

## ⚙️ Configuration

**Model:** roberta-base (125M parameters)  
**Datasets:** 2 sources combined  
**Training:** SFT (full fine-tuning)  
**Epochs:** 5  
**Batch size:** 8 (GPU) / 2 (CPU)  
**Hardware:** RTX 4060 or any 8GB+ GPU

---

**Ready to proceed with training! 🎉**

