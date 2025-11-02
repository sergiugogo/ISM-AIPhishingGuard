import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, classification_report
)

print('='*70)
print('PHISHGUARD MODEL EVALUATION')
print('='*70)

# Configuration
MODEL_DIR = 'models/phishguard-model'
TEST_FILE = 'data/test.csv'

print(f"\n Loading test set: {TEST_FILE}")
test_df = pd.read_csv(TEST_FILE)
print(f'   Total samples: {len(test_df):,}')
print(f'\n   Label distribution:')
print(test_df['label'].value_counts())

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n Loading model from: {MODEL_DIR}')
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()
print(f'   Model loaded on {device}')

# Batch prediction for speed
print('\n Running predictions...')
batch_size = 32
predictions = []
probabilities = []

for i in range(0, len(test_df), batch_size):
    batch = test_df.iloc[i:i+batch_size]
    texts = [f'{row["subject"]} . {row["body"]}' for _, row in batch.iterrows()]
    
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, max_length=256, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        predictions.extend(preds.cpu().numpy())
        probabilities.extend(probs[:, 1].cpu().numpy())  # Probability of phishing class
    
    if (i + batch_size) % 500 == 0 or i + batch_size >= len(test_df):
        print(f'   Progress: {min(i + batch_size, len(test_df))}/{len(test_df)}')

# Convert to numpy arrays for safety
predictions = np.array(predictions)
probabilities = np.array(probabilities)

# Prepare true labels
true_labels = test_df['label'].tolist()
y_true = np.array([1 if l == 'phishing' else 0 for l in true_labels])
y_pred = predictions

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
roc_auc = roc_auc_score(y_true, probabilities)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Per-class metrics
precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
    y_true, y_pred, average=None, labels=[0, 1]
)

print('\n' + '='*70)
print(' OVERALL METRICS')
print('='*70)
print(f'\n Accuracy: {accuracy:.2%}')
print(f'   Precision: {precision:.2%}')
print(f'   Recall: {recall:.2%}')
print(f'   F1 Score: {f1:.2%}')
print(f'   ROC-AUC: {roc_auc:.4f}')

print('\n' + '='*70)
print(' CONFUSION MATRIX')
print('='*70)
print(f'\n                Predicted')
print(f'                Benign  Phishing')
print(f'   Actual Benign   {tn:5d}   {fp:5d}')
print(f'          Phishing {fn:5d}   {tp:5d}')

print('\n' + '='*70)
print(' PER-CLASS METRICS')
print('='*70)
print(f'\n   BENIGN:')
print(f'      Precision: {precision_per_class[0]:.2%}')
print(f'      Recall: {recall_per_class[0]:.2%}')
print(f'      F1: {f1_per_class[0]:.2%}')
print(f'      Support: {support[0]:,} samples')

print(f'\n   PHISHING:')
print(f'      Precision: {precision_per_class[1]:.2%}')
print(f'      Recall: {recall_per_class[1]:.2%}')
print(f'      F1: {f1_per_class[1]:.2%}')
print(f'      Support: {support[1]:,} samples')

# Misclassification analysis
print('\n' + '='*70)
print('  MISCLASSIFICATION ANALYSIS')
print('='*70)

# False Positives (predicted phishing, actually benign)
fp_indices = np.where((y_pred == 1) & (y_true == 0))[0]
print(f'\n False Positives: {len(fp_indices)} ({len(fp_indices)/len(y_true)*100:.2f}%)')
print(f'   (Benign emails incorrectly flagged as phishing)')

if len(fp_indices) > 0:
    print(f'\n   Top {min(5, len(fp_indices))} False Positives:')
    # Sort by confidence (highest confidence mistakes are most concerning)
    fp_confidences = probabilities[fp_indices]
    top_fp_idx = fp_indices[np.argsort(fp_confidences)[::-1][:min(5, len(fp_indices))]]
    
    for i, idx in enumerate(top_fp_idx, 1):
        subject = str(test_df.iloc[idx]["subject"])[:70]
        body = str(test_df.iloc[idx]["body"])[:100]
        print(f'\n   {i}. Subject: {subject}...')
        print(f'      Body: {body}...')
        print(f'      Confidence: {probabilities[idx]:.2%}')

# False Negatives (predicted benign, actually phishing)
fn_indices = np.where((y_pred == 0) & (y_true == 1))[0]
print(f'\n False Negatives: {len(fn_indices)} ({len(fn_indices)/len(y_true)*100:.2f}%)')
print(f'   (Phishing emails missed by the model)')

if len(fn_indices) > 0:
    print(f'\n   Top {min(5, len(fn_indices))} False Negatives:')
    # Sort by confidence (lowest confidence = most certain it was benign)
    fn_confidences = probabilities[fn_indices]
    top_fn_idx = fn_indices[np.argsort(fn_confidences)[:min(5, len(fn_indices))]]
    
    for i, idx in enumerate(top_fn_idx, 1):
        subject = str(test_df.iloc[idx]["subject"])[:70]
        body = str(test_df.iloc[idx]["body"])[:100]
        print(f'\n   {i}. Subject: {subject}...')
        print(f'      Body: {body}...')
        print(f'      Confidence: {probabilities[idx]:.2%} (predicted benign)')

print('\n' + '='*70)
print(' INSIGHTS')
print('='*70)

# Calculate false positive rate on benign emails
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f'\n   False Positive Rate: {fpr:.2%}')
print(f'   (Out of {tn + fp:,} benign emails, {fp} were incorrectly flagged)')

# Calculate false negative rate on phishing emails
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
print(f'\n   False Negative Rate: {fnr:.2%}')
print(f'   (Out of {fn + tp:,} phishing emails, {fn} were missed)')

# Check for bias
if fp > fn * 2:
    print(f'\n     Model appears biased toward flagging emails as phishing')
    print(f'      (More false positives than false negatives)')
elif fn > fp * 2:
    print(f'\n     Model appears conservative (may miss real phishing)')
    print(f'      (More false negatives than false positives)')
else:
    print(f'\n    Model has balanced error distribution')

print('\n' + '='*70)
