import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

print('='*70)
print('PHISHGUARD MODEL EVALUATION')
print('='*70)

val_df = pd.read_csv('data/validation.csv')
print(f'\nLoaded {len(val_df):,} validation emails')

test_df = val_df.sample(n=1000, random_state=42).reset_index(drop=True)
print(f'Testing on {len(test_df):,} samples\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('models/phishguard-model')
model = AutoModelForSequenceClassification.from_pretrained('models/phishguard-model').to(device)
model.eval()
print(f'Model loaded on {device}\n')

print('Running predictions...')
predictions = []
for idx, row in test_df.iterrows():
    text = f'{row["subject"]} . {row["body"]}'
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_idx = torch.argmax(outputs.logits, dim=1).item()
        predictions.append('phishing' if pred_idx == 1 else 'benign')
    if (idx + 1) % 200 == 0:
        print(f'  Progress: {idx + 1}/{len(test_df)}')

true_labels = test_df['label'].tolist()
y_true = [1 if l == 'phishing' else 0 for l in true_labels]
y_pred = [1 if p == 'phishing' else 0 for p in predictions]

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print('\n' + '='*70)
print('RESULTS')
print('='*70)
print(f'\nAccuracy: {accuracy:.2%}')
print(f'Precision: {precision:.2%}')
print(f'Recall: {recall:.2%}')
print(f'F1 Score: {f1:.2%}')
print(f'\nConfusion Matrix:')
print(f'  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')
print('='*70)
