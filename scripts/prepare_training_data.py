"""
Download and prepare training data from HuggingFace.

Combines multiple datasets:
- zefang-liu/phishing-email-dataset (18,650 emails)
- SetFit/enron_spam (31,716 emails)

Total: ~50,000 emails
Splits: 80% training, 20% validation (kept separate for evaluation)
"""
import pandas as pd
from datasets import load_dataset
from pathlib import Path
import sys
import requests
import random


TRAIN_FILE = Path("data/train.csv")
VAL_FILE = Path("data/validation.csv")

# Multiple datasets for more data
DATASETS = [
    {
        'name': 'zefang-liu/phishing-email-dataset',
        'text_col': 'Email Text',
        'label_col': 'Email Type',
        'phishing_value': 'Phishing Email'
    },
    {
        'name': 'SetFit/enron_spam',
        'text_col': 'message',
        'subject_col': 'subject',
        'label_col': 'label',
        'phishing_value': 1  # 1 = spam, 0 = ham
    }
]

# Add PhishTank URLs as phishing samples
ADD_PHISHTANK_URLS = True
PHISHTANK_URL = "http://data.phishtank.com/data/online-valid.csv"


def download_from_huggingface():
    """Download multiple datasets from HuggingFace."""
    print("="*70)
    print(" DOWNLOADING FROM HUGGINGFACE")
    print("="*70)
    
    all_emails = []
    
    for dataset_info in DATASETS:
        dataset_name = dataset_info['name']
        print(f"\n Dataset: {dataset_name}")
        print("   Loading...")
        
        try:
            # Load full dataset
            dataset = load_dataset(dataset_name, split="train")
            df = dataset.to_pandas()
            
            print(f" Downloaded {len(df):,} emails")
            print(f"   Columns: {list(df.columns)}")
            
            # Extract emails from this dataset
            for idx, row in df.iterrows():
                # Get text content
                if 'subject_col' in dataset_info:
                    subject = str(row.get(dataset_info['subject_col'], ''))
                    body = str(row.get(dataset_info['text_col'], ''))
                else:
                    # Split email text into subject and body
                    text = str(row.get(dataset_info['text_col'], ''))
                    if '\n' in text:
                        parts = text.split('\n', 1)
                        subject = parts[0].strip()
                        body = parts[1].strip() if len(parts) > 1 else text
                    else:
                        subject = ''
                        body = text
                
                # Determine label
                label_value = row.get(dataset_info['label_col'])
                phishing_value = dataset_info['phishing_value']
                
                if isinstance(phishing_value, str):
                    is_phishing = phishing_value in str(label_value)
                else:
                    is_phishing = (label_value == phishing_value)
                
                label_text = 'phishing' if is_phishing else 'benign'
                
                all_emails.append({
                    'subject': subject,
                    'body': body,
                    'label': label_text,
                    'source': dataset_name
                })
            
            print(f"   Extracted {len(all_emails)} total emails so far")
        
        except Exception as e:
            print(f"\n Error downloading {dataset_name}: {e}")
            print(f"   Skipping this dataset...")
            continue
    
    if not all_emails:
        print("\n Failed to download any datasets!")
        sys.exit(1)
    
    df = pd.DataFrame(all_emails)
    
    print(f"\n" + "="*70)
    print(f" Downloaded total: {len(df):,} emails from {len(DATASETS)} datasets")
    print(f"\n Combined dataset info:")
    print(f"   Total emails: {len(df):,}")
    print(f"\n   By source:")
    print(df['source'].value_counts())
    print(f"\n   By label:")
    print(df['label'].value_counts())
    
    return df


def download_phishtank_urls():
    """Download PhishTank verified phishing URLs and create phishing emails."""
    if not ADD_PHISHTANK_URLS:
        return []
    
    print("\n" + "="*70)
    print(" DOWNLOADING PHISHTANK PHISHING URLS")
    print("="*70)
    
    try:
        print(f"\n Downloading from: {PHISHTANK_URL}")
        response = requests.get(PHISHTANK_URL, timeout=30)
        response.raise_for_status()
        
        # Save to temp file
        temp_file = Path("data/temp_phishtank.csv")
        temp_file.write_bytes(response.content)
        
        # Load CSV
        df = pd.read_csv(temp_file)
        print(f" Downloaded {len(df):,} verified phishing URLs")
        
        # Create phishing emails from URLs
        print(f"\n Creating phishing emails from URLs...")
        
        phishing_templates = [
            "URGENT: Your account requires immediate verification. Click here: {url}",
            "Security Alert: Unusual activity detected. Verify your identity: {url}",
            "Action Required: Update your payment information at: {url}",
            "Your package delivery failed. Reschedule here: {url}",
            "Congratulations! You've won. Claim your prize: {url}",
            "Invoice attached. View and pay here: {url}",
            "Your account will be suspended. Prevent this by clicking: {url}",
            "Password reset requested. If this wasn't you, click here: {url}",
            "Limited time offer! Don't miss out: {url}",
            "Your payment was declined. Update your details: {url}",
        ]
        
        subject_templates = [
            "URGENT: Account Verification Required",
            "Security Alert - Action Needed",
            "Payment Update Required",
            "Package Delivery Failed",
            "You've Won!",
            "Invoice #{rand}",
            "Account Suspension Warning",
            "Password Reset Request",
            "Exclusive Offer Inside",
            "Payment Failed",
        ]
        
        emails = []
        # Limit to 5000 URLs to avoid overwhelming the dataset
        sample_size = min(5000, len(df))
        sampled_urls = df['url'].head(sample_size).tolist()
        
        for url in sampled_urls:
            # Pick random template
            body_template = random.choice(phishing_templates)
            subject_template = random.choice(subject_templates)
            
            # Replace placeholders
            body = body_template.format(url=url)
            subject = subject_template.replace('{rand}', str(random.randint(1000, 9999)))
            
            emails.append({
                'subject': subject,
                'body': body,
                'label': 'phishing',
                'source': 'PhishTank'
            })
        
        print(f" Created {len(emails):,} phishing emails from PhishTank URLs")
        
        # Cleanup
        temp_file.unlink()
        
        return emails
    
    except Exception as e:
        print(f"  Failed to download PhishTank data: {e}")
        print(f"   Continuing with other datasets...")
        return []


def prepare_for_training(df):
    """Convert to PhishGuard format and split into train/val."""
    print("\n" + "="*70)
    print(" PREPARING DATA FOR TRAINING")
    print("="*70)
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['subject', 'body'])
    removed = initial_count - len(df)
    if removed > 0:
        print(f"\n  Removed {removed:,} duplicate emails")
    
    # Remove empty entries
    df = df[(df['subject'].str.len() > 0) | (df['body'].str.len() > 0)]
    
    # Show statistics
    print(f"\n Final combined dataset:")
    print(f"   Total emails: {len(df):,}")
    print(f"   Phishing: {len(df[df['label'] == 'phishing']):,}")
    print(f"   Benign: {len(df[df['label'] == 'benign']):,}")
    
    # Check if balanced
    phishing_count = len(df[df['label'] == 'phishing'])
    benign_count = len(df[df['label'] == 'benign'])
    ratio = max(phishing_count, benign_count) / min(phishing_count, benign_count) if min(phishing_count, benign_count) > 0 else 0
    
    if ratio > 2.0:
        print(f"\n   Dataset is imbalanced (ratio: {ratio:.2f}:1)")
        print(f"   Consider balancing - using all data for now...")
    else:
        print(f"\n   ✅ Dataset is reasonably balanced (ratio: {ratio:.2f}:1)")
    
    # Shuffle dataset before splitting
    print(f"\n Shuffling dataset...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split 80/20 train/validation
    print(f"\n  Splitting dataset (80% train, 20% validation)...")
    total_size = len(df)
    train_size = int(0.8 * total_size)
    
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    # Drop source column before saving
    train_df = train_df.drop(columns=['source'])
    val_df = val_df.drop(columns=['source'])
    
    print(f"   Training set: {len(train_df):,} emails")
    print(f"   Validation set: {len(val_df):,} emails")
    print(f"\n   Training labels:")
    print(f"   {train_df['label'].value_counts()}")
    print(f"\n   Validation labels:")
    print(f"   {val_df['label'].value_counts()}")
    
    return train_df, val_df


def save_datasets(train_df, val_df):
    """Save train and validation sets to separate CSV files."""
    print("\n" + "="*70)
    print(" SAVING DATASETS")
    print("="*70)
    
    # Ensure data directory exists
    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Save training set
    train_df.to_csv(TRAIN_FILE, index=False, encoding='utf-8')
    train_size = TRAIN_FILE.stat().st_size / (1024 * 1024)
    print(f"\n Training set saved to: {TRAIN_FILE.absolute()}")
    print(f"   Size: {train_size:.2f} MB")
    
    # Save validation set
    val_df.to_csv(VAL_FILE, index=False, encoding='utf-8')
    val_size = VAL_FILE.stat().st_size / (1024 * 1024)
    print(f"\n Validation set saved to: {VAL_FILE.absolute()}")
    print(f"   Size: {val_size:.2f} MB")
    
    # Preview
    print(f"\n Training set preview:")
    print("="*70)
    for i, row in train_df.head(2).iterrows():
        print(f"\n{i+1}. Label: {row['label']}")
        print(f"   Subject: {row['subject'][:60]}...")
        print(f"   Body: {row['body'][:100]}...")


def main():
    """Main workflow."""
    print("\n" + "="*70)
    print(" PHISHGUARD DATA PREPARATION")
    print("="*70)
    print(f"\nThis will download real phishing/spam emails from HuggingFace")
    print(f"and prepare them for training.\n")
    
    # Download from HuggingFace
    df = download_from_huggingface()
    
    # Add PhishTank URLs
    phishtank_emails = download_phishtank_urls()
    if phishtank_emails:
        phishtank_df = pd.DataFrame(phishtank_emails)
        df = pd.concat([df, phishtank_df], ignore_index=True)
        print(f"\n Combined total: {len(df):,} emails")
    
    # Prepare and split
    train_df, val_df = prepare_for_training(df)
    
    # Save
    save_datasets(train_df, val_df)
    
    # Summary
    print("\n" + "="*70)
    print(" DATA PREPARATION COMPLETE")
    print("="*70)
    print(f"\n Files created:")
    print(f"     Training: {TRAIN_FILE.absolute()}")
    print(f"    Validation: {VAL_FILE.absolute()}")
    print(f"\n  IMPORTANT: Validation set is NEVER used in training!")
    print(f"   It's kept separate for honest evaluation.")
    print(f"\n Next steps:")
    print(f"   1. Train model: python src/train.py")
    print(f"   2. Evaluate: python scripts/evaluate_model.py")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
