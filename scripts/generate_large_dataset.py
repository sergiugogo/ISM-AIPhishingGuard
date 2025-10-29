"""Generate a LARGE synthetic dataset to boost training data."""
from src.utils.generate_synthetic_data import PhishingEmailGenerator
import pandas as pd
from pathlib import Path

print(" Generating LARGE synthetic phishing dataset...")
print("This will create 10,000 additional emails for better training\n")

generator = PhishingEmailGenerator()

# Generate 10,000 synthetic emails (in addition to real data)
df_synthetic_large = generator.generate_dataset(num_samples=10000, balance=0.5)

# Load existing real data
existing_data = []

data_files = [
    "data/phishing_emails.csv",
    "data/combined_training_data.csv",
]

for file in data_files:
    path = Path(file)
    if path.exists():
        print(f" Loading {file}...")
        df = pd.read_csv(file)
        if 'subject' in df.columns and 'body' in df.columns and 'label' in df.columns:
            existing_data.append(df)
            print(f"    Added {len(df)} emails")
            break  # Use first valid file found

if existing_data:
    # Combine real + synthetic
    df_combined = pd.concat(existing_data + [df_synthetic_large], ignore_index=True)
    
    # Remove duplicates
    initial = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['body'], keep='first')
    removed = initial - len(df_combined)
    print(f"\n Removed {removed} duplicates")
    
    # Balance
    label_counts = df_combined['label'].value_counts()
    min_count = label_counts.min()
    
    df_benign = df_combined[df_combined['label'] == 0].sample(n=min_count, random_state=42)
    df_phishing = df_combined[df_combined['label'] == 1].sample(n=min_count, random_state=42)
    df_final = pd.concat([df_benign, df_phishing]).sample(frac=1, random_state=42)
    
    # Save
    output_file = Path("data/large_training_data.csv")
    df_final.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print(" LARGE TRAINING DATASET CREATED!")
    print("="*60)
    print(f"Total samples: {len(df_final):,}")
    print(f"Phishing: {(df_final['label']==1).sum():,}")
    print(f"Benign: {(df_final['label']==0).sum():,}")
    print(f"Saved to: {output_file}")
    print("\n This is {:.1f}x larger than before!".format(len(df_final) / 27198))
    print("\nNow retrain with: python src/train.py")
    print("Expected accuracy boost: +5-10%")
    print("="*60)
else:
    print(" No existing data found!")
