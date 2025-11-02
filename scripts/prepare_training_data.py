import pandas as pd
import re
from sklearn.model_selection import train_test_split

print("=" * 70)
print("PHISHGUARD – CLEAN DATA SPLITTING (STRICT PHISHING EVALUATION)")
print("=" * 70)

INPUT_FILE = "data/train.csv"  # Original merged real dataset
OUTPUT_DIR = "data/"

print("\nLoading dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"Total samples: {len(df):,}")

# Normalize text
df["subject"] = df["subject"].fillna("").astype(str).str.strip().str.lower()
df["body"] = df["body"].fillna("").astype(str).str.strip().str.lower()

# -------------------------------------------------------------------
# 1️⃣ REMOVE SYNTHETIC & TEMPLATE PHISHING FROM EVAL SPLITS
# -------------------------------------------------------------------
synthetic_phish_patterns = [
    r"click here:", r"verify( your)? account",
    r"security alert", r"payment (failed|declined|required)",
    r"congratulations!? you'?ve won",
    r"package delivery failed", r"limited time offer",
]
synthetic_regex = re.compile("|".join(synthetic_phish_patterns), flags=re.IGNORECASE)

df["is_synthetic_pattern"] = df["body"].str.contains(synthetic_regex, na=False)

# -------------------------------------------------------------------
# 2️⃣ REMOVE GENERIC SPAM FROM EVAL SPLITS
# -------------------------------------------------------------------
spam_patterns = [
    r"viagra", r"cialis", r"porn", r"xxx",
    r"search engine", r"loan", r"mortgage", r"casino",
    r"investment opportunity", r"make money", r"lose weight",
]
spam_regex = re.compile("|".join(spam_patterns), flags=re.IGNORECASE)

df["is_generic_spam"] = df["body"].str.contains(spam_regex, na=False)

# -------------------------------------------------------------------
# 3️⃣ CLASSIFY TRUE PHISHING FOR EVALUATION SETS
# -------------------------------------------------------------------
df["is_eval_eligible"] = ~(df["is_synthetic_pattern"] | df["is_generic_spam"])

print("\nLABEL COUNTS BEFORE FILTERING:")
print(df["label"].value_counts())

print("\nEval-eligible samples:")
print(df[df["is_eval_eligible"]]["label"].value_counts())

# -------------------------------------------------------------------
# 4️⃣ DEDUPLICATE SUBJECT+BODY ACROSS SPLITS
# -------------------------------------------------------------------
df = df.drop_duplicates(subset=["subject", "body"])  # exact dup removal

# -------------------------------------------------------------------
# 5️⃣ BUILD CLEAN EVAL POOL AND SAMPLE VAL/TEST FROM IT
#    - Ensure validation and test contain only eval-eligible (no synthetic/templates/generic spam)
#    - Ensure both phishing and benign are present in each split (fall back to best-effort)
# -------------------------------------------------------------------

# Desired sizes (fractions of original dataset)
total_n = len(df)
val_n = int(round(0.10 * total_n))
test_n = int(round(0.10 * total_n))

print(f"\nTotal samples: {total_n:,}; target VAL: {val_n:,}, TEST: {test_n:,}")

# Eval pool: only non-synthetic, non-generic-spam
eval_pool = df[df["is_eval_eligible"]].copy()
print(f"Eval-eligible pool size: {len(eval_pool):,}")

if len(eval_pool) < (val_n + test_n):
    print("\n  Not enough eval-eligible samples to fill VAL+TEST targets.")
    print("   Will allocate as many as possible from eval pool and fill remaining with non-synthetic benign samples.")

# Helper to safely sample with stratification when possible
def safe_sample(df_pool, n, stratify_col='label'):
    if n <= 0 or len(df_pool) == 0:
        return df_pool.iloc[0:0].copy(), df_pool.copy()
    n = min(n, len(df_pool))
    # If multiple classes and enough samples, use stratified split
    try:
        if df_pool[estrat := stratify_col].nunique() > 1:
            # Use train_test_split to get a sampled set of size n
            sampled, rest = train_test_split(df_pool, train_size=n, stratify=df_pool[estrat], random_state=42)
            return sampled, rest
        else:
            sampled = df_pool.sample(n=n, random_state=42)
            return sampled, df_pool.drop(sampled.index)
    except Exception:
        sampled = df_pool.sample(n=n, random_state=42)
        return sampled, df_pool.drop(sampled.index)

# First, sample VAL from eval_pool
val_df, remaining = safe_sample(eval_pool, val_n)
# Then, sample TEST from remaining eval_pool
test_df, remaining = safe_sample(remaining, test_n)

# If we still don't have enough for val/test, try to fill from df (prefer non-synthetic benign)
needed_val = val_n - len(val_df)
needed_test = test_n - len(test_df)
if needed_val > 0 or needed_test > 0:
    filler_pool = df[~df["is_synthetic_pattern"] & ~df["is_generic_spam"] & ~df.index.isin(val_df.index) & ~df.index.isin(test_df.index)].copy()
    if len(filler_pool) > 0:
        if needed_val > 0:
            add_val = filler_pool.sample(n=min(needed_val, len(filler_pool)), random_state=42)
            val_df = pd.concat([val_df, add_val])
            filler_pool = filler_pool.drop(add_val.index)
        if needed_test > 0 and len(filler_pool) > 0:
            add_test = filler_pool.sample(n=min(needed_test, len(filler_pool)), random_state=43)
            test_df = pd.concat([test_df, add_test])

# Finally, make TRAIN the remainder of the original df excluding val/test
train_df = df[~df.index.isin(val_df.index) & ~df.index.isin(test_df.index)].copy()

# Recompute counts
print(f"\nFinal split counts -> TRAIN: {len(train_df):,}, VAL: {len(val_df):,}, TEST: {len(test_df):,}")

print("\nFinal Label Distribution:")
print("\nTRAIN:")
print(train_df["label"].value_counts())
print("\nVALIDATION:")
print(val_df["label"].value_counts())
print("\nTEST:")
print(test_df["label"].value_counts())

# -------------------------------------------------------------------
# 6️⃣ SAVE OUTPUT
# -------------------------------------------------------------------
train_df[["subject", "body", "label"]].to_csv(OUTPUT_DIR + "train_clean.csv", index=False)
val_df[["subject", "body", "label"]].to_csv(OUTPUT_DIR + "val_clean.csv", index=False)
test_df[["subject", "body", "label"]].to_csv(OUTPUT_DIR + "test_clean.csv", index=False)

print("\nSaved:")
print(f" - {OUTPUT_DIR}train_clean.csv ({len(train_df):,})")
print(f" - {OUTPUT_DIR}val_clean.csv ({len(val_df):,})")
print(f" - {OUTPUT_DIR}test_clean.csv ({len(test_df):,})")
print("\n DONE — Synthetic & spam removed from eval sets (val/test include real benign & phishing)")
print("=" * 70)
