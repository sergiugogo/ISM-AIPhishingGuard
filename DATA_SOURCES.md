"""
Data Sources for PhishGuard Fine-tuning
========================================

This document lists recommended datasets for training/fine-tuning the phishing detection model.

PUBLIC DATASETS:
---------------

1. Hugging Face Datasets:
   - mlexchange/phishing-email-detection
     URL: https://huggingface.co/datasets/mlexchange/phishing-email-detection
     Size: ~18K emails
     Labels: Binary (phishing/legitimate)
   
   - christophschuhmann/phishing-emails
     URL: https://huggingface.co/datasets/christophschuhmann/phishing-emails
     Size: Varies
   
   - CEAS-08 Phishing Email Corpus
     URL: Available through academic sources
     Size: ~10K emails

2. Kaggle Datasets:
   - Phishing Email Detection Dataset
     URL: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
     Size: ~18K emails
   
   - Email Spam Classification Dataset
     URL: https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv
     Note: Contains spam, but includes phishing examples

3. Academic Repositories:
   - Enron Email Dataset (filtered for phishing)
     URL: https://www.cs.cmu.edu/~enron/
     Size: 500K+ emails (need labeling for phishing)
   
   - Nazario's Phishing Corpus
     URL: https://monkey.org/~jose/phishing/
     Size: Historical phishing emails

4. GitHub Datasets:
   - awesome-phishing-datasets
     URL: https://github.com/topics/phishing-dataset
     Multiple curated collections

COMMERCIAL/RESEARCH ACCESS:
--------------------------

1. PhishTank API
   URL: https://www.phishtank.com/
   - Real-time phishing URL database
   - Free API access (rate-limited)
   - Can correlate with email samples

2. OpenPhish
   URL: https://openphish.com/
   - Community-driven phishing intelligence
   - Premium feeds available

3. Anti-Phishing Working Group (APWG)
   URL: https://apwg.org/
   - Research-focused datasets
   - Requires academic/research affiliation

SYNTHETIC DATA GENERATION:
-------------------------

1. Using GPT/Claude to generate phishing templates
2. Augment existing legitimate emails with phishing patterns
3. Template-based generation with common phishing tactics

DATA COLLECTION METHODS:
------------------------

1. Email Honeypots
   - Set up honeypot email addresses
   - Collect incoming spam/phishing attempts
   - Requires time to accumulate data

2. User Reporting Systems
   - Collect emails reported by users
   - Requires manual labeling/verification

3. Public Email Archives
   - Mailing list archives (with permission)
   - Filter and label for training

MULTILINGUAL DATASETS:
---------------------

1. For international phishing detection:
   - PhishFarm (multiple languages)
   - European phishing datasets
   - Language-specific spam corpora

DATA AUGMENTATION TECHNIQUES:
-----------------------------

1. Back-translation (translate to another language and back)
2. Synonym replacement for non-critical words
3. Paraphrasing using LLMs
4. URL obfuscation variations
5. Combining legitimate email bodies with phishing subjects

RECOMMENDED APPROACH:
--------------------

For best results, combine:
1. Base dataset: Hugging Face phishing-email-detection (quick start)
2. Augmentation: Add Kaggle datasets
3. Real-world data: PhishTank URLs + email samples
4. Synthetic: Generate edge cases with LLMs
5. Validation: Manual review of 10% of data

TARGET SIZE:
-----------
- Minimum: 5,000 emails (2,500 per class)
- Good: 20,000 emails (10,000 per class)
- Excellent: 50,000+ emails (balanced classes)

LABELING QUALITY:
----------------
- Ensure balanced classes (50/50 or 60/40 split)
- Review edge cases manually
- Include diverse phishing types (spear phishing, whaling, etc.)
- Validate with multiple annotators for ambiguous cases

PRIVACY & LEGAL:
---------------
- Remove PII (personal identifiable information)
- Check dataset licenses
- Comply with GDPR/privacy regulations
- Anonymize real email addresses
"""
