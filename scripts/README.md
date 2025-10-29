# Scripts Directory

Utility scripts for PhishGuard data collection and verification.

## Quick Dataset Testing

### Test Which Datasets Work
```powershell
python scripts/test_datasets.py
```

This will:
- ✅ Test connection to recommended datasets
- ✅ Show which ones are accessible
- ✅ Display column information
- ✅ Provide recommendations

**Takes:** ~30 seconds

---

### Preview Dataset Details
```powershell
python scripts/preview_datasets.py
```

This will:
- ✅ Load sample data from each dataset
- ✅ Show label distribution
- ✅ Display sample emails
- ✅ Check for quality issues
- ✅ Compare datasets side-by-side

**Takes:** ~2-3 minutes (interactive)

---

## Recommended Workflow

1. **Quick Test** (30 seconds):
   ```powershell
   python scripts/test_datasets.py
   ```

2. **If successful**, preview details:
   ```powershell
   python scripts/preview_datasets.py
   ```

3. **If satisfied**, download full dataset:
   ```powershell
   python src/utils/data_collector.py
   ```

---

## Scripts Overview

| Script | Purpose | Time | Output |
|--------|---------|------|--------|
| `test_datasets.py` | Quick connectivity test | 30s | Which datasets work |
| `preview_datasets.py` | Detailed inspection | 2-3min | Sample data + quality report |
