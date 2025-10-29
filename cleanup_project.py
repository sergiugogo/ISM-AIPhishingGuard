"""
Clean up PhishGuard project - remove redundant files and checkpoints.
"""
import os
import shutil
from pathlib import Path


def get_folder_size(path):
    """Calculate folder size in MB."""
    total = 0
    for entry in Path(path).rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total / (1024 * 1024)


def main():
    print("="*70)
    print("🧹 PHISHGUARD PROJECT CLEANUP")
    print("="*70)
    
    base_dir = Path("d:/projects/ISM_Showcases/phishguard")
    
    # Items to clean
    cleanup_items = []
    
    # 1. Find checkpoint directories
    print("\n1️⃣ Scanning for training checkpoints...")
    checkpoint_dirs = list(base_dir.rglob("checkpoint-*"))
    if checkpoint_dirs:
        for checkpoint in checkpoint_dirs:
            size = get_folder_size(checkpoint)
            cleanup_items.append({
                'path': checkpoint,
                'type': 'checkpoint',
                'size_mb': size,
                'reason': 'Training checkpoint (no longer needed)'
            })
    
    # 2. Find redundant data files
    print("\n2️⃣ Scanning for redundant data files...")
    data_dir = base_dir / "data"
    if data_dir.exists():
        data_files = {
            'phishing_emails.csv': 'Original small dataset',
            'processed_emails.csv': 'Intermediate processed file',
            'synthetic_emails.csv': 'Old synthetic data',
            'combined_training_data.csv': 'Intermediate combined file',
            'enhanced_training_data.csv': 'Intermediate enhanced file'
        }
        
        for filename, reason in data_files.items():
            filepath = data_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size / (1024 * 1024)
                cleanup_items.append({
                    'path': filepath,
                    'type': 'data',
                    'size_mb': size,
                    'reason': reason
                })
    
    # 3. Find redundant scripts
    print("\n3️⃣ Scanning for redundant scripts...")
    scripts_dir = base_dir / "scripts"
    redundant_scripts = {
        'download_more_data.py': 'Outdated data downloader',
        'train_roberta.py': 'Alternative training script (not using)',
        'preview_datasets.py': 'Testing script',
        'test_model.py': 'Duplicate of evaluate_model.py'
    }
    
    for filename, reason in redundant_scripts.items():
        filepath = scripts_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / (1024 * 1024)
            cleanup_items.append({
                'path': filepath,
                'type': 'script',
                'size_mb': size,
                'reason': reason
            })
    
    # 4. Find redundant documentation
    print("\n4️⃣ Scanning for redundant documentation...")
    redundant_docs = {
        'HUGGINGFACE_DATASETS.md': 'Outdated dataset info',
        'DATA_COLLECTION_GUIDE.md': 'Superseded by DATA_SOURCES.md'
    }
    
    for filename, reason in redundant_docs.items():
        filepath = base_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / (1024 * 1024)
            cleanup_items.append({
                'path': filepath,
                'type': 'docs',
                'size_mb': size,
                'reason': reason
            })
    
    # Show cleanup plan
    if not cleanup_items:
        print("\n✨ Project is already clean!")
        return
    
    print("\n" + "="*70)
    print("📋 CLEANUP PLAN")
    print("="*70)
    
    total_size = sum(item['size_mb'] for item in cleanup_items)
    
    by_type = {}
    for item in cleanup_items:
        item_type = item['type']
        if item_type not in by_type:
            by_type[item_type] = []
        by_type[item_type].append(item)
    
    for item_type, items in by_type.items():
        type_size = sum(i['size_mb'] for i in items)
        print(f"\n📁 {item_type.upper()} ({len(items)} items, {type_size:.2f} MB)")
        for item in items:
            print(f"   ├─ {item['path'].name} ({item['size_mb']:.2f} MB)")
            print(f"   │  {item['reason']}")
    
    print(f"\n💾 Total space to free: {total_size:.2f} MB")
    
    # Confirm deletion
    print("\n" + "="*70)
    response = input("🗑️  Delete these files? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("\n❌ Cleanup cancelled")
        return
    
    # Perform cleanup
    print("\n🗑️  Deleting files...")
    deleted_count = 0
    freed_space = 0
    
    for item in cleanup_items:
        try:
            if item['path'].is_dir():
                shutil.rmtree(item['path'])
            else:
                item['path'].unlink()
            
            print(f"   ✅ Deleted: {item['path'].name}")
            deleted_count += 1
            freed_space += item['size_mb']
        
        except Exception as e:
            print(f"   ❌ Failed to delete {item['path'].name}: {e}")
    
    print("\n" + "="*70)
    print("✅ CLEANUP COMPLETE")
    print("="*70)
    print(f"\n📊 Results:")
    print(f"   Files deleted: {deleted_count}/{len(cleanup_items)}")
    print(f"   Space freed: {freed_space:.2f} MB")
    
    # What to keep
    print("\n" + "="*70)
    print("📦 KEEPING (Core Project Files)")
    print("="*70)
    print("\n📂 Data:")
    print("   ✅ large_training_data.csv - Main training dataset (29,256 emails)")
    print("\n📂 Model:")
    print("   ✅ models/phishguard-model/ - Fine-tuned DistilRoBERTa (330MB)")
    print("\n📂 Scripts:")
    print("   ✅ test_datasets.py - Verify HuggingFace datasets")
    print("   ✅ evaluate_model.py - Evaluate model performance")
    print("   ✅ generate_large_dataset.py - Generate synthetic data if needed")
    print("\n📂 Source:")
    print("   ✅ src/train.py - Training script")
    print("   ✅ src/api/main.py - Production API")
    print("   ✅ src/utils/ - Preprocessing utilities")
    print("\n📂 Config:")
    print("   ✅ requirements.txt, Dockerfile, docker-compose.yml")
    print("   ✅ README.md, TECHNICAL_EXPLANATION.md")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
