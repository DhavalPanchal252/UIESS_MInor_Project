import os
import shutil
import argparse
import random
from pathlib import Path

def setup_dataset(raw_dir, ref_dir, target_root, train_ratio=0.9):
    target_root = Path(target_root)
    
    # Define directories
    # Note: datasets.py REQUIRES 'trainB_label' and 'testB_label' even if README omits them.
    dirs = {
        'trainA': target_root / 'trainA',       # Real Underwater (UIEB Raw)
        'trainB': target_root / 'trainB',       # Synthetic Underwater (Placeholder: UIEB Ref)
        'trainB_label': target_root / 'trainB_label', # Clean Ground Truth (UIEB Ref)
        'testA': target_root / 'testA',         # Real Test (UIEB Raw)
        'testB': target_root / 'testB',         # Syn Test (Placeholder: UIEB Ref)
        'testB_label': target_root / 'testB_label'    # Clean Test GT (UIEB Ref)
    }
    
    # Create directories
    for d in dirs.values():
        if d.exists():
            print(f"Cleaning up {d}...")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        
    # Get all images
    raw_files = sorted([f for f in Path(raw_dir).iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
    ref_files = sorted([f for f in Path(ref_dir).iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
    
    # Filter to matching pairs (assuming filenames match or correspond)
    # UIEB raw: 1.png, ref: 1.png
    raw_names = {f.name: f for f in raw_files}
    ref_names = {f.name: f for f in ref_files}
    
    common_names = sorted(list(set(raw_names.keys()) & set(ref_names.keys())))
    
    print(f"Found {len(common_names)} paired images.")
    
    # Shuffle and split
    random.seed(123)
    random.shuffle(common_names)
    
    split_idx = int(len(common_names) * train_ratio)
    train_names = common_names[:split_idx]
    test_names = common_names[split_idx:]
    
    print(f"Copying {len(train_names)} pairs to train and {len(test_names)} to test...")

    # Copy files
    for name in train_names:
        raw_path = raw_names[name]
        ref_path = ref_names[name]
        
        # Train A: Real Raw
        shutil.copy2(raw_path, dirs['trainA'] / name)
        
        # Train B: Synthetic (Placeholder -> Using Ref)
        # Ideally this should be synthetic underwater images.
        shutil.copy2(ref_path, dirs['trainB'] / name)
        
        # Train B Label: Clean Reference
        shutil.copy2(ref_path, dirs['trainB_label'] / name)
        
    for name in test_names:
        raw_path = raw_names[name]
        ref_path = ref_names[name]
        
        shutil.copy2(raw_path, dirs['testA'] / name)
        shutil.copy2(ref_path, dirs['testB'] / name)
        shutil.copy2(ref_path, dirs['testB_label'] / name)
        
    print("Dataset setup complete.")
    print("NOTE: 'trainB' and 'testB' are populated with Reference images as placeholders for Synthetic data.")
    print("      'trainB_label' and 'testB_label' contain the Reference (Clean) images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize UIEB dataset for UIESS")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to UIEB Raw images folder")
    parser.add_argument("--ref_dir", type=str, required=True, help="Path to UIEB Reference images folder")
    parser.add_argument("--target_root", type=str, default="data", help="Target data directory")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.raw_dir) or not os.path.isdir(args.ref_dir):
        print(f"Error: Provided directories do not exist:\n{args.raw_dir}\n{args.ref_dir}")
        exit(1)
        
    setup_dataset(args.raw_dir, args.ref_dir, args.target_root)
