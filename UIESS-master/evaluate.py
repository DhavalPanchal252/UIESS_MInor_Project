import os
import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse
from pathlib import Path

def calculate_psnr(img1, img2):
    return psnr(img1, img2)

def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2, data_range=255)

def evaluate(pred_dir, gt_dir):
    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)
    
    pred_files = sorted([f for f in pred_path.iterdir() if f.suffix in ['.png', '.jpg']])
    gt_files = sorted([f for f in gt_path.iterdir() if f.suffix in ['.png', '.jpg']])
    
    # Map by filename
    pred_map = {f.name: f for f in pred_files}
    gt_map = {f.name: f for f in gt_files}
    
    common_names = sorted(list(set(pred_map.keys()) & set(gt_map.keys())))
    
    if not common_names:
        print("No matching files found between Prediction and Ground Truth directories.")
        return

    psnr_values = []
    ssim_values = []
    
    print(f"Evaluating {len(common_names)} images...")
    
    for name in common_names:
        # Load images
        pred_img = cv2.imread(str(pred_map[name]))
        gt_img = cv2.imread(str(gt_map[name]))
        
        # Ensure same size (resize pred to gt if needed)
        if pred_img.shape != gt_img.shape:
            pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
            
        p = calculate_psnr(gt_img, pred_img)
        s = calculate_ssim(gt_img, pred_img)
        
        psnr_values.append(p)
        ssim_values.append(s)
        
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    print(f"\nResults on {len(common_names)} images:")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    with open("metrics.txt", "w") as f:
        f.write(f"PSNR: {avg_psnr:.4f}\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True, help="Path to predicted/enhanced images")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to ground truth/reference images")
    args = parser.parse_args()
    
    if not os.path.isdir(args.pred_dir) or not os.path.isdir(args.gt_dir):
        print("Error: Invalid directory paths.")
        exit(1)
        
    evaluate(args.pred_dir, args.gt_dir)
