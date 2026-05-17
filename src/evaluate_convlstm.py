"""
Evaluate ConvLSTM Autoencoder baseline on UCSD Ped2.
Uses sequence-level reconstruction error.
"""

import os
import sys
import glob
import yaml
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from models.conv_lstm_autoencoder import ConvLSTMAutoencoder
from preprocessing.load_datasets import load_ucsd_ped2
from preprocessing.sequence_dataset import VideoSequenceDataset
from preprocessing.transforms import get_transform
from utils.metrics import compute_auc_roc


def main():
    # Load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    data_root = os.path.normpath(os.path.join(script_dir, "..", config["data"]["root_dir"]))
    _, test_videos = load_ucsd_ped2(data_root)
    
    # Filter to videos with ground truth
    gt_path = os.path.normpath(os.path.join(
        script_dir, "..", config["data"]["root_dir"], "UCSDped2/Test"))
    gt_folders = set([f.replace("_gt", "") for f in os.listdir(gt_path) if f.endswith("_gt")])
    test_videos = sorted([v for v in test_videos if os.path.basename(v) in gt_folders])
    
    transform = get_transform(config["data"]["image_size"])
    seq_len = config["training"].get("sequence_length", 8)
    
    test_dataset = VideoSequenceDataset(test_videos, sequence_length=seq_len, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], 
                             shuffle=False, num_workers=0)
    
    # Load model
    model = ConvLSTMAutoencoder(
        input_channels=3,
        hidden_dim=config["model"].get("hidden_dim", 64)
    ).to(device)
    
    checkpoint = torch.load("checkpoints/conv_lstm_autoencoder.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Compute anomaly scores
    scores = []
    with torch.no_grad():
        for sequences in test_loader:
            sequences = sequences.to(device)  # (B, T, C, H, W)
            output = model(sequences)
            # Per-sequence error
            errors = ((sequences - output) ** 2).mean(dim=(1, 2, 3, 4))
            scores.extend(errors.cpu().numpy())
    
    scores = np.array(scores)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    # Load ground truth (frame-level → sequence-level)
    y_true = []
    for gt_folder in sorted([f for f in os.listdir(gt_path) if f.endswith("_gt")]):
        gt_frames = sorted(glob.glob(os.path.join(gt_path, gt_folder, "*.bmp")))
        frame_labels = []
        for fp in gt_frames:
            gt_img = np.array(Image.open(fp).convert("L"))
            frame_labels.append(1 if gt_img.max() > 0 else 0)
        
        # Convert to sequence-level labels
        for i in range(len(frame_labels) - seq_len + 1):
            label = 1 if any(frame_labels[i:i+seq_len]) else 0
            y_true.append(label)
    
    y_true = np.array(y_true)
    
    # Align lengths
    min_len = min(len(scores), len(y_true))
    scores = scores[:min_len]
    y_true = y_true[:min_len]
    
    auc = compute_auc_roc(y_true, scores)
    
    print("=" * 60)
    print("ConvLSTM BASELINE RESULTS")
    print("=" * 60)
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Samples: {min_len}")
    print("=" * 60)
    
    np.save("results/convlstm_anomaly_scores.npy", scores)
    
    return auc


if __name__ == "__main__":
    main()