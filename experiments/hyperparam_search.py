import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
import torch.nn as nn
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from preprocessing.load_datasets import load_ucsd_ped2
from preprocessing.sequence_dataset import VideoSequenceDataset
from preprocessing.transforms import get_transform
from models.conv3d_autoencoder import Conv3DAutoencoder as Model
from utils.gt_processing import get_sequence_gt_labels
from utils.anomaly_scoring import compute_anomaly_scores,normalize_scores
from utils.metrics import compute_auc_roc


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config():
    """Load config from default.yaml"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/default.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def train_and_eval(config, lr, hidden_dim, epochs=10):
    print(f"\nRunning LR={lr}, HiddenDim={hidden_dim}")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # -----------------------
    # Data
    # -----------------------

    data_root = config["data"]["root_dir"]
    if not os.path.isabs(data_root):
        # If relative, resolve relative to the project root (parent of experiments/)
        data_root = os.path.join(script_dir, "../", data_root)
    data_root = os.path.abspath(data_root)
    
    print(f"Data root: {data_root}")
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"\n Dataset not found at:\n{data_root}\nCheck default.yaml path!")
    
    train_videos, test_videos = load_ucsd_ped2(dataset_root=data_root)
    
    if len(train_videos) == 0:
        raise ValueError(f"No training videos found in {data_root}")
    if len(test_videos) == 0:
        raise ValueError(f"No test videos found in {data_root}")
    
    transform = get_transform((config["data"]["image_size"],
                            config["data"]["image_size"]))
    
    SEQ_LEN = config["training"]["sequence_length"]
    train_ds = VideoSequenceDataset(train_videos, sequence_length=SEQ_LEN, transform=transform)
    test_ds  = VideoSequenceDataset(test_videos,  sequence_length=SEQ_LEN, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False,num_workers=2, pin_memory=True)

    # -----------------------
    # Model + Loss + Optimizer
    # -----------------------
    model = Model(
        in_channels=3,
        hidden_dim=hidden_dim
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss setup
    l1 = nn.L1Loss()
    ssim = SSIM(data_range=1.0, size_average=True, channel=3)

    # -----------------------
    # Training (quick)
    # -----------------------
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        print(f"Epoch {epoch+1}/{epochs} started...")
        for seq in train_loader:
            seq = seq.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                reconstructed = model(seq)
                l1_loss = nn.L1Loss()(reconstructed, seq)

                # SSIM over sequence
                B, T, C, H, W = reconstructed.shape
                ssim_loss = 0
                for t in range(T):
                    ssim_loss += 1 - ssim(reconstructed[:, t], seq[:, t])
                ssim_loss /= T

                # Total loss
                loss = 0.85 * l1_loss + 0.15 * ssim_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.5f}")

    # -----------------------
    # Evaluation
    # -----------------------
    model.eval()
    gt_root = os.path.join(data_root, "UCSDped2", "Test")

    # Compute anomaly scores (sequence-level)
    scores = compute_anomaly_scores(model, test_loader, DEVICE)

    # 2. Normalize scores before AUC 
    scores = normalize_scores(scores)

    # 3. Convert frame labels → sequence labels
    y_true = get_sequence_gt_labels(
    gt_root,
    sequence_length=SEQ_LEN
    )

    # safety check
    # ALIGN SCORES & LABELS (fix length mismatch)
    min_len = min(len(scores), len(y_true))
    scores = scores[:min_len]
    y_true = y_true[:min_len]


    # 4. Compute AUC-ROC
    auc = compute_auc_roc(y_true, scores)

    print(f"AUC-ROC: {auc:.4f}")
    return auc


# -----------------------
# Hyperparameter Search
# -----------------------


if __name__ == "__main__":

    config = load_config()

    experiments = [
    {"lr": 3e-4, "hidden_dim": 128},
    {"lr": 2e-4, "hidden_dim": 128},
    {"lr": 3e-4, "hidden_dim": 64},
    ]


    results = []

    for exp in experiments:
        auc = train_and_eval(config, exp["lr"], exp["hidden_dim"])
        results.append((exp["lr"], exp["hidden_dim"], auc))

    print("\nFinal Summary:")
    for lr, hd, auc in results:
        print(f"LR={lr}, HiddenDim={hd} → AUC={auc:.4f}")
