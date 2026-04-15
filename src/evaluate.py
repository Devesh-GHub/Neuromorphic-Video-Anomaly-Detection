import os
import sys
import yaml
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from preprocessing.load_datasets import load_ucsd_ped2
from preprocessing.video_dataset import VideoFrameDataset
from preprocessing.transforms import get_transform
from models.conv_autoencoder import ConvAutoencoder
from utils.anomaly_scoring import compute_anomaly_scores, normalize_scores
from utils.metrics import compute_auc_roc


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # -----------------------
    # Load config
    # -----------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/default.yaml")
    config = load_config(config_path)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------
    # Load test dataset
    # -----------------------
    # Convert relative data path to absolute path from script directory
    data_root = os.path.normpath(os.path.join(script_dir, "..", config["data"]["root_dir"]))
    _, all_test_videos = load_ucsd_ped2(data_root)
    
    # Filter to only test videos that have ground truth annotations
    gt_path = os.path.normpath(os.path.join(script_dir, "..", config["data"]["root_dir"], "UCSDped2/Test"))
    gt_folders = set([f.replace("_gt", "") for f in os.listdir(gt_path) if f.endswith("_gt")])
    test_videos = sorted([v for v in all_test_videos if os.path.basename(v) in gt_folders])
    
    print(f"Test videos with ground truth: {len(test_videos)}")
    
    transform = get_transform(config["data"]["image_size"])

    test_dataset = VideoFrameDataset(test_videos, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0
    )

    print(f"Total test frames: {len(test_dataset)}")

    # -----------------------
    # Load trained model
    # -----------------------
    model = ConvAutoencoder().to(DEVICE)
    checkpoint_path = "checkpoints/conv_autoencoder.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    print("Model loaded successfully")

    # -----------------------
    # Compute anomaly scores
    # -----------------------
    scores = compute_anomaly_scores(model, test_loader, DEVICE)
    scores = normalize_scores(scores)

    # -----------------------
    # Load ground-truth labels
    # -----------------------
    # UCSD Ped2 provides frame-level binary masks in Test*_gt folders
    # Convert to binary labels (0=normal, 1=anomaly)
    gt_path = os.path.normpath(os.path.join(script_dir, "..", config["data"]["root_dir"], "UCSDped2/Test"))
    gt_folders = sorted([f for f in os.listdir(gt_path) if f.endswith("_gt")])
    
    y_true = []
    for gt_folder in gt_folders:
        gt_frames = sorted(glob.glob(os.path.join(gt_path, gt_folder, "*.bmp")))
        for frame_path in gt_frames:
            # Open ground truth mask: if any pixel is non-zero, frame is anomalous
            gt_img = np.array(Image.open(frame_path).convert("L"))
            label = 1 if gt_img.max() > 0 else 0
            y_true.append(label)
    
    y_true = np.array(y_true)

    # -----------------------
    # Compute AUC-ROC
    # -----------------------
    auc = compute_auc_roc(y_true, scores)
    print(f"AUC-ROC: {auc:.4f}")

    # -----------------------
    # Save results
    # -----------------------
    os.makedirs("results", exist_ok=True)
    np.save("results/anomaly_scores.npy", scores)
    np.save("results/ground_truth.npy", y_true)

    print("Evaluation completed. Results saved.")


if __name__ == "__main__":
    main()
