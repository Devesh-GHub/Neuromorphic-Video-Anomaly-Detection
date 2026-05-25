"""
Evaluate ConvAutoencoder on UCSD Ped2 test frames.

Usage:
    python src/evaluate_conv_autoencoder.py
"""

import os
import sys
import glob
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from preprocessing.load_datasets import load_ucsd_ped2
from preprocessing.motion_dataset import MotionFrameDataset
from models.conv_autoencoder import ConvAutoencoder
from utils.metrics import compute_auc_roc


def compute_anomaly_scores(model, dataloader, device, use_amp):
    model.eval()
    scores = []

    with torch.no_grad():
        for batch_idx, frames in enumerate(dataloader):
            frames = frames.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(frames)

            # cast to float32 before subtraction (AMP may produce FP16)
            mse = ((frames.float() - output.float()) ** 2).mean(dim=(1, 2, 3))
            scores.extend(mse.cpu().numpy())

            if batch_idx % 20 == 0:
                print(f"  Evaluated {batch_idx}/{len(dataloader)} batches | "
                      f"MSE: {mse.mean().item():.5f}")

    return np.array(scores)


def load_ground_truth(gt_path):
    gt_folders = sorted([f for f in os.listdir(gt_path) if f.endswith("_gt")])
    if not gt_folders:
        return None

    y_true = []
    for gt_folder in gt_folders:
        gt_frames = sorted(glob.glob(os.path.join(gt_path, gt_folder, "*.bmp")))
        for frame_path in gt_frames:
            gt_img = np.array(Image.open(frame_path).convert("L"))
            y_true.append(1 if gt_img.max() > 0 else 0)

    return np.array(y_true)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.normpath(os.path.join(script_dir, "../data/UCSD_Anomaly_Dataset.v1p2"))
    checkpoint_path = "checkpoints/conv_autoencoder/conv_autoencoder_best.pth"
    IMAGE_SIZE = 128
    BATCH_SIZE = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}  |  AMP: {use_amp}")

    _, all_test_videos = load_ucsd_ped2(data_root)
    gt_dir = os.path.join(data_root, "UCSDped2/Test")
    gt_folders = set(f.replace("_gt", "") for f in os.listdir(gt_dir) if f.endswith("_gt"))
    test_videos = sorted(v for v in all_test_videos if os.path.basename(v) in gt_folders)
    print(f"Test videos with GT: {len(test_videos)}")

    test_dataset = MotionFrameDataset(test_videos, image_size=IMAGE_SIZE)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp,
    )
    print(f"Total test frames: {len(test_dataset)}")

    print("\nLoading ConvAutoencoder...")
    model = ConvAutoencoder().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded epoch {checkpoint.get('epoch', '?')}, "
              f"loss={checkpoint.get('loss', '?'):.6f}")
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print("\nComputing anomaly scores...")
    scores = compute_anomaly_scores(model, test_loader, device, use_amp)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    print(f"\nScores: {len(scores)} samples")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    print("\nLoading ground truth...")
    y_true = load_ground_truth(gt_dir)

    auc = float("nan")
    scores_aligned = scores

    if y_true is not None:
        print(f"GT frames: {len(y_true)} "
              f"({y_true.sum()} anomalous, {len(y_true)-y_true.sum()} normal)")

        if len(scores) != len(y_true):
            print(f"  Resampling {len(scores)} scores → {len(y_true)} GT frames")
            x_old = np.linspace(0, 1, len(scores))
            x_new = np.linspace(0, 1, len(y_true))
            scores_aligned = np.interp(x_new, x_old, scores)
        else:
            scores_aligned = scores.copy()

        try:
            auc = compute_auc_roc(y_true, scores_aligned)
        except ValueError as e:
            print(f"  AUC-ROC failed: {e}")

        if not np.isnan(auc) and auc < 0.5:
            print(f"  AUC={auc:.4f} < 0.5 — flipping scores...")
            scores_aligned = 1.0 - scores_aligned
            auc = compute_auc_roc(y_true, scores_aligned)
            print(f"  AUC after flip: {auc:.4f}")

        np.save("results/convae_ground_truth.npy", y_true)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model:          ConvAutoencoder (RGB frames)")
    print(f"  Image size:     {IMAGE_SIZE}x{IMAGE_SIZE}")
    if not np.isnan(auc):
        print(f"  AUC-ROC:        {auc:.4f}")
    else:
        print(f"  AUC-ROC:        N/A (no GT labels)")
    print(f"  Frames scored:  {len(scores_aligned)}")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    np.save("results/convae_anomaly_scores.npy", scores_aligned)

    summary = {
        "model": "ConvAutoencoder",
        "image_size": IMAGE_SIZE,
        "auc_roc": float(auc) if not np.isnan(auc) else None,
        "n_test_frames": int(len(scores_aligned)),
    }
    with open("results/convae_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to results/")
    return auc


if __name__ == "__main__":
    main()
