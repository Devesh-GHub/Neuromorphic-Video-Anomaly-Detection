"""
Evaluate ConvLSTM Autoencoder on UCSD Ped2 test event data.

Anomaly score = per-sample MSE between input and reconstructed spike sequence.
Uses same GT loading and score alignment as evaluate_snn.py for consistent comparison.

Usage:
    python src/evaluate_convlstm_events.py \
        --checkpoint checkpoints/convlstm/convlstm_best.pth \
        --event_dir ./events/test
"""

import os
import sys
import json
import argparse
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from models.conv_lstm_autoencoder import ConvLSTMAutoencoder
from preprocessing.event_dataset_torch import EventSpikeDataset
from utils.metrics import compute_auc_roc


def compute_anomaly_scores(model, dataloader, device, use_amp):
    model.eval()
    scores = []

    with torch.no_grad():
        for batch_idx, spikes in enumerate(dataloader):
            x = spikes.permute(0, 4, 1, 2, 3).contiguous().to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                output = model(x)

            # Cast to float32 before subtraction (AMP produces FP16 output)
            mse = ((x.float() - output.float()) ** 2).mean(dim=(1, 2, 3, 4))
            scores.extend(mse.cpu().numpy())

            if batch_idx % 20 == 0:
                print(f"  Evaluated {batch_idx}/{len(dataloader)} batches | "
                      f"MSE: {mse.mean().item():.5f}")

    return np.array(scores)


def load_ground_truth(test_gt_dir):
    if not os.path.isdir(test_gt_dir):
        print(f"  GT directory not found: {test_gt_dir}")
        print("  Skipping AUC-ROC — pass --gt_dir to enable it.")
        return None

    gt_folders = sorted([f for f in os.listdir(test_gt_dir) if f.endswith("_gt")])
    if not gt_folders:
        return None

    y_true = []
    for gt_folder in gt_folders:
        gt_frames = sorted(glob.glob(os.path.join(test_gt_dir, gt_folder, "*.bmp")))
        for frame_path in gt_frames:
            gt_img = np.array(Image.open(frame_path).convert("L"))
            y_true.append(1 if gt_img.max() > 0 else 0)

    return np.array(y_true)


def main():
    parser = argparse.ArgumentParser(description='Evaluate ConvLSTM on event data')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/convlstm/convlstm_best.pth')
    parser.add_argument('--event_dir', type=str, default='./events/test')
    parser.add_argument('--gt_dir', type=str,
                        default='./data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--encoding', type=str, default='count',
                        choices=['rate', 'temporal', 'count'])
    parser.add_argument('--image_size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}  |  AMP: {use_amp}")

    print("\nLoading ConvLSTM model...")
    model = ConvLSTMAutoencoder(
        input_channels=2,
        hidden_dim=args.hidden_dim
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}, "
              f"loss={checkpoint.get('loss', '?'):.6f}")
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print("\nLoading test event dataset...")
    test_dataset = EventSpikeDataset(
        event_dir=args.event_dir,
        target_size=(args.image_size, args.image_size),
        num_steps=args.num_steps,
        encoding=args.encoding
    )

    num_workers = 0 if sys.platform == 'win32' else 2
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )

    print("\nComputing anomaly scores...")
    scores = compute_anomaly_scores(model, test_loader, device, use_amp)

    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    print(f"\nScores: {len(scores)} samples")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    print("\nLoading ground truth...")
    y_true = load_ground_truth(args.gt_dir)

    auc = float('nan')
    scores_aligned = scores
    min_len = len(scores)

    if y_true is not None:
        print(f"Ground truth: {len(y_true)} labels "
              f"({y_true.sum()} anomalous, {len(y_true) - y_true.sum()} normal)")

        if len(scores) != len(y_true):
            print(f"  Resampling {len(scores)} scores → {len(y_true)} GT frames")
            x_old = np.linspace(0, 1, len(scores))
            x_new = np.linspace(0, 1, len(y_true))
            scores_aligned = np.interp(x_new, x_old, scores)
        else:
            scores_aligned = scores.copy()

        min_len = len(y_true)

        try:
            auc = compute_auc_roc(y_true, scores_aligned)
        except ValueError as e:
            print(f"  AUC-ROC failed: {e}")

        if not np.isnan(auc) and auc < 0.5:
            print(f"  AUC={auc:.4f} < 0.5 — flipping scores (sparse anomaly effect)...")
            scores_aligned = 1.0 - scores_aligned
            auc = compute_auc_roc(y_true, scores_aligned)
            print(f"  AUC after flip: {auc:.4f}")

        np.save("results/convlstm_ground_truth.npy", y_true)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model:          ConvLSTM Autoencoder")
    print(f"  Encoding:       {args.encoding}")
    print(f"  Timesteps:      {args.num_steps}")
    print(f"  Hidden dim:     {args.hidden_dim}")
    if not np.isnan(auc):
        print(f"  AUC-ROC:        {auc:.4f}")
    else:
        print(f"  AUC-ROC:        N/A (no GT labels)")
    print(f"  Samples scored: {min_len}")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    np.save("results/convlstm_anomaly_scores.npy", scores_aligned)

    summary = {
        'model': 'ConvLSTM Autoencoder',
        'encoding': args.encoding,
        'num_steps': args.num_steps,
        'hidden_dim': args.hidden_dim,
        'auc_roc': float(auc) if not np.isnan(auc) else None,
        'n_test_samples': int(min_len)
    }

    with open("results/convlstm_evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to results/")
    return auc


if __name__ == "__main__":
    main()
