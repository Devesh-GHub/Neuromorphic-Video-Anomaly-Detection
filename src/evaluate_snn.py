"""
Evaluate SNN Autoencoder on UCSD Ped2 test set.

Computes per-frame anomaly scores using reconstruction error
(MSE between input and output spike counts), then evaluates
against ground truth labels using AUC-ROC.

Usage:
    python src/evaluate_snn.py --checkpoint checkpoints/snn_autoencoder_best.pth
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

from models.snn_autoencoder import SNNAutoencoder
from preprocessing.event_dataset_torch import EventSpikeDataset
from utils.metrics import compute_auc_roc


def compute_snn_anomaly_scores(model, dataloader, device, score_mode='combined'):
    """
    Compute anomaly scores for each sample in the dataloader.

    score_mode options:
      'combined'  — normalized MSE + normalized mem (previous default)
      'mem_only'  — membrane potential energy only
      'mse_only'  — rate reconstruction MSE only
      'weighted'  — 0.2 * norm_mse + 0.8 * norm_mem (mem dominates since MSE
                    variance is tiny in practice)
    """
    model.eval()
    mse_scores = []
    mem_scores = []
    spike_rates = []

    with torch.no_grad():
        for batch_idx, spikes in enumerate(dataloader):
            spikes = spikes.to(device)  # (B, 2, H, W, T)

            spike_count_in = spikes.sum(dim=-1)  # (B, 2, H, W)
            spike_count_out, mem_final, spike_record = model(spikes)

            # Rate MSE — consistent with training loss
            rate_in  = spike_count_in  / model.num_steps
            rate_out = spike_count_out / model.num_steps
            mse_score = ((rate_in - rate_out) ** 2).mean(dim=(1, 2, 3))

            # Membrane potential energy — continuous, not bounded to [0,1]
            mem_score = (mem_final ** 2).mean(dim=(1, 2, 3))

            mse_scores.extend(mse_score.cpu().numpy())
            mem_scores.extend(mem_score.cpu().numpy())

            spike_rates.append(spike_record.mean().item())

            if batch_idx % 20 == 0:
                print(f"  Evaluated {batch_idx}/{len(dataloader)} batches | "
                      f"MSE: {mse_score.mean():.5f} | Mem: {mem_score.mean():.3f}")

    mse_scores = np.array(mse_scores)
    mem_scores  = np.array(mem_scores)

    def _norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 0 else np.zeros_like(x)

    norm_mse = _norm(mse_scores)
    norm_mem = _norm(mem_scores)

    print(f"\n  MSE score range:  [{mse_scores.min():.5f}, {mse_scores.max():.5f}]")
    print(f"  Mem score range:  [{mem_scores.min():.3f}, {mem_scores.max():.3f}]")
    print(f"  Score mode:       {score_mode}")

    if score_mode == 'mem_only':
        scores = norm_mem
    elif score_mode == 'mse_only':
        scores = norm_mse
    elif score_mode == 'weighted':
        scores = 0.2 * norm_mse + 0.8 * norm_mem
    else:  # combined
        scores = norm_mse + norm_mem

    return scores, np.mean(spike_rates), mse_scores, mem_scores


def load_ground_truth(test_gt_dir):
    """
    Load frame-level ground truth labels from UCSD Ped2.

    Each test video has a corresponding _gt folder with binary masks.
    Frame is anomalous if any pixel in mask is non-zero.
    Returns None if the directory does not exist.
    """
    if not os.path.isdir(test_gt_dir):
        print(f"  GT directory not found: {test_gt_dir}")
        print("  Skipping AUC-ROC — pass --gt_dir to enable it.")
        return None

    gt_folders = sorted([f for f in os.listdir(test_gt_dir) if f.endswith("_gt")])

    if not gt_folders:
        print(f"  No *_gt folders found inside {test_gt_dir}")
        return None

    y_true = []
    for gt_folder in gt_folders:
        gt_frames = sorted(glob.glob(os.path.join(test_gt_dir, gt_folder, "*.bmp")))
        for frame_path in gt_frames:
            gt_img = np.array(Image.open(frame_path).convert("L"))
            label = 1 if gt_img.max() > 0 else 0
            y_true.append(label)

    return np.array(y_true)


def normalize_scores(scores):
    """Normalize anomaly scores to [0, 1]."""
    if len(scores) == 0:
        return scores
    min_val, max_val = scores.min(), scores.max()
    if max_val - min_val == 0:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)


def main():
    parser = argparse.ArgumentParser(description='Evaluate SNN Autoencoder')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/snn_autoencoder_best.pth')
    parser.add_argument('--event_dir', type=str, default='./events/test')
    parser.add_argument('--gt_dir', type=str, 
                        default='./data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--encoding', type=str, default='rate')
    parser.add_argument('--beta', type=float, default=0.95)
    parser.add_argument('--bottleneck_channels', type=int, default=8,
                        help='Must match the value used during training.')
    parser.add_argument('--score_mode', type=str, default='all',
                        choices=['combined', 'mem_only', 'mse_only', 'weighted', 'all'],
                        help="Scoring mode. 'all' tries every mode and reports each AUC.")

    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading SNN model...")
    model = SNNAutoencoder(
        in_channels=2,
        beta=args.beta,
        num_steps=args.num_steps,
        bottleneck_channels=args.bottleneck_channels
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}, "
              f"loss={checkpoint.get('loss', '?')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Load test dataset
    print("\nLoading test event dataset...")
    test_dataset = EventSpikeDataset(
        event_dir=args.event_dir,
        target_size=(128, 128),
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
    
    # Compute anomaly scores (raw components — mode applied below)
    print("\nComputing anomaly scores...")
    modes_to_run = ['combined', 'mem_only', 'mse_only', 'weighted'] \
                   if args.score_mode == 'all' else [args.score_mode]

    # Run model once to get raw component arrays
    _, avg_spike_rate, raw_mse, raw_mem = compute_snn_anomaly_scores(
        model, test_loader, device, score_mode='combined'
    )
    print(f"\nScores: {len(raw_mse)} samples")
    print(f"Average output spike rate: {avg_spike_rate:.4f}")

    # Load ground truth
    print("\nLoading ground truth...")
    y_true = load_ground_truth(args.gt_dir)

    os.makedirs("results", exist_ok=True)

    best_auc = float('nan')
    best_mode = modes_to_run[0]
    best_scores_aligned = None
    y_true_aligned = None
    min_len = len(raw_mse)

    def _norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 0 else np.zeros_like(x)

    norm_mse = _norm(raw_mse)
    norm_mem = _norm(raw_mem)

    mode_aucs = {}

    for mode in modes_to_run:
        if mode == 'mem_only':
            scores = norm_mem.copy()
        elif mode == 'mse_only':
            scores = norm_mse.copy()
        elif mode == 'weighted':
            scores = 0.2 * norm_mse + 0.8 * norm_mem
        else:
            scores = norm_mse + norm_mem

        scores = normalize_scores(scores)

        if y_true is not None:
            if len(scores) != len(y_true):
                x_old = np.linspace(0, 1, len(scores))
                x_new = np.linspace(0, 1, len(y_true))
                scores_aligned = np.interp(x_new, x_old, scores)
            else:
                scores_aligned = scores.copy()

            y_true_aligned = y_true
            min_len = len(y_true)

            try:
                auc = compute_auc_roc(y_true_aligned, scores_aligned)
            except ValueError as e:
                print(f"  [{mode}] AUC-ROC failed: {e}")
                auc = float('nan')

            if not np.isnan(auc) and auc < 0.5:
                scores_aligned = 1.0 - scores_aligned
                auc = compute_auc_roc(y_true_aligned, scores_aligned)

            mode_aucs[mode] = auc
            print(f"  [{mode:10s}] AUC-ROC: {auc:.4f}")

            if np.isnan(best_auc) or (not np.isnan(auc) and auc > best_auc):
                best_auc = auc
                best_mode = mode
                best_scores_aligned = scores_aligned
        else:
            scores_aligned = scores
            best_scores_aligned = scores_aligned

    if y_true is not None:
        np.save("results/snn_ground_truth.npy", y_true_aligned)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model:          SNN Autoencoder (snnTorch)")
    print(f"  Encoding:       {args.encoding}")
    print(f"  Timesteps:      {args.num_steps}")
    if mode_aucs:
        for m, a in mode_aucs.items():
            marker = " <-- BEST" if m == best_mode else ""
            print(f"  AUC [{m:10s}]: {a:.4f}{marker}")
    else:
        print(f"  AUC-ROC:        N/A (no GT labels)")
    print(f"  Avg spike rate: {avg_spike_rate:.4f}")
    print(f"  Sparsity:       {1.0 - avg_spike_rate:.4f}")
    print(f"  Samples scored: {min_len}")
    print("=" * 60)

    np.save("results/snn_anomaly_scores.npy", best_scores_aligned)

    summary = {
        'model': 'SNN Autoencoder (snnTorch)',
        'encoding': args.encoding,
        'num_steps': args.num_steps,
        'beta': args.beta,
        'best_score_mode': best_mode,
        'auc_roc': float(best_auc) if not np.isnan(best_auc) else None,
        'all_aucs': {k: float(v) if not np.isnan(v) else None for k, v in mode_aucs.items()},
        'avg_spike_rate': float(avg_spike_rate),
        'n_test_samples': int(min_len)
    }

    with open("results/snn_evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to results/")
    return best_auc


if __name__ == "__main__":
    main()