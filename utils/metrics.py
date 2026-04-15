import cv2
import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_reconstruction_error(original, reconstructed, reduction="mean"):
    """
    Compute reconstruction error between original and reconstructed frames.

    Args:
        original (Tensor): shape (B, C, H, W)
        reconstructed (Tensor): shape (B, C, H, W)
        reduction (str): "mean" or "sum"

    Returns:
        Tensor: reconstruction error per sample (B,)
    """
    error = (original - reconstructed) ** 2  # MSE

    # Average over channels and spatial dimensions
    error = error.view(error.size(0), -1).mean(dim=1)

    return error


def compute_auc_roc(y_true, y_scores):
    """
    Compute AUC-ROC score.

    Args:
        y_true (array-like): ground truth labels (0=normal, 1=anomaly)
        y_scores (array-like): anomaly scores

    Returns:
        float: AUC-ROC value
    """
    return roc_auc_score(y_true, y_scores)


def compute_psnr(original, reconstructed, max_pixel_value=1.0):
    """
    Compute PSNR (Peak Signal-to-Noise Ratio).

    Args:
        original (Tensor)
        reconstructed (Tensor)
        max_pixel_value (float): 1.0 if normalized

    Returns:
        float: PSNR value
    """
    mse = torch.mean((original - reconstructed) ** 2)

    if mse == 0:
        return float("inf")

    psnr = 20 * torch.log10(torch.tensor(max_pixel_value)) - 10 * torch.log10(mse)
    return psnr.item()


def load_ped2_frame_labels(test_gt_dir):
    """
    Convert Ped2 pixel-level GT masks into frame-level labels.
    0 = normal, 1 = anomaly
    """
    gt_files = sorted(os.listdir(test_gt_dir))
    labels = []

    for gt_file in gt_files:
        # Skip non-image files (e.g., .DS_Store on macOS)
        if not gt_file.lower().endswith(('.bmp', '.tiff')):
            continue
            
        gt_path = os.path.join(test_gt_dir, gt_file)
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"Could not read {gt_path}")

        # If any anomalous pixel exists → anomaly frame
        label = 1 if np.any(mask > 0) else 0
        labels.append(label)

    return np.array(labels)


def frame_labels_to_sequence_labels(frame_labels, sequence_length):
    seq_labels = []
    for i in range(len(frame_labels) - sequence_length + 1):
        seq_labels.append(
            1 if np.any(frame_labels[i:i+sequence_length]) else 0
        )
    return np.array(seq_labels)
