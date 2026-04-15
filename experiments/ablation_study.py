import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from preprocessing.load_datasets import load_ucsd_ped2
from preprocessing.video_dataset import VideoFrameDataset
from preprocessing.transforms import get_transform
from models.conv_autoencoder import ConvAutoencoder
from utils.anomaly_scoring import compute_anomaly_scores, normalize_scores
from utils.metrics import compute_auc_roc
from utils.metrics import load_ped2_frame_labels


def get_loss_function(name):
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "mse_mae":
        return lambda x, y: nn.MSELoss()(x, y) + nn.L1Loss()(x, y)
    else:
        raise ValueError("Unknown loss")


def train_quick(model, dataloader, loss_fn, optimizer, device, epochs=10):
    model.train()
    for _ in range(epochs):
        for batch in dataloader:
            batch = batch.to(device)
            out = model(batch)
            loss = loss_fn(out, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def run_experiment(exp_name, loss_name, lr, latent_dim):
    print(f"\nRunning {exp_name}")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Data
    print("Loading datasets...")
    train_videos, test_videos = load_ucsd_ped2("./data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2")
    print(f"Loaded {len(train_videos)} train, {len(test_videos)} test videos")
    transform = get_transform(128)

    train_ds = VideoFrameDataset(train_videos, transform)
    train_ds = Subset(train_ds, list(range(1000)))  # quick training

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    # Model
    model = ConvAutoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = get_loss_function(loss_name)

    # Train
    train_quick(model, train_loader, loss_fn, optimizer, DEVICE)

   
    # Evaluate
    test_ds = VideoFrameDataset(test_videos, transform)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    scores = compute_anomaly_scores(model, test_loader, DEVICE)
    scores = normalize_scores(scores)

    # ---- Load GT labels (frame-level) ----
    DATA_ROOT = "./data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test"

    y_true_all = []

    for test_video in sorted(os.listdir(DATA_ROOT)):
        if test_video.endswith("_gt"):
            gt_dir = os.path.join(DATA_ROOT, test_video)
            labels = load_ped2_frame_labels(gt_dir)
            y_true_all.extend(labels)

    y_true = torch.tensor(y_true_all).numpy()

    # Ensure same length (truncate scores if needed)
    min_len = min(len(y_true), len(scores))
    y_true = y_true[:min_len]
    scores = scores[:min_len]
    
    print(f"Using {min_len} frames for evaluation (y_true: {len(y_true)}, scores: {len(scores)})")

    auc = compute_auc_roc(y_true, scores)
    print(f"AUC-ROC: {auc:.4f}")
    return {
        "experiment": exp_name,
        "loss": loss_name,
        "learning_rate": lr,
        "latent_dim": latent_dim,
        "auc_roc": auc,
    }



if __name__ == "__main__":
    results = []

    results.append(run_experiment("Exp1_MSE", "mse", 1e-3, 128))
    results.append(run_experiment("Exp2_MAE", "mae", 1e-3, 128))
    results.append(run_experiment("Exp3_MSE_MAE", "mse_mae", 1e-3, 128))
    results.append(run_experiment("Exp4_LR_1e4", "mse", 1e-4, 128))
    results.append(run_experiment("Exp5_Latent_64", "mse", 1e-3, 64))

    print("\nFinal Results:")
    for r in results:
        print(r)
