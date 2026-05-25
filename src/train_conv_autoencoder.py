"""
Train ConvAutoencoder on UCSD Ped2 RGB frames.

Usage:
    python src/train_conv_autoencoder.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from preprocessing.load_datasets import load_ucsd_ped2
from preprocessing.motion_dataset import MotionFrameDataset
from models.conv_autoencoder import ConvAutoencoder


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.normpath(os.path.join(script_dir, "../data/UCSD_Anomaly_Dataset.v1p2"))

    EPOCHS = 50
    BATCH_SIZE = 16
    LR = 1e-3
    IMAGE_SIZE = 128
    CHECKPOINT_DIR = "checkpoints/conv_autoencoder"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}")
    torch.backends.cudnn.benchmark = True

    train_videos, _ = load_ucsd_ped2(data_root)
    train_dataset = MotionFrameDataset(train_videos, image_size=IMAGE_SIZE)

    use_workers = sys.platform != "win32"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2 if use_workers else 0,
        pin_memory=use_amp,
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
        drop_last=True,
    )

    model = ConvAutoencoder().to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)
        print("  torch.compile() enabled")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")
    print(f"Train frames: {len(train_dataset)}  |  Batches/epoch: {len(train_loader)}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_loss = float("inf")

    print(f"\nTraining for {EPOCHS} epochs")
    print("=" * 60)

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0

        for frames in train_loader:
            frames = frames.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(frames)
                loss = criterion(output, frames)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch [{epoch+1}/{EPOCHS}] ({elapsed:.1f}s) | "
              f"Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            state = model._orig_mod.state_dict() \
                if hasattr(model, "_orig_mod") else model.state_dict()
            torch.save({
                "epoch": epoch,
                "model_state_dict": state,
                "loss": best_loss,
            }, os.path.join(CHECKPOINT_DIR, "conv_autoencoder_best.pth"))
            print(f"  ✓ Best model saved (loss={best_loss:.6f})")

        if (epoch + 1) % 10 == 0:
            state = model._orig_mod.state_dict() \
                if hasattr(model, "_orig_mod") else model.state_dict()
            torch.save({
                "epoch": epoch,
                "model_state_dict": state,
                "loss": avg_loss,
            }, os.path.join(CHECKPOINT_DIR, f"conv_autoencoder_epoch{epoch+1}.pth"))

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE  |  Best loss: {best_loss:.6f}")
    print(f"Checkpoint: {CHECKPOINT_DIR}/conv_autoencoder_best.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
