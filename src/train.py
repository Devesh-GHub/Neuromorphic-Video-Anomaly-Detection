import yaml
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from preprocessing.load_datasets import load_ucsd_ped2
from preprocessing.sequence_dataset import VideoSequenceDataset
from preprocessing.transforms import get_transform
from models.conv_lstm_autoencoder import ConvLSTMAutoencoder
from utils.logger import Logger


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/default.yaml")
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}")

    # Picks fastest cuDNN kernel for each conv shape
    torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------ data
    train_videos, _ = load_ucsd_ped2(config["data"]["root_dir"])
    transform = get_transform(config["data"]["image_size"])
    seq_len = config["training"].get("sequence_length", 8)

    train_dataset = VideoSequenceDataset(
        train_videos, sequence_length=seq_len, transform=transform
    )

    use_workers = sys.platform != "win32"
    num_workers = 2 if use_workers else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_amp,
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
    )

    # ----------------------------------------------------------------- model
    model = ConvLSTMAutoencoder(
        input_channels=3,
        hidden_dim=config["model"].get("hidden_dim", 64)
    ).to(device)

    # Fuses the T-step LSTM loop, reduces Python dispatch overhead
    if hasattr(torch, "compile"):
        model = torch.compile(model)
        print("  torch.compile() enabled")

    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"], eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger = Logger(config["logging"]["log_dir"])

    os.makedirs("checkpoints", exist_ok=True)
    epochs = config["training"]["epochs"]
    global_step = 0
    best_loss = float("inf")

    print(f"\nTraining for {epochs} epochs  |  {len(train_loader)} batches/epoch")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0

        for sequences in train_loader:
            # non_blocking overlaps CPU→GPU transfer with compute
            sequences = sequences.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(sequences)
                loss = criterion(output, sequences)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # Log every 10 steps — loss.item() forces a CUDA sync, keep it rare
            if global_step % 10 == 0:
                logger.log_loss(loss.item(), global_step)
                logger.log_lr(optimizer, global_step)

            global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch [{epoch+1}/{epochs}] ({elapsed:.1f}s) | "
              f"Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        logger.log_images(sequences[:, 0], output[:, 0].float(), epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            state = model._orig_mod.state_dict() \
                if hasattr(model, "_orig_mod") else model.state_dict()
            torch.save({
                "epoch": epoch,
                "model_state_dict": state,
                "loss": best_loss,
            }, "checkpoints/conv_lstm_autoencoder_best.pth")
            print(f"  ✓ Best model saved (loss={best_loss:.6f})")

        if (epoch + 1) % 10 == 0:
            state = model._orig_mod.state_dict() \
                if hasattr(model, "_orig_mod") else model.state_dict()
            torch.save(state, f"checkpoints/conv_lstm_autoencoder_epoch{epoch+1}.pth")

    state = model._orig_mod.state_dict() \
        if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save(state, "checkpoints/conv_lstm_autoencoder.pth")
    logger.close()

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE  |  Best loss: {best_loss:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
