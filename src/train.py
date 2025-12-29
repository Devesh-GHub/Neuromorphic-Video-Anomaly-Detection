import yaml
import os
import sys
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


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for sequences in dataloader:
        sequences = sequences.to(device)   # (B, T, C, H, W)

        output = model(sequences)
        loss = criterion(output, sequences)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # -----------------------
    # Load config
    # -----------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/default.yaml")
    config = load_config(config_path)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", DEVICE)

    # -----------------------
    # Dataset (SEQUENCES)
    # -----------------------
    train_videos, _ = load_ucsd_ped2(config["data"]["root_dir"])

    transform = get_transform(config["data"]["image_size"])

    sequence_length = config["training"].get("sequence_length", 8)

    train_dataset = VideoSequenceDataset(
        train_videos,
        sequence_length=sequence_length,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],  # recommend 4–8
        shuffle=True,
        num_workers=0
    )

    # -----------------------
    # Model
    # -----------------------
    model = ConvLSTMAutoencoder(
        input_channels=3,
        hidden_dim=config["model"].get("hidden_dim", 64)
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    # -----------------------
    # Logger
    # -----------------------
    logger = Logger(config["logging"]["log_dir"])

    # -----------------------
    # Training loop
    # -----------------------
    global_step = 0

    for epoch in range(config["training"]["epochs"]):
        epoch_loss = 0.0

        for sequences in train_loader:
            sequences = sequences.to(DEVICE)

            output = model(sequences)
            loss = criterion(output, sequences)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.log_loss(loss.item(), global_step)
            logger.log_lr(optimizer, global_step)

            global_step += 1
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}] - Loss: {avg_loss:.6f}")

        # Log reconstructed sequences (first batch only)
        logger.log_images(sequences[:, 0], output[:, 0], epoch)

    # -----------------------
    # Save model
    # -----------------------
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        model.state_dict(),
        "checkpoints/conv_lstm_autoencoder.pth"
    )

    logger.close()


if __name__ == "__main__":
    main()
