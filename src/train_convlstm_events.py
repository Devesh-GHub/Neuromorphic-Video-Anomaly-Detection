"""
Train ConvLSTM Autoencoder on event spike data.

Fair comparison with SNN: same EventSpikeDataset, same image size, same timesteps.

Speed optimisations (over a naive implementation):
  - torch.compile()       — fuses ops, reduces Python overhead (~20-50% faster)
  - cudnn.benchmark       — picks fastest cuDNN conv algorithm per shape
  - AMP (FP16)            — halves memory, enables larger batch on free-tier T4
  - persistent_workers    — keeps DataLoader workers alive between epochs
  - prefetch_factor=2     — overlaps data loading with GPU compute

Usage:
    python src/train_convlstm_events.py --event_dir ./events/train --epochs 50
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from models.conv_lstm_autoencoder import ConvLSTMAutoencoder
from preprocessing.event_dataset_torch import EventSpikeDataset


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, use_amp):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, spikes in enumerate(dataloader):
        # (B, 2, H, W, T) → (B, T, 2, H, W)
        x = spikes.permute(0, 4, 1, 2, 3).contiguous().cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            output = model(x)
            loss = criterion(output, x)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx:>4d}/{len(dataloader)} | "
                  f"Loss: {loss.item():.6f}")

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description='Train ConvLSTM on event data')
    parser.add_argument('--event_dir', type=str, default='./events/train')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Reduce to 4 if OOM on Colab free tier')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_steps', type=int, default=25,
                        help='Keep same as SNN for fair comparison')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='ConvLSTM hidden channels')
    parser.add_argument('--encoding', type=str, default='count',
                        choices=['rate', 'temporal', 'count'])
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/convlstm')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Device: {device}  |  AMP: {use_amp}")

    # Fastest cuDNN kernel per input shape
    torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------ dataset
    print("\nLoading event dataset...")
    dataset = EventSpikeDataset(
        event_dir=args.event_dir,
        target_size=(args.image_size, args.image_size),
        num_steps=args.num_steps,
        encoding=args.encoding
    )

    use_workers = sys.platform != 'win32'
    num_workers = 2 if use_workers else 0
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_amp,
        drop_last=True,
        persistent_workers=use_workers,   # keep workers alive between epochs
        prefetch_factor=2 if use_workers else None,  # overlap IO with compute
    )

    # ------------------------------------------------------------------ model
    print("\nCreating ConvLSTM Autoencoder (2-channel event input)...")
    model = ConvLSTMAutoencoder(
        input_channels=2,
        hidden_dim=args.hidden_dim
    ).to(device)

    # torch.compile fuses loops and reduces Python dispatch overhead
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("  torch.compile() enabled")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")
    print(f"Encoding: {args.encoding}  |  T={args.num_steps}  |  "
          f"hidden={args.hidden_dim}  |  {args.image_size}x{args.image_size}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float('inf')

    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(args.epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, criterion, scaler, use_amp
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch+1}/{args.epochs}] ({elapsed:.1f}s) | "
              f"Loss: {avg_loss:.6f} | LR: {lr:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save underlying module when model is compiled
            state = model._orig_mod.state_dict() \
                if hasattr(model, '_orig_mod') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'args': vars(args)
            }, os.path.join(args.checkpoint_dir, 'convlstm_best.pth'))
            print(f"  ✓ Best model saved (loss={best_loss:.6f})")

        if (epoch + 1) % 10 == 0:
            state = model._orig_mod.state_dict() \
                if hasattr(model, '_orig_mod') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state,
                'loss': avg_loss,
            }, os.path.join(args.checkpoint_dir, f'convlstm_epoch{epoch+1}.pth'))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoint: {args.checkpoint_dir}/convlstm_best.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
