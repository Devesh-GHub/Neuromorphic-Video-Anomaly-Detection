"""
Training script for SNN Autoencoder on event data.

Usage:
    python src/train_snn.py --event_dir ./events/train --epochs 50

Loss function:
    L = MSE(rate_out, rate_in) + rate_weight * (actual_rate - target_rate)²

    where:
    - rate_out = spike_count_out / T  (normalised to [0,1])
    - rate_in  = spike_count_in  / T  (normalised to [0,1])
    - rate homeostasis term pulls the network toward target_rate (default 0.1)
      instead of collapsing to zero — critical for anomaly detection quality.
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from models.snn_autoencoder import SNNAutoencoder, count_parameters
from preprocessing.event_dataset_torch import EventSpikeDataset


def snn_loss(spike_count_out, spike_count_in, spike_record,
             num_steps, target_rate=0.1, rate_weight=1.0):
    """
    Combined loss for SNN autoencoder.

    Args:
        spike_count_out: Accumulated output spikes (B, C, H, W)
        spike_count_in:  Input spike counts (B, C, H, W) — sum over T
        spike_record:    Per-timestep spikes (T, B, C, H, W)
        num_steps:       T — used to normalise counts to rates
        target_rate:     Desired mean firing rate (0–1). Default 0.1 = 10%.
                         Prevents the network from collapsing to zero output.
        rate_weight:     Weight on the homeostasis term vs. reconstruction MSE.

    Returns:
        total_loss, mse_loss, rate_loss
    """
    # Normalise to firing rates so MSE is scale-invariant regardless of T
    rate_out = spike_count_out / num_steps  # (B, C, H, W), values in [0, 1]
    rate_in  = spike_count_in  / num_steps
    mse_loss = F.mse_loss(rate_out, rate_in)

    # Rate homeostasis: penalise deviation from target_rate.
    # Unlike a plain sparsity penalty (which rewards zero), this steers the
    # network toward a healthy firing rate — essential for anomaly detection.
    actual_rate = spike_record.mean()
    rate_loss = rate_weight * (actual_rate - target_rate) ** 2

    total_loss = mse_loss + rate_loss

    return total_loss, mse_loss, rate_loss


def train_one_epoch(model, dataloader, optimizer, device,
                    num_steps, target_rate=0.1, rate_weight=1.0):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_rate_loss = 0.0
    n_batches = 0

    for batch_idx, spikes in enumerate(dataloader):
        # spikes shape: (B, 2, H, W, T)
        spikes = spikes.to(device)

        # Input spike count (target for reconstruction)
        spike_count_in = spikes.sum(dim=-1)  # (B, 2, H, W)

        # Forward pass
        spike_count_out, mem_final, spike_record = model(spikes)

        # Compute loss
        loss, mse, rate_loss = snn_loss(
            spike_count_out, spike_count_in, spike_record,
            num_steps=num_steps,
            target_rate=target_rate,
            rate_weight=rate_weight
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (important for SNN stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_rate_loss += rate_loss.item()
        n_batches += 1

        if batch_idx % 10 == 0:
            actual_rate = spike_record.mean().item()
            print(f"  Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.6f} | "
                  f"MSE: {mse.item():.6f} | "
                  f"RateLoss: {rate_loss.item():.6f} | "
                  f"ActualRate: {actual_rate:.4f}")

    return {
        'loss':      total_loss      / n_batches,
        'mse':       total_mse       / n_batches,
        'rate_loss': total_rate_loss / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train SNN Autoencoder')
    parser.add_argument('--event_dir', type=str, default='./events/train',
                        help='Directory with training event files')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_steps', type=int, default=25,
                        help='SNN timesteps per sample')
    parser.add_argument('--beta', type=float, default=0.95,
                        help='LIF membrane decay rate')
    parser.add_argument('--target_rate', type=float, default=0.1,
                        help='Target firing rate for homeostasis (0–1). '
                             '0.1 = 10%% — prevents zero-output collapse.')
    parser.add_argument('--rate_weight', type=float, default=1.0,
                        help='Weight on rate homeostasis loss vs. MSE.')
    parser.add_argument('--encoding', type=str, default='rate',
                        choices=['rate', 'temporal', 'count'])
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--image_size', type=int, default=128)
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset
    print("\nLoading event dataset...")
    dataset = EventSpikeDataset(
        event_dir=args.event_dir,
        target_size=(args.image_size, args.image_size),
        num_steps=args.num_steps,
        encoding=args.encoding
    )
    
    num_workers = 0 if sys.platform == 'win32' else 2
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    
    # Model
    print("\nCreating SNN Autoencoder...")
    model = SNNAutoencoder(
        in_channels=2,
        beta=args.beta,
        num_steps=args.num_steps
    ).to(device)
    
    print(f"Parameters: {count_parameters(model):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    best_loss = float('inf')
    loss_history = []
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        metrics = train_one_epoch(
            model, dataloader, optimizer, device,
            num_steps=args.num_steps,
            target_rate=args.target_rate,
            rate_weight=args.rate_weight
        )
        
        scheduler.step()
        
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch [{epoch+1}/{args.epochs}] ({elapsed:.1f}s)")
        print(f"  Loss: {metrics['loss']:.6f} | "
              f"MSE: {metrics['mse']:.6f} | "
              f"RateLoss: {metrics['rate_loss']:.6f} | "
              f"LR: {current_lr:.6f}")
        
        loss_history.append(metrics)
        
        # Save best model
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'args': vars(args)
            }, os.path.join(args.checkpoint_dir, 'snn_autoencoder_best.pth'))
            print(f"  ✓ Best model saved (loss={best_loss:.6f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': metrics['loss'],
                'args': vars(args)
            }, os.path.join(args.checkpoint_dir, f'snn_autoencoder_epoch{epoch+1}.pth'))
    
    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join(args.checkpoint_dir, 'snn_autoencoder.pth')
    )
    
    # Save loss history
    np.save(
        os.path.join(args.checkpoint_dir, 'snn_loss_history.npy'),
        [{'epoch': i, **m} for i, m in enumerate(loss_history)],
        allow_pickle=True
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoint: {args.checkpoint_dir}/snn_autoencoder_best.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()