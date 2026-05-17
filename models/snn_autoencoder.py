"""
SNN Autoencoder using snnTorch for GPU-accelerated training.

Architecture mirrors the Brian2 version but uses surrogate gradient
descent for training instead of STDP.

Input:  (B, 2, 128, 128) per timestep — 2 channels (ON/OFF events)
Output: (B, 2, 128, 128) per timestep — reconstructed event frame

LIF neurons with surrogate gradient (ATan) enable backprop through spikes.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNEncoder(nn.Module):
    """Spiking convolutional encoder."""

    def __init__(self, in_channels=2, beta=0.95, bottleneck_channels=8):
        super().__init__()

        # Surrogate gradient for backprop through spikes
        spike_grad = surrogate.atan(alpha=2.0)

        # Encoder Layer 1: Conv 5×5, stride=2
        # (B, 2, 128, 128) → (B, 128, 64, 64)
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        # Encoder Layer 2: Conv 3×3, stride=2
        # (B, 128, 64, 64) → (B, 64, 32, 32)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        # Bottleneck: Conv 1×1 — small channel count forces compression
        # (B, 64, 32, 32) → (B, bottleneck_channels, 32, 32)
        self.conv3 = nn.Conv2d(64, bottleneck_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.bottleneck_channels = bottleneck_channels
    
    def forward(self, x, mem1, mem2, mem3):
        """
        Process one timestep.
        
        Args:
            x: Input tensor (B, 2, 128, 128) — one timestep
            mem1, mem2, mem3: Membrane potentials from previous timestep
            
        Returns:
            spk3: Output spikes from bottleneck (B, 16, 32, 32)
            mem1, mem2, mem3: Updated membrane potentials
        """
        # Layer 1
        cur1 = self.bn1(self.conv1(x))
        spk1, mem1 = self.lif1(cur1, mem1)
        
        # Layer 2
        cur2 = self.bn2(self.conv2(spk1))
        spk2, mem2 = self.lif2(cur2, mem2)
        
        # Bottleneck
        cur3 = self.bn3(self.conv3(spk2))
        spk3, mem3 = self.lif3(cur3, mem3)
        
        return spk3, mem1, mem2, mem3
    
    def init_mem(self, batch_size, device):
        """Initialize membrane potentials to zero."""
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        return mem1, mem2, mem3


class SNNDecoder(nn.Module):
    """Spiking transposed convolutional decoder."""

    def __init__(self, out_channels=2, beta=0.95, bottleneck_channels=8):
        super().__init__()

        spike_grad = surrogate.atan(alpha=2.0)

        # Decoder Layer 1: TransConv 3×3, stride=2
        # (B, bottleneck_channels, 32, 32) → (B, 64, 64, 64)
        self.deconv1 = nn.ConvTranspose2d(bottleneck_channels, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        # Decoder Layer 2: TransConv 5×5, stride=2
        # (B, 64, 64, 64) → (B, out_channels, 128, 128)
        self.deconv2 = nn.ConvTranspose2d(64, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
    
    def forward(self, x, mem1, mem2):
        """
        Process one timestep.
        
        Args:
            x: Input spikes from bottleneck (B, 16, 32, 32)
            mem1, mem2: Membrane potentials
            
        Returns:
            spk2: Reconstructed spikes (B, 2, 128, 128)
            mem1, mem2: Updated membrane potentials
        """
        cur1 = self.bn1(self.deconv1(x))
        spk1, mem1 = self.lif1(cur1, mem1)
        
        cur2 = self.bn2(self.deconv2(spk1))
        spk2, mem2 = self.lif2(cur2, mem2)
        
        return spk2, mem1, mem2
    
    def init_mem(self, batch_size, device):
        """Initialize membrane potentials."""
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        return mem1, mem2


class SNNAutoencoder(nn.Module):
    """
    Complete SNN Autoencoder for event-based anomaly detection.

    Uses Leaky Integrate-and-Fire (LIF) neurons with surrogate
    gradient descent for end-to-end training.

    Architecture:
        Input:  (B, 2, 128, 128) per timestep
        Enc1:   Conv2d(2→128, 5×5, s=2) + LIF → (B, 128, 64, 64)
        Enc2:   Conv2d(128→64, 3×3, s=2) + LIF → (B, 64, 32, 32)
        Bottleneck: Conv2d(64→B, 1×1) + LIF  → (B, bottleneck_channels, 32, 32)
        Dec1:   ConvTranspose2d(B→64, 3×3, s=2) + LIF → (B, 64, 64, 64)
        Dec2:   ConvTranspose2d(64→2, 5×5, s=2) + LIF → (B, 2, 128, 128)

    bottleneck_channels controls compression. Smaller = harder to reconstruct
    anomalies = higher anomaly scores. Default 8 (vs old 32) forces the model
    to only efficiently encode patterns seen in training (normal pedestrians).
    """

    def __init__(self, in_channels=2, beta=0.95, num_steps=25, bottleneck_channels=8):
        super().__init__()

        self.num_steps = num_steps
        self.bottleneck_channels = bottleneck_channels
        self.encoder = SNNEncoder(in_channels=in_channels, beta=beta,
                                  bottleneck_channels=bottleneck_channels)
        self.decoder = SNNDecoder(out_channels=in_channels, beta=beta,
                                  bottleneck_channels=bottleneck_channels)
    
    def forward(self, x):
        """
        Full forward pass over all timesteps.
        
        Args:
            x: Input spike tensor (B, C, H, W, T) where T=num_steps
               OR (B, C, H, W) — single frame replicated over T steps
               
        Returns:
            spike_out: Accumulated output spikes (B, C, H, W)
            mem_out: Final membrane potential (B, C, H, W) 
            spike_record: List of per-timestep output spikes
        """
        # Handle single-frame input (replicate over time)
        if x.dim() == 4:
            # (B, C, H, W) → treat as constant input for each timestep
            input_is_static = True
        elif x.dim() == 5:
            # (B, C, H, W, T) — different input per timestep
            input_is_static = False
            assert x.shape[-1] == self.num_steps, \
                f"Expected T={self.num_steps}, got {x.shape[-1]}"
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")
        
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize membrane potentials
        enc_mems = self.encoder.init_mem(batch_size, device)
        dec_mems = self.decoder.init_mem(batch_size, device)
        
        # Collect outputs
        spike_record = []
        mem_record = []
        
        for step in range(self.num_steps):
            # Get input for this timestep
            if input_is_static:
                x_t = x  # Same input each step
            else:
                x_t = x[:, :, :, :, step]  # (B, C, H, W)
            
            # Encode
            spk_bottleneck, *enc_mems = self.encoder(x_t, *enc_mems)
            
            # Decode
            spk_out, *dec_mems = self.decoder(spk_bottleneck, *dec_mems)
            
            spike_record.append(spk_out)
            mem_record.append(dec_mems[-1])  # Last decoder membrane
        
        # Stack: list of (B,C,H,W) → (T, B, C, H, W)
        spike_record = torch.stack(spike_record, dim=0)
        
        # Accumulated spike count over time: (B, C, H, W)
        spike_count = spike_record.sum(dim=0)
        
        # Final membrane potential
        mem_final = mem_record[-1]
        
        return spike_count, mem_final, spike_record
    
    def get_spike_rates(self, spike_record):
        """
        Compute per-layer spike rates for efficiency analysis.
        
        Args:
            spike_record: (T, B, C, H, W) output spike tensor
            
        Returns:
            dict with spike rate statistics
        """
        # Output layer spike rate
        total_possible = spike_record.numel()
        total_spikes = spike_record.sum().item()
        avg_rate = total_spikes / total_possible
        
        return {
            'output_spike_rate': avg_rate,
            'total_spikes': int(total_spikes),
            'total_possible': total_possible,
            'sparsity': 1.0 - avg_rate  # Higher = sparser = more efficient
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_default_snn(num_steps=25, beta=0.95, bottleneck_channels=8):
    """Create SNN autoencoder with default config for UCSD Ped2."""
    return SNNAutoencoder(
        in_channels=2,
        beta=beta,
        num_steps=num_steps,
        bottleneck_channels=bottleneck_channels
    )


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing SNN Autoencoder (snnTorch)...")
    print("=" * 60)
    
    model = create_default_snn(num_steps=25)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test with 5D input (B, C, H, W, T)
    x = torch.randn(4, 2, 128, 128, 25)
    spike_count, mem_final, spike_record = model(x)
    
    print(f"Input shape:        {x.shape}")
    print(f"Spike count shape:  {spike_count.shape}")
    print(f"Mem final shape:    {mem_final.shape}")
    print(f"Spike record shape: {spike_record.shape}")
    
    rates = model.get_spike_rates(spike_record)
    print(f"Output spike rate:  {rates['output_spike_rate']:.4f}")
    print(f"Output sparsity:    {rates['sparsity']:.4f}")
    
    # Test with 4D input (static frame)
    x_static = torch.randn(4, 2, 128, 128)
    spike_count2, _, _ = model(x_static)
    print(f"\nStatic input:       {x_static.shape}")
    print(f"Spike count shape:  {spike_count2.shape}")
    
    print("\n✓ All tests passed!")