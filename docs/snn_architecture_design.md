Input shape: (Batch, Channels, Height, Width, Time)
Example: (16, 2, 240, 360, 100)
         # 100 time steps = 100ms at 1ms resolution
```

**Why this matters:**
- LIF neurons integrate inputs over time
- Spike patterns emerge temporally
- Rate coding needs multiple time steps to work

### **3. Architecture Refinement**

Here's my **recommended architecture** (simple but effective):
```
┌─────────────────────────────────────────────────────┐
│ INPUT: (Batch, 2, 240, 360, T=25-50)                │
│   2 channels: ON/OFF events                         │
│   T=100 time steps                                  │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ ENCODER LAYER 1                                     │
│   Conv2d + LIF: 2→128 channels, 5×5, stride=2       │
│   Output: (B, 128, 120, 180, T)                     │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ ENCODER LAYER 2                                     │
│   Conv2d + LIF: 128→64 channels, 3×3, stride=2      │
│   Output: (B, 64, 60, 90, T)                        │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ BOTTLENECK (Choose one)                             │
│                                                     │
│   Conv 1×1: 64→32 channels                          │
│   Output: (B, 32, 60, 90, T)                        │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ DECODER LAYER 1                                     │
│   ConvTranspose2d + LIF: 32→64, 3×3, stride=2       │
│   Output: (B, 128, 120, 180, T)                     │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ DECODER LAYER 2                                     │
│   ConvTranspose2d + LIF: 64→2, 5×5, stride=2        │
│   Output: (B, 2, 240, 360, T)                       │
└─────────────────────┬───────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ OUTPUT: Reconstructed event frame                   │
│   Compare with input → Anomaly score                │
└─────────────────────────────────────────────────────┘

# Spike count loss (main)
L_count = MSE(spike_count_out, spike_count_in)

# Temporal correlation loss (optional but helpful)
L_temporal = 1 - correlation(spike_train_out, spike_train_in) (ADD LATER)

# Sparsity regularization (prevent too many spikes)
L_sparse = λ * mean(spike_rate)

# Total
Loss = L_count + α * L_temporal (LATER) + β * L_sparse
