# Neuromorphic Video Anomaly Detection
### Comparing SNN, ConvAutoencoder, and ConvLSTM on UCSD Ped2

> A systematic comparison of spiking neural networks (neuromorphic computing)
> against conventional deep learning baselines for video anomaly detection —
> evaluated on both accuracy and computational efficiency.

---

## Results

| Model | Input | AUC-ROC | Parameters | Energy (μJ) |
|---|---|---|---|---|
| **ConvAutoencoder** | Motion frames | **0.8269** | 949 K | 182,872 (GPU) |
| SNN Count | Event spikes | 0.6994 | **89 K** | **16,062 (Loihi)** |
| SNN Temporal | Event spikes | 0.6558 | 89 K | — |
| SNN Rate | Event spikes | 0.6416 | 89 K | — |
| ConvLSTM | Raw RGB | 0.5995 | 450 K | 17,673,958 (GPU) |

**Key finding:** SNN sacrifices 12.7% AUC to achieve **11.4× lower energy** on
Intel Loihi neuromorphic hardware — enabling always-on edge deployment.
ConvLSTM performs worst on every metric (accuracy, speed, energy).

---

## Architecture Overview

```
                    UCSD Ped2 Dataset
                          │
          ┌───────────────┼───────────────┐
          │               │               │
     Raw RGB          Frame Diff      Event Spikes
     (T frames)      (|f_t - f_{t-1}|) (ON/OFF channels)
          │               │               │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌────▼──────┐
    │ ConvLSTM  │   │  ConvAE   │   │    SNN    │
    │ (seq AE)  │   │ (frame AE)│   │ (LIF AE)  │
    └─────┬─────┘   └─────┬─────┘   └────┬──────┘
          │               │               │
     AUC 0.5995      AUC 0.8269      AUC 0.6994
     GPU heavy        Best AUC       10.6× smaller
                                     11.4× efficient
```

---

## Project Structure

```
project/
├── run_pipeline.py                  ← One-command end-to-end runner
│
├── models/
│   ├── snn_autoencoder.py           ← LIF spiking autoencoder (snnTorch)
│   ├── conv_autoencoder.py          ← ConvAE, no skip connections, bottleneck=32
│   └── conv_lstm_autoencoder.py     ← ConvLSTM autoencoder
│
├── preprocessing/
│   ├── motion_dataset.py            ← Absolute frame difference dataset
│   ├── event_dataset_torch.py       ← Spike encoding (rate/temporal/count)
│   ├── video_dataset.py             ← Raw frame dataset
│   └── sequence_dataset.py          ← Sequence dataset for ConvLSTM
│
├── src/
│   ├── train_snn.py                 ← SNN training (rate homeostasis loss)
│   ├── train_conv_autoencoder.py    ← ConvAE training on motion frames
│   ├── train.py                     ← ConvLSTM training
│   ├── evaluate_snn.py              ← SNN evaluation + AUC
│   ├── evaluate_conv_autoencoder.py ← ConvAE evaluation + AUC
│   ├── evaluate_convlstm.py         ← ConvLSTM evaluation + AUC
│   └── visualize_results.py         ← All publication figures
│
├── experiments/
│   └── efficiency_analysis.py       ← MACs, SynOps, energy projection
│
├── configs/
│   └── default.yaml                 ← Shared hyperparameters
│
└── results/
    ├── figures/                     ← PNG plots (ROC, efficiency, sparsity)
    └── efficiency_analysis.json     ← Detailed efficiency metrics
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision snnTorch scikit-learn matplotlib numpy PyYAML
```

### 2. Download dataset

Download [UCSD Anomaly Detection Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html)
and extract to `./data/UCSD_Anomaly_Dataset.v1p2/`.

### 3. Run the full pipeline

```bash
# Full pipeline (event generation → training → evaluation → visualization)
python run_pipeline.py

# Skip training (use saved checkpoints)
python run_pipeline.py --skip_training

# SNN experiments only
python run_pipeline.py --snn_only

# See what would run without executing
python run_pipeline.py --dry_run
```

---

## Running Individual Scripts

### Train

```bash
# SNN with different spike encodings
python src/train_snn.py --encoding count    --checkpoint_dir checkpoints/snn_count
python src/train_snn.py --encoding temporal --checkpoint_dir checkpoints/snn_temporal
python src/train_snn.py --encoding rate     --checkpoint_dir checkpoints/snn_rate

# ConvAutoencoder (motion frames)
python src/train_conv_autoencoder.py

# ConvLSTM (RGB sequences)
python src/train.py
```

### Evaluate

```bash
python src/evaluate_snn.py \
    --checkpoint checkpoints/snn_count/snn_autoencoder_best.pth \
    --encoding count

python src/evaluate_conv_autoencoder.py
python src/evaluate_convlstm.py
```

### Efficiency Analysis

```bash
python experiments/efficiency_analysis.py \
    --checkpoint checkpoints/snn_count/snn_autoencoder_best.pth
```

### Visualize

```bash
python src/visualize_results.py
# → results/figures/{auc_comparison, roc_curves, efficiency_comparison,
#                    spike_sparsity, auc_vs_energy}.png
```

---

## Model Details

### SNN Autoencoder
- **Neurons:** Leaky Integrate-and-Fire (LIF) with ATan surrogate gradient
- **Architecture:** Conv(2→128, 5×5, s=2) → Conv(128→64, 3×3, s=2) → Bottleneck(64→8) → Deconv × 2
- **Input:** Event spikes `(B, 2, H, W, T)` — ON/OFF channels × 25 timesteps
- **Loss:** MSE + rate homeostasis regularization
- **Key stat:** 83.9% internal spike sparsity → maps to hardware efficiency

### ConvAutoencoder
- **Architecture:** 3-level encoder → bottleneck (32 ch @ 16×16) → 3-level decoder
- **No skip connections** — forces information through bottleneck, critical for anomaly detection
- **Input:** Absolute frame differences `|f_t - f_{t-1}|` — captures motion only
- **Why motion?** Background dominates raw RGB; frame diff isolates anomalous motion

### ConvLSTM Autoencoder
- **Architecture:** ConvLSTMCell (3→64 hidden) encoder + decoder at full 128×128 resolution
- **Input:** RGB sequences `(B, T=12, 3, 128, 128)`
- **Why it fails:** 88B MACs per inference; raw RGB contains redundant background

---

## Efficiency Analysis

| Metric | ConvAE (GPU) | ConvLSTM (GPU) | SNN (Loihi) |
|---|---|---|---|
| Parameters | 949 K | 450 K | **89 K** |
| MACs / SynOps | 914 M | 88,370 M | 698 M |
| CPU inference | 42 ms | 2,211 ms | 515 ms* |
| Peak GPU memory | 22 MB | 118 MB | 27 MB |
| Energy/inference | 182,872 μJ | 17,673,958 μJ | **16,062 μJ** |
| Input sparsity | 0% | 0% | **90%** |
| Internal sparsity | N/A | N/A | **83.9%** |

*SNN CPU timing reflects PyTorch timestep loop overhead — on dedicated
neuromorphic hardware (Loihi) this runs in parallel, yielding the 11.4× energy advantage.

Energy estimates: 200 pJ/MAC (GPU 7nm, Horowitz 2014), 23 pJ/SynOp (Loihi, Davies 2018).

---

## Spike Encoding Methods

| Encoding | Description | SNN AUC |
|---|---|---|
| **Count** | Binary spike if pixel intensity > threshold | **0.6994** |
| **Temporal** | Spike time inversely proportional to intensity | 0.6558 |
| **Rate** | Poisson spike trains at rate proportional to intensity | 0.6416 |

---

## Environment

- Python 3.10+
- PyTorch 2.0+
- snnTorch 0.9+
- CUDA 11.8+ (for GPU training)
- Tested on Google Colab T4 GPU
