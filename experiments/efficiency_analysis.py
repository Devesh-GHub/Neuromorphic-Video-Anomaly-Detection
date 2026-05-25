"""
Computational Efficiency Analysis: SNN vs ConvAE vs ConvLSTM

Measures:
  1. Parameter count
  2. Inference time (CPU, avg over 50 runs)
  3. Peak GPU memory
  4. MACs (ANN) / SynOps (SNN)
  5. Spike sparsity per layer
  6. Energy projection on neuromorphic hardware vs GPU

The "why neuromorphic matters" argument:
  SNNs trade ~13% AUC for 18x energy reduction on dedicated hardware.
  ConvLSTM is both the worst in AUC AND the most expensive to run.

Usage:
    python experiments/efficiency_analysis.py
    python experiments/efficiency_analysis.py --checkpoint checkpoints/snn_count/snn_autoencoder_best.pth
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from models.snn_autoencoder import SNNAutoencoder
from models.conv_lstm_autoencoder import ConvLSTMAutoencoder
from models.conv_autoencoder import ConvAutoencoder


# ---------------------------------------------------------------------------
# Measured AUC results from our experiments
# ---------------------------------------------------------------------------
AUC_RESULTS = {
    'ConvAutoencoder':  0.8269,   # motion frames, batch=8, 50 epochs
    'SNN Count':        0.6994,   # event spikes, count encoding
    'SNN Temporal':     0.6558,
    'SNN Rate':         0.6416,
    'ConvLSTM RGB':     0.5995,   # raw RGB sequences
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, input_tensor, n_runs=50, warmup=10):
    model.eval()
    model.cpu()
    x = input_tensor.cpu()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter() - t0) * 1000)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms':  float(np.std(times)),
        'min_ms':  float(np.min(times)),
    }


def measure_gpu_memory(model, input_tensor):
    if not torch.cuda.is_available():
        return None
    model.cuda()
    x = input_tensor.cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(x)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    model.cpu()
    return round(peak_mb, 1)


# ---------------------------------------------------------------------------
# MAC / SynOp calculators
# ---------------------------------------------------------------------------

def conv_macs(in_c, out_c, kH, kW, out_H, out_W):
    """MACs for one Conv2d layer (= in_c × kH × kW × out_c × out_H × out_W)."""
    return in_c * kH * kW * out_c * out_H * out_W


def convT_macs(in_c, out_c, kH, kW, out_H, out_W):
    """Approximate MACs for ConvTranspose2d (same formula, output spatial)."""
    return in_c * kH * kW * out_c * out_H * out_W


def compute_convae_macs():
    """
    Updated ConvAutoencoder architecture (no skip connections, bottleneck=32):
      enc1:  Conv2d(3→32,  3×3) @ 128×128
      enc2:  Conv2d(32→64, 3×3) @ 64×64   (after pool)
      enc3:  Conv2d(64→128,3×3) @ 32×32   (after pool)
      bottleneck: Conv2d(128→32,3×3) @ 16×16 (after pool)
      up3:   ConvTranspose2d(32→128,2×2) → 32×32
      dec3:  Conv2d(128→128,3×3) @ 32×32
      up2:   ConvTranspose2d(128→64,2×2) → 64×64
      dec2:  Conv2d(64→64, 3×3) @ 64×64
      up1:   ConvTranspose2d(64→32, 2×2) → 128×128
      dec1:  Conv2d(32→32, 3×3) @ 128×128
      final: Conv2d(32→3,  1×1) @ 128×128
    """
    macs = 0
    macs += conv_macs(3,   32,  3, 3, 128, 128)   # enc1
    macs += conv_macs(32,  64,  3, 3,  64,  64)   # enc2
    macs += conv_macs(64,  128, 3, 3,  32,  32)   # enc3
    macs += conv_macs(128, 32,  3, 3,  16,  16)   # bottleneck (128→32!)
    macs += convT_macs(32, 128, 2, 2,  32,  32)   # up3
    macs += conv_macs(128, 128, 3, 3,  32,  32)   # dec3
    macs += convT_macs(128, 64, 2, 2,  64,  64)   # up2
    macs += conv_macs(64,  64,  3, 3,  64,  64)   # dec2
    macs += convT_macs(64, 32,  2, 2, 128, 128)   # up1
    macs += conv_macs(32,  32,  3, 3, 128, 128)   # dec1
    macs += conv_macs(32,  3,   1, 1, 128, 128)   # final
    return macs


def compute_convlstm_macs(T=12, hidden_dim=64):
    """
    ConvLSTMCell runs at FULL spatial resolution (128×128).
    Cell conv: (in_dim + hidden_dim) → 4*hidden_dim, kernel 3×3.
      Encoder: in_dim=3,          T steps
      Decoder: in_dim=hidden_dim, T steps
    Output layer: Conv2d(hidden_dim→3, 1×1) × T
    """
    enc_cell_macs = conv_macs(3 + hidden_dim, 4 * hidden_dim, 3, 3, 128, 128)
    dec_cell_macs = conv_macs(hidden_dim + hidden_dim, 4 * hidden_dim, 3, 3, 128, 128)
    out_macs      = conv_macs(hidden_dim, 3, 1, 1, 128, 128)
    total = (enc_cell_macs + dec_cell_macs + out_macs) * T
    return total


def compute_snn_equivalent_macs():
    """
    SNN architecture (bottleneck=8, stride convolutions):
      enc_conv1: Conv2d(2→128,  5×5, s=2)  @ output 64×64
      enc_conv2: Conv2d(128→64, 3×3, s=2)  @ output 32×32
      bottleneck: Conv2d(64→8,  1×1)       @ 32×32
      dec_deconv1: ConvTranspose2d(8→64,  3×3, s=2) @ output 64×64
      dec_deconv2: ConvTranspose2d(64→2,  5×5, s=2) @ output 128×128

    Equivalent MACs if this were an ANN (before applying spike rate).
    """
    macs = 0
    macs += conv_macs(2,   128, 5, 5,  64,  64)   # enc_conv1  (output 64×64)
    macs += conv_macs(128,  64, 3, 3,  32,  32)   # enc_conv2  (output 32×32)
    macs += conv_macs(64,    8, 1, 1,  32,  32)   # bottleneck
    macs += convT_macs(8,   64, 3, 3,  64,  64)   # dec_deconv1
    macs += convT_macs(64,   2, 5, 5, 128, 128)   # dec_deconv2
    return macs


# ---------------------------------------------------------------------------
# Spike sparsity analysis
# ---------------------------------------------------------------------------

def analyze_snn_sparsity(model, input_tensor):
    """
    Capture per-layer spike rates via forward hooks on LIF neurons.
    input_tensor should be a realistic sparse binary event tensor.
    """
    model.eval()
    layer_spikes = {}

    def make_hook(name):
        def hook(module, _, output):
            spk = output[0] if isinstance(output, tuple) else output
            layer_spikes[name] = spk.detach().cpu()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if 'lif' in name.lower():
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        spike_count, _, spike_record = model(input_tensor.cpu())

    for h in hooks:
        h.remove()

    results = {}
    for name, spikes in layer_spikes.items():
        total  = spikes.numel()
        active = int((spikes > 0).sum().item())
        rate   = active / total if total > 0 else 0.0
        results[name] = {
            'spike_rate':    round(rate, 4),
            'activity_pct':  round(rate * 100, 2),
            'sparsity_pct':  round((1 - rate) * 100, 2),
            'active_neurons': active,
            'total_neurons':  total,
        }

    # Output layer over all timesteps
    out_rate = spike_record.float().mean().item()
    results['decoder_output'] = {
        'spike_rate':   round(out_rate, 4),
        'activity_pct': round(out_rate * 100, 2),
        'sparsity_pct': round((1 - out_rate) * 100, 2),
    }

    return results, spike_record


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained SNN checkpoint for realistic spike rates')
    args = parser.parse_args()

    print("=" * 70)
    print("COMPUTATIONAL EFFICIENCY ANALYSIS")
    print("SNN (neuromorphic) vs ConvAutoencoder vs ConvLSTM")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Instantiate models (match exact configs used in training)
    # ------------------------------------------------------------------
    snn = SNNAutoencoder(in_channels=2, beta=0.95, num_steps=25, bottleneck_channels=8)
    conv_ae   = ConvAutoencoder()
    conv_lstm = ConvLSTMAutoencoder(input_channels=3, hidden_dim=64)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        snn.load_state_dict(state)
        print(f"  Loaded SNN checkpoint: {args.checkpoint}")
    else:
        print("  Note: using untrained SNN weights — spike rates reflect init bias.")
        print("  Pass --checkpoint for realistic rates from trained model.")

    # ------------------------------------------------------------------
    # Input tensors (match training shapes exactly)
    # ------------------------------------------------------------------

    # SNN: realistic sparse binary events — UCSD Ped2 events are ~10% active
    # (frame differencing with threshold produces ~5-15% active pixels)
    EVENT_RATE = 0.10
    snn_input  = (torch.rand(1, 2, 128, 128, 25) < EVENT_RATE).float()

    # ConvAE: motion frame (absolute frame difference) — dense 3-channel
    conv_input = torch.rand(1, 3, 128, 128)

    # ConvLSTM: RGB sequences, T=12 as used in training config
    lstm_input = torch.randn(1, 12, 3, 128, 128)

    results = {}

    # ------------------------------------------------------------------
    # 1. Parameter count
    # ------------------------------------------------------------------
    print("\n1. PARAMETER COUNT")
    print("-" * 50)
    for name, model in [('SNN Autoencoder', snn),
                         ('ConvAutoencoder', conv_ae),
                         ('ConvLSTM',        conv_lstm)]:
        n = count_params(model)
        results.setdefault(name, {})['parameters'] = n
        print(f"  {name:<22}: {n:>10,} params")

    # ------------------------------------------------------------------
    # 2. CPU inference time
    # ------------------------------------------------------------------
    print("\n2. CPU INFERENCE TIME  (avg over 50 runs, batch=1)")
    print("-" * 50)
    for name, model, inp in [
        ('SNN Autoencoder', snn,       snn_input),
        ('ConvAutoencoder', conv_ae,   conv_input),
        ('ConvLSTM',        conv_lstm, lstm_input),
    ]:
        t = measure_inference_time(model, inp)
        results[name]['cpu_ms']     = t['mean_ms']
        results[name]['cpu_std_ms'] = t['std_ms']
        print(f"  {name:<22}: {t['mean_ms']:>7.1f} ± {t['std_ms']:.1f} ms")

    # ------------------------------------------------------------------
    # 3. Peak GPU memory
    # ------------------------------------------------------------------
    print("\n3. PEAK GPU MEMORY")
    print("-" * 50)
    for name, model, inp in [
        ('SNN Autoencoder', snn,       snn_input),
        ('ConvAutoencoder', conv_ae,   conv_input),
        ('ConvLSTM',        conv_lstm, lstm_input),
    ]:
        mb = measure_gpu_memory(model, inp)
        results[name]['gpu_peak_mb'] = mb
        label = f"{mb:.1f} MB" if mb is not None else "N/A (no GPU)"
        print(f"  {name:<22}: {label}")

    # ------------------------------------------------------------------
    # 4. Input sparsity
    # ------------------------------------------------------------------
    print("\n4. INPUT SPARSITY")
    print("-" * 50)
    snn_input_sparsity  = 1.0 - snn_input.mean().item()
    conv_input_sparsity = 0.0   # dense RGB / motion frames

    print(f"  SNN   event input : {snn_input_sparsity*100:.1f}% zeros "
          f"  ({EVENT_RATE*100:.0f}% event rate — realistic UCSD Ped2)")
    print(f"  ConvAE motion frame: {conv_input_sparsity*100:.1f}% zeros  (dense)")
    print(f"  ConvLSTM RGB frame : {conv_input_sparsity*100:.1f}% zeros  (dense)")

    results['SNN Autoencoder']['input_sparsity_pct'] = round(snn_input_sparsity * 100, 1)
    results['ConvAutoencoder']['input_sparsity_pct'] = 0.0
    results['ConvLSTM']['input_sparsity_pct']        = 0.0

    # ------------------------------------------------------------------
    # 5. SNN per-layer spike sparsity
    # ------------------------------------------------------------------
    print("\n5. SNN SPIKE SPARSITY PER LAYER  (realistic 10% event input)")
    print("-" * 50)
    sparsity_stats, _ = analyze_snn_sparsity(snn, snn_input)

    total_active   = 0
    total_possible = 0
    for layer_name, stats in sparsity_stats.items():
        print(f"  {layer_name:<35}: {stats['activity_pct']:>5.2f}% active  "
              f"({stats['sparsity_pct']:.2f}% sparse)")
        if 'active_neurons' in stats:
            total_active   += stats['active_neurons']
            total_possible += stats['total_neurons']

    overall_sparsity = 1 - (total_active / max(total_possible, 1))
    print(f"\n  Overall internal sparsity: {overall_sparsity*100:.1f}%")
    results['SNN Autoencoder']['layer_sparsity'] = sparsity_stats
    results['SNN Autoencoder']['overall_internal_sparsity_pct'] = round(overall_sparsity * 100, 1)

    # Average spike rate for SynOps calculation
    avg_spike_rate = 1.0 - overall_sparsity

    # ------------------------------------------------------------------
    # 6. Operations: MACs (ANN) vs SynOps (SNN)
    # ------------------------------------------------------------------
    print("\n6. COMPUTATIONAL OPERATIONS")
    print("-" * 50)

    # ConvAutoencoder MACs (updated architecture: bottleneck=32, no skip)
    convae_macs = compute_convae_macs()

    # ConvLSTM MACs (T=12, hidden=64, full 128×128 spatial — very expensive)
    convlstm_macs = compute_convlstm_macs(T=12, hidden_dim=64)

    # SNN: equivalent MACs × spike rate × T timesteps
    snn_equiv_macs = compute_snn_equivalent_macs()
    snn_synops     = avg_spike_rate * snn_equiv_macs * 25   # per-timestep × T

    print(f"  ConvAutoencoder  MACs  : {convae_macs/1e6:>9.1f} M")
    print(f"  ConvLSTM         MACs  : {convlstm_macs/1e9:>9.2f} B  "
          f"({convlstm_macs/convae_macs:.0f}x ConvAE)")
    print(f"  SNN equiv MACs/step    : {snn_equiv_macs/1e6:>9.1f} M  × {25} steps")
    print(f"  SNN SynOps (×{avg_spike_rate:.2f} rate): {snn_synops/1e6:>9.1f} M  "
          f"({snn_synops/convae_macs*100:.1f}% of ConvAE MACs)")

    results['ConvAutoencoder']['macs']           = int(convae_macs)
    results['ConvLSTM']['macs']                  = int(convlstm_macs)
    results['SNN Autoencoder']['synops']         = int(snn_synops)
    results['SNN Autoencoder']['equiv_macs']     = int(snn_equiv_macs)
    results['SNN Autoencoder']['avg_spike_rate'] = round(avg_spike_rate, 4)

    # ------------------------------------------------------------------
    # 7. Energy projection
    # ------------------------------------------------------------------
    print("\n7. ENERGY PROJECTION")
    print("-" * 50)
    print("  Hardware references:")
    print("    GPU 7nm  MAC  : ~200 pJ  (Horowitz, 2014 / industry estimates)")
    print("    Intel Loihi  : ~23 pJ/SynOp  (Davies et al., 2018)")
    print()

    pJ_per_mac_gpu    = 200
    pJ_per_synop_loihi = 23

    convae_energy_uJ   = (convae_macs   * pJ_per_mac_gpu)    / 1e6
    convlstm_energy_uJ = (convlstm_macs * pJ_per_mac_gpu)    / 1e6
    snn_energy_uJ      = (snn_synops    * pJ_per_synop_loihi) / 1e6

    print(f"  ConvAutoencoder  (GPU)   : {convae_energy_uJ:>10.1f} μJ")
    print(f"  ConvLSTM         (GPU)   : {convlstm_energy_uJ:>10.1f} μJ  "
          f"({convlstm_energy_uJ/convae_energy_uJ:.0f}x ConvAE)")
    print(f"  SNN              (Loihi) : {snn_energy_uJ:>10.2f} μJ  "
          f"({convae_energy_uJ/snn_energy_uJ:.1f}x more efficient than ConvAE GPU)")

    results['ConvAutoencoder']['energy_uJ']  = round(convae_energy_uJ, 1)
    results['ConvLSTM']['energy_uJ']         = round(convlstm_energy_uJ, 1)
    results['SNN Autoencoder']['energy_uJ']  = round(snn_energy_uJ, 3)

    # ------------------------------------------------------------------
    # 8. AUC vs Efficiency tradeoff
    # ------------------------------------------------------------------
    print("\n8. AUC vs EFFICIENCY TRADEOFF")
    print("-" * 50)
    auc_convae  = AUC_RESULTS['ConvAutoencoder']
    auc_snn     = AUC_RESULTS['SNN Count']
    auc_lstm    = AUC_RESULTS['ConvLSTM RGB']

    # AUC per log10(energy) — higher is better
    def efficiency_score(auc, energy_uJ):
        return auc / np.log10(max(energy_uJ, 1e-6))

    print(f"  {'Model':<22}  {'AUC':>6}  {'Energy (μJ)':>12}  {'AUC/log₁₀E':>12}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*12}  {'-'*12}")

    for label, auc, energy in [
        ('SNN Count (Loihi)',   auc_snn,   snn_energy_uJ),
        ('ConvAutoencoder GPU', auc_convae, convae_energy_uJ),
        ('ConvLSTM GPU',        auc_lstm,  convlstm_energy_uJ),
    ]:
        score = efficiency_score(auc, energy)
        print(f"  {label:<22}  {auc:>6.4f}  {energy:>12.2f}  {score:>12.4f}")

    print(f"\n  AUC gap  ConvAE → SNN : {(auc_convae - auc_snn)*100:.1f} percentage points")
    print(f"  Energy   ConvAE → SNN : {convae_energy_uJ/snn_energy_uJ:.1f}x reduction on Loihi")
    print(f"  Tradeoff: SNN sacrifices {(auc_convae-auc_snn)*100:.1f}% AUC for "
          f"{convae_energy_uJ/snn_energy_uJ:.0f}x energy reduction")

    # ------------------------------------------------------------------
    # 9. Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"  {'Metric':<30} {'ConvAE':>12} {'ConvLSTM':>12} {'SNN':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'AUC-ROC':<30} {auc_convae:>12.4f} {auc_lstm:>12.4f} {auc_snn:>12.4f}")
    print(f"  {'Parameters':<30} "
          f"{results['ConvAutoencoder']['parameters']:>12,} "
          f"{results['ConvLSTM']['parameters']:>12,} "
          f"{results['SNN Autoencoder']['parameters']:>12,}")
    print(f"  {'CPU inference (ms)':<30} "
          f"{results['ConvAutoencoder']['cpu_ms']:>12.1f} "
          f"{results['ConvLSTM']['cpu_ms']:>12.1f} "
          f"{results['SNN Autoencoder']['cpu_ms']:>12.1f}")
    print(f"  {'MACs / SynOps (M)':<30} "
          f"{convae_macs/1e6:>12.1f} "
          f"{convlstm_macs/1e6:>12.1f} "
          f"{snn_synops/1e6:>12.1f}")
    print(f"  {'Energy μJ (GPU/Loihi)':<30} "
          f"{convae_energy_uJ:>12.1f} "
          f"{convlstm_energy_uJ:>12.1f} "
          f"{snn_energy_uJ:>12.2f}")
    print(f"  {'Input sparsity':<30} "
          f"{'0%':>12} {'0%':>12} "
          f"{snn_input_sparsity*100:>11.1f}%")
    print(f"  {'Internal spike sparsity':<30} "
          f"{'N/A':>12} {'N/A':>12} "
          f"{overall_sparsity*100:>11.1f}%")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    def _convert(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return obj

    os.makedirs("results", exist_ok=True)
    out_path = "results/efficiency_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
