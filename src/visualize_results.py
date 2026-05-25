"""
Generate publication-quality visualizations for all results.

Run all evaluate scripts first, then:
    python src/visualize_results.py

Outputs (results/figures/):
  1. auc_comparison.png       — bar chart, all 5 models
  2. roc_curves.png           — overlaid ROC curves
  3. efficiency_comparison.png — params / ops / energy subplots
  4. spike_sparsity.png       — per-layer SNN sparsity
  5. auc_vs_energy.png        — accuracy-efficiency tradeoff scatter
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc as sk_auc

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

# ---------------------------------------------------------------------------
# Known AUC results from all experiments
# ---------------------------------------------------------------------------
MODEL_RESULTS = [
    # (display_name, auc, color, family)
    ('ConvAutoencoder\n(Motion frames)', 0.8269, '#2E7D32', 'ANN'),
    ('SNN Count\n(Events)',              0.6994, '#1565C0', 'SNN'),
    ('SNN Temporal\n(Events)',           0.6558, '#1976D2', 'SNN'),
    ('SNN Rate\n(Events)',               0.6416, '#42A5F5', 'SNN'),
    ('ConvLSTM\n(RGB)',                  0.5995, '#BF360C', 'ANN'),
]

# File mapping: (scores_file, gt_file) relative to results/
SCORE_FILES = {
    'ConvAutoencoder': ('convae_anomaly_scores.npy',    'convae_ground_truth.npy'),
    'SNN Count':       ('snn_anomaly_scores.npy',       'snn_ground_truth.npy'),
    'ConvLSTM':        ('convlstm_anomaly_scores.npy',  'convlstm_ground_truth.npy'),
}

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')

FIGURE_DIR = 'results/figures'


# ---------------------------------------------------------------------------
# 1. AUC Comparison Bar Chart
# ---------------------------------------------------------------------------
def plot_auc_comparison(output_dir=FIGURE_DIR):
    os.makedirs(output_dir, exist_ok=True)

    names  = [r[0] for r in MODEL_RESULTS]
    aucs   = [r[1] for r in MODEL_RESULTS]
    colors = [r[2] for r in MODEL_RESULTS]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, aucs, color=colors, width=0.55, edgecolor='white', linewidth=0.8)

    # Value labels on each bar
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Reference line at 0.70
    ax.axhline(0.70, color='gray', linestyle='--', linewidth=1.2, alpha=0.7,
               label='AUC = 0.70 reference')

    # Legend for model families
    ann_patch = mpatches.Patch(color='#2E7D32', label='ANN (frame-based)')
    snn_patch = mpatches.Patch(color='#1565C0', label='SNN (event-based)')
    ax.legend(handles=[ann_patch, snn_patch], fontsize=10, loc='lower right')

    ax.set_ylim(0.45, 0.92)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('Anomaly Detection Performance — UCSD Ped2', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'auc_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ AUC comparison saved → {path}")


# ---------------------------------------------------------------------------
# 2. ROC Curves
# ---------------------------------------------------------------------------
def plot_roc_curves(results_dir='results', output_dir=FIGURE_DIR):
    os.makedirs(output_dir, exist_ok=True)

    roc_models = [
        ('ConvAutoencoder (Motion, AUC=0.8269)', '#2E7D32',
         'convae_anomaly_scores.npy', 'convae_ground_truth.npy'),
        ('SNN Count (AUC=0.6994)',               '#1565C0',
         'snn_anomaly_scores.npy',    'snn_ground_truth.npy'),
        ('ConvLSTM RGB (AUC=0.5995)',            '#BF360C',
         'convlstm_anomaly_scores.npy', 'convlstm_ground_truth.npy'),
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = 0

    for label, color, score_file, gt_file in roc_models:
        sp = os.path.join(results_dir, score_file)
        gp = os.path.join(results_dir, gt_file)

        if not os.path.exists(sp) or not os.path.exists(gp):
            print(f"  Skipping {label} — missing {score_file} or {gt_file}")
            continue

        scores = np.load(sp)
        y_true = np.load(gp)
        n = min(len(scores), len(y_true))
        scores, y_true = scores[:n], y_true[:n]

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.2,
                label=f'{label} — AUC={roc_auc:.4f}')
        plotted += 1

    if plotted == 0:
        print("  No ROC data found — run evaluate scripts first.")
        plt.close()
        return

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4, label='Random baseline')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — UCSD Ped2 Anomaly Detection', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ROC curves saved → {path}")


# ---------------------------------------------------------------------------
# 3. Efficiency Comparison (4 subplots)
# ---------------------------------------------------------------------------
def plot_efficiency_comparison(results_dir='results', output_dir=FIGURE_DIR):
    os.makedirs(output_dir, exist_ok=True)

    eff_path = os.path.join(results_dir, 'efficiency_analysis.json')
    if not os.path.exists(eff_path):
        print("  Skipping efficiency plot — run experiments/efficiency_analysis.py first")
        return

    with open(eff_path) as f:
        data = json.load(f)

    labels = ['ConvAutoencoder', 'ConvLSTM', 'SNN Autoencoder']
    display = ['ConvAE\n(Motion)', 'ConvLSTM\n(RGB)', 'SNN\n(Events)']
    colors  = ['#2E7D32', '#BF360C', '#1565C0']

    params   = [data[m]['parameters'] / 1e3        for m in labels]
    cpu_ms   = [data[m].get('cpu_ms', 0)           for m in labels]
    # MACs for ANN, SynOps for SNN — all in millions
    ops_M    = [
        data['ConvAutoencoder'].get('macs', 0)   / 1e6,
        data['ConvLSTM'].get('macs', 0)          / 1e6,
        data['SNN Autoencoder'].get('synops', 0) / 1e6,
    ]
    energy   = [data[m].get('energy_uJ', 0)        for m in labels]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    def _bar(ax, vals, title, ylabel, logy=False):
        bars = ax.bar(display, vals, color=colors, edgecolor='white')
        for bar, v in zip(bars, vals):
            fmt = f'{v:.1f}' if v < 1e4 else f'{v/1e3:.1f}K'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * (1.05 if not logy else 1.3),
                    fmt, ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10)
        if logy:
            ax.set_yscale('log')
        ax.tick_params(axis='x', labelsize=9)

    _bar(axes[0], params,  'Parameters',              'Thousands (K)')
    _bar(axes[1], cpu_ms,  'CPU Inference Time',      'Milliseconds (ms)', logy=True)
    _bar(axes[2], ops_M,   'MACs / SynOps',           'Millions (M)',      logy=True)
    _bar(axes[3], energy,  'Energy (GPU or Loihi)',   'μJ per inference',  logy=True)

    # Annotate ops chart to clarify ANN=MACs, SNN=SynOps
    axes[2].text(0, ops_M[0] * 0.6, 'MACs', ha='center', fontsize=8, color='white', fontweight='bold')
    axes[2].text(1, ops_M[1] * 0.6, 'MACs', ha='center', fontsize=8, color='white', fontweight='bold')
    axes[2].text(2, ops_M[2] * 0.6, 'SynOps', ha='center', fontsize=8, color='white', fontweight='bold')

    fig.suptitle('Computational Efficiency Comparison', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'efficiency_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Efficiency comparison saved → {path}")


# ---------------------------------------------------------------------------
# 4. Spike Sparsity Per Layer
# ---------------------------------------------------------------------------
def plot_spike_sparsity(results_dir='results', output_dir=FIGURE_DIR):
    os.makedirs(output_dir, exist_ok=True)

    eff_path = os.path.join(results_dir, 'efficiency_analysis.json')
    if not os.path.exists(eff_path):
        print("  Skipping sparsity plot — run experiments/efficiency_analysis.py first")
        return

    with open(eff_path) as f:
        data = json.load(f)

    sparsity = data.get('SNN Autoencoder', {}).get('layer_sparsity', {})
    if not sparsity:
        print("  No sparsity data in efficiency_analysis.json")
        return

    layer_labels = {
        'encoder.lif1':  'Enc LIF-1\n(2→128, 64×64)',
        'encoder.lif2':  'Enc LIF-2\n(128→64, 32×32)',
        'encoder.lif3':  'Enc LIF-3\n(Bottleneck 8ch)',
        'decoder.lif1':  'Dec LIF-1\n(8→64, 64×64)',
        'decoder.lif2':  'Dec LIF-2\n(64→2, 128×128)',
        'decoder_output':'Output',
    }

    layers   = [k for k in layer_labels if k in sparsity]
    names    = [layer_labels[k] for k in layers]
    active   = [sparsity[k]['activity_pct'] for k in layers]
    sparse   = [sparsity[k]['sparsity_pct']  for k in layers]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(layers))
    w = 0.38

    b1 = ax.bar(x - w/2, active, w, label='Active (%)',  color='#1565C0', alpha=0.85)
    b2 = ax.bar(x + w/2, sparse, w, label='Sparse (%)', color='#90CAF9', alpha=0.85)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8.5)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title('SNN Spike Sparsity Per Layer\n(10% event rate input, trained model)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(80, color='green', linestyle='--', linewidth=1, alpha=0.5,
               label='80% sparsity target')

    plt.tight_layout()
    path = os.path.join(output_dir, 'spike_sparsity.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Spike sparsity saved → {path}")


# ---------------------------------------------------------------------------
# 5. AUC vs Energy Tradeoff Scatter
# ---------------------------------------------------------------------------
def plot_auc_vs_energy(results_dir='results', output_dir=FIGURE_DIR):
    os.makedirs(output_dir, exist_ok=True)

    eff_path = os.path.join(results_dir, 'efficiency_analysis.json')
    if not os.path.exists(eff_path):
        print("  Skipping tradeoff plot — run experiments/efficiency_analysis.py first")
        return

    with open(eff_path) as f:
        data = json.load(f)

    points = [
        ('ConvAutoencoder\n(Motion, GPU)',  0.8269,
         data['ConvAutoencoder'].get('energy_uJ', 182871), '#2E7D32', 120),
        ('SNN Count\n(Events, Loihi)',       0.6994,
         data['SNN Autoencoder'].get('energy_uJ', 16061),  '#1565C0', 120),
        ('ConvLSTM\n(RGB, GPU)',             0.5995,
         data['ConvLSTM'].get('energy_uJ', 17673958),      '#BF360C', 120),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for label, auc_val, energy, color, ms in points:
        ax.scatter(energy, auc_val, color=color, s=ms, zorder=5, edgecolors='white', linewidth=1.5)
        offset_x = energy * 0.15
        offset_y = 0.008
        ax.annotate(label, (energy, auc_val),
                    xytext=(energy + offset_x, auc_val + offset_y),
                    fontsize=9, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color=color, lw=0.8))

    # Shaded "ideal" region
    ax.axhspan(0.75, 1.0, alpha=0.05, color='green', label='High accuracy zone')

    # Efficiency annotation arrow (SNN → ConvAE direction on energy axis)
    snn_e = data['SNN Autoencoder'].get('energy_uJ', 16061)
    ae_e  = data['ConvAutoencoder'].get('energy_uJ', 182871)
    ratio = ae_e / snn_e
    ax.annotate('', xy=(snn_e, 0.58), xytext=(ae_e, 0.58),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text((snn_e * ae_e) ** 0.5, 0.565,
            f'{ratio:.0f}× energy gap', ha='center', va='top', fontsize=9.5, color='gray')

    ax.set_xscale('log')
    ax.set_xlabel('Energy per Inference — μJ  (log scale)', fontsize=11)
    ax.set_ylabel('AUC-ROC', fontsize=11)
    ax.set_title('Accuracy–Efficiency Tradeoff\nSNN (Loihi) vs ANN (GPU)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0.52, 0.88)
    ax.grid(True, which='both', alpha=0.3)

    # Legend: hardware platform
    gpu_patch  = mpatches.Patch(color='#888888', label='ANN on GPU')
    loihi_patch = mpatches.Patch(color='#1565C0', label='SNN on Loihi (projected)')
    ax.legend(handles=[gpu_patch, loihi_patch], fontsize=9.5, loc='upper right')

    plt.tight_layout()
    path = os.path.join(output_dir, 'auc_vs_energy.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ AUC vs Energy tradeoff saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 55)
    print("GENERATING VISUALIZATIONS")
    print("=" * 55)

    plot_auc_comparison()
    plot_roc_curves()
    plot_efficiency_comparison()
    plot_spike_sparsity()
    plot_auc_vs_energy()

    print("\n" + "=" * 55)
    print(f"All figures saved to {FIGURE_DIR}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
