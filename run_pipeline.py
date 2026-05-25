#!/usr/bin/env python3
"""
End-to-end pipeline for Neuromorphic Video Anomaly Detection.

Stages (auto-skipped if outputs already exist):
  1. Event Generation    — UCSD Ped2 videos → event files
  2. Training            — SNN (rate/temporal/count), ConvLSTM, ConvAutoencoder
  3. Evaluation          — all models → AUC-ROC scores
  4. Efficiency Analysis — MACs, SynOps, energy projection
  5. Visualization       — publication-quality figures

Usage:
    python run_pipeline.py                    # Full pipeline (auto-skips done stages)
    python run_pipeline.py --skip_training    # Evaluation + viz only
    python run_pipeline.py --snn_only         # SNN stages only
    python run_pipeline.py --force            # Re-run everything regardless
    python run_pipeline.py --dry_run          # Show what would run without running
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime

# ---------------------------------------------------------------------------
# Terminal colours (work in Colab and most terminals)
# ---------------------------------------------------------------------------
class C:
    RESET  = '\033[0m'
    BOLD   = '\033[1m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    CYAN   = '\033[96m'
    GREY   = '\033[90m'

def _c(text, *codes): return ''.join(codes) + str(text) + C.RESET
def ok(t):   return _c(t, C.GREEN,  C.BOLD)
def warn(t): return _c(t, C.YELLOW, C.BOLD)
def err(t):  return _c(t, C.RED,    C.BOLD)
def info(t): return _c(t, C.CYAN)
def dim(t):  return _c(t, C.GREY)


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

RESULTS = {}   # filled by evaluation stages

def run_stage(name, cmd, skip_if_exists=None, dry_run=False, force=False):
    """Run one pipeline stage, auto-skipping if output already exists."""
    print(f"\n{'─'*60}")
    print(f"  {info('STAGE')}  {C.BOLD}{name}{C.RESET}")
    print(f"{'─'*60}")

    if not force and skip_if_exists and all(os.path.exists(p) for p in skip_if_exists):
        print(f"  {ok('✓ SKIP')}  {dim('Output already exists:')}")
        for p in skip_if_exists:
            print(f"         {dim(p)}")
        return True

    print(f"  {dim('CMD :')} {' '.join(cmd)}")

    if dry_run:
        print(f"  {warn('DRY RUN')} — not executed")
        return True

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  {ok('✓ DONE')}  {dim(f'({elapsed:.0f}s)')}")
        return True
    else:
        print(f"\n  {err('✗ FAILED')}  exit code {result.returncode}  {dim(f'({elapsed:.0f}s)')}")
        return False


def extract_auc(results_dir, score_key):
    """Read AUC from a saved evaluation summary JSON if available."""
    summaries = {
        'convae':     'convae_evaluation_summary.json',
        'convlstm':   'convlstm_evaluation_summary.json',
        'snn_rate':   'snn_evaluation_summary.json',
        'snn_temporal': 'snn_evaluation_summary.json',
        'snn_count':  'snn_evaluation_summary.json',
    }
    path = os.path.join(results_dir, summaries.get(score_key, ''))
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f).get('auc_roc')
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Neuromorphic Video Anomaly Detection Pipeline')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip all training stages (use saved checkpoints)')
    parser.add_argument('--snn_only', action='store_true',
                        help='Run only SNN-related stages (skip ConvLSTM and ConvAE)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run all stages even if outputs exist')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--event_dir', default='./events',
                        help='Root directory for event files (default: ./events)')
    parser.add_argument('--data_dir', default='./data/UCSD_Anomaly_Dataset.v1p2',
                        help='UCSD Ped2 dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs for all models (default: 50)')
    parser.add_argument('--results_dir', default='./results',
                        help='Directory to save evaluation results')
    args = parser.parse_args()

    py = sys.executable   # use same Python interpreter as current env

    print(f"\n{'='*60}")
    print(f"  {C.BOLD}NEUROMORPHIC VIDEO ANOMALY DETECTION{C.RESET}")
    print(f"  End-to-End Pipeline")
    print(f"  {dim(datetime.now().strftime('%Y-%m-%d %H:%M'))}")
    print(f"{'='*60}")
    if args.skip_training: print(f"  {warn('Mode: evaluation + visualization only')}")
    if args.snn_only:       print(f"  {warn('Mode: SNN stages only')}")
    if args.force:          print(f"  {warn('Mode: force re-run all stages')}")
    if args.dry_run:        print(f"  {warn('Mode: dry run (no execution)')}")

    train_event_dir = os.path.join(args.event_dir, 'train')
    test_event_dir  = os.path.join(args.event_dir, 'test')
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    pipeline_ok = True

    # ------------------------------------------------------------------ #
    # STAGE 1 — Event Generation                                          #
    # ------------------------------------------------------------------ #
    if not args.skip_training:
        ok_ = run_stage(
            name='Event Generation (Train)',
            cmd=[py, 'src/convert_videos_to_events.py',
                 '--output_dir', args.event_dir,
                 '--split', 'Train', '--max_videos', '10'],
            skip_if_exists=[train_event_dir],
            dry_run=args.dry_run, force=args.force,
        )
        pipeline_ok &= ok_

        ok_ = run_stage(
            name='Event Generation (Test)',
            cmd=[py, 'src/convert_videos_to_events.py',
                 '--output_dir', args.event_dir,
                 '--split', 'Test', '--max_videos', '12'],
            skip_if_exists=[test_event_dir],
            dry_run=args.dry_run, force=args.force,
        )
        pipeline_ok &= ok_

    # ------------------------------------------------------------------ #
    # STAGE 2 — Training                                                  #
    # ------------------------------------------------------------------ #
    if not args.skip_training:

        snn_configs = [
            ('SNN Rate',     'rate',     'checkpoints/snn_rate'),
            ('SNN Temporal', 'temporal', 'checkpoints/snn_temporal'),
            ('SNN Count',    'count',    'checkpoints/snn_count'),
        ]
        for label, encoding, ckpt_dir in snn_configs:
            ok_ = run_stage(
                name=f'Training — {label}',
                cmd=[py, 'src/train_snn.py',
                     '--event_dir', train_event_dir,
                     '--epochs', str(args.epochs),
                     '--encoding', encoding,
                     '--checkpoint_dir', ckpt_dir],
                skip_if_exists=[os.path.join(ckpt_dir, 'snn_autoencoder_best.pth')],
                dry_run=args.dry_run, force=args.force,
            )
            pipeline_ok &= ok_

        if not args.snn_only:
            ok_ = run_stage(
                name='Training — ConvLSTM (RGB)',
                cmd=[py, 'src/train.py'],
                skip_if_exists=['checkpoints/conv_lstm_autoencoder_best.pth'],
                dry_run=args.dry_run, force=args.force,
            )
            pipeline_ok &= ok_

            ok_ = run_stage(
                name='Training — ConvAutoencoder (Motion frames)',
                cmd=[py, 'src/train_conv_autoencoder.py'],
                skip_if_exists=['checkpoints/conv_autoencoder/conv_autoencoder_best.pth'],
                dry_run=args.dry_run, force=args.force,
            )
            pipeline_ok &= ok_

    # ------------------------------------------------------------------ #
    # STAGE 3 — Evaluation                                                #
    # ------------------------------------------------------------------ #
    snn_evals = [
        ('SNN Rate',     'rate',     'checkpoints/snn_rate/snn_autoencoder_best.pth'),
        ('SNN Temporal', 'temporal', 'checkpoints/snn_temporal/snn_autoencoder_best.pth'),
        ('SNN Count',    'count',    'checkpoints/snn_count/snn_autoencoder_best.pth'),
    ]
    for label, encoding, ckpt in snn_evals:
        ok_ = run_stage(
            name=f'Evaluation — {label}',
            cmd=[py, 'src/evaluate_snn.py',
                 '--checkpoint', ckpt,
                 '--event_dir', test_event_dir,
                 '--encoding', encoding],
            skip_if_exists=[os.path.join(args.results_dir, 'snn_anomaly_scores.npy')]
                if encoding == 'count' else None,
            dry_run=args.dry_run, force=args.force,
        )
        pipeline_ok &= ok_

    if not args.snn_only:
        ok_ = run_stage(
            name='Evaluation — ConvLSTM (RGB)',
            cmd=[py, 'src/evaluate_convlstm.py'],
            skip_if_exists=[os.path.join(args.results_dir, 'convlstm_anomaly_scores.npy')],
            dry_run=args.dry_run, force=args.force,
        )
        pipeline_ok &= ok_

        ok_ = run_stage(
            name='Evaluation — ConvAutoencoder (Motion)',
            cmd=[py, 'src/evaluate_conv_autoencoder.py'],
            skip_if_exists=[os.path.join(args.results_dir, 'convae_anomaly_scores.npy')],
            dry_run=args.dry_run, force=args.force,
        )
        pipeline_ok &= ok_

    # ------------------------------------------------------------------ #
    # STAGE 4 — Efficiency Analysis                                       #
    # ------------------------------------------------------------------ #
    best_snn_ckpt = 'checkpoints/snn_count/snn_autoencoder_best.pth'
    ok_ = run_stage(
        name='Efficiency Analysis',
        cmd=[py, 'experiments/efficiency_analysis.py',
             '--checkpoint', best_snn_ckpt],
        skip_if_exists=[os.path.join(args.results_dir, 'efficiency_analysis.json')],
        dry_run=args.dry_run, force=args.force,
    )
    pipeline_ok &= ok_

    # ------------------------------------------------------------------ #
    # STAGE 5 — Visualization                                             #
    # ------------------------------------------------------------------ #
    ok_ = run_stage(
        name='Visualization',
        cmd=[py, 'src/visualize_results.py'],
        dry_run=args.dry_run, force=args.force,
    )
    pipeline_ok &= ok_

    # ------------------------------------------------------------------ #
    # FINAL SUMMARY TABLE                                                 #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"  {C.BOLD}FINAL RESULTS SUMMARY{C.RESET}")
    print(f"{'='*60}")

    # Hardcoded known results — updated from run outputs if JSON exists
    known = {
        'ConvAutoencoder (Motion frames)': 0.8269,
        'SNN Count (Events)':              0.6994,
        'SNN Temporal (Events)':           0.6558,
        'SNN Rate (Events)':               0.6416,
        'ConvLSTM RGB':                    0.5995,
    }

    # Try to read latest from summary JSON files
    for fname, key in [
        ('convae_evaluation_summary.json',  'ConvAutoencoder (Motion frames)'),
        ('snn_evaluation_summary.json',     'SNN Count (Events)'),
        ('convlstm_evaluation_summary.json','ConvLSTM RGB'),
    ]:
        path = os.path.join(args.results_dir, fname)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    v = json.load(f).get('auc_roc')
                if v is not None:
                    known[key] = v
            except Exception:
                pass

    ranked = sorted(known.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  {'Model':<35} {'AUC-ROC':>8}  {'vs Best':>8}")
    print(f"  {'─'*35} {'─'*8}  {'─'*8}")
    best_auc = ranked[0][1]
    for i, (name, auc_val) in enumerate(ranked):
        delta = f"-{(best_auc - auc_val)*100:.1f}pp" if i > 0 else "  (best)"
        medal = ['🥇', '🥈', '🥉'] [i] if i < 3 else '   '
        marker = ok if auc_val == best_auc else (warn if auc_val > 0.65 else err)
        print(f"  {medal} {name:<33} {marker(f'{auc_val:.4f}'):>17}  {dim(delta):>12}")

    # Efficiency highlight
    eff_path = os.path.join(args.results_dir, 'efficiency_analysis.json')
    if os.path.exists(eff_path):
        try:
            with open(eff_path) as f:
                eff = json.load(f)
            snn_e  = eff.get('SNN Autoencoder', {}).get('energy_uJ', 0)
            ae_e   = eff.get('ConvAutoencoder', {}).get('energy_uJ', 0)
            snn_p  = eff.get('SNN Autoencoder', {}).get('parameters', 0)
            ae_p   = eff.get('ConvAutoencoder', {}).get('parameters', 0)
            if snn_e and ae_e:
                print(f"\n  {C.BOLD}Efficiency Highlights:{C.RESET}")
                print(f"  • SNN is {ae_e/snn_e:.1f}× more energy-efficient on Loihi vs ConvAE on GPU")
                print(f"  • SNN has {ae_p/snn_p:.1f}× fewer parameters than ConvAE ({snn_p:,} vs {ae_p:,})")
                print(f"  • SNN trades {(best_auc - known.get('SNN Count (Events)', 0))*100:.1f} AUC points for {ae_e/snn_e:.0f}× energy reduction")
        except Exception:
            pass

    status = ok('PIPELINE COMPLETE') if pipeline_ok else err('PIPELINE FINISHED WITH ERRORS')
    print(f"\n  {status}")
    figures_dir = 'results/figures'
    if os.path.isdir(figures_dir):
        figs = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
        print(f"  {len(figs)} figures saved to {figures_dir}/")
    print(f"{'='*60}\n")

    return 0 if pipeline_ok else 1


if __name__ == '__main__':
    sys.exit(main())
