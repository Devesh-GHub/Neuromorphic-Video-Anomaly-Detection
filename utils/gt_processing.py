import os
import numpy as np
from utils.metrics import load_ped2_frame_labels


def get_sequence_gt_labels(gt_root, sequence_length):
    """
    Converts frame-level GT labels to sequence-level GT labels
    """
    all_frame_labels = []

    # ONLY *_gt folders
    gt_folders = sorted([
        f for f in os.listdir(gt_root)
        if f.endswith("_gt")
    ])

    assert len(gt_folders) > 0, f"No GT folders found in {gt_root}"

    for gt_folder in gt_folders:
        gt_path = os.path.join(gt_root, gt_folder)

        if not os.path.isdir(gt_path):
            raise ValueError(f"GT path does not exist: {gt_path}")

        frame_labels = load_ped2_frame_labels(gt_path)
        all_frame_labels.extend(frame_labels)

    all_frame_labels = np.array(all_frame_labels)

    # Frame → Sequence labels
    seq_labels = []
    for i in range(len(all_frame_labels) - sequence_length + 1):
        seq_labels.append(int(np.any(all_frame_labels[i:i+sequence_length])))

    return np.array(seq_labels)
