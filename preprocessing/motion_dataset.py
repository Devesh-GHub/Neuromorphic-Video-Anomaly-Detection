"""
Dataset that returns frame-difference (motion) images instead of raw frames.

For each video, yields abs(frame_t - frame_{t-1}) for t = 1..N-1.
This gives the same motion signal as event cameras but with continuous values.
Normal pedestrian motion → small consistent differences → low AE reconstruction error.
Anomalous motion (bikes, skaters) → unusual differences → high reconstruction error.
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class MotionFrameDataset(Dataset):
    def __init__(self, video_paths, image_size=128):
        self.pairs = []
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0, 1]
        ])

        for folder in video_paths:
            frames = sorted(
                glob.glob(os.path.join(folder, "*.tif")) +
                glob.glob(os.path.join(folder, "*.bmp"))
            )
            for i in range(1, len(frames)):
                self.pairs.append((frames[i - 1], frames[i]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path_prev, path_curr = self.pairs[idx]

        prev = self.transform(Image.open(path_prev).convert("RGB"))  # (3, H, W)
        curr = self.transform(Image.open(path_curr).convert("RGB"))  # (3, H, W)

        # Absolute frame difference — values in [0, 1]
        diff = torch.abs(curr - prev)
        return diff
