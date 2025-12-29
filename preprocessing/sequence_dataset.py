import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image


class VideoSequenceDataset(Dataset):
    def __init__(self, video_paths, sequence_length=8, transform=None):
        """
        video_paths: list of directories (each directory = frames of one video)
        """
        self.sequence_length = sequence_length
        self.transform = transform
        self.sequences = []

        for video_path in video_paths:
            frames = sorted(
                glob.glob(os.path.join(video_path, "*.tif")) +
                glob.glob(os.path.join(video_path, "*.bmp")) 
            )

            # Sliding window
            for i in range(len(frames) - sequence_length + 1):
                self.sequences.append(frames[i:i + sequence_length])

        print(f"Total sequences created: {len(self.sequences)}")

    def __getitem__(self, idx):
        frame_paths = self.sequences[idx]
        frames = []

        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        return torch.stack(frames)  # (T, C, H, W)

    def __len__(self):
        return len(self.sequences)
