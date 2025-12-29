# Implements a custom PyTorch Dataset for video frames.
# - Collects individual frames from multiple videos.
# - Converts frame-based videos into a flat dataset.
# - Applies image preprocessing transforms.
# - Enables scalable batch loading via DataLoader.


import os
import glob                   # used for file pattern matching , like finding all .tif or .bmp files
from torch.utils.data import Dataset 
from PIL import Image  # uaed for images like .tif or .bmp files loading and processing

class VideoFrameDataset(Dataset):   # this fun returns individual frames from multiple videos
    def __init__(self, video_paths, transform=None):
        """
        video_paths: List of folders
        Each folder contains many *.tif or *.bmp files
        """
        self.transform = transform
        self.frame_paths = []

        # Collect ALL frame paths from ALL videos
        for folder in video_paths:
            frame_list = sorted(
                glob.glob(os.path.join(folder, "*.tif")) +
                glob.glob(os.path.join(folder, "*.bmp"))
            )
            self.frame_paths.extend(frame_list)

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        img = Image.open(frame_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img
