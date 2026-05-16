"""
PyTorch-compatible Event Dataset for SNN training.

Wraps the existing EventDataset to return PyTorch tensors
compatible with DataLoader and GPU training.

Returns spike tensors of shape (2, H, W, T) ready for SNN input.
"""

import sys
import torch
import numpy as np
import h5py
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import torch.nn.functional as F


class EventSpikeDataset(Dataset):
    """
    PyTorch Dataset that loads events from HDF5 files and converts
    to spike tensors for SNN training.
    
    Each sample is a spike tensor of shape (2, H, W, T):
        - 2 channels: ON events (polarity=1) and OFF events (polarity=0)
        - H, W: spatial resolution (resized to target_size)
        - T: number of timesteps
    """
    
    def __init__(self,
                 event_dir: str,
                 target_size: Tuple[int, int] = (128, 128),
                 num_steps: int = 25,
                 time_window: float = 0.05,
                 encoding: str = 'rate',
                 max_rate: float = 100.0,
                 overlap: float = 0.5):
        """
        Args:
            event_dir: Directory containing .h5 event files
            target_size: (H, W) to resize event frames
            num_steps: Number of timesteps T for spike trains
            time_window: Duration of each sample window in seconds
            encoding: 'rate', 'temporal', or 'count'
            max_rate: Max spike rate for rate encoding (Hz)
            overlap: Overlap fraction between consecutive windows
        """
        self.target_size = target_size
        self.num_steps = num_steps
        self.time_window = time_window
        self.encoding = encoding
        self.max_rate = max_rate
        self.overlap = overlap
        self._file_cache = {}  # file_idx → (t, x, y, p)

        # Find all event files
        self.event_dir = Path(event_dir)
        self.event_files = sorted(self.event_dir.rglob("*.h5"))
        
        if len(self.event_files) == 0:
            # Try .npy files
            self.event_files = sorted(self.event_dir.rglob("*.npy"))
        
        if len(self.event_files) == 0:
            raise FileNotFoundError(f"No .h5 or .npy files found in {event_dir}")
        
        # Build sample index: list of (file_idx, window_start_time)
        self.samples = self._build_sample_index()
        
        print(f"EventSpikeDataset initialized:")
        print(f"  Files: {len(self.event_files)}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Target size: {target_size}")
        print(f"  Timesteps: {num_steps}")
        print(f"  Encoding: {encoding}")
    
    def _build_sample_index(self):
        """Create index of (file_idx, window_start_time) for each sample."""
        samples = []
        step = self.time_window * (1.0 - self.overlap)
        
        for file_idx, event_file in enumerate(self.event_files):
            try:
                duration = self._get_duration(event_file)
                if duration <= 0:
                    continue
                
                n_windows = max(1, int((duration - self.time_window) / step) + 1)
                
                for w in range(n_windows):
                    t_start = w * step
                    samples.append((file_idx, t_start))
                    
            except Exception as e:
                print(f"  Warning: Could not read {event_file.name}: {e}")
                continue
        
        return samples
    
    def _get_duration(self, event_file):
        """Get total duration of event file in seconds."""
        if event_file.suffix == '.h5':
            with h5py.File(event_file, 'r') as f:
                t = f['events/t'][:]
                return (t.max() - t.min()) * 1e-6  # μs → seconds
        elif event_file.suffix == '.npy':
            data = np.load(event_file, allow_pickle=True)
            if hasattr(data, 'dtype') and data.dtype.names and 't' in data.dtype.names:
                return (data['t'].max() - data['t'].min()) * 1e-6
            return 0
        return 0
    
    def _load_events(self, file_idx):
        """Load events from file."""
        event_file = self.event_files[file_idx]
        
        if event_file.suffix == '.h5':
            with h5py.File(event_file, 'r') as f:
                t = f['events/t'][:].astype(np.float64) * 1e-6  # μs → seconds
                x = f['events/x'][:].astype(np.int64)
                y = f['events/y'][:].astype(np.int64)
                p = f['events/p'][:].astype(np.int64)
            return t, x, y, p
        
        elif event_file.suffix == '.npy':
            data = np.load(event_file, allow_pickle=True)
            if hasattr(data, 'dtype') and data.dtype.names:
                t = data['t'].astype(np.float64) * 1e-6
                x = data['x'].astype(np.int64)
                y = data['y'].astype(np.int64)
                p = data['p'].astype(np.int64)
                return t, x, y, p
        
        raise ValueError(f"Cannot read {event_file}")
    
    def _extract_window(self, t, x, y, p, t_start):
        """Extract events within a time window."""
        t_end = t_start + self.time_window
        mask = (t >= t_start) & (t < t_end)
        return t[mask] - t_start, x[mask], y[mask], p[mask]
    
    def _events_to_frame(self, x, y, p, original_h, original_w):
        """
        Convert events to 2-channel frame at target resolution.
        
        Returns: (2, target_H, target_W) float32 tensor
        """
        H, W = self.target_size
        frame = np.zeros((2, H, W), dtype=np.float32)
        
        if len(x) == 0:
            return frame
        
        # Scale coordinates to target size
        x_scaled = (x * W / original_w).astype(np.int64)
        y_scaled = (y * H / original_h).astype(np.int64)
        
        # Clip to valid range
        x_scaled = np.clip(x_scaled, 0, W - 1)
        y_scaled = np.clip(y_scaled, 0, H - 1)
        
        # Accumulate events — vectorized, no Python loop
        # Channel 0 = OFF events (p=0), Channel 1 = ON events (p=1)
        np.add.at(frame, (p, y_scaled, x_scaled), 1.0)

        return frame
    
    def _encode_spikes(self, event_frame):
        """
        Encode event frame as spike trains.
        
        Args:
            event_frame: (2, H, W) float32
            
        Returns:
            spikes: (2, H, W, T) float32
        """
        C, H, W = event_frame.shape
        T = self.num_steps
        
        if self.encoding == 'rate':
            # Normalize to [0, 1]
            max_val = event_frame.max()
            if max_val > 0:
                normalized = event_frame / max_val
            else:
                normalized = event_frame
            
            # Spike probability per timestep
            dt_sec = self.time_window / T
            rates = normalized * self.max_rate  # Hz
            probs = rates * dt_sec  # Probability per step
            probs = np.clip(probs, 0, 1)
            
            # Poisson spike generation
            rand = np.random.rand(C, H, W, T)
            spikes = (rand < probs[..., None]).astype(np.float32)
            
        elif self.encoding == 'temporal':
            spikes = np.zeros((C, H, W, T), dtype=np.float32)
            max_val = event_frame.max()
            if max_val > 0:
                normalized = event_frame / max_val
                # Higher intensity → earlier spike
                spike_times = ((1.0 - normalized) * (T - 1)).astype(np.int64)
                c_idx, h_idx, w_idx = np.nonzero(normalized > 0)
                t_idx = spike_times[c_idx, h_idx, w_idx]
                spikes[c_idx, h_idx, w_idx, t_idx] = 1.0
            
        elif self.encoding == 'count':
            # Binary: fire every timestep if any events present
            binary = (event_frame > 0).astype(np.float32)
            spikes = np.tile(binary[..., None], (1, 1, 1, T))
        
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")
        
        return spikes
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            spikes: torch.FloatTensor of shape (2, H, W, T)
        """
        file_idx, t_start = self.samples[idx]
        
        try:
            if file_idx not in self._file_cache:
                self._file_cache[file_idx] = self._load_events(file_idx)
            t, x, y, p = self._file_cache[file_idx]
            
            # Get original resolution from event coordinates
            original_h = max(y.max() + 1, 240)
            original_w = max(x.max() + 1, 360)
            
            # Extract window
            t_w, x_w, y_w, p_w = self._extract_window(t, x, y, p, t_start)
            
            # Convert to frame
            event_frame = self._events_to_frame(x_w, y_w, p_w, original_h, original_w)
            
            # Encode as spikes
            spikes = self._encode_spikes(event_frame)
            
            return torch.FloatTensor(spikes)
        
        except Exception as e:
            # Return zeros on error
            H, W = self.target_size
            return torch.zeros(2, H, W, self.num_steps)


def create_event_dataloaders(event_train_dir, event_test_dir=None,
                              batch_size=16, num_steps=25,
                              encoding='rate'):
    """
    Convenience function to create train and test DataLoaders.
    
    Args:
        event_train_dir: Path to training event files
        event_test_dir: Path to test event files (optional)
        batch_size: Batch size
        num_steps: SNN timesteps
        encoding: Spike encoding method
        num_workers: DataLoader workers
        
    Returns:
        train_loader, test_loader (or just train_loader if no test dir)
    """
    train_dataset = EventSpikeDataset(
        event_dir=event_train_dir,
        num_steps=num_steps,
        encoding=encoding
    )
    
    num_workers = 0 if sys.platform == 'win32' else 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    if event_test_dir:
        test_dataset = EventSpikeDataset(
            event_dir=event_test_dir,
            num_steps=num_steps,
            encoding=encoding
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    return train_loader


if __name__ == "__main__":
    print("Testing EventSpikeDataset...")
    
    # Test with your existing event data
    # Adjust this path to where your events are stored
    event_dir = "./events/train"
    
    try:
        dataset = EventSpikeDataset(
            event_dir=event_dir,
            target_size=(128, 128),
            num_steps=25,
            encoding='rate'
        )
        
        sample = dataset[0]
        print(f"Sample shape: {sample.shape}")
        print(f"Sample dtype: {sample.dtype}")
        print(f"Spike rate: {sample.mean().item():.4f}")
        print(f"Non-zero: {(sample > 0).sum().item()} / {sample.numel()}")
        
        # Test DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        print(f"Batch shape: {batch.shape}")
        
        print("✓ Dataset test passed!")
        
    except FileNotFoundError as e:
        print(f"Event files not found at {event_dir}")
        print("Run: python src/convert_videos_to_events.py first")
        print(f"Error: {e}")