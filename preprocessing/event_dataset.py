"""
Event Dataset for SNN Training

Provides efficient data loading and spike encoding for event-based
video anomaly detection using Spiking Neural Networks.

Features:
- Lazy loading (memory efficient)
- Multiple spike encoding methods
- Temporal windowing
- On-the-fly preprocessing
- Caching for frequently accessed data

"""

import numpy as np
import h5py
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union


class EventDataset:
    """
    Dataset class for loading and encoding event streams for SNN training.
    
    Uses lazy loading to be memory efficient - events are loaded on demand.
    """
    
    def __init__(self,
                 event_files: List[Union[str, Path]],
                 sequence_length: int = 100,
                 time_window: float = 0.05,
                 encoding: str = 'rate',
                 resolution: Tuple[int, int] = (240, 360),
                 max_rate: float = 100.0,
                 overlap: float = 0.0):
        """
        Initialize event dataset.
        
        Args:
            event_files: List of paths to event files (.h5, .npy)
            sequence_length: Number of time steps per sample (T)
            time_window: Duration of each time window in seconds
            encoding: Spike encoding method ('rate', 'temporal', 'count')
            resolution: (height, width) of event frames
            max_rate: Maximum spike rate for rate encoding (Hz)
            overlap: Overlap between consecutive sequences (0.0 to 0.99)
        """
        self.event_files = [Path(f) for f in event_files]
        self.sequence_length = sequence_length
        self.time_window = time_window
        self.encoding = encoding
        self.resolution = resolution
        self.max_rate = max_rate
        self.overlap = overlap
        
        # Verify files exist
        self._verify_files()
        
        # Create index mapping: (file_idx, window_idx) for each sample
        self.sample_index = self._create_sample_index()
        
        
        print("=" * 70)
        print("EVENT DATASET INITIALIZED")
        print("=" * 70)
        print(f"Files: {len(self.event_files)}")
        print(f"Total samples: {len(self.sample_index)}")
        print(f"Sequence length: {sequence_length} time steps")
        print(f"Time window: {time_window * 1000:.1f} ms")
        print(f"Resolution: {resolution[0]}x{resolution[1]}")
        print(f"Encoding: {encoding}")
        print(f"Overlap: {overlap * 100:.0f}%")
        print("=" * 70)
    
    def _verify_files(self):
        """Verify that all event files exist."""
        missing_files = [f for f in self.event_files if not f.exists()]
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing {len(missing_files)} event files:\n" +
                "\n".join(str(f) for f in missing_files[:5])
            )
    
    def _create_sample_index(self) -> List[Tuple[int, int]]:
        """
        Create index mapping each sample to (file_idx, window_start_idx).
        
        Returns:
            List of tuples (file_idx, window_idx)
        """
        sample_index = []
        
        for file_idx, event_file in enumerate(self.event_files):
            # Load file metadata to get duration
            metadata = self._load_metadata(event_file)
            
            if metadata is None:
                continue
            
            # Calculate number of windows from this file
            duration = metadata.get('duration', 0)
            
            if duration <= 0:
                continue
            
            # Number of samples from this file
            step_size = self.time_window * (1.0 - self.overlap)
            n_windows = int((duration - self.time_window) / step_size) + 1
            
            if n_windows < 1:
                n_windows = 1
            
            # Add to index
            for window_idx in range(n_windows):
                sample_index.append((file_idx, window_idx))
        
        return sample_index
    

    def _load_metadata(self, event_file: Path) -> Optional[Dict]:
        try:
            if event_file.suffix == '.h5':
                with h5py.File(event_file, 'r') as f:
                    t = f['events/t'][:]
            elif event_file.suffix == '.npy':
                data = np.load(event_file)
                t = data['t']
            else:
                return None

            duration = (t.max() - t.min()) * 1e-6  # seconds
            return {'duration': duration}
        except Exception:
            return None


    def _load_events(self, file_idx: int) -> Optional[np.ndarray]:
        """
        Load events from file with caching.
        
        Args:
            file_idx: Index of file to load
            
        Returns:
            Event array or None
        """
        
        # Load from disk
        event_file = self.event_files[file_idx]
        
        try:
            if event_file.suffix == '.h5':
                events = self._load_h5(event_file)
            elif event_file.suffix == '.npy':
                events = self._load_npy(event_file)
            else:
                return None
        
        except Exception as e:
            return None
        
        return events
    
    def _load_h5(self, file_path: Path) -> np.ndarray:
        """Load events from HDF5 file."""
        with h5py.File(file_path, 'r') as f:
            t = f['events/t'][:]
            x = f['events/x'][:]
            y = f['events/y'][:]
            p = f['events/p'][:]
            
            n_events = len(t)
            events = np.zeros(n_events, dtype=[
                ('t', np.float64),
                ('x', np.uint16),
                ('y', np.uint16),
                ('p', np.uint8)
            ])
            
            events['t'] = t * 1e-6 
            events['x'] = x
            events['y'] = y
            events['p'] = p
            
            return events
    
    def _load_npy(self, file_path: Path) -> np.ndarray:
        """Load events from NumPy file."""
        return np.load(file_path)
    
    def _extract_window(self,
                       events: np.ndarray,
                       window_idx: int) -> np.ndarray:
        """
        Extract time window from event stream.
        
        Args:
            events: Full event array
            window_idx: Which window to extract
            
        Returns:
            Events within the specified window
        """
        if len(events) == 0:
            return events
        
        # Calculate time range
        t_min = events['t'].min()
        step_size = self.time_window * (1.0 - self.overlap)  # to microseconds
        
        t_start = events['t'].min() + window_idx * step_size
        t_end = t_start + self.time_window 
        
        # Extract events in window
        mask = (events['t'] >= t_start) & (events['t'] < t_end)
        window_events = events[mask]
        
        # Normalize timestamps to start at 0
        if len(window_events) > 0:
            window_events = window_events.copy()
            window_events['t'] = window_events['t'] - t_start
        
        return window_events
    
    def _events_to_frame(self, events: np.ndarray) -> np.ndarray:
        """
        Convert events to 2-channel frame (ON/OFF).
        
        Args:
            events: Event array
            
        Returns:
            Frame array (2, H, W) with ON and OFF channels
        """
        height, width = self.resolution
        frame = np.zeros((2, height, width), dtype=np.float32)
        if len(events) == 0:
            return frame
        
        x = events['x'].astype(np.int64)
        y = events['y'].astype(np.int64)
        p = events['p'].astype(np.int64)

        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        np.add.at(frame, (p[valid], y[valid], x[valid]), 1)
        
        return frame
    
    def _encode_spikes(self, event_frame: np.ndarray) -> np.ndarray:
        """
        Encode event frame as spike trains.
        
        Args:
            event_frame: Event frame (2, H, W)
            
        Returns:
            Spike trains (2, H, W, T)
        """
        if self.encoding == 'rate':
            return self._rate_encode(event_frame)
        elif self.encoding == 'temporal':
            return self._temporal_encode(event_frame)
        elif self.encoding == 'count':
            return self._count_encode(event_frame)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")
    
    def _rate_encode(self, event_frame: np.ndarray) -> np.ndarray:
        """
        Rate encoding: spike rate proportional to event count.
        
        Args:
            event_frame: Event counts (2, H, W)
            
        Returns:
            Spike trains (2, H, W, T)
        """
        C, H, W = event_frame.shape
        T = self.sequence_length
        
        # Normalize event counts to [0, 1]
        max_count = event_frame.max()
        if max_count > 0:
            normalized = event_frame / max_count
        else:
            normalized = event_frame
        
        # Spike probability per time step
        dt = (self.time_window * 1000) / T  # ms per step
        rates = normalized * self.max_rate
        probs = rates * dt / 1000.0  # shape (C,H,W)
        
        # Broadcast to time
        rand = np.random.rand(C, H, W, T)
        spikes = (rand < probs[..., None]).astype(np.float32)
        
        return spikes
    
    def _temporal_encode(self, event_frame: np.ndarray) -> np.ndarray:
        """
        Temporal encoding: spike time encodes intensity.
        
        Higher values → earlier spikes
        
        Args:
            event_frame: Event counts (2, H, W)
            
        Returns:
            Spike trains (2, H, W, T)
        """
        C, H, W = event_frame.shape
        T = self.sequence_length
        
        spikes = np.zeros((C, H, W, T), dtype=np.float32)
        
        # Normalize to [0, 1]
        max_count = event_frame.max()
        if max_count > 0:
            normalized = event_frame / max_count
        else:
            normalized = event_frame
        
        spike_times = ((1.0 - normalized) * (T - 1)).astype(np.int64)

        c_idx, h_idx, w_idx = np.nonzero(normalized > 0)
        t_idx = spike_times[c_idx, h_idx, w_idx]

        spikes[c_idx, h_idx, w_idx, t_idx] = 1.0
        
        return spikes
    
    def _count_encode(self, event_frame: np.ndarray) -> np.ndarray:
        """
        Count encoding: binary presence of events.
        
        Args:
            event_frame: Event counts (2, H, W)
            
        Returns:
            Spike trains (2, H, W, T)
        """
        C, H, W = event_frame.shape
        T = self.sequence_length
        
        # Binary: 0 if no events, 1 if any events
        binary_frame = (event_frame > 0).astype(np.float32)
        
        # Repeat across time steps
        spike_trains = np.tile(
            binary_frame[:, :, :, np.newaxis],
            (1, 1, 1, T)
        )
        
        return spike_trains
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.sample_index)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get event sequence at index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - 'spikes': Spike trains (2, H, W, T)
                - 'events': Original event frame (2, H, W)
                - 'file_idx': Source file index
                - 'window_idx': Window index within file
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        # Get file and window indices
        file_idx, window_idx = self.sample_index[idx]
        
        # Load events from file
        events = self._load_events(file_idx)
        
        if events is None:
            # Return empty sample
            return self._get_empty_sample(file_idx, window_idx)
        
        # Extract time window
        window_events = self._extract_window(events, window_idx)
        
        # Convert to event frame
        event_frame = self._events_to_frame(window_events)
        
        # Encode as spikes
        spike_trains = self._encode_spikes(event_frame)

        assert spike_trains.shape[-1] == self.sequence_length
        
        return {
            'spikes': spike_trains,
            'events': event_frame,
            'file_idx': file_idx,
            'window_idx': window_idx,
            'n_events': len(window_events)
        }
    
    def _get_empty_sample(self,
                         file_idx: int,
                         window_idx: int) -> Dict[str, np.ndarray]:
        """Return empty sample when data cannot be loaded."""
        C, H, W = 2, *self.resolution
        T = self.sequence_length
        
        return {
            'spikes': np.zeros((C, H, W, T), dtype=np.float32),
            'events': np.zeros((C, H, W), dtype=np.float32),
            'file_idx': file_idx,
            'window_idx': window_idx,
            'n_events': 0
        }
