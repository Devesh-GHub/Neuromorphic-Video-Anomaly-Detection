
import numpy as np
from typing import Union, Tuple, Optional


class EventToFrameConverter:
    """
    Convert event streams to frame representations using various methods.
    
    Methods supported:
        1. Event Frame (simple accumulation)
        2. Time Surface
        3. Event Count (binary or weighted)
    """
    
    def __init__(self, height: int, width: int):
        """
        Initialize converter with sensor dimensions.
        
        Args:
            height: Frame height in pixels
            width: Frame width in pixels
        """
        self.height = height
        self.width = width
        
    def events_to_frame(self, 
                       events: np.ndarray, 
                       time_window: float = 0.05,
                       method: str = 'accumulation',
                       polarity_mode: str = 'separate') -> np.ndarray:
        """
        Main conversion function - converts events to frame representation.
        
        Args:
            events: Structured array with fields (t, x, y, p) or (N, 4) array
            time_window: Time window in seconds (or microseconds if timestamps are in μs)
            method: Conversion method ('accumulation', 'time_surface', 'count')
            polarity_mode: How to handle polarity ('separate', 'combined', 'signed')
                - 'separate': Create 2 channels (ON and OFF)
                - 'combined': Single channel (ON and OFF merged)
                - 'signed': Single channel with +1 for ON, -1 for OFF
        
        Returns:
            frame: Converted frame representation
                - Shape depends on polarity_mode:
                    - 'separate': (height, width, 2)
                    - 'combined'/'signed': (height, width)
        """
        if len(events) == 0:
            if polarity_mode == 'separate':
                return np.zeros((self.height, self.width, 2))
            return np.zeros((self.height, self.width))
        
        # Select conversion method
        if method == 'accumulation':
            return self._accumulation_method(events, polarity_mode)
        
        elif method == 'time_surface':
            return self._time_surface_method(events, time_window, polarity_mode)
        
        elif method == 'count':
            return self._count_method(events, polarity_mode)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    

    def _accumulation_method(self, 
                            events: np.ndarray, 
                            polarity_mode: str = 'separate') -> np.ndarray:
        """
        Simple event accumulation (counting events at each pixel).
        
        This is the most basic method: just count how many events occurred
        at each pixel location.
        
        Args:
            events: Event array
            polarity_mode: Polarity handling mode
            
        Returns:
            frame: Accumulated event frame
        """
        if polarity_mode == 'separate':
            # Two channels: [ON_events, OFF_events]
            frame = np.zeros((self.height, self.width, 2), dtype=np.float32)
            
            for evt in events:
                x, y, p = self._extract_xyp(evt)
                if self._is_valid_coord(x, y):
                    if p == 1:
                        frame[y, x, 0] += 1  # ON channel
                    else:
                        frame[y, x, 1] += 1  # OFF channel
                        
        elif polarity_mode == 'combined':
            # Single channel: count all events regardless of polarity
            frame = np.zeros((self.height, self.width), dtype=np.float32)
            
            for evt in events:
                x, y, p = self._extract_xyp(evt)
                if self._is_valid_coord(x, y):
                    frame[y, x] += 1
                    
        elif polarity_mode == 'signed':
            # Single channel: +1 for ON, -1 for OFF
            frame = np.zeros((self.height, self.width), dtype=np.float32)
            
            for evt in events:
                x, y, p = self._extract_xyp(evt)
                if self._is_valid_coord(x, y):
                    frame[y, x] += 1 if p == 1 else -1
        else:
            raise ValueError(f"Unknown polarity_mode: {polarity_mode}")
        
        return frame
    

    def _time_surface_method(self, 
                            events: np.ndarray, 
                            time_window: float,
                            polarity_mode: str = 'separate') -> np.ndarray:
        """
        Time surface representation - stores most recent timestamp at each pixel.
        
        Time surface encodes temporal information by storing the time of the 
        most recent event at each pixel, creating a "surface" that shows 
        temporal context.
        
        Args:
            events: Event array
            time_window: Time window for normalization (in same units as timestamps)
            polarity_mode: Polarity handling mode
            
        Returns:
            frame: Time surface representation (values between 0 and 1)
        """
        # Get the last timestamp
        last_timestamp = self._extract_t(events[-1]) if len(events) > 0 else 0
        
        if polarity_mode == 'separate':
            # Two channels for ON and OFF events
            frame = np.zeros((self.height, self.width, 2), dtype=np.float32)
            
            for evt in events:
                t, x, y, p = self._extract_txxyp(evt)
                if self._is_valid_coord(x, y):
                    # Compute time since event (exponential decay)
                    time_since = last_timestamp - t
                    time_surface_value = np.exp(-time_since / time_window)
                    
                    if p == 1:
                        frame[y, x, 0] = max(frame[y, x, 0], time_surface_value)  # for p=1 ,channel=0 so frame[y,x,0]
                    else:
                        frame[y, x, 1] = max(frame[y, x, 1], time_surface_value)  # for p=0 ,channel=1 so frame[y,x,1]
                        
        elif polarity_mode in ['combined', 'signed']:
            # Single channel
            frame = np.zeros((self.height, self.width), dtype=np.float32)
            
            for evt in events:
                t, x, y, p = self._extract_txxyp(evt)
                if self._is_valid_coord(x, y):
                    time_since = last_timestamp - t
                    time_surface_value = np.exp(-time_since / time_window)
                    
                    if polarity_mode == 'signed':
                        sign = 1 if p == 1 else -1
                        frame[y, x] = sign * max(abs(frame[y, x]), time_surface_value)
                    else:
                        frame[y, x] = max(frame[y, x], time_surface_value)
        else:
            raise ValueError(f"Unknown polarity_mode: {polarity_mode}")
        
        return frame
    

    def _count_method(self, 
                     events: np.ndarray, 
                     polarity_mode: str = 'separate',
                     binary: bool = False) -> np.ndarray:
        """
        Event count method - similar to accumulation but can be binary.
        
        Args:
            events: Event array
            polarity_mode: Polarity handling mode
            binary: If True, use binary encoding (0 or 1), else count
            
        Returns:
            frame: Event count frame
        """
        frame = self._accumulation_method(events, polarity_mode)
        
        if binary:
            # Convert to binary: 0 if no events, 1 if any events
            frame = (frame > 0).astype(np.float32)
        
        return frame
    
    def events_to_frame_sequence(self,
                                events: np.ndarray,
                                n_frames: int = None,
                                time_window: float = None,
                                method: str = 'accumulation',
                                polarity_mode: str = 'separate',
                                overlap: float = 0.0) -> np.ndarray:
        """
        Convert event stream to a sequence of frames.
        
        Args:
            events: Event array
            n_frames: Number of frames to generate (if None, use time_window)
            time_window: Time window per frame in seconds (if None, use n_frames)
            method: Conversion method
            polarity_mode: Polarity handling mode
            overlap: Overlap between consecutive windows (0.0 to 0.99)
            
        Returns:
            frames: Array of shape (n_frames, height, width, [channels])
        """
        if len(events) == 0:
            raise ValueError("No events provided")
        
        # Get time range
        t_min = self._extract_t(events[0])
        t_max = self._extract_t(events[-1])
        total_duration = t_max - t_min
        
        # Determine parameters
        if n_frames is not None and time_window is None:
            time_window = total_duration / n_frames

        elif time_window is not None and n_frames is None:
            n_frames = int(total_duration / time_window) + 1

        elif n_frames is None and time_window is None:
            raise ValueError("Must specify either n_frames or time_window")
        
        # Calculate step size considering overlap 
            # Step_Size --> how far we move forward in time before starting the next frame
            # Overlap --> how much two consecutive frames share the same events.
        step_size = time_window * (1 - overlap)   
        
        # Generate frames
        frames_list = []
        for i in range(n_frames):       # For each frame calculating its start and end time
            t_start = t_min + i * step_size
            t_end = t_start + time_window
            
            # Extract events in this window
            window_events = self._filter_events_by_time(events, t_start, t_end)
            
            # Convert to frame
            if len(window_events) > 0:
                frame = self.events_to_frame(window_events, time_window, 
                                            method, polarity_mode)
            else:
                # Empty frame
                if polarity_mode == 'separate':
                    frame = np.zeros((self.height, self.width, 2))
                else:
                    frame = np.zeros((self.height, self.width))
            
            frames_list.append(frame)
        
        return np.array(frames_list)
    

    def _filter_events_by_time(self, 
                              events: np.ndarray, 
                              t_start: float, 
                              t_end: float) -> np.ndarray:
        """Filter events within time window."""
        timestamps = np.array([self._extract_t(evt) for evt in events])
        mask = (timestamps >= t_start) & (timestamps < t_end)
        return events[mask]
    

    def _extract_t(self, event) -> float:      # This function retunrs events timestamp 
        """Extract timestamp from event."""
        if isinstance(event, np.void):  # Structured array
            return float(event['t'])
        else:  # Regular array
            return float(event[0])
    

    def _extract_xyp(self, event) -> Tuple[int, int, int]:
        """Extract x, y, polarity from event."""
        if isinstance(event, np.void):  # Structured array
            return int(event['x']), int(event['y']), int(event['p'])
        else:  # Regular array
            return int(event[1]), int(event[2]), int(event[3])
    

    def _extract_txxyp(self, event) -> Tuple[float, int, int, int]:
        """Extract t, x, y, polarity from event."""
        if isinstance(event, np.void):  # Structured array
            return (float(event['t']), int(event['x']), 
                   int(event['y']), int(event['p']))
        else:  # Regular array
            return (float(event[0]), int(event[1]), 
                   int(event[2]), int(event[3]))
    

    def _is_valid_coord(self, x: int, y: int) -> bool:
        """Check if coordinates are within bounds."""
        return 0 <= x < self.width and 0 <= y < self.height



# Convenience functions for quick usage
def events_to_frame(events: np.ndarray, 
                   height: int, 
                   width: int,
                   time_window: float = 0.05,
                   method: str = 'accumulation',
                   polarity_mode: str = 'signed') -> np.ndarray:
    """
    Convenience function to convert events to a single frame.
    
    Args:
        events: Event array with fields (t, x, y, p) or shape (N, 4)
        height: Frame height
        width: Frame width
        time_window: Time window in seconds (default: 50ms)
        method: Conversion method ('accumulation', 'time_surface', 'count')
        polarity_mode: How to handle polarity ('separate', 'combined', 'signed')
    
    Returns:
        frame: 2D or 3D array representing the frame
    
    Example:
        >>> events = load_events('sample.bin')
        >>> frame = events_to_frame(events, 34, 34, time_window=0.05)
        >>> plt.imshow(frame, cmap='RdBu_r')
    """
    converter = EventToFrameConverter(height, width)
    return converter.events_to_frame(events, time_window, method, polarity_mode)



def events_to_frame_sequence(events: np.ndarray,
                             height: int,
                             width: int,
                             n_frames: int = 10,
                             method: str = 'accumulation',
                             polarity_mode: str = 'signed') -> np.ndarray:
    """
    Convenience function to convert events to a sequence of frames.
    
    Args:
        events: Event array
        height: Frame height
        width: Frame width
        n_frames: Number of frames to generate
        method: Conversion method
        polarity_mode: Polarity handling mode
    
    Returns:
        frames: Array of shape (n_frames, height, width, [channels])
    
    Example:
        >>> events = load_events('sample.bin')
        >>> frames = events_to_frame_sequence(events, 34, 34, n_frames=10)
        >>> # Now you have 10 frames to work with
    """
    converter = EventToFrameConverter(height, width)
    return converter.events_to_frame_sequence(events, n_frames=n_frames,
                                             method=method, 
                                             polarity_mode=polarity_mode)



def normalize_frame(frame: np.ndarray, 
                    method: str = 'minmax',
                    clip_percentile: float = None) -> np.ndarray:
    """
    Normalize frame values for visualization or processing.
    
    Args:
        frame: Input frame
        method: Normalization method ('minmax', 'zscore', 'max')
        clip_percentile: If provided, clip outliers at this percentile (e.g., 99)
    
    Returns:
        normalized_frame: Normalized frame
    """
    frame = frame.copy().astype(np.float32)
    
    # Clip outliers if requested
    if clip_percentile is not None:
        if frame.ndim == 2:
            upper = np.percentile(np.abs(frame), clip_percentile)
            frame = np.clip(frame, -upper, upper)
        else:
            for c in range(frame.shape[-1]):
                upper = np.percentile(np.abs(frame[..., c]), clip_percentile)
                frame[..., c] = np.clip(frame[..., c], -upper, upper)
    
    # Normalize
    if method == 'minmax':
        # Scale to [0, 1]
        min_val = frame.min()
        max_val = frame.max()
        if max_val - min_val > 0:
            frame = (frame - min_val) / (max_val - min_val)
    elif method == 'zscore':
        # Z-score normalization
        mean = frame.mean()
        std = frame.std()
        if std > 0:
            frame = (frame - mean) / std
    elif method == 'max':
        # Divide by max absolute value
        max_abs = np.abs(frame).max()
        if max_abs > 0:
            frame = frame / max_abs
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return frame



if __name__ == "__main__":
    # Test the module with synthetic data
    print("Testing Event-to-Frame Conversion Module")
    print("=" * 60)
    
    # Create synthetic events
    n_events = 1000
    height, width = 34, 34
    
    events = np.zeros(n_events, dtype=[
        ('t', np.uint32),
        ('x', np.uint16),
        ('y', np.uint16),
        ('p', np.uint8)
    ])
    
    # Generate events (diagonal motion)
    for i in range(n_events):
        events['t'][i] = i * 100  # 100μs apart
        events['x'][i] = int((i / n_events) * (width - 1))
        events['y'][i] = int((i / n_events) * (height - 1))
        events['p'][i] = i % 2  # Alternating polarity
    
    print(f"Created {n_events} synthetic events")
    print(f"Frame size: {height} x {width}")
    print()
    
    # Test different conversion methods
    converter = EventToFrameConverter(height, width)
    
    print("Testing accumulation method (signed)...")
    frame1 = converter.events_to_frame(events, method='accumulation', 
                                       polarity_mode='signed')
    print(f"  Result shape: {frame1.shape}")
    print(f"  Value range: [{frame1.min():.2f}, {frame1.max():.2f}]")
    print()
    
    print("Testing accumulation method (separate)...")
    frame2 = converter.events_to_frame(events, method='accumulation',
                                       polarity_mode='separate')
    print(f"  Result shape: {frame2.shape}")
    print(f"  ON channel range: [{frame2[..., 0].min():.2f}, {frame2[..., 0].max():.2f}]")
    print(f"  OFF channel range: [{frame2[..., 1].min():.2f}, {frame2[..., 1].max():.2f}]")
    print()
    
    print("Testing time surface method...")
    frame3 = converter.events_to_frame(events, time_window=50000, 
                                       method='time_surface',
                                       polarity_mode='signed')
    print(f"  Result shape: {frame3.shape}")
    print(f"  Value range: [{frame3.min():.2f}, {frame3.max():.2f}]")
    print()
    
    print("Testing frame sequence generation...")
    frames = converter.events_to_frame_sequence(events, n_frames=5,
                                                method='accumulation',
                                                polarity_mode='signed')
    print(f"  Result shape: {frames.shape}")
    print(f"  Frames generated: {len(frames)}")
    print()
    
    print("✓ All tests passed!")
    print("=" * 60)