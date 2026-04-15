
import numpy as np
import h5py
from pathlib import Path
from typing import Union, Tuple, Optional, List
import warnings


# ============================================================================
# LOADING FUNCTIONS
# ============================================================================

def load_events(file_path: Union[str, Path],
                format: str = 'auto') -> np.ndarray:
    """
    Load events from file (.h5, .npy, or .bin).
    
    Args:
        file_path: Path to event file
        format: File format ('auto', 'h5', 'npy', 'bin')
                'auto' detects from extension
    
    Returns:
        events: Structured array with fields (t, x, y, p)
        
    Example:
        >>> events = load_events('events/train/Train001/events.h5')
        >>> print(f"Loaded {len(events)} events")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect format
    if format == 'auto':
        ext = file_path.suffix.lower()
        if ext == '.h5':
            format = 'h5'
        elif ext == '.npy':
            format = 'npy'
        else:
            raise ValueError(f"Unknown file extension: {ext}")
    
    # Load based on format
    if format == 'h5':
        return load_events_h5(file_path)
    elif format == 'npy':
        return load_events_npy(file_path)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_events_h5(file_path: Path) -> np.ndarray:
    """Load events from HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        # Check structure
        if 'events' in f:
            t = f['events/t'][:]
            x = f['events/x'][:]
            y = f['events/y'][:]
            p = f['events/p'][:]
        else:
            # Alternative structure
            t = f['t'][:]
            x = f['x'][:]
            y = f['y'][:]
            p = f['p'][:]
        
        # Create structured array
        n_events = len(t)
        events = np.zeros(n_events, dtype=[
            ('t', np.float64),
            ('x', np.uint16),
            ('y', np.uint16),
            ('p', np.uint8)
        ])
        
        events['t'] = t
        events['x'] = x
        events['y'] = y
        events['p'] = p
    
    return events


def load_events_npy(file_path: Path) -> np.ndarray:
    """Load events from NumPy file."""
    events = np.load(file_path)
    
    # Ensure correct dtype
    if events.dtype.names is None:
        # Convert from regular array to structured array
        n_events = len(events)
        structured = np.zeros(n_events, dtype=[
            ('t', np.float64),
            ('x', np.uint16),
            ('y', np.uint16),
            ('p', np.uint8)
        ])
        structured['t'] = events[:, 0]
        structured['x'] = events[:, 1]
        structured['y'] = events[:, 2]
        structured['p'] = events[:, 3]
        events = structured
    
    return events


# ============================================================================
# FILTERING FUNCTIONS
# ============================================================================

def filter_events(events: np.ndarray,
                 spatial_filter: Optional[Tuple[int, int, int, int]] = None,
                 temporal_filter: Optional[Tuple[float, float]] = None,
                 polarity_filter: Optional[int] = None) -> np.ndarray:
    """
    Apply spatial and/or temporal filtering to events.
    
    Args:
        events: Event array
        spatial_filter: (x_min, x_max, y_min, y_max) → keep only events inside a rectangular region
        temporal_filter: (t_start, t_end) → keep only events within a time window
        polarity_filter: 0 or 1 → keep only OFF or ON events 
    
    Returns:
        filtered_events: Filtered event array
        
    Example:
        >>> # Keep only events in center region, first half of time
        >>> filtered = filter_events(events,
        ...                          spatial_filter=(100, 260, 60, 180),
        ...                          temporal_filter=(0, events['t'].max()/2))
    """
    if len(events) == 0:
        return events

    mask = np.ones(len(events), dtype=bool)
    
    # Spatial filtering
    if spatial_filter is not None:
        x_min, x_max, y_min, y_max = spatial_filter
        mask &= (events['x'] >= x_min) & (events['x'] <= x_max)  # &= means “update mask” → only keep events that satisfy these conditions
        mask &= (events['y'] >= y_min) & (events['y'] <= y_max)
    
    # Temporal filtering
    if temporal_filter is not None:
        t_start, t_end = temporal_filter
        mask &= (events['t'] >= t_start) & (events['t'] <= t_end)
    
    # Polarity filtering
    if polarity_filter is not None:
        mask &= (events['p'] == polarity_filter)
    
    return events[mask]


# def denoise_events(events: np.ndarray,
                #   method: str = 'spatial',
                #   radius: int = 1,
                #   time_window: float = 10000.0) -> np.ndarray:
    """
    Remove noise events using spatial or temporal filtering.
    
    Args:
        events: Event array
        method: 'spatial' or 'temporal'
        radius: Spatial radius for neighborhood (pixels)
        time_window: Time window for temporal filtering (microseconds)
    
    Returns:
        denoised_events: Events with noise removed
        
    Example:
        >>> denoised = denoise_events(events, method='spatial', radius=1)
    """
    # if method == 'spatial':                          # Remove events that are isolated in space (no nearby pixels firing around the same time)
    #     return _denoise_spatial(events, radius) 
    # elif method == 'temporal':                       # Remove events that are isolated in time (no repeated firing at the same pixel within a short window)
    #     return _denoise_temporal(events, time_window)
    # else:
    #     raise ValueError(f"Unknown denoising method: {method}")


# def _denoise_spatial(events: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Remove isolated events (no neighbors within radius).

    Imagine a pixel at (100, 50) fires once, but no other nearby pixels fire.
    That's likely noise → discard.
    If multiple pixels around (100, 50) fire together, that's real motion → keep.
    
    Simple implementation: Keep event if it has at least one neighbor within spatial radius.
    """
    if len(events) == 0:
        return events
    
    keep_mask = np.zeros(len(events), dtype=bool)
    
    # Sort by time for efficiency
    time_sorted_idx = np.argsort(events['t'])
    sorted_events = events[time_sorted_idx]
    
    # Check each event for neighbors
    for i in range(len(sorted_events)):
        # Look in small time window around this event
        t_center = sorted_events['t'][i]
        time_mask = np.abs(sorted_events['t'] - t_center) < 50000  # 50ms window
        
        nearby_events = sorted_events[time_mask]
        
        # Check for spatial neighbors
        x_diff = np.abs(nearby_events['x'] - sorted_events['x'][i])
        y_diff = np.abs(nearby_events['y'] - sorted_events['y'][i])
        
        has_neighbor = np.any((x_diff <= radius) & (y_diff <= radius) & 
                             ((x_diff + y_diff) > 0))  # Exclude self
        
        if has_neighbor:
            keep_mask[time_sorted_idx[i]] = True
    
    return events[keep_mask]


# def _denoise_temporal(events: np.ndarray, time_window: float) -> np.ndarray:
    """
    Remove events with no temporal neighbors within time window.

    Imagine pixel (200, 100) fires once at time 1.0s, but never again.
    That's likely noise → discard.
    If the same pixel fires multiple times within a short window, that's real motion → keep.
    """
    if len(events) == 0:
        return events
    
    keep_mask = np.zeros(len(events), dtype=bool)
    
    for i in range(len(events)):
        t = events['t'][i]
        x, y = events['x'][i], events['y'][i]
        
        # Find events at same location within time window
        same_location = (events['x'] == x) & (events['y'] == y)
        in_time_window = np.abs(events['t'] - t) < time_window
        
        neighbors = same_location & in_time_window & (np.arange(len(events)) != i)
        
        if np.any(neighbors):
            keep_mask[i] = True
    
    return events[keep_mask]


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_events(events: np.ndarray,
                    time_normalization: bool = True,
                    spatial_normalization: bool = False,
                    resolution: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Normalize event timestamps and/or spatial coordinates.
    
    Args:
        events: Event array
        time_normalization: Normalize timestamps to [0, 1]
        spatial_normalization: Normalize coordinates to [0, 1]
        resolution: (height, width) for spatial normalization
    
    Returns:
        normalized_events: Normalized event array (copy)
        
    Example:
        >>> norm_events = normalize_events(events, 
        ...                                time_normalization=True,
        ...                                spatial_normalization=True,
        ...                                resolution=(240, 360))
    """
    if len(events) == 0:
        return events.copy()

    normalized = events.copy()
    
    if time_normalization and len(events) > 0:
        t_min = events['t'].min()
        t_max = events['t'].max()
        
        if t_max > t_min:
            # Normalize to [0, 1]
            normalized['t'] = (events['t'] - t_min) / (t_max - t_min)
    
    if spatial_normalization:
        if resolution is None:
            # Use max coordinates
            x_max = events['x'].max()
            y_max = events['y'].max()
        else:
            y_max, x_max = resolution
        
        if x_max > 0:
            normalized['x'] = events['x'] / x_max
        if y_max > 0:
            normalized['y'] = events['y'] / y_max
    
    return normalized


def standardize_timestamps(events: np.ndarray,
                          start_time: float = 0.0) -> np.ndarray:
    """
    Shift timestamps to start from specified time.
    
    Args:
        events: Event array
        start_time: Desired start time (default: 0.0)
    
    Returns:
        shifted_events: Events with shifted timestamps
    """
    shifted = events.copy()
    
    if len(events) > 0:
        t_min = events['t'].min()
        shifted['t'] = events['t'] - t_min + start_time
    
    return shifted


# ============================================================================
# TIME WINDOW FUNCTIONS
# ============================================================================

def split_events_by_time(events: np.ndarray,
                        window_size: float = 100000.0,
                        overlap: float = 0.0) -> List[np.ndarray]:
    """
    Split event stream into time windows.
    
    Args:
        events: Event array (must be sorted by time)
        window_size: Window size in microseconds (or number of events if method='count')
        overlap: Overlap fraction (0.0 to 0.99)
        method: 'fixed' (fixed time windows) or 'count' (fixed event count)
    
    Returns:
        windows: List of event arrays, one per window
        
    Example:
        >>> # Split into 100ms windows with 50% overlap
        >>> windows = split_events_by_time(events, 
        ...                                window_size=100000, 
        ...                                overlap=0.5)
        >>> print(f"Created {len(windows)} windows")
    """
    if len(events) == 0:
        return []
    
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0)")

    times = events['t']
    windows = []

    step_size = window_size * (1.0 - overlap)

    t_start = times[0]
    t_end = times[-1]

    idx_start = 0

    while t_start < t_end:
        t_window_end = t_start + window_size

        idx_end = np.searchsorted(times, t_window_end, side='left')

        if idx_end > idx_start:
            windows.append(events[idx_start:idx_end])

        t_start += step_size
        idx_start = np.searchsorted(times, t_start, side='left')

    return windows


def _split_by_fixed_time(events: np.ndarray,
                         window_size: float,
                         overlap: float) -> List[np.ndarray]:
    """Split by fixed time windows."""
    windows = []
    
    t_min = events['t'].min()
    t_max = events['t'].max()
    
    # Calculate step size
    step_size = window_size * (1.0 - overlap)
    
    t_start = t_min
    while t_start < t_max:
        t_end = t_start + window_size
        
        # Extract events in window
        mask = (events['t'] >= t_start) & (events['t'] < t_end)
        window_events = events[mask]
        
        if len(window_events) > 0:
            windows.append(window_events)
        
        t_start += step_size
    
    return windows


# def _split_by_count(events: np.ndarray,
    #                 events_per_window: int,
    #                 overlap: float) -> List[np.ndarray]:
    # """Split by fixed event count."""
    # windows = []
    
    # step_size = int(events_per_window * (1.0 - overlap))
    
    # start_idx = 0
    # while start_idx < len(events):
    #     end_idx = start_idx + events_per_window
        
    #     window_events = events[start_idx:end_idx]
        
    #     if len(window_events) > 0:
    #         windows.append(window_events)
        
    #     start_idx += step_size
        
    #     # Stop if we can't make a full window
    #     if end_idx >= len(events) and start_idx < len(events):
    #         # Add final partial window
    #         windows.append(events[start_idx:])
    #         break
    
    # return windows


# ============================================================================
# SUBSAMPLING FUNCTIONS
# ============================================================================

# # def subsample_events(events: np.ndarray,
#                     method: str = 'random',
#                     factor: float = 0.5) -> np.ndarray:
    """
    Subsample events to reduce density.
    
    Args:
        events: Event array
        method: 'random' (random selection) or 'uniform' (keep every Nth)
        factor: Fraction to keep (0.0 to 1.0)
    
    Returns:
        subsampled_events: Reduced event array
        
    Example:
        >>> # Keep 50% of events randomly
        >>> subsampled = subsample_events(events, method='random', factor=0.5)
    # """
    # if factor >= 1.0:
    #     return events.copy()
    
    # n_keep = int(len(events) * factor)
    
    # if method == 'random':
    #     indices = np.random.choice(len(events), n_keep, replace=False)
    #     indices = np.sort(indices)  # Maintain temporal order
    #     return events[indices]
    
    # elif method == 'uniform':
    #     step = int(1.0 / factor)
    #     return events[::step]
    
    # else:
    #     raise ValueError(f"Unknown method: {method}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_event_statistics(events: np.ndarray,
                        resolution: Optional[Tuple[int, int]] = None) -> dict:
    """
    Compute comprehensive statistics for event stream.
    
    Args:
        events: Event array
        resolution: (height, width) for spatial statistics
    
    Returns:
        stats: Dictionary with statistics
    """
    if len(events) == 0:
        return {'n_events': 0}
    
    stats = {
        # Basic counts
        'n_events': len(events),
        'n_on': int(np.sum(events['p'] == 1)),
        'n_off': int(np.sum(events['p'] == 0)),
        
        # Temporal
        't_min': float(events['t'].min()),
        't_max': float(events['t'].max()),
        'duration': float(events['t'].max() - events['t'].min()),
        
        # Spatial
        'x_min': int(events['x'].min()),
        'x_max': int(events['x'].max()),
        'y_min': int(events['y'].min()),
        'y_max': int(events['y'].max()),
        
        # Polarity
        'on_ratio': float(np.sum(events['p'] == 1) / len(events)),
        'off_ratio': float(np.sum(events['p'] == 0) / len(events)),
    }
    
    # Event rate
    if stats['duration'] > 0:
        stats['event_rate'] = stats['n_events'] / (stats['duration'] / 1e6)  # events/sec
    else:
        stats['event_rate'] = 0.0
    
    # Spatial coverage
    if resolution is not None:
        height, width = resolution
        active_pixels = len(np.unique(events['y'] * width + events['x']))
        total_pixels = height * width
        stats['spatial_coverage'] = active_pixels / total_pixels
    
    return stats


def print_event_statistics(events: np.ndarray,
                          name: str = "Event Stream") -> None:
    """
    Print formatted event statistics.
    
    Args:
        events: Event array
        name: Name for the event stream
    """
    stats = get_event_statistics(events)
    
    print(f"\n{name}")
    print("=" * 60)
    print(f"Total events:     {stats['n_events']:,}")
    print(f"  ON events:      {stats['n_on']:,} ({stats['on_ratio']*100:.1f}%)")
    print(f"  OFF events:     {stats['n_off']:,} ({stats['off_ratio']*100:.1f}%)")
    print(f"\nTemporal:")
    print(f"  Duration:       {stats['duration']/1000:.2f} ms")
    print(f"  Event rate:     {stats['event_rate']:.0f} events/sec")
    print(f"\nSpatial:")
    print(f"  X range:        [{stats['x_min']}, {stats['x_max']}]")
    print(f"  Y range:        [{stats['y_min']}, {stats['y_max']}]")
    print("=" * 60)


if __name__ == "__main__":
    # Test the module
    print("=" * 70)
    print("EVENT PREPROCESSING MODULE - TESTS")
    print("=" * 70)
    print()
    
    # Create synthetic test data
    print("Creating synthetic event data...")
    n_events = 10000
    test_events = np.zeros(n_events, dtype=[
        ('t', np.float64),
        ('x', np.uint16),
        ('y', np.uint16),
        ('p', np.uint8)
    ])
    
    test_events['t'] = np.sort(np.random.rand(n_events) * 100000)  # 100ms
    test_events['x'] = np.random.randint(0, 360, n_events)
    test_events['y'] = np.random.randint(0, 240, n_events)
    test_events['p'] = np.random.randint(0, 2, n_events)
    
    print(f"Created {len(test_events)} test events")
    print()
    
    # Test statistics
    print("TEST 1: Event Statistics")
    print("-" * 70)
    print_event_statistics(test_events, "Test Events")
    print()
    
    # Test filtering
    print("TEST 2: Spatial Filtering")
    print("-" * 70)
    filtered = filter_events(test_events, spatial_filter=(100, 260, 60, 180))
    print(f"Original: {len(test_events)} events")
    print(f"Filtered: {len(filtered)} events ({len(filtered)/len(test_events)*100:.1f}%)")
    print()
    
    # Test normalization
    print("TEST 3: Normalization")
    print("-" * 70)
    normalized = normalize_events(test_events, time_normalization=True)
    print(f"Original time range: [{test_events['t'].min():.0f}, {test_events['t'].max():.0f}]")
    print(f"Normalized time range: [{normalized['t'].min():.3f}, {normalized['t'].max():.3f}]")
    print()
    
    # Test time windows
    print("TEST 4: Time Window Splitting")
    print("-" * 70)
    windows = split_events_by_time(test_events, window_size=20000, overlap=0.0)
    print(f"Split into {len(windows)} windows")
    print(f"Events per window: {[len(w) for w in windows[:5]]}...")
    print()
    

