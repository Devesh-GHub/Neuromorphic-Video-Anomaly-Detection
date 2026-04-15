
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import h5py
import json
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class VideoToEventConverter:
    """
    Convert video frames to event streams using frame differencing.
    
    This simulates a Dynamic Vision Sensor (DVS) camera by detecting
    brightness changes between consecutive frames.
    """
    
    def __init__(self, 
                 contrast_threshold: float = 0.2,
                 refractory_period: int = 0,
                 sigma_threshold: float = 0.03,
                 cutoff_hz: float = 0,
                 leak_rate_hz: float = 0.1,
                 shot_noise_rate_hz: float = 0.0):
        """
        Initialize converter with v2e-like parameters.
        
        Args:
            contrast_threshold: Controls how much brightness change is needed before an event is triggered
                               Lower = more sensitive (lots of events),Higher = less sensitive (fewer events).
            refractory_period: A “cooldown” in frames before the same pixel can fire another event (0-2)
            sigma_threshold: Adds Gaussian noise to the threshold.Makes the simulation more realistic (0.01-0.05)
            cutoff_hz: Frequency cutoff for filtering.If >0, it smooths out high-frequency flickers (like removing jitter) (0 = no filter)
            leak_rate_hz: Models how pixel values “leak” over time.Prevents pixels from holding infinite memory of past brightness
            shot_noise_rate_hz: Adds random “shot noise” events.Mimics real sensors where pixels sometimes fire spontaneously
        """
        self.contrast_threshold = contrast_threshold
        self.refractory_period = refractory_period
        self.sigma_threshold = sigma_threshold
        self.cutoff_hz = cutoff_hz
        self.leak_rate_hz = leak_rate_hz
        self.shot_noise_rate_hz = shot_noise_rate_hz
        
        # State variables
        self.reference_frame = None
        self.last_event_time = None
        self.current_timestamp = 0
        
    def reset_state(self):
        """Reset converter state for new video."""
        self.reference_frame = None
        self.last_event_time = None
        self.current_timestamp = 0
    
    def load_frame(self, frame_path: Path) -> np.ndarray:
        """
        Load and preprocess a frame.
        
        Args:
            frame_path: Path to frame image
            
        Returns:
            frame: Normalized grayscale frame [0, 1]
        """
        img = Image.open(frame_path)
        frame = np.array(img, dtype=np.float32)
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]  # Grey = .299*R + 0.587*G + 0.114*B
        
        # Normalize to [0, 1]
        if frame.max() > 1.0:
            frame = frame / 255.0
        
        return frame
    
    def generate_events(self, 
                       current_frame: np.ndarray,
                       dt: float = 1000.0) -> np.ndarray:
        """
        Generate events from frame difference.
        
        Args:
            current_frame: Current frame (normalized [0, 1])
            dt: Time delta in microseconds
            
        Returns:
            events: Array of events with fields (t, x, y, p)
        """
        # Initialize reference on first frame
        if self.reference_frame is None:
            self.reference_frame = current_frame.copy()
            self.last_event_time = np.zeros_like(current_frame, dtype=np.float32)
            return np.array([], dtype=[('t', np.float64), ('x', np.uint16), 
                                      ('y', np.uint16), ('p', np.uint8)])
        
        ''' Compute log intensity difference, after normalizing we will do log for all pixel values 
            bcz event cameras respond to relative changes in brightness, not absolute
            eg: if intensity changes form 10 --> 20 it is 2x change and diff is 10 ,
                if intensity changes form 100 --> 200 here it is 2x change and diff is 100, 
                   diff of 100 > 10 this is absolute but relative it is 2x change ,so to avoid this we do log()'''
        epsilon = 1e-3   # adding some small value bcs log(0)=0 ,if we add some constant like 0.003 then value will not be 0 this increases mathematical stability 
        current_log = np.log(current_frame + epsilon)
        reference_log = np.log(self.reference_frame + epsilon)
        diff = current_log - reference_log
        
        # Add threshold noise, Each pixel has a contrast threshold: how much change is needed to trigger an event.
            # If sigma_threshold > 0, we add Gaussian noise to make thresholds slightly random.
        threshold = self.contrast_threshold
        if self.sigma_threshold > 0:
            threshold_noise = np.random.normal(0, self.sigma_threshold, diff.shape)
            threshold = threshold + threshold_noise
        
        # Detect ON and OFF events
        on_events = diff > threshold    # If brightness increased beyond threshold → ON event 
        off_events = diff < -threshold  # If brightness decreased beyond threshold → OFF event 
        
        # Apply refractory period
        if self.refractory_period > 0:
            time_since_last = self.current_timestamp - self.last_event_time
            on_events = on_events & (time_since_last >= self.refractory_period)
            off_events = off_events & (time_since_last >= self.refractory_period)
        
        on_y, on_x = np.where(on_events)
        off_y, off_x = np.where(off_events)  
        
        # Create event list
        events_list = []

        
        # ON events (polarity = 1)
        if len(on_x) > 0:
            on_t = np.full(len(on_x), self.current_timestamp, dtype=np.int64)
            on_p = np.ones(len(on_x), dtype=np.uint8)

            events_list.append(
                np.core.records.fromarrays(
                    [on_t, on_x, on_y, on_p],
                    names='t,x,y,p'
                )
            )
            self.last_event_time[on_y, on_x] = self.current_timestamp
            self.reference_frame[on_y, on_x] = current_frame[on_y, on_x]
        
        # OFF events (polarity = 0)
        if len(off_x) > 0:
            off_t = np.full(len(off_x), self.current_timestamp, dtype=np.int64)
            off_p = np.zeros(len(off_x), dtype=np.uint8)

            events_list.append(
                np.core.records.fromarrays(
                    [off_t, off_x, off_y, off_p],
                    names='t,x,y,p'
                )
            )
            self.last_event_time[off_y, off_x] = self.current_timestamp
            self.reference_frame[off_y, off_x] = current_frame[off_y, off_x]
        
        # Increment timestamp
        self.current_timestamp += dt
        
        # Convert to structured array
        if len(events_list) > 0:
            events = np.concatenate(events_list)
        else:
            events = np.empty(
                0,
                dtype=[('t', np.float64), ('x', np.uint16), ('y', np.uint16), ('p', np.uint8)]
            )
                
        return events
    
    def convert_video_sequence(self,
                              frame_paths: list,
                              fps: float = 30.0,
                              show_progress: bool = True) -> np.ndarray:
        """
        Convert sequence of frames to events.
        
        Args:
            frame_paths: List of paths to frame images (sorted)
            fps: Frames per second (for timestamp calculation)
            show_progress: Show progress bar
            
        Returns:
            events: Combined event array
        """
        self.reset_state()
        
        # Calculate time delta per frame
        dt = (1.0 / fps) * 1e6  # Convert to microseconds
        
        all_events = []
        iterator = tqdm(frame_paths, desc="Converting frames") if show_progress else frame_paths
        
        for frame_path in iterator:
            # Load frame
            frame = self.load_frame(frame_path)
            
            # Generate events
            events = self.generate_events(frame, dt=dt)
            
            if len(events) > 0:
                all_events.append(events)
        
        # Combine all events
        if len(all_events) > 0:
            combined_events = np.concatenate(all_events)
        else:
            combined_events = np.array([], dtype=[('t', np.float64), ('x', np.uint16),
                                                   ('y', np.uint16), ('p', np.uint8)])
        
        return combined_events


def convert_video_to_events(video_dir: Path,
                           output_path: Path,
                           contrast_threshold: float = 0.2,
                           fps: float = 30.0) -> dict:
    """
    Convert single video (frame sequence) to event stream.
    
    Args:
        video_dir: Directory containing frame sequence (.tif files)
        output_path: Path to save events (will create .h5 and .npy)
        contrast_threshold: Threshold for event generation
        fps: Frame rate
        
    Returns:
        stats: Dictionary with conversion statistics
    """
    # Get all frame files
    frame_files = sorted(list(video_dir.glob("*.tif")))
    
    if len(frame_files) == 0:
        print(f"Warning: No .tif files found in {video_dir}")
        return {}
    
    print(f"\nProcessing: {video_dir.name}")
    print(f"  Frames found: {len(frame_files)}")
    
    # Get frame dimensions
    first_frame = np.array(Image.open(frame_files[0]))
    height, width = first_frame.shape[:2]
    print(f"  Resolution: {width}x{height}")
    
    # Initialize converter
    converter = VideoToEventConverter(contrast_threshold=contrast_threshold)
    
    # Convert frames to events
    start_time = datetime.now()
    events = converter.convert_video_sequence(frame_files, fps=fps, show_progress=True)
    conversion_time = (datetime.now() - start_time).total_seconds()
    
    # Statistics
    stats = {
        'video_name': video_dir.name,
        'n_frames': len(frame_files),
        'resolution': (height, width),
        'n_events': len(events),
        'duration_sec': len(frame_files) / fps,
        'event_rate': len(events) / (len(frame_files) / fps) if len(frame_files) > 0 else 0,
        'conversion_time_sec': conversion_time,
        'contrast_threshold': contrast_threshold,
        'fps': fps
    }
    
    print(f"  Events generated: {stats['n_events']:,}")
    print(f"  Event rate: {stats['event_rate']:.0f} events/sec")
    print(f"  Conversion time: {conversion_time:.1f} sec")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save events in multiple formats
    if len(events) > 0:
        # HDF5 format (recommended for large files)
        h5_path = output_path.with_suffix('.h5')
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('events/t', data=events['t'], compression='gzip')
            f.create_dataset('events/x', data=events['x'], compression='gzip')
            f.create_dataset('events/y', data=events['y'], compression='gzip')
            f.create_dataset('events/p', data=events['p'], compression='gzip')
            
            # Metadata
            f.attrs['height'] = height
            f.attrs['width'] = width
            f.attrs['n_events'] = len(events)
            f.attrs['fps'] = fps
            f.attrs['contrast_threshold'] = contrast_threshold
        
        print(f"  ✓ Saved HDF5: {h5_path}")
        
        # NumPy format (for quick loading)
        npy_path = output_path.with_suffix('.npy')
        np.save(npy_path, events)
        print(f"  ✓ Saved NumPy: {npy_path}")
        
        # Save statistics as JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Saved stats: {json_path}")
    else:
        print(f"  ⚠ No events generated (check parameters)")
    
    return stats


def batch_convert(input_base_dir: Path,
                 output_base_dir: Path,
                 max_videos: int = None,
                 contrast_threshold: float = 0.2,
                 fps: float = 30.0,
                 split: str = 'Train') -> list:
    """
    Batch convert multiple video sequences to events.
    
    Args:
        input_base_dir: Base directory (e.g., data/UCSDped2)
        output_base_dir: Output base directory (e.g., data/events)
        max_videos: Maximum number of videos to convert (None = all)
        contrast_threshold: Threshold for event generation
        fps: Frame rate
        split: 'Train' or 'Test'
        
    Returns:
        all_stats: List of statistics for each video
    """
    # Get video directories
    split_dir = input_base_dir / split
    
    if not split_dir.exists():
        print(f"Error: Directory not found: {split_dir}")
        return []
    
    # Find all video directories
    video_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    
    if max_videos is not None:
        video_dirs = video_dirs[:max_videos]
    
    print("=" * 70)
    print(f"BATCH VIDEO-TO-EVENTS CONVERSION")
    print("=" * 70)
    print(f"Input directory: {input_base_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Split: {split}")
    print(f"Videos to process: {len(video_dirs)}")
    print(f"Contrast threshold: {contrast_threshold}")
    print(f"FPS: {fps}")
    print("=" * 70)
    
    # Process each video
    all_stats = []
    
    for i, video_dir in enumerate(video_dirs, 1):
        print(f"\n[{i}/{len(video_dirs)}]", end=" ")
        
        # Determine output path
        output_subdir = output_base_dir / split.lower() / video_dir.name
        output_path = output_subdir / "events"
        
        try:
            stats = convert_video_to_events(
                video_dir=video_dir,
                output_path=output_path,
                contrast_threshold=contrast_threshold,
                fps=fps
            )
            
            if stats:
                all_stats.append(stats)
        
        except Exception as e:
            print(f"  ✗ Error processing {video_dir.name}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    print(f"Videos processed: {len(all_stats)}/{len(video_dirs)}")
    
    if len(all_stats) > 0:
        total_events = sum(s['n_events'] for s in all_stats)
        total_frames = sum(s['n_frames'] for s in all_stats)
        total_time = sum(s['conversion_time_sec'] for s in all_stats)
        
        print(f"Total frames: {total_frames:,}")
        print(f"Total events: {total_events:,}")
        print(f"Average events/frame: {total_events/total_frames:.1f}")
        print(f"Total conversion time: {total_time:.1f} sec ({total_time/60:.1f} min)")
        
        # Save summary
        summary_path = output_base_dir / f"{split.lower()}_conversion_summary.json"
        summary = {
            'n_videos': len(all_stats),
            'total_frames': total_frames,
            'total_events': int(total_events),
            'avg_events_per_frame': total_events / total_frames,
            'total_conversion_time_sec': total_time,
            'parameters': {
                'contrast_threshold': contrast_threshold,
                'fps': fps
            },
            'videos': all_stats
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved: {summary_path}")
    
    print("=" * 70)
    
    return all_stats


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Convert UCSD Ped2 videos to event streams'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default='./data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2',
        help='Input directory (UCSDped2 base folder)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./events',
        help='Output directory for event data'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='Train',
        choices=['Train', 'Test'],
        help='Dataset split to process'
    )
    
    parser.add_argument(
        '--max_videos',
        type=int,
        default=10,
        help='Maximum number of videos to convert (None = all)'
    )
    
    parser.add_argument(
        '--contrast_threshold',
        type=float,
        default=0.2,
        help='Contrast threshold for event generation (0.1-0.3)'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Frame rate (frames per second)'
    )
    
    parser.add_argument(
        '--single_video',
        type=str,
        default=None,
        help='Convert single video directory (e.g., Train001)'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Check input directory
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Please update --input_dir to point to your UCSDped2 folder")
        return
    
    # Single video or batch
    if args.single_video:
        video_path = input_dir / args.split / args.single_video
        output_path = output_dir / args.split.lower() / args.single_video / "events"
        
        if video_path.exists():
            convert_video_to_events(
                video_dir=video_path,
                output_path=output_path,
                contrast_threshold=args.contrast_threshold,
                fps=args.fps
            )
        else:
            print(f"Error: Video directory not found: {video_path}")
    else:
        # Batch conversion
        batch_convert(
            input_base_dir=input_dir,
            output_base_dir=output_dir,
            max_videos=args.max_videos,
            contrast_threshold=args.contrast_threshold,
            fps=args.fps,
            split=args.split
        )


if __name__ == "__main__":
    main()