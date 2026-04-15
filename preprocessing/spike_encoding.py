
import numpy as np
from typing import Union, List, Tuple, Optional


class SpikeEncoder:
    """
    Encode analog signals into spike trains for SNNs.
    
    Supports multiple encoding schemes suitable for different applications.
    """
    
    def __init__(self, duration: float = 100.0, dt: float = 1.0):
        """
        Initialize spike encoder.
        
        Args:
            duration: Encoding duration in milliseconds (default: 100ms)
            dt: Time step in milliseconds (default: 1ms)
        """
        self.duration = duration
        self.dt = dt
        self.time_steps = int(duration / dt)
        
    def rate_encode(self, 
                   value: Union[float, np.ndarray],
                   max_rate: float = 100.0,   # It means neuron can fire at most 100 spikes per sec
                   method: str = 'poisson') -> Union[np.ndarray, List[float]]:
        """
        Rate coding: Encode value as firing rate (Hz).
        
        Higher values → more spikes per unit time
        
        Args:
            value: Input value(s) normalized to [0, 1]
            max_rate: Maximum firing rate in Hz
            method: 'poisson' for stochastic or 'regular' for deterministic
            
        Returns:
            spike_train: Binary spike train or list of spike times
        """
        if np.isscalar(value):
            return self._rate_encode_single(value, max_rate, method)  # Simple value will be converted into range(0,1) eg:value = 5 --> 0.05
        else:
            # Handle array of values
            if value.ndim == 1:
                return np.array([self._rate_encode_single(v, max_rate, method)  # value = 1D [] then we convert each value into range(0,1), eg:[22,45,63] --> [0.22,0.45,0.63]
                               for v in value])
            else:
                # Handle 2D image
                shape = value.shape                    
                flat_values = value.flatten()                               # If value = 2D [[]] then we covert 2D --> 1D --> range(0,1)
                encoded = [self._rate_encode_single(v, max_rate, method) 
                          for v in flat_values]
                
                # Reshape back --> will add time for each value for single value --> (num,time); for 1D --> 2D(becomes) [[22,time],[33,time],[56,time]]; for 2D --> 3D(becomes) [[22,32,time],[11,34,time],[59,65,78,time]]
                if isinstance(encoded[0], np.ndarray):                          
                    return np.array(encoded).reshape(shape + (self.time_steps,))   # time_steps dimension indicates when spikes occur
                else:
                    return np.array(encoded).reshape(shape)
    
    def _rate_encode_single(self, 
                           value: float, 
                           max_rate: float,
                           method: str) -> Union[np.ndarray, List[float]]:
        """Encode single value with rate coding."""
        # Ensure value is in [0, 1]
        value = np.clip(value, 0.0, 1.0)
        
        # Calculate firing rate
        rate = value * max_rate  # Hz
        
        if method == 'poisson':  # --> Poisson returns binary spike train eg:-[1,0,0,1,0,0,1]
            ''' Poisson spike train (stochastic) --> spikes are not generated at perticular intervalus they are randomly generated,
                but at last all spikes will be generated witin duration '''
            spike_train = self._generate_poisson_spikes(rate)
            return spike_train
        
        elif method == 'regular': # --> Regular returns spike times eg:= [20,40,60,80,100] this is time 20ms,40ms,60ms etc ..
            ''' Regular spike train (deterministic) --> this will give the time frame of each spike generated
               eg:- [20ms,40ms,60ms,80ms] first spike at 20ms then at eavery 20ms duration each spike will be generated '''
            spike_times = self._generate_regular_spikes(rate)
            return spike_times
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _generate_poisson_spikes(self, rate: float) -> np.ndarray:
        """
        Generate Poisson spike train.
        
        Args:
            rate: Firing rate in Hz
            
        Returns:
            spike_train: Binary array of length time_steps
        """
        # Probability of spike in each time bin
        # rate (Hz) * dt (ms) / 1000 (ms/s) = probability per bin
        spike_prob = rate * self.dt / 1000.0
        
        # Generate random spikes
        spike_train = np.random.rand(self.time_steps) < spike_prob
        
        return spike_train.astype(np.float32)
    
    def _generate_regular_spikes(self, rate: float) -> List[float]:
        """
        Generate regular (periodic) spike train.
        
        Args:
            rate: Firing rate in Hz
            
        Returns:
            spike_times: List of spike times in ms
        """
        if rate <= 0:
            return []
        
        # Inter-spike interval in ms
        isi = 1000.0 / rate  # Convert Hz to ms
        
        # Generate spike times
        spike_times = []
        t = isi  # Start after first ISI
        while t < self.duration:
            spike_times.append(t)
            t += isi
        
        return spike_times
    
    def temporal_encode(self,
                       value: Union[float, np.ndarray],
                       method: str = 'latency') -> Union[float, np.ndarray]:
        """
        Temporal coding: Encode value as spike timing.
        
        Early spikes → higher values
        Late spikes → lower values
        
        Args:
            value: Input value(s) normalized to [0, 1]
            method: 'latency' or 'ttfs' (time-to-first-spike)
            
        Returns:
            spike_time: Time of first spike in ms
        """
        if method in ['latency', 'ttfs']:
            return self._latency_encode(value)
        else:
            raise ValueError(f"Unknown temporal method: {method}")
    
    def _latency_encode(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Latency coding: First spike time encodes value.
        
        High values → early spikes
        Low values → late spikes
        """
        # Ensure values in [0, 1]
        value = np.clip(value, 0.0, 1.0)
        
        # Map value to spike time: high value → early spike
        # latency = duration * (1 - value)
        spike_time = self.duration * (1.0 - value)
        
        return spike_time
    
    def population_encode(self,
                         value: float,
                         n_neurons: int = 10,
                         encoding_range: Tuple[float, float] = (0.0, 1.0),
                         sigma: float = 0.15) -> np.ndarray:
        """
        Population coding: Multiple neurons with overlapping receptive fields.
        
        Each neuron has a preferred value and fires maximally for that value.
        
        Args:
            value: Input value to encode
            n_neurons: Number of neurons in population
            encoding_range: Range of values (min, max)
            sigma: Width of Gaussian receptive field
            
        Returns:
            firing_rates: Array of firing rates for each neuron (normalized 0-1)
        """
        # Create neuron preferred values (centers)
        min_val, max_val = encoding_range
        centers = np.linspace(min_val, max_val, n_neurons)
        
        # Compute Gaussian responses
        responses = np.exp(-((value - centers) ** 2) / (2 * sigma ** 2))
        
        # Normalize
        responses = responses / (responses.sum() + 1e-8)
        
        return responses
    
    def encode_image(self,
                    image: np.ndarray,
                    method: str = 'rate',
                    max_rate: float = 100.0,
                    normalize: bool = True) -> np.ndarray:
        """
        Encode entire image into spike trains.
        
        Args:
            image: 2D image array
            method: Encoding method ('rate', 'latency')
            max_rate: Maximum firing rate for rate coding
            normalize: Whether to normalize image to [0, 1]
            
        Returns:
            encoded: Encoded spike trains
                - For rate: (height, width, time_steps)
                - For latency: (height, width) with spike times
        """
        # Normalize image if needed
        if normalize:
            image = self._normalize_image(image)
        
        if method == 'rate':
            return self.rate_encode(image, max_rate, method='poisson')
        elif method == 'latency':
            return self.temporal_encode(image, method='latency')
        else:
            raise ValueError(f"Unknown encoding method: {method}")
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1]."""
        img_min = image.min()
        img_max = image.max()
        
        if img_max - img_min > 0:
            return (image - img_min) / (img_max - img_min)
        else:
            return np.zeros_like(image)
    
    def decode_rate(self,
                   spike_train: np.ndarray,
                   max_rate: float = 100.0) -> float:
        """
        Decode rate-coded spike train back to value.
        
        Args:
            spike_train: Binary spike train
            max_rate: Maximum firing rate used in encoding
            
        Returns:
            value: Decoded value in [0, 1]
        """
        # Count spikes
        spike_count = np.sum(spike_train)
        
        # Calculate rate in Hz
        rate = spike_count / (self.duration / 1000.0)
        
        # Decode to value
        value = rate / max_rate
        
        return np.clip(value, 0.0, 1.0)
    
    def visualize_spike_train(self, spike_train: np.ndarray) -> None:
        """
        Print text visualization of spike train.
        
        Args:
            spike_train: Binary spike train
        """
        visualization = ''.join(['|' if s else '.' for s in spike_train])
        print(f"Spike train: {visualization}")
        print(f"Total spikes: {np.sum(spike_train)}")


# Convenience functions
def rate_encode(pixel_value: Union[float, np.ndarray],
               max_rate: float = 100.0,
               duration: float = 100.0,
               dt: float = 1.0) -> Union[np.ndarray, List[float]]:
    """
    Convert pixel intensity to spike train using rate coding.
    
    Args:
        pixel_value: Normalized value [0, 1] or array of values
        max_rate: Maximum firing rate (Hz)
        duration: Encoding duration (ms)
        dt: Time step (ms)
    
    Returns:
        spike_train: Binary spike train or array of spike trains
    
    Example:
        >>> spikes = rate_encode(0.8, max_rate=100, duration=100)
        >>> print(f"Generated {np.sum(spikes)} spikes")
    """
    encoder = SpikeEncoder(duration=duration, dt=dt)
    return encoder.rate_encode(pixel_value, max_rate, method='poisson')


def temporal_encode(pixel_value: Union[float, np.ndarray],
                   duration: float = 100.0) -> Union[float, np.ndarray]:
    """
    Convert pixel intensity to spike time using temporal coding.
    
    Args:
        pixel_value: Normalized value [0, 1] or array of values
        duration: Maximum encoding time (ms)
    
    Returns:
        spike_time: Time of spike in ms
    
    Example:
        >>> spike_time = temporal_encode(0.8, duration=100)
        >>> print(f"Spike at {spike_time:.1f} ms")
    """
    encoder = SpikeEncoder(duration=duration)
    return encoder.temporal_encode(pixel_value, method='latency')


def population_encode(value: float,
                     n_neurons: int = 10,
                     sigma: float = 0.15) -> np.ndarray:
    """
    Encode value using population of neurons.
    
    Args:
        value: Input value [0, 1]
        n_neurons: Number of neurons in population
        sigma: Receptive field width
    
    Returns:
        firing_rates: Normalized firing rates for each neuron
    
    Example:
        >>> rates = population_encode(0.5, n_neurons=10)
        >>> print(f"Peak neuron: {np.argmax(rates)}")
    """
    encoder = SpikeEncoder()
    return encoder.population_encode(value, n_neurons, (0.0, 1.0), sigma)


if __name__ == "__main__":
    # Test the module
    print("=" * 70)
    print("SPIKE ENCODING MODULE - TESTS")
    print("=" * 70)
    print()
    
    # Test 1: Rate encoding (single value)
    print("TEST 1: Rate Encoding (Single Value)")
    print("-" * 70)
    encoder = SpikeEncoder(duration=100, dt=1.0)
    
    test_values = [0.2, 0.5, 0.8, 1.0]
    for val in test_values:
        spikes = encoder.rate_encode(val, max_rate=100, method='poisson')
        spike_count = np.sum(spikes)
        expected_rate = val * 100
        actual_rate = spike_count / 0.1  # 100ms = 0.1s
        print(f"Value: {val:.1f} → Spikes: {spike_count} "
              f"(Expected rate: {expected_rate:.0f} Hz, Actual: {actual_rate:.0f} Hz)")
    print()
    
    # Test 2: Regular vs Poisson
    print("TEST 2: Regular vs Poisson Spike Trains")
    print("-" * 70)
    value = 0.5
    
    poisson_spikes = encoder.rate_encode(value, max_rate=100, method='poisson')
    regular_times = encoder.rate_encode(value, max_rate=100, method='regular')
    
    print(f"Value: {value}")
    print(f"Poisson: {np.sum(poisson_spikes)} spikes")
    print(f"Regular: {len(regular_times)} spikes at times: {regular_times[:5]}...")
    print()
    
    # Test 3: Temporal encoding
    print("TEST 3: Temporal Encoding (Latency)")
    print("-" * 70)
    for val in test_values:
        spike_time = encoder.temporal_encode(val, method='latency')
        print(f"Value: {val:.1f} → Spike time: {spike_time:.1f} ms")
    print()
    
    # Test 4: Population encoding
    print("TEST 4: Population Encoding")
    print("-" * 70)
    value = 0.5
    n_neurons = 10
    rates = encoder.population_encode(value, n_neurons=n_neurons)
    
    print(f"Value: {value}")
    print(f"Population size: {n_neurons}")
    print(f"Firing rates: {rates}")
    print(f"Peak neuron: {np.argmax(rates)} (rate: {rates.max():.3f})")
    print()
    
    # Test 5: Image encoding
    print("TEST 5: Image Encoding")
    print("-" * 70)
    test_image = np.random.rand(5, 5)
    print(f"Test image shape: {test_image.shape}")
    
    # Rate encoding
    encoded_rate = encoder.encode_image(test_image, method='rate', max_rate=100)
    print(f"Rate encoded shape: {encoded_rate.shape}")
    print(f"Total spikes: {np.sum(encoded_rate)}")
    
    # Latency encoding
    encoded_latency = encoder.encode_image(test_image, method='latency')
    print(f"Latency encoded shape: {encoded_latency.shape}")
    print(f"Spike time range: [{encoded_latency.min():.1f}, {encoded_latency.max():.1f}] ms")
    print()
    
    # Test 6: Decoding
    print("TEST 6: Rate Decoding")
    print("-" * 70)
    original = 0.7
    spikes = encoder.rate_encode(original, max_rate=100, method='poisson')
    decoded = encoder.decode_rate(spikes, max_rate=100)
    error = abs(original - decoded)
    print(f"Original value: {original:.2f}")
    print(f"Decoded value:  {decoded:.2f}")
    print(f"Error:          {error:.2f}")
    print()
    
    # Test 7: Visualization
    print("TEST 7: Spike Train Visualization")
    print("-" * 70)
    spikes = encoder.rate_encode(0.3, max_rate=50, method='poisson')
    encoder.visualize_spike_train(spikes[:50])  # Show first 50ms
    print()
    
    print("=" * 70)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)