
from brian2 import *
import numpy as np
from typing import Tuple, Optional, Dict


class SNNAutoencoder:
    """
    Spiking Neural Network Autoencoder for anomaly detection in event-based video.
    
    Uses Brian2 for SNN simulation with LIF neurons.
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (2, 240, 360),
                 hidden_channels: Tuple[int, int, int] = (128, 64, 32),
                 tau: float = 20.0,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 v_rest: float = 0.0,
                 dt: float = 1.0,
                 simulation_time: float = 100.0):
        """
        Initialize SNN Autoencoder.
        
        Args:
            input_shape: (channels, height, width) - (2, 240, 360) for UCSD Ped2
            hidden_channels: (enc1, enc2, bottleneck) - (128, 64, 32)
            tau: Membrane time constant (ms)
            v_threshold: Spike threshold voltage
            v_reset: Reset voltage after spike
            v_rest: Resting potential
            dt: Simulation time step (ms)
            simulation_time: Total simulation time per sample (ms)
        """
        self.input_shape = input_shape
        self.c_in, self.h_in, self.w_in = input_shape
        
        # Hidden layer sizes
        self.c_enc1, self.c_enc2, self.c_bottleneck = hidden_channels
        
        # LIF neuron parameters
        self.tau = tau * ms
        self.v_threshold = v_threshold * volt
        self.v_reset = v_reset * volt
        self.v_rest = v_rest * volt
        
        # Simulation parameters
        self.dt = dt * ms
        self.simulation_time = simulation_time * ms
        
        # Calculate layer dimensions
        self._calculate_layer_dimensions()
        
        # Define neuron equations
        self._define_neuron_model()
        
        # Build network
        self._build_network()
        
    
    def _calculate_layer_dimensions(self):
        """Calculate spatial dimensions for each layer."""
        # Encoder Layer 1: 5×5, stride=2
        self.h_enc1 = (self.h_in - 5 + 2 * 2) // 2 + 1  # 240 → 120 (This is the foumula for conv2D)
        self.w_enc1 = (self.w_in - 5 + 2 * 2) // 2 + 1  # 360 → 180
        
        # Encoder Layer 2: 3×3, stride=2
        self.h_enc2 = (self.h_enc1 - 3 + 2 * 1) // 2 + 1  # 120 → 60
        self.w_enc2 = (self.w_enc1 - 3 + 2 * 1) // 2 + 1  # 180 → 90
        
        # Bottleneck: 1×1 (same spatial size)
        self.h_bottleneck = self.h_enc2  # 60
        self.w_bottleneck = self.w_enc2  # 90
        
        # Decoder mirrors encoder
        self.h_dec1 = self.h_enc1  # 120  (There is a forumula for ConvTranspose2D --> this will give same result as enc1 so we directly assigned value of enc1)
        self.w_dec1 = self.w_enc1  # 180
        
        self.h_out = self.h_in  # 240
        self.w_out = self.w_in  # 360
    
    def _define_neuron_model(self):
        """Define LIF neuron equations for Brian2."""
        # Leaky Integrate-and-Fire model
        self.neuron_eqs = '''
        dv/dt = (v_rest - v) / tau : volt (unless refractory)
        v_rest : volt (constant)
        tau : second (constant)
        '''
        
        # Threshold and reset - extract numeric values for string expressions
        # Brian2 expects clean numeric values with explicit units in threshold/reset strings
        v_thresh_val = float(self.v_threshold / volt)  # converting v_threshould into volt
        v_reset_val = float(self.v_reset / volt)
        
        self.threshold = f'v > {v_thresh_val}*volt'
        self.reset = f'v = {v_reset_val}*volt'
        self.refractory = 2 * ms  # 2ms refractory period
        
    
    def _build_network(self):
        """Build the complete SNN architecture."""
        print("\nBuilding Network...")
        
        # Set Brian2 clock
        defaultclock.dt = self.dt
        
        # ===================================================================
        # INPUT LAYER
        # ===================================================================
        n_input = self.c_in * self.h_in * self.w_in
        self.input_layer = NeuronGroup(
            n_input,
            model='v : volt',  # Simple input neurons (no dynamics)
            threshold='v > 0*volt',  # Fire immediately when stimulated
            reset='v = 0*volt',
            name='input_layer'
        )
        print(f"  ✓ Input Layer: {n_input:,} neurons")
        
        # ===================================================================
        # ENCODER LAYER 1: Conv 5×5, stride=2, 2→128 channels
        # ===================================================================
        n_enc1 = self.c_enc1 * self.h_enc1 * self.w_enc1
        self.encoder1 = NeuronGroup(
            n_enc1,
            model=self.neuron_eqs,
            threshold=self.threshold,
            reset=self.reset,
            refractory=self.refractory,  
            method='euler',
            name='encoder1'
        )
        # Set neuron parameters
        self.encoder1.v = self.v_rest     # it should not be self.v it should be v_rest consider v_rest=-0.5 but v = 0 then neuron should go to 0 to -0.5(Decrease) for stable position so we directly begin wit -0.5
        self.encoder1.v_rest = self.v_rest
        self.encoder1.tau = self.tau
        print(f"  ✓ Encoder 1: {n_enc1:,} neurons (LIF)")
        
        # ===================================================================
        # ENCODER LAYER 2: Conv 3×3, stride=2, 128→64 channels
        # ===================================================================
        n_enc2 = self.c_enc2 * self.h_enc2 * self.w_enc2
        self.encoder2 = NeuronGroup(
            n_enc2,
            model=self.neuron_eqs,
            threshold=self.threshold,
            reset=self.reset,
            refractory=self.refractory,
            method='euler',
            name='encoder2'
        )
        self.encoder2.v = self.v_rest
        self.encoder2.v_rest = self.v_rest
        self.encoder2.tau = self.tau
        print(f"  ✓ Encoder 2: {n_enc2:,} neurons (LIF)")
        
        # ===================================================================
        # BOTTLENECK: Conv 1×1, 64→32 channels
        # ===================================================================
        n_bottleneck = self.c_bottleneck * self.h_bottleneck * self.w_bottleneck
        self.bottleneck = NeuronGroup(
            n_bottleneck,
            model=self.neuron_eqs,
            threshold=self.threshold,
            reset=self.reset,
            refractory=self.refractory,
            method='euler',
            name='bottleneck'
        )
        self.bottleneck.v = self.v_rest
        self.bottleneck.v_rest = self.v_rest
        self.bottleneck.tau = self.tau
        print(f"  ✓ Bottleneck: {n_bottleneck:,} neurons (LIF)")
        
        # ===================================================================
        # DECODER LAYER 1: TransConv 3×3, stride=2, 32→64 channels
        # ===================================================================
        n_dec1 = self.c_enc2 * self.h_dec1 * self.w_dec1
        self.decoder1 = NeuronGroup(
            n_dec1,
            model=self.neuron_eqs,
            threshold=self.threshold,
            reset=self.reset,
            refractory=self.refractory,
            method='euler',
            name='decoder1'
        )
        self.decoder1.v = self.v_rest
        self.decoder1.v_rest = self.v_rest
        self.decoder1.tau = self.tau
        print(f"   Decoder 1: {n_dec1:,} neurons (LIF)")
        
        # ===================================================================
        # DECODER LAYER 2 (OUTPUT): TransConv 5×5, stride=2, 64→2 channels
        # ===================================================================
        n_output = self.c_in * self.h_out * self.w_out
        self.output_layer = NeuronGroup(
            n_output,
            model=self.neuron_eqs,
            threshold=self.threshold,
            reset=self.reset,
            refractory=self.refractory,
            method='euler',
            name='output_layer'
        )
        self.output_layer.v = self.v_rest
        self.output_layer.v_rest = self.v_rest
        self.output_layer.tau = self.tau
        print(f"   Output Layer: {n_output:,} neurons (LIF)")
        
        # ===================================================================
        # SYNAPSES (Connections between layers)
        # ===================================================================
        print("\nCreating Synaptic Connections...")
        
        # These will be created with convolutional connectivity patterns
        # (Implementation in _create_conv_synapses method)
        
        # Input → Encoder1 (Conv 5×5, stride=2)
        self.syn_input_enc1 = Synapses(
            self.input_layer,
            self.encoder1,
            model='w : 1',  # Synaptic weight
            on_pre='v_post += w*volt',  # Add weighted input to postsynaptic voltage
            name='syn_input_enc1'
        )
        print(f"  → Input → Encoder1 (Conv 5x5, s=2)")
        
        # Encoder1 → Encoder2 (Conv 3×3, stride=2)
        self.syn_enc1_enc2 = Synapses(
            self.encoder1,
            self.encoder2,
            model='w : 1',
            on_pre='v_post += w*volt',
            name='syn_enc1_enc2'
        )
        print(f"  → Encoder1 → Encoder2 (Conv 3x3, s=2)")
        
        # Encoder2 → Bottleneck (Conv 1×1)
        self.syn_enc2_bottleneck = Synapses(
            self.encoder2,
            self.bottleneck,
            model='w : 1',
            on_pre='v_post += w*volt',
            name='syn_enc2_bottleneck'
        )
        print(f"  → Encoder2 → Bottleneck (Conv 1x1)")
        
        # Bottleneck → Decoder1 (TransConv 3×3, stride=2)
        self.syn_bottleneck_dec1 = Synapses(
            self.bottleneck,
            self.decoder1,
            model='w : 1',
            on_pre='v_post += w*volt',
            name='syn_bottleneck_dec1'
        )
        print(f"  → Bottleneck → Decoder1 (TransConv 3x3, s=2)")
        
        # Decoder1 → Output (TransConv 5×5, stride=2)
        self.syn_dec1_output = Synapses(
            self.decoder1,
            self.output_layer,
            model='w : 1',
            on_pre='v_post += w*volt',
            name='syn_dec1_output'
        )
        print(f"  → Decoder1 → Output (TransConv 5x5, s=2)")
        
        # ===================================================================
        # SPIKE MONITORS
        # ===================================================================
        print("\nCreating Spike Monitors...")
        self.spike_mon_input = SpikeMonitor(self.input_layer, name='spikes_input')
        self.spike_mon_enc1 = SpikeMonitor(self.encoder1, name='spikes_enc1')
        self.spike_mon_enc2 = SpikeMonitor(self.encoder2, name='spikes_enc2')
        self.spike_mon_bottleneck = SpikeMonitor(self.bottleneck, name='spikes_bottleneck')
        self.spike_mon_dec1 = SpikeMonitor(self.decoder1, name='spikes_dec1')
        self.spike_mon_output = SpikeMonitor(self.output_layer, name='spikes_output')
        print("   Spike monitors created for all layers")
        
        # ===================================================================
        # STATE MONITORS (for debugging/visualization)
        # ===================================================================
        self.state_mon_bottleneck = StateMonitor(
            self.bottleneck,
            variables=['v'],
            record=True,
            name='state_bottleneck'
        )
        print("   State monitor created for bottleneck")
        
        print("\n✓ Network built successfully!")
    
    def _create_conv_synapses(self,
                             syn: Synapses,
                             in_shape: Tuple[int, int, int],
                             out_shape: Tuple[int, int, int],
                             kernel_size: int,
                             stride: int,
                             is_transpose: bool = False):
        """
        Create convolutional connectivity pattern.
        
        Args:
            syn: Synapse object to connect
            in_shape: (channels, height, width) of input layer
            out_shape: (channels, height, width) of output layer
            kernel_size: Convolution kernel size
            stride: Convolution stride
            is_transpose: True for transposed convolution
        """
        print(f"    Creating {kernel_size}x{kernel_size} conv connections...")
        
        c_in, h_in, w_in = in_shape
        c_out, h_out, w_out = out_shape
        
        # For simplified implementation: create sparse random connectivity
        # that approximates convolutional structure
        
        if not is_transpose:
            # Regular convolution
            i_list = []
            j_list = []
            
            for c_o in range(c_out):
                for h_o in range(h_out):
                    for w_o in range(w_out):
                        # Output neuron index
                        out_idx = c_o * h_out * w_out + h_o * w_out + w_o
                        
                        # Calculate receptive field in input
                        h_start = h_o * stride
                        w_start = w_o * stride
                        
                        # Connect to all input channels within kernel
                        for c_i in range(c_in):
                            for kh in range(kernel_size):
                                for kw in range(kernel_size):
                                    h_i = h_start + kh - kernel_size // 2
                                    w_i = w_start + kw - kernel_size // 2
                                    
                                    # Check bounds
                                    if 0 <= h_i < h_in and 0 <= w_i < w_in:
                                        in_idx = c_i * h_in * w_in + h_i * w_in + w_i
                                        i_list.append(in_idx)
                                        j_list.append(out_idx)
            
            # Connect synapses
            if len(i_list) > 0:
                syn.connect(i=i_list, j=j_list)
                # Initialize weights with values large enough to cause spiking
                syn.w = 'rand() * 0.3 + 0.2'  # [0.2, 0.5]
                print(f"      Connected {len(i_list):,} synapses")
        
        else:
            # Transposed convolution (upsampling)
            # Simplified: reverse of regular conv
            i_list = []
            j_list = []
            
            for c_o in range(c_out):
                for h_o in range(h_out):
                    for w_o in range(w_out):
                        out_idx = c_o * h_out * w_out + h_o * w_out + w_o
                        
                        # For transpose conv, input is smaller
                        h_i = h_o // stride
                        w_i = w_o // stride
                        
                        for c_i in range(c_in):
                            if h_i < h_in and w_i < w_in:
                                in_idx = c_i * h_in * w_in + h_i * w_in + w_i
                                i_list.append(in_idx)
                                j_list.append(out_idx)
            
            if len(i_list) > 0:
                syn.connect(i=i_list, j=j_list)
                syn.w = 'rand() * 0.3 + 0.2'  # [0.2, 0.5]
                print(f"      Connected {len(i_list):,} synapses (transpose)")
    
    def initialize_weights(self):
        """Initialize synaptic weights for all connections."""
        print("\nInitializing Synaptic Weights...")
        
        # Input → Encoder1
        self._create_conv_synapses(
            self.syn_input_enc1,
            (self.c_in, self.h_in, self.w_in),
            (self.c_enc1, self.h_enc1, self.w_enc1),
            kernel_size=5,
            stride=2
        )
        
        # Encoder1 → Encoder2
        self._create_conv_synapses(
            self.syn_enc1_enc2,
            (self.c_enc1, self.h_enc1, self.w_enc1),
            (self.c_enc2, self.h_enc2, self.w_enc2),
            kernel_size=3,
            stride=2
        )
        
        # Encoder2 → Bottleneck
        self._create_conv_synapses(
            self.syn_enc2_bottleneck,
            (self.c_enc2, self.h_enc2, self.w_enc2),
            (self.c_bottleneck, self.h_bottleneck, self.w_bottleneck),
            kernel_size=1,
            stride=1
        )
        
        # Bottleneck → Decoder1
        self._create_conv_synapses(
            self.syn_bottleneck_dec1,
            (self.c_bottleneck, self.h_bottleneck, self.w_bottleneck),
            (self.c_enc2, self.h_dec1, self.w_dec1),
            kernel_size=3,
            stride=2,
            is_transpose=True
        )
        
        # Decoder1 → Output
        self._create_conv_synapses(
            self.syn_dec1_output,
            (self.c_enc2, self.h_dec1, self.w_dec1),
            (self.c_in, self.h_out, self.w_out),
            kernel_size=5,
            stride=2,
            is_transpose=True
        )
        
        print("✓ Weights initialized")
    
    def _count_total_neurons(self) -> int:
        """Count total number of neurons in network."""
        total = (
            self.c_in * self.h_in * self.w_in +
            self.c_enc1 * self.h_enc1 * self.w_enc1 +
            self.c_enc2 * self.h_enc2 * self.w_enc2 +
            self.c_bottleneck * self.h_bottleneck * self.w_bottleneck +
            self.c_enc2 * self.h_dec1 * self.w_dec1 +
            self.c_in * self.h_out * self.w_out
        )
        return total
    
    def set_input_spikes(self, spike_times_dict: Dict[int, np.ndarray]):
        """
        Set spike times for input neurons.
        
        Args:
            spike_times_dict: Dictionary mapping neuron index to spike times
                             e.g., {0: [1*ms, 5*ms, 10*ms], 1: [2*ms, 7*ms], ...}
        """
        # Create SpikeGeneratorGroup for input  
        indices = []
        times = []
        
        for neuron_idx, spike_times in spike_times_dict.items():
            for t in spike_times:
                indices.append(neuron_idx)
                times.append(t)
        
        if len(indices) > 0:
            self.input_generator = SpikeGeneratorGroup(          #SpikeGeneratorGroup:- generates spikes at specific times for specific neurons
                    len(self.input_layer),                       # number of neurons in the group
                    indices,                                     # A list of neuron IDs (integers) that should fire
                times * ms if not isinstance(times[0], Quantity) else times,
                name='input_generator'
            )
            
            # Connect to input layer
            self.syn_generator_input = Synapses(
                self.input_generator,
                self.input_layer,
                on_pre='v_post = 10*volt',        # Strong input to ensure firing
                name='syn_generator_input'
            )
            self.syn_generator_input.connect(j='i')
    
    def forward(self, 
                input_spikes: np.ndarray,
                simulation_time: float = None) -> Dict[str, np.ndarray]:
        """
        Process input through network (forward pass).
        
        Args:
            input_spikes: Input spike trains
                         Shape: (C, H, W, T) where spikes are binary (0 or 1)
                         OR dictionary of {neuron_idx: [spike_times]}
            simulation_time: Simulation duration in ms (default: self.simulation_time)
        
        Returns:
            Dictionary containing:
                - 'output_spikes': Output spike trains (C, H, W, T)
                - 'spike_counts': Spike counts per layer
                - 'bottleneck_activity': Bottleneck layer spikes for analysis
        """
        if simulation_time is None:
            simulation_time = float(self.simulation_time / ms)
        
        # Reset network state
        self._reset_network()   # Clears all voltages, spike monitors, and synapses so the simulation starts fresh
        
        # Convert input spikes to spike times
        if isinstance(input_spikes, np.ndarray):                     # isinstance()-->checks if input_spikes is a NumPy array.
            spike_times_dict = self._spikes_to_times(input_spikes)   # if yes --> convert to dictionary of spike times
        else:
            spike_times_dict = input_spikes
        
        # Set input spikes
        self.set_input_spikes(spike_times_dict)
        
        # Create network and run
        net = Network()
        net.add(self.input_layer,
                self.encoder1, self.encoder2,
                self.bottleneck,
                self.decoder1, self.output_layer,
                self.syn_input_enc1, self.syn_enc1_enc2,
                self.syn_enc2_bottleneck,
                self.syn_bottleneck_dec1, self.syn_dec1_output,
                self.spike_mon_input, self.spike_mon_enc1,
                self.spike_mon_enc2, self.spike_mon_bottleneck,
                self.spike_mon_dec1, self.spike_mon_output)
        
        # Add the input generator and its synapses if they exist
        if hasattr(self, 'input_generator') and self.input_generator is not None:
            net.add(self.input_generator, self.syn_generator_input)

        net.run(simulation_time * ms)
        
        # Collect output spikes
        output_spike_times = self.spike_mon_output.t / ms
        output_spike_indices = self.spike_mon_output.i
        
        # Convert back to array format
        output_array = self._times_to_spikes(
            output_spike_indices,
            output_spike_times,
            n_neurons=len(self.output_layer),
            duration=simulation_time
        )
        
        # Reshape to (C, H, W, T)
        output_spikes = self._reshape_to_spatial(
            output_array,
            (self.c_in, self.h_out, self.w_out)
        )
        
        # Collect statistics
        spike_counts = {
            'input': len(self.spike_mon_input),
            'encoder1': len(self.spike_mon_enc1),
            'encoder2': len(self.spike_mon_enc2),
            'bottleneck': len(self.spike_mon_bottleneck),
            'decoder1': len(self.spike_mon_dec1),
            'output': len(self.spike_mon_output)
        }
        
        # Get bottleneck activity for analysis
        bottleneck_spikes = self._times_to_spikes(
            self.spike_mon_bottleneck.i,
            self.spike_mon_bottleneck.t / ms,
            n_neurons=len(self.bottleneck),
            duration=simulation_time
        )
        
        return {
            'output_spikes': output_spikes,
            'spike_counts': spike_counts,
            'bottleneck_activity': bottleneck_spikes
        }
    
    def _reset_network(self):
        """Reset all neuron states to initial conditions."""
        # Reset membrane potentials
        self.encoder1.v = self.v_rest
        self.encoder2.v = self.v_rest
        self.bottleneck.v = self.v_rest
        self.decoder1.v = self.v_rest
        self.output_layer.v = self.v_rest
        
        # Clear spike monitors
        # Note: Brian2 automatically resets monitors when using Network.run()
    
    def _spikes_to_times(self, spike_array: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Convert binary spike array to spike times dictionary.
        
        Args:
            spike_array: Binary array (N_neurons, T_steps) or (C, H, W, T)
        
        Returns:
            Dictionary {neuron_idx: [spike_times in ms]}
        """
        # Flatten spatial dimensions if needed
        if spike_array.ndim == 4:  # (C, H, W, T)
            C, H, W, T = spike_array.shape
            spike_array = spike_array.reshape(C * H * W, T)
        
        spike_times_dict = {}
        dt = float(self.dt / ms)
        
        for neuron_idx in range(spike_array.shape[0]):
            spike_indices = np.where(spike_array[neuron_idx] > 0)[0]
            if len(spike_indices) > 0:
                spike_times_dict[neuron_idx] = spike_indices * dt
        
        return spike_times_dict
    
    def _times_to_spikes(self,
                        spike_indices: np.ndarray,
                        spike_times: np.ndarray,
                        n_neurons: int,
                        duration: float) -> np.ndarray:
        """
        Convert spike times to binary spike array.
        
        Args:
            spike_indices: Neuron indices that spiked
            spike_times: Spike times in ms
            n_neurons: Total number of neurons
            duration: Total duration in ms
        
        Returns:
            Binary spike array (n_neurons, n_timesteps)
        """
        dt = float(self.dt / ms)
        n_timesteps = int(duration / dt)
        
        spike_array = np.zeros((n_neurons, n_timesteps), dtype=np.float32)
        
        for idx, t in zip(spike_indices, spike_times):
            time_idx = int(t / dt)
            if 0 <= time_idx < n_timesteps and 0 <= idx < n_neurons:
                spike_array[idx, time_idx] = 1.0
        
        return spike_array
    
    def _reshape_to_spatial(self,
                           spike_array: np.ndarray,
                           shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Reshape flat spike array to spatial format.
        
        Args:
            spike_array: Array of shape (C*H*W, T)
            shape: Target shape (C, H, W)
        
        Returns:
            Reshaped array (C, H, W, T)
        """
        C, H, W = shape
        T = spike_array.shape[1]
        
        return spike_array.reshape(C, H, W, T)
    
    def compute_loss(self,
                    input_spikes: np.ndarray,
                    output_spikes: np.ndarray,
                    beta: float = 0.01) -> Dict[str, float]:
        """
        Compute reconstruction loss + sparsity regularization.
        
        Loss = MSE(spike_count_out, spike_count_in) + β * mean(spike_rate)
        
        Args:
            input_spikes: Input spike trains
            output_spikes: Reconstructed spike trains
            beta: Sparsity regularization weight
            
        Returns:
            Dictionary with loss components
        """
        # Spike counts (sum over time)
        input_count = np.sum(input_spikes, axis=-1)
        output_count = np.sum(output_spikes, axis=-1)
        
        # MSE reconstruction loss
        mse_loss = np.mean((input_count - output_count) ** 2)
        
        # Sparsity regularization
        spike_rate = np.mean(output_spikes)
        sparsity_loss = beta * spike_rate
        
        # Total loss
        total_loss = mse_loss + sparsity_loss
        
        return {
            'total': total_loss,
            'mse': mse_loss,
            'sparsity': sparsity_loss,
            'spike_rate': spike_rate
        }
    
    # def train_step(self, input_data: np.ndarray) -> Dict[str, float]:
        """
        Single training step (to be implemented).
        
        Args:
            input_data: Batch of input event frames
            
        Returns:
            Loss dictionary
        """
        # TODO: Implement training
        # 1. Forward pass
        # 2. Compute loss
        # 3. Backpropagate (or use STDP)
        # 4. Update weights
        
        raise NotImplementedError("Training to be implemented in Part 2")
    
    def get_network_info(self) -> Dict:
        """Get network architecture information."""
        return {
            'input_shape': self.input_shape,
            'encoder_channels': (self.c_enc1, self.c_enc2),
            'bottleneck_channels': self.c_bottleneck,
            'total_neurons': self._count_total_neurons(),
            'neuron_type': 'LIF',
            'tau': float(self.tau / ms),
            'v_threshold': float(self.v_threshold / volt),
            'simulation_time': float(self.simulation_time / ms)
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_default_autoencoder() -> SNNAutoencoder:
    """Create autoencoder with default UCSD Ped2 configuration."""
    return SNNAutoencoder(
        input_shape=(2, 240, 360),
        hidden_channels=(128, 64, 32),
        tau=20.0,
        v_threshold=1.0,
        dt=1.0,
        simulation_time=100.0
    )


if __name__ == "__main__":
    # Test network creation
    print("\nTesting SNN Autoencoder Creation...")
    print("=" * 70)
    
    # Create network
    autoencoder = create_default_autoencoder()
    
    # Initialize weights
    autoencoder.initialize_weights()
    
    # Print network info
    print("\n" + "=" * 70)
    print("NETWORK INFORMATION")
    print("=" * 70)
    info = autoencoder.get_network_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✓ SNN AUTOENCODER STRUCTURE COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Implement forward pass")
    print("  2. Implement training procedure")
    print("  3. Test on real event data")
    print("=" * 70)