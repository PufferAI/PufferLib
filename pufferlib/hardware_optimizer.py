#!/usr/bin/env python3
"""
Hardware-specific optimization for PufferLib
Automatically detects hardware and applies optimal training configurations
"""

import os
import platform
import torch
import psutil
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class HardwareProfile:
    """Hardware characteristics for optimization"""
    device_type: str  # 'cuda', 'mps', 'cpu'
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    cpu_cores: int = 1
    cpu_model: str = ""
    system_memory_gb: float = 0
    is_apple_silicon: bool = False
    compute_capability: Optional[tuple] = None  # For CUDA


class HardwareOptimizer:
    """Automatically optimize PufferLib configuration based on hardware"""
    
    def __init__(self):
        self.profile = self._detect_hardware()
        self._base_config = self._calculate_dynamic_config()
    
    def _detect_hardware(self) -> HardwareProfile:
        """Detect current hardware configuration"""
        profile = HardwareProfile(device_type='cpu')
        
        # CPU info
        profile.cpu_cores = psutil.cpu_count(logical=False) or 1
        profile.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Platform info
        system = platform.system()
        machine = platform.machine()
        profile.is_apple_silicon = system == 'Darwin' and machine == 'arm64'
        
        # Get CPU model
        if system == 'Darwin':
            try:
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                profile.cpu_model = result.stdout.strip()
            except:
                profile.cpu_model = platform.processor()
        else:
            profile.cpu_model = platform.processor()
        
        # GPU detection
        if torch.cuda.is_available():
            profile.device_type = 'cuda'
            profile.gpu_name = torch.cuda.get_device_name(0)
            profile.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            profile.compute_capability = torch.cuda.get_device_capability(0)
        elif torch.backends.mps.is_available():
            profile.device_type = 'mps'
            profile.gpu_name = "Apple Silicon GPU"
            # Estimate GPU memory based on system memory (unified architecture)
            # M4: ~75% of system memory available to GPU
            profile.gpu_memory_gb = profile.system_memory_gb * 0.75
        
        return profile
    
    def _calculate_dynamic_config(self) -> Dict[str, Any]:
        """Calculate optimal configuration based on actual hardware specs"""
        config = {
            'device': self.profile.device_type,
            'gradient_accumulation_steps': 1,
            'cpu_offload': False,
            'compile': False,
            'non_blocking': False,
            'min_params_for_gpu': float('inf'),
        }
        
        # CPU core-based scaling
        cores = self.profile.cpu_cores
        
        if self.profile.device_type == 'mps':
            # Apple Silicon specific
            config['device'] = 'mps'
            config['non_blocking'] = True
            config['min_params_for_gpu'] = 100_000  # MPS has higher overhead
            
            # Scale based on GPU memory (unified memory architecture)
            gpu_mem_gb = self.profile.gpu_memory_gb
            
            # num_envs: scale with cores but cap for stability
            config['num_envs'] = min(max(4, cores - 2), 16)  # Leave 2 cores for system
            
            # Batch size: scale with memory
            # Rule of thumb: ~1K batch per GB of GPU memory
            config['batch_size'] = int(min(gpu_mem_gb * 1024, 16384))
            config['minibatch_size'] = config['batch_size'] // 2
            
            # Adjust for smaller memory systems
            if gpu_mem_gb < 16:
                config['batch_size'] = 4096
                config['minibatch_size'] = 2048
            elif gpu_mem_gb < 32:
                config['batch_size'] = 8192
                config['minibatch_size'] = 4096
            
        elif self.profile.device_type == 'cuda':
            # NVIDIA GPU specific
            config['device'] = 'cuda'
            config['compile'] = True
            config['non_blocking'] = True
            config['min_params_for_gpu'] = 10_000  # CUDA has lower overhead
            
            gpu_mem_gb = self.profile.gpu_memory_gb
            
            # num_envs: more aggressive scaling for CUDA
            config['num_envs'] = min(cores * 4, 128)  # CUDA can handle more envs
            
            # Batch size: CUDA can handle larger batches
            if gpu_mem_gb >= 40:  # A100 class
                config['batch_size'] = 65536
                config['minibatch_size'] = 16384
                config['num_envs'] = min(cores * 8, 128)
            elif gpu_mem_gb >= 20:  # RTX 4090 class
                config['batch_size'] = 32768
                config['minibatch_size'] = 8192
                config['num_envs'] = min(cores * 4, 64)
            elif gpu_mem_gb >= 10:  # RTX 3060+ class
                config['batch_size'] = 16384
                config['minibatch_size'] = 4096
                config['num_envs'] = min(cores * 2, 32)
            else:  # Smaller GPUs
                config['batch_size'] = 8192
                config['minibatch_size'] = 2048
                config['num_envs'] = min(cores * 2, 16)
                
        else:
            # CPU only
            config['device'] = 'cpu'
            config['min_params_for_gpu'] = float('inf')
            
            # Conservative scaling for CPU
            config['num_envs'] = min(cores, 16)
            config['batch_size'] = config['num_envs'] * 1024
            config['minibatch_size'] = max(256, config['batch_size'] // 8)
        
        return config
    
    def get_optimal_config(self, env_name: str = None, network_params: int = None) -> Dict[str, Any]:
        """Get optimal configuration for current hardware and environment"""
        
        # Start with dynamically calculated base config
        config = self._base_config.copy()
        
        # Environment-specific adjustments
        if env_name:
            config = self._adjust_for_environment(config, env_name)
        
        # Network size adjustments
        if network_params:
            config = self._adjust_for_network_size(config, network_params)
        
        return config
    
    
    def _adjust_for_environment(self, config: Dict[str, Any], env_name: str) -> Dict[str, Any]:
        """Adjust configuration based on environment characteristics"""
        
        # Ocean environments (small observation spaces)
        if 'ocean' in env_name.lower() or any(e in env_name.lower() for e in 
                                              ['cartpole', 'acrobot', 'pendulum', 'mountaincar']):
            # These have very small networks, reduce batch sizes
            config['batch_size'] = min(config['batch_size'], 8192)
            config['minibatch_size'] = min(config['minibatch_size'], 2048)
        
        # Atari environments
        elif 'atari' in env_name.lower() or 'pong' in env_name.lower():
            # Medium-sized networks, standard config is good
            pass
        
        # Large environments (NethHack, NMMO)
        elif any(e in env_name.lower() for e in ['nethack', 'nmmo', 'pokered']):
            # These benefit from larger batch sizes
            config['batch_size'] = int(config['batch_size'] * 1.5)
            config['minibatch_size'] = int(config['minibatch_size'] * 1.5)
        
        return config
    
    def _adjust_for_network_size(self, config: Dict[str, Any], network_params: int) -> Dict[str, Any]:
        """Adjust configuration based on neural network size"""
        
        # Check if network is too small for GPU
        if network_params < config['min_params_for_gpu'] and config['device'] != 'cpu':
            print(f"Network has {network_params:,} parameters, below GPU threshold of {config['min_params_for_gpu']:,}")
            print("Switching to CPU for better performance")
            config['device'] = 'cpu'
            config['num_envs'] = min(self.profile.cpu_cores, 16)
            config['batch_size'] = config['num_envs'] * 1024
            config['minibatch_size'] = max(256, config['batch_size'] // 8)
        
        # Adjust batch sizes based on network size and memory
        elif config['device'] in ['cuda', 'mps']:
            # Estimate memory usage (rough approximation)
            # params * 4 bytes * (forward + backward + optimizer state)
            estimated_memory_mb = (network_params * 4 * 6) / (1024 * 1024)
            
            # Add buffer for activations (rough estimate: 2x params)
            estimated_memory_mb += (network_params * 4 * 2) / (1024 * 1024)
            
            # Scale batch size based on available memory
            available_memory_mb = self.profile.gpu_memory_gb * 1024
            memory_ratio = available_memory_mb / estimated_memory_mb
            
            if memory_ratio < 10:  # Less than 10x headroom
                # Need to reduce batch size
                scale = min(1.0, memory_ratio / 20)  # Conservative scaling
                config['batch_size'] = int(config['batch_size'] * scale)
                config['minibatch_size'] = int(config['minibatch_size'] * scale)
                print(f"Large network detected ({network_params:,} params)")
                print(f"Adjusting batch size to {config['batch_size']:,} to fit in memory")
        
        return config
    
    def print_hardware_info(self):
        """Print detected hardware information"""
        print("=" * 60)
        print("Hardware Detection Results")
        print("=" * 60)
        print(f"System: {platform.system()} {platform.machine()}")
        print(f"CPU: {self.profile.cpu_model}")
        print(f"CPU Cores: {self.profile.cpu_cores}")
        print(f"System Memory: {self.profile.system_memory_gb:.1f} GB")
        print(f"Device Type: {self.profile.device_type}")
        
        if self.profile.gpu_name:
            print(f"GPU: {self.profile.gpu_name}")
            print(f"GPU Memory: {self.profile.gpu_memory_gb:.1f} GB")
        
        if self.profile.is_apple_silicon:
            print("Apple Silicon Detected: Yes")
        
        print("=" * 60)
    
    def apply_optimal_config(self, args: Dict[str, Any], env_name: str = None, 
                           network_params: int = None) -> Dict[str, Any]:
        """Apply optimal configuration to existing args"""
        optimal = self.get_optimal_config(env_name, network_params)
        
        # Only override if not explicitly set by user
        if 'train' not in args:
            args['train'] = {}
        if 'vec' not in args:
            args['vec'] = {}
        
        # Track what was optimized
        optimized = []
        
        # Device optimization - handle both 'auto' and default 'cuda' intelligently
        current_device = args['train'].get('device', 'cuda')
        if current_device == 'auto' or (current_device == 'cuda' and not torch.cuda.is_available() and self.profile.device_type == 'mps'):
            args['train']['device'] = optimal['device']
            optimized.append(f"Device: {optimal['device']} (auto-detected)")
        
        # Batch size optimization
        if args['train'].get('batch_size') == 'auto':
            args['train']['batch_size'] = optimal['batch_size']
            optimized.append(f"Batch Size: {optimal['batch_size']:,}")
        
        # Minibatch size optimization - apply if auto or if using non-optimal defaults
        current_minibatch = args['train'].get('minibatch_size', 8192)
        if current_minibatch == 'auto' or (self.profile.device_type == 'mps' and current_minibatch > optimal['minibatch_size']):
            args['train']['minibatch_size'] = optimal['minibatch_size']
            optimized.append(f"Minibatch Size: {optimal['minibatch_size']:,}")
        
        # Num envs optimization
        current_num_envs = args['vec'].get('num_envs', 2)
        if current_num_envs == 'auto' or current_num_envs == 2:  # 2 is the unhelpful default
            args['vec']['num_envs'] = optimal['num_envs']
            optimized.append(f"Num Envs: {optimal['num_envs']}")
        
        # Apply non-blocking for MPS/CUDA (doesn't hurt if already set)
        if optimal['device'] in ['mps', 'cuda']:
            args['train']['non_blocking'] = True
        
        # Only apply these if not already set
        if 'cpu_offload' not in args['train']:
            args['train']['cpu_offload'] = optimal['cpu_offload']
        if 'compile' not in args['train']:
            args['train']['compile'] = optimal['compile']
        
        # Report what was optimized
        if optimized:
            print(f"\n✨ Hardware-Optimized Configuration Applied:")
            for item in optimized:
                print(f"   {item}")
            
            # Add recommendations if on MPS
            if self.profile.device_type == 'mps':
                print(f"\n📌 MPS Performance Tips:")
                print(f"   • Networks <100K params run faster on CPU")
                print(f"   • Current config optimized for networks >100K params")
                print(f"   • Monitor GPU usage in Activity Monitor > Window > GPU History")
        
        return args


# Convenience function for direct use
def optimize_for_hardware(args: Dict[str, Any], env_name: str = None, 
                         network_params: int = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Optimize PufferLib configuration for current hardware
    
    Args:
        args: Current configuration dictionary
        env_name: Name of the environment
        network_params: Number of parameters in the neural network
        verbose: Whether to print hardware info
    
    Returns:
        Updated configuration dictionary
    """
    optimizer = HardwareOptimizer()
    
    if verbose:
        optimizer.print_hardware_info()
    
    return optimizer.apply_optimal_config(args, env_name, network_params)


if __name__ == "__main__":
    # Test hardware detection
    optimizer = HardwareOptimizer()
    optimizer.print_hardware_info()
    
    # Test configurations
    print("\nOptimal configs for different scenarios:")
    print("\n1. Small network (10K params):")
    config = optimizer.get_optimal_config(network_params=10_000)
    print(f"   Device: {config['device']}")
    print(f"   Batch Size: {config['batch_size']:,}")
    
    print("\n2. Medium network (500K params):")
    config = optimizer.get_optimal_config(network_params=500_000)
    print(f"   Device: {config['device']}")
    print(f"   Batch Size: {config['batch_size']:,}")
    
    print("\n3. Large network (5M params):")
    config = optimizer.get_optimal_config(network_params=5_000_000)
    print(f"   Device: {config['device']}")
    print(f"   Batch Size: {config['batch_size']:,}")