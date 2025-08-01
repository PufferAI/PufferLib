#!/usr/bin/env python3
"""
Quick performance test focusing on the core PufferRL training loop
to validate performance claims without full training runs.
"""

import time
import torch
import numpy as np
from pufferlib.pufferl import load_config, load_env, load_policy, PuffeRL

def measure_sps_capability(env_name, duration_seconds=10, device='mps'):
    """Measure steps per second capability of the training setup"""
    print(f"🔥 Measuring {env_name} SPS capability")
    print(f"Duration: {duration_seconds}s, Device: {device}")
    print("=" * 50)
    
    # Load minimal config
    args = load_config(env_name)
    args['train']['device'] = device
    args['train']['total_timesteps'] = 1000000  # High limit
    args['train']['batch_size'] = 4096 if 'cartpole' in env_name.lower() else 2048
    args['train']['bptt_horizon'] = 128 if 'cartpole' in env_name.lower() else 64
    args['train']['update_epochs'] = 1  # Minimal training
    args['wandb'] = False
    args['neptune'] = False
    
    try:
        print("Loading environment and policy...")
        vecenv = load_env(env_name, args)
        policy = load_policy(args, vecenv, env_name)
        
        print("Initializing PufferRL...")
        train_config = dict(**args['train'], env=env_name)
        pufferl = PuffeRL(train_config, vecenv, policy, logger=None)
        
        print(f"Setup complete. Batch size: {train_config['batch_size']}")
        print(f"Environment: {vecenv.num_agents} agents")
        
        # Warm up
        print("Warming up...")
        pufferl.evaluate()
        pufferl.train()
        
        # Measure performance
        print(f"Measuring performance for {duration_seconds}s...")
        start_time = time.time()
        start_steps = pufferl.global_step
        
        eval_count = 0
        train_count = 0
        
        while time.time() - start_time < duration_seconds:
            pufferl.evaluate()
            eval_count += 1
            
            logs = pufferl.train()
            train_count += 1
            
            if logs and train_count % 5 == 0:  # Print every 5th update
                current_sps = logs.get('SPS', 0)
                elapsed = time.time() - start_time
                print(f"  Update {train_count}: {current_sps:.0f} SPS ({elapsed:.1f}s elapsed)")
        
        end_time = time.time()
        end_steps = pufferl.global_step
        
        # Calculate results
        total_time = end_time - start_time
        total_steps = end_steps - start_steps
        average_sps = total_steps / total_time
        
        print(f"\n🎯 Performance Results:")
        print(f"  Environment: {env_name}")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average SPS: {average_sps:.0f}")
        print(f"  Updates: {train_count} train, {eval_count} eval")
        print(f"  Batch size: {train_config['batch_size']}")
        
        # Cleanup
        pufferl.close()
        
        return {
            'env_name': env_name,
            'total_steps': total_steps,
            'total_time': total_time,
            'average_sps': average_sps,
            'updates': train_count,
            'batch_size': train_config['batch_size']
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("🚀 PufferLib Quick Performance Test")
    print("=" * 60)
    
    # Test environments
    environments = ['puffer_cartpole', 'puffer_pong']
    results = []
    
    for env_name in environments:
        print(f"\n{'='*20} {env_name.upper()} {'='*20}")
        result = measure_sps_capability(env_name, duration_seconds=15, device='mps')
        if result:
            results.append(result)
        print()
    
    # Summary
    if results:
        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"{'Environment':<20} {'Steps':<10} {'Time (s)':<10} {'SPS':<12} {'Batch':<8}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['env_name']:<20} {result['total_steps']:<10,} "
                  f"{result['total_time']:<10.1f} {result['average_sps']:<12.0f} "
                  f"{result['batch_size']:<8}")
        
        # Validate CartPole claim
        cartpole_results = [r for r in results if 'cartpole' in r['env_name'].lower()]
        if cartpole_results:
            cartpole_sps = cartpole_results[0]['average_sps']
            print(f"\n🎯 CartPole Performance Analysis:")
            print(f"  Measured SPS: {cartpole_sps:.0f}")
            print(f"  Claimed SPS: 370,000+")
            
            if cartpole_sps >= 370000:
                print(f"  ✅ CLAIM VALIDATED - Exceeds by {cartpole_sps-370000:.0f} SPS")
            elif cartpole_sps >= 300000:
                print(f"  ⚠️  CLOSE TO CLAIM - {cartpole_sps/370000*100:.1f}% of claimed")
            elif cartpole_sps >= 200000:
                print(f"  📈 REASONABLE - {cartpole_sps/370000*100:.1f}% of claimed (may vary by config)")
            else:
                print(f"  ❌ BELOW EXPECTATION - Only {cartpole_sps/370000*100:.1f}% of claimed")
                
            print(f"\n💡 Note: Performance can vary based on:")
            print(f"  - Batch size and horizon configuration")
            print(f"  - System load and thermal throttling")
            print(f"  - Memory bandwidth and Numba JIT warmup")
            print(f"  - Environment complexity and vectorization")

if __name__ == '__main__':
    main()