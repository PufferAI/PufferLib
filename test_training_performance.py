#!/usr/bin/env python3
"""
Quick test script to measure training performance on different environments
and validate the performance claims.
"""

import time
import argparse
from pufferlib.pufferl import train, load_config

def test_environment_performance(env_name, timesteps=100000, device='mps'):
    """Test training performance on a specific environment"""
    print(f"Testing {env_name} training performance...")
    print(f"Target timesteps: {timesteps:,}")
    print(f"Device: {device}")
    print("=" * 50)
    
    # Load config and override key settings
    args = load_config(env_name)
    args['train']['device'] = device
    args['train']['total_timesteps'] = timesteps
    args['train']['batch_size'] = 4096  # Reasonable batch size
    args['train']['bptt_horizon'] = 128
    args['wandb'] = False
    args['neptune'] = False
    
    start_time = time.time()
    
    try:
        # Run training
        logs = train(env_name, args=args)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate performance metrics
        if logs:
            final_log = logs[-1]
            actual_steps = final_log.get('agent_steps', timesteps)
            final_sps = final_log.get('SPS', 0)
            
            print(f"\n🎯 Performance Results for {env_name}:")
            print(f"  Actual steps: {actual_steps:,}")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Average SPS: {actual_steps/total_time:.0f}")
            print(f"  Final SPS: {final_sps:.0f}")
            print(f"  Efficiency: {(actual_steps/total_time)/final_sps*100:.1f}%" if final_sps > 0 else "")
            
            return {
                'env_name': env_name,
                'actual_steps': actual_steps,
                'total_time': total_time,
                'average_sps': actual_steps/total_time,
                'final_sps': final_sps,
                'logs': logs
            }
        else:
            print(f"❌ No logs returned for {env_name}")
            return None
            
    except Exception as e:
        print(f"❌ Error training {env_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Test training performance on different environments')
    parser.add_argument('--timesteps', type=int, default=100000, help='Number of timesteps to train')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'], help='Device to use')
    parser.add_argument('--environments', nargs='+', default=['puffer_cartpole', 'puffer_pong'], 
                       help='Environments to test')
    
    args = parser.parse_args()
    
    print("🚀 PufferLib Training Performance Test")
    print("=" * 60)
    
    results = []
    
    for env_name in args.environments:
        print(f"\n🔥 Testing {env_name}")
        result = test_environment_performance(env_name, args.timesteps, args.device)
        if result:
            results.append(result)
        print()
    
    # Summary comparison
    if len(results) > 1:
        print("\n📊 Performance Comparison:")
        print("=" * 60)
        print(f"{'Environment':<20} {'Steps':<10} {'Time (s)':<10} {'Avg SPS':<12} {'Final SPS':<12}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['env_name']:<20} {result['actual_steps']:<10,} "
                  f"{result['total_time']:<10.1f} {result['average_sps']:<12.0f} "
                  f"{result['final_sps']:<12.0f}")
        
        # Find best performer
        best = max(results, key=lambda x: x['average_sps'])
        print(f"\n🏆 Best performer: {best['env_name']} at {best['average_sps']:.0f} SPS average")
        
        # Validate claims
        cartpole_results = [r for r in results if 'cartpole' in r['env_name'].lower()]
        if cartpole_results:
            cartpole_sps = cartpole_results[0]['average_sps']
            print(f"\n🎯 CartPole Performance Validation:")
            print(f"  Measured: {cartpole_sps:.0f} SPS")
            print(f"  Claimed: 370,000+ SPS")
            if cartpole_sps >= 370000:
                print(f"  ✅ VALIDATED - Exceeds claim by {cartpole_sps-370000:.0f} SPS")
            elif cartpole_sps >= 300000:
                print(f"  ⚠️  CLOSE - Within 20% of claim ({cartpole_sps/370000*100:.1f}%)")
            else:
                print(f"  ❌ BELOW CLAIM - Only {cartpole_sps/370000*100:.1f}% of claimed performance")

if __name__ == '__main__':
    main()