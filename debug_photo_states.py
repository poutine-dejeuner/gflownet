#!/usr/bin/env python3
"""
Debug script to understand Photo state structure
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_photo_states():
    """Debug Photo environment states to understand the structure"""
    try:
        from gflownet.envs.photo import Photo
        
        # Initialize Photo environment
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env_kwargs = {
            'device': device,
            'fixed_distr_params': {
                "beta_weights": 1.0,
                "beta_alpha": 10.0,
                "beta_beta": 10.0,
                "bernoulli_bts_prob": 0.1,
                "bernoulli_eos_prob": 0.1,
            },
            'random_distr_params': {
                "beta_weights": 1.0,
                "beta_alpha": 10.0,
                "beta_beta": 10.0,
                "bernoulli_bts_prob": 0.1,
                "bernoulli_eos_prob": 0.1,
            }
        }
        
        env = Photo(**env_kwargs)
        
        print("=== Photo Environment State Debug ===")
        
        # Test initial state
        env.reset()
        print(f"\\nInitial state type: {type(env.state)}")
        print(f"Initial state: {env.state}")
        
        # Test state2proxy
        proxy_state = env.state2proxy(env.state)
        print(f"\\nProxy state type: {type(proxy_state)}")
        print(f"Proxy state shape: {proxy_state.shape if hasattr(proxy_state, 'shape') else 'no shape'}")
        print(f"Proxy state: {proxy_state}")
        
        # Test with a simple action
        try:
            action = (-1, 5, 0, 0, 0)  # Simple action
            print(f"\\nTaking action: {action}")
            env.step(action)
            
            print(f"State after action type: {type(env.state)}")
            print(f"State after action: {env.state}")
            
            proxy_state_after = env.state2proxy(env.state)
            print(f"\\nProxy state after action type: {type(proxy_state_after)}")
            print(f"Proxy state after action shape: {proxy_state_after.shape if hasattr(proxy_state_after, 'shape') else 'no shape'}")
            print(f"Proxy state after action: {proxy_state_after}")
            
        except Exception as e:
            print(f"Error taking action: {e}")
        
        # Test rbf_function directly
        try:
            from gflownet.proxy.photo import rbf_function, make_grid
            
            print(f"\\n=== Testing rbf_function directly ===")
            
            # Create a simple RBF parameter tensor
            # Shape should be (batch_size, n_functions, 4) where 4 = [weight, center_x, center_y, width]
            simple_params = torch.tensor([
                [[1.0, 0.5, 0.5, 0.1],   # Function 1: weight=1, center=(0.5,0.5), width=0.1
                 [0.8, 0.3, 0.7, 0.2]]   # Function 2: weight=0.8, center=(0.3,0.7), width=0.2
            ], dtype=torch.float32)  # Shape: (1, 2, 4)
            
            print(f"Simple params shape: {simple_params.shape}")
            print(f"Simple params: {simple_params}")
            
            # Create grid
            rows, cols = 101, 91
            grid_points = make_grid(rows, cols)
            print(f"Grid points shapes: {[g.shape for g in grid_points]}")
            
            # Test rbf_function
            image = rbf_function(simple_params, grid_points)
            print(f"Generated image shape: {image.shape}")
            print(f"Image value range: [{image.min():.3f}, {image.max():.3f}]")
            
        except Exception as e:
            print(f"Error testing rbf_function: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error in debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_photo_states()