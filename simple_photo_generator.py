#!/usr/bin/env python3
"""
Simple script to generate random action sequences for the Photo environment
"""

import sys
import pickle
import random
from pathlib import Path
import torch
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def generate_simple_photo_sequences(n_sequences=20, max_steps=15):
    """
    Generate random action sequences by actually running the Photo environment
    """
    try:
        from gflownet.envs.photo import Photo
        
        print("Initializing Photo environment...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize with required parameters
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
        print(f"Environment initialized on {device}")
        
        sequences = []
        final_states = []
        
        for seq_idx in range(n_sequences):
            print(f"Generating sequence {seq_idx + 1}/{n_sequences}")
            
            # Reset environment
            env.reset()
            actions = []
            
            for step in range(max_steps):
                try:
                    # Try to get valid actions
                    if hasattr(env, 'get_mask_invalid_actions_forward'):
                        mask = env.get_mask_invalid_actions_forward()
                        if mask is not None:
                            # Get action space
                            if hasattr(env, 'get_action_space'):
                                action_space = env.get_action_space()
                                if len(action_space) > 0:
                                    # Filter valid actions - handle different mask types
                                    if isinstance(mask, torch.Tensor):
                                        valid_indices = torch.where(~mask)[0]
                                    elif isinstance(mask, (list, np.ndarray)):
                                        mask_array = np.array(mask)
                                        valid_indices = np.where(~mask_array)[0]
                                    else:
                                        # Unknown mask type, use all actions
                                        valid_indices = list(range(len(action_space)))
                                    
                                    if len(valid_indices) > 0:
                                        # Choose random valid action
                                        if isinstance(valid_indices, torch.Tensor):
                                            action_idx = random.choice(valid_indices.tolist())
                                        else:
                                            action_idx = random.choice(valid_indices)
                                        
                                        if action_idx < len(action_space):
                                            action = action_space[action_idx]
                                        else:
                                            break  # EOS or invalid
                                    else:
                                        break  # No valid actions
                                else:
                                    break  # No action space
                            else:
                                # Generate random action as fallback
                                action = generate_random_action()
                        else:
                            # No mask available, generate random action
                            action = generate_random_action()
                    else:
                        # No mask method, generate random action
                        action = generate_random_action()
                    
                    # Execute action
                    prev_state = str(env.state) if hasattr(env, 'state') else "unknown"
                    env.step(action)
                    actions.append(action)
                    
                    # Check if done
                    if hasattr(env, 'done') and env.done:
                        break
                        
                except Exception as e:
                    print(f"  Error at step {step}: {e}")
                    break
            
            if len(actions) > 0:
                sequences.append(actions)
                final_state = str(env.state) if hasattr(env, 'state') else "unknown"
                final_states.append(final_state)
                print(f"  Generated sequence with {len(actions)} actions")
            else:
                print(f"  Failed to generate valid sequence")
        
        return sequences, final_states
        
    except ImportError as e:
        print(f"Could not import Photo environment: {e}")
        return [], []
    except Exception as e:
        print(f"Error during sequence generation: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def generate_random_action():
    """Generate a random action based on the Photo environment structure"""
    # Based on photo.py test function
    a, b, c, d = [random.random() for _ in range(4)]
    
    # Create action components as seen in photo.py
    action_components = [
        (0, a, b, 0),
        (1, c, 0, 0),
        (2, d, 0, 0)
    ]
    
    # Flatten into single tuple
    action = tuple(sum(action_components, ()))
    return action

def main():
    """Main function to generate and save sequences"""
    print("Starting Photo environment sequence generation...")
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate sequences
    sequences, final_states = generate_simple_photo_sequences(n_sequences=10, max_steps=8)
    
    if sequences:
        # Prepare data for saving
        data = {
            'sequences': sequences,
            'final_states': final_states,
            'n_sequences': len(sequences),
            'metadata': {
                'environment': 'Photo',
                'generation_method': 'random_sampling',
                'max_steps': 8
            }
        }
        
        # Save data
        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "photo_random_sequences.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\\nSaved {len(sequences)} sequences to {output_file}")
        
        # Save human-readable version
        csv_file = output_dir / "photo_random_sequences.txt"
        with open(csv_file, 'w') as f:
            f.write(f"Generated {len(sequences)} random action sequences for Photo environment\\n")
            f.write("=" * 60 + "\\n\\n")
            
            for i, (seq, final_state) in enumerate(zip(sequences, final_states)):
                f.write(f"Sequence {i+1}:\\n")
                f.write(f"  Length: {len(seq)}\\n")
                f.write(f"  Actions: {seq}\\n")
                f.write(f"  Final state: {final_state}\\n\\n")
        
        print(f"Saved readable format to {csv_file}")
        
        # Print summary
        print(f"\\nSummary:")
        print(f"- Total sequences: {len(sequences)}")
        print(f"- Average length: {np.mean([len(s) for s in sequences]):.1f}")
        print(f"- Length range: {min(len(s) for s in sequences)}-{max(len(s) for s in sequences)}")
        
        if sequences:
            print(f"\\nExample sequence:")
            print(f"  Actions: {sequences[0]}")
    else:
        print("No valid sequences were generated.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())