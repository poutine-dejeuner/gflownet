#!/usr/bin/env python3
"""
Generate random action sequences for Photo environment with enhanced functionality
"""

import sys
import pickle
import random
import argparse
from pathlib import Path
import torch
import numpy as np
import json

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def generate_enhanced_photo_sequences(n_sequences=50, max_steps=20, seed=42, output_dir="data"):
    """
    Generate random action sequences with enhanced functionality
    """
    # Set seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    try:
        from gflownet.envs.photo import Photo
        
        print(f"Initializing Photo environment with seed={seed}...")
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
        sequence_lengths = []
        action_types = {}
        
        successful_sequences = 0
        
        for seq_idx in range(n_sequences):
            if (seq_idx + 1) % 10 == 0:
                print(f"Generating sequence {seq_idx + 1}/{n_sequences} (success rate: {successful_sequences}/{seq_idx + 1})")
            
            # Reset environment
            env.reset()
            actions = []
            step_count = 0
            
            for step in range(max_steps):
                try:
                    # Get valid actions
                    action = None
                    
                    if hasattr(env, 'get_mask_invalid_actions_forward') and hasattr(env, 'get_action_space'):
                        mask = env.get_mask_invalid_actions_forward()
                        action_space = env.get_action_space()
                        
                        if mask is not None and len(action_space) > 0:
                            # Handle different mask types
                            if isinstance(mask, torch.Tensor):
                                valid_indices = torch.where(~mask)[0].tolist()
                            elif isinstance(mask, (list, np.ndarray)):
                                mask_array = np.array(mask)
                                valid_indices = np.where(~mask_array)[0].tolist()
                            else:
                                valid_indices = list(range(len(action_space)))
                            
                            if valid_indices:
                                action_idx = random.choice(valid_indices)
                                if action_idx < len(action_space):
                                    action = action_space[action_idx]
                    
                    # Fallback: generate random action
                    if action is None:
                        action = generate_enhanced_random_action()
                    
                    # Execute action
                    try:
                        env.step(action)
                        actions.append(action)
                        step_count += 1
                        
                        # Track action types
                        action_key = str(action[:2]) if len(action) >= 2 else str(action)
                        action_types[action_key] = action_types.get(action_key, 0) + 1
                        
                        # Check if done
                        if hasattr(env, 'done') and env.done:
                            break
                            
                    except Exception as step_error:
                        # Step failed, but we have some actions, so save partial sequence
                        if len(actions) > 0:
                            break
                        else:
                            raise step_error
                        
                except Exception as e:
                    # Action generation or other error
                    if len(actions) > 0:
                        break  # Save partial sequence
                    else:
                        break  # Skip this sequence
            
            # Save sequence if it has actions
            if len(actions) > 0:
                sequences.append(actions)
                final_state = str(env.state) if hasattr(env, 'state') else "unknown"
                final_states.append(final_state)
                sequence_lengths.append(len(actions))
                successful_sequences += 1
        
        if not sequences:
            print("No valid sequences were generated!")
            return None
        
        # Prepare data for saving
        data = {
            'sequences': sequences,
            'final_states': final_states,
            'sequence_lengths': sequence_lengths,
            'action_types': action_types,
            'n_sequences_requested': n_sequences,
            'n_sequences_generated': len(sequences),
            'metadata': {
                'environment': 'Photo',
                'generation_method': 'enhanced_random_sampling',
                'max_steps': max_steps,
                'seed': seed,
                'device': str(device),
                'success_rate': successful_sequences / n_sequences
            }
        }
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save data
        output_file = output_dir / f"photo_sequences_{n_sequences}_{seed}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\\nSaved {len(sequences)} sequences to {output_file}")
        
        # Save metadata as JSON
        metadata_file = output_dir / f"photo_sequences_{n_sequences}_{seed}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(data['metadata'], f, indent=2)
        
        # Save human-readable summary
        summary_file = output_dir / f"photo_sequences_{n_sequences}_{seed}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Photo Environment Random Action Sequences\\n")
            f.write("=" * 60 + "\\n\\n")
            f.write(f"Generation Parameters:\\n")
            f.write(f"  Requested sequences: {n_sequences}\\n")
            f.write(f"  Generated sequences: {len(sequences)}\\n")
            f.write(f"  Success rate: {successful_sequences/n_sequences:.2%}\\n")
            f.write(f"  Max steps per sequence: {max_steps}\\n")
            f.write(f"  Random seed: {seed}\\n")
            f.write(f"  Device: {device}\\n\\n")
            
            f.write(f"Sequence Statistics:\\n")
            f.write(f"  Average length: {np.mean(sequence_lengths):.1f}\\n")
            f.write(f"  Length range: {min(sequence_lengths)}-{max(sequence_lengths)}\\n")
            f.write(f"  Total actions: {sum(sequence_lengths)}\\n\\n")
            
            f.write(f"Action Type Distribution:\\n")
            for action_type, count in sorted(action_types.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {action_type}: {count} ({count/sum(action_types.values()):.1%})\\n")
            
            f.write(f"\\nFirst 5 sequences:\\n")
            for i, seq in enumerate(sequences[:5]):
                f.write(f"  Sequence {i+1} (length {len(seq)}): {seq}\\n")
        
        print(f"Saved summary to {summary_file}")
        
        return data
        
    except ImportError as e:
        print(f"Could not import Photo environment: {e}")
        return None
    except Exception as e:
        print(f"Error during sequence generation: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_enhanced_random_action():
    """Generate a more varied random action"""
    action_type = random.choice(['simple', 'complex'])
    
    if action_type == 'simple':
        # Simple action with one component
        idx = random.randint(0, 31)
        return (-1, idx, 0, 0, 0)
    else:
        # More complex action
        a, b, c = [random.random() for _ in range(3)]
        subenv_idx = random.randint(0, 2)
        action_idx = random.randint(0, 1)
        return (subenv_idx, action_idx, a, b, c)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Generate random action sequences for Photo environment')
    parser.add_argument('--n_sequences', type=int, default=100, help='Number of sequences to generate')
    parser.add_argument('--max_steps', type=int, default=15, help='Maximum steps per sequence')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Starting enhanced Photo environment sequence generation...")
    print(f"Parameters: n_sequences={args.n_sequences}, max_steps={args.max_steps}, seed={args.seed}")
    
    data = generate_enhanced_photo_sequences(
        n_sequences=args.n_sequences,
        max_steps=args.max_steps,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    if data:
        print(f"\\nGeneration Summary:")
        print(f"- Success rate: {data['metadata']['success_rate']:.2%}")
        print(f"- Average length: {np.mean(data['sequence_lengths']):.1f}")
        print(f"- Total actions generated: {sum(data['sequence_lengths'])}")
        print(f"- Unique action types: {len(data['action_types'])}")
        return 0
    else:
        print("Generation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())