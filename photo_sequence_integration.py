#!/usr/bin/env python3
"""
Example script showing how to integrate random Photo sequences into training
"""

import sys
from pathlib import Path
import pickle
import torch
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_sequences_for_training(sequence_file, gflownet_agent):
    """
    Load action sequences and add them to the training workflow
    
    Parameters
    ----------
    sequence_file : str or Path
        Path to the pickle file with sequences
    gflownet_agent : GFlowNetAgent
        Initialized GFlowNet agent
    """
    sequence_file = Path(sequence_file)
    
    if not sequence_file.exists():
        print(f"Sequence file {sequence_file} not found!")
        return False
    
    try:
        # Load sequences
        with open(sequence_file, 'rb') as f:
            data = pickle.load(f)
        
        sequences = data['sequences']
        print(f"Loaded {len(sequences)} action sequences")
        
        # Convert sequences to states and compute rewards
        states = []
        trajectories = []
        rewards = []
        
        for i, action_sequence in enumerate(sequences):
            try:
                # Reset environment
                gflownet_agent.env.reset()
                
                # Execute action sequence to get final state
                sequence_states = []
                for action in action_sequence:
                    sequence_states.append(gflownet_agent.env.state)
                    gflownet_agent.env.step(action)
                
                # Get final state
                final_state = gflownet_agent.env.state
                
                # Compute reward using proxy
                proxy_state = gflownet_agent.env.state2proxy(final_state)
                energy = gflownet_agent.proxy(proxy_state.unsqueeze(0))
                reward = gflownet_agent.proxy.proxy2reward(energy)
                
                states.append(final_state)
                trajectories.append(action_sequence)
                rewards.append(reward.item())
                
            except Exception as e:
                print(f"Error processing sequence {i}: {e}")
                continue
        
        if states:
            # Add to replay buffer
            print(f"Adding {len(states)} valid sequences to replay buffer...")
            gflownet_agent.buffer.add(
                samples=states,
                trajectories=trajectories,
                rewards=rewards,
                it=0,  # Mark as pre-training data
                buffer="replay",
                criterion="greater"
            )
            
            print(f"Replay buffer now contains {len(gflownet_agent.buffer.replay)} samples")
            return True
        else:
            print("No valid states could be generated from sequences")
            return False
            
    except Exception as e:
        print(f"Error loading sequences: {e}")
        return False

def example_training_with_sequences():
    """
    Example of how to use sequences in training
    """
    print("Example: Integrating Photo sequences into training")
    print("="*60)
    
    # This is a conceptual example - in practice you'd use hydra config
    print("\\n1. Initialize GFlowNet with Photo environment:")
    print("   from gflownet.utils.common import gflownet_from_config")
    print("   config = load_config('config/experiments/photo.yaml')")
    print("   gflownet = gflownet_from_config(config)")
    
    print("\\n2. Load pre-generated sequences:")
    print("   load_sequences_for_training('data/photo_sequences_50_42.pkl', gflownet)")
    
    print("\\n3. Train with enhanced replay buffer:")
    print("   gflownet.train()  # Will use the loaded sequences in replay buffer")
    
    print("\\n4. Alternative: Use sequences as training data:")
    print("   # Modify buffer config to use sequences as train data:")
    print("   buffer:")
    print("     train:")
    print("       type: pkl")
    print("       path: 'data/photo_sequences_processed.pkl'")

def create_example_config():
    """
    Create an example config file for using sequences
    """
    config_content = """# Example config for using pre-generated Photo sequences

defaults:
  - override /env: photo
  - override /gflownet: trajectorybalance
  - override /proxy: photo
  - override /logger: wandb

# Buffer with replay buffer enabled
buffer:
  replay_capacity: 1000  # Enable replay buffer
  train:
    type: null  # No predefined training data
  test:
    type: random
    n: 100

# GFlowNet with replay sampling
gflownet:
  replay_sampling: weighted  # Use weighted sampling for better performance
  train_sampling: permutation
  optimizer:
    batch_size:
      forward: 32
      backward_replay: 16  # Use replay buffer samples
    lr: 0.001
    n_train_steps: 5000

# Additional sequences to load into replay buffer
additional_replay_data:
  path: "data/photo_sequences_50_42.pkl"
  format: "custom"  # Custom format for our generated sequences
"""
    
    output_file = Path("config/experiments/photo_with_sequences.yaml")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(config_content)
    
    print(f"Created example configuration: {output_file}")
    return output_file

def main():
    """Main demonstration"""
    print("Photo Environment Action Sequence Integration")
    print("="*60)
    
    # Show example usage
    example_training_with_sequences()
    
    # Create example config
    config_file = create_example_config()
    
    print(f"\\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\\nGenerated files:")
    print(f"- Action sequences: data/photo_sequences_50_42.pkl")
    print(f"- Sequence analysis: data/photo_sequences_50_42_summary.txt")
    print(f"- Example config: {config_file}")
    
    print(f"\\nScripts created:")
    print(f"- simple_photo_generator.py: Basic sequence generation")
    print(f"- enhanced_photo_generator.py: Advanced sequence generation")
    print(f"- analyze_photo_sequences.py: Sequence analysis")
    print(f"- photo_sequence_integration.py: Training integration example")
    
    print(f"\\nUsage examples:")
    print(f"1. Generate sequences:")
    print(f"   python enhanced_photo_generator.py --n_sequences 100")
    
    print(f"\\n2. Analyze sequences:")
    print(f"   python analyze_photo_sequences.py data/photo_sequences_50_42.pkl")
    
    print(f"\\n3. Train with sequences (conceptual):")
    print(f"   python train.py --config-name photo_with_sequences")
    
    print(f"\\nThe sequences contain various action types for the Photo environment")
    print(f"and can be used to initialize replay buffers or as training data.")

if __name__ == "__main__":
    main()