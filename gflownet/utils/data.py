from pathlib import Path
import pandas as pd
import pickle
import random
import torch
import numpy as np


def load_additional_replay_data(gflownet, data_config):
    """
    Load additional data into the replay buffer from various sources.
    
    Parameters
    ----------
    gflownet : GFlowNetAgent
        The GFlowNet agent with initialized buffer
    data_config : dict
        Configuration specifying data source and format
        Expected keys:
        - path: path to the data file
        - format: 'csv', 'pkl', or 'samples_only'
        - reward_key: key for rewards in the data (if applicable)
    """
    if gflownet.buffer.replay_capacity == 0:
        print("Warning: Replay buffer capacity is 0. Cannot add additional data.")
        return
        
    path = Path(data_config.path)
    if not path.exists():
        print(f"Warning: Additional replay data file {path} does not exist.")
        return
    
    print(f"Loading additional replay data from {path}")
    
    if data_config.format == 'csv':
        # Load CSV with full replay buffer format
        additional_replay = gflownet.buffer.load_replay_from_path(path)
        # Merge with existing replay buffer
        gflownet.buffer.replay = pd.concat([gflownet.buffer.replay, additional_replay], 
                                         ignore_index=True)
        # Ensure we don't exceed capacity
        if len(gflownet.buffer.replay) > gflownet.buffer.replay_capacity:
            # Keep the highest reward samples
            gflownet.buffer.replay = gflownet.buffer.replay.nlargest(
                gflownet.buffer.replay_capacity, 'rewards'
            )
    
    elif data_config.format == 'pkl':
        # Load pickled data (expecting dict with 'x' and 'energy' keys like the samples)
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if 'x' in data and 'energy' in data:
            samples = data['x']
            energies = data['energy']
            rewards = gflownet.proxy.proxy2reward(energies)
            
            # Create dummy trajectories (this is a limitation)
            trajectories = [[] for _ in samples]  # Empty trajectories
            
            # Add to buffer
            gflownet.buffer.add(
                samples=samples,
                trajectories=trajectories,
                rewards=rewards,
                it=0,  # Mark as pre-training data
                buffer="replay",
                criterion="greater"
            )
    
    elif data_config.format == 'samples_only':
        # Load just samples and compute rewards
        # Format depends on file extension
        if path.suffix == '.csv':
            df = pd.read_csv(path)
            if 'readable' in df.columns:
                samples = [gflownet.env.readable2state(readable) for readable in df['readable']]
            else:
                raise ValueError("CSV must contain 'readable' column for samples")
        else:
            raise ValueError(f"Unsupported file format for samples_only: {path.suffix}")
        
        # Compute rewards using the proxy
        energies = gflownet.proxy(gflownet.env.states2proxy(samples))
        rewards = gflownet.proxy.proxy2reward(energies)
        
        # Create dummy trajectories
        trajectories = [[] for _ in samples]
        
        # Add to buffer
        gflownet.buffer.add(
            samples=samples,
            trajectories=trajectories,
            rewards=rewards,
            it=0,
            buffer="replay",
            criterion="greater"
        )
    
    print(f"Replay buffer now contains {len(gflownet.buffer.replay)} samples")


def generate_random_photo_sequences(n_sequences=100, min_length=5, max_length=20, output_path="random_photo_sequences.pkl"):
    """
    Generate random action sequences for the Photo environment and save them to a file.
    
    Parameters
    ----------
    n_sequences : int
        Number of random sequences to generate
    min_length : int
        Minimum length of each sequence
    max_length : int
        Maximum length of each sequence
    output_path : str
        Path to save the sequences
        
    Returns
    -------
    dict
        Dictionary containing the generated sequences and metadata
    """
    from gflownet.envs.photo import Photo
    
    # Initialize the Photo environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Photo(device=device)
    
    sequences = []
    states = []
    trajectories = []
    valid_sequences = []
    
    print(f"Generating {n_sequences} random action sequences...")
    
    for i in range(n_sequences):
        # Reset environment for new sequence
        env.reset()
        sequence_length = random.randint(min_length, max_length)
        
        actions = []
        sequence_states = [env.state.copy() if hasattr(env.state, 'copy') else env.state]
        valid = True
        
        for step in range(sequence_length):
            try:
                # Get valid actions mask
                action_mask = env.get_mask_invalid_actions_forward()
                
                if action_mask is not None and hasattr(action_mask, 'shape'):
                    # Find valid action indices
                    if isinstance(action_mask, torch.Tensor):
                        valid_indices = torch.where(~action_mask)[0]
                    else:
                        valid_indices = np.where(~action_mask)[0]
                    
                    if len(valid_indices) == 0:
                        # No valid actions, end sequence
                        break
                    
                    # Sample a random valid action index
                    action_idx = random.choice(valid_indices.tolist())
                    
                    # Convert action index to actual action
                    # For photo environment, we need to handle the action space
                    action_space = env.get_action_space()
                    if action_idx < len(action_space):
                        action = action_space[action_idx]
                    else:
                        # EOS action or invalid, break
                        break
                else:
                    # Generate random action directly (fallback)
                    action = generate_random_photo_action()
                
                # Take the action
                old_state = env.state.copy() if hasattr(env.state, 'copy') else env.state
                env.step(action)
                
                actions.append(action)
                sequence_states.append(env.state.copy() if hasattr(env.state, 'copy') else env.state)
                
                # Check if environment is done
                if env.done:
                    break
                    
            except Exception as e:
                print(f"Error in sequence {i}, step {step}: {e}")
                valid = False
                break
        
        if valid and len(actions) > 0:
            sequences.append(actions)
            states.append(sequence_states)
            trajectories.append(actions)  # In this case, actions are the trajectories
            valid_sequences.append(i)
            
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{n_sequences} sequences ({len(valid_sequences)} valid)")
    
    # Prepare data for saving
    data = {
        'sequences': sequences,
        'states': states,
        'trajectories': trajectories,
        'valid_indices': valid_sequences,
        'metadata': {
            'n_sequences_requested': n_sequences,
            'n_sequences_generated': len(valid_sequences),
            'min_length': min_length,
            'max_length': max_length,
            'environment': 'Photo',
            'device': str(device)
        }
    }
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved {len(valid_sequences)} valid sequences to {output_path}")
    
    # Also save as CSV for human readability
    csv_path = output_path.with_suffix('.csv')
    df_data = []
    for i, (seq, states_seq) in enumerate(zip(sequences, states)):
        df_data.append({
            'sequence_id': i,
            'length': len(seq),
            'actions': str(seq),
            'final_state': str(states_seq[-1]) if states_seq else None
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_path, index=False)
    print(f"Also saved readable format to {csv_path}")
    
    return data


def generate_random_photo_action():
    """
    Generate a random action for the Photo environment.
    Based on the structure seen in photo.py test functions.
    
    Returns
    -------
    tuple
        Random action tuple
    """
    # From the photo.py file, actions seem to be tuples with 4 elements each
    # and there are multiple sub-actions combined
    a, b, c, d = [random.random() for _ in range(4)]
    
    # Based on the test function in photo.py
    action_components = [
        (0, a, b, 0),
        (1, c, 0, 0), 
        (2, d, 0, 0)
    ]
    
    # Flatten the action components
    action = tuple(sum(action_components, ()))
    return action


def generate_photo_action_sequences_script():
    """
    Standalone script function to generate random photo sequences.
    Can be called from command line or other scripts.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate random action sequences for Photo environment')
    parser.add_argument('--n_sequences', type=int, default=100, help='Number of sequences to generate')
    parser.add_argument('--min_length', type=int, default=5, help='Minimum sequence length')
    parser.add_argument('--max_length', type=int, default=20, help='Maximum sequence length')
    parser.add_argument('--output_path', type=str, default='random_photo_sequences.pkl', help='Output file path')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    data = generate_random_photo_sequences(
        n_sequences=args.n_sequences,
        min_length=args.min_length,
        max_length=args.max_length,
        output_path=args.output_path
    )
    
    print(f"Generation complete! Generated {data['metadata']['n_sequences_generated']} valid sequences.")
    return data
