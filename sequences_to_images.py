#!/usr/bin/env python3
"""
Convert generated Photo sequences to images using the rbf_function and save as PNG
"""

import sys
import pickle
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_photo_sequences(sequence_file):
    """Load sequences from pickle file"""
    sequence_file = Path(sequence_file)
    
    if not sequence_file.exists():
        print(f"Sequence file {sequence_file} not found!")
        return None
    
    with open(sequence_file, 'rb') as f:
        data = pickle.load(f)
    
    return data

def sequences_to_states(sequences):
    """
    Convert action sequences to final states by executing them in the Photo environment
    """
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
        
        states = []
        proxy_states = []
        
        print("Converting action sequences to states...")
        
        for i, action_sequence in enumerate(tqdm(sequences)):
            try:
                # Reset environment
                env.reset()
                
                # Execute action sequence
                for action in action_sequence:
                    env.step(action)
                
                # Get final state
                final_state = env.state
                states.append(final_state)
                
                # Convert to proxy format
                proxy_state = env.state2proxy(final_state)
                proxy_states.append(proxy_state)
                
            except Exception as e:
                print(f"Error processing sequence {i}: {e}")
                # Add empty state for consistency
                states.append(None)
                proxy_states.append(None)
        
        return states, proxy_states
        
    except ImportError as e:
        print(f"Could not import Photo environment: {e}")
        return None, None
    except Exception as e:
        print(f"Error converting sequences: {e}")
        return None, None

def states_to_images(proxy_states, image_shape=(101, 91)):
    """
    Convert proxy states to images using the rbf_function
    """
    try:
        from gflownet.proxy.photo import rbf_function, make_grid, list_to_tensor
        
        # Filter out None states
        valid_states = [state for state in proxy_states if state is not None]
        
        if not valid_states:
            print("No valid states to convert to images")
            return []
        
        print(f"Converting {len(valid_states)} states to images...")
        
        # Create grid coordinates
        rows, cols = image_shape
        grid_points = make_grid(rows, cols)
        
        images = []
        
        # Process states in batches for efficiency
        batch_size = 8
        for batch_start in range(0, len(valid_states), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_states))
            batch_states = valid_states[batch_start:batch_end]
            
            try:
                # Use list_to_tensor to convert the complex state structure
                state_tensors = list_to_tensor(batch_states)
                print(f"Batch {batch_start//batch_size + 1}: converted states to tensor shape {state_tensors.shape}")
                
                # Generate images using rbf_function
                batch_images = rbf_function(state_tensors, grid_points)
                
                # Convert to numpy and add to results
                batch_images = batch_images.detach().cpu().numpy()
                
                for i in range(batch_images.shape[0]):
                    images.append(batch_images[i])
                
            except Exception as e:
                print(f"Error converting batch {batch_start//batch_size + 1}: {e}")
                # Add empty images for this batch
                for _ in range(len(batch_states)):
                    images.append(np.zeros(image_shape))
        
        return images
        
    except ImportError as e:
        print(f"Could not import rbf_function: {e}")
        return []
    except Exception as e:
        print(f"Error in states_to_images: {e}")
        import traceback
        traceback.print_exc()
        return []

def save_images_as_png(images, sequences, output_dir="images", prefix="photo_sequence"):
    """
    Save images as PNG files with metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Saving {len(images)} images to {output_dir}...")
    
    saved_files = []
    
    for i, (image, sequence) in enumerate(tqdm(zip(images, sequences), total=len(images))):
        try:
            # Create filename
            filename = f"{prefix}_{i:03d}_len{len(sequence)}.png"
            filepath = output_dir / filename
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Plot image
            plt.imshow(image, cmap='viridis', aspect='auto')
            plt.colorbar(label='Intensity')
            plt.title(f"Photo Sequence {i+1} (Length: {len(sequence)})")
            plt.xlabel("X coordinate")
            plt.ylabel("Y coordinate")
            
            # Add sequence info as text
            sequence_str = str(sequence[:3]) + "..." if len(sequence) > 3 else str(sequence)
            plt.figtext(0.02, 0.02, f"Actions: {sequence_str}", fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Save image
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            saved_files.append(filepath)
            
        except Exception as e:
            print(f"Error saving image {i}: {e}")
    
    return saved_files

def create_summary_image(images, sequences, output_dir="images", grid_size=(4, 4)):
    """
    Create a summary image showing multiple sequences in a grid
    """
    n_images = min(len(images), grid_size[0] * grid_size[1])
    
    if n_images == 0:
        return None
    
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(n_images):
        ax = axes[i]
        ax.imshow(images[i], cmap='viridis', aspect='auto')
        ax.set_title(f"Seq {i+1} (L={len(sequences[i])})", fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Photo Environment Sequences as Images", fontsize=16, y=0.98)
    
    summary_path = Path(output_dir) / "photo_sequences_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return summary_path

def main():
    """Main function to convert sequences to images"""
    print("Converting Photo sequences to images using rbf_function")
    print("="*60)
    
    # Load sequences
    sequence_file = "data/photo_sequences_50_42.pkl"
    data = load_photo_sequences(sequence_file)
    
    if data is None:
        print("Failed to load sequences")
        return 1
    
    sequences = data['sequences']
    print(f"Loaded {len(sequences)} sequences")
    
    # Convert sequences to states
    states, proxy_states = sequences_to_states(sequences)
    
    if proxy_states is None:
        print("Failed to convert sequences to states")
        return 1
    
    # Filter out None states for image conversion
    valid_sequences = []
    valid_proxy_states = []
    
    for seq, proxy_state in zip(sequences, proxy_states):
        if proxy_state is not None:
            valid_sequences.append(seq)
            valid_proxy_states.append(proxy_state)
    
    print(f"Found {len(valid_proxy_states)} valid states for image conversion")
    
    if not valid_proxy_states:
        print("No valid states to convert")
        return 1
    
    # Convert states to images using rbf_function
    images = states_to_images(valid_proxy_states)
    
    if not images:
        print("Failed to convert states to images")
        return 1
    
    print(f"Generated {len(images)} images")
    
    # Save individual images
    output_dir = "images/photo_sequences"
    saved_files = save_images_as_png(images, valid_sequences, output_dir)
    
    print(f"Saved {len(saved_files)} individual images to {output_dir}")
    
    # Create summary image
    summary_path = create_summary_image(images, valid_sequences, output_dir)
    
    if summary_path:
        print(f"Created summary image: {summary_path}")
    
    # Print statistics
    print(f"\nImage Statistics:")
    print(f"- Total images: {len(images)}")
    print(f"- Image shape: {images[0].shape}")
    print(f"- Value range: [{np.min(images):.3f}, {np.max(images):.3f}]")
    print(f"- Average sequence length: {np.mean([len(s) for s in valid_sequences]):.1f}")
    
    print(f"\nFiles saved in: {output_dir}/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())