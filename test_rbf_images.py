#!/usr/bin/env python3
"""
Create images using rbf_function with synthetic data to test the pipeline
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_synthetic_rbf_images():
    """Create images using synthetic RBF parameters to test the pipeline"""
    try:
        from gflownet.proxy.photo import rbf_function, make_grid
        
        print("Creating synthetic RBF images...")
        
        # Create grid
        rows, cols = 101, 91
        grid_points = make_grid(rows, cols)
        
        # Create several different RBF parameter sets
        synthetic_params = []
        
        # Image 1: Single central Gaussian
        params1 = torch.tensor([
            [[1.0, 50.0, 45.0, 10.0]]  # weight=1.0, center=(50,45), width=10.0
        ], dtype=torch.float32)
        synthetic_params.append(("Single Central Gaussian", params1))
        
        # Image 2: Two Gaussians
        params2 = torch.tensor([
            [[0.8, 30.0, 30.0, 8.0],   # First Gaussian
             [0.6, 70.0, 60.0, 12.0]]  # Second Gaussian
        ], dtype=torch.float32)
        synthetic_params.append(("Two Gaussians", params2))
        
        # Image 3: Four corner Gaussians
        params3 = torch.tensor([
            [[0.5, 20.0, 20.0, 6.0],   # Top-left
             [0.5, 80.0, 20.0, 6.0],   # Top-right
             [0.5, 20.0, 70.0, 6.0],   # Bottom-left
             [0.5, 80.0, 70.0, 6.0]]   # Bottom-right
        ], dtype=torch.float32)
        synthetic_params.append(("Four Corner Gaussians", params3))
        
        # Image 4: Random pattern
        np.random.seed(42)
        n_funcs = 6
        random_weights = np.random.uniform(0.2, 1.0, n_funcs)
        random_centers_x = np.random.uniform(10, 90, n_funcs)
        random_centers_y = np.random.uniform(10, 80, n_funcs)
        random_widths = np.random.uniform(3, 15, n_funcs)
        
        params4 = torch.tensor([[
            [w, cx, cy, wd] for w, cx, cy, wd in zip(random_weights, random_centers_x, random_centers_y, random_widths)
        ]], dtype=torch.float32)
        synthetic_params.append(("Random Pattern", params4))
        
        # Generate images
        images = []
        labels = []
        
        for label, params in synthetic_params:
            print(f"Generating {label}...")
            print(f"  Params shape: {params.shape}")
            print(f"  Params: {params}")
            
            # Generate image
            image = rbf_function(params, grid_points)
            image = image.squeeze(0).detach().cpu().numpy()
            
            images.append(image)
            labels.append(label)
            
            print(f"  Image shape: {image.shape}")
            print(f"  Value range: [{image.min():.3f}, {image.max():.3f}]")
        
        # Save images
        output_dir = Path("images/synthetic_rbf")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Individual images
        for i, (image, label) in enumerate(zip(images, labels)):
            plt.figure(figsize=(10, 8))
            plt.imshow(image, cmap='viridis', aspect='auto', origin='lower')
            plt.colorbar(label='Intensity')
            plt.title(f"Synthetic RBF Image: {label}")
            plt.xlabel("X coordinate")
            plt.ylabel("Y coordinate")
            
            filename = f"synthetic_rbf_{i:02d}_{label.replace(' ', '_').lower()}.png"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved: {filepath}")
        
        # Summary image
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (image, label) in enumerate(zip(images, labels)):
            ax = axes[i]
            im = ax.imshow(image, cmap='viridis', aspect='auto', origin='lower')
            ax.set_title(label)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.suptitle("Synthetic RBF Images", fontsize=16, y=0.98)
        
        summary_path = output_dir / "synthetic_rbf_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\\nCreated summary image: {summary_path}")
        print(f"All images saved in: {output_dir}")
        
        return images, labels
        
    except Exception as e:
        print(f"Error creating synthetic images: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def extract_rbf_params_from_photo_state(state):
    """
    Extract RBF parameters from Photo environment state
    """
    try:
        # State structure: [setflex_state, cubestack_dict]
        # setflex_state: [active_idx, count, active_mask, inactive_mask]
        # cubestack_dict: {idx: [tensor1, tensor2, tensor3]}
        
        if len(state) != 2:
            print(f"Unexpected state structure: {len(state)} elements")
            return None
        
        setflex_state, cubestack_dict = state
        
        # Extract parameters from cubestack_dict
        rbf_params = []
        
        for idx in sorted(cubestack_dict.keys()):
            cube_data = cubestack_dict[idx]
            
            if len(cube_data) >= 3:  # Should have 3 tensors
                tensor1, tensor2, tensor3 = cube_data[:3]
                
                # Convert tensors to CPU and extract values
                if hasattr(tensor1, 'cpu'):
                    tensor1 = tensor1.cpu()
                if hasattr(tensor2, 'cpu'):
                    tensor2 = tensor2.cpu()
                if hasattr(tensor3, 'cpu'):
                    tensor3 = tensor3.cpu()
                
                # Create RBF parameters from tensor values
                # tensor1 might be 2D coordinates, tensor2 and tensor3 might be widths/weights
                if len(tensor1) >= 2 and len(tensor2) >= 1 and len(tensor3) >= 1:
                    # Use first elements as parameters
                    x, y = float(tensor1[0]), float(tensor1[1])
                    width = float(tensor2[0])
                    weight = float(tensor3[0])
                    
                    # Skip if values are invalid (-1 means inactive)
                    if x != -1 and y != -1 and width != -1 and weight != -1:
                        # Normalize coordinates to image space
                        x_norm = (x + 1) * 50  # Map [-1,1] to [0,100]
                        y_norm = (y + 1) * 45  # Map [-1,1] to [0,90]
                        width_norm = max(0.1, abs(width) * 10)  # Scale width
                        weight_norm = max(0.1, abs(weight))  # Use weight as-is
                        
                        rbf_params.append([weight_norm, x_norm, y_norm, width_norm])
        
        if rbf_params:
            # Convert to tensor
            params_tensor = torch.tensor([rbf_params], dtype=torch.float32)
            return params_tensor
        else:
            # Return default single Gaussian
            return torch.tensor([[[0.5, 50.0, 45.0, 5.0]]], dtype=torch.float32)
            
    except Exception as e:
        print(f"Error extracting RBF params: {e}")
        return None

def create_images_from_photo_sequences():
    """Create images from actual Photo sequences using proper parameter extraction"""
    try:
        import pickle
        
        # Load sequences
        with open("data/photo_sequences_50_42.pkl", "rb") as f:
            data = pickle.load(f)
        
        sequences = data['sequences']
        print(f"Loaded {len(sequences)} sequences")
        
        # Convert sequences to states
        states, _ = sequences_to_states(sequences)
        
        # Extract valid states
        valid_states = [state for state in states if state is not None]
        print(f"Found {len(valid_states)} valid states")
        
        if not valid_states:
            print("No valid states found")
            return
        
        # Convert states to RBF parameters and generate images
        from gflownet.proxy.photo import rbf_function, make_grid
        
        rows, cols = 101, 91
        grid_points = make_grid(rows, cols)
        
        images = []
        valid_sequences = []
        
        for i, state in enumerate(valid_states[:10]):  # Limit to first 10 for testing
            try:
                # Extract RBF parameters
                rbf_params = extract_rbf_params_from_photo_state(state)
                
                if rbf_params is not None:
                    # Generate image
                    image = rbf_function(rbf_params, grid_points)
                    image = image.squeeze(0).detach().cpu().numpy()
                    
                    images.append(image)
                    valid_sequences.append(sequences[i])
                    
                    print(f"Generated image {len(images)} from state {i}")
                else:
                    print(f"Could not extract RBF params from state {i}")
                    
            except Exception as e:
                print(f"Error processing state {i}: {e}")
        
        if images:
            # Save images
            output_dir = Path("images/photo_sequences_fixed")
            output_dir.mkdir(exist_ok=True, parents=True)
            
            for i, (image, sequence) in enumerate(zip(images, valid_sequences)):
                plt.figure(figsize=(10, 8))
                plt.imshow(image, cmap='viridis', aspect='auto', origin='lower')
                plt.colorbar(label='Intensity')
                plt.title(f"Photo Sequence {i+1} (Length: {len(sequence)})")
                plt.xlabel("X coordinate")
                plt.ylabel("Y coordinate")
                
                # Add sequence info
                sequence_str = str(sequence[:2]) + "..." if len(sequence) > 2 else str(sequence)
                plt.figtext(0.02, 0.02, f"Actions: {sequence_str}", fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                filename = f"photo_sequence_{i:03d}_len{len(sequence)}.png"
                filepath = output_dir / filename
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"\\nSaved {len(images)} images to {output_dir}")
        else:
            print("No images were generated")
            
    except Exception as e:
        print(f"Error creating images from sequences: {e}")
        import traceback
        traceback.print_exc()

def sequences_to_states(sequences):
    """Convert sequences to states (reused from main script)"""
    try:
        from gflownet.envs.photo import Photo
        
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
        
        for i, action_sequence in enumerate(sequences):
            try:
                env.reset()
                for action in action_sequence:
                    env.step(action)
                states.append(env.state)
            except Exception as e:
                print(f"Error processing sequence {i}: {e}")
                states.append(None)
        
        return states, None
        
    except Exception as e:
        print(f"Error converting sequences: {e}")
        return [], []

def main():
    """Main function"""
    print("Testing RBF function and creating Photo images")
    print("="*60)
    
    # First test with synthetic data
    print("\\n1. Creating synthetic RBF images...")
    synthetic_images, labels = create_synthetic_rbf_images()
    
    if synthetic_images:
        print(f"✓ Successfully created {len(synthetic_images)} synthetic images")
    
    # Then try with actual Photo sequences
    print("\\n2. Creating images from Photo sequences...")
    create_images_from_photo_sequences()
    
    print("\\nDone!")

if __name__ == "__main__":
    main()