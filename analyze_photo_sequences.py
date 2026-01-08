#!/usr/bin/env python3
"""
Utility script to load and analyze Photo environment action sequences
"""

import sys
import pickle
import json
from pathlib import Path
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_and_analyze_sequences(filepath):
    """
    Load and analyze generated action sequences
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"File {filepath} does not exist!")
        return None
    
    print(f"Loading sequences from {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\\nLoaded data with keys: {list(data.keys())}")
        
        sequences = data['sequences']
        metadata = data['metadata']
        
        print(f"\\nDataset Summary:")
        print(f"- Environment: {metadata['environment']}")
        print(f"- Number of sequences: {len(sequences)}")
        print(f"- Generation method: {metadata['generation_method']}")
        print(f"- Success rate: {metadata.get('success_rate', 'N/A')}")
        print(f"- Device: {metadata.get('device', 'N/A')}")
        print(f"- Seed: {metadata.get('seed', 'N/A')}")
        
        # Sequence length analysis
        lengths = [len(seq) for seq in sequences]
        print(f"\\nSequence Length Statistics:")
        print(f"- Average length: {np.mean(lengths):.2f}")
        print(f"- Length std: {np.std(lengths):.2f}")
        print(f"- Min length: {min(lengths)}")
        print(f"- Max length: {max(lengths)}")
        print(f"- Total actions: {sum(lengths)}")
        
        # Action type analysis
        if 'action_types' in data:
            action_types = data['action_types']
            print(f"\\nAction Type Analysis:")
            print(f"- Unique action types: {len(action_types)}")
            print(f"- Most common actions:")
            sorted_actions = sorted(action_types.items(), key=lambda x: x[1], reverse=True)
            for action_type, count in sorted_actions[:10]:
                percentage = count / sum(action_types.values()) * 100
                print(f"  {action_type}: {count} ({percentage:.1f}%)")
        
        # Show some example sequences
        print(f"\\nExample Sequences:")
        for i, seq in enumerate(sequences[:5]):
            print(f"  {i+1}. Length {len(seq)}: {seq}")
        
        return data
        
    except Exception as e:
        print(f"Error loading sequences: {e}")
        return None

def demonstrate_sequence_usage(data):
    """
    Demonstrate how to use the loaded sequences
    """
    if not data:
        return
    
    print(f"\\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    sequences = data['sequences']
    
    print(f"\\n1. Iterating through all sequences:")
    print(f"   for i, sequence in enumerate(sequences):")
    print(f"       print(f'Sequence {{i}}: {{len(sequence)}} actions')")
    print(f"       for action in sequence:")
    print(f"           # Process each action")
    print(f"           pass")
    
    print(f"\\n2. Filtering sequences by length:")
    long_sequences = [seq for seq in sequences if len(seq) >= 3]
    print(f"   long_sequences = [seq for seq in sequences if len(seq) >= 3]")
    print(f"   # Found {len(long_sequences)} sequences with 3+ actions")
    
    print(f"\\n3. Converting to replay buffer format:")
    print(f"   # To add these sequences to a replay buffer, you would need:")
    print(f"   # - Convert actions to proper trajectory format")
    print(f"   # - Compute rewards using the proxy")
    print(f"   # - Use gflownet.buffer.add() method")
    
    print(f"\\n4. Action analysis:")
    all_actions = [action for seq in sequences for action in seq]
    print(f"   all_actions = [action for seq in sequences for action in seq]")
    print(f"   # Total actions across all sequences: {len(all_actions)}")
    
    if all_actions:
        print(f"   # Example action structure: {all_actions[0]}")
        print(f"   # Action components: {len(all_actions[0])} elements")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Photo environment action sequences')
    parser.add_argument('filepath', nargs='?', 
                       default='data/photo_sequences_50_42.pkl',
                       help='Path to the pickle file containing sequences')
    
    args = parser.parse_args()
    
    # Load and analyze
    data = load_and_analyze_sequences(args.filepath)
    
    if data:
        # Demonstrate usage
        demonstrate_sequence_usage(data)
        
        print(f"\\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print(f"\\n1. To generate more sequences:")
        print(f"   python enhanced_photo_generator.py --n_sequences 100 --max_steps 20")
        
        print(f"\\n2. To add these to a replay buffer:")
        print(f"   # Modify your training config to include:")
        print(f"   additional_replay_data:")
        print(f"     path: '{args.filepath}'")
        print(f"     format: 'custom'")
        
        print(f"\\n3. To analyze specific action patterns:")
        print(f"   # Use the loaded data to study action distributions")
        print(f"   # and sequence patterns for your research")
        
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())