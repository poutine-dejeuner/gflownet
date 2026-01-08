#!/usr/bin/env python3
"""
Script to generate random action sequences for the Photo environment
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gflownet.utils.data import generate_random_photo_sequences

def main():
    """Generate random photo sequences and save to file"""
    
    print("Generating random action sequences for Photo environment...")
    
    # Generate sequences
    try:
        data = generate_random_photo_sequences(
            n_sequences=50,  # Start with smaller number for testing
            min_length=3,
            max_length=10,
            output_path="data/random_photo_sequences.pkl"
        )
        
        print("\nGeneration Summary:")
        print(f"- Requested: {data['metadata']['n_sequences_requested']} sequences")
        print(f"- Generated: {data['metadata']['n_sequences_generated']} valid sequences")
        print(f"- Length range: {data['metadata']['min_length']}-{data['metadata']['max_length']}")
        print(f"- Environment: {data['metadata']['environment']}")
        print(f"- Device: {data['metadata']['device']}")
        
        if data['sequences']:
            print(f"\nExample sequence (first one):")
            print(f"- Length: {len(data['sequences'][0])}")
            print(f"- Actions: {data['sequences'][0][:3]}..." if len(data['sequences'][0]) > 3 else data['sequences'][0])
            
    except Exception as e:
        print(f"Error generating sequences: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())