#!/bin/bash

# Quick debug run with only 2 training steps on CPU to avoid device issues
python train.py gflownet.optimizer.n_train_steps=3 env=photo proxy=photo

# Alternative: Run with GPU (if you have CUDA properly configured)
# python train.py gflownet.optimizer.n_train_steps=2 env=photo proxy=photo device=cuda
