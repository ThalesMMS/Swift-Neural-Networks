# Swift Neural Network Models

Authors: Antonio Neto and Thales Matheus

## Overview

This repository contains small neural networks for:

- MNIST digit classification (MLP, CNN, and single-head self-attention + FFN)
- XOR toy example (2→4→1)

The primary implementations are in Swift, with Python utilities. The design and binary model format are inspired by https://github.com/djbyrne/mlp.c.

## Repository layout

Source code:

- `mnist_mlp.swift`, `mnist_cnn.swift`, `mnist_attention_pool.swift`, `mlp_simple.swift` (Swift, single-file implementations)

Scripts:

- `digit_recognizer.py` (draw digits and run inference with a saved model)
- `plot_comparison.py` (plot training loss from `logs/`)
- `requirements.txt` (Python dependencies)

Data and outputs:

- `data/` (MNIST IDX files)
- `logs/` (training loss logs; generated and ignored by git)
- `mnist_model.bin` (saved model; generated and ignored by git)
- `logs/training_loss_cnn.txt` (CNN loss log)
- `logs/training_loss_attention_mnist.txt` (attention loss log)

## Models

### MNIST MLP

Architecture:

- Input: 784 neurons (28×28 pixels)
- Hidden: 512 neurons (ReLU)
- Output: 10 neurons (Softmax)

Default training parameters (Swift):

- Learning rate: 0.01
- Batch size: 64
- Epochs: 10

Expected accuracy: ~94–97% depending on backend and hyperparameters.

### MNIST CNN

Architecture:

- Input: 28×28 image
- Conv: 8 filters (3×3) + ReLU
- MaxPool: 2×2
- FC: 1568 → 10

Default training parameters:

- Learning rate: 0.01
- Batch size: 32
- Epochs: 3

### MNIST attention model

Architecture:

- 4×4 patches → 49 tokens
- Token projection + position embeddings + ReLU
- Self-attention (1 head, Q/K/V, 49×49 scores)
- Feed-forward MLP per token (D → FF → D)
- Mean-pool tokens → 10 classes

Default training parameters:

- D model: 16
- FF dim: 32
- Learning rate: 0.01
- Batch size: 32
- Epochs: 5

### XOR model

Architecture:

- Input: 2 neurons
- Hidden: 4 neurons (Sigmoid)
- Output: 1 neuron (Sigmoid)

Training uses 1,000,000 epochs by default.

## Swift GPU acceleration

`mnist_mlp.swift` includes GPU paths for faster training and testing:

- MPS GEMM with shared CPU/GPU buffers
- Custom Metal kernels for bias add, ReLU, softmax, reductions, loss, and SGD
- MPSGraph to run forward, loss, gradients, and updates fully on GPU
- GPU testing path that uses argmax on logits (no softmax needed)

Note: `--mpsgraph` uses a fixed batch size; leftover samples are dropped to keep the graph static.

## MLX Swift implementation

The MLX Swift versions provide GPU acceleration on Apple Silicon and automatic differentiation.

Requirements:

- macOS 14.0+ (Sonoma)
- Apple Silicon (M1/M2/M3)
- Swift 5.9+ or Xcode 15+

Build and run:

```bash
# Build
swift build

# Run MLP (default)
swift run MNISTMLX

# Run CNN
swift run MNISTMLX --model cnn --epochs 3

# Run Attention model
swift run MNISTMLX --model attention --epochs 5

# Custom hyperparameters
swift run MNISTMLX -m mlp -e 10 -b 64 -l 0.005
```

Available models:

| Model | Architecture | Best accuracy |
| --- | --- | --- |
| `mlp` | 784→512→10 (ReLU) | ~97% |
| `cnn` | Conv(3×3,8)→MaxPool→Linear | ~98% |
| `attention` | Patches→Attention→Pool→Linear | ~95% |

See `docs/mlx_migration.md` for the migration guide.

Original vs MLX code:

| Component | Original lines | MLX lines | Reduction |
| --- | --- | --- | --- |
| CNN | 583 | 100 | 83% |
| MLP | 2,223 | 80 | 96% |
| Attention | 972 | 180 | 81% |

## MNISTClassic (modular implementation)

The original monolithic `mnist_mlp.swift` has been refactored into a modular structure for maintainability, testability, and readability.

Architecture:

| Module | Purpose | Lines |
| --- | --- | --- |
| `RNG.swift` | Random number generation (xorshift) | ~40 |
| `Types.swift` | Core neural network types | ~20 |
| `GemmEngine.swift` | GEMM backend interface & selection | ~30 |
| `CPUBackend.swift` | Accelerate/vDSP CPU implementation | ~200 |
| `MetalKernels.swift` | Custom Metal GPU kernels | ~230 |
| `GPUBackend.swift` | MPS GPU backend | ~110 |
| `Activations.swift` | ReLU, softmax, loss functions | ~120 |
| `Training.swift` | Training loops & weight initialization | ~440 |
| `Testing.swift` | Model evaluation & accuracy | ~195 |
| `MPSGraphTraining.swift` | MPSGraph GPU training path | ~340 |
| `DataLoading.swift` | MNIST IDX file parsing | ~75 |
| `CLI.swift` | Command-line argument parsing | ~52 |
| `main.swift` | Entry point & orchestration | ~130 |

Requirements:

- macOS 11.0+ (Big Sur or later)
- Swift 5.5+ or Xcode 13+
- GPU acceleration requires Apple Silicon

Build and run:

```bash
# Build the modular version
swift build --target MNISTClassic

# Run with CPU backend (default)
swift run MNISTClassic

# Run with MPS GPU backend
swift run MNISTClassic --mps

# Run with MPSGraph (fully on-device training)
swift run MNISTClassic --mpsgraph

# Custom hyperparameters
swift run MNISTClassic --mps --epochs 10 --batch 128 --lr 0.005 --hidden 512
```

Backend comparison (local runs):

| Backend | Training speed (1 epoch) | Accuracy | Best for |
| --- | --- | --- | --- |
| CPU | ~73s | 94–97% | CPU-only machines, debugging |
| MPS | ~0.8s | 94–97% | Fast training on Apple Silicon |
| MPSGraph | ~1.2s | 94–97% | On-device ML, graph optimization |

Command-line options:

```bash
--mps          # Use MPS GEMM + Metal kernels (GPU)
--mpsgraph     # Use MPSGraph (fully on-device)
--batch N      # Batch size (default: 64)
--hidden N     # Hidden layer size (default: 512)
--epochs N     # Number of epochs (default: 10)
--lr F         # Learning rate (default: 0.01)
--seed N       # RNG seed for reproducibility
```

The legacy monolithic `mnist_mlp.swift` is kept for reference and can still be built directly.

## Build and run (single-file Swift)

Build:

```bash
swiftc -O mnist_mlp.swift -o mnist_mlp_swift
swiftc -O mlp_simple.swift -o mlp_simple_swift
swiftc -O mnist_cnn.swift -o mnist_cnn_swift
swiftc -O mnist_attention_pool.swift -o mnist_attention_pool_swift
```

Run MNIST MLP:

```bash
./mnist_mlp_swift --mps
./mnist_mlp_swift --mpsgraph
```

Run MNIST CNN:

```bash
./mnist_cnn_swift
```

Run MNIST attention:

```bash
./mnist_attention_pool_swift
```

Run XOR:

```bash
./mlp_simple_swift
```

Options for `mnist_mlp_swift`:

```bash
--mps          # Use MPS GEMM + Metal kernels
--mpsgraph     # Use MPSGraph (train and test on GPU)
--batch N      # Batch size (default: 64)
--hidden N     # Hidden layer size (default: 512)
--epochs N     # Epochs (default: 10)
--lr F         # Learning rate (default: 0.01)
--seed N       # RNG seed (default: 1)
--help         # Print usage
```

Options for `mnist_attention_pool_swift`:

```bash
--batch N      # Batch size (default: 32)
--epochs N     # Epochs (default: 5)
--lr F         # Learning rate (default: 0.01)
--seed N       # RNG seed (default: 1)
--help         # Print usage
```

Note: `mnist_cnn_swift` uses fixed defaults and has no CLI flags.

## Benchmarks (local runs)

All runs used the default settings unless noted. Training time is reported as total training time; for CNN/attention it is the sum of per-epoch times. XOR accuracy is computed with a 0.5 threshold on the final outputs.

| Model | Language | Command | Epochs | Batch | Train time (s) | Test accuracy (%) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST MLP | Swift | `./mnist_mlp_swift` | 10 | 64 | 7.30 | 11.90 | CPU backend (no `--mps`) |
| MNIST CNN | Swift | `./mnist_cnn_swift` | 3 | 32 | 53.21 | 92.35 | Conv8/3×3 + MaxPool |
| MNIST Attention | Swift | `./mnist_attention_pool_swift` | 5 | 32 | 101.71 | 24.53 | D=16, FF=32 |
| XOR MLP | Swift | `./mlp_simple_swift` | 1,000,000 | - | 1.78 | 100.00 | Threshold 0.5 |

Note: results vary by hardware and build flags. The Swift MLP CPU run above did not converge well; try `--mps` or `--mpsgraph` for faster and more stable training.

## MNIST dataset

Expected files under `data/`:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

Download from:

- https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- http://yann.lecun.com/exdb/mnist/

## Visualization

To plot training curves:

```bash
python plot_comparison.py
```

## Digit recognizer UI

The drawing app loads `mnist_model.bin` and runs inference:

```bash
python digit_recognizer.py
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## References

- https://github.com/djbyrne/mlp.c
- http://yann.lecun.com/exdb/mnist/
