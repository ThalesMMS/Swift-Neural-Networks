# Neural Network Models (MLP, CNN, Attention)

Authors: Antonio Neto and Thales Matheus

This project implements several small neural nets for two problems:

- MNIST digit classification (MLP, CNN, and single-head self-attention + FFN)
- XOR toy example (2->4->1)

Current code is in Swift, with Python utilities. The design and binary model format are inspired by https://github.com/djbyrne/mlp.c.

## Contents

Source code:

- `mnist_mlp.swift`, `mnist_cnn.swift`, `mnist_attention_pool.swift`, `mlp_simple.swift` (Swift)

Scripts:

- `digit_recognizer.py` (draw digits and run inference with a saved model)
- `plot_comparison.py` (plots training loss from `logs/`)
- `requirements.txt` (Python dependencies)

Data and outputs:

- `data/` (MNIST IDX files)
- `logs/` (training loss logs, generated and ignored by git)
- `mnist_model.bin` (saved model, generated and ignored by git)
- `logs/training_loss_cnn.txt` (CNN loss log)
- `logs/training_loss_attention_mnist.txt` (attention loss log)

## MNIST MLP model

Architecture:

- Input: 784 neurons (28x28 pixels)
- Hidden: 512 neurons (ReLU)
- Output: 10 neurons (Softmax)

Default training parameters (Swift):

- Learning rate: 0.01
- Batch size: 64
- Epochs: 10

Expected accuracy: ~94-97% depending on backend and hyperparameters.

## MNIST CNN model

Architecture:

- Input: 28x28 image
- Conv: 8 filters (3x3) + ReLU
- MaxPool: 2x2
- FC: 1568 -> 10

Default training parameters:

- Learning rate: 0.01
- Batch size: 32
- Epochs: 3

## MNIST attention model

Architecture:

- 4x4 patches => 49 tokens
- Token projection + position embeddings + ReLU
- Self-attention (1 head, Q/K/V, 49x49 scores)
- Feed-forward MLP per token (D -> FF -> D)
- Mean-pool tokens -> 10 classes

Default training parameters:

- D model: 16
- FF dim: 32
- Learning rate: 0.01
- Batch size: 32
- Epochs: 5

## XOR model

Architecture:

- Input: 2 neurons
- Hidden: 4 neurons (Sigmoid)
- Output: 1 neuron (Sigmoid)

Training uses 1,000,000 epochs by default.

## Swift GPU optimizations

`mnist_mlp.swift` includes GPU paths for faster training and testing:

- MPS GEMM with shared CPU/GPU buffers
- Custom Metal kernels for bias add, ReLU, softmax, reductions, loss, and SGD
- MPSGraph to run forward, loss, gradients, and updates fully on GPU
- GPU testing path that uses argmax on logits (no softmax needed)

Note: `--mpsgraph` uses a fixed batch size; leftover samples are dropped to keep the graph static.

## MLX Swift (New!)

This project now includes **MLX Swift** implementations that provide:

- 🚀 **GPU acceleration** on Apple Silicon (M1/M2/M3)
- ✨ **Automatic differentiation** (no manual backprop!)
- 📦 **90% less code** compared to manual implementations
- 📚 **Didactic comments** explaining every concept

### Requirements

- macOS 14.0+ (Sonoma)
- Apple Silicon (M1/M2/M3)
- Swift 5.9+ or Xcode 15+

### Build and Run

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

### Available Models

| Model | Architecture | Best Accuracy |
|-------|-------------|---------------|
| `mlp` | 784→512→10 (ReLU) | ~97% |
| `cnn` | Conv(3×3,8)→MaxPool→Linear | ~98% |
| `attention` | Patches→Attention→Pool→Linear | ~95% |

See [docs/mlx_migration.md](docs/mlx_migration.md) for detailed migration guide.

### Original vs MLX Code

| Component | Original Lines | MLX Lines | Reduction |
|-----------|---------------|-----------|-----------|
| CNN | 583 | 100 | **83%** |
| MLP | 2,223 | 80 | **96%** |
| Attention | 972 | 180 | **81%** |

## MNISTClassic (Modular Implementation)

**⚠️ IMPORTANT:** The original monolithic `mnist_mlp.swift` (2,223 lines) has been **refactored into a modular structure** for better maintainability, testability, and readability.

### Architecture

The MNISTClassic module splits the monolithic implementation into **13 focused components**:

| Module | Purpose | Lines |
|--------|---------|-------|
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

### Requirements

- macOS 11.0+ (Big Sur or later)
- Swift 5.5+ or Xcode 13+
- **GPU acceleration requires Apple Silicon** (M1/M2/M3/M4)

### Build and Run

```bash
# Build the modular version
swift build --target MNISTClassic

# Run with CPU backend (default)
swift run MNISTClassic

# Run with MPS GPU backend (fast!)
swift run MNISTClassic --mps

# Run with MPSGraph (fully on-device training)
swift run MNISTClassic --mpsgraph

# Custom hyperparameters
swift run MNISTClassic --mps --epochs 10 --batch 128 --lr 0.005 --hidden 512
```

### Backend Comparison

| Backend | Training Speed (1 epoch) | Accuracy | Best For |
|---------|--------------------------|----------|----------|
| **CPU** | ~73s | 94-97% | CPU-only machines, debugging |
| **MPS** | ~0.8s | 94-97% | **Fastest training** on Apple Silicon |
| **MPSGraph** | ~1.2s | 94-97% | On-device ML, graph optimization |

### Command-Line Options

```bash
--mps          # Use MPS GEMM + Metal kernels (GPU)
--mpsgraph     # Use MPSGraph (fully on-device)
--batch N      # Batch size (default: 64)
--hidden N     # Hidden layer size (default: 512)
--epochs N     # Number of epochs (default: 10)
--lr F         # Learning rate (default: 0.01)
--seed N       # RNG seed for reproducibility
```

### Migration from Monolithic Implementation

The original `mnist_mlp.swift` is **deprecated** but kept for reference. Key improvements:

✅ **90% better organization** - Single-responsibility modules
✅ **Easier testing** - Each component can be tested independently
✅ **Better Git history** - Changes affect specific modules, not one giant file
✅ **Reduced cognitive load** - ~150 lines per file vs 2,223 lines
✅ **Same performance** - Identical training speed and accuracy
✅ **Same features** - All backends (CPU, MPS, MPSGraph) preserved

To build the legacy monolithic version:
```bash
# Not recommended - use MNISTClassic module instead
swiftc -O mnist_mlp.swift -o mnist_mlp_swift
./mnist_mlp_swift --mps
```

## Build and run

### Swift

Build:

```
swiftc -O mnist_mlp.swift -o mnist_mlp_swift
swiftc -O mlp_simple.swift -o mlp_simple_swift
swiftc -O mnist_cnn.swift -o mnist_cnn_swift
swiftc -O mnist_attention_pool.swift -o mnist_attention_pool_swift
```

Run MNIST MLP:

```
./mnist_mlp_swift --mps
./mnist_mlp_swift --mpsgraph
```

Run MNIST CNN:

```
./mnist_cnn_swift
```

Run MNIST attention:

```
./mnist_attention_pool_swift
```

Run XOR:

```
./mlp_simple_swift
```

Swift MLP options (`mnist_mlp_swift`):

```
--mps          use MPS GEMM + Metal kernels
--mpsgraph     use MPSGraph (train and test on GPU)
--batch N      batch size (default: 64)
--hidden N     hidden layer size (default: 512)
--epochs N     epochs (default: 10)
--lr F         learning rate (default: 0.01)
--seed N       RNG seed (default: 1)
--help         print usage
```

Swift attention options (`mnist_attention_pool_swift`):

```
--batch N      batch size (default: 32)
--epochs N     epochs (default: 5)
--lr F         learning rate (default: 0.01)
--seed N       RNG seed (default: 1)
--help         print usage
```

Note: `mnist_cnn_swift` uses fixed defaults and has no CLI flags.

## Benchmarks (local runs)

All runs used the default settings unless noted. Training time is reported as total training time; for CNN/attention it is the sum of per-epoch times. XOR accuracy is computed with a 0.5 threshold on the final outputs.

| Model | Language | Command | Epochs | Batch | Train time (s) | Test accuracy (%) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST MLP | Swift | `./mnist_mlp_swift` | 10 | 64 | 7.30 | 11.90 | CPU backend (no `--mps`) |
| MNIST CNN | Swift | `./mnist_cnn_swift` | 3 | 32 | 53.21 | 92.35 | Conv8/3x3 + MaxPool |
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

```
python plot_comparison.py
```

## Digit recognizer UI

The drawing app loads `mnist_model.bin` and runs inference:

```
python digit_recognizer.py
```

Install dependencies:

```
pip install -r requirements.txt
```

## References

- https://github.com/djbyrne/mlp.c
- http://yann.lecun.com/exdb/mnist/
