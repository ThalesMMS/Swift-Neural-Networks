# Swift Neural Network Models

[![CI](https://github.com/ThalesMMS/Swift-Neural-Networks-dev/workflows/CI/badge.svg)](https://github.com/ThalesMMS/Swift-Neural-Networks-dev/actions)

Authors: Antonio Neto and Thales Matheus

## Overview

This repository contains small neural networks for:

- MNIST digit classification (MLP, CNN, and single-head self-attention + FFN)
- XOR toy example (2→4→1)

The primary implementations are in Swift, with Python utilities. The design and binary model format are inspired by https://github.com/djbyrne/mlp.c.

## Quick Start

### Production Use (Recommended)

For production ML workflows and modern development, use **MNISTMLX** with Apple's MLX framework:

```bash
# Build and run (MLP by default)
swift run MNISTMLX

# Train CNN model
swift run MNISTMLX --model cnn --epochs 3

# Train with compilation for 2-3x speedup
swift run MNISTMLX --model mlp --compile --epochs 10
```

**Why MNISTMLX?**
- Modern automatic differentiation (no manual gradients)
- GPU acceleration on Apple Silicon
- Concise, maintainable code (80-180 lines vs 1700-2200 lines)
- Easy to extend and experiment with
- Production-ready performance

### Learning & Educational Use

To **understand neural networks from scratch**, see **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)** for:
- Step-by-step learning paths (beginner → advanced)
- Manual backpropagation implementations
- Side-by-side comparison of manual vs auto-diff approaches
- Detailed explanations of how neural networks work under the hood

**Quick learning recommendations:**
- **New to neural networks?** Start with `mlp_simple.swift` (218 lines, simple XOR problem)
- **Want to understand MNIST classification?** Study `MNISTClassic` (modular, well-documented)
- **Ready for production?** Jump to `MNISTMLX` (modern, concise, fast)

---

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

## Models (Educational Reference)

> **Note:** The models described in this section are **educational reference implementations** with manual backpropagation. They are excellent for learning how neural networks work, but for production use, we recommend [MNISTMLX](#mlx-swift-implementation-production-recommended) which uses automatic differentiation and modern best practices. See [LEARNING_GUIDE.md](LEARNING_GUIDE.md) for detailed learning paths.

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

### MNIST ResNet

Architecture:

- Input: 28×28 image
- Initial Conv: 1 → 16 channels (3×3) + BatchNorm + ReLU
- ResidualBlock × 3: Skip connections with identity shortcuts
- GlobalAveragePool: 28×28 → 1×1 per channel
- FC: 16 → 10

Default training parameters:

- Learning rate: 0.01
- Batch size: 32
- Epochs: 5
- Parameters: ~10,000

Expected accuracy: ~98%

The ResNet implementation demonstrates residual learning with skip connections that enable training of deeper networks by solving the vanishing gradient problem. Each residual block learns F(x) and adds it to the input via a skip connection: output = F(x) + x.

### MNIST attention model

Architecture:

- 4×4 patches → 49 tokens
- Token projection + position embeddings + ReLU
- Self-attention (1 head, Q/K/V, 49×49 scores)
- Feed-forward MLP per token (D → FF → D)
- Mean-pool tokens → 10 classes

Default training parameters:

- D model: 32
- FF dim: 64
- Learning rate: 0.005
- Batch size: 32
- Epochs: 5

### XOR model

Architecture:

- Input: 2 neurons
- Hidden: 4 neurons (Sigmoid)
- Output: 1 neuron (Sigmoid)

Training uses 1,000,000 epochs by default.

## Unified CLI Interface

> **Note:** This section covers both production ([MNISTMLX](#mlx-swift-implementation-production-recommended)) and educational ([MNISTClassic](#mnistclassic-modular-educational-implementation), single-file) implementations.

All implementations now support a consistent set of command-line flags for easy experimentation:

| Flag | Short | Description | Default |
| --- | --- | --- | --- |
| `--batch` | `-b` | Batch size | 64 (MLP), 32 (CNN/Attention) |
| `--epochs` | `-e` | Number of training epochs | 10 (MLP), 3 (CNN), 5 (Attention) |
| `--lr` | `-l` | Learning rate | 0.01 (MLP/CNN), 0.005 (Attention) |
| `--seed` | `-s` | RNG seed for reproducibility | 1 |
| `--help` | `-h` | Display usage information | - |

Model-specific flags (where applicable):
- `--hidden` / `-n`: Hidden layer size (MLP only, default: 512)
- `--model` / `-m`: Model architecture (MNISTMLX: mlp, cnn, attention)
- `--mps`: Use MPS GPU backend (Classic/single-file)
- `--mpsgraph`: Use MPSGraph backend (Classic/single-file)
- `--data` / `-d`: Path to MNIST data directory (default: `./data`)
- `--compile`: Enable MLX function compilation for faster training (MNISTMLX only)

Examples:
```bash
# MNISTClassic with custom hyperparameters
swift run MNISTClassic -b 128 -e 5 -l 0.005 -s 42

# MNISTMLX CNN with reproducible seed
swift run MNISTMLX -m cnn -e 3 -s 42

# MNISTMLX MLP with compilation for faster training
swift run MNISTMLX -m mlp --compile -e 10 -s 42

# Single-file MLP with GPU and custom params
./mnist_mlp_swift --mps -b 64 -e 10 -l 0.01 -s 123
```

## ANSI Color-Coded Output

All Swift executables support optional ANSI color coding for terminal output, making it easier to quickly scan logs and identify important information.

### Enabling Colors

Set the `ANSI_COLORS` environment variable to `1` to enable colored output:

```bash
# Enable colors for MNISTMLX
ANSI_COLORS=1 swift run MNISTMLX --model mlp --epochs 10

# Enable colors for MNISTClassic
ANSI_COLORS=1 swift run MNISTClassic --epochs 5

# Without the environment variable, output remains plain (backwards compatible)
swift run MNISTMLX --model mlp --epochs 10
```

### Color Meanings

Color coding helps visually distinguish different types of messages:

| Color | Message Type | Example |
| --- | --- | --- |
| Red | Errors | File loading failures, invalid arguments |
| Yellow | Warnings | GPU unavailable, gradient issues |
| Green | Success | Training completed, test accuracy results |
| Cyan | Progress | Epoch updates, training metrics |
| Default | Info | General status messages |

### Implementation Details

- **Opt-in**: Colors are disabled by default for backwards compatibility
- **Environment variable**: Set `ANSI_COLORS=1` to enable
- **Clean output**: When disabled, no ANSI escape codes appear in output
- **All executables**: Works with both MNISTMLX and MNISTClassic

The colored output is particularly useful when training for many epochs or when monitoring long-running experiments, as it allows you to quickly spot warnings, errors, and key metrics in dense training logs.

## Swift GPU Acceleration (Educational Reference)

> **Educational:** This section describes manual GPU implementation details in the educational reference files. For production GPU acceleration, use [MNISTMLX](#mlx-swift-implementation-production-recommended) which handles GPU operations automatically.

`mnist_mlp.swift` includes GPU paths for faster training and testing:

- MPS GEMM with shared CPU/GPU buffers
- Custom Metal kernels for bias add, ReLU, softmax, reductions, loss, and SGD
- MPSGraph to run forward, loss, gradients, and updates fully on GPU
- GPU testing path that uses argmax on logits (no softmax needed)

Note: `--mpsgraph` uses a fixed batch size; leftover samples are dropped to keep the graph static.

## MLX Swift Implementation (Production Recommended)

> **Production:** This is the **recommended implementation** for production use and modern ML development. It provides automatic differentiation, GPU acceleration, and concise code that's easy to maintain and extend.

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

# Custom hyperparameters with short-form flags
swift run MNISTMLX -m mlp -e 10 -b 64 -l 0.005 -s 42

# Display help
swift run MNISTMLX --help

# Enable compilation for faster training
swift run MNISTMLX --compile -e 10
```

Common flags: `--batch` / `-b`, `--epochs` / `-e`, `--lr` / `-l`, `--seed` / `-s`, `--help` / `-h`
Model selection: `--model` / `-m` (mlp, cnn, attention)

### Performance: MLX Compilation

The `--compile` flag enables MLX's just-in-time (JIT) compilation of training functions, significantly improving performance:

**Performance benefits:**
- 2-3x faster training for MLP and CNN models
- Reduced per-iteration overhead through function fusion
- Automatic kernel optimization for Apple Silicon GPUs

**When to use compilation:**
- Training for multiple epochs (compilation overhead amortized)
- Production training runs with fixed hyperparameters
- Benchmarking and performance optimization

**When NOT to use compilation:**
- Quick experimentation with changing model architectures
- Single-epoch runs (compilation overhead not worth it)
- Debugging training code (compiled functions are harder to inspect)

Example:
```bash
# Without compilation (first run for comparison)
swift run MNISTMLX -m mlp -e 10 -s 42

# With compilation (2-3x faster after warm-up)
swift run MNISTMLX -m mlp --compile -e 10 -s 42
```

Note: The first training iteration with `--compile` includes compilation overhead (~1-2 seconds), but subsequent iterations run much faster.

Available models:

| Model | Architecture | Best accuracy |
| --- | --- | --- |
| `mlp` | 784→512→10 (ReLU) | ~97% |
| `cnn` | Conv(3×3,8)→MaxPool→Linear | ~98% |
| `attention` | Patches→Attention→Pool→Linear | ~90% |

See `docs/mlx_migration.md` for the migration guide.

Original vs MLX code:

| Component | Original lines | MLX lines | Reduction |
| --- | --- | --- | --- |
| CNN | 583 | 100 | 83% |
| MLP | 2,223 | 80 | 96% |
| Attention | 972 | 180 | 81% |

## MNISTClassic (Modular Educational Implementation)

> **Educational:** This is a **refactored educational implementation** that demonstrates modular code organization and multiple backend options (CPU, GPU, MPSGraph). Great for understanding code architecture and GPU acceleration, but for production, prefer [MNISTMLX](#mlx-swift-implementation-production-recommended).

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

# Custom hyperparameters (using short-form flags)
swift run MNISTClassic --mps -e 10 -b 128 -l 0.005 --hidden 512 -s 42

# Display help
swift run MNISTClassic --help
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
--batch, -b    # Batch size (default: 64)
--hidden, -n   # Hidden layer size (default: 512)
--epochs, -e   # Number of epochs (default: 10)
--lr, -l       # Learning rate (default: 0.01)
--seed, -s     # RNG seed for reproducibility (default: 1)
--help, -h     # Display usage information
```

The legacy monolithic `mnist_mlp.swift` is kept for reference and can still be built directly.

## Build and Run (Single-File Educational Implementations)

> **Educational Reference:** These are **standalone educational implementations** with manual backpropagation. They are kept for backward compatibility and learning purposes. For production use, see [MNISTMLX](#mlx-swift-implementation-production-recommended).

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
./mnist_mlp_swift --mpsgraph -b 128 -e 5 -l 0.005 -s 42
```

Run MNIST CNN:

```bash
./mnist_cnn_swift
./mnist_cnn_swift -b 64 -e 5 -l 0.01 -s 42
```

Run MNIST attention:

```bash
./mnist_attention_pool_swift
./mnist_attention_pool_swift -b 32 -e 5 -l 0.005 -s 42
```

Run XOR:

```bash
./mlp_simple_swift
```

Options for `mnist_mlp_swift`:

```bash
--mps, --mpsgraph  # GPU backends (MPS or MPSGraph)
--batch, -b        # Batch size (default: 64)
--hidden, -n       # Hidden layer size (default: 512)
--epochs, -e       # Epochs (default: 10)
--lr, -l           # Learning rate (default: 0.01)
--seed, -s         # RNG seed (default: 1)
--data, -d         # Data directory (default: ./data)
--help, -h         # Print usage
```

Options for `mnist_cnn_swift`:

```bash
--batch, -b    # Batch size (default: 32)
--epochs, -e   # Epochs (default: 3)
--lr, -l       # Learning rate (default: 0.01)
--seed, -s     # RNG seed (default: 1)
--data, -d     # Data directory (default: ./data)
--help, -h     # Print usage
```

Options for `mnist_attention_pool_swift`:

```bash
--batch, -b    # Batch size (default: 32)
--epochs, -e   # Epochs (default: 5)
--lr, -l       # Learning rate (default: 0.005)
--seed, -s     # RNG seed (default: 1)
--help, -h     # Print usage
```

## Benchmarks (local runs)

All runs used the default settings unless noted. Training time is reported as total training time; for CNN/attention it is the sum of per-epoch times. XOR accuracy is computed with a 0.5 threshold on the final outputs. MNIST attention accuracy is projected (not yet empirically measured) for the D=32/FF=64/lr=0.005 configuration.

| Model | Language | Command | Epochs | Batch | Train time (s) | Test accuracy (%) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST MLP | Swift | `./mnist_mlp_swift` | 10 | 64 | 7.30 | 11.90 | CPU backend (no `--mps`) |
| MNIST CNN | Swift | `./mnist_cnn_swift` | 3 | 32 | 53.21 | 92.35 | Conv8/3×3 + MaxPool |
| MNIST Attention | Swift | `./mnist_attention_pool_swift` | 5 | 32 | 101.71 | ~90 (projected) | D=32, FF=64, lr=0.005 |
| XOR MLP | Swift | `./mlp_simple_swift` | 1,000,000 | - | 1.78 | 100.00 | Threshold 0.5 |

Note: results vary by hardware and build flags. The Swift MLP CPU run above did not converge well; try `--mps` or `--mpsgraph` for faster and more stable training.

## Test Coverage

This project includes a comprehensive test suite covering data loading, neural network models, activation functions, and backend implementations. The test suite consists of **87 passing tests** across 4 test modules with coverage for CPU and GPU code paths.

### Quick Start

```bash
# Run all tests
swift test

# Run specific test module
swift test --filter MNISTCommonTests
swift test --filter MNISTClassicTests

# Run tests in parallel
swift test --parallel
```

### Test Modules

| Module | Tests | Status | Coverage |
| --- | --- | --- | --- |
| **MNISTCommonTests** | 66 | All passing | Activation functions, RNG |
| **MNISTClassicTests** | 21 | All passing | CPU GEMM operations |
| **MNISTMLXTests** | ~76 | MLX limitation | MLX models (MLP, CNN, loss functions) |
| **MNISTDataTests** | ~47 | MLX limitation | MNIST data loading and batching |

**Total Passing:** 87 tests

### What's Covered

- **Activation Functions**: Softmax correctness, numerical stability, pointer version equivalence
- **Random Number Generation**: Reproducibility, statistical distribution, multiple RNG methods
- **CPU Backend**: GEMM operations, matrix transposition, alpha/beta scaling, numerical precision
- **MLX Models**: MLP/CNN forward pass, gradient flow, loss computation (blocked by MLX Metal library limitation)
- **Data Loading**: IDX format parsing, normalization, batching, shuffling (blocked by MLX Metal library limitation)

For comprehensive testing documentation, test organization, coverage details, known limitations, and guidelines for adding new tests, see **[TESTING.md](TESTING.md)**.

## Continuous Integration

This repository uses GitHub Actions for automated testing and build verification. The CI pipeline runs on every push and pull request to ensure code quality and prevent regressions.

### What runs automatically

The CI workflow (`.github/workflows/ci.yml`) performs the following checks:

- **Build verification**: Compiles all Swift targets using `swift build`
- **Test execution**: Runs the test suite with `swift test`
- **Platform**: Tests run on macOS 14 (Apple Silicon) to support GPU-accelerated code

The build status badge at the top of this README shows the current CI status. You can view detailed workflow runs in the [Actions tab](https://github.com/ThalesMMS/Swift-Neural-Networks-dev/actions).

### Running tests locally

To run the same checks that CI performs:

```bash
# Build all targets
swift build

# Run the test suite
swift test

# Build specific targets
swift build --target MNISTClassic
swift build --target MNISTMLX
```

The test suite includes:

- **Smoke tests**: Verify basic module imports and package structure
- **RNG tests**: Validate random number generator consistency and determinism
- **Integration tests**: Ensure components work together correctly

### Adding new tests

Tests are located in `Tests/Swift-Neural-NetworksTests/`. To add new tests:

1. Create a new test file or add to existing ones
2. Import the modules you want to test: `@testable import MNISTCommon`
3. Write test methods (must start with `test`)
4. Run locally with `swift test` before pushing

Example test:

```swift
import XCTest
@testable import MNISTCommon

final class MyTests: XCTestCase {
    func testExample() {
        // Your test code here
        XCTAssertEqual(2 + 2, 4)
    }
}
```

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

The drawing app loads `mnist_model.bin` and runs inference. Note: currently supports MLP-only model format.

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
