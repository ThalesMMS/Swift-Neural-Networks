# Learning Guide: Swift Neural Networks

## Overview

This repository contains multiple implementations of neural networks in Swift, ranging from **educational manual backpropagation** to **production-ready automatic differentiation**. This guide helps you navigate the codebase based on your learning goals and experience level.

## Quick Navigation

| Learning Goal | Recommended Path | Files to Study |
| --- | --- | --- |
| **Understand neural network fundamentals** | [Beginner Path](#beginner-path) | `mlp_simple.swift`, `MNISTClassic` |
| **Learn backpropagation from scratch** | [Manual Backprop Path](#manual-backprop-path) | `mnist_mlp.swift`, `mnist_cnn.swift` |
| **Build production ML models** | [Modern MLX Path](#modern-mlx-path) | `MNISTMLX`, `MNISTData` |
| **Understand GPU acceleration** | [GPU Optimization Path](#gpu-optimization-path) | `MNISTClassic/GPUBackend.swift` |
| **Compare manual vs auto-diff** | [Comparative Study](#comparative-study) | Both Classic and MLX implementations |

---

## Implementation Categories

### Educational Reference Implementations (Manual Backprop)

**Purpose:** Learn how neural networks work under the hood by implementing everything from scratch.

**Characteristics:**
-Explicit forward and backward passes
-Manual gradient calculations
-No automatic differentiation frameworks
-Single-file implementations for easy study
- Note:Not recommended for production use
- Note:More verbose code (~2000+ lines per model)

**Files:**
1. **`mlp_simple.swift`** (218 lines)
   - **Best starting point for beginners**
   - Simple XOR problem (2→4→1 architecture)
   - Sigmoid activations throughout
   - Manual backprop with clear variable names
   - No external dependencies

2. **`mnist_mlp.swift`** (2053 lines)
   - Full MLP for MNIST (784→512→10)
   - ReLU + Softmax activations
   - Manual backprop with detailed comments
   - CPU, MPS GPU, and MPSGraph backends
   - IDX file parsing for MNIST data

3. **`mnist_cnn.swift`** (1724 lines)
   - Convolutional neural network
   - Manual convolution and pooling operations
   - Explicit gradient calculations for conv/pool layers
   - Demonstrates spatial feature learning

4. **`mnist_attention_pool.swift`** (2281 lines)
   - Self-attention mechanism from scratch
   - Manual Q/K/V projections and attention scores
   - Position embeddings and token pooling
   - Most advanced manual implementation

**When to study these:**
- You're learning neural networks for the first time
- You want to understand backpropagation deeply
- You're debugging gradient flow issues
- You're implementing a new architecture and need intuition

### Modular Educational Implementation (MNISTClassic)

**Purpose:** Refactored manual backprop with clean separation of concerns.

**Characteristics:**
-Organized into logical modules
-Reusable components (backends, activations, data loading)
-Multiple backend options (CPU, MPS, MPSGraph)
-Production-quality code structure
- Note:Still manual backprop (educational focus)
- Note:More complex than single-file versions

**Module Structure:**
```
Sources/MNISTClassic/
├── main.swift              # Entry point and CLI
├── Types.swift             # Core data structures
├── RNG.swift               # Random number generation
├── DataLoading.swift       # MNIST data loading
├── Activations.swift       # ReLU, Softmax
├── CPUBackend.swift        # Pure Swift + Accelerate GEMM
├── GPUBackend.swift        # Metal Performance Shaders
├── MPSGraphTraining.swift  # High-level graph API
├── GemmEngine.swift        # Matrix multiplication abstraction
├── MetalKernels.swift      # Custom GPU kernels
├── Training.swift          # Training loop logic
└── Testing.swift           # Evaluation logic
```

**Study order:**
1. **`Types.swift`** - Understand core data structures (NeuralNet, GemmParams)
2. **`CPUBackend.swift`** - See how GEMM powers forward/backward passes
3. **`Training.swift`** - Follow the training loop logic
4. **`GPUBackend.swift`** - Compare CPU vs GPU implementations
5. **`MPSGraphTraining.swift`** - See higher-level abstractions

**Run it:**
```bash
# Build the project
swift build

# CPU backend (pure Swift + Accelerate)
swift run MNISTClassic --epochs 5 --batch 64

# MPS GPU backend
swift run MNISTClassic --mps --epochs 5

# MPSGraph backend (most automated)
swift run MNISTClassic --mpsgraph --epochs 5
```

### Production Implementation (MNISTMLX)

**Purpose:** Modern, maintainable ML with automatic differentiation.

**Characteristics:**
-**Automatic differentiation** (no manual gradients!)
-**GPU-accelerated** by default (Apple Silicon)
-**Modular architecture** (easy to extend)
-**Multiple models** (MLP, CNN, Attention, ResNet)
-**Production-ready** (proper error handling, checkpointing)
-**~100 lines per model** (vs ~2000 for manual)
-**Actively maintained** and recommended for new projects

**Module Structure:**
```
Sources/MNISTMLX/
├── main.swift              # Entry point and CLI
├── MLPModel.swift          # Multi-layer perceptron (~80 lines)
├── CNNModel.swift          # Convolutional network (~120 lines)
├── AttentionModel.swift    # Transformer-style attention (~150 lines)
├── ResNetModel.swift       # Residual network (~180 lines)
├── TransformerBlock.swift  # Reusable transformer components
├── CompiledTraining.swift  # Optimized training loops
├── Checkpointing.swift     # Model save/load
├── TrainingSummary.swift   # Progress reporting
└── ProgressBar.swift       # Terminal UI
```

**Study order:**
1. **`MLPModel.swift`** - Simplest model, understand MLX basics
2. **`main.swift`** - See how models are trained
3. **`CNNModel.swift`** - Learn convolutional layers
4. **`CompiledTraining.swift`** - Performance optimization techniques
5. **`AttentionModel.swift`** - Advanced architecture patterns

**Run it:**
```bash
# MLP (simple and fast)
swift run MNISTMLX --model mlp --epochs 10

# CNN (better accuracy)
swift run MNISTMLX --model cnn --epochs 3

# Attention (transformer-style)
swift run MNISTMLX --model attention --epochs 5

# ResNet (state-of-the-art)
swift run MNISTMLX --model resnet --epochs 5

# Use compiled training for 2-3× speedup
swift run MNISTMLX --model cnn --compile --epochs 3
```

### Shared Utilities (MNISTCommon & MNISTData)

**Purpose:** Reusable components extracted from duplicate code.

**MNISTCommon (Pure Swift):**
```
Sources/MNISTCommon/
├── SimpleRng.swift       # Reproducible random number generation
├── Activations.swift     # Softmax implementations
├── DataLoading.swift     # IDX file parsing (for Classic)
└── ANSIColors.swift      # Optional colored terminal output
```

**MNISTData (MLX-based):**
```
Sources/MNISTData/
└── MNISTLoader.swift     # Modern data loading with MLXArray
```

**When to study these:**
- You need to load MNIST data in your own project
- You're implementing a custom RNG for reproducibility
- You want to understand IDX file format parsing
- You're adding colored output to CLI tools

---

## Learning Paths

### Beginner Path: "I'm new to neural networks"

**Estimated time:** 2-3 days

**Step 1: XOR Problem (1-2 hours)**
- Read `mlp_simple.swift` from top to bottom
- Understand: forward pass → loss → backward pass → update weights
- Compile and run: `swift mlp_simple.swift`
- Experiment: Change learning rate, hidden layer size, epochs

**Step 2: MNIST MLP (4-6 hours)**
- Study `mnist_mlp.swift` sections in order:
  1. Data structures (lines 1-100)
  2. Forward pass (lines 300-400)
  3. Backward pass (lines 500-700)
  4. Training loop (lines 1500-1700)
- Compile and run: `swift mnist_mlp.swift`
- Experiment: Adjust batch size, learning rate, hidden units

**Step 3: Modern MLX (2-3 hours)**
- Read `Sources/MNISTMLX/MLPModel.swift` (~80 lines)
- Compare to manual MLP: notice **no backward pass code**!
- Run: `swift run MNISTMLX --model mlp --epochs 5`
- Appreciate the power of automatic differentiation

**Key Takeaway:**
You now understand what frameworks like PyTorch/TensorFlow are automating!

---

### Manual Backprop Path: "I want to implement backprop from scratch"

**Estimated time:** 1-2 weeks

**Phase 1: Fundamentals (2-3 days)**
1. **XOR with Sigmoid** (`mlp_simple.swift`)
   - Implement sigmoid derivative: `σ'(x) = σ(x) × (1 - σ(x))`
   - Trace gradient flow for 1 training step by hand
   - Verify: Loss should decrease consistently

2. **MNIST with ReLU** (`mnist_mlp.swift`)
   - Study ReLU backward: `∂L/∂x = ∂L/∂y × (x > 0 ? 1 : 0)`
   - Understand mini-batch gradient accumulation
   - Implement your own version from scratch (don't peek!)

**Phase 2: Convolutional Layers (3-4 days)**
3. **CNN Forward Pass** (`mnist_cnn.swift`)
   - Study `convForward()`: sliding window, im2col optimization
   - Understand max pooling: keep index of max value
   - Trace shapes through network: [N,1,28,28] → [N,10]

4. **CNN Backward Pass** (`mnist_cnn.swift`)
   - Study `convBackward()`: gradient redistribution to conv filters
   - Understand pooling gradient: route to max locations only
   - Debug: Check gradient shapes match forward activations

**Phase 3: Attention Mechanisms (4-5 days)**
5. **Self-Attention Forward** (`mnist_attention_pool.swift`)
   - Study Q/K/V projections: `Q=XW_q, K=XW_k, V=XW_v`
   - Understand attention scores: `softmax(Q·K^T / √d_k)`
   - Trace token mixing: `output = attention_scores · V`

6. **Self-Attention Backward** (`mnist_attention_pool.swift`)
   - Study gradient flow through softmax
   - Understand chain rule through matmuls: `∂L/∂W = X^T · ∂L/∂Y`
   - Most complex backprop in the repo - take your time!

**Final Project:**
Implement a new architecture (e.g., ResNet) with manual backprop. Then compare to MLX version.

---

### Modern MLX Path: "I want to build production models"

**Estimated time:** 3-5 days

**Phase 1: MLX Basics (1 day)**
1. **Read Package.swift** - Understand MLX dependencies
2. **Study MLPModel.swift** - See how `Linear` and `ReLU` are used
3. **Run training:** `swift run MNISTMLX --model mlp --epochs 5`
4. **Check tests:** `swift test --filter MLPModelTests`

**Phase 2: Model Zoo (2 days)**
5. **CNN** (`CNNModel.swift`)
   - Learn: `Conv2d`, `maxPool2d`, `flatten()`
   - Compare to manual conv in `mnist_cnn.swift`
   - Run: `swift run MNISTMLX --model cnn --epochs 3`

6. **Attention** (`AttentionModel.swift`)
   - Learn: `scaled_dot_product_attention()`, `Embedding`
   - Compare to manual attention in `mnist_attention_pool.swift`
   - Run: `swift run MNISTMLX --model attention --epochs 5`

7. **ResNet** (`ResNetModel.swift`)
   - Learn: Skip connections, BatchNorm
   - Understand residual learning: `F(x) + x`
   - Run: `swift run MNISTMLX --model resnet --epochs 5`

**Phase 3: Production Features (1-2 days)**
8. **Checkpointing** (`Checkpointing.swift`)
   - Save/load models with `.safetensors` format
   - Resume training from checkpoints

9. **Compiled Training** (`CompiledTraining.swift`)
   - Use `compile()` for 2-3× speedup
   - Understand JIT compilation tradeoffs

10. **Testing** (`Tests/MNISTMLXTests/`)
    - Study convergence tests
    - Write tests for your own models

**Final Project:**
Build a custom model for a new dataset using MLX patterns.

---

### GPU Optimization Path: "I want to understand GPU acceleration"

**Estimated time:** 1 week

**Phase 1: CPU Baseline (1-2 days)**
1. **Accelerate Framework** (`CPUBackend.swift`)
   - Study `cblas_sgemm()`: Apple's optimized BLAS
   - Understand GEMM parameters: `transposeA`, `transposeB`, `alpha`, `beta`
   - Run: `swift run MNISTClassic --epochs 5` (CPU only)

**Phase 2: Metal Basics (2 days)**
2. **MPS Buffers** (`GPUBackend.swift`)
   - Study shared CPU/GPU memory allocation
   - Understand: `MTLBuffer.contents()` for zero-copy access
   - Compare CPU vs MPS performance: `--mps` flag

3. **Custom Kernels** (`MetalKernels.swift`)
   - Study Metal shader language basics
   - Understand threadgroup size and grid dimensions
   - Implement your own kernel (e.g., custom activation)

**Phase 3: High-Level APIs (2 days)**
4. **MPSGraph** (`MPSGraphTraining.swift`)
   - Study graph construction: forward + loss + gradients
   - Understand automatic gradient generation
   - Compare to manual backprop: way less code!

5. **MLX GPU** (MNISTMLX)
   - Study unified memory architecture
   - Understand lazy evaluation and `eval()`
   - Compare to MPS: simpler API, automatic optimization

**Phase 4: Performance (1-2 days)**
6. **Benchmarking**
   - Run: `scripts/run_benchmarks.sh`
   - Compare: CPU vs MPS vs MPSGraph vs MLX

7. **Optimization**
   - Study compiled training: `CompiledTraining.swift`
   - Profile with Instruments
   - Minimize memory allocations in hot loops

**Final Project:**
Implement a custom GPU kernel for a unique operation (e.g., custom loss function).

---

## Comparative Study

### Side-by-Side: Manual vs Auto-Diff MLP

**Manual Backprop (MNISTClassic):**
```swift
// Forward pass (explicit)
func forward(input: [Float], hidden: inout [Float], output: inout [Float]) {
    // hidden = input · W1 + b1
    gemm(&hidden, input, W1, M: batchSize, N: hiddenSize, K: inputSize)
    addBias(&hidden, b1, batchSize, hiddenSize)
    relu(&hidden, batchSize * hiddenSize)

    // output = hidden · W2 + b2
    gemm(&output, hidden, W2, M: batchSize, N: outputSize, K: hiddenSize)
    addBias(&output, b2, batchSize, outputSize)
}

// Backward pass (explicit - you must write this!)
func backward(dOutput: [Float], dHidden: inout [Float], dInput: inout [Float]) {
    // Gradient through W2
    gemmTranspose(&dW2, hidden, dOutput, M: hiddenSize, N: outputSize, K: batchSize)
    // Gradient through hidden
    gemmTranspose(&dHidden, dOutput, W2, M: batchSize, N: hiddenSize, K: outputSize)
    // Gradient through ReLU
    reluBackward(&dHidden, hidden, batchSize * hiddenSize)
    // Gradient through W1
    gemmTranspose(&dW1, input, dHidden, M: inputSize, N: hiddenSize, K: batchSize)
}
```

**Auto-Diff (MNISTMLX):**
```swift
// Define model structure only
class MLP: Module, UnaryLayer {
    let fc1: Linear
    let fc2: Linear

    init(inputSize: Int, hiddenSize: Int, outputSize: Int) {
        fc1 = Linear(inputSize, hiddenSize)
        fc2 = Linear(hiddenSize, outputSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x.sequential(fc1, relu, fc2)
    }
}

// Backward pass is automatic!
let lossGrad = grad(lossFunction)
let gradients = lossGrad(model, x, y)  // Automatic!
```

**Lines of code:**
- Manual: ~400 lines (forward + backward + weight updates)
- Auto-diff: ~80 lines (forward only)

**Maintainability:**
- Manual: Every architecture change requires updating backward pass
- Auto-diff: Just change forward pass, gradients update automatically

**Performance:**
- Manual: Optimized for specific architecture, can be faster with tuning
- Auto-diff: MLX uses optimized kernels, compiled training for speed

---

## File Reference Guide

### Educational Files (Study These to Learn)

| File | Lines | Purpose | Difficulty | Best For |
| --- | --- | --- | --- | --- |
| `mlp_simple.swift` | 218 | XOR problem, sigmoid | Easy | Beginners |
| `mnist_mlp.swift` | 2053 | MNIST MLP, ReLU | Medium | Learning backprop |
| `mnist_cnn.swift` | 1724 | CNN from scratch | Hard | Conv/pooling details |
| `mnist_attention_pool.swift` | 2281 | Self-attention | Expert | Attention mechanisms |

### Production Files (Use These for Projects)

| Module | Files | Purpose | When to Use |
| --- | --- | --- | --- |
| **MNISTMLX** | `Sources/MNISTMLX/*.swift` | Modern auto-diff models | New projects, production code |
| **MNISTData** | `Sources/MNISTData/MNISTLoader.swift` | Data loading utilities | Need MNIST data in MLXArray format |
| **MNISTCommon** | `Sources/MNISTCommon/*.swift` | Shared utilities (RNG, colors) | Need RNG or colored output |
| **MNISTClassic** | `Sources/MNISTClassic/*.swift` | Modular manual backprop | Learning backends, GPU comparison |

### Documentation Files (Read These First)

| File | Purpose |
| --- | --- |
| **README.md** | Project overview, quick start, model descriptions |
| **LEARNING_GUIDE.md** | This file - learning paths and file navigation |
| **docs/migration.md** | How the project evolved, refactoring history |
| **TESTING.md** | Test suite organization, how to run tests |

---

## Frequently Asked Questions

### Q: Which implementation should I use for my project?

**A:** Use **MNISTMLX** for any new project. It's:
- Maintained and tested
- 5-10× less code than manual implementations
- GPU-accelerated by default
- Easy to extend with new architectures

### Q: Why keep the manual backprop implementations?

**A:** Educational value:
- Understand what PyTorch/TensorFlow automate
- Debug gradient flow issues
- Implement custom operations not in frameworks
- Appreciate the complexity frameworks hide

### Q: Can I use the standalone .swift files directly?

**A:** Not recommended. They're missing the utilities (SimpleRng, data loaders) that were extracted to `MNISTCommon`. Instead:
- Use `swift run MNISTClassic` for manual backprop
- Use `swift run MNISTMLX` for modern auto-diff
- See `docs/migration.md` for how to use standalone files if needed

### Q: I want to implement a new architecture. Where do I start?

**A:** Add to MNISTMLX:
1. Create `Sources/MNISTMLX/MyModel.swift`
2. Define `class MyModel: Module, UnaryLayer`
3. Implement `callAsFunction(_ x: MLXArray) -> MLXArray`
4. Add to `main.swift` model selection
5. Write tests in `Tests/MNISTMLXTests/MyModelTests.swift`

See `MLPModel.swift` as a template (~80 lines).

### Q: How do I run tests?

**A:** See `TESTING.md` for full details. Quick start:
```bash
# All tests
swift test

# Specific module
swift test --filter MNISTMLXTests

# Specific test
swift test --filter testMLPForwardPass
```

### Q: What are the performance differences between implementations?

**A:** Rough benchmarks (M1 MacBook, 1 epoch MNIST):
- **CPU (Accelerate):** ~45 seconds
- **MPS GPU:** ~12 seconds
- **MPSGraph:** ~10 seconds
- **MLX (no compile):** ~15 seconds
- **MLX (compiled):** ~6 seconds

Run `scripts/run_benchmarks.sh` for detailed benchmarks.

### Q: Can I use this code in production?

**A:** MNISTMLX: Yes, with caveats:
-Well-tested, modular, production code quality
-Proper error handling and checkpointing
- Note:Designed for MNIST (small scale)
- Note:May need modifications for larger datasets/models

MNISTClassic/Standalone: No
-Educational reference only
-Manual backprop is error-prone at scale
-Use MLX, PyTorch, or TensorFlow instead

---

## Next Steps

1. **Choose your path** from the sections above based on your goals
2. **Clone the repository** and explore the recommended files
3. **Run the code** - seeing it work helps understanding
4. **Experiment** - modify hyperparameters, architectures, datasets
5. **Read the tests** - they show how each component should behave
6. **Build something new** - apply what you learned to a custom project

## Additional Resources

- **MLX Documentation:** https://ml-explore.github.io/mlx/build/html/index.html
- **MLX Swift Examples:** https://github.com/ml-explore/mlx-swift-examples
- **Original C Implementation:** https://github.com/djbyrne/mlp.c
- **MNIST Dataset:** http://yann.lecun.com/exdb/mnist/

---

If you find issues or have suggestions for improving this guide, please open an issue or pull request.
