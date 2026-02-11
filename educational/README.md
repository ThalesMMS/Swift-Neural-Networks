# Educational Standalone Implementations

Standalone Swift files demonstrating neural network fundamentals through manual backpropagation.

## Purpose

This directory contains single-file implementations that teach neural network concepts from first principles. Each file is a complete, runnable example that implements forward and backward passes explicitly, without relying on automatic differentiation frameworks.

These files are preserved for:
- **Backward Compatibility**: Existing tutorials and references may link to these files
- **Easy Access**: Single-file format makes it easy to study complete implementations
- **Educational Value**: Clear demonstration of how neural networks work under the hood
- **Reference Material**: Useful when implementing custom layers or debugging gradient flow

## Files

### 1. `mlp_simple.swift` (218 lines)
**Best starting point for beginners**

A minimal multi-layer perceptron that learns the XOR function.

**Key Concepts:**
- Basic neural network architecture (2→4→1)
- Sigmoid activation function
- Manual backpropagation with clear variable names
- Gradient descent optimization

**Difficulty:** Beginner

**Compile and run:**
```bash
swift mlp_simple.swift
```

**Expected output:**
```
Training simple MLP to learn XOR...
Epoch 0, Loss: 0.6931
Epoch 1000, Loss: 0.0123
...
Final predictions close to [0, 1, 1, 0]
```

---

### 2. `mnist_mlp.swift` (2053 lines)
**Full MNIST classifier with multiple backend options**

A complete multi-layer perceptron for MNIST digit classification.

**Key Concepts:**
- Full-scale MLP architecture (784→512→10)
- ReLU activation and Softmax output layer
- Cross-entropy loss function
- Three backend implementations:
  - CPU (pure Swift + Accelerate GEMM)
  - MPS GPU (Metal Performance Shaders)
  - MPSGraph (high-level graph API)
- MNIST IDX file format parsing
- Mini-batch training

**Difficulty:** Intermediate

**Compile and run:**
```bash
swift mnist_mlp.swift
```

**Expected accuracy:** ~97% on MNIST test set

---

### 3. `mnist_cnn.swift` (1724 lines)
**Convolutional neural network from scratch**

A CNN implementation demonstrating spatial feature learning.

**Key Concepts:**
- Convolutional layers with explicit im2col/col2im
- Max pooling operations
- Manual gradient calculations for conv/pool layers
- 2D spatial transformations
- Feature map visualization

**Difficulty:** Advanced

**Compile and run:**
```bash
swift mnist_cnn.swift
```

**Expected accuracy:** ~98% on MNIST test set

---

### 4. `mnist_attention_pool.swift` (2281 lines)
**Self-attention mechanism implementation**

The most advanced manual implementation, demonstrating modern attention mechanisms.

**Key Concepts:**
- Multi-head self-attention from scratch
- Query/Key/Value projections
- Attention score calculations
- Position embeddings
- Token pooling
- Softmax attention weights

**Difficulty:** Expert

**Compile and run:**
```bash
swift mnist_attention_pool.swift
```

**Expected accuracy:** ~98% on MNIST test set

---

## Compilation Requirements

All standalone files require:
- **Swift 5.9+**
- **macOS** (for Metal acceleration in some files)
- **MNIST dataset** in `./data/` directory (for mnist_*.swift files)

The MNIST dataset files should be placed at:
```
./data/train-images.idx3-ubyte
./data/train-labels.idx1-ubyte
./data/t10k-images.idx3-ubyte
./data/t10k-labels.idx1-ubyte
```

Download MNIST data from: http://yann.lecun.com/exdb/mnist/

### Quick Compilation

Each standalone file can be compiled and run directly:

```bash
# Simple XOR example (no data files needed)
swift mlp_simple.swift

# MNIST examples (require ./data/ directory)
swift mnist_mlp.swift
swift mnist_cnn.swift
swift mnist_attention_pool.swift
```

### Integration with Swift Package

These files can also use shared utilities from the Swift Package:

```bash
# Build the package first
swift build

# Then run with library linking
swift -I .build/debug -L .build/debug -lMNISTCommon mlp_simple.swift
```

## Recommended Learning Order

We recommend studying these files in the following progression:

### Path 1: Complete Beginner
1. **Start here:** `mlp_simple.swift` - Learn basic neural network concepts
2. **Next:** `MNISTClassic` package - See modular, organized code
3. **Then:** `MNISTMLX` package - Learn modern automatic differentiation
4. **Optional:** Return to manual implementations to deepen understanding

### Path 2: Already Familiar with Neural Networks
1. **Start here:** `mnist_mlp.swift` - See full MNIST classifier
2. **Compare:** `Sources/MNISTMLX/MLPModel.swift` - Manual vs auto-diff
3. **Deep dive:** `mnist_cnn.swift` or `mnist_attention_pool.swift` - Advanced architectures
4. **Optional:** `MNISTClassic/GPUBackend.swift` - GPU optimization techniques

### Path 3: Want Production-Ready Code
**Skip standalone files entirely** and go straight to:
1. **`MNISTMLX`** - Modern automatic differentiation with MLX
2. **`MNISTData`** - Efficient data loading utilities
3. **`MNISTCommon`** - Shared utility functions

Run production code:
```bash
swift run MNISTMLX --model mlp --epochs 10
swift run MNISTMLX --model cnn --epochs 5
swift run MNISTMLX --model attention --epochs 5
```

## Why Standalone Files?

**Advantages:**
- **Self-contained**: Everything in one file, easy to understand the full picture
- **No dependencies**: Can read and run without understanding package structure
- **Educational clarity**: Shows complete data flow from input to output
- **Easy to share**: Single-file format is perfect for tutorials and blog posts
- **Debugging aid**: Explicit code helps understand what automatic tools are doing

**Limitations:**
- **Code duplication**: Similar code repeated across files
- **Not production-ready**: Verbose and harder to maintain
- **Limited scalability**: Difficult to extend to new architectures
- **No automatic differentiation**: Manual gradient calculations are error-prone

## Migration to Production Code

When you're ready to build real applications, migrate to the modular Swift Package:

**Instead of:**
```bash
swift mnist_mlp.swift
```

**Use:**
```bash
swift run MNISTMLX --model mlp
```

**Benefits of migration:**
- **10-20x less code**: Automatic differentiation eliminates manual gradients
- **Faster development**: Add new layers without computing gradients by hand
- **Better performance**: Optimized MLX backend with GPU acceleration
- **Modern features**: Checkpointing, progress bars, configurable architectures
- **Production-ready**: Clean APIs, error handling, comprehensive testing

## Detailed Learning Resources

For comprehensive learning guidance, see:

**[LEARNING_GUIDE.md](../LEARNING_GUIDE.md)** - Complete learning paths with detailed progression

This guide includes:
- Learning path recommendations based on your background
- Side-by-side comparison of manual vs automatic differentiation
- Detailed file reference with difficulty ratings
- FAQ addressing common questions
- Architectural comparison charts

## Design Philosophy

These standalone files follow a consistent design philosophy:

1. **Clarity over efficiency**: Code is written to be readable and educational
2. **Explicit over implicit**: All operations are spelled out, no hidden magic
3. **Comments explain "why"**: Not just what the code does, but why it's needed
4. **Complete examples**: Each file runs end-to-end without external setup
5. **Progressive complexity**: Files build on each other in logical progression

## Common Questions

### "Which file should I read first?"
**Answer:** Start with `mlp_simple.swift` if you're new to neural networks. It's only 218 lines and demonstrates all core concepts.

### "Should I use these for production?"
**Answer:** No. Use `swift run MNISTMLX` for production applications. These files are educational references only.

### "Why keep these if there's a better modular version?"
**Answer:**
1. Many existing tutorials link to these files
2. Single-file format is easier for learning and sharing
3. Explicit implementations help debug automatic differentiation
4. Some learners prefer complete examples over modular code

### "Can I compile these with the Swift Package?"
**Answer:** Yes! The standalone files can import shared utilities like `MNISTCommon`:
```bash
swift build
swift -I .build/debug -L .build/debug -lMNISTCommon mlp_simple.swift
```

### "Where do I get MNIST data?"
**Answer:** Download from http://yann.lecun.com/exdb/mnist/ and extract to `./data/` directory.

### "What's the difference between these and MNISTClassic?"
**Answer:**
- **Standalone files**: Single-file, educational focus, manual backprop
- **MNISTClassic**: Modular refactoring with multiple backends (CPU/GPU/MPSGraph)
- Both use manual backpropagation, but MNISTClassic is better organized

### "What's the difference between MNISTClassic and MNISTMLX?"
**Answer:**
- **MNISTClassic**: Manual backprop, educational, explicit gradients
- **MNISTMLX**: Automatic differentiation (MLX), production-ready, modern API

---

## Summary

| File | Lines | Difficulty | Key Concepts | Runtime |
|------|-------|-----------|--------------|---------|
| `mlp_simple.swift` | 218 | Beginner | Basic NN, backprop, XOR | <1 sec |
| `mnist_mlp.swift` | 2053 | Intermediate | Full MLP, MNIST, multiple backends | ~30 sec |
| `mnist_cnn.swift` | 1724 | Advanced | Conv layers, pooling, spatial features | ~2 min |
| `mnist_attention_pool.swift` | 2281 | Expert | Self-attention, Q/K/V, position encoding | ~2 min |

**Total code saved by using MNISTMLX instead:** ~6000+ lines of manual gradient calculations!

---

**For production use:** `swift run MNISTMLX --help`

**For complete learning guidance:** See [LEARNING_GUIDE.md](../LEARNING_GUIDE.md)

**For package documentation:** See [Sources/*/README.md](../Sources/)
