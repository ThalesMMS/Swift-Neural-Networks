# MLX Swift Migration Guide

This document explains the migration from manual neural network implementations to MLX Swift, Apple's GPU-accelerated machine learning framework.

## Why MLX Swift?

### Before: Manual Implementation
The original code required:
- **Manual forward passes** with nested loops
- **Hand-coded backpropagation** (chain rule applied manually)
- **Custom Metal shaders** for GPU acceleration
- **Buffer management** for CPU/GPU data transfer

### After: MLX Swift
MLX provides:
- **Automatic differentiation** via `valueAndGrad()`
- **Native GPU acceleration** on Apple Silicon
- **Pre-built layers**: `Conv2d`, `Linear`, `MaxPool2d`, etc.
- **Built-in optimizers**: `SGD`, `Adam`, etc.

## Code Comparison

### CNN Forward Pass

**Before** (`mnist_cnn.swift`, ~50 lines):
```swift
for b in 0..<batch {
    for oc in 0..<convOut {
        for oy in 0..<imgH {
            for ox in 0..<imgW {
                var sum = bias
                for ky in 0..<kernel {
                    for kx in 0..<kernel {
                        // Manual convolution...
                    }
                }
            }
        }
    }
}
```

**After** (`CNNModel.swift`, ~10 lines):
```swift
var h = conv1(x)      // Conv2d handles everything
h = relu(h)           // Built-in activation
h = pool(h)           // MaxPool2d
h = h.reshaped(...)   // Flatten
h = fc(h)             // Linear
```

### Gradient Computation

**Before** (~200 lines of backprop):
```swift
func fcBackward(...) { /* 35 lines */ }
func maxPoolBackwardRelu(...) { /* 40 lines */ }
func convBackward(...) { /* 35 lines */ }
// Plus ReLU backward, softmax backward, etc.
```

**After** (1 line):
```swift
let lossAndGrad = valueAndGrad(model: model, cnnLoss)
let (loss, grads) = lossAndGrad(model, images, labels)
```

## File Structure

```
Swift-Neural-Networks/
├── Package.swift                    # SPM manifest
├── Sources/
│   ├── MNISTData/
│   │   └── MNISTLoader.swift        # Data loading
│   └── MNISTMLX/
│       ├── CNNModel.swift           # CNN implementation
│       ├── MLPModel.swift           # MLP implementation
│       ├── AttentionModel.swift     # Attention implementation
│       └── main.swift               # CLI entry point
├── mnist_cnn.swift                  # Original (preserved)
├── mnist_mlp.swift                  # Original (preserved)
└── mnist_attention_pool.swift       # Original (preserved)
```

## Lines of Code Comparison

| Component | Original | MLX Swift | Reduction |
|-----------|----------|-----------|-----------|
| CNN Model | 583 | 100 | **83%** |
| MLP Model | 2,223 | 80 | **96%** |
| Attention | 972 | 180 | **81%** |
| **Total** | **3,778** | **360** | **90%** |

## Key MLX Concepts

### 1. Modules and Layers

```swift
// Create a model by subclassing Module
class MyModel: Module {
    @ModuleInfo(key: "layer") var layer: Linear
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return layer(x)
    }
}
```

### 2. Automatic Differentiation

```swift
// Define a loss function
func loss(model: MyModel, x: MLXArray, y: MLXArray) -> MLXArray {
    return crossEntropy(logits: model(x), targets: y, reduction: .mean)
}

// Create gradient function (MLX magic!)
let lossAndGrad = valueAndGrad(model: model, loss)

// Get loss AND gradients in one call
let (lossValue, gradients) = lossAndGrad(model, inputs, labels)
```

### 3. Optimizers

```swift
let optimizer = SGD(learningRate: 0.01)

// After computing gradients:
optimizer.update(model: model, gradients: grads)
```

### 4. Lazy Evaluation

MLX uses lazy evaluation for efficiency. Force evaluation with:

```swift
eval(model, optimizer)  // Triggers actual computation
```

## Requirements

- **macOS 14.0+** (Sonoma or later)
- **Apple Silicon** (M1/M2/M3)
- **Swift 5.9+** or Xcode 15+

## Running the MLX Version

```bash
# Build
swift build

# Run with default MLP
swift run MNISTMLX

# Run CNN
swift run MNISTMLX --model cnn --epochs 3

# Run Attention
swift run MNISTMLX --model attention --epochs 5

# Custom hyperparameters
swift run MNISTMLX -m mlp -e 10 -b 64 -l 0.005
```

## Performance Notes

- **GPU Acceleration**: MLX automatically uses Metal on Apple Silicon
- **Unified Memory**: No explicit CPU↔GPU transfers needed
- **Lazy Evaluation**: Operations are batched for efficiency
- **Compiled Graphs**: Repeated operations are optimized

## Original vs MLX Training Time

| Model | Original (CPU) | MLX (GPU) | Speedup |
|-------|---------------|-----------|---------|
| MLP | ~7s/epoch | ~0.5s/epoch | **14×** |
| CNN | ~53s/epoch | ~2s/epoch | **26×** |
| Attention | ~100s/epoch | ~3s/epoch | **33×** |

*Times are approximate and depend on hardware.*

## Further Reading

- [MLX Swift Documentation](https://ml-explore.github.io/mlx-swift/)
- [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples)
- [Original MLX Paper](https://arxiv.org/abs/2307.05714)
