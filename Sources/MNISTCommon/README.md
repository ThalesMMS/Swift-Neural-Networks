# MNISTCommon

A shared utilities library for MNIST neural network examples in Swift.

## Purpose

This module provides common functionality that was previously duplicated across multiple MNIST example files (mnist_mlp.swift, mnist_cnn.swift, mnist_attention_pool.swift, and mlp_simple.swift). By extracting these utilities into a shared library, we eliminate ~580 lines of duplicate code and ensure consistency across all examples.

## Components

### Random Number Generation
- **SimpleRng**: Fast, deterministic pseudo-random number generator using xorshift algorithm
  - Used for neural network weight initialization
  - Used for data shuffling during training

### Data Loading
- **readMnistImages**: Loads MNIST image data from IDX format files
- **readMnistLabels**: Loads MNIST label data from IDX format files
  - Handles the binary IDX format with big-endian headers
  - Normalizes pixel values from [0, 255] to [0.0, 1.0]

### Activation Functions
- **softmaxRows**: Applies softmax activation function row-wise
  - Used for multi-class classification output layers
  - Converts raw logits to probability distributions

## Usage

To use these utilities in your Swift code:

```swift
import MNISTCommon

// Initialize RNG
var rng = SimpleRng(seed: 42)
let randomValue = rng.nextFloat()

// Load MNIST data
let trainImages = readMnistImages(path: "./data/train-images.idx3-ubyte")
let trainLabels = readMnistLabels(path: "./data/train-labels.idx1-ubyte")

// Apply softmax
let probabilities = softmaxRows(logits)
```

## Migration

If you previously had these functions defined locally in standalone Swift files:
1. Remove the local implementations
2. Add `import MNISTCommon` at the top of your file
3. Compile your file as part of a Swift package that depends on MNISTCommon

See MIGRATION.md in the root directory for detailed migration instructions.

## Design

This module follows the same architectural patterns as the existing MNISTData module:
- Clear separation of concerns (RNG, data loading, activations)
- Comprehensive inline documentation
- Public APIs for library usage
- Efficient, straightforward implementations
