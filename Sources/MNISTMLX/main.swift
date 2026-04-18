// ============================================================================
// main.swift - CLI Entry Point for MNIST Neural Network Training
// ============================================================================
//
// This is the main executable that provides a command-line interface for
// training and testing different neural network architectures on MNIST.
//
// USAGE:
//   swift run MNISTMLX --model cnn --epochs 3 --batch 32 --lr 0.01
//
// AVAILABLE MODELS:
//   - mlp:         Multi-Layer Perceptron (fastest, ~97% accuracy)
//   - cnn:         Convolutional Neural Network (best accuracy, ~98%)
//   - resnet:      Residual Network with skip connections (~98% accuracy)
//   - attention:   Transformer-style attention (educational, ~95%)
//   - transformer: Full Transformer block with multi-head attention and FFN
//
// COMMAND-LINE OPTIONS:
//   --model <name>   Model to train: mlp, cnn, resnet, attention, or transformer (default: mlp)
//   --epochs <n>     Number of training epochs (default: 5)
//   --batch <n>      Batch size (default: 32)
//   --lr <f>         Learning rate (default: 0.01)
//   --data <path>    Path to MNIST data directory (default: ./data)
//   --help           Show usage information
//
// ============================================================================

import Foundation
import MLX
import MNISTData
import MNISTCommon

// =============================================================================
// MARK: - Main Entry Point
// =============================================================================

private func splitTrainingValidation(
    images: MLXArray,
    labels: MLXArray,
    validationFraction: Float = 0.1
) -> (trainImages: MLXArray, trainLabels: MLXArray, validationImages: MLXArray, validationLabels: MLXArray) {
    guard !images.shape.isEmpty else {
        ColoredPrint.error("❌ Training images must have a batch dimension")
        exit(1)
    }
    guard !labels.shape.isEmpty else {
        ColoredPrint.error("❌ Training labels must have a batch dimension")
        exit(1)
    }

    let sampleCount = images.shape[0]
    let labelCount = labels.shape[0]
    guard sampleCount == labelCount else {
        ColoredPrint.error("❌ Training images and labels count mismatch: \(sampleCount) images, \(labelCount) labels")
        exit(1)
    }
    guard sampleCount > 1 else {
        ColoredPrint.error("❌ Need at least 2 training samples to create a validation split")
        exit(1)
    }
    guard validationFraction.isFinite, (0.0...1.0).contains(validationFraction) else {
        ColoredPrint.error("❌ Validation fraction must be between 0.0 and 1.0")
        exit(1)
    }

    let validationCount = min(max(1, Int(Float(sampleCount) * validationFraction)), sampleCount - 1)
    let trainCount = sampleCount - validationCount

    return (
        trainImages: images[0..<trainCount],
        trainLabels: labels[0..<trainCount],
        validationImages: images[trainCount..<sampleCount],
        validationLabels: labels[trainCount..<sampleCount]
    )
}

/// Program entry point that parses command-line options, loads the MNIST dataset, and trains the selected model.
/// 
/// Parses CLI configuration, prints the chosen configuration, and adjusts the default learning rate for the attention
/// model when the user did not override it. Loads training and test MNIST data from the configured directory and
/// dispatches to the appropriate training routine for `mlp`, `cnn`, or `attention`. On failure to load data or when
/// an unknown model type is specified, the program prints an error and exits with code 1.
func main() {
    // =========================================================================
    // Parse Command-Line Arguments
    // =========================================================================
    var config = Config.parse()

    // Use optimal learning rate for attention model with increased capacity
    // (dModel=32, ffDim=64). If user explicitly set --lr, respect that.
    // Otherwise, use 0.005 which was found optimal during investigation.
    if config.modelType == "attention" && !config.learningRateProvided {
        config.learningRate = 0.005
    }

    print("╔═══════════════════════════════════════════════════════╗")
    print("║   MNIST Neural Networks with MLX Swift                ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    print("Configuration:")
    print("  Model:         \(config.modelType)")
    print("  Epochs:        \(config.epochs)")
    print("  Batch Size:    \(config.batchSize)")
    print("  Learning Rate: \(config.learningRate)")
    print("  Data Path:     \(config.dataPath)")
    print("  Seed:          \(config.seed)")
    print("  Compile:       \(config.useCompile ? "enabled" : "disabled")")

    // =========================================================================
    // Set Random Seed
    // =========================================================================
    MLX.seed(config.seed)

    // =========================================================================
    // Load MNIST Dataset
    // =========================================================================
    print("\n📁 Loading MNIST dataset...")
    
    let trainImages: MLXArray
    let trainLabels: MLXArray
    let validationImages: MLXArray
    let validationLabels: MLXArray
    let testImages: MLXArray
    let testLabels: MLXArray
    
    do {
        let loadedTrainImages: MLXArray
        let loadedTrainLabels: MLXArray
        (loadedTrainImages, loadedTrainLabels) = try loadMNIST(directory: config.dataPath, train: true)
        let split = splitTrainingValidation(images: loadedTrainImages, labels: loadedTrainLabels)
        trainImages = split.trainImages
        trainLabels = split.trainLabels
        validationImages = split.validationImages
        validationLabels = split.validationLabels
        (testImages, testLabels) = try loadMNIST(directory: config.dataPath, train: false)
        
        print("   Training samples: \(trainImages.shape[0])")
        print("   Validation samples: \(validationImages.shape[0])")
        print("   Test samples:     \(testImages.shape[0])")
    } catch {
        print("❌ Error loading MNIST data: \(error)")
        print()
        print("Make sure the MNIST files exist in '\(config.dataPath)/':")
        print("  - train-images.idx3-ubyte")
        print("  - train-labels.idx1-ubyte")
        print("  - t10k-images.idx3-ubyte")
        print("  - t10k-labels.idx1-ubyte")
        print()
        print("Download from: http://yann.lecun.com/exdb/mnist/")
        exit(1)
    }
    
    // =========================================================================
    // Train Selected Model
    // =========================================================================
    switch config.modelType {
    case "mlp":
        trainMLP(config: config, trainImages: trainImages, trainLabels: trainLabels,
                 validationImages: validationImages, validationLabels: validationLabels,
                 testImages: testImages, testLabels: testLabels)

    case "cnn":
        trainCNN(config: config, trainImages: trainImages, trainLabels: trainLabels,
                 validationImages: validationImages, validationLabels: validationLabels,
                 testImages: testImages, testLabels: testLabels)

    case "resnet":
        trainResNet(config: config, trainImages: trainImages, trainLabels: trainLabels,
                    validationImages: validationImages, validationLabels: validationLabels,
                    testImages: testImages, testLabels: testLabels)

    case "attention":
        trainAttention(config: config, trainImages: trainImages, trainLabels: trainLabels,
                       validationImages: validationImages, validationLabels: validationLabels,
                       testImages: testImages, testLabels: testLabels)

    case "transformer":
        trainTransformer(config: config, trainImages: trainImages, trainLabels: trainLabels,
                         validationImages: validationImages, validationLabels: validationLabels,
                         testImages: testImages, testLabels: testLabels)

    default:
        ColoredPrint.error("❌ Unknown model type: \(config.modelType)")
        print("   Available models: mlp, cnn, resnet, attention, transformer")
        exit(1)
    }

    ColoredPrint.success("\n✅ Done!")
}

// Run main
main()
