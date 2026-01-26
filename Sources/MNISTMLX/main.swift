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
//   - mlp:       Multi-Layer Perceptron (fastest, ~97% accuracy)
//   - cnn:       Convolutional Neural Network (best accuracy, ~98%)
//   - attention: Transformer-style attention (educational, ~95%)
//
// COMMAND-LINE OPTIONS:
//   --model <name>   Model to train: mlp, cnn, or attention (default: mlp)
//   --epochs <n>     Number of training epochs (default: 5)
//   --batch <n>      Batch size (default: 32)
//   --lr <f>         Learning rate (default: 0.01)
//   --data <path>    Path to MNIST data directory (default: ./data)
//   --help           Show usage information
//
// ============================================================================

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MNISTData

// =============================================================================
// MARK: - Command-Line Argument Parsing
// =============================================================================

/// Configuration parsed from command-line arguments
struct Config {
    var modelType: String = "mlp"
    var epochs: Int = 5
    var batchSize: Int = 32
    var learningRate: Float = 0.01
    var dataPath: String = "./data"
    var seed: UInt64 = 1
    
    /// Parses command-line arguments into configuration
    ///
    /// This is a simple hand-rolled parser. For production code,
    /// consider using Swift Argument Parser package.
    static func parse() -> Config {
        var config = Config()
        let args = CommandLine.arguments
        var i = 1
        
        while i < args.count {
            let arg = args[i]
            
            switch arg {
            case "--model", "-m":
                i += 1
                if i < args.count {
                    config.modelType = args[i].lowercased()
                }
                
            case "--epochs", "-e":
                i += 1
                if i < args.count, let val = Int(args[i]) {
                    config.epochs = val
                }
                
            case "--batch", "-b":
                i += 1
                if i < args.count, let val = Int(args[i]) {
                    config.batchSize = val
                }
                
            case "--lr", "-l":
                i += 1
                if i < args.count, let val = Float(args[i]) {
                    config.learningRate = val
                }
                
            case "--data", "-d":
                i += 1
                if i < args.count {
                    config.dataPath = args[i]
                }

            case "--seed", "-s":
                i += 1
                if i < args.count, let val = UInt64(args[i]) {
                    config.seed = val
                }

            case "--help", "-h":
                printUsage()
                exit(0)
                
            default:
                print("Unknown argument: \(arg)")
                printUsage()
                exit(1)
            }
            
            i += 1
        }
        
        return config
    }
}

/// Prints usage information
func printUsage() {
    print("""
    MNIST Neural Networks with MLX Swift
    =====================================
    
    USAGE:
      swift run MNISTMLX [OPTIONS]
    
    OPTIONS:
      --model, -m <name>    Model to train: mlp, cnn, or attention (default: mlp)
      --epochs, -e <n>      Number of training epochs (default: 5)
      --batch, -b <n>       Batch size (default: 32)
      --lr, -l <f>          Learning rate (default: 0.01)
      --data, -d <path>     Path to MNIST data directory (default: ./data)
      --seed, -s <n>        Random seed for reproducibility (default: 1)
      --help, -h            Show this help message
    
    EXAMPLES:
      swift run MNISTMLX --model cnn --epochs 3
      swift run MNISTMLX -m mlp -e 10 -b 64 -l 0.005
      swift run MNISTMLX --model attention --epochs 5
    
    MODELS:
      mlp        Multi-Layer Perceptron (784→512→10)
                 - Fastest training
                 - Good baseline (~97% accuracy)
                 
      cnn        Convolutional Neural Network
                 - Conv(3×3, 8 filters) → MaxPool → Linear
                 - Best accuracy (~98%)
                 
      attention  Transformer-style self-attention
                 - 4×4 patches → 49 tokens → attention → pooling
                 - Educational (demonstrates attention mechanism)
    """)
}

// =============================================================================
// MARK: - Training Functions
// =============================================================================

/// Trains an MLP model and evaluates it
func trainMLP(config: Config, trainImages: MLXArray, trainLabels: MLXArray, 
              testImages: MLXArray, testLabels: MLXArray) {
    print("\n🧠 Training MLP Model")
    print("   Architecture: 784 → 512 → 10")
    print("   Parameters:   ~407,000")
    print()
    
    // -------------------------------------------------------------------------
    // Initialize Model and Optimizer
    // -------------------------------------------------------------------------
    let model = MLPModel()
    
    // Evaluate model to initialize parameters
    // MLX uses lazy evaluation, so we need to "touch" the model first
    eval(model)
    
    // SGD (Stochastic Gradient Descent) optimizer
    // - learningRate: how big of a step to take in the gradient direction
    // - Lower = more stable but slower; higher = faster but may overshoot
    let optimizer = SGD(learningRate: config.learningRate)
    
    // -------------------------------------------------------------------------
    // Training Loop
    // -------------------------------------------------------------------------
    print("Epoch | Loss     | Time")
    print("------|----------|--------")
    
    for epoch in 1...config.epochs {
        let startTime = Date()
        
        // Train for one epoch
        let loss = trainMLPEpoch(
            model: model,
            optimizer: optimizer,
            trainImages: trainImages,
            trainLabels: trainLabels,
            batchSize: config.batchSize
        )
        
        let elapsed = Date().timeIntervalSince(startTime)
        print(String(format: "%5d | %.6f | %.2fs", epoch, loss, elapsed))
    }
    
    // -------------------------------------------------------------------------
    // Evaluation
    // -------------------------------------------------------------------------
    print("\n📊 Evaluating on test set...")
    let accuracy = mlpAccuracy(model: model, images: testImages, labels: testLabels)
    print(String(format: "   Test Accuracy: %.2f%%", accuracy * 100))
}

/// Trains a CNN model and evaluates it
func trainCNN(config: Config, trainImages: MLXArray, trainLabels: MLXArray,
              testImages: MLXArray, testLabels: MLXArray) {
    print("\n🧠 Training CNN Model")
    print("   Architecture: Conv(3×3, 8) → ReLU → MaxPool(2×2) → Linear(10)")
    print("   Parameters:   ~16,000")
    print()
    
    // -------------------------------------------------------------------------
    // Initialize Model and Optimizer
    // -------------------------------------------------------------------------
    let model = CNNModel()
    eval(model)
    
    let optimizer = SGD(learningRate: config.learningRate)
    
    // -------------------------------------------------------------------------
    // Training Loop
    // -------------------------------------------------------------------------
    print("Epoch | Loss     | Time")
    print("------|----------|--------")
    
    for epoch in 1...config.epochs {
        let startTime = Date()
        
        let loss = trainCNNEpoch(
            model: model,
            optimizer: optimizer,
            trainImages: trainImages,
            trainLabels: trainLabels,
            batchSize: config.batchSize
        )
        
        let elapsed = Date().timeIntervalSince(startTime)
        print(String(format: "%5d | %.6f | %.2fs", epoch, loss, elapsed))
    }
    
    // -------------------------------------------------------------------------
    // Evaluation
    // -------------------------------------------------------------------------
    print("\n📊 Evaluating on test set...")
    
    // Reshape for CNN (add channel dimension)
    let testImagesReshaped = testImages.reshaped([-1, 1, 28, 28])
    let accuracy = cnnAccuracy(model: model, images: testImagesReshaped, labels: testLabels)
    print(String(format: "   Test Accuracy: %.2f%%", accuracy * 100))
}

/// Trains an Attention model and evaluates it
func trainAttention(config: Config, trainImages: MLXArray, trainLabels: MLXArray,
                    testImages: MLXArray, testLabels: MLXArray) {
    print("\n🧠 Training Attention Model")
    print("   Architecture: Patches(4×4) → Attention → FFN → Pool → Linear")
    print("   Parameters:   ~8,000")
    print()
    
    // -------------------------------------------------------------------------
    // Initialize Model and Optimizer
    // -------------------------------------------------------------------------
    let model = AttentionModel()
    eval(model)
    
    let optimizer = SGD(learningRate: config.learningRate)
    
    // -------------------------------------------------------------------------
    // Training Loop
    // -------------------------------------------------------------------------
    print("Epoch | Loss     | Time")
    print("------|----------|--------")
    
    for epoch in 1...config.epochs {
        let startTime = Date()
        
        let loss = trainAttentionEpoch(
            model: model,
            optimizer: optimizer,
            trainImages: trainImages,
            trainLabels: trainLabels,
            batchSize: config.batchSize
        )
        
        let elapsed = Date().timeIntervalSince(startTime)
        print(String(format: "%5d | %.6f | %.2fs", epoch, loss, elapsed))
    }
    
    // -------------------------------------------------------------------------
    // Evaluation
    // -------------------------------------------------------------------------
    print("\n📊 Evaluating on test set...")
    let accuracy = attentionAccuracy(model: model, images: testImages, labels: testLabels)
    print(String(format: "   Test Accuracy: %.2f%%", accuracy * 100))
}

// =============================================================================
// MARK: - Main Entry Point
// =============================================================================

/// Main function
func main() {
    // =========================================================================
    // Parse Command-Line Arguments
    // =========================================================================
    let config = Config.parse()
    
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
    let testImages: MLXArray
    let testLabels: MLXArray
    
    do {
        (trainImages, trainLabels) = try loadMNIST(directory: config.dataPath, train: true)
        (testImages, testLabels) = try loadMNIST(directory: config.dataPath, train: false)
        
        print("   Training samples: \(trainImages.shape[0])")
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
                 testImages: testImages, testLabels: testLabels)
        
    case "cnn":
        trainCNN(config: config, trainImages: trainImages, trainLabels: trainLabels,
                 testImages: testImages, testLabels: testLabels)
        
    case "attention":
        trainAttention(config: config, trainImages: trainImages, trainLabels: trainLabels,
                       testImages: testImages, testLabels: testLabels)
        
    default:
        print("❌ Unknown model type: \(config.modelType)")
        print("   Available models: mlp, cnn, attention")
        exit(1)
    }
    
    print("\n✅ Done!")
}

// Run main
main()
