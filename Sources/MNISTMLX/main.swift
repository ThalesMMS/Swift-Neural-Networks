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
import MLXNN
import MLXOptimizers
import MNISTData
import MNISTCommon

// =============================================================================
// MARK: - Command-Line Argument Parsing
// =============================================================================

/// Configuration parsed from command-line arguments
struct Config {
    var modelType: String = "mlp"
    var epochs: Int = 5
    var batchSize: Int = 32
    var learningRate: Float = 0.01
    var learningRateProvided: Bool = false
    var dataPath: String = "./data"
    var seed: UInt64 = 1
    var useCompile: Bool = false
    var exportJson: Bool = false
    var checkpointInterval: Int? = nil
    var resumeFrom: String? = nil

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
                    config.learningRateProvided = true
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

            case "--compile", "-c":
                config.useCompile = true

            case "--export-json":
                config.exportJson = true

            case "--checkpoint-interval":
                i += 1
                if i < args.count, let val = Int(args[i]) {
                    config.checkpointInterval = val
                }

            case "--resume":
                i += 1
                if i < args.count {
                    config.resumeFrom = args[i]
                }

            case "--help", "-h":
                printUsage()
                exit(0)
                
            default:
                ColoredPrint.error("Unknown argument: \(arg)")
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
      --model, -m <name>    Model to train: mlp, cnn, resnet, attention, or transformer (default: mlp)
      --epochs, -e <n>      Number of training epochs (default: 5)
      --batch, -b <n>       Batch size (default: 32)
      --lr, -l <f>          Learning rate (default: 0.01)
      --data, -d <path>     Path to MNIST data directory (default: ./data)
      --seed, -s <n>        Random seed for reproducibility (default: 1)
      --compile, -c         Enable compiled training for faster execution
      --export-json         Export training results to JSON file
      --checkpoint-interval <n>  Save checkpoint every N epochs (default: disabled)
      --resume <path>       Resume training from checkpoint file
      --help, -h            Show this help message

    ENVIRONMENT:
      ANSI_COLORS=1         Enable colored terminal output
                            (errors=red, warnings=yellow, success=green, progress=cyan)

    EXAMPLES:
      swift run MNISTMLX --model cnn --epochs 3
      swift run MNISTMLX -m mlp -e 10 -b 64 -l 0.005
      ANSI_COLORS=1 swift run MNISTMLX --model attention --epochs 5

    MODELS:
      mlp        Multi-Layer Perceptron (784→512→10)
                 - Fastest training
                 - Good baseline (~97% accuracy)

      cnn        Convolutional Neural Network
                 - Conv(3×3, 8 filters) → MaxPool → Linear
                 - Best accuracy (~98%)

      resnet     Residual Network with skip connections
                 - ResidualBlock × 3 with skip connections
                 - Demonstrates how ResNet enables deeper networks (~98% accuracy)

      attention  Transformer-style self-attention
                 - 4×4 patches → 49 tokens → attention → pooling
                 - Educational (demonstrates attention mechanism)

      transformer Full Transformer block with multi-head attention
                 - Patches → Multi-head Self-Attention → LayerNorm → FFN → LayerNorm
                 - Demonstrates complete transformer architecture
    """)
}

// =============================================================================
// MARK: - Training Functions
// =============================================================================

/// Trains an MLP model and evaluates it
func trainMLP(config: Config, trainImages: MLXArray, trainLabels: MLXArray,
              testImages: MLXArray, testLabels: MLXArray) {
    ColoredPrint.progress("\n🧠 Training MLP Model")
    ColoredPrint.info("   Architecture: 784 → 512 → 10")
    ColoredPrint.info("   Parameters:   ~407,000")
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
    // Resume from Checkpoint (if specified)
    // -------------------------------------------------------------------------
    var startEpoch = 1

    if let resumePath = config.resumeFrom {
        ColoredPrint.progress("\n📂 Loading checkpoint from: \(resumePath)")

        do {
            // Load checkpoint from file
            let checkpoint = try Checkpoint.load(from: resumePath)

            // Validate model type matches
            guard checkpoint.validateModelType("mlp") else {
                ColoredPrint.error("❌ Model type mismatch: checkpoint is '\(checkpoint.modelType)', expected 'mlp'")
                exit(1)
            }

            // Restore model weights
            try loadCheckpoint(checkpoint: checkpoint, into: model)

            // Resume from next epoch
            startEpoch = checkpoint.epoch + 1

            ColoredPrint.success("✅ Checkpoint loaded successfully")
            ColoredPrint.info("   Resuming from epoch: \(startEpoch)")
            ColoredPrint.info("   Previous loss: \(String(format: "%.6f", checkpoint.metrics.trainLoss))")
            ColoredPrint.info("   Learning rate: \(checkpoint.optimizerState.learningRate)")
            print()
        } catch {
            ColoredPrint.error("❌ Failed to load checkpoint: \(error)")
            exit(1)
        }
    }

    // -------------------------------------------------------------------------
    // Training Loop
    // -------------------------------------------------------------------------
    var epochMetrics: [EpochMetrics] = []
    var bestValidationAccuracy: Float = 0.0
    var bestEpoch: Int = 0

    if config.useCompile {
        ColoredPrint.info("   Compilation: enabled ⚡")
    }

    ColoredPrint.info("Epoch | Loss     | Time    | Validation Accuracy")
    ColoredPrint.info("------|----------|---------|--------------------")

    for epoch in startEpoch...config.epochs {
        let startTime = Date()

        // Train for one epoch (compiled or uncompiled based on config)
        let loss: Float
        if config.useCompile {
            loss = trainMLPEpochCompiled(
                model: model,
                optimizer: optimizer,
                trainImages: trainImages,
                trainLabels: trainLabels,
                batchSize: config.batchSize
            )
        } else {
            loss = trainMLPEpoch(
                model: model,
                optimizer: optimizer,
                trainImages: trainImages,
                trainLabels: trainLabels,
                batchSize: config.batchSize
            )
        }

        let elapsed = Date().timeIntervalSince(startTime)

        // Evaluate validation accuracy after epoch
        let validationAccuracy = mlpAccuracy(model: model, images: testImages, labels: testLabels)

        ColoredPrint.progress(String(format: "%5d | %.6f | %.2fs | Validation: %.2f%%",
                                     epoch, loss, elapsed, validationAccuracy * 100))

        // Collect epoch metrics
        epochMetrics.append(EpochMetrics(epoch: epoch, loss: loss, duration: elapsed))

        // -------------------------------------------------------------------------
        // Track and Save Best Model
        // -------------------------------------------------------------------------
        if validationAccuracy > bestValidationAccuracy {
            bestValidationAccuracy = validationAccuracy
            bestEpoch = epoch

            // Create optimizer state
            let optimState = OptimizerState(learningRate: config.learningRate)

            // Create checkpoint metrics with validation accuracy
            let bestModelMetrics = CheckpointMetrics(
                trainLoss: loss,
                validationAccuracy: validationAccuracy
            )

            // Create hyperparameters
            let hyperparams = TrainingHyperparameters(
                epochs: config.epochs,
                batchSize: config.batchSize,
                learningRate: config.learningRate,
                seed: config.seed
            )

            // Save best model
            do {
                let savedPath = try saveBestModel(
                    model: model,
                    modelType: "mlp",
                    epoch: epoch,
                    validationAccuracy: validationAccuracy,
                    optimizerState: optimState,
                    hyperparameters: hyperparams,
                    metrics: bestModelMetrics
                )
                ColoredPrint.success("🌟 New best model saved: \(savedPath) (Validation: \(String(format: "%.2f%%", validationAccuracy * 100)))")
            } catch {
                ColoredPrint.error("Failed to save best model: \(error)")
            }
        }

        // -------------------------------------------------------------------------
        // Save Checkpoint (if checkpoint interval is set)
        // -------------------------------------------------------------------------
        if let checkpointInterval = config.checkpointInterval, epoch % checkpointInterval == 0 {
            let checkpointsDir = "./checkpoints"
            let filename = "checkpoint_mlp_epoch_\(epoch).json"
            let filePath = "\(checkpointsDir)/\(filename)"

            // Create optimizer state
            let optimState = OptimizerState(learningRate: config.learningRate)

            // Create checkpoint metrics
            let checkpointMetrics = CheckpointMetrics(trainLoss: loss)

            // Create hyperparameters
            let hyperparams = TrainingHyperparameters(
                epochs: config.epochs,
                batchSize: config.batchSize,
                learningRate: config.learningRate,
                seed: config.seed
            )

            // Save checkpoint
            do {
                try saveCheckpoint(
                    model: model,
                    modelType: "mlp",
                    epoch: epoch,
                    optimizerState: optimState,
                    hyperparameters: hyperparams,
                    metrics: checkpointMetrics,
                    filePath: filePath
                )
                ColoredPrint.success("💾 Checkpoint saved: \(filePath)")
            } catch {
                ColoredPrint.error("Failed to save checkpoint: \(error)")
            }
        }
    }

    // -------------------------------------------------------------------------
    // Evaluation
    // -------------------------------------------------------------------------
    ColoredPrint.progress("\n📊 Evaluating on test set...")
    let accuracy = mlpAccuracy(model: model, images: testImages, labels: testLabels)
    ColoredPrint.info(String(format: "   Test Accuracy: %.2f%%", accuracy * 100))

    // -------------------------------------------------------------------------
    // Training Summary
    // -------------------------------------------------------------------------
    let hyperparameters = TrainingHyperparameters(
        epochs: config.epochs,
        batchSize: config.batchSize,
        learningRate: config.learningRate,
        seed: config.seed
    )

    let benchmarkComparison = BenchmarkComparison(
        expectedAccuracy: 0.97,
        actualAccuracy: accuracy
    )

    let summary = TrainingSummary(
        modelType: "mlp",
        hyperparameters: hyperparameters,
        epochMetrics: epochMetrics,
        finalAccuracy: accuracy,
        benchmarkComparison: benchmarkComparison,
        bestValidationAccuracy: bestEpoch > 0 ? bestValidationAccuracy : nil,
        bestEpoch: bestEpoch > 0 ? bestEpoch : nil
    )

    summary.printSummary()
    summary.printBenchmarkComparison()

    // -------------------------------------------------------------------------
    // JSON Export (if requested)
    // -------------------------------------------------------------------------
    if config.exportJson {
        let fileManager = FileManager.default
        let logsDir = "./logs"

        // Create logs directory if it doesn't exist
        if !fileManager.fileExists(atPath: logsDir) {
            do {
                try fileManager.createDirectory(atPath: logsDir, withIntermediateDirectories: true)
            } catch {
                ColoredPrint.error("Failed to create logs directory: \(error)")
                return
            }
        }

        // Generate timestamped filename
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = dateFormatter.string(from: Date())
        let filename = "training_summary_\(config.modelType)_\(timestamp).json"
        let filePath = "\(logsDir)/\(filename)"

        // Export to JSON
        do {
            try summary.exportToJSON(filePath: filePath)
            ColoredPrint.success("📄 Training summary exported to: \(filePath)")
        } catch {
            ColoredPrint.error("Failed to export JSON: \(error)")
        }
    }
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
    // Resume from Checkpoint (if specified)
    // -------------------------------------------------------------------------
    var startEpoch = 1

    if let resumePath = config.resumeFrom {
        ColoredPrint.progress("\n📂 Loading checkpoint from: \(resumePath)")

        do {
            // Load checkpoint from file
            let checkpoint = try Checkpoint.load(from: resumePath)

            // Validate model type matches
            guard checkpoint.validateModelType("cnn") else {
                ColoredPrint.error("❌ Model type mismatch: checkpoint is '\(checkpoint.modelType)', expected 'cnn'")
                exit(1)
            }

            // Restore model weights
            try loadCheckpoint(checkpoint: checkpoint, into: model)

            // Resume from next epoch
            startEpoch = checkpoint.epoch + 1

            ColoredPrint.success("✅ Checkpoint loaded successfully")
            ColoredPrint.info("   Resuming from epoch: \(startEpoch)")
            ColoredPrint.info("   Previous loss: \(String(format: "%.6f", checkpoint.metrics.trainLoss))")
            ColoredPrint.info("   Learning rate: \(checkpoint.optimizerState.learningRate)")
            print()
        } catch {
            ColoredPrint.error("❌ Failed to load checkpoint: \(error)")
            exit(1)
        }
    }

    // -------------------------------------------------------------------------
    // Training Loop
    // -------------------------------------------------------------------------
    var epochMetrics: [EpochMetrics] = []
    var bestValidationAccuracy: Float = 0.0
    var bestEpoch: Int = 0

    if config.useCompile {
        ColoredPrint.info("   Compilation: enabled ⚡")
    }

    ColoredPrint.info("Epoch | Loss     | Time    | Validation Accuracy")
    ColoredPrint.info("------|----------|---------|--------------------")

    for epoch in startEpoch...config.epochs {
        let startTime = Date()

        // Train for one epoch (compiled or uncompiled based on config)
        let loss: Float
        if config.useCompile {
            loss = trainCNNEpochCompiled(
                model: model,
                optimizer: optimizer,
                trainImages: trainImages,
                trainLabels: trainLabels,
                batchSize: config.batchSize
            )
        } else {
            loss = trainCNNEpoch(
                model: model,
                optimizer: optimizer,
                trainImages: trainImages,
                trainLabels: trainLabels,
                batchSize: config.batchSize
            )
        }

        let elapsed = Date().timeIntervalSince(startTime)

        // Evaluate validation accuracy after epoch
        let testImagesReshaped = testImages.reshaped([-1, 1, 28, 28])
        let validationAccuracy = cnnAccuracy(model: model, images: testImagesReshaped, labels: testLabels)

        ColoredPrint.progress(String(format: "%5d | %.6f | %.2fs | Validation: %.2f%%",
                                     epoch, loss, elapsed, validationAccuracy * 100))

        // Collect epoch metrics
        epochMetrics.append(EpochMetrics(epoch: epoch, loss: loss, duration: elapsed))

        // -------------------------------------------------------------------------
        // Track and Save Best Model
        // -------------------------------------------------------------------------
        if validationAccuracy > bestValidationAccuracy {
            bestValidationAccuracy = validationAccuracy
            bestEpoch = epoch

            // Create optimizer state
            let optimState = OptimizerState(learningRate: config.learningRate)

            // Create checkpoint metrics with validation accuracy
            let bestModelMetrics = CheckpointMetrics(
                trainLoss: loss,
                validationAccuracy: validationAccuracy
            )

            // Create hyperparameters
            let hyperparams = TrainingHyperparameters(
                epochs: config.epochs,
                batchSize: config.batchSize,
                learningRate: config.learningRate,
                seed: config.seed
            )

            // Save best model
            do {
                let savedPath = try saveBestModel(
                    model: model,
                    modelType: "cnn",
                    epoch: epoch,
                    validationAccuracy: validationAccuracy,
                    optimizerState: optimState,
                    hyperparameters: hyperparams,
                    metrics: bestModelMetrics
                )
                ColoredPrint.success("🌟 New best model saved: \(savedPath) (Validation: \(String(format: "%.2f%%", validationAccuracy * 100)))")
            } catch {
                ColoredPrint.error("Failed to save best model: \(error)")
            }
        }

        // -------------------------------------------------------------------------
        // Save Checkpoint (if checkpoint interval is set)
        // -------------------------------------------------------------------------
        if let checkpointInterval = config.checkpointInterval, epoch % checkpointInterval == 0 {
            let checkpointsDir = "./checkpoints"
            let filename = "checkpoint_cnn_epoch_\(epoch).json"
            let filePath = "\(checkpointsDir)/\(filename)"

            // Create optimizer state
            let optimState = OptimizerState(learningRate: config.learningRate)

            // Create checkpoint metrics
            let checkpointMetrics = CheckpointMetrics(trainLoss: loss)

            // Create hyperparameters
            let hyperparams = TrainingHyperparameters(
                epochs: config.epochs,
                batchSize: config.batchSize,
                learningRate: config.learningRate,
                seed: config.seed
            )

            // Save checkpoint
            do {
                try saveCheckpoint(
                    model: model,
                    modelType: "cnn",
                    epoch: epoch,
                    optimizerState: optimState,
                    hyperparameters: hyperparams,
                    metrics: checkpointMetrics,
                    filePath: filePath
                )
                ColoredPrint.success("💾 Checkpoint saved: \(filePath)")
            } catch {
                ColoredPrint.error("Failed to save checkpoint: \(error)")
            }
        }
    }

    // -------------------------------------------------------------------------
    // Evaluation
    // -------------------------------------------------------------------------
    ColoredPrint.progress("\n📊 Evaluating on test set...")

    // Reshape for CNN (add channel dimension)
    let testImagesReshaped = testImages.reshaped([-1, 1, 28, 28])
    let accuracy = cnnAccuracy(model: model, images: testImagesReshaped, labels: testLabels)
    ColoredPrint.info(String(format: "   Test Accuracy: %.2f%%", accuracy * 100))

    // -------------------------------------------------------------------------
    // Training Summary
    // -------------------------------------------------------------------------
    let hyperparameters = TrainingHyperparameters(
        epochs: config.epochs,
        batchSize: config.batchSize,
        learningRate: config.learningRate,
        seed: config.seed
    )

    let benchmarkComparison = BenchmarkComparison(
        expectedAccuracy: 0.98,
        actualAccuracy: accuracy
    )

    let summary = TrainingSummary(
        modelType: "cnn",
        hyperparameters: hyperparameters,
        epochMetrics: epochMetrics,
        finalAccuracy: accuracy,
        benchmarkComparison: benchmarkComparison,
        bestValidationAccuracy: bestEpoch > 0 ? bestValidationAccuracy : nil,
        bestEpoch: bestEpoch > 0 ? bestEpoch : nil
    )

    summary.printSummary()
    summary.printBenchmarkComparison()

    // -------------------------------------------------------------------------
    // JSON Export (if requested)
    // -------------------------------------------------------------------------
    if config.exportJson {
        let fileManager = FileManager.default
        let logsDir = "./logs"

        // Create logs directory if it doesn't exist
        if !fileManager.fileExists(atPath: logsDir) {
            do {
                try fileManager.createDirectory(atPath: logsDir, withIntermediateDirectories: true)
            } catch {
                ColoredPrint.error("Failed to create logs directory: \(error)")
                return
            }
        }

        // Generate timestamped filename
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = dateFormatter.string(from: Date())
        let filename = "training_summary_\(config.modelType)_\(timestamp).json"
        let filePath = "\(logsDir)/\(filename)"

        // Export to JSON
        do {
            try summary.exportToJSON(filePath: filePath)
            ColoredPrint.success("📄 Training summary exported to: \(filePath)")
        } catch {
            ColoredPrint.error("Failed to export JSON: \(error)")
        }
    }
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
    // Resume from Checkpoint (if specified)
    // -------------------------------------------------------------------------
    var startEpoch = 1

    if let resumePath = config.resumeFrom {
        ColoredPrint.progress("\n📂 Loading checkpoint from: \(resumePath)")

        do {
            // Load checkpoint from file
            let checkpoint = try Checkpoint.load(from: resumePath)

            // Validate model type matches
            guard checkpoint.validateModelType("attention") else {
                ColoredPrint.error("❌ Model type mismatch: checkpoint is '\(checkpoint.modelType)', expected 'attention'")
                exit(1)
            }

            // Restore model weights
            try loadCheckpoint(checkpoint: checkpoint, into: model)

            // Resume from next epoch
            startEpoch = checkpoint.epoch + 1

            ColoredPrint.success("✅ Checkpoint loaded successfully")
            ColoredPrint.info("   Resuming from epoch: \(startEpoch)")
            ColoredPrint.info("   Previous loss: \(String(format: "%.6f", checkpoint.metrics.trainLoss))")
            ColoredPrint.info("   Learning rate: \(checkpoint.optimizerState.learningRate)")
            print()
        } catch {
            ColoredPrint.error("❌ Failed to load checkpoint: \(error)")
            exit(1)
        }
    }

    // -------------------------------------------------------------------------
    // Training Loop
    // -------------------------------------------------------------------------
    var epochMetrics: [EpochMetrics] = []
    var bestValidationAccuracy: Float = 0.0
    var bestEpoch: Int = 0

    if config.useCompile {
        ColoredPrint.info("   Compilation: enabled ⚡")
    }

    ColoredPrint.info("Epoch | Loss     | Time    | Validation Accuracy")
    ColoredPrint.info("------|----------|---------|--------------------")

    for epoch in startEpoch...config.epochs {
        let startTime = Date()

        // Train for one epoch (compiled or uncompiled based on config)
        let loss: Float
        if config.useCompile {
            loss = trainAttentionEpochCompiled(
                model: model,
                optimizer: optimizer,
                trainImages: trainImages,
                trainLabels: trainLabels,
                batchSize: config.batchSize
            )
        } else {
            loss = trainAttentionEpoch(
                model: model,
                optimizer: optimizer,
                trainImages: trainImages,
                trainLabels: trainLabels,
                batchSize: config.batchSize
            )
        }

        let elapsed = Date().timeIntervalSince(startTime)

        // Evaluate validation accuracy after epoch
        let validationAccuracy = attentionAccuracy(model: model, images: testImages, labels: testLabels)

        ColoredPrint.progress(String(format: "%5d | %.6f | %.2fs | Validation: %.2f%%",
                                     epoch, loss, elapsed, validationAccuracy * 100))

        // Collect epoch metrics
        epochMetrics.append(EpochMetrics(epoch: epoch, loss: loss, duration: elapsed))

        // -------------------------------------------------------------------------
        // Track and Save Best Model
        // -------------------------------------------------------------------------
        if validationAccuracy > bestValidationAccuracy {
            bestValidationAccuracy = validationAccuracy
            bestEpoch = epoch

            // Create optimizer state
            let optimState = OptimizerState(learningRate: config.learningRate)

            // Create checkpoint metrics with validation accuracy
            let bestModelMetrics = CheckpointMetrics(
                trainLoss: loss,
                validationAccuracy: validationAccuracy
            )

            // Create hyperparameters
            let hyperparams = TrainingHyperparameters(
                epochs: config.epochs,
                batchSize: config.batchSize,
                learningRate: config.learningRate,
                seed: config.seed
            )

            // Save best model
            do {
                let savedPath = try saveBestModel(
                    model: model,
                    modelType: "attention",
                    epoch: epoch,
                    validationAccuracy: validationAccuracy,
                    optimizerState: optimState,
                    hyperparameters: hyperparams,
                    metrics: bestModelMetrics
                )
                ColoredPrint.success("🌟 New best model saved: \(savedPath) (Validation: \(String(format: "%.2f%%", validationAccuracy * 100)))")
            } catch {
                ColoredPrint.error("Failed to save best model: \(error)")
            }
        }

        // -------------------------------------------------------------------------
        // Save Checkpoint (if checkpoint interval is set)
        // -------------------------------------------------------------------------
        if let checkpointInterval = config.checkpointInterval, epoch % checkpointInterval == 0 {
            let checkpointsDir = "./checkpoints"
            let filename = "checkpoint_attention_epoch_\(epoch).json"
            let filePath = "\(checkpointsDir)/\(filename)"

            // Create optimizer state
            let optimState = OptimizerState(learningRate: config.learningRate)

            // Create checkpoint metrics
            let checkpointMetrics = CheckpointMetrics(trainLoss: loss)

            // Create hyperparameters
            let hyperparams = TrainingHyperparameters(
                epochs: config.epochs,
                batchSize: config.batchSize,
                learningRate: config.learningRate,
                seed: config.seed
            )

            // Save checkpoint
            do {
                try saveCheckpoint(
                    model: model,
                    modelType: "attention",
                    epoch: epoch,
                    optimizerState: optimState,
                    hyperparameters: hyperparams,
                    metrics: checkpointMetrics,
                    filePath: filePath
                )
                ColoredPrint.success("💾 Checkpoint saved: \(filePath)")
            } catch {
                ColoredPrint.error("Failed to save checkpoint: \(error)")
            }
        }
    }

    // -------------------------------------------------------------------------
    // Evaluation
    // -------------------------------------------------------------------------
    ColoredPrint.progress("\n📊 Evaluating on test set...")
    let accuracy = attentionAccuracy(model: model, images: testImages, labels: testLabels)
    ColoredPrint.info(String(format: "   Test Accuracy: %.2f%%", accuracy * 100))

    // -------------------------------------------------------------------------
    // Training Summary
    // -------------------------------------------------------------------------
    let hyperparameters = TrainingHyperparameters(
        epochs: config.epochs,
        batchSize: config.batchSize,
        learningRate: config.learningRate,
        seed: config.seed
    )

    let benchmarkComparison = BenchmarkComparison(
        expectedAccuracy: 0.90,
        actualAccuracy: accuracy
    )

    let summary = TrainingSummary(
        modelType: "attention",
        hyperparameters: hyperparameters,
        epochMetrics: epochMetrics,
        finalAccuracy: accuracy,
        benchmarkComparison: benchmarkComparison,
        bestValidationAccuracy: bestEpoch > 0 ? bestValidationAccuracy : nil,
        bestEpoch: bestEpoch > 0 ? bestEpoch : nil
    )

    summary.printSummary()
    summary.printBenchmarkComparison()

    // -------------------------------------------------------------------------
    // JSON Export (if requested)
    // -------------------------------------------------------------------------
    if config.exportJson {
        let fileManager = FileManager.default
        let logsDir = "./logs"

        // Create logs directory if it doesn't exist
        if !fileManager.fileExists(atPath: logsDir) {
            do {
                try fileManager.createDirectory(atPath: logsDir, withIntermediateDirectories: true)
            } catch {
                ColoredPrint.error("Failed to create logs directory: \(error)")
                return
            }
        }

        // Generate timestamped filename
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = dateFormatter.string(from: Date())
        let filename = "training_summary_\(config.modelType)_\(timestamp).json"
        let filePath = "\(logsDir)/\(filename)"

        // Export to JSON
        do {
            try summary.exportToJSON(filePath: filePath)
            ColoredPrint.success("📄 Training summary exported to: \(filePath)")
        } catch {
            ColoredPrint.error("Failed to export JSON: \(error)")
        }
    }
}

/// Trains a Transformer model and evaluates it
func trainTransformer(config: Config, trainImages: MLXArray, trainLabels: MLXArray,
                      testImages: MLXArray, testLabels: MLXArray) {
    print("\n🧠 Training Transformer Model")
    print("   Architecture: Patches → Multi-head Self-Attention → LayerNorm → FFN → LayerNorm")
    print("   Parameters:   ~15,000")
    print()

    // -------------------------------------------------------------------------
    // Initialize Model and Optimizer
    // -------------------------------------------------------------------------
    let model = TransformerModel()
    eval(model)

    let optimizer = SGD(learningRate: config.learningRate)

    // -------------------------------------------------------------------------
    // Resume from Checkpoint (if specified)
    // -------------------------------------------------------------------------
    var startEpoch = 1

    if let resumePath = config.resumeFrom {
        ColoredPrint.progress("\n📂 Loading checkpoint from: \(resumePath)")

        do {
            // Load checkpoint from file
            let checkpoint = try Checkpoint.load(from: resumePath)

            // Validate model type matches
            guard checkpoint.validateModelType("transformer") else {
                ColoredPrint.error("❌ Model type mismatch: checkpoint is '\(checkpoint.modelType)', expected 'transformer'")
                exit(1)
            }

            // Restore model weights
            try loadCheckpoint(checkpoint: checkpoint, into: model)

            // Resume from next epoch
            startEpoch = checkpoint.epoch + 1

            ColoredPrint.success("✅ Checkpoint loaded successfully")
            ColoredPrint.info("   Resuming from epoch: \(startEpoch)")
            ColoredPrint.info("   Previous loss: \(String(format: "%.6f", checkpoint.metrics.trainLoss))")
            ColoredPrint.info("   Learning rate: \(checkpoint.optimizerState.learningRate)")
            print()
        } catch {
            ColoredPrint.error("❌ Failed to load checkpoint: \(error)")
            exit(1)
        }
    }

    // -------------------------------------------------------------------------
    // Training Loop
    // -------------------------------------------------------------------------
    var epochMetrics: [EpochMetrics] = []
    var bestValidationAccuracy: Float = 0.0
    var bestEpoch: Int = 0

    if config.useCompile {
        ColoredPrint.info("   Compilation: enabled ⚡")
    }

    ColoredPrint.info("Epoch | Loss     | Time    | Validation Accuracy")
    ColoredPrint.info("------|----------|---------|--------------------")

    for epoch in startEpoch...config.epochs {
        let startTime = Date()

        // Train for one epoch
        let loss = trainTransformerEpoch(
            model: model,
            optimizer: optimizer,
            trainImages: trainImages,
            trainLabels: trainLabels,
            batchSize: config.batchSize
        )

        let elapsed = Date().timeIntervalSince(startTime)

        // Evaluate validation accuracy after epoch
        let validationAccuracy = transformerAccuracy(model: model, images: testImages, labels: testLabels)

        ColoredPrint.progress(String(format: "%5d | %.6f | %.2fs | Validation: %.2f%%",
                                     epoch, loss, elapsed, validationAccuracy * 100))

        // Collect epoch metrics
        epochMetrics.append(EpochMetrics(epoch: epoch, loss: loss, duration: elapsed))

        // -------------------------------------------------------------------------
        // Track and Save Best Model
        // -------------------------------------------------------------------------
        if validationAccuracy > bestValidationAccuracy {
            bestValidationAccuracy = validationAccuracy
            bestEpoch = epoch

            // Create optimizer state
            let optimState = OptimizerState(learningRate: config.learningRate)

            // Create checkpoint metrics with validation accuracy
            let bestModelMetrics = CheckpointMetrics(
                trainLoss: loss,
                validationAccuracy: validationAccuracy
            )

            // Create hyperparameters
            let hyperparams = TrainingHyperparameters(
                epochs: config.epochs,
                batchSize: config.batchSize,
                learningRate: config.learningRate,
                seed: config.seed
            )

            // Save best model
            do {
                let savedPath = try saveBestModel(
                    model: model,
                    modelType: "transformer",
                    epoch: epoch,
                    validationAccuracy: validationAccuracy,
                    optimizerState: optimState,
                    hyperparameters: hyperparams,
                    metrics: bestModelMetrics
                )
                ColoredPrint.success("🌟 New best model saved: \(savedPath) (Validation: \(String(format: "%.2f%%", validationAccuracy * 100)))")
            } catch {
                ColoredPrint.error("Failed to save best model: \(error)")
            }
        }

        // -------------------------------------------------------------------------
        // Save Checkpoint (if checkpoint interval is set)
        // -------------------------------------------------------------------------
        if let checkpointInterval = config.checkpointInterval, epoch % checkpointInterval == 0 {
            let checkpointsDir = "./checkpoints"
            let filename = "checkpoint_transformer_epoch_\(epoch).json"
            let filePath = "\(checkpointsDir)/\(filename)"

            // Create optimizer state
            let optimState = OptimizerState(learningRate: config.learningRate)

            // Create checkpoint metrics
            let checkpointMetrics = CheckpointMetrics(trainLoss: loss)

            // Create hyperparameters
            let hyperparams = TrainingHyperparameters(
                epochs: config.epochs,
                batchSize: config.batchSize,
                learningRate: config.learningRate,
                seed: config.seed
            )

            // Save checkpoint
            do {
                try saveCheckpoint(
                    model: model,
                    modelType: "transformer",
                    epoch: epoch,
                    optimizerState: optimState,
                    hyperparameters: hyperparams,
                    metrics: checkpointMetrics,
                    filePath: filePath
                )
                ColoredPrint.success("💾 Checkpoint saved: \(filePath)")
            } catch {
                ColoredPrint.error("Failed to save checkpoint: \(error)")
            }
        }
    }

    // -------------------------------------------------------------------------
    // Evaluation
    // -------------------------------------------------------------------------
    ColoredPrint.progress("\n📊 Evaluating on test set...")
    let accuracy = transformerAccuracy(model: model, images: testImages, labels: testLabels)
    ColoredPrint.info(String(format: "   Test Accuracy: %.2f%%", accuracy * 100))

    // -------------------------------------------------------------------------
    // Training Summary
    // -------------------------------------------------------------------------
    let hyperparameters = TrainingHyperparameters(
        epochs: config.epochs,
        batchSize: config.batchSize,
        learningRate: config.learningRate,
        seed: config.seed
    )

    let benchmarkComparison = BenchmarkComparison(
        expectedAccuracy: 0.92,
        actualAccuracy: accuracy
    )

    let summary = TrainingSummary(
        modelType: "transformer",
        hyperparameters: hyperparameters,
        epochMetrics: epochMetrics,
        finalAccuracy: accuracy,
        benchmarkComparison: benchmarkComparison,
        bestValidationAccuracy: bestEpoch > 0 ? bestValidationAccuracy : nil,
        bestEpoch: bestEpoch > 0 ? bestEpoch : nil
    )

    summary.printSummary()
    summary.printBenchmarkComparison()

    // -------------------------------------------------------------------------
    // JSON Export (if requested)
    // -------------------------------------------------------------------------
    if config.exportJson {
        let fileManager = FileManager.default
        let logsDir = "./logs"

        // Create logs directory if it doesn't exist
        if !fileManager.fileExists(atPath: logsDir) {
            do {
                try fileManager.createDirectory(atPath: logsDir, withIntermediateDirectories: true)
            } catch {
                ColoredPrint.error("Failed to create logs directory: \(error)")
                return
            }
        }

        // Generate timestamped filename
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = dateFormatter.string(from: Date())
        let filename = "training_summary_\(config.modelType)_\(timestamp).json"
        let filePath = "\(logsDir)/\(filename)"

        // Export to JSON
        do {
            try summary.exportToJSON(filePath: filePath)
            ColoredPrint.success("📄 Training summary exported to: \(filePath)")
        } catch {
            ColoredPrint.error("Failed to export JSON: \(error)")
        }
    }
}

/// Trains a ResNet model and evaluates it
func trainResNet(config: Config, trainImages: MLXArray, trainLabels: MLXArray,
                 testImages: MLXArray, testLabels: MLXArray) {
    print("\n🧠 Training ResNet Model")
    print("   Architecture: Conv → ResidualBlock × 3 → GlobalAvgPool → Linear")
    print("   Parameters:   ~10,000")
    print()

    // -------------------------------------------------------------------------
    // Initialize Model and Optimizer
    // -------------------------------------------------------------------------
    let model = ResNetModel()
    eval(model)

    let optimizer = SGD(learningRate: config.learningRate)

    // -------------------------------------------------------------------------
    // Resume from Checkpoint (if specified)
    // -------------------------------------------------------------------------
    var startEpoch = 1

    if let resumePath = config.resumeFrom {
        ColoredPrint.progress("\n📂 Loading checkpoint from: \(resumePath)")

        do {
            // Load checkpoint from file
            let checkpoint = try Checkpoint.load(from: resumePath)

            // Validate model type matches
            guard checkpoint.validateModelType("resnet") else {
                ColoredPrint.error("❌ Model type mismatch: checkpoint is '\(checkpoint.modelType)', expected 'resnet'")
                exit(1)
            }

            // Restore model weights
            try loadCheckpoint(checkpoint: checkpoint, into: model)

            // Resume from next epoch
            startEpoch = checkpoint.epoch + 1

            ColoredPrint.success("✅ Checkpoint loaded successfully")
            ColoredPrint.info("   Resuming from epoch: \(startEpoch)")
            ColoredPrint.info("   Previous loss: \(String(format: "%.6f", checkpoint.metrics.trainLoss))")
            ColoredPrint.info("   Learning rate: \(checkpoint.optimizerState.learningRate)")
            print()
        } catch {
            ColoredPrint.error("❌ Failed to load checkpoint: \(error)")
            exit(1)
        }
    }

    // -------------------------------------------------------------------------
    // Training Loop
    // -------------------------------------------------------------------------
    var epochMetrics: [EpochMetrics] = []
    var bestValidationAccuracy: Float = 0.0
    var bestEpoch: Int = 0

    if config.useCompile {
        ColoredPrint.info("   Compilation: enabled ⚡")
    }

    ColoredPrint.info("Epoch | Loss     | Time    | Validation Accuracy")
    ColoredPrint.info("------|----------|---------|--------------------")

    for epoch in startEpoch...config.epochs {
        let startTime = Date()

        // Train for one epoch (compiled or uncompiled based on config)
        let loss: Float
        if config.useCompile {
            loss = trainResNetEpochCompiled(
                model: model,
                optimizer: optimizer,
                trainImages: trainImages,
                trainLabels: trainLabels,
                batchSize: config.batchSize
            )
        } else {
            loss = trainResNetEpoch(
                model: model,
                optimizer: optimizer,
                trainImages: trainImages,
                trainLabels: trainLabels,
                batchSize: config.batchSize
            )
        }

        let elapsed = Date().timeIntervalSince(startTime)

        // Evaluate validation accuracy after epoch
        let testImagesReshaped = testImages.reshaped([-1, 1, 28, 28])
        let validationAccuracy = resnetAccuracy(model: model, images: testImagesReshaped, labels: testLabels)

        ColoredPrint.progress(String(format: "%5d | %.6f | %.2fs | Validation: %.2f%%",
                                     epoch, loss, elapsed, validationAccuracy * 100))

        // Collect epoch metrics
        epochMetrics.append(EpochMetrics(epoch: epoch, loss: loss, duration: elapsed))

        // -------------------------------------------------------------------------
        // Track and Save Best Model
        // -------------------------------------------------------------------------
        if validationAccuracy > bestValidationAccuracy {
            bestValidationAccuracy = validationAccuracy
            bestEpoch = epoch

            // Create optimizer state
            let optimState = OptimizerState(learningRate: config.learningRate)

            // Create checkpoint metrics with validation accuracy
            let bestModelMetrics = CheckpointMetrics(
                trainLoss: loss,
                validationAccuracy: validationAccuracy
            )

            // Create hyperparameters
            let hyperparams = TrainingHyperparameters(
                epochs: config.epochs,
                batchSize: config.batchSize,
                learningRate: config.learningRate,
                seed: config.seed
            )

            // Save best model
            do {
                let savedPath = try saveBestModel(
                    model: model,
                    modelType: "resnet",
                    epoch: epoch,
                    validationAccuracy: validationAccuracy,
                    optimizerState: optimState,
                    hyperparameters: hyperparams,
                    metrics: bestModelMetrics
                )
                ColoredPrint.success("🌟 New best model saved: \(savedPath) (Validation: \(String(format: "%.2f%%", validationAccuracy * 100)))")
            } catch {
                ColoredPrint.error("Failed to save best model: \(error)")
            }
        }

        // -------------------------------------------------------------------------
        // Save Checkpoint (if checkpoint interval is set)
        // -------------------------------------------------------------------------
        if let checkpointInterval = config.checkpointInterval, epoch % checkpointInterval == 0 {
            let checkpointsDir = "./checkpoints"
            let filename = "checkpoint_resnet_epoch_\(epoch).json"
            let filePath = "\(checkpointsDir)/\(filename)"

            // Create optimizer state
            let optimState = OptimizerState(learningRate: config.learningRate)

            // Create checkpoint metrics
            let checkpointMetrics = CheckpointMetrics(trainLoss: loss)

            // Create hyperparameters
            let hyperparams = TrainingHyperparameters(
                epochs: config.epochs,
                batchSize: config.batchSize,
                learningRate: config.learningRate,
                seed: config.seed
            )

            // Save checkpoint
            do {
                try saveCheckpoint(
                    model: model,
                    modelType: "resnet",
                    epoch: epoch,
                    optimizerState: optimState,
                    hyperparameters: hyperparams,
                    metrics: checkpointMetrics,
                    filePath: filePath
                )
                ColoredPrint.success("💾 Checkpoint saved: \(filePath)")
            } catch {
                ColoredPrint.error("Failed to save checkpoint: \(error)")
            }
        }
    }

    // -------------------------------------------------------------------------
    // Evaluation
    // -------------------------------------------------------------------------
    ColoredPrint.progress("\n📊 Evaluating on test set...")

    // Reshape for ResNet (add channel dimension)
    let testImagesReshaped = testImages.reshaped([-1, 1, 28, 28])
    let accuracy = resnetAccuracy(model: model, images: testImagesReshaped, labels: testLabels)
    ColoredPrint.info(String(format: "   Test Accuracy: %.2f%%", accuracy * 100))

    // -------------------------------------------------------------------------
    // Training Summary
    // -------------------------------------------------------------------------
    let hyperparameters = TrainingHyperparameters(
        epochs: config.epochs,
        batchSize: config.batchSize,
        learningRate: config.learningRate,
        seed: config.seed
    )

    let benchmarkComparison = BenchmarkComparison(
        expectedAccuracy: 0.98,
        actualAccuracy: accuracy
    )

    let summary = TrainingSummary(
        modelType: "resnet",
        hyperparameters: hyperparameters,
        epochMetrics: epochMetrics,
        finalAccuracy: accuracy,
        benchmarkComparison: benchmarkComparison,
        bestValidationAccuracy: bestEpoch > 0 ? bestValidationAccuracy : nil,
        bestEpoch: bestEpoch > 0 ? bestEpoch : nil
    )

    summary.printSummary()
    summary.printBenchmarkComparison()

    // -------------------------------------------------------------------------
    // JSON Export (if requested)
    // -------------------------------------------------------------------------
    if config.exportJson {
        let fileManager = FileManager.default
        let logsDir = "./logs"

        // Create logs directory if it doesn't exist
        if !fileManager.fileExists(atPath: logsDir) {
            do {
                try fileManager.createDirectory(atPath: logsDir, withIntermediateDirectories: true)
            } catch {
                ColoredPrint.error("Failed to create logs directory: \(error)")
                return
            }
        }

        // Generate timestamped filename
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
        let timestamp = dateFormatter.string(from: Date())
        let filename = "training_summary_\(config.modelType)_\(timestamp).json"
        let filePath = "\(logsDir)/\(filename)"

        // Export to JSON
        do {
            try summary.exportToJSON(filePath: filePath)
            ColoredPrint.success("📄 Training summary exported to: \(filePath)")
        } catch {
            ColoredPrint.error("Failed to export JSON: \(error)")
        }
    }
}

// =============================================================================
// MARK: - Main Entry Point
// =============================================================================

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

    case "resnet":
        trainResNet(config: config, trainImages: trainImages, trainLabels: trainLabels,
                    testImages: testImages, testLabels: testLabels)

    case "attention":
        trainAttention(config: config, trainImages: trainImages, trainLabels: trainLabels,
                       testImages: testImages, testLabels: testLabels)

    case "transformer":
        trainTransformer(config: config, trainImages: trainImages, trainLabels: trainLabels,
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
