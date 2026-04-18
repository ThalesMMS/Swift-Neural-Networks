// ============================================================================
// TrainingOrchestration.swift - Model Training Orchestration
// ============================================================================

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MNISTCommon

// =============================================================================
// MARK: - Shared Training Infrastructure
// =============================================================================

private func makeHyperparameters(config: Config) -> TrainingHyperparameters {
    TrainingHyperparameters(
        epochs: config.epochs,
        batchSize: config.batchSize,
        learningRate: config.learningRate,
        seed: config.seed
    )
}

private func exportSummaryIfNeeded(config: Config, summary: TrainingSummary) {
    guard config.exportJson else { return }

    let logsDir = "./logs"
    let fileManager = FileManager.default
    if !fileManager.fileExists(atPath: logsDir) {
        do {
            try fileManager.createDirectory(atPath: logsDir, withIntermediateDirectories: true)
        } catch {
            ColoredPrint.error("Failed to create logs directory: \(error)")
            return
        }
    }

    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
    let timestamp = dateFormatter.string(from: Date())
    let filePath = "\(logsDir)/training_summary_\(config.modelType)_\(timestamp).json"

    do {
        try summary.exportToJSON(filePath: filePath)
        ColoredPrint.success("📄 Training summary exported to: \(filePath)")
    } catch {
        ColoredPrint.error("Failed to export JSON: \(error)")
    }
}

/// Runs the full training loop, checkpoint management, evaluation, and summary
/// for any model that conforms to `Module`.
///
/// - Parameters:
///   - config: Training configuration
///   - modelType: String identifier for this model (e.g. "mlp", "cnn")
///   - model: The model to train (generic over `Module`)
///   - optimizer: Optimizer to use
///   - trainImages: Training images
///   - trainLabels: Training labels
///   - validationImages: Validation images used for model selection
///   - validationLabels: Validation labels used for model selection
///   - testImages: Test images (in whatever shape the model expects)
///   - testLabels: Test labels used only for final evaluation
///   - trainEpoch: Closure that runs one epoch and returns average loss
///   - evalAccuracy: Closure that evaluates and returns accuracy in [0, 1]
///   - expectedAccuracy: Benchmark accuracy for the summary comparison
private func runTraining<M: Module>(
    config: Config,
    modelType: String,
    model: M,
    optimizer: SGD,
    trainImages: MLXArray,
    trainLabels: MLXArray,
    validationImages: MLXArray,
    validationLabels: MLXArray,
    testImages: MLXArray,
    testLabels: MLXArray,
    trainEpoch: (MLXArray, MLXArray) -> Float,
    evalAccuracy: (MLXArray, MLXArray) -> Float,
    expectedAccuracy: Float
) {
    // -------------------------------------------------------------------------
    // Resume from Checkpoint (if specified)
    // -------------------------------------------------------------------------
    var startEpoch = 1
    var bestValidationAccuracy: Float = 0.0
    var bestEpoch: Int = 0

    if let resumePath = config.resumeFrom {
        ColoredPrint.progress("\n📂 Loading checkpoint from: \(resumePath)")
        do {
            let checkpoint = try Checkpoint.load(from: resumePath)
            guard checkpoint.validateModelType(modelType) else {
                ColoredPrint.error("❌ Model type mismatch: checkpoint is '\(checkpoint.modelType)', expected '\(modelType)'")
                exit(1)
            }
            try loadCheckpoint(checkpoint: checkpoint, into: model)
            bestValidationAccuracy = checkpoint.bestValidationAccuracy ?? checkpoint.metrics.validationAccuracy ?? 0.0
            bestEpoch = checkpoint.bestEpoch ?? (checkpoint.metrics.validationAccuracy == nil ? 0 : checkpoint.epoch)
            if checkpoint.epoch >= config.epochs {
                ColoredPrint.info("Nothing left to train: checkpoint is already at epoch \(checkpoint.epoch), target is \(config.epochs).")
                return
            }
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

    if config.useCompile {
        ColoredPrint.info("   Compilation: enabled ⚡")
    }

    ColoredPrint.info("Epoch | Loss     | Time    | Validation Accuracy")
    ColoredPrint.info("------|----------|---------|--------------------")

    let hyperparams = makeHyperparameters(config: config)
    let optimState = OptimizerState(learningRate: config.learningRate)

    for epoch in startEpoch...config.epochs {
        let startTime = Date()
        let loss = trainEpoch(trainImages, trainLabels)
        let elapsed = Date().timeIntervalSince(startTime)
        let validationAccuracy = evalAccuracy(validationImages, validationLabels)

        ColoredPrint.progress(String(format: "%5d | %.6f | %.2fs | Validation: %.2f%%",
                                     epoch, loss, elapsed, validationAccuracy * 100))
        epochMetrics.append(EpochMetrics(epoch: epoch, loss: loss, duration: elapsed))

        // Save best model
        if validationAccuracy > bestValidationAccuracy {
            bestValidationAccuracy = validationAccuracy
            bestEpoch = epoch
            do {
                let savedPath = try saveBestModel(
                    model: model,
                    modelType: modelType,
                    epoch: epoch,
                    validationAccuracy: validationAccuracy,
                    optimizerState: optimState,
                    hyperparameters: hyperparams,
                    metrics: CheckpointMetrics(trainLoss: loss, validationAccuracy: validationAccuracy)
                )
                ColoredPrint.success("🌟 New best model saved: \(savedPath) (Validation: \(String(format: "%.2f%%", validationAccuracy * 100)))")
            } catch {
                ColoredPrint.error("Failed to save best model: \(error)")
            }
        }

        // Save periodic checkpoint
        if let interval = config.checkpointInterval, epoch % interval == 0 {
            let filePath = "./checkpoints/checkpoint_\(modelType)_epoch_\(epoch).json"
            do {
                try saveCheckpoint(
                    model: model,
                    modelType: modelType,
                    epoch: epoch,
                    optimizerState: optimState,
                    hyperparameters: hyperparams,
                    metrics: CheckpointMetrics(trainLoss: loss, validationAccuracy: validationAccuracy),
                    bestValidationAccuracy: bestValidationAccuracy,
                    bestEpoch: bestEpoch,
                    filePath: filePath
                )
                ColoredPrint.success("💾 Checkpoint saved: \(filePath)")
            } catch {
                ColoredPrint.error("Failed to save checkpoint: \(error)")
            }
        }
    }

    // -------------------------------------------------------------------------
    // Final Evaluation and Summary
    // -------------------------------------------------------------------------
    ColoredPrint.progress("\n📊 Evaluating on test set...")
    let accuracy = evalAccuracy(testImages, testLabels)
    ColoredPrint.info(String(format: "   Test Accuracy: %.2f%%", accuracy * 100))

    let summary = TrainingSummary(
        modelType: modelType,
        hyperparameters: hyperparams,
        epochMetrics: epochMetrics,
        finalAccuracy: accuracy,
        benchmarkComparison: BenchmarkComparison(expectedAccuracy: expectedAccuracy, actualAccuracy: accuracy),
        bestValidationAccuracy: bestEpoch > 0 ? bestValidationAccuracy : nil,
        bestEpoch: bestEpoch > 0 ? bestEpoch : nil
    )

    summary.printSummary()
    summary.printBenchmarkComparison()
    exportSummaryIfNeeded(config: config, summary: summary)
}

// =============================================================================
// MARK: - Per-Model Training Entry Points
// =============================================================================

/// Trains an MLP model and evaluates it
func trainMLP(config: Config, trainImages: MLXArray, trainLabels: MLXArray,
              validationImages: MLXArray, validationLabels: MLXArray,
              testImages: MLXArray, testLabels: MLXArray) {
    ColoredPrint.progress("\n🧠 Training MLP Model")
    ColoredPrint.info("   Architecture: 784 → 512 → 10")
    ColoredPrint.info("   Parameters:   ~407,000")
    print()

    let model = MLPModel()
    eval(model)
    let optimizer = SGD(learningRate: config.learningRate)

    runTraining(
        config: config, modelType: "mlp", model: model, optimizer: optimizer,
        trainImages: trainImages, trainLabels: trainLabels,
        validationImages: validationImages, validationLabels: validationLabels,
        testImages: testImages, testLabels: testLabels,
        trainEpoch: { images, labels in
            config.useCompile
                ? trainMLPEpochCompiled(model: model, optimizer: optimizer, trainImages: images, trainLabels: labels, batchSize: config.batchSize)
                : trainMLPEpoch(model: model, optimizer: optimizer, trainImages: images, trainLabels: labels, batchSize: config.batchSize)
        },
        evalAccuracy: { images, labels in mlpAccuracy(model: model, images: images, labels: labels) },
        expectedAccuracy: 0.97
    )
}

/// Trains a CNN model and evaluates it
func trainCNN(config: Config, trainImages: MLXArray, trainLabels: MLXArray,
              validationImages: MLXArray, validationLabels: MLXArray,
              testImages: MLXArray, testLabels: MLXArray) {
    print("\n🧠 Training CNN Model")
    print("   Architecture: Conv(3×3, 8) → ReLU → MaxPool(2×2) → Linear(10)")
    print("   Parameters:   ~16,000")
    print()

    let model = CNNModel()
    eval(model)
    let optimizer = SGD(learningRate: config.learningRate)
    let reshapedValidationImages = validationImages.reshaped([-1, 1, 28, 28])
    let reshapedTestImages = testImages.reshaped([-1, 1, 28, 28])

    runTraining(
        config: config, modelType: "cnn", model: model, optimizer: optimizer,
        trainImages: trainImages, trainLabels: trainLabels,
        validationImages: reshapedValidationImages, validationLabels: validationLabels,
        testImages: reshapedTestImages, testLabels: testLabels,
        trainEpoch: { images, labels in
            config.useCompile
                ? trainCNNEpochCompiled(model: model, optimizer: optimizer, trainImages: images, trainLabels: labels, batchSize: config.batchSize)
                : trainCNNEpoch(model: model, optimizer: optimizer, trainImages: images, trainLabels: labels, batchSize: config.batchSize)
        },
        evalAccuracy: { images, labels in cnnAccuracy(model: model, images: images, labels: labels) },
        expectedAccuracy: 0.98
    )
}

/// Trains an Attention model and evaluates it
func trainAttention(config: Config, trainImages: MLXArray, trainLabels: MLXArray,
                    validationImages: MLXArray, validationLabels: MLXArray,
                    testImages: MLXArray, testLabels: MLXArray) {
    print("\n🧠 Training Attention Model")
    print("   Architecture: Patches(4×4) → Attention → FFN → Pool → Linear")
    print("   Parameters:   ~8,000")
    print()

    let model = AttentionModel()
    eval(model)
    let optimizer = SGD(learningRate: config.learningRate)

    runTraining(
        config: config, modelType: "attention", model: model, optimizer: optimizer,
        trainImages: trainImages, trainLabels: trainLabels,
        validationImages: validationImages, validationLabels: validationLabels,
        testImages: testImages, testLabels: testLabels,
        trainEpoch: { images, labels in
            config.useCompile
                ? trainAttentionEpochCompiled(model: model, optimizer: optimizer, trainImages: images, trainLabels: labels, batchSize: config.batchSize)
                : trainAttentionEpoch(model: model, optimizer: optimizer, trainImages: images, trainLabels: labels, batchSize: config.batchSize)
        },
        evalAccuracy: { images, labels in attentionAccuracy(model: model, images: images, labels: labels) },
        expectedAccuracy: 0.90
    )
}

/// Trains a Transformer model and evaluates it
func trainTransformer(config: Config, trainImages: MLXArray, trainLabels: MLXArray,
                      validationImages: MLXArray, validationLabels: MLXArray,
                      testImages: MLXArray, testLabels: MLXArray) {
    print("\n🧠 Training Transformer Model")
    print("   Architecture: Patches → Multi-head Self-Attention → LayerNorm → FFN → LayerNorm")
    print("   Parameters:   ~15,000")
    print()

    let model = TransformerModel()
    eval(model)
    let optimizer = SGD(learningRate: config.learningRate)

    runTraining(
        config: config, modelType: "transformer", model: model, optimizer: optimizer,
        trainImages: trainImages, trainLabels: trainLabels,
        validationImages: validationImages, validationLabels: validationLabels,
        testImages: testImages, testLabels: testLabels,
        trainEpoch: { images, labels in
            config.useCompile
                ? trainTransformerEpochCompiled(model: model, optimizer: optimizer, trainImages: images, trainLabels: labels, batchSize: config.batchSize)
                : trainTransformerEpoch(model: model, optimizer: optimizer, trainImages: images, trainLabels: labels, batchSize: config.batchSize)
        },
        evalAccuracy: { images, labels in transformerAccuracy(model: model, images: images, labels: labels) },
        expectedAccuracy: 0.92
    )
}

/// Trains a ResNet model and evaluates it
func trainResNet(config: Config, trainImages: MLXArray, trainLabels: MLXArray,
                 validationImages: MLXArray, validationLabels: MLXArray,
                 testImages: MLXArray, testLabels: MLXArray) {
    print("\n🧠 Training ResNet Model")
    print("   Architecture: Conv → ResidualBlock × 3 → GlobalAvgPool → Linear")
    print("   Parameters:   ~10,000")
    print()

    let model = ResNetModel()
    eval(model)
    let optimizer = SGD(learningRate: config.learningRate)
    let reshapedValidationImages = validationImages.reshaped([-1, 1, 28, 28])
    let reshapedTestImages = testImages.reshaped([-1, 1, 28, 28])

    runTraining(
        config: config, modelType: "resnet", model: model, optimizer: optimizer,
        trainImages: trainImages, trainLabels: trainLabels,
        validationImages: reshapedValidationImages, validationLabels: validationLabels,
        testImages: reshapedTestImages, testLabels: testLabels,
        trainEpoch: { images, labels in
            config.useCompile
                ? trainResNetEpochCompiled(model: model, optimizer: optimizer, trainImages: images, trainLabels: labels, batchSize: config.batchSize)
                : trainResNetEpoch(model: model, optimizer: optimizer, trainImages: images, trainLabels: labels, batchSize: config.batchSize)
        },
        evalAccuracy: { images, labels in resnetAccuracy(model: model, images: images, labels: labels) },
        expectedAccuracy: 0.98
    )
}
