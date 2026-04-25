// ============================================================================
// Evaluation.swift - Checkpoint Evaluation for MNISTMLX
// ============================================================================

import Foundation
import MLX
import MLXNN
import MNISTCommon
import MNISTData

struct EvaluationResult {
    let modelType: String
    let epoch: Int
    let testAccuracy: Float
    let testLoss: Float
    let sampleCount: Int
}

enum EvaluationError: Error, CustomStringConvertible {
    case modelTypeMismatch(checkpoint: String, override: String)
    case unsupportedModelType(String)
    case missingCheckpoint(String)
    case invalidTestData(String)

    var description: String {
        switch self {
        case .modelTypeMismatch(let checkpoint, let override):
            return "Checkpoint contains '\(checkpoint)' model but --model \(override) was specified"
        case .unsupportedModelType(let modelType):
            return "Unsupported checkpoint model type '\(modelType)'. Supported types: \(supportedModelTypes.joined(separator: ", "))"
        case .missingCheckpoint(let path):
            return "Checkpoint file not found: \(path)"
        case .invalidTestData(let message):
            return "Invalid test data: \(message)"
        }
    }
}

func runEvaluation(
    checkpointPath: String,
    dataDirectory: String,
    modelTypeOverride: String?
) {
    do {
        ColoredPrint.progress("\n📁 Loading MNIST test dataset...")
        let (testImages, testLabels) = try loadMNIST(directory: dataDirectory, train: false)
        ColoredPrint.info("   Test samples: \(testImages.shape[0])")

        let result = try evaluateCheckpoint(
            checkpointPath: checkpointPath,
            testImages: testImages,
            testLabels: testLabels,
            modelTypeOverride: modelTypeOverride
        )

        printEvaluationSummary(result)
    } catch let error as EvaluationError {
        ColoredPrint.error("❌ Evaluation failed: \(error.description)")
        exit(1)
    } catch let error as CheckpointError {
        ColoredPrint.error("❌ Evaluation failed: \(error.description)")
        exit(1)
    } catch {
        ColoredPrint.error("❌ Evaluation failed: \(error)")
        exit(1)
    }
}

func evaluateCheckpoint(
    checkpointPath: String,
    testImages: MLXArray,
    testLabels: MLXArray,
    modelTypeOverride: String? = nil
) throws -> EvaluationResult {
    guard !testImages.shape.isEmpty, !testLabels.shape.isEmpty else {
        throw EvaluationError.invalidTestData("images and labels must have a batch dimension")
    }

    let sampleCount = testImages.shape[0]
    guard sampleCount == testLabels.shape[0] else {
        throw EvaluationError.invalidTestData("\(sampleCount) images but \(testLabels.shape[0]) labels")
    }
    guard sampleCount > 0 else {
        throw EvaluationError.invalidTestData("empty test set: 0 images and 0 labels")
    }

    guard FileManager.default.fileExists(atPath: checkpointPath) else {
        throw EvaluationError.missingCheckpoint(checkpointPath)
    }

    let checkpoint = try Checkpoint.load(from: checkpointPath)
    let checkpointModelType = checkpoint.modelType.lowercased()

    if let override = modelTypeOverride?.lowercased(), override != checkpointModelType {
        throw EvaluationError.modelTypeMismatch(checkpoint: checkpointModelType, override: override)
    }

    guard supportedModelTypes.contains(checkpointModelType) else {
        throw EvaluationError.unsupportedModelType(checkpoint.modelType)
    }

    switch checkpointModelType {
    case "mlp":
        let model = MLPModel()
        return try evaluateModel(
            model: model,
            checkpoint: checkpoint,
            images: testImages,
            labels: testLabels,
            sampleCount: sampleCount,
            loss: mlpLoss,
            accuracy: mlpAccuracy
        )

    case "cnn":
        let model = CNNModel()
        let images = testImages.reshaped([-1, 1, 28, 28])
        return try evaluateModel(
            model: model,
            checkpoint: checkpoint,
            images: images,
            labels: testLabels,
            sampleCount: sampleCount,
            loss: cnnLoss,
            accuracy: cnnAccuracy
        )

    case "resnet":
        let model = ResNetModel()
        let images = testImages.reshaped([-1, 1, 28, 28])
        return try evaluateModel(
            model: model,
            checkpoint: checkpoint,
            images: images,
            labels: testLabels,
            sampleCount: sampleCount,
            loss: resnetLoss,
            accuracy: resnetAccuracy
        )

    case "attention":
        let model = AttentionModel()
        return try evaluateModel(
            model: model,
            checkpoint: checkpoint,
            images: testImages,
            labels: testLabels,
            sampleCount: sampleCount,
            loss: attentionLoss,
            accuracy: attentionAccuracy
        )

    case "transformer":
        let model = TransformerModel()
        return try evaluateModel(
            model: model,
            checkpoint: checkpoint,
            images: testImages,
            labels: testLabels,
            sampleCount: sampleCount,
            loss: transformerLoss,
            accuracy: transformerAccuracy
        )

    default:
        throw EvaluationError.unsupportedModelType(checkpoint.modelType)
    }
}

private func evaluateModel<M: Module>(
    model: M,
    checkpoint: Checkpoint,
    images: MLXArray,
    labels: MLXArray,
    sampleCount: Int,
    loss: (M, MLXArray, MLXArray) -> MLXArray,
    accuracy: (M, MLXArray, MLXArray) -> Float
) throws -> EvaluationResult {
    eval(model)
    try loadCheckpoint(checkpoint: checkpoint, into: model)

    let lossArray = loss(model, images, labels)
    eval(lossArray)

    return EvaluationResult(
        modelType: checkpoint.modelType.lowercased(),
        epoch: checkpoint.epoch,
        testAccuracy: accuracy(model, images, labels),
        testLoss: lossArray.item(Float.self),
        sampleCount: sampleCount
    )
}

private func printEvaluationSummary(_ result: EvaluationResult) {
    ColoredPrint.success("\n✅ Evaluation complete")
    print("Model Type:   \(result.modelType)")
    print("Checkpoint Epoch: \(result.epoch)")
    print("Test Samples: \(result.sampleCount)")
    print(String(format: "Test Accuracy: %.2f%%", result.testAccuracy * 100))
    print(String(format: "Test Loss:     %.6f", result.testLoss))
}
