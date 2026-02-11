// ============================================================================
// Checkpointing.swift - Model Checkpoint Save/Load Infrastructure
// ============================================================================
//
// This file defines data structures and functions for saving and loading
// model checkpoints during training. It enables training to be resumed from
// any saved checkpoint, preserving model weights, optimizer state, and
// training progress.
//
// ============================================================================

import Foundation
import MLX
import MLXNN
import MLXOptimizers

// =============================================================================
// MARK: - Optimizer State
// =============================================================================

/// State of the optimizer at checkpoint time
public struct OptimizerState: Codable {
    /// Learning rate being used
    public let learningRate: Float

    /// Optional momentum value (for optimizers that use it)
    public let momentum: Float?

    /// Optional weight decay value (for regularization)
    public let weightDecay: Float?

    public init(learningRate: Float, momentum: Float? = nil, weightDecay: Float? = nil) {
        self.learningRate = learningRate
        self.momentum = momentum
        self.weightDecay = weightDecay
    }
}

// =============================================================================
// MARK: - Checkpoint Metrics
// =============================================================================

/// Training metrics at the time of checkpoint creation
public struct CheckpointMetrics: Codable {
    /// Training loss at this checkpoint
    public let trainLoss: Float

    /// Validation accuracy at this checkpoint (if available)
    public let validationAccuracy: Float?

    /// Training accuracy at this checkpoint (if available)
    public let trainAccuracy: Float?

    public init(trainLoss: Float, validationAccuracy: Float? = nil, trainAccuracy: Float? = nil) {
        self.trainLoss = trainLoss
        self.validationAccuracy = validationAccuracy
        self.trainAccuracy = trainAccuracy
    }
}

// =============================================================================
// MARK: - Model Weights Storage
// =============================================================================

/// Serializable storage for model weights
///
/// Model weights are stored as a dictionary mapping parameter names to
/// flattened arrays. This allows for flexible serialization while maintaining
/// the ability to restore the exact model state.
///
/// ## Example
/// For an MLP model:
/// ```
/// {
///   "hidden.weight": [784 × 512 values],
///   "hidden.bias": [512 values],
///   "output.weight": [512 × 10 values],
///   "output.bias": [10 values]
/// }
/// ```
public struct ModelWeights: Codable {
    /// Dictionary mapping parameter names to their flattened values
    public let parameters: [String: [Float]]

    /// Metadata about parameter shapes for reconstruction
    /// Maps parameter name to its shape dimensions
    public let shapes: [String: [Int]]

    public init(parameters: [String: [Float]], shapes: [String: [Int]]) {
        self.parameters = parameters
        self.shapes = shapes
    }
}

// =============================================================================
// MARK: - Checkpoint
// =============================================================================

/// Complete checkpoint containing all information needed to resume training
///
/// A checkpoint captures the complete state of a training run at a specific
/// epoch, including:
/// - Model architecture type (mlp, cnn, attention)
/// - Current epoch number
/// - Model weights and biases
/// - Optimizer configuration
/// - Training hyperparameters
/// - Current training metrics
///
/// ## Usage
/// ```swift
/// // Create a checkpoint
/// let checkpoint = Checkpoint(
///     modelType: "mlp",
///     epoch: 5,
///     weights: modelWeights,
///     optimizerState: optimState,
///     hyperparameters: hyperparams,
///     metrics: metrics,
///     timestamp: Date()
/// )
///
/// // Save to file
/// try checkpoint.save(to: "checkpoints/checkpoint_mlp_epoch_5.json")
///
/// // Load from file
/// let loaded = try Checkpoint.load(from: "checkpoints/checkpoint_mlp_epoch_5.json")
/// ```
public struct Checkpoint: Codable {
    /// Type of model (e.g., "mlp", "cnn", "attention")
    public let modelType: String

    /// Epoch number when this checkpoint was created (1-indexed)
    public let epoch: Int

    /// Model weights and their shapes
    public let weights: ModelWeights

    /// Optimizer state (learning rate, momentum, etc.)
    public let optimizerState: OptimizerState

    /// Training hyperparameters
    public let hyperparameters: TrainingHyperparameters

    /// Training metrics at checkpoint time
    public let metrics: CheckpointMetrics

    /// Timestamp when checkpoint was created
    public let timestamp: Date

    /// Optional notes or metadata
    public let notes: String?

    public init(
        modelType: String,
        epoch: Int,
        weights: ModelWeights,
        optimizerState: OptimizerState,
        hyperparameters: TrainingHyperparameters,
        metrics: CheckpointMetrics,
        timestamp: Date = Date(),
        notes: String? = nil
    ) {
        self.modelType = modelType
        self.epoch = epoch
        self.weights = weights
        self.optimizerState = optimizerState
        self.hyperparameters = hyperparameters
        self.metrics = metrics
        self.timestamp = timestamp
        self.notes = notes
    }

    // =========================================================================
    // MARK: - File I/O
    // =========================================================================

    /// Saves the checkpoint to a JSON file
    /// - Parameter filePath: The path where the checkpoint should be saved
    /// - Throws: Encoding or file writing errors
    public func save(to filePath: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let jsonData = try encoder.encode(self)
        try jsonData.write(to: URL(fileURLWithPath: filePath))
    }

    /// Loads a checkpoint from a JSON file
    /// - Parameter filePath: The path to the checkpoint file
    /// - Returns: The loaded checkpoint
    /// - Throws: Decoding or file reading errors
    public static func load(from filePath: String) throws -> Checkpoint {
        let fileURL = URL(fileURLWithPath: filePath)
        let jsonData = try Data(contentsOf: fileURL)

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        return try decoder.decode(Checkpoint.self, from: jsonData)
    }

    // =========================================================================
    // MARK: - Validation
    // =========================================================================

    /// Validates that this checkpoint matches the expected model type
    /// - Parameter expectedType: The expected model type (e.g., "mlp", "cnn")
    /// - Returns: True if the checkpoint matches the expected type
    public func validateModelType(_ expectedType: String) -> Bool {
        return modelType.lowercased() == expectedType.lowercased()
    }

    /// Returns a human-readable description of the checkpoint
    public var description: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short

        return """
        Checkpoint Summary:
          Model Type:    \(modelType)
          Epoch:         \(epoch)
          Train Loss:    \(String(format: "%.4f", metrics.trainLoss))
          Learning Rate: \(optimizerState.learningRate)
          Created:       \(formatter.string(from: timestamp))
        """
    }
}

// =============================================================================
// MARK: - Checkpoint Save/Load Functions
// =============================================================================

/// Saves a model checkpoint to disk
///
/// This function extracts model weights from an MLX Module, packages them with
/// training metadata, and saves everything to a JSON file.
///
/// ## Usage
/// ```swift
/// let optimState = OptimizerState(learningRate: 0.01)
/// let hyperparams = TrainingHyperparameters(epochs: 10, batchSize: 32, learningRate: 0.01, seed: 42)
/// let metrics = CheckpointMetrics(trainLoss: 0.123)
///
/// try saveCheckpoint(
///     model: mlpModel,
///     modelType: "mlp",
///     epoch: 5,
///     optimizerState: optimState,
///     hyperparameters: hyperparams,
///     metrics: metrics,
///     filePath: "checkpoints/checkpoint_mlp_epoch_5.json"
/// )
/// ```
///
/// - Parameters:
///   - model: The MLX Module (MLPModel, CNNModel, or AttentionModel)
///   - modelType: Type identifier (e.g., "mlp", "cnn", "attention")
///   - epoch: Current epoch number (1-indexed)
///   - optimizerState: Current optimizer configuration
///   - hyperparameters: Training hyperparameters
///   - metrics: Current training metrics
///   - filePath: Path where checkpoint should be saved
///   - notes: Optional notes or metadata
/// - Throws: Encoding or file writing errors
public func saveCheckpoint<M: Module>(
    model: M,
    modelType: String,
    epoch: Int,
    optimizerState: OptimizerState,
    hyperparameters: TrainingHyperparameters,
    metrics: CheckpointMetrics,
    filePath: String,
    notes: String? = nil
) throws {
    // =========================================================================
    // Step 1: Extract Model Parameters
    // =========================================================================
    // Get all model parameters (weights and biases) using MLX's parameters() API.
    // This returns a nested structure of parameters organized by module hierarchy.
    let modelParams = model.parameters()
    let flattenedParams = modelParams.flattened()

    // =========================================================================
    // Step 2: Convert MLXArrays to Swift Arrays
    // =========================================================================
    // We need to convert MLX's GPU arrays to plain Swift arrays for JSON serialization.
    // We also store the shapes so we can reconstruct the arrays later.
    var parameterDict: [String: [Float]] = [:]
    var shapesDict: [String: [Int]] = [:]

    for (key, param) in flattenedParams {
        // Force evaluation to ensure data is available
        eval(param)

        // Store the shape for later reconstruction
        shapesDict[key] = param.shape

        // Convert to Float array
        parameterDict[key] = param.asArray(Float.self)
    }

    // =========================================================================
    // Step 3: Create ModelWeights
    // =========================================================================
    let weights = ModelWeights(parameters: parameterDict, shapes: shapesDict)

    // =========================================================================
    // Step 4: Create Checkpoint
    // =========================================================================
    let checkpoint = Checkpoint(
        modelType: modelType,
        epoch: epoch,
        weights: weights,
        optimizerState: optimizerState,
        hyperparameters: hyperparameters,
        metrics: metrics,
        timestamp: Date(),
        notes: notes
    )

    // =========================================================================
    // Step 5: Ensure Directory Exists
    // =========================================================================
    // Create parent directory if it doesn't exist
    let fileURL = URL(fileURLWithPath: filePath)
    let directory = fileURL.deletingLastPathComponent()
    try FileManager.default.createDirectory(
        at: directory,
        withIntermediateDirectories: true,
        attributes: nil
    )

    // =========================================================================
    // Step 6: Save to File
    // =========================================================================
    try checkpoint.save(to: filePath)
}

/// Loads a checkpoint and restores model state
///
/// This function takes a loaded Checkpoint and applies the saved weights to
/// an existing MLX Module. It validates that parameter names and shapes match
/// before restoring the weights.
///
/// ## Usage
/// ```swift
/// // Load checkpoint from file
/// let checkpoint = try Checkpoint.load(from: "checkpoints/checkpoint_mlp_epoch_5.json")
///
/// // Validate checkpoint type
/// guard checkpoint.validateModelType("mlp") else {
///     throw CheckpointError.modelTypeMismatch
/// }
///
/// // Restore model state
/// try loadCheckpoint(checkpoint: checkpoint, into: mlpModel)
/// ```
///
/// - Parameters:
///   - checkpoint: The loaded checkpoint containing model weights
///   - model: The MLX Module to restore weights into
/// - Throws: Errors if parameter names/shapes don't match or conversion fails
public func loadCheckpoint<M: Module>(
    checkpoint: Checkpoint,
    into model: M
) throws {
    // =========================================================================
    // Step 1: Get Current Model Parameters
    // =========================================================================
    // Get the model's parameter structure to validate against checkpoint
    let modelParams = model.parameters()
    let flattenedParams = modelParams.flattened()

    // Convert flattened params array to dictionary for easier lookup
    var modelParamsDict: [String: MLXArray] = [:]
    for (key, param) in flattenedParams {
        modelParamsDict[key] = param
    }

    // =========================================================================
    // Step 2: Validate Parameter Names Match
    // =========================================================================
    // Ensure all checkpoint parameters exist in the model
    let checkpointKeys = Set(checkpoint.weights.parameters.keys)
    let modelKeys = Set(modelParamsDict.keys)

    // Check for missing parameters in model
    let missingInModel = checkpointKeys.subtracting(modelKeys)
    if !missingInModel.isEmpty {
        let missingList = missingInModel.sorted().joined(separator: ", ")
        throw CheckpointError.parameterMismatch(
            "Checkpoint contains parameters not in model: \(missingList)"
        )
    }

    // Check for parameters in model not found in checkpoint
    let missingInCheckpoint = modelKeys.subtracting(checkpointKeys)
    if !missingInCheckpoint.isEmpty {
        let missingList = missingInCheckpoint.sorted().joined(separator: ", ")
        throw CheckpointError.parameterMismatch(
            "Model contains parameters not in checkpoint: \(missingList)"
        )
    }

    // =========================================================================
    // Step 3: Convert Float Arrays to MLXArrays and Validate Shapes
    // =========================================================================
    var restoredParams: [String: MLXArray] = [:]

    for (key, floatArray) in checkpoint.weights.parameters {
        // Get expected shape from checkpoint metadata
        guard let expectedShape = checkpoint.weights.shapes[key] else {
            throw CheckpointError.missingShape(
                "Missing shape information for parameter: \(key)"
            )
        }

        // Validate that the current model parameter has the same shape
        guard let currentParam = modelParamsDict[key] else {
            throw CheckpointError.parameterMismatch(
                "Parameter \(key) exists in checkpoint but not in model"
            )
        }

        if currentParam.shape != expectedShape {
            throw CheckpointError.shapeMismatch(
                "Shape mismatch for parameter \(key): " +
                "checkpoint has \(expectedShape), model has \(currentParam.shape)"
            )
        }

        // Convert Float array to MLXArray with the correct shape
        let mlxArray = MLXArray(floatArray, expectedShape)
        restoredParams[key] = mlxArray
    }

    // =========================================================================
    // Step 4: Update Model Parameters
    // =========================================================================
    // Apply the restored parameters to the model
    model.update(parameters: ModuleParameters.unflattened(restoredParams))

    // =========================================================================
    // Step 5: Evaluate to Ensure Parameters are Applied
    // =========================================================================
    // Force evaluation to ensure all parameters are properly loaded
    let updatedParams = model.parameters()
    for (_, param) in updatedParams.flattened() {
        eval(param)
    }
}

// =============================================================================
// MARK: - Checkpoint Errors
// =============================================================================

/// Errors that can occur during checkpoint operations
public enum CheckpointError: Error, CustomStringConvertible {
    case parameterMismatch(String)
    case shapeMismatch(String)
    case missingShape(String)
    case modelTypeMismatch(String)
    case fileNotFound(String)

    public var description: String {
        switch self {
        case .parameterMismatch(let message):
            return "Parameter mismatch: \(message)"
        case .shapeMismatch(let message):
            return "Shape mismatch: \(message)"
        case .missingShape(let message):
            return "Missing shape: \(message)"
        case .modelTypeMismatch(let message):
            return "Model type mismatch: \(message)"
        case .fileNotFound(let message):
            return "File not found: \(message)"
        }
    }
}

// =============================================================================
// MARK: - Best Model Tracking
// =============================================================================

/// Information about the best model seen during training
public struct BestModelInfo: Codable {
    /// Epoch number when best model was achieved
    public let epoch: Int

    /// Validation accuracy of the best model
    public let validationAccuracy: Float

    /// Path to the saved best model checkpoint
    public let checkpointPath: String

    /// Timestamp when best model was saved
    public let timestamp: Date

    public init(epoch: Int, validationAccuracy: Float, checkpointPath: String, timestamp: Date = Date()) {
        self.epoch = epoch
        self.validationAccuracy = validationAccuracy
        self.checkpointPath = checkpointPath
        self.timestamp = timestamp
    }
}

/// Saves the best model to a dedicated file
///
/// This function saves the model with the best validation accuracy seen during
/// training. It creates a checkpoint with metadata indicating it's the best model.
///
/// ## Usage
/// ```swift
/// try saveBestModel(
///     model: mlpModel,
///     modelType: "mlp",
///     epoch: 5,
///     validationAccuracy: 0.98,
///     optimizerState: optimState,
///     hyperparameters: hyperparams,
///     metrics: metrics
/// )
/// ```
///
/// - Parameters:
///   - model: The MLX Module (MLPModel, CNNModel, or AttentionModel)
///   - modelType: Type identifier (e.g., "mlp", "cnn", "attention")
///   - epoch: Epoch number when best model was achieved
///   - validationAccuracy: Validation accuracy of the best model
///   - optimizerState: Current optimizer configuration
///   - hyperparameters: Training hyperparameters
///   - metrics: Current training metrics
/// - Throws: Encoding or file writing errors
/// - Returns: Path to the saved best model file
@discardableResult
public func saveBestModel<M: Module>(
    model: M,
    modelType: String,
    epoch: Int,
    validationAccuracy: Float,
    optimizerState: OptimizerState,
    hyperparameters: TrainingHyperparameters,
    metrics: CheckpointMetrics
) throws -> String {
    // =========================================================================
    // Create Best Model Filename
    // =========================================================================
    let filename = "best_model_\(modelType).json"
    let filePath = "./\(filename)"

    // =========================================================================
    // Add Note Indicating This is the Best Model
    // =========================================================================
    let notes = "Best model with validation accuracy: \(String(format: "%.2f%%", validationAccuracy * 100))"

    // =========================================================================
    // Save Checkpoint with Best Model Metadata
    // =========================================================================
    try saveCheckpoint(
        model: model,
        modelType: modelType,
        epoch: epoch,
        optimizerState: optimizerState,
        hyperparameters: hyperparameters,
        metrics: metrics,
        filePath: filePath,
        notes: notes
    )

    return filePath
}
