// ============================================================================
// CheckpointingTests.swift - Tests for Checkpoint Save/Load Functionality
// ============================================================================
//
// This test suite validates the checkpoint save/load infrastructure:
// - Checkpoint serialization (saveCheckpoint creates valid JSON)
// - Checkpoint deserialization (loadCheckpoint restores correctly)
// - Model type validation (architecture mismatch detection)
// - File I/O error handling
// - Weight restoration accuracy
// - Best model tracking
//
// ============================================================================

import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import MNISTMLX

final class CheckpointingTests: MLXTestCase {

    // =============================================================================
    // MARK: - Test Utilities
    // =============================================================================

    /// Creates a temporary directory for test checkpoints
    /// - Returns: Path to temporary directory
    private func createTempDirectory() throws -> String {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("checkpoint_tests_\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tempDir,
            withIntermediateDirectories: true,
            attributes: nil
        )
        return tempDir.path
    }

    /// Removes a temporary directory and its contents
    /// - Parameter path: Path to directory to remove
    private func removeTempDirectory(_ path: String) {
        try? FileManager.default.removeItem(atPath: path)
    }

    /// Creates test hyperparameters
    private func createTestHyperparameters() -> TrainingHyperparameters {
        return TrainingHyperparameters(
            epochs: 10,
            batchSize: 32,
            learningRate: 0.01,
            seed: 42
        )
    }

    /// Creates test optimizer state
    private func createTestOptimizerState() -> OptimizerState {
        return OptimizerState(
            learningRate: 0.01,
            momentum: 0.9,
            weightDecay: 0.0001
        )
    }

    /// Creates test checkpoint metrics
    private func createTestMetrics() -> CheckpointMetrics {
        return CheckpointMetrics(
            trainLoss: 0.123,
            validationAccuracy: 0.95,
            trainAccuracy: 0.96
        )
    }

    /// Verifies that two MLXArrays have the same shape and values within tolerance
    private func assertArraysEqual(
        _ array1: MLXArray,
        _ array2: MLXArray,
        tolerance: Float = 1e-5,
        _ message: String = "Arrays should be equal",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        // Check shapes match
        XCTAssertEqual(array1.shape, array2.shape, "Shapes should match", file: file, line: line)

        // Force evaluation
        eval(array1, array2)

        // Convert to Float arrays
        let values1 = array1.asArray(Float.self)
        let values2 = array2.asArray(Float.self)

        // Check all values are close
        XCTAssertEqual(values1.count, values2.count, "Array sizes should match", file: file, line: line)

        for i in 0..<values1.count {
            let diff = abs(values1[i] - values2[i])
            XCTAssertLessThanOrEqual(
                diff,
                tolerance,
                "\(message): value at index \(i) differs by \(diff)",
                file: file,
                line: line
            )
        }
    }

    // =============================================================================
    // MARK: - Checkpoint Save Tests
    // =============================================================================

    func testSaveCheckpointCreatesFile() throws {
        // Test that saveCheckpoint creates a valid JSON file
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let model = MLPModel()
        let checkpointPath = "\(tempDir)/test_checkpoint.json"

        // Save checkpoint
        try saveCheckpoint(
            model: model,
            modelType: "mlp",
            epoch: 1,
            optimizerState: createTestOptimizerState(),
            hyperparameters: createTestHyperparameters(),
            metrics: createTestMetrics(),
            filePath: checkpointPath
        )

        // Verify file was created
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: checkpointPath),
            "Checkpoint file should exist"
        )
    }

    func testSaveCheckpointCreatesValidJSON() throws {
        // Test that saveCheckpoint creates valid, parseable JSON
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let model = MLPModel()
        let checkpointPath = "\(tempDir)/test_checkpoint.json"

        // Save checkpoint
        try saveCheckpoint(
            model: model,
            modelType: "mlp",
            epoch: 5,
            optimizerState: createTestOptimizerState(),
            hyperparameters: createTestHyperparameters(),
            metrics: createTestMetrics(),
            filePath: checkpointPath,
            notes: "Test checkpoint"
        )

        // Load and verify JSON structure
        let checkpoint = try Checkpoint.load(from: checkpointPath)

        XCTAssertEqual(checkpoint.modelType, "mlp", "Model type should match")
        XCTAssertEqual(checkpoint.epoch, 5, "Epoch should match")
        XCTAssertEqual(checkpoint.notes, "Test checkpoint", "Notes should match")
        XCTAssertEqual(checkpoint.optimizerState.learningRate, 0.01, accuracy: 0.0001)
        XCTAssertEqual(checkpoint.metrics.trainLoss, 0.123, accuracy: 0.0001)
    }

    func testSaveCheckpointCreatesDirectoryIfNeeded() throws {
        // Test that saveCheckpoint creates parent directories
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let model = MLPModel()
        let checkpointPath = "\(tempDir)/subdir/nested/checkpoint.json"

        // Save checkpoint (should create subdir/nested/)
        try saveCheckpoint(
            model: model,
            modelType: "mlp",
            epoch: 1,
            optimizerState: createTestOptimizerState(),
            hyperparameters: createTestHyperparameters(),
            metrics: createTestMetrics(),
            filePath: checkpointPath
        )

        // Verify file was created in nested directory
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: checkpointPath),
            "Checkpoint should be created in nested directory"
        )
    }

    func testSaveCheckpointIncludesAllParameters() throws {
        // Test that all model parameters are saved
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let model = MLPModel()
        let checkpointPath = "\(tempDir)/test_checkpoint.json"

        // Save checkpoint
        try saveCheckpoint(
            model: model,
            modelType: "mlp",
            epoch: 1,
            optimizerState: createTestOptimizerState(),
            hyperparameters: createTestHyperparameters(),
            metrics: createTestMetrics(),
            filePath: checkpointPath
        )

        // Load checkpoint and verify parameters
        let checkpoint = try Checkpoint.load(from: checkpointPath)

        // MLP should have 4 parameters: hidden.weight, hidden.bias, output.weight, output.bias
        XCTAssertEqual(checkpoint.weights.parameters.count, 4, "Should have 4 parameters")
        XCTAssertTrue(
            checkpoint.weights.parameters.keys.contains("hidden.weight"),
            "Should contain hidden.weight"
        )
        XCTAssertTrue(
            checkpoint.weights.parameters.keys.contains("hidden.bias"),
            "Should contain hidden.bias"
        )
        XCTAssertTrue(
            checkpoint.weights.parameters.keys.contains("output.weight"),
            "Should contain output.weight"
        )
        XCTAssertTrue(
            checkpoint.weights.parameters.keys.contains("output.bias"),
            "Should contain output.bias"
        )

        // Verify shapes are stored
        XCTAssertEqual(checkpoint.weights.shapes.count, 4, "Should have 4 shapes")
        XCTAssertEqual(
            checkpoint.weights.shapes["hidden.weight"],
            [784, 512],
            "Hidden weight shape should be [784, 512]"
        )
        XCTAssertEqual(
            checkpoint.weights.shapes["hidden.bias"],
            [512],
            "Hidden bias shape should be [512]"
        )
        XCTAssertEqual(
            checkpoint.weights.shapes["output.weight"],
            [512, 10],
            "Output weight shape should be [512, 10]"
        )
        XCTAssertEqual(
            checkpoint.weights.shapes["output.bias"],
            [10],
            "Output bias shape should be [10]"
        )
    }

    // =============================================================================
    // MARK: - Checkpoint Load Tests
    // =============================================================================

    func testLoadCheckpointRestoresWeights() throws {
        // Test that loadCheckpoint correctly restores model weights
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let originalModel = MLPModel()
        let checkpointPath = "\(tempDir)/test_checkpoint.json"

        // Save original model
        try saveCheckpoint(
            model: originalModel,
            modelType: "mlp",
            epoch: 1,
            optimizerState: createTestOptimizerState(),
            hyperparameters: createTestHyperparameters(),
            metrics: createTestMetrics(),
            filePath: checkpointPath
        )

        // Create new model with different weights
        let newModel = MLPModel()

        // Load checkpoint into new model
        let checkpoint = try Checkpoint.load(from: checkpointPath)
        try loadCheckpoint(checkpoint: checkpoint, into: newModel)

        // Verify weights match
        let originalParams = originalModel.parameters().flattened()
        let newParams = newModel.parameters().flattened()

        for (key, originalParam) in originalParams {
            guard let newParam = newParams.first(where: { $0.0 == key })?.1 else {
                XCTFail("Parameter \(key) not found in restored model")
                continue
            }

            assertArraysEqual(
                originalParam,
                newParam,
                "Parameter \(key) should match after restoration"
            )
        }
    }

    func testLoadCheckpointPreservesModelBehavior() throws {
        // Test that a restored model produces the same outputs as the original
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let originalModel = MLPModel()
        let checkpointPath = "\(tempDir)/test_checkpoint.json"

        // Create test input
        let testInput = MLXRandom.normal([4, 784])

        // Get original output
        let originalOutput = originalModel(testInput)
        eval(originalOutput)

        // Save checkpoint
        try saveCheckpoint(
            model: originalModel,
            modelType: "mlp",
            epoch: 1,
            optimizerState: createTestOptimizerState(),
            hyperparameters: createTestHyperparameters(),
            metrics: createTestMetrics(),
            filePath: checkpointPath
        )

        // Create new model and restore from checkpoint
        let restoredModel = MLPModel()
        let checkpoint = try Checkpoint.load(from: checkpointPath)
        try loadCheckpoint(checkpoint: checkpoint, into: restoredModel)

        // Get restored output
        let restoredOutput = restoredModel(testInput)
        eval(restoredOutput)

        // Verify outputs match
        assertArraysEqual(
            originalOutput,
            restoredOutput,
            "Model outputs should match after restoration"
        )
    }

    // =============================================================================
    // MARK: - Model Type Validation Tests
    // =============================================================================

    func testCheckpointValidateModelType() throws {
        // Test that checkpoint validation catches model type mismatches
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let model = MLPModel()
        let checkpointPath = "\(tempDir)/test_checkpoint.json"

        // Save MLP checkpoint
        try saveCheckpoint(
            model: model,
            modelType: "mlp",
            epoch: 1,
            optimizerState: createTestOptimizerState(),
            hyperparameters: createTestHyperparameters(),
            metrics: createTestMetrics(),
            filePath: checkpointPath
        )

        let checkpoint = try Checkpoint.load(from: checkpointPath)

        // Valid model type should pass
        XCTAssertTrue(
            checkpoint.validateModelType("mlp"),
            "Should validate matching model type"
        )
        XCTAssertTrue(
            checkpoint.validateModelType("MLP"),
            "Should be case-insensitive"
        )

        // Invalid model type should fail
        XCTAssertFalse(
            checkpoint.validateModelType("cnn"),
            "Should reject mismatched model type"
        )
        XCTAssertFalse(
            checkpoint.validateModelType("attention"),
            "Should reject mismatched model type"
        )
    }

    func testLoadCheckpointDetectsShapeMismatch() throws {
        // Test that loading fails when parameter shapes don't match
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        // Save checkpoint from model with default hidden size (512)
        let model512 = MLPModel(hiddenSize: 512)
        let checkpointPath = "\(tempDir)/test_checkpoint.json"

        try saveCheckpoint(
            model: model512,
            modelType: "mlp",
            epoch: 1,
            optimizerState: createTestOptimizerState(),
            hyperparameters: createTestHyperparameters(),
            metrics: createTestMetrics(),
            filePath: checkpointPath
        )

        // Try to load into model with different hidden size (256)
        let model256 = MLPModel(hiddenSize: 256)
        let checkpoint = try Checkpoint.load(from: checkpointPath)

        // Should throw an error due to shape mismatch
        XCTAssertThrowsError(
            try loadCheckpoint(checkpoint: checkpoint, into: model256),
            "Should throw error on shape mismatch"
        ) { error in
            guard let checkpointError = error as? CheckpointError else {
                XCTFail("Should throw CheckpointError")
                return
            }

            // Verify it's a shape mismatch error
            let errorDescription = checkpointError.description
            XCTAssertTrue(
                errorDescription.contains("Shape mismatch") ||
                errorDescription.contains("shape mismatch"),
                "Error should mention shape mismatch: \(errorDescription)"
            )
        }
    }

    // =============================================================================
    // MARK: - File I/O Error Handling Tests
    // =============================================================================

    func testLoadCheckpointFromNonexistentFile() {
        // Test that loading from nonexistent file throws appropriate error
        let nonexistentPath = "/tmp/nonexistent_\(UUID().uuidString).json"

        XCTAssertThrowsError(
            try Checkpoint.load(from: nonexistentPath),
            "Should throw error for nonexistent file"
        )
    }

    func testLoadCheckpointFromInvalidJSON() throws {
        // Test that loading from invalid JSON throws appropriate error
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let invalidJSONPath = "\(tempDir)/invalid.json"

        // Create file with invalid JSON
        let invalidJSON = "{ this is not valid JSON }"
        try invalidJSON.write(toFile: invalidJSONPath, atomically: true, encoding: .utf8)

        XCTAssertThrowsError(
            try Checkpoint.load(from: invalidJSONPath),
            "Should throw error for invalid JSON"
        )
    }

    func testLoadCheckpointFromCorruptedData() throws {
        // Test that loading from corrupted checkpoint data fails gracefully
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let corruptedPath = "\(tempDir)/corrupted.json"

        // Create valid JSON but with wrong structure (missing required fields)
        let corruptedJSON = """
        {
            "modelType": "mlp",
            "epoch": 1
        }
        """
        try corruptedJSON.write(toFile: corruptedPath, atomically: true, encoding: .utf8)

        XCTAssertThrowsError(
            try Checkpoint.load(from: corruptedPath),
            "Should throw error for corrupted checkpoint data"
        )
    }

    // =============================================================================
    // MARK: - Best Model Tracking Tests
    // =============================================================================

    func testSaveBestModelCreatesFile() throws {
        // Test that saveBestModel creates a file with correct naming
        let bestModelPath = "./best_model_mlp.json"
        defer { try? FileManager.default.removeItem(atPath: bestModelPath) }

        let model = MLPModel()

        // Save best model
        let savedPath = try saveBestModel(
            model: model,
            modelType: "mlp",
            epoch: 5,
            validationAccuracy: 0.98,
            optimizerState: createTestOptimizerState(),
            hyperparameters: createTestHyperparameters(),
            metrics: createTestMetrics()
        )

        XCTAssertEqual(savedPath, bestModelPath, "Should return correct path")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: bestModelPath),
            "Best model file should exist"
        )
    }

    func testSaveBestModelIncludesMetadata() throws {
        // Test that best model checkpoint includes validation accuracy in metadata
        let bestModelPath = "./best_model_mlp.json"
        defer { try? FileManager.default.removeItem(atPath: bestModelPath) }

        let model = MLPModel()
        let validationAcc: Float = 0.98

        // Save best model
        try saveBestModel(
            model: model,
            modelType: "mlp",
            epoch: 5,
            validationAccuracy: validationAcc,
            optimizerState: createTestOptimizerState(),
            hyperparameters: createTestHyperparameters(),
            metrics: createTestMetrics()
        )

        // Load and verify metadata
        let checkpoint = try Checkpoint.load(from: bestModelPath)

        XCTAssertNotNil(checkpoint.notes, "Best model should have notes")
        XCTAssertTrue(
            checkpoint.notes?.contains("Best model") ?? false,
            "Notes should indicate this is the best model"
        )
        XCTAssertTrue(
            checkpoint.notes?.contains("98.00%") ?? false,
            "Notes should include validation accuracy"
        )
    }

    // =============================================================================
    // MARK: - Checkpoint Metadata Tests
    // =============================================================================

    func testCheckpointDescription() throws {
        // Test that checkpoint description provides useful information
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let model = MLPModel()
        let checkpointPath = "\(tempDir)/test_checkpoint.json"

        try saveCheckpoint(
            model: model,
            modelType: "mlp",
            epoch: 5,
            optimizerState: createTestOptimizerState(),
            hyperparameters: createTestHyperparameters(),
            metrics: createTestMetrics(),
            filePath: checkpointPath
        )

        let checkpoint = try Checkpoint.load(from: checkpointPath)
        let description = checkpoint.description

        // Verify description contains key information
        XCTAssertTrue(description.contains("mlp"), "Description should mention model type")
        XCTAssertTrue(description.contains("5"), "Description should mention epoch")
        XCTAssertTrue(description.contains("0.123"), "Description should mention loss")
        XCTAssertTrue(description.contains("0.01"), "Description should mention learning rate")
    }

    func testCheckpointTimestamp() throws {
        // Test that checkpoint includes accurate timestamp
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let model = MLPModel()
        let checkpointPath = "\(tempDir)/test_checkpoint.json"

        let beforeSave = Date()

        try saveCheckpoint(
            model: model,
            modelType: "mlp",
            epoch: 1,
            optimizerState: createTestOptimizerState(),
            hyperparameters: createTestHyperparameters(),
            metrics: createTestMetrics(),
            filePath: checkpointPath
        )

        let afterSave = Date()

        let checkpoint = try Checkpoint.load(from: checkpointPath)

        // Timestamp should be between beforeSave and afterSave
        XCTAssertGreaterThanOrEqual(
            checkpoint.timestamp,
            beforeSave,
            "Timestamp should be after save started"
        )
        XCTAssertLessThanOrEqual(
            checkpoint.timestamp,
            afterSave,
            "Timestamp should be before save completed"
        )
    }

    // =============================================================================
    // MARK: - Edge Cases and Stress Tests
    // =============================================================================

    func testSaveLoadMultipleCheckpoints() throws {
        // Test saving and loading multiple checkpoints in sequence
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let model = MLPModel()

        // Save multiple checkpoints
        for epoch in 1...5 {
            let checkpointPath = "\(tempDir)/checkpoint_epoch_\(epoch).json"
            let metrics = CheckpointMetrics(
                trainLoss: Float(epoch) * 0.1,
                validationAccuracy: 0.9 + Float(epoch) * 0.01
            )

            try saveCheckpoint(
                model: model,
                modelType: "mlp",
                epoch: epoch,
                optimizerState: createTestOptimizerState(),
                hyperparameters: createTestHyperparameters(),
                metrics: metrics,
                filePath: checkpointPath
            )
        }

        // Verify all checkpoints were created and have correct data
        for epoch in 1...5 {
            let checkpointPath = "\(tempDir)/checkpoint_epoch_\(epoch).json"
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: checkpointPath),
                "Checkpoint for epoch \(epoch) should exist"
            )

            let checkpoint = try Checkpoint.load(from: checkpointPath)
            XCTAssertEqual(checkpoint.epoch, epoch, "Epoch should match")
        }
    }

    func testCheckpointWithCustomHiddenSize() throws {
        // Test checkpoint/restore with non-standard model architecture
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let customHiddenSize = 256
        let originalModel = MLPModel(hiddenSize: customHiddenSize)
        let checkpointPath = "\(tempDir)/test_checkpoint.json"

        // Save checkpoint
        try saveCheckpoint(
            model: originalModel,
            modelType: "mlp",
            epoch: 1,
            optimizerState: createTestOptimizerState(),
            hyperparameters: createTestHyperparameters(),
            metrics: createTestMetrics(),
            filePath: checkpointPath
        )

        // Restore into model with same architecture
        let restoredModel = MLPModel(hiddenSize: customHiddenSize)
        let checkpoint = try Checkpoint.load(from: checkpointPath)
        try loadCheckpoint(checkpoint: checkpoint, into: restoredModel)

        // Verify weights match
        let testInput = MLXRandom.normal([4, 784])
        let originalOutput = originalModel(testInput)
        let restoredOutput = restoredModel(testInput)

        eval(originalOutput, restoredOutput)

        assertArraysEqual(
            originalOutput,
            restoredOutput,
            "Custom architecture should restore correctly"
        )
    }

    func testCheckpointWithZeroWeights() throws {
        // Test checkpoint/restore handles edge case of zero-initialized weights
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let model = MLPModel()

        // Zero out all weights
        let params = model.parameters()
        let flatParams = params.flattened()
        var zeroParams: [String: MLXArray] = [:]
        for (key, param) in flatParams {
            zeroParams[key] = MLXArray.zeros(param.shape)
        }
        model.update(parameters: ModuleParameters.unflattened(zeroParams))

        let checkpointPath = "\(tempDir)/zero_checkpoint.json"

        // Save checkpoint
        try saveCheckpoint(
            model: model,
            modelType: "mlp",
            epoch: 1,
            optimizerState: createTestOptimizerState(),
            hyperparameters: createTestHyperparameters(),
            metrics: createTestMetrics(),
            filePath: checkpointPath
        )

        // Restore into new model
        let restoredModel = MLPModel()
        let checkpoint = try Checkpoint.load(from: checkpointPath)
        try loadCheckpoint(checkpoint: checkpoint, into: restoredModel)

        // Verify all weights are zero
        let restoredParams = restoredModel.parameters().flattened()
        for (key, param) in restoredParams {
            eval(param)
            let values = param.asArray(Float.self)
            let allZero = values.allSatisfy { $0 == 0.0 }
            XCTAssertTrue(allZero, "Parameter \(key) should be all zeros")
        }
    }

    // =============================================================================
    // MARK: - Resume Training Tests
    // =============================================================================

    func testResumeTraining() throws {
        // Test that training can be resumed from a checkpoint
        //
        // WHAT WE'RE TESTING:
        // - Model weights are correctly restored from checkpoint
        // - Training continues from the correct epoch
        // - Loss continues to decrease after resume
        // - Resumed model produces same outputs as saved model
        //
        // WORKFLOW:
        // 1. Train model for N epochs
        // 2. Save checkpoint
        // 3. Create new model instance
        // 4. Resume training from checkpoint
        // 5. Verify continuation is seamless

        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        // Create toy dataset for training
        let numSamples = 100
        let images = abs(MLXRandom.normal([numSamples, 784])) * 0.1
        let labelData = (0..<numSamples).map { _ in Int32.random(in: 0..<10) }
        let labels = MLXArray(labelData)

        // PHASE 1: Initial training for 3 epochs
        let initialModel = MLPModel()
        let initialOptimizer = SGD(learningRate: 0.01)

        var initialLosses: [Float] = []
        for _ in 0..<3 {
            let epochLoss = trainEpoch(
                model: initialModel,
                optimizer: initialOptimizer,
                images: images,
                labels: labels,
                batchSize: 32
            )
            initialLosses.append(epochLoss)
        }

        // Verify initial training showed convergence
        XCTAssertLessThan(
            initialLosses[2],
            initialLosses[0],
            "Loss should decrease during initial training"
        )

        // Create test input for verifying model behavior
        let testInput = MLXRandom.normal([4, 784])
        let initialOutput = initialModel(testInput)
        eval(initialOutput)

        // Save checkpoint after epoch 3
        let checkpointPath = "\(tempDir)/resume_checkpoint.json"
        let checkpointEpoch = 3
        let checkpointMetrics = CheckpointMetrics(
            trainLoss: initialLosses[2],
            validationAccuracy: 0.85,
            trainAccuracy: 0.90
        )

        try saveCheckpoint(
            model: initialModel,
            modelType: "mlp",
            epoch: checkpointEpoch,
            optimizerState: OptimizerState(
                learningRate: initialOptimizer.learningRate,
                momentum: 0.0,
                weightDecay: 0.0
            ),
            hyperparameters: createTestHyperparameters(),
            metrics: checkpointMetrics,
            filePath: checkpointPath
        )

        // PHASE 2: Resume training with new model instance
        let resumedModel = MLPModel()

        // Verify new model initially produces different outputs
        let initialResumedOutput = resumedModel(testInput)
        eval(initialResumedOutput)

        // Models should differ before loading checkpoint (different random weights)
        let initialOutputValues = initialOutput.asArray(Float.self)
        let initialResumedValues = initialResumedOutput.asArray(Float.self)
        var hasDifference = false
        for i in 0..<min(10, initialOutputValues.count) {
            if abs(initialOutputValues[i] - initialResumedValues[i]) > 1e-4 {
                hasDifference = true
                break
            }
        }
        XCTAssertTrue(
            hasDifference,
            "New model should have different weights before loading checkpoint"
        )

        // Load checkpoint into resumed model
        let checkpoint = try Checkpoint.load(from: checkpointPath)
        try loadCheckpoint(checkpoint: checkpoint, into: resumedModel)

        // Verify checkpoint metadata
        XCTAssertEqual(checkpoint.epoch, checkpointEpoch, "Checkpoint should record correct epoch")
        XCTAssertEqual(checkpoint.modelType, "mlp", "Checkpoint should record correct model type")

        // Verify resumed model now produces same outputs as saved model
        let resumedOutput = resumedModel(testInput)
        eval(resumedOutput)

        assertArraysEqual(
            initialOutput,
            resumedOutput,
            "Resumed model should produce same outputs as saved model"
        )

        // PHASE 3: Continue training from checkpoint
        let resumedOptimizer = SGD(learningRate: 0.01)
        var resumedLosses: [Float] = []

        // Train for 2 more epochs (epochs 4 and 5)
        for _ in 0..<2 {
            let epochLoss = trainEpoch(
                model: resumedModel,
                optimizer: resumedOptimizer,
                images: images,
                labels: labels,
                batchSize: 32
            )
            resumedLosses.append(epochLoss)
        }

        // Verify training continued successfully
        XCTAssertEqual(resumedLosses.count, 2, "Should train for 2 additional epochs")

        // Verify loss continues to decrease (or at least doesn't increase significantly)
        // We compare the last initial loss with the resumed losses
        let lastInitialLoss = initialLosses[2]
        let firstResumedLoss = resumedLosses[0]

        // First resumed epoch might increase slightly due to optimizer state reset,
        // but should generally continue the trend
        XCTAssertLessThan(
            firstResumedLoss,
            lastInitialLoss * 1.5,
            "Resumed training loss should not spike too much (last initial: \(lastInitialLoss), first resumed: \(firstResumedLoss))"
        )

        // By the end of resumed training, loss should be lower than checkpoint
        let finalResumedLoss = resumedLosses[1]
        XCTAssertLessThan(
            finalResumedLoss,
            lastInitialLoss,
            "Final resumed loss should be lower than checkpoint loss (checkpoint: \(lastInitialLoss), final: \(finalResumedLoss))"
        )

        // Overall convergence check: final loss should be much lower than very first loss
        let veryFirstLoss = initialLosses[0]
        XCTAssertLessThan(
            finalResumedLoss,
            veryFirstLoss * 0.8,
            "Final loss should be significantly lower than initial loss (initial: \(veryFirstLoss), final: \(finalResumedLoss))"
        )
    }

    // =============================================================================
    // MARK: - Training Helper for Resume Tests
    // =============================================================================

    /// Trains a model for one epoch (simplified training loop for testing)
    ///
    /// - Parameters:
    ///   - model: The model to train
    ///   - optimizer: The optimizer
    ///   - images: Training images
    ///   - labels: Training labels
    ///   - batchSize: Batch size
    /// - Returns: Average loss for the epoch
    private func trainEpoch(
        model: MLPModel,
        optimizer: SGD,
        images: MLXArray,
        labels: MLXArray,
        batchSize: Int
    ) -> Float {
        let n = images.shape[0]
        var totalLoss: Float = 0
        var batchCount = 0

        // Setup automatic differentiation
        let lossAndGrad = valueAndGrad(model: model, mlpLoss)

        // Shuffle indices for SGD
        var indices = Array(0..<n)
        indices.shuffle()

        // Training loop
        var start = 0
        while start < n {
            let end = min(start + batchSize, n)
            let batchIndices = Array(indices[start..<end]).map { Int32($0) }
            let idxArray = MLXArray(batchIndices)

            let batchImages = images[idxArray]
            let batchLabels = labels[idxArray]

            // Forward + backward pass
            let (loss, grads) = lossAndGrad(model, batchImages, batchLabels)

            // Update parameters
            optimizer.update(model: model, gradients: grads)
            eval(model, optimizer)

            totalLoss += loss.item(Float.self)
            batchCount += 1

            start = end
        }

        return totalLoss / Float(batchCount)
    }
}
