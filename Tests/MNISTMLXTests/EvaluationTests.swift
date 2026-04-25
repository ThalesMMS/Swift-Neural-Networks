// ============================================================================
// EvaluationTests.swift - Tests for MNISTMLX Checkpoint Evaluation
// ============================================================================

import XCTest
import MLX
@testable import MNISTMLX

final class EvaluationTests: MLXTestCase {

    private func createTempDirectory() throws -> String {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("evaluation_tests_\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tempDir,
            withIntermediateDirectories: true,
            attributes: nil
        )
        return tempDir.path
    }

    private func removeTempDirectory(_ path: String) {
        try? FileManager.default.removeItem(atPath: path)
    }

    private func createTestHyperparameters() -> TrainingHyperparameters {
        TrainingHyperparameters(
            epochs: 1,
            batchSize: 4,
            learningRate: 0.01,
            seed: 1
        )
    }

    private func createTestImagesAndLabels() -> (MLXArray, MLXArray) {
        let images = MLXArray.zeros([4, 784])
        let labels = MLXArray([Int32(0), Int32(1), Int32(2), Int32(3)])
        return (images, labels)
    }

    func testEvaluateCheckpointRejectsEmptyTestData() {
        let images = MLXArray([Float](), [0, 784])
        let labels = MLXArray([Int32](), [0])

        XCTAssertThrowsError(
            try evaluateCheckpoint(
                checkpointPath: "/tmp/not_needed_for_empty_data.json",
                testImages: images,
                testLabels: labels
            )
        ) { error in
            guard let evaluationError = error as? EvaluationError else {
                XCTFail("Should throw EvaluationError")
                return
            }

            XCTAssertTrue(
                evaluationError.description.contains("empty test set"),
                "Error should describe empty test data: \(evaluationError.description)"
            )
        }
    }

    func testEvaluateCheckpointRejectsMismatchedImageAndLabelCounts() {
        let images = MLXArray.zeros([3, 784])
        let labels = MLXArray([Int32(0), Int32(1)])

        XCTAssertThrowsError(
            try evaluateCheckpoint(
                checkpointPath: "/tmp/not_needed_for_mismatched_data.json",
                testImages: images,
                testLabels: labels
            )
        ) { error in
            guard let evaluationError = error as? EvaluationError else {
                XCTFail("Should throw EvaluationError")
                return
            }

            XCTAssertTrue(
                evaluationError.description.contains("3 images but 2 labels"),
                "Error should describe image/label count mismatch: \(evaluationError.description)"
            )
        }
    }

    func testEvaluateCheckpointLoadsMLPArtifact() throws {
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let model = MLPModel()
        let checkpointPath = "\(tempDir)/best_model_mlp.json"
        try saveCheckpoint(
            model: model,
            modelType: "mlp",
            epoch: 1,
            optimizerState: OptimizerState(learningRate: 0.01),
            hyperparameters: createTestHyperparameters(),
            metrics: CheckpointMetrics(trainLoss: 0.5),
            filePath: checkpointPath
        )

        let (images, labels) = createTestImagesAndLabels()
        let result = try evaluateCheckpoint(
            checkpointPath: checkpointPath,
            testImages: images,
            testLabels: labels
        )

        XCTAssertEqual(result.modelType, "mlp")
        XCTAssertEqual(result.epoch, 1)
        XCTAssertEqual(result.sampleCount, 4)
        XCTAssertTrue(result.testLoss.isFinite)
        XCTAssertGreaterThanOrEqual(result.testAccuracy, 0.0)
        XCTAssertLessThanOrEqual(result.testAccuracy, 1.0)
    }

    func testEvaluateCheckpointAutoDetectsMLPModelType() throws {
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let model = MLPModel()
        let checkpointPath = "\(tempDir)/auto_detect_mlp.json"
        try saveCheckpoint(
            model: model,
            modelType: "mlp",
            epoch: 2,
            optimizerState: OptimizerState(learningRate: 0.01),
            hyperparameters: createTestHyperparameters(),
            metrics: CheckpointMetrics(trainLoss: 0.4),
            filePath: checkpointPath
        )

        let (images, labels) = createTestImagesAndLabels()
        let result = try evaluateCheckpoint(
            checkpointPath: checkpointPath,
            testImages: images,
            testLabels: labels,
            modelTypeOverride: nil
        )

        XCTAssertEqual(result.modelType, "mlp")
        XCTAssertEqual(result.epoch, 2)
    }

    func testEvaluateCheckpointRejectsModelOverrideMismatch() throws {
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let model = MLPModel()
        let checkpointPath = "\(tempDir)/best_model_mlp.json"
        try saveCheckpoint(
            model: model,
            modelType: "mlp",
            epoch: 1,
            optimizerState: OptimizerState(learningRate: 0.01),
            hyperparameters: createTestHyperparameters(),
            metrics: CheckpointMetrics(trainLoss: 0.5),
            filePath: checkpointPath
        )

        let (images, labels) = createTestImagesAndLabels()
        XCTAssertThrowsError(
            try evaluateCheckpoint(
                checkpointPath: checkpointPath,
                testImages: images,
                testLabels: labels,
                modelTypeOverride: "cnn"
            )
        ) { error in
            guard let evaluationError = error as? EvaluationError else {
                XCTFail("Should throw EvaluationError")
                return
            }

            XCTAssertTrue(
                evaluationError.description.contains("Checkpoint contains 'mlp' model but --model cnn was specified"),
                "Error should explain the model mismatch: \(evaluationError.description)"
            )
        }
    }

    func testEvaluateCheckpointSurfacesShapeMismatch() throws {
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let customModel = MLPModel(hiddenSize: 256)
        let checkpointPath = "\(tempDir)/custom_mlp.json"
        try saveCheckpoint(
            model: customModel,
            modelType: "mlp",
            epoch: 1,
            optimizerState: OptimizerState(learningRate: 0.01),
            hyperparameters: createTestHyperparameters(),
            metrics: CheckpointMetrics(trainLoss: 0.5),
            filePath: checkpointPath
        )

        let (images, labels) = createTestImagesAndLabels()
        XCTAssertThrowsError(
            try evaluateCheckpoint(
                checkpointPath: checkpointPath,
                testImages: images,
                testLabels: labels
            )
        ) { error in
            guard let checkpointError = error as? CheckpointError else {
                XCTFail("Should throw CheckpointError")
                return
            }

            XCTAssertTrue(
                checkpointError.description.contains("Shape mismatch"),
                "Error should include shape mismatch details: \(checkpointError.description)"
            )
        }
    }

    func testEvaluateCheckpointMissingFileThrowsDescriptiveError() {
        let missingPath = "/tmp/missing_mnistmlx_checkpoint_\(UUID().uuidString).json"
        let (images, labels) = createTestImagesAndLabels()

        XCTAssertThrowsError(
            try evaluateCheckpoint(
                checkpointPath: missingPath,
                testImages: images,
                testLabels: labels
            )
        ) { error in
            guard let evaluationError = error as? EvaluationError else {
                XCTFail("Should throw EvaluationError")
                return
            }

            XCTAssertEqual(
                evaluationError.description,
                "Checkpoint file not found: \(missingPath)"
            )
        }
    }

    func testEvaluateCheckpointCorruptedJSONThrowsDecodingError() throws {
        let tempDir = try createTempDirectory()
        defer { removeTempDirectory(tempDir) }

        let checkpointPath = "\(tempDir)/corrupted_checkpoint.json"
        try "{ this is not valid JSON }".write(
            toFile: checkpointPath,
            atomically: true,
            encoding: .utf8
        )

        let (images, labels) = createTestImagesAndLabels()
        XCTAssertThrowsError(
            try evaluateCheckpoint(
                checkpointPath: checkpointPath,
                testImages: images,
                testLabels: labels
            )
        ) { error in
            guard case DecodingError.dataCorrupted = error else {
                XCTFail("Should throw DecodingError.dataCorrupted, got \(error)")
                return
            }
        }
    }
}
