// ============================================================================
// TrainingSummaryTests.swift - Tests for Training Summary Module
// ============================================================================
//
// This test suite validates the TrainingSummary module functionality:
// - JSON encoding/decoding with all fields
// - Benchmark comparison logic and computed properties
// - Computed properties calculation (total/average time)
// - JSON export functionality
//
// ============================================================================

import XCTest
import Foundation
@testable import MNISTMLX

final class TrainingSummaryTests: XCTestCase {

    // =============================================================================
    // MARK: - Test Fixtures
    // =============================================================================

    /// Creates sample hyperparameters for testing
    private func createSampleHyperparameters() -> TrainingHyperparameters {
        return TrainingHyperparameters(
            epochs: 10,
            batchSize: 32,
            learningRate: 0.001,
            seed: 42
        )
    }

    /// Creates sample epoch metrics for testing
    private func createSampleEpochMetrics() -> [EpochMetrics] {
        return [
            EpochMetrics(epoch: 1, loss: 0.5, duration: 10.5),
            EpochMetrics(epoch: 2, loss: 0.3, duration: 10.2),
            EpochMetrics(epoch: 3, loss: 0.2, duration: 10.8)
        ]
    }

    /// Creates a complete sample training summary
    private func createSampleTrainingSummary() -> TrainingSummary {
        let hyperparameters = createSampleHyperparameters()
        let epochMetrics = createSampleEpochMetrics()
        let benchmarkComparison = BenchmarkComparison(
            expectedAccuracy: 0.95,
            actualAccuracy: 0.97
        )

        return TrainingSummary(
            modelType: "mlp",
            hyperparameters: hyperparameters,
            epochMetrics: epochMetrics,
            finalAccuracy: 0.97,
            benchmarkComparison: benchmarkComparison
        )
    }

    // =============================================================================
    // MARK: - JSON Encoding Tests
    // =============================================================================

    func testJSONEncodingIncludesAllFields() throws {
        // Create a TrainingSummary with sample data
        let summary = createSampleTrainingSummary()

        // Encode to JSON
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let jsonData = try encoder.encode(summary)

        // Convert to string for inspection
        let jsonString = String(data: jsonData, encoding: .utf8)
        XCTAssertNotNil(jsonString, "JSON string should not be nil")

        // Decode to dictionary to verify all fields are present
        let jsonDict = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
        XCTAssertNotNil(jsonDict, "JSON dictionary should not be nil")

        // Verify all expected fields are present
        XCTAssertNotNil(jsonDict?["modelType"], "modelType field should be present")
        XCTAssertNotNil(jsonDict?["hyperparameters"], "hyperparameters field should be present")
        XCTAssertNotNil(jsonDict?["epochMetrics"], "epochMetrics field should be present")
        XCTAssertNotNil(jsonDict?["finalAccuracy"], "finalAccuracy field should be present")
        XCTAssertNotNil(jsonDict?["benchmarkComparison"], "benchmarkComparison field should be present")

        // Verify computed properties are included
        XCTAssertNotNil(jsonDict?["totalTrainingTime"], "totalTrainingTime computed property should be included")
        XCTAssertNotNil(jsonDict?["averageEpochTime"], "averageEpochTime computed property should be included")

        // Verify values
        XCTAssertEqual(jsonDict?["modelType"] as? String, "mlp")
        if let finalAcc = jsonDict?["finalAccuracy"] as? Double {
            XCTAssertEqual(finalAcc, 0.97, accuracy: 0.001)
        } else {
            XCTFail("finalAccuracy should be present and a Double")
        }
    }

    func testJSONDecodingRestoresAllFields() throws {
        // Create and encode a summary
        let originalSummary = createSampleTrainingSummary()
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(originalSummary)

        // Decode back
        let decoder = JSONDecoder()
        let decodedSummary = try decoder.decode(TrainingSummary.self, from: jsonData)

        // Verify all fields match
        XCTAssertEqual(decodedSummary.modelType, originalSummary.modelType)
        XCTAssertEqual(decodedSummary.finalAccuracy, originalSummary.finalAccuracy, accuracy: 0.001)
        XCTAssertEqual(decodedSummary.hyperparameters.epochs, originalSummary.hyperparameters.epochs)
        XCTAssertEqual(decodedSummary.hyperparameters.batchSize, originalSummary.hyperparameters.batchSize)
        XCTAssertEqual(decodedSummary.hyperparameters.learningRate, originalSummary.hyperparameters.learningRate, accuracy: 0.0001)
        XCTAssertEqual(decodedSummary.hyperparameters.seed, originalSummary.hyperparameters.seed)
        XCTAssertEqual(decodedSummary.epochMetrics.count, originalSummary.epochMetrics.count)

        // Verify benchmark comparison
        XCTAssertNotNil(decodedSummary.benchmarkComparison)
        if let expected = decodedSummary.benchmarkComparison?.expectedAccuracy {
            XCTAssertEqual(Double(expected), 0.95, accuracy: 0.001)
        }
        if let actual = decodedSummary.benchmarkComparison?.actualAccuracy {
            XCTAssertEqual(Double(actual), 0.97, accuracy: 0.001)
        }
    }

    // =============================================================================
    // MARK: - Benchmark Comparison Tests
    // =============================================================================

    func testBenchmarkComparisonMetExpectations() {
        // Test case 1: Actual >= Expected (met expectations)
        let comparisonMet = BenchmarkComparison(
            expectedAccuracy: 0.90,
            actualAccuracy: 0.92
        )
        XCTAssertTrue(comparisonMet.metExpectations, "Should meet expectations when actual >= expected")
        XCTAssertEqual(comparisonMet.difference, 0.02, accuracy: 0.001, "Difference should be actual - expected")

        // Test case 2: Actual < Expected (did not meet expectations)
        let comparisonNotMet = BenchmarkComparison(
            expectedAccuracy: 0.95,
            actualAccuracy: 0.92
        )
        XCTAssertFalse(comparisonNotMet.metExpectations, "Should not meet expectations when actual < expected")
        XCTAssertEqual(comparisonNotMet.difference, -0.03, accuracy: 0.001, "Difference should be negative when below expected")

        // Test case 3: Exact match
        let comparisonExact = BenchmarkComparison(
            expectedAccuracy: 0.95,
            actualAccuracy: 0.95
        )
        XCTAssertTrue(comparisonExact.metExpectations, "Should meet expectations when actual == expected")
        XCTAssertEqual(comparisonExact.difference, 0.0, accuracy: 0.001, "Difference should be zero for exact match")
    }

    func testBenchmarkComparisonEncodesComputedProperties() throws {
        let comparison = BenchmarkComparison(
            expectedAccuracy: 0.90,
            actualAccuracy: 0.92
        )

        // Encode to JSON
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(comparison)

        // Convert to dictionary
        let jsonDict = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
        XCTAssertNotNil(jsonDict)

        // Verify computed properties are included
        XCTAssertNotNil(jsonDict?["difference"], "difference computed property should be encoded")
        XCTAssertNotNil(jsonDict?["metExpectations"], "metExpectations computed property should be encoded")

        // Verify values
        if let diff = jsonDict?["difference"] as? Double {
            XCTAssertEqual(diff, 0.02, accuracy: 0.001)
        } else {
            XCTFail("difference should be present and a Double")
        }
        XCTAssertEqual(jsonDict?["metExpectations"] as? Bool, true)
    }

    // =============================================================================
    // MARK: - Computed Properties Tests
    // =============================================================================

    func testComputedPropertiesCalculation() {
        let epochMetrics = [
            EpochMetrics(epoch: 1, loss: 0.5, duration: 10.0),
            EpochMetrics(epoch: 2, loss: 0.3, duration: 12.0),
            EpochMetrics(epoch: 3, loss: 0.2, duration: 11.0)
        ]

        let summary = TrainingSummary(
            modelType: "test",
            hyperparameters: createSampleHyperparameters(),
            epochMetrics: epochMetrics,
            finalAccuracy: 0.95
        )

        // Test totalTrainingTime: 10 + 12 + 11 = 33
        XCTAssertEqual(summary.totalTrainingTime, 33.0, accuracy: 0.001,
                      "Total training time should be sum of all epoch durations")

        // Test averageEpochTime: 33 / 3 = 11
        XCTAssertEqual(summary.averageEpochTime, 11.0, accuracy: 0.001,
                      "Average epoch time should be total time / number of epochs")
    }

    func testComputedPropertiesWithEmptyEpochMetrics() {
        let summary = TrainingSummary(
            modelType: "test",
            hyperparameters: createSampleHyperparameters(),
            epochMetrics: [],
            finalAccuracy: 0.95
        )

        // Test edge case: empty epochMetrics array
        XCTAssertEqual(summary.totalTrainingTime, 0.0, accuracy: 0.001,
                      "Total training time should be 0 for empty metrics")
        XCTAssertEqual(summary.averageEpochTime, 0.0, accuracy: 0.001,
                      "Average epoch time should be 0 for empty metrics")
    }

    func testComputedPropertiesWithSingleEpoch() {
        let epochMetrics = [
            EpochMetrics(epoch: 1, loss: 0.5, duration: 15.5)
        ]

        let summary = TrainingSummary(
            modelType: "test",
            hyperparameters: createSampleHyperparameters(),
            epochMetrics: epochMetrics,
            finalAccuracy: 0.95
        )

        // Single epoch: total and average should be the same
        XCTAssertEqual(summary.totalTrainingTime, 15.5, accuracy: 0.001)
        XCTAssertEqual(summary.averageEpochTime, 15.5, accuracy: 0.001)
    }

    // =============================================================================
    // MARK: - JSON Export Tests
    // =============================================================================

    func testExportToJSONCreatesValidFile() throws {
        let summary = createSampleTrainingSummary()

        // Create a temporary file path
        let tempDir = FileManager.default.temporaryDirectory
        let fileName = "test_summary_\(UUID().uuidString).json"
        let filePath = tempDir.appendingPathComponent(fileName).path

        // Export to JSON
        try summary.exportToJSON(filePath: filePath)

        // Verify file exists
        XCTAssertTrue(FileManager.default.fileExists(atPath: filePath),
                     "JSON file should be created at specified path")

        // Read and verify file content
        let fileData = try Data(contentsOf: URL(fileURLWithPath: filePath))
        XCTAssertTrue(fileData.count > 0, "JSON file should not be empty")

        // Verify JSON is valid and parseable
        let jsonDict = try JSONSerialization.jsonObject(with: fileData) as? [String: Any]
        XCTAssertNotNil(jsonDict, "File should contain valid JSON")
        XCTAssertEqual(jsonDict?["modelType"] as? String, "mlp")

        // Verify it can be decoded back to TrainingSummary
        let decoder = JSONDecoder()
        let decodedSummary = try decoder.decode(TrainingSummary.self, from: fileData)
        XCTAssertEqual(decodedSummary.modelType, summary.modelType)
        XCTAssertEqual(decodedSummary.finalAccuracy, summary.finalAccuracy, accuracy: 0.001)

        // Clean up
        try? FileManager.default.removeItem(atPath: filePath)
    }

    func testExportToJSONWithPrettyPrinting() throws {
        let summary = createSampleTrainingSummary()

        let tempDir = FileManager.default.temporaryDirectory
        let fileName = "test_summary_pretty_\(UUID().uuidString).json"
        let filePath = tempDir.appendingPathComponent(fileName).path

        try summary.exportToJSON(filePath: filePath)

        // Read file as string
        let fileContent = try String(contentsOfFile: filePath, encoding: .utf8)

        // Verify pretty printing (should contain newlines and indentation)
        XCTAssertTrue(fileContent.contains("\n"), "JSON should be pretty-printed with newlines")
        XCTAssertTrue(fileContent.contains("  "), "JSON should be pretty-printed with indentation")

        // Clean up
        try? FileManager.default.removeItem(atPath: filePath)
    }

    // =============================================================================
    // MARK: - TrainingHyperparameters Tests
    // =============================================================================

    func testTrainingHyperparametersEncodeDecode() throws {
        let hyperparameters = TrainingHyperparameters(
            epochs: 20,
            batchSize: 64,
            learningRate: 0.005,
            seed: 1234
        )

        // Encode and decode
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let jsonData = try encoder.encode(hyperparameters)
        let decoded = try decoder.decode(TrainingHyperparameters.self, from: jsonData)

        // Verify all fields
        XCTAssertEqual(decoded.epochs, 20)
        XCTAssertEqual(decoded.batchSize, 64)
        XCTAssertEqual(decoded.learningRate, 0.005, accuracy: 0.0001)
        XCTAssertEqual(decoded.seed, 1234)
    }

    // =============================================================================
    // MARK: - EpochMetrics Tests
    // =============================================================================

    func testEpochMetricsEncodeDecode() throws {
        let metrics = EpochMetrics(epoch: 5, loss: 0.123, duration: 12.34)

        // Encode and decode
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let jsonData = try encoder.encode(metrics)
        let decoded = try decoder.decode(EpochMetrics.self, from: jsonData)

        // Verify all fields
        XCTAssertEqual(decoded.epoch, 5)
        XCTAssertEqual(decoded.loss, 0.123, accuracy: 0.001)
        XCTAssertEqual(decoded.duration, 12.34, accuracy: 0.01)
    }

    // =============================================================================
    // MARK: - Integration Tests
    // =============================================================================

    func testTrainingSummaryWithoutBenchmarkComparison() throws {
        // Create a summary without benchmark comparison
        let summary = TrainingSummary(
            modelType: "cnn",
            hyperparameters: createSampleHyperparameters(),
            epochMetrics: createSampleEpochMetrics(),
            finalAccuracy: 0.98,
            benchmarkComparison: nil
        )

        // Verify it can be encoded/decoded
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let jsonData = try encoder.encode(summary)
        let decoded = try decoder.decode(TrainingSummary.self, from: jsonData)

        XCTAssertEqual(decoded.modelType, "cnn")
        XCTAssertEqual(decoded.finalAccuracy, 0.98, accuracy: 0.001)
        XCTAssertNil(decoded.benchmarkComparison, "Benchmark comparison should be nil")
    }

    func testMultipleModelsAccuracyValues() {
        // Test MLP benchmark (0.97)
        let mlpComparison = BenchmarkComparison(expectedAccuracy: 0.97, actualAccuracy: 0.975)
        XCTAssertTrue(mlpComparison.metExpectations)
        XCTAssertEqual(mlpComparison.difference, 0.005, accuracy: 0.001)

        // Test CNN benchmark (0.98)
        let cnnComparison = BenchmarkComparison(expectedAccuracy: 0.98, actualAccuracy: 0.982)
        XCTAssertTrue(cnnComparison.metExpectations)
        XCTAssertEqual(cnnComparison.difference, 0.002, accuracy: 0.001)

        // Test Attention benchmark (0.90) - CORRECTED FROM 0.95
        let attentionComparison = BenchmarkComparison(expectedAccuracy: 0.90, actualAccuracy: 0.92)
        XCTAssertTrue(attentionComparison.metExpectations)
        XCTAssertEqual(attentionComparison.difference, 0.02, accuracy: 0.001)
    }
}
