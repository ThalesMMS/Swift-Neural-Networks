// ============================================================================
// TrainingSummary.swift - Training Metrics and Summary Data Structures
// ============================================================================
//
// This file defines data structures for collecting and storing training
// metrics throughout the training process. It provides a structured way to
// track hyperparameters, epoch-wise metrics, and final results.
//
// ============================================================================

import Foundation
import MNISTCommon

// =============================================================================
// MARK: - Training Hyperparameters
// =============================================================================

/// Hyperparameters used during training
public struct TrainingHyperparameters: Codable {
    /// Number of training epochs
    public let epochs: Int

    /// Batch size for mini-batch gradient descent
    public let batchSize: Int

    /// Learning rate for optimization
    public let learningRate: Float

    /// Random seed for reproducibility
    public let seed: UInt64

    public init(epochs: Int, batchSize: Int, learningRate: Float, seed: UInt64) {
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.seed = seed
    }
}

// =============================================================================
// MARK: - Epoch Metrics
// =============================================================================

/// Metrics collected for a single training epoch
public struct EpochMetrics: Codable {
    /// Epoch number (1-indexed)
    public let epoch: Int

    /// Average loss for this epoch
    public let loss: Float

    /// Time taken to complete this epoch (in seconds)
    public let duration: Double

    public init(epoch: Int, loss: Float, duration: Double) {
        self.epoch = epoch
        self.loss = loss
        self.duration = duration
    }
}

// =============================================================================
// MARK: - Benchmark Comparison
// =============================================================================

/// Comparison of achieved accuracy against expected benchmark
public struct BenchmarkComparison: Codable {
    /// Expected accuracy for this model type (from README benchmarks)
    public let expectedAccuracy: Float

    /// Actual achieved accuracy
    public let actualAccuracy: Float

    /// Difference between actual and expected (actual - expected)
    public var difference: Float {
        return actualAccuracy - expectedAccuracy
    }

    /// Whether the model met or exceeded expectations
    public var metExpectations: Bool {
        return actualAccuracy >= expectedAccuracy
    }

    public init(expectedAccuracy: Float, actualAccuracy: Float) {
        self.expectedAccuracy = expectedAccuracy
        self.actualAccuracy = actualAccuracy
    }

    // Custom encoding to include computed properties
    enum CodingKeys: String, CodingKey {
        case expectedAccuracy
        case actualAccuracy
        case difference
        case metExpectations
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(expectedAccuracy, forKey: .expectedAccuracy)
        try container.encode(actualAccuracy, forKey: .actualAccuracy)
        try container.encode(difference, forKey: .difference)
        try container.encode(metExpectations, forKey: .metExpectations)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        expectedAccuracy = try container.decode(Float.self, forKey: .expectedAccuracy)
        actualAccuracy = try container.decode(Float.self, forKey: .actualAccuracy)
        // Note: difference and metExpectations are computed from the stored properties
    }
}

// =============================================================================
// MARK: - Training Summary
// =============================================================================

/// Complete summary of a training run including all metrics and results
public struct TrainingSummary: Codable {
    /// Type of model trained (e.g., "mlp", "cnn", "attention")
    public let modelType: String

    /// Hyperparameters used during training
    public let hyperparameters: TrainingHyperparameters

    /// Metrics collected for each epoch
    public let epochMetrics: [EpochMetrics]

    /// Final test accuracy achieved
    public let finalAccuracy: Float

    /// Comparison against benchmark accuracy
    public let benchmarkComparison: BenchmarkComparison?

    /// Best validation accuracy achieved during training
    public let bestValidationAccuracy: Float?

    /// Epoch number when best validation accuracy was achieved
    public let bestEpoch: Int?

    /// Total training time (sum of all epoch durations)
    public var totalTrainingTime: Double {
        return epochMetrics.reduce(0.0) { $0 + $1.duration }
    }

    /// Average time per epoch
    public var averageEpochTime: Double {
        guard !epochMetrics.isEmpty else { return 0.0 }
        return totalTrainingTime / Double(epochMetrics.count)
    }

    public init(
        modelType: String,
        hyperparameters: TrainingHyperparameters,
        epochMetrics: [EpochMetrics],
        finalAccuracy: Float,
        benchmarkComparison: BenchmarkComparison? = nil,
        bestValidationAccuracy: Float? = nil,
        bestEpoch: Int? = nil
    ) {
        self.modelType = modelType
        self.hyperparameters = hyperparameters
        self.epochMetrics = epochMetrics
        self.finalAccuracy = finalAccuracy
        self.benchmarkComparison = benchmarkComparison
        self.bestValidationAccuracy = bestValidationAccuracy
        self.bestEpoch = bestEpoch
    }

    // Custom encoding to include computed properties
    enum CodingKeys: String, CodingKey {
        case modelType
        case hyperparameters
        case epochMetrics
        case finalAccuracy
        case benchmarkComparison
        case bestValidationAccuracy
        case bestEpoch
        case totalTrainingTime
        case averageEpochTime
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(hyperparameters, forKey: .hyperparameters)
        try container.encode(epochMetrics, forKey: .epochMetrics)
        try container.encode(finalAccuracy, forKey: .finalAccuracy)
        try container.encodeIfPresent(benchmarkComparison, forKey: .benchmarkComparison)
        try container.encodeIfPresent(bestValidationAccuracy, forKey: .bestValidationAccuracy)
        try container.encodeIfPresent(bestEpoch, forKey: .bestEpoch)
        try container.encode(totalTrainingTime, forKey: .totalTrainingTime)
        try container.encode(averageEpochTime, forKey: .averageEpochTime)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        hyperparameters = try container.decode(TrainingHyperparameters.self, forKey: .hyperparameters)
        epochMetrics = try container.decode([EpochMetrics].self, forKey: .epochMetrics)
        finalAccuracy = try container.decode(Float.self, forKey: .finalAccuracy)
        benchmarkComparison = try container.decodeIfPresent(BenchmarkComparison.self, forKey: .benchmarkComparison)
        bestValidationAccuracy = try container.decodeIfPresent(Float.self, forKey: .bestValidationAccuracy)
        bestEpoch = try container.decodeIfPresent(Int.self, forKey: .bestEpoch)
        // Note: totalTrainingTime and averageEpochTime are computed from epochMetrics
    }

    // =========================================================================
    // MARK: - JSON Export
    // =========================================================================

    /// Exports the training summary to a JSON file
    /// - Parameter filePath: The path where the JSON file should be saved
    /// - Throws: Encoding or file writing errors
    public func exportToJSON(filePath: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        let jsonData = try encoder.encode(self)
        try jsonData.write(to: URL(fileURLWithPath: filePath))
    }

    // =========================================================================
    // MARK: - Display Methods
    // =========================================================================

    /// Prints a formatted training summary to the console
    public func printSummary() {
        print()
        ColoredPrint.progress("═══════════════════════════════════════════════════════════")
        ColoredPrint.progress("                    Training Summary                        ")
        ColoredPrint.progress("═══════════════════════════════════════════════════════════")
        print()

        // Model Information
        ColoredPrint.info("Model Type:        \(modelType.uppercased())")
        print()

        // Hyperparameters
        ColoredPrint.info("Hyperparameters:")
        ColoredPrint.info("  Epochs:          \(hyperparameters.epochs)")
        ColoredPrint.info("  Batch Size:      \(hyperparameters.batchSize)")
        ColoredPrint.info("  Learning Rate:   \(hyperparameters.learningRate)")
        ColoredPrint.info("  Random Seed:     \(hyperparameters.seed)")
        print()

        // Epoch-by-Epoch Metrics
        ColoredPrint.info("Epoch-by-Epoch Metrics:")
        ColoredPrint.info("  Epoch | Loss     | Time (s)")
        ColoredPrint.info("  ------|----------|----------")
        for metrics in epochMetrics {
            let lossStr = String(format: "%.4f", metrics.loss)
            let timeStr = String(format: "%.2f", metrics.duration)
            ColoredPrint.info(String(format: "  %-5d | %-8s | %s", metrics.epoch, lossStr, timeStr))
        }
        print()

        // Training Summary Statistics
        ColoredPrint.info("Training Statistics:")
        ColoredPrint.info(String(format: "  Total Training Time:   %.2f seconds", totalTrainingTime))
        ColoredPrint.info(String(format: "  Average Epoch Time:    %.2f seconds", averageEpochTime))
        print()

        // Best Model Information (if available)
        if let bestValAcc = bestValidationAccuracy, let bestEp = bestEpoch {
            ColoredPrint.info("Best Model:")
            ColoredPrint.info(String(format: "  Best Epoch:            %d", bestEp))
            ColoredPrint.info(String(format: "  Best Validation Acc:   %.2f%%", bestValAcc * 100))
            print()
        }

        // Final Results
        ColoredPrint.success(String(format: "Final Test Accuracy:   %.2f%%", finalAccuracy * 100))
        print()
    }

    /// Prints a formatted benchmark comparison if available
    public func printBenchmarkComparison() {
        guard let comparison = benchmarkComparison else {
            return
        }

        ColoredPrint.progress("───────────────────────────────────────────────────────────")
        ColoredPrint.progress("                 Benchmark Comparison                       ")
        ColoredPrint.progress("───────────────────────────────────────────────────────────")
        print()

        ColoredPrint.info(String(format: "Expected Accuracy:     %.2f%%", comparison.expectedAccuracy * 100))
        ColoredPrint.info(String(format: "Actual Accuracy:       %.2f%%", comparison.actualAccuracy * 100))

        let diffPercent = comparison.difference * 100
        let diffStr = String(format: "%+.2f%%", diffPercent)

        if comparison.metExpectations {
            ColoredPrint.success(String(format: "Difference:            %s ✓", diffStr))
            ColoredPrint.success("Status:                PASSED - Met or exceeded expectations")
        } else {
            ColoredPrint.warning(String(format: "Difference:            %s ⚠", diffStr))
            ColoredPrint.warning("Status:                WARNING - Below expected accuracy")
        }

        print()
        ColoredPrint.progress("═══════════════════════════════════════════════════════════")
        print()
    }
}
