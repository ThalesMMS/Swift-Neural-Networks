import XCTest
import MLX
import MLXNN
@testable import MNISTMLX

final class CrossEntropyCoreTests: LossFunctionTestSupport {
    // =============================================================================
    // MARK: - Basic Cross-Entropy Tests
    // =============================================================================

    func testCrossEntropyBasicComputation() {
        // Test that cross-entropy loss computes correctly for simple case
        // Single sample with known logits
        let logits = MLXArray([-1.0, 2.0, -0.5], [1, 3])  // Favors class 1
        let labels = MLXArray([Int32(1)])  // Correct class is 1

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        // Loss should be computed
        XCTAssertEqual(loss.size, 1, "Loss should be a scalar")
        assertAllFinite(loss, "Loss should be finite")

        // For correct prediction with moderate confidence, loss should be small but positive
        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThan(lossValue, 0.0, "Loss should be positive")
        XCTAssertLessThan(lossValue, 5.0, "Loss should be reasonable for correct prediction")
    }

    func testCrossEntropyIsPositive() {
        // Test that cross-entropy loss is always positive (or zero for perfect predictions)
        let batchSize = 32
        let logits = createRandomLogits(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThanOrEqual(lossValue, 0.0,
                                   "Cross-entropy loss should always be non-negative")
    }

    func testCrossEntropyIsFinite() {
        // Test that cross-entropy loss produces finite values
        let batchSize = 64
        let logits = createRandomLogits(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)

        assertAllFinite(loss, "Cross-entropy loss should be finite (no NaN or Inf)")
    }

    func testCrossEntropyScalarOutput() {
        // Test that cross-entropy with mean reduction produces scalar output
        let batchSize = 16
        let logits = createRandomLogits(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        XCTAssertEqual(loss.size, 1, "Loss with mean reduction should be a scalar")
        XCTAssertEqual(loss.ndim, 0, "Loss should be 0-dimensional (scalar)")
    }

    // =============================================================================
    // MARK: - Perfect Prediction Tests
    // =============================================================================

    func testCrossEntropyPerfectPredictions() {
        // Test that perfect predictions yield very low loss (near zero)
        // When model predicts correct class with high confidence, loss → 0
        let batchSize = 16
        let labels = createRandomLabels(batchSize: batchSize)
        let logits = createPerfectLogits(labels: labels)

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        let lossValue = loss.item(Float.self)

        // Perfect predictions should give very small loss
        // Loss = -log(p) where p ≈ 1, so loss ≈ 0
        XCTAssertLessThan(lossValue, 0.1,
                         "Perfect predictions should yield very low loss (< 0.1)")
        XCTAssertGreaterThanOrEqual(lossValue, 0.0,
                                   "Loss should still be non-negative")
    }

    func testCrossEntropyPerfectSingleSample() {
        // Test perfect prediction for single sample
        // Logits: [10, -10, -10, ...] with label = 0 should give near-zero loss
        let logits = MLXArray([10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0], [1, 10])
        let labels = MLXArray([Int32(0)])

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        let lossValue = loss.item(Float.self)
        XCTAssertLessThan(lossValue, 0.001,
                         "Perfect single sample prediction should yield near-zero loss")
    }

    // =============================================================================
    // MARK: - Worst Case Prediction Tests
    // =============================================================================

    func testCrossEntropyWorstPredictions() {
        // Test that worst-case predictions yield high loss
        // When model confidently predicts wrong class, loss should be large
        let batchSize = 16
        let labels = createRandomLabels(batchSize: batchSize)
        let logits = createWorstLogits(labels: labels)

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        let lossValue = loss.item(Float.self)

        // Worst predictions should give high loss
        // Loss = -log(p) where p ≈ 0, so loss → ∞ (but capped by numerical precision)
        XCTAssertGreaterThan(lossValue, 5.0,
                            "Worst case predictions should yield high loss (> 5.0)")
    }

    func testCrossEntropyWorstSingleSample() {
        // Test worst prediction for single sample
        // Logits: [-10, 10, -10, ...] with label = 0 (correct is 0, predicted is 1)
        let logits = MLXArray([-10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0], [1, 10])
        let labels = MLXArray([Int32(0)])  // Correct class is 0, but we predict 1

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThan(lossValue, 10.0,
                            "Confidently wrong prediction should yield very high loss")
    }

    // =============================================================================
    // MARK: - Random Prediction Tests
    // =============================================================================

    func testCrossEntropyRandomPredictions() {
        // Test that random predictions yield moderate loss
        // Random guessing should give loss ≈ -log(1/10) ≈ 2.3 for 10 classes
        let batchSize = 100
        let logits = createRandomLogits(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        let lossValue = loss.item(Float.self)

        // For random predictions on 10 classes, expected loss ≈ -log(0.1) ≈ 2.3
        // Allow some variance: should be in range [1.0, 4.0]
        assertInRange(lossValue, 0.5, 5.0,
                     "Random predictions should yield moderate loss (roughly -log(1/num_classes))")
    }

    func testCrossEntropyUniformLogits() {
        // Test loss with uniform logits (all classes equally likely)
        // Uniform distribution over 10 classes gives loss = -log(1/10) ≈ 2.3
        let batchSize = 32
        let logits = MLXArray.zeros([batchSize, 10])  // All logits equal → uniform probabilities
        let labels = createRandomLabels(batchSize: batchSize)

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        let lossValue = loss.item(Float.self)

        // Theoretical loss for uniform distribution: -log(1/10) = log(10) ≈ 2.302
        XCTAssertEqual(lossValue, 2.302, accuracy: 0.1,
                      "Uniform logits should give loss ≈ log(num_classes)")
    }

    // =============================================================================
    // MARK: - Batch Size Variation Tests
    // =============================================================================

    func testCrossEntropyVariousBatchSizes() {
        // Test that cross-entropy works correctly with various batch sizes
        let batchSizes = [1, 2, 4, 8, 16, 32, 64, 128]

        for batchSize in batchSizes {
            let logits = createRandomLogits(batchSize: batchSize)
            let labels = createRandomLabels(batchSize: batchSize)

            let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
            eval(loss)

            // Verify loss is computed and reasonable
            let lossValue = loss.item(Float.self)
            XCTAssertGreaterThan(lossValue, 0.0,
                                "Loss should be positive for batch size \(batchSize)")
            XCTAssertLessThan(lossValue, 20.0,
                             "Loss should be reasonable for batch size \(batchSize)")
            assertAllFinite(loss, "Loss should be finite for batch size \(batchSize)")
        }
    }

    func testCrossEntropySingleSample() {
        // Test cross-entropy with single sample (batch size = 1)
        let logits = createRandomLogits(batchSize: 1)
        let labels = createRandomLabels(batchSize: 1)

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        XCTAssertEqual(loss.size, 1, "Single sample loss should be scalar")
        assertAllFinite(loss, "Single sample loss should be finite")
    }

    func testCrossEntropyLargeBatch() {
        // Test cross-entropy with large batch (simulating full dataset evaluation)
        let batchSize = 1000
        let logits = createRandomLogits(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThan(lossValue, 0.0, "Large batch loss should be positive")
        assertAllFinite(loss, "Large batch loss should be finite")
    }

    // =============================================================================
    // MARK: - MLP Model Loss Function Tests
    // =============================================================================

}
