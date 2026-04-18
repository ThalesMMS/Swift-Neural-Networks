import XCTest
import MLX
import MLXNN
@testable import MNISTMLX

final class CrossEntropyNumericalTests: LossFunctionTestSupport {
    func testCrossEntropyWithLargeLogits() {
        // Test numerical stability with very large logit values
        let batchSize = 16
        let (logits, labels) = createExtremeLogits(batchSize: batchSize, low: -100.0, high: 100.0)

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        assertAllFinite(loss, "Loss should be finite even with large logits")

        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThanOrEqual(lossValue, 0.0, "Loss should be non-negative")
    }

    func testCrossEntropyWithSmallLogits() {
        // Test numerical stability with very small logit values
        let batchSize = 16
        let (logits, labels) = createExtremeLogits(batchSize: batchSize, low: -1_000.0, high: -100.0)

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        assertAllFinite(loss, "Loss should be finite even with very negative logits")

        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThanOrEqual(lossValue, 0.0, "Loss should be non-negative")
    }

    private func createExtremeLogits(batchSize: Int, numClasses: Int = 10, low: Float, high: Float) -> (MLXArray, MLXArray) {
        var values = [Float](repeating: low, count: batchSize * numClasses)
        var labels = [Int32]()
        labels.reserveCapacity(batchSize)

        for row in 0..<batchSize {
            let highClass = row % numClasses
            labels.append(Int32(highClass))
            values[row * numClasses + highClass] = high
        }

        return (MLXArray(values, [batchSize, numClasses]), MLXArray(labels))
    }

    func testCrossEntropyWithMixedLogits() {
        // Test with mixed positive and negative logits (typical scenario)
        let batchSize = 32
        let logits = MLXRandom.uniform(low: -10.0, high: 10.0, [batchSize, 10])
        let labels = createRandomLabels(batchSize: batchSize)

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        assertAllFinite(loss, "Loss should be finite with mixed logits")

        let lossValue = loss.item(Float.self)
        assertInRange(lossValue, 0.0, 30.0,
                     "Loss with mixed logits should be in reasonable range")
    }

    func testCrossEntropyAllCorrectLabels() {
        // Test with all samples having the same label
        let batchSize = 16
        let logits = createRandomLogits(batchSize: batchSize)
        let labels = MLXArray(Array(repeating: Int32(5), count: batchSize))  // All label 5

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        assertAllFinite(loss, "Loss should be finite with uniform labels")

        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThan(lossValue, 0.0, "Loss should be positive")
    }

    // =============================================================================
    // MARK: - Loss Comparison Tests
    // =============================================================================

    func testCrossEntropyPerfectVsRandom() {
        // Test that perfect predictions have lower loss than random predictions
        let batchSize = 32
        let labels = createRandomLabels(batchSize: batchSize)

        let perfectLogits = createPerfectLogits(labels: labels)
        let randomLogits = createRandomLogits(batchSize: batchSize)

        let perfectLoss = crossEntropy(logits: perfectLogits, targets: labels, reduction: .mean)
        let randomLoss = crossEntropy(logits: randomLogits, targets: labels, reduction: .mean)

        eval(perfectLoss, randomLoss)

        let perfectValue = perfectLoss.item(Float.self)
        let randomValue = randomLoss.item(Float.self)

        XCTAssertLessThan(perfectValue, randomValue,
                         "Perfect predictions should have lower loss than random predictions")
    }

    func testCrossEntropyRandomVsWorst() {
        // Test that random predictions have lower loss than worst-case predictions
        let batchSize = 32
        let labels = createRandomLabels(batchSize: batchSize)

        let randomLogits = createRandomLogits(batchSize: batchSize)
        let worstLogits = createWorstLogits(labels: labels)

        let randomLoss = crossEntropy(logits: randomLogits, targets: labels, reduction: .mean)
        let worstLoss = crossEntropy(logits: worstLogits, targets: labels, reduction: .mean)

        eval(randomLoss, worstLoss)

        let randomValue = randomLoss.item(Float.self)
        let worstValue = worstLoss.item(Float.self)

        XCTAssertLessThan(randomValue, worstValue,
                         "Random predictions should have lower loss than worst predictions")
    }

    func testCrossEntropyLossOrdering() {
        // Test the complete ordering: perfect < random < worst
        let batchSize = 32
        let labels = createRandomLabels(batchSize: batchSize)

        let perfectLogits = createPerfectLogits(labels: labels)
        let randomLogits = createRandomLogits(batchSize: batchSize)
        let worstLogits = createWorstLogits(labels: labels)

        let perfectLoss = crossEntropy(logits: perfectLogits, targets: labels, reduction: .mean)
        let randomLoss = crossEntropy(logits: randomLogits, targets: labels, reduction: .mean)
        let worstLoss = crossEntropy(logits: worstLogits, targets: labels, reduction: .mean)

        eval(perfectLoss, randomLoss, worstLoss)

        let perfectValue = perfectLoss.item(Float.self)
        let randomValue = randomLoss.item(Float.self)
        let worstValue = worstLoss.item(Float.self)

        XCTAssertLessThan(perfectValue, randomValue,
                         "Perfect loss < Random loss")
        XCTAssertLessThan(randomValue, worstValue,
                         "Random loss < Worst loss")
        XCTAssertLessThan(perfectValue, worstValue,
                         "Perfect loss < Worst loss")
    }

    // =============================================================================
    // MARK: - Gradient Flow Tests
    // =============================================================================

}
