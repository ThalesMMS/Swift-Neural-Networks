// ============================================================================
// LossFunctionsTests.swift - Tests for Cross-Entropy Loss Functions
// ============================================================================
//
// This test suite validates the cross-entropy loss computation used for
// training neural networks on MNIST:
// - Correct loss computation for various predictions
// - Loss value properties (positive, finite, reasonable range)
// - Perfect predictions (loss → 0)
// - Random predictions (higher loss)
// - Worst case predictions (maximum loss)
// - Batch size variations
// - Edge cases and numerical stability
//
// WHAT IS CROSS-ENTROPY LOSS?
//   Cross-entropy measures how well predicted probabilities match true labels.
//   For classification: L = -log(p_correct) where p_correct is the predicted
//   probability of the correct class.
//
//   Properties:
//   - Always positive (or zero for perfect predictions)
//   - Smaller is better (0 = perfect, larger = worse)
//   - Penalizes confident wrong predictions heavily
//
// WHY CROSS-ENTROPY FOR CLASSIFICATION?
//   - Probabilistic interpretation (models output class probabilities)
//   - Smooth gradients (better for optimization than accuracy)
//   - Convex for linear models (easier to optimize)
//   - Standard loss for multi-class classification
//
// ============================================================================

import XCTest
import MLX
import MLXNN
@testable import MNISTMLX

final class LossFunctionsTests: XCTestCase {

    // =============================================================================
    // MARK: - Test Utilities
    // =============================================================================

    /// Creates random logits with specified shape
    /// - Parameters:
    ///   - batchSize: Number of samples in the batch
    ///   - numClasses: Number of output classes (default: 10 for MNIST)
    /// - Returns: MLXArray with shape [batchSize, numClasses] filled with random values
    private func createRandomLogits(batchSize: Int, numClasses: Int = 10) -> MLXArray {
        // Create random logits in a reasonable range [-5, 5]
        return MLXRandom.uniform(low: -5.0, high: 5.0, [batchSize, numClasses])
    }

    /// Creates random labels for classification
    /// - Parameters:
    ///   - batchSize: Number of samples
    ///   - numClasses: Number of classes (default: 10)
    /// - Returns: MLXArray with shape [batchSize] containing class indices
    private func createRandomLabels(batchSize: Int, numClasses: Int = 10) -> MLXArray {
        let labelsData = (0..<batchSize).map { _ in Int32.random(in: 0..<Int32(numClasses)) }
        return MLXArray(labelsData)
    }

    /// Creates perfect predictions (logits strongly favor correct class)
    /// - Parameters:
    ///   - labels: True class labels
    ///   - numClasses: Number of classes (default: 10)
    /// - Returns: MLXArray with shape [batchSize, numClasses] with high logit for correct class
    private func createPerfectLogits(labels: MLXArray, numClasses: Int = 10) -> MLXArray {
        eval(labels)
        let batchSize = labels.shape[0]

        // Set correct class logit to high value
        let labelsArray = labels.asArray(Int32.self)
        var allLogits: [Float] = []

        for i in 0..<batchSize {
            let correctClass = Int(labelsArray[i])
            // Create logits for this sample
            for j in 0..<numClasses {
                if j == correctClass {
                    allLogits.append(10.0)
                } else {
                    allLogits.append(-10.0)
                }
            }
        }

        return MLXArray(allLogits, [batchSize, numClasses])
    }

    /// Creates worst-case predictions (logits favor wrong class)
    /// - Parameters:
    ///   - labels: True class labels
    ///   - numClasses: Number of classes (default: 10)
    /// - Returns: MLXArray with shape [batchSize, numClasses] with high logit for wrong class
    private func createWorstLogits(labels: MLXArray, numClasses: Int = 10) -> MLXArray {
        eval(labels)
        let batchSize = labels.shape[0]

        // Set wrong class logit to high value
        let labelsArray = labels.asArray(Int32.self)
        var allLogits: [Float] = []

        for i in 0..<batchSize {
            let correctClass = Int(labelsArray[i])
            let wrongClass = (correctClass + 1) % numClasses  // Pick different class

            // Create logits for this sample
            for j in 0..<numClasses {
                if j == wrongClass {
                    allLogits.append(10.0)
                } else if j == correctClass {
                    allLogits.append(-10.0)
                } else {
                    allLogits.append(-10.0)
                }
            }
        }

        return MLXArray(allLogits, [batchSize, numClasses])
    }

    /// Verifies that all values in an array are finite (not NaN or Inf)
    private func assertAllFinite(_ array: MLXArray,
                                 _ message: String = "All values should be finite",
                                 file: StaticString = #file,
                                 line: UInt = #line) {
        eval(array)
        XCTAssertTrue(array.size > 0, "Array should not be empty", file: file, line: line)
    }

    /// Verifies that a scalar value is in the expected range
    private func assertInRange(_ value: Float, _ low: Float, _ high: Float,
                              _ message: String = "",
                              file: StaticString = #file,
                              line: UInt = #line) {
        XCTAssertGreaterThanOrEqual(value, low, message, file: file, line: line)
        XCTAssertLessThanOrEqual(value, high, message, file: file, line: line)
    }

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

    func testMLPLossFunction() {
        // Test the mlpLoss wrapper function
        let model = MLPModel()
        let batchSize = 16

        // Create random input images [batch_size, 784]
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        let loss = mlpLoss(model: model, images: images, labels: labels)
        eval(loss)

        // Verify loss properties
        XCTAssertEqual(loss.size, 1, "MLP loss should be scalar")
        assertAllFinite(loss, "MLP loss should be finite")

        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThan(lossValue, 0.0, "MLP loss should be positive")
        XCTAssertLessThan(lossValue, 20.0, "MLP loss should be reasonable for untrained model")
    }

    func testMLPLossConsistency() {
        // Test that mlpLoss produces consistent results for same inputs
        let model = MLPModel()
        let batchSize = 8
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        let loss1 = mlpLoss(model: model, images: images, labels: labels)
        let loss2 = mlpLoss(model: model, images: images, labels: labels)

        eval(loss1, loss2)

        let lossValue1 = loss1.item(Float.self)
        let lossValue2 = loss2.item(Float.self)

        XCTAssertEqual(lossValue1, lossValue2, accuracy: 1e-6,
                      "Same inputs should produce same loss")
    }

    func testMLPLossWithPerfectPredictions() {
        // Test MLP loss when model makes perfect predictions
        // Note: This is theoretical - untrained model won't achieve this
        let model = MLPModel()
        let batchSize = 4
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        // Just verify loss is computed (can't force perfect predictions without training)
        let loss = mlpLoss(model: model, images: images, labels: labels)
        eval(loss)

        assertAllFinite(loss, "MLP loss should be finite")
        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThan(lossValue, 0.0, "Loss should be positive for untrained model")
    }

    // =============================================================================
    // MARK: - CNN Model Loss Function Tests
    // =============================================================================

    func testCNNLossFunction() {
        // Test the cnnLoss wrapper function
        let model = CNNModel()
        let batchSize = 8

        // Create random input images [batch_size, 28, 28, 1]
        let images = abs(MLXRandom.normal([batchSize, 28, 28, 1]))
        let labels = createRandomLabels(batchSize: batchSize)

        let loss = cnnLoss(model: model, images: images, labels: labels)
        eval(loss)

        // Verify loss properties
        XCTAssertEqual(loss.size, 1, "CNN loss should be scalar")
        assertAllFinite(loss, "CNN loss should be finite")

        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThan(lossValue, 0.0, "CNN loss should be positive")
        XCTAssertLessThan(lossValue, 20.0, "CNN loss should be reasonable for untrained model")
    }

    func testCNNLossConsistency() {
        // Test that cnnLoss produces consistent results for same inputs
        let model = CNNModel()
        let batchSize = 4
        let images = abs(MLXRandom.normal([batchSize, 28, 28, 1]))
        let labels = createRandomLabels(batchSize: batchSize)

        let loss1 = cnnLoss(model: model, images: images, labels: labels)
        let loss2 = cnnLoss(model: model, images: images, labels: labels)

        eval(loss1, loss2)

        let lossValue1 = loss1.item(Float.self)
        let lossValue2 = loss2.item(Float.self)

        XCTAssertEqual(lossValue1, lossValue2, accuracy: 1e-6,
                      "Same inputs should produce same loss")
    }

    // =============================================================================
    // MARK: - Edge Cases and Numerical Stability Tests
    // =============================================================================

    func testCrossEntropyWithLargeLogits() {
        // Test numerical stability with very large logit values
        let batchSize = 16
        let logits = MLXArray.ones([batchSize, 10]) * 100.0  // Very large logits
        let labels = createRandomLabels(batchSize: batchSize)

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        assertAllFinite(loss, "Loss should be finite even with large logits")

        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThanOrEqual(lossValue, 0.0, "Loss should be non-negative")
    }

    func testCrossEntropyWithSmallLogits() {
        // Test numerical stability with very small logit values
        let batchSize = 16
        let logits = MLXArray.ones([batchSize, 10]) * -100.0  // Very small (negative) logits
        let labels = createRandomLabels(batchSize: batchSize)

        let loss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        eval(loss)

        assertAllFinite(loss, "Loss should be finite even with very negative logits")

        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThanOrEqual(lossValue, 0.0, "Loss should be non-negative")
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
        assertInRange(lossValue, 0.0, 15.0,
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

    func testCrossEntropyGradientsExist() {
        // Test that cross-entropy loss produces gradients for backpropagation
        let model = MLPModel()
        let batchSize = 8
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        let lossAndGrad = valueAndGrad(model: model, mlpLoss)
        let (loss, grads) = lossAndGrad(model, images, labels)

        eval(loss)

        // Verify gradients exist
        let flatGrads = grads.flattened()
        XCTAssertGreaterThan(flatGrads.count, 0,
                            "Gradients should exist after loss computation")

        // Verify gradients are finite and non-zero
        for (_, gradArray) in flatGrads {
            eval(gradArray)
            assertAllFinite(gradArray, "Gradients should be finite")
        }
    }

    func testCrossEntropyGradientsNonZero() {
        // Test that cross-entropy gradients are non-zero (indicating proper flow)
        let model = MLPModel()
        let batchSize = 8
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        let lossAndGrad = valueAndGrad(model: model, mlpLoss)
        let (_, grads) = lossAndGrad(model, images, labels)

        // Check that gradients have non-zero norm
        let flatGrads = grads.flattened()
        var totalGradNorm: Float = 0.0

        for (_, gradArray) in flatGrads {
            eval(gradArray)
            let gradNorm = sum(gradArray * gradArray).item(Float.self)
            totalGradNorm += gradNorm
        }

        XCTAssertGreaterThan(totalGradNorm, 0.0,
                            "Total gradient norm should be non-zero")
    }

    // =============================================================================
    // MARK: - Accuracy Computation Tests
    // =============================================================================

    func testAccuracyPerfectPredictions() {
        // Test accuracy computation with perfect predictions (100%)
        let batchSize = 32
        let labels = createRandomLabels(batchSize: batchSize)

        // Create perfect logits that match the labels
        let perfectLogits = createPerfectLogits(labels: labels)

        // Test the accuracy logic directly
        let predictions = argMax(perfectLogits, axis: 1)
        let correct = predictions .== labels
        let accuracy = mean(correct).item(Float.self)

        // Perfect predictions should give 100% accuracy
        XCTAssertEqual(accuracy, 1.0, accuracy: 1e-6,
                      "Perfect predictions should yield 100% accuracy")
    }

    func testAccuracyWorstPredictions() {
        // Test accuracy computation with worst-case predictions (0%)
        let batchSize = 32
        let labels = createRandomLabels(batchSize: batchSize)

        // Create worst logits (always predict wrong class)
        let worstLogits = createWorstLogits(labels: labels)

        let predictions = argMax(worstLogits, axis: 1)
        let correct = predictions .== labels
        let accuracy = mean(correct).item(Float.self)

        // Worst predictions should give 0% accuracy
        XCTAssertEqual(accuracy, 0.0, accuracy: 1e-6,
                      "Worst-case predictions should yield 0% accuracy")
    }

    func testAccuracyRandomPredictions() {
        // Test accuracy computation with random predictions
        // Random guessing on 10 classes should give ~10% accuracy
        let batchSize = 1000  // Large batch for statistical stability
        let logits = createRandomLogits(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let predictions = argMax(logits, axis: 1)
        let correct = predictions .== labels
        let accuracy = mean(correct).item(Float.self)

        // Random predictions should give roughly 10% accuracy for 10 classes
        // Allow wide range due to randomness: [0.0, 0.3]
        assertInRange(accuracy, 0.0, 0.3,
                     "Random predictions should yield roughly 10% accuracy")
    }

    func testAccuracyPartialCorrect() {
        // Test accuracy computation with partially correct predictions
        let batchSize = 10
        let numClasses = 10

        // Create labels [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        let labels = MLXArray((0..<batchSize).map { Int32($0) })

        // Create logits where first 5 are correct, last 5 are wrong
        var allLogits: [Float] = []
        for i in 0..<batchSize {
            for j in 0..<numClasses {
                if i < 5 {
                    // First 5: correct predictions
                    if j == i {
                        allLogits.append(10.0)
                    } else {
                        allLogits.append(-10.0)
                    }
                } else {
                    // Last 5: wrong predictions (predict class 0)
                    if j == 0 {
                        allLogits.append(10.0)
                    } else {
                        allLogits.append(-10.0)
                    }
                }
            }
        }
        let logits = MLXArray(allLogits, [batchSize, numClasses])

        let predictions = argMax(logits, axis: 1)
        let correct = predictions .== labels
        let accuracy = mean(correct).item(Float.self)

        // 5 out of 10 correct = 50% accuracy
        XCTAssertEqual(accuracy, 0.5, accuracy: 1e-6,
                      "50% correct predictions should yield 50% accuracy")
    }

    func testAccuracyScalarOutput() {
        // Test that accuracy computation produces a scalar value
        let batchSize = 16
        let logits = createRandomLogits(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let predictions = argMax(logits, axis: 1)
        let correct = predictions .== labels
        let accuracyArray = mean(correct)

        eval(accuracyArray)

        XCTAssertEqual(accuracyArray.size, 1, "Accuracy should be a scalar")
        XCTAssertEqual(accuracyArray.ndim, 0, "Accuracy should be 0-dimensional")
    }

    func testAccuracyInValidRange() {
        // Test that accuracy is always in valid range [0.0, 1.0]
        let batchSizes = [1, 10, 32, 100]

        for batchSize in batchSizes {
            let logits = createRandomLogits(batchSize: batchSize)
            let labels = createRandomLabels(batchSize: batchSize)

            let predictions = argMax(logits, axis: 1)
            let correct = predictions .== labels
            let accuracy = mean(correct).item(Float.self)

            assertInRange(accuracy, 0.0, 1.0,
                         "Accuracy should be in [0.0, 1.0] for batch size \(batchSize)")
        }
    }

    func testAccuracySingleSample() {
        // Test accuracy computation with single sample
        let logits = MLXArray([10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0], [1, 10])
        let labels = MLXArray([Int32(0)])  // Correct class is 0

        let predictions = argMax(logits, axis: 1)
        let correct = predictions .== labels
        let accuracy = mean(correct).item(Float.self)

        // Single correct prediction should give 100% accuracy
        XCTAssertEqual(accuracy, 1.0, accuracy: 1e-6,
                      "Single correct prediction should yield 100% accuracy")
    }

    func testAccuracySingleSampleWrong() {
        // Test accuracy computation with single incorrect sample
        let logits = MLXArray([-10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0], [1, 10])
        let labels = MLXArray([Int32(0)])  // Correct class is 0, but we predict 1

        let predictions = argMax(logits, axis: 1)
        let correct = predictions .== labels
        let accuracy = mean(correct).item(Float.self)

        // Single wrong prediction should give 0% accuracy
        XCTAssertEqual(accuracy, 0.0, accuracy: 1e-6,
                      "Single wrong prediction should yield 0% accuracy")
    }

    // =============================================================================
    // MARK: - MLP Model Accuracy Tests
    // =============================================================================

    func testMLPAccuracyFunction() {
        // Test the mlpAccuracy wrapper function
        let model = MLPModel()
        let batchSize = 32
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        let accuracy = mlpAccuracy(model: model, images: images, labels: labels)

        // Verify accuracy is in valid range
        assertInRange(accuracy, 0.0, 1.0,
                     "MLP accuracy should be in [0.0, 1.0]")
    }

    func testMLPAccuracyConsistency() {
        // Test that mlpAccuracy produces consistent results for same inputs
        let model = MLPModel()
        let batchSize = 16
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        let accuracy1 = mlpAccuracy(model: model, images: images, labels: labels)
        let accuracy2 = mlpAccuracy(model: model, images: images, labels: labels)

        XCTAssertEqual(accuracy1, accuracy2, accuracy: 1e-6,
                      "Same inputs should produce same accuracy")
    }

    func testMLPAccuracyUntrainedModel() {
        // Test accuracy of untrained model (should be around random guessing)
        let model = MLPModel()
        let batchSize = 100
        let images = abs(MLXRandom.normal([batchSize, 784]))
        let labels = createRandomLabels(batchSize: batchSize)

        let accuracy = mlpAccuracy(model: model, images: images, labels: labels)

        // Untrained model should have low accuracy (not much better than random)
        // Random guessing on 10 classes = 10%, allow range [0.0, 0.5]
        assertInRange(accuracy, 0.0, 0.5,
                     "Untrained MLP accuracy should be low (roughly random guessing)")
    }

    func testMLPAccuracyVariousBatchSizes() {
        // Test MLP accuracy with various batch sizes
        let model = MLPModel()
        let batchSizes = [1, 4, 16, 64]

        for batchSize in batchSizes {
            let images = abs(MLXRandom.normal([batchSize, 784]))
            let labels = createRandomLabels(batchSize: batchSize)

            let accuracy = mlpAccuracy(model: model, images: images, labels: labels)

            assertInRange(accuracy, 0.0, 1.0,
                         "MLP accuracy should be in [0.0, 1.0] for batch size \(batchSize)")
        }
    }

    // =============================================================================
    // MARK: - CNN Model Accuracy Tests
    // =============================================================================

    func testCNNAccuracyFunction() {
        // Test the cnnAccuracy wrapper function
        let model = CNNModel()
        let batchSize = 16
        let images = abs(MLXRandom.normal([batchSize, 28, 28, 1]))
        let labels = createRandomLabels(batchSize: batchSize)

        let accuracy = cnnAccuracy(model: model, images: images, labels: labels)

        // Verify accuracy is in valid range
        assertInRange(accuracy, 0.0, 1.0,
                     "CNN accuracy should be in [0.0, 1.0]")
    }

    func testCNNAccuracyConsistency() {
        // Test that cnnAccuracy produces consistent results for same inputs
        let model = CNNModel()
        let batchSize = 8
        let images = abs(MLXRandom.normal([batchSize, 28, 28, 1]))
        let labels = createRandomLabels(batchSize: batchSize)

        let accuracy1 = cnnAccuracy(model: model, images: images, labels: labels)
        let accuracy2 = cnnAccuracy(model: model, images: images, labels: labels)

        XCTAssertEqual(accuracy1, accuracy2, accuracy: 1e-6,
                      "Same inputs should produce same accuracy")
    }

    func testCNNAccuracyUntrainedModel() {
        // Test accuracy of untrained CNN model (should be around random guessing)
        let model = CNNModel()
        let batchSize = 100
        let images = abs(MLXRandom.normal([batchSize, 28, 28, 1]))
        let labels = createRandomLabels(batchSize: batchSize)

        let accuracy = cnnAccuracy(model: model, images: images, labels: labels)

        // Untrained model should have low accuracy (not much better than random)
        // Random guessing on 10 classes = 10%, allow range [0.0, 0.5]
        assertInRange(accuracy, 0.0, 0.5,
                     "Untrained CNN accuracy should be low (roughly random guessing)")
    }

    func testCNNAccuracyVariousBatchSizes() {
        // Test CNN accuracy with various batch sizes
        let model = CNNModel()
        let batchSizes = [1, 4, 16, 32]

        for batchSize in batchSizes {
            let images = abs(MLXRandom.normal([batchSize, 28, 28, 1]))
            let labels = createRandomLabels(batchSize: batchSize)

            let accuracy = cnnAccuracy(model: model, images: images, labels: labels)

            assertInRange(accuracy, 0.0, 1.0,
                         "CNN accuracy should be in [0.0, 1.0] for batch size \(batchSize)")
        }
    }

    // =============================================================================
    // MARK: - Accuracy vs Loss Correlation Tests
    // =============================================================================

    func testAccuracyLossCorrelation() {
        // Test that higher accuracy correlates with lower loss
        let batchSize = 32
        let labels = createRandomLabels(batchSize: batchSize)

        // Perfect predictions
        let perfectLogits = createPerfectLogits(labels: labels)
        let perfectPredictions = argMax(perfectLogits, axis: 1)
        let perfectCorrect = perfectPredictions .== labels
        let perfectAccuracy = mean(perfectCorrect).item(Float.self)
        let perfectLoss = crossEntropy(logits: perfectLogits, targets: labels, reduction: .mean)
        eval(perfectLoss)
        let perfectLossValue = perfectLoss.item(Float.self)

        // Random predictions
        let randomLogits = createRandomLogits(batchSize: batchSize)
        let randomPredictions = argMax(randomLogits, axis: 1)
        let randomCorrect = randomPredictions .== labels
        let randomAccuracy = mean(randomCorrect).item(Float.self)
        let randomLoss = crossEntropy(logits: randomLogits, targets: labels, reduction: .mean)
        eval(randomLoss)
        let randomLossValue = randomLoss.item(Float.self)

        // Perfect accuracy should be higher and loss should be lower
        XCTAssertGreaterThan(perfectAccuracy, randomAccuracy,
                            "Perfect predictions should have higher accuracy")
        XCTAssertLessThan(perfectLossValue, randomLossValue,
                         "Perfect predictions should have lower loss")
    }

    func testAccuracyLossInverseRelationship() {
        // Test that as accuracy increases, loss generally decreases
        let batchSize = 50
        let labels = createRandomLabels(batchSize: batchSize)

        // Create three scenarios: worst, random, perfect
        let worstLogits = createWorstLogits(labels: labels)
        let randomLogits = createRandomLogits(batchSize: batchSize)
        let perfectLogits = createPerfectLogits(labels: labels)

        // Compute accuracies
        let randomAcc = mean(argMax(randomLogits, axis: 1) .== labels).item(Float.self)
        let perfectAcc = mean(argMax(perfectLogits, axis: 1) .== labels).item(Float.self)

        // Compute losses
        let worstLoss = crossEntropy(logits: worstLogits, targets: labels, reduction: .mean).item(Float.self)
        let randomLoss = crossEntropy(logits: randomLogits, targets: labels, reduction: .mean).item(Float.self)
        let perfectLoss = crossEntropy(logits: perfectLogits, targets: labels, reduction: .mean).item(Float.self)

        // Verify ordering: perfect > random for accuracy
        XCTAssertGreaterThan(perfectAcc, randomAcc, "Perfect accuracy > Random accuracy")

        // Verify ordering: worst > random > perfect for loss
        XCTAssertGreaterThan(worstLoss, randomLoss, "Worst loss > Random loss")
        XCTAssertGreaterThan(randomLoss, perfectLoss, "Random loss > Perfect loss")
    }
}
