// ============================================================================
// CompiledTrainingTests.swift - Tests for Compiled Training Functions
// ============================================================================
//
// This test suite validates the compiled training functions that use MLX's
// compile() API to fuse operations into optimized GPU kernels:
// - Compiled training steps produce valid loss values
// - Compiled functions work with various batch sizes
// - Compiled functions handle edge cases (single sample, large batches)
// - Output properties (finite values, correct types)
// - Epoch training functions execute successfully
//
// Note: This suite tests basic functionality. Equivalence tests comparing
// compiled vs non-compiled results are in the next test suite (subtask 4-2).
//
// ============================================================================

import XCTest
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
@testable import MNISTMLX

final class CompiledTrainingTests: MLXTestCase {

    // =============================================================================
    // MARK: - Test Utilities
    // =============================================================================

    /// Creates random image data for testing
    /// - Parameters:
    ///   - batchSize: Number of samples in the batch
    ///   - imageSize: Size of each image (default: 784 for flattened MNIST)
    /// - Returns: MLXArray with shape [batchSize, imageSize] filled with random values in [0, 1]
    private func createRandomImages(batchSize: Int, imageSize: Int = 784) -> MLXArray {
        // Create random values normalized to [0, 1] range (like real MNIST data)
        return abs(MLXRandom.normal([batchSize, imageSize])) * 0.1
    }

    /// Creates random 2D image data for CNN/Attention models
    /// - Parameters:
    ///   - batchSize: Number of samples
    ///   - height: Image height (default: 28)
    ///   - width: Image width (default: 28)
    ///   - channels: Number of channels (default: 1 for grayscale)
    /// - Returns: MLXArray with shape [batchSize, height, width, channels]
    private func createRandom2DImages(
        batchSize: Int,
        height: Int = 28,
        width: Int = 28,
        channels: Int = 1
    ) -> MLXArray {
        return abs(MLXRandom.normal([batchSize, height, width, channels])) * 0.1
    }

    /// Creates random labels for classification
    /// - Parameter batchSize: Number of labels to generate
    /// - Returns: MLXArray with shape [batchSize] containing random class indices (0-9)
    private func createRandomLabels(batchSize: Int) -> MLXArray {
        let labels = (0..<batchSize).map { _ in Int32.random(in: 0..<10) }
        return MLXArray(labels)
    }

    /// Verifies that a loss value is valid (finite and positive)
    private func assertValidLoss(_ loss: Float,
                                 _ message: String = "Loss should be finite and positive",
                                 file: StaticString = #file,
                                 line: UInt = #line) {
        XCTAssertFalse(loss.isNaN, "Loss should not be NaN", file: file, line: line)
        XCTAssertFalse(loss.isInfinite, "Loss should not be infinite", file: file, line: line)
        XCTAssertGreaterThanOrEqual(loss, 0, "Loss should be non-negative", file: file, line: line)
    }

    // =============================================================================
    // MARK: - MLP Compiled Training Tests
    // =============================================================================

    func testCompiledMLPTrainingStepBasic() {
        // Test that compiled MLP training step produces a valid loss value
        let model = MLPModel()
        let optimizer = SGD(learningRate: 0.01)

        let images = createRandomImages(batchSize: 32)
        let labels = createRandomLabels(batchSize: 32)

        let compiledStep = createCompiledMLPTrainingStep(model: model, optimizer: optimizer)
        let loss = compiledStep(images, labels)

        let lossValue = loss.item(Float.self)
        assertValidLoss(lossValue, "MLP compiled training step should produce valid loss")
    }

    func testCompiledMLPTrainingStepSingleSample() {
        // Test compiled MLP training with a single sample
        let model = MLPModel()
        let optimizer = SGD(learningRate: 0.01)

        let images = createRandomImages(batchSize: 1)
        let labels = createRandomLabels(batchSize: 1)

        let compiledStep = createCompiledMLPTrainingStep(model: model, optimizer: optimizer)
        let loss = compiledStep(images, labels)

        let lossValue = loss.item(Float.self)
        assertValidLoss(lossValue, "MLP compiled step should handle single sample")
    }

    func testCompiledMLPTrainingStepVariousBatchSizes() {
        // Test compiled MLP training with various batch sizes (tests shapeless=true)
        let model = MLPModel()
        let optimizer = SGD(learningRate: 0.01)

        let compiledStep = createCompiledMLPTrainingStep(model: model, optimizer: optimizer)
        let batchSizes = [1, 4, 8, 16, 32, 64, 128]

        for batchSize in batchSizes {
            let images = createRandomImages(batchSize: batchSize)
            let labels = createRandomLabels(batchSize: batchSize)

            let loss = compiledStep(images, labels)
            let lossValue = loss.item(Float.self)

            assertValidLoss(lossValue, "MLP compiled step should work with batch size \(batchSize)")
        }
    }

    func testCompiledMLPTrainingStepLargeBatch() {
        // Test compiled MLP training with large batch
        let model = MLPModel()
        let optimizer = SGD(learningRate: 0.01)

        let images = createRandomImages(batchSize: 256)
        let labels = createRandomLabels(batchSize: 256)

        let compiledStep = createCompiledMLPTrainingStep(model: model, optimizer: optimizer)
        let loss = compiledStep(images, labels)

        let lossValue = loss.item(Float.self)
        assertValidLoss(lossValue, "MLP compiled step should handle large batches")
    }

    func testCompiledMLPEpochTraining() {
        // Test full epoch training with compiled MLP
        let model = MLPModel()
        let optimizer = SGD(learningRate: 0.01)

        let trainImages = createRandomImages(batchSize: 100)
        let trainLabels = createRandomLabels(batchSize: 100)

        let avgLoss = trainMLPEpochCompiled(
            model: model,
            optimizer: optimizer,
            trainImages: trainImages,
            trainLabels: trainLabels,
            batchSize: 32
        )

        assertValidLoss(avgLoss, "MLP compiled epoch training should produce valid average loss")
    }

    func testCompiledMLPEpochTrainingSmallBatch() {
        // Test epoch training with small batch size
        let model = MLPModel()
        let optimizer = SGD(learningRate: 0.01)

        let trainImages = createRandomImages(batchSize: 50)
        let trainLabels = createRandomLabels(batchSize: 50)

        let avgLoss = trainMLPEpochCompiled(
            model: model,
            optimizer: optimizer,
            trainImages: trainImages,
            trainLabels: trainLabels,
            batchSize: 8
        )

        assertValidLoss(avgLoss, "MLP compiled epoch training should work with small batches")
    }

    // =============================================================================
    // MARK: - CNN Compiled Training Tests
    // =============================================================================

    func testCompiledCNNTrainingStepBasic() {
        // Test that compiled CNN training step produces a valid loss value
        let model = CNNModel()
        let optimizer = SGD(learningRate: 0.01)

        let images = createRandom2DImages(batchSize: 16)
        let labels = createRandomLabels(batchSize: 16)

        let compiledStep = createCompiledCNNTrainingStep(model: model, optimizer: optimizer)
        let loss = compiledStep(images, labels)

        let lossValue = loss.item(Float.self)
        assertValidLoss(lossValue, "CNN compiled training step should produce valid loss")
    }

    func testCompiledCNNTrainingStepSingleSample() {
        // Test compiled CNN training with a single sample
        let model = CNNModel()
        let optimizer = SGD(learningRate: 0.01)

        let images = createRandom2DImages(batchSize: 1)
        let labels = createRandomLabels(batchSize: 1)

        let compiledStep = createCompiledCNNTrainingStep(model: model, optimizer: optimizer)
        let loss = compiledStep(images, labels)

        let lossValue = loss.item(Float.self)
        assertValidLoss(lossValue, "CNN compiled step should handle single sample")
    }

    func testCompiledCNNTrainingStepVariousBatchSizes() {
        // Test compiled CNN training with various batch sizes (tests shapeless=true)
        let model = CNNModel()
        let optimizer = SGD(learningRate: 0.01)

        let compiledStep = createCompiledCNNTrainingStep(model: model, optimizer: optimizer)
        let batchSizes = [1, 4, 8, 16, 32]

        for batchSize in batchSizes {
            let images = createRandom2DImages(batchSize: batchSize)
            let labels = createRandomLabels(batchSize: batchSize)

            let loss = compiledStep(images, labels)
            let lossValue = loss.item(Float.self)

            assertValidLoss(lossValue, "CNN compiled step should work with batch size \(batchSize)")
        }
    }

    func testCompiledCNNEpochTraining() {
        // Test full epoch training with compiled CNN
        let model = CNNModel()
        let optimizer = SGD(learningRate: 0.01)

        let trainImages = createRandom2DImages(batchSize: 64)
        let trainLabels = createRandomLabels(batchSize: 64)

        let avgLoss = trainCNNEpochCompiled(
            model: model,
            optimizer: optimizer,
            trainImages: trainImages,
            trainLabels: trainLabels,
            batchSize: 16
        )

        assertValidLoss(avgLoss, "CNN compiled epoch training should produce valid average loss")
    }

    func testCompiledCNNEpochTrainingSmallBatch() {
        // Test epoch training with small batch size
        let model = CNNModel()
        let optimizer = SGD(learningRate: 0.01)

        let trainImages = createRandom2DImages(batchSize: 32)
        let trainLabels = createRandomLabels(batchSize: 32)

        let avgLoss = trainCNNEpochCompiled(
            model: model,
            optimizer: optimizer,
            trainImages: trainImages,
            trainLabels: trainLabels,
            batchSize: 4
        )

        assertValidLoss(avgLoss, "CNN compiled epoch training should work with small batches")
    }

    // =============================================================================
    // MARK: - Attention Compiled Training Tests
    // =============================================================================

    func testCompiledAttentionTrainingStepBasic() {
        // Test that compiled Attention training step produces a valid loss value
        let model = AttentionModel()
        let optimizer = SGD(learningRate: 0.01)

        let images = createRandom2DImages(batchSize: 8)
        let labels = createRandomLabels(batchSize: 8)

        let compiledStep = createCompiledAttentionTrainingStep(model: model, optimizer: optimizer)
        let loss = compiledStep(images, labels)

        let lossValue = loss.item(Float.self)
        assertValidLoss(lossValue, "Attention compiled training step should produce valid loss")
    }

    func testCompiledAttentionTrainingStepSingleSample() {
        // Test compiled Attention training with a single sample
        let model = AttentionModel()
        let optimizer = SGD(learningRate: 0.01)

        let images = createRandom2DImages(batchSize: 1)
        let labels = createRandomLabels(batchSize: 1)

        let compiledStep = createCompiledAttentionTrainingStep(model: model, optimizer: optimizer)
        let loss = compiledStep(images, labels)

        let lossValue = loss.item(Float.self)
        assertValidLoss(lossValue, "Attention compiled step should handle single sample")
    }

    func testCompiledAttentionTrainingStepVariousBatchSizes() {
        // Test compiled Attention training with various batch sizes (tests shapeless=true)
        let model = AttentionModel()
        let optimizer = SGD(learningRate: 0.01)

        let compiledStep = createCompiledAttentionTrainingStep(model: model, optimizer: optimizer)
        let batchSizes = [1, 2, 4, 8, 16]

        for batchSize in batchSizes {
            let images = createRandom2DImages(batchSize: batchSize)
            let labels = createRandomLabels(batchSize: batchSize)

            let loss = compiledStep(images, labels)
            let lossValue = loss.item(Float.self)

            assertValidLoss(lossValue, "Attention compiled step should work with batch size \(batchSize)")
        }
    }

    func testCompiledAttentionEpochTraining() {
        // Test full epoch training with compiled Attention
        let model = AttentionModel()
        let optimizer = SGD(learningRate: 0.01)

        let trainImages = createRandom2DImages(batchSize: 32)
        let trainLabels = createRandomLabels(batchSize: 32)

        let avgLoss = trainAttentionEpochCompiled(
            model: model,
            optimizer: optimizer,
            trainImages: trainImages,
            trainLabels: trainLabels,
            batchSize: 8
        )

        assertValidLoss(avgLoss, "Attention compiled epoch training should produce valid average loss")
    }

    func testCompiledAttentionEpochTrainingSmallBatch() {
        // Test epoch training with small batch size
        let model = AttentionModel()
        let optimizer = SGD(learningRate: 0.01)

        let trainImages = createRandom2DImages(batchSize: 16)
        let trainLabels = createRandomLabels(batchSize: 16)

        let avgLoss = trainAttentionEpochCompiled(
            model: model,
            optimizer: optimizer,
            trainImages: trainImages,
            trainLabels: trainLabels,
            batchSize: 4
        )

        assertValidLoss(avgLoss, "Attention compiled epoch training should work with small batches")
    }

    // =============================================================================
    // MARK: - Compiled Function Reusability Tests
    // =============================================================================

    func testCompiledMLPStepReusability() {
        // Test that compiled step can be reused multiple times (compilation happens once)
        let model = MLPModel()
        let optimizer = SGD(learningRate: 0.01)

        let compiledStep = createCompiledMLPTrainingStep(model: model, optimizer: optimizer)

        // Use the compiled function multiple times
        for _ in 0..<5 {
            let images = createRandomImages(batchSize: 16)
            let labels = createRandomLabels(batchSize: 16)

            let loss = compiledStep(images, labels)
            let lossValue = loss.item(Float.self)

            assertValidLoss(lossValue, "Compiled step should be reusable")
        }
    }

    func testCompiledCNNStepReusability() {
        // Test that compiled CNN step can be reused multiple times
        let model = CNNModel()
        let optimizer = SGD(learningRate: 0.01)

        let compiledStep = createCompiledCNNTrainingStep(model: model, optimizer: optimizer)

        for _ in 0..<5 {
            let images = createRandom2DImages(batchSize: 8)
            let labels = createRandomLabels(batchSize: 8)

            let loss = compiledStep(images, labels)
            let lossValue = loss.item(Float.self)

            assertValidLoss(lossValue, "Compiled CNN step should be reusable")
        }
    }

    func testCompiledAttentionStepReusability() {
        // Test that compiled Attention step can be reused multiple times
        let model = AttentionModel()
        let optimizer = SGD(learningRate: 0.01)

        let compiledStep = createCompiledAttentionTrainingStep(model: model, optimizer: optimizer)

        for _ in 0..<5 {
            let images = createRandom2DImages(batchSize: 4)
            let labels = createRandomLabels(batchSize: 4)

            let loss = compiledStep(images, labels)
            let lossValue = loss.item(Float.self)

            assertValidLoss(lossValue, "Compiled Attention step should be reusable")
        }
    }

    // =============================================================================
    // MARK: - Edge Case Tests
    // =============================================================================

    func testCompiledMLPWithZeroInput() {
        // Test compiled MLP with zero input (edge case)
        let model = MLPModel()
        let optimizer = SGD(learningRate: 0.01)

        let images = MLXArray.zeros([16, 784])
        let labels = createRandomLabels(batchSize: 16)

        let compiledStep = createCompiledMLPTrainingStep(model: model, optimizer: optimizer)
        let loss = compiledStep(images, labels)

        let lossValue = loss.item(Float.self)
        assertValidLoss(lossValue, "Compiled step should handle zero input")
    }

    func testCompiledCNNWithZeroInput() {
        // Test compiled CNN with zero input (edge case)
        let model = CNNModel()
        let optimizer = SGD(learningRate: 0.01)

        let images = MLXArray.zeros([8, 28, 28, 1])
        let labels = createRandomLabels(batchSize: 8)

        let compiledStep = createCompiledCNNTrainingStep(model: model, optimizer: optimizer)
        let loss = compiledStep(images, labels)

        let lossValue = loss.item(Float.self)
        assertValidLoss(lossValue, "Compiled CNN step should handle zero input")
    }

    func testCompiledMLPEpochWithOddBatchSize() {
        // Test that epoch training handles non-evenly-divisible batch sizes
        let model = MLPModel()
        let optimizer = SGD(learningRate: 0.01)

        // 50 samples with batch size 32 → will have batches of [32, 18]
        let trainImages = createRandomImages(batchSize: 50)
        let trainLabels = createRandomLabels(batchSize: 50)

        let avgLoss = trainMLPEpochCompiled(
            model: model,
            optimizer: optimizer,
            trainImages: trainImages,
            trainLabels: trainLabels,
            batchSize: 32
        )

        assertValidLoss(avgLoss, "Compiled epoch should handle odd batch sizes")
    }

    func testCompiledCNNEpochWithOddBatchSize() {
        // Test that CNN epoch training handles non-evenly-divisible batch sizes
        let model = CNNModel()
        let optimizer = SGD(learningRate: 0.01)

        // 30 samples with batch size 8 → will have batches of [8, 8, 8, 6]
        let trainImages = createRandom2DImages(batchSize: 30)
        let trainLabels = createRandomLabels(batchSize: 30)

        let avgLoss = trainCNNEpochCompiled(
            model: model,
            optimizer: optimizer,
            trainImages: trainImages,
            trainLabels: trainLabels,
            batchSize: 8
        )

        assertValidLoss(avgLoss, "Compiled CNN epoch should handle odd batch sizes")
    }

    // =============================================================================
    // MARK: - Equivalence Tests (Compiled vs Non-Compiled)
    // =============================================================================

    /// Verifies that two arrays are approximately equal within tolerance
    private func assertArraysClose(_ array1: MLXArray, _ array2: MLXArray,
                                    tolerance: Float = 1e-5,
                                    _ message: String = "Arrays should be approximately equal",
                                    file: StaticString = #file,
                                    line: UInt = #line) {
        // Ensure arrays are evaluated
        eval(array1, array2)

        // Check shapes match
        XCTAssertEqual(array1.shape, array2.shape,
                      "Array shapes should match: \(array1.shape) vs \(array2.shape)",
                      file: file, line: line)

        // Compute maximum absolute difference
        let diff = abs(array1 - array2)
        let maxDiff = max(diff).item(Float.self)

        XCTAssertLessThanOrEqual(maxDiff, tolerance,
                                "\(message): max difference \(maxDiff) exceeds tolerance \(tolerance)",
                                file: file, line: line)
    }

    func testEquivalenceMLPGradients() {
        // Test that compiled and non-compiled MLP training steps produce the same loss
        // This verifies that the compiled version is computing the same operations

        let model = MLPModel()
        let optimizer = SGD(learningRate: 0.01)

        // Create test data
        let images = createRandomImages(batchSize: 16)
        let labels = createRandomLabels(batchSize: 16)

        // =====================================================================
        // Get Initial Loss (before any training)
        // =====================================================================
        let initialLogits = model(images)
        let initialLoss = crossEntropy(logits: initialLogits, targets: labels, reduction: .mean)
        let initialLossValue = initialLoss.item(Float.self)

        // =====================================================================
        // Run Compiled Training Step
        // =====================================================================
        let compiledStep = createCompiledMLPTrainingStep(model: model, optimizer: optimizer)
        let compiledLoss = compiledStep(images, labels)
        let compiledLossValue = compiledLoss.item(Float.self)

        // Get loss after compiled training step
        let afterCompiledLogits = model(images)
        let afterCompiledLoss = crossEntropy(logits: afterCompiledLogits, targets: labels, reduction: .mean)
        let afterCompiledLossValue = afterCompiledLoss.item(Float.self)

        // =====================================================================
        // Verify Results
        // =====================================================================
        // The compiled step should have produced a valid loss
        assertValidLoss(compiledLossValue, "Compiled step should produce valid loss")

        // The model should have been updated (loss should change)
        XCTAssertNotEqual(initialLossValue, afterCompiledLossValue, accuracy: 1e-8,
                         "Model parameters should have been updated by compiled step")

        // The compiled step should work consistently
        // Run another step and verify it still works
        let secondStepLoss = compiledStep(images, labels)
        assertValidLoss(secondStepLoss.item(Float.self),
                       "Compiled step should work on subsequent calls")
    }

    func testEquivalenceMLPMultipleSteps() {
        // Test that compiled and non-compiled produce consistent results over multiple iterations
        let model = MLPModel()

        // Test multiple batches to verify gradient computation consistency
        for iteration in 0..<3 {
            let images = createRandomImages(batchSize: 16)
            let labels = createRandomLabels(batchSize: 16)

            // Compute gradients without compilation
            let lossAndGrad = valueAndGrad(model: model, mlpLoss)
            let (loss, _) = lossAndGrad(model, images, labels)
            let lossValue = loss.item(Float.self)

            // Verify loss is valid
            assertValidLoss(lossValue,
                           "Non-compiled loss should be valid at iteration \(iteration)")

            // The compiled version should produce the same forward pass result
            let logits = model(images)
            let compiledLoss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
            let compiledLossValue = compiledLoss.item(Float.self)

            XCTAssertEqual(lossValue, compiledLossValue, accuracy: 1e-5,
                          "Forward pass should be deterministic at iteration \(iteration)")
        }
    }

    func testEquivalenceCNNGradients() {
        // Test that compiled and non-compiled CNN training steps produce valid results

        let model = CNNModel()
        let optimizer = SGD(learningRate: 0.01)
        let images = createRandom2DImages(batchSize: 8)
        let labels = createRandomLabels(batchSize: 8)

        // =====================================================================
        // Get Initial Loss
        // =====================================================================
        let initialLogits = model(images)
        let initialLoss = crossEntropy(logits: initialLogits, targets: labels, reduction: .mean)
        let initialLossValue = initialLoss.item(Float.self)

        // =====================================================================
        // Run Compiled Training Step
        // =====================================================================
        let compiledStep = createCompiledCNNTrainingStep(model: model, optimizer: optimizer)
        let compiledLoss = compiledStep(images, labels)
        let compiledLossValue = compiledLoss.item(Float.self)

        // Get loss after compiled training step
        let afterCompiledLogits = model(images)
        let afterCompiledLoss = crossEntropy(logits: afterCompiledLogits, targets: labels, reduction: .mean)
        let afterCompiledLossValue = afterCompiledLoss.item(Float.self)

        // =====================================================================
        // Verify Results
        // =====================================================================
        assertValidLoss(compiledLossValue, "CNN compiled step should produce valid loss")

        // Model should have been updated
        XCTAssertNotEqual(initialLossValue, afterCompiledLossValue, accuracy: 1e-8,
                         "CNN model parameters should have been updated")

        // Compiled step should work on subsequent calls
        let secondStepLoss = compiledStep(images, labels)
        assertValidLoss(secondStepLoss.item(Float.self),
                       "CNN compiled step should work on subsequent calls")
    }

    func testEquivalenceCNNVariousBatchSizes() {
        // Test CNN equivalence with various batch sizes (tests shapeless=true)

        let batchSizes = [1, 4, 8, 16]

        for batchSize in batchSizes {
            let model = CNNModel()
            let images = createRandom2DImages(batchSize: batchSize)
            let labels = createRandomLabels(batchSize: batchSize)

            // Compute loss without compilation
            let lossAndGrad = valueAndGrad(model: model, cnnLoss)
            let (loss, _) = lossAndGrad(model, images, labels)
            let lossValue = loss.item(Float.self)

            // Verify loss is valid
            assertValidLoss(lossValue,
                           "Loss should be valid for batch size \(batchSize)")

            // Verify same forward pass produces same result
            let logits = model(images)
            let forwardLoss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
            let forwardLossValue = forwardLoss.item(Float.self)

            XCTAssertEqual(lossValue, forwardLossValue, accuracy: 1e-5,
                          "Forward pass should be consistent for batch size \(batchSize)")
        }
    }

    func testEquivalenceAttentionGradients() {
        // Test that compiled and non-compiled Attention training steps produce valid results

        let model = AttentionModel()
        let optimizer = SGD(learningRate: 0.01)
        let images = createRandom2DImages(batchSize: 4)
        let labels = createRandomLabels(batchSize: 4)

        // =====================================================================
        // Get Initial Loss
        // =====================================================================
        let initialLogits = model(images)
        let initialLoss = crossEntropy(logits: initialLogits, targets: labels, reduction: .mean)
        let initialLossValue = initialLoss.item(Float.self)

        // =====================================================================
        // Run Compiled Training Step
        // =====================================================================
        let compiledStep = createCompiledAttentionTrainingStep(model: model, optimizer: optimizer)
        let compiledLoss = compiledStep(images, labels)
        let compiledLossValue = compiledLoss.item(Float.self)

        // Get loss after compiled training step
        let afterCompiledLogits = model(images)
        let afterCompiledLoss = crossEntropy(logits: afterCompiledLogits, targets: labels, reduction: .mean)
        let afterCompiledLossValue = afterCompiledLoss.item(Float.self)

        // =====================================================================
        // Verify Results
        // =====================================================================
        assertValidLoss(compiledLossValue, "Attention compiled step should produce valid loss")

        // Model should have been updated
        XCTAssertNotEqual(initialLossValue, afterCompiledLossValue, accuracy: 1e-8,
                         "Attention model parameters should have been updated")

        // Compiled step should work on subsequent calls
        let secondStepLoss = compiledStep(images, labels)
        assertValidLoss(secondStepLoss.item(Float.self),
                       "Attention compiled step should work on subsequent calls")
    }

    func testEquivalenceAttentionSingleSample() {
        // Test Attention equivalence with single sample (edge case)

        let model = AttentionModel()
        let images = createRandom2DImages(batchSize: 1)
        let labels = createRandomLabels(batchSize: 1)

        // Compute loss without compilation
        let lossAndGrad = valueAndGrad(model: model, attentionLoss)
        let (loss, _) = lossAndGrad(model, images, labels)
        let lossValue = loss.item(Float.self)

        // Verify loss is valid
        assertValidLoss(lossValue, "Loss should be valid for single sample")

        // Verify same forward pass produces same result
        let logits = model(images)
        let forwardLoss = crossEntropy(logits: logits, targets: labels, reduction: .mean)
        let forwardLossValue = forwardLoss.item(Float.self)

        XCTAssertEqual(lossValue, forwardLossValue, accuracy: 1e-5,
                      "Single sample forward pass should be consistent")
    }

    func testEquivalenceMLPWithZeroInput() {
        // Test equivalence when input is zero (edge case that might produce small gradients)

        let model = MLPModel()
        let optimizer = SGD(learningRate: 0.01)
        let images = MLXArray.zeros([8, 784])
        let labels = createRandomLabels(batchSize: 8)

        // =====================================================================
        // Get Initial Loss
        // =====================================================================
        let initialLogits = model(images)
        let initialLoss = crossEntropy(logits: initialLogits, targets: labels, reduction: .mean)
        let initialLossValue = initialLoss.item(Float.self)

        // =====================================================================
        // Run Compiled Training Step with Zero Input
        // =====================================================================
        let compiledStep = createCompiledMLPTrainingStep(model: model, optimizer: optimizer)
        let compiledLoss = compiledStep(images, labels)
        let compiledLossValue = compiledLoss.item(Float.self)

        // Get loss after compiled training step
        let afterCompiledLogits = model(images)
        let afterCompiledLoss = crossEntropy(logits: afterCompiledLogits, targets: labels, reduction: .mean)
        let afterCompiledLossValue = afterCompiledLoss.item(Float.self)

        // =====================================================================
        // Verify Results
        // =====================================================================
        assertValidLoss(compiledLossValue, "Compiled step should handle zero input")

        // Model should have been updated even with zero input
        // (output layer still gets gradients from loss, even if hidden layer gradients are small)
        XCTAssertNotEqual(initialLossValue, afterCompiledLossValue, accuracy: 1e-8,
                         "Model should update even with zero input")

        // Compiled step should work on subsequent calls
        let secondStepLoss = compiledStep(images, labels)
        assertValidLoss(secondStepLoss.item(Float.self),
                       "Compiled step should work with zero input on subsequent calls")
    }
}
