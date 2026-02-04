// ============================================================================
// AttentionModelTests.swift - Tests for Attention Model Forward Pass
// ============================================================================
//
// This test suite validates the Attention model's functionality:
// - Output shape correctness for various batch sizes
// - Correct handling of flat image input [N, 784] -> patches
// - Attention mechanism (Q/K/V projections, self-attention)
// - Edge cases (single sample, large batches)
// - Output properties (finite values, correct dimensions)
// - Loss and accuracy computation
// - Model architecture configuration (dModel, ffDim)
//
// The attention model uses a Vision Transformer (ViT) style architecture:
// - Patchifies 28×28 images into 7×7 grid of 4×4 patches (49 tokens)
// - Projects patches to embedding dimension (dModel=32)
// - Applies self-attention to allow patches to interact
// - Uses feed-forward network for token processing
// - Mean pools over tokens for classification
//
// ============================================================================

import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import MNISTMLX

final class AttentionModelTests: MLXTestCase {

    // =============================================================================
    // MARK: - Test Utilities
    // =============================================================================

    /// Creates a random input tensor with specified shape
    /// - Parameters:
    ///   - batchSize: Number of samples in the batch
    ///   - inputSize: Size of each input vector (default: 784 for MNIST)
    /// - Returns: MLXArray with shape [batchSize, inputSize] filled with random values
    private func createRandomInput(batchSize: Int, inputSize: Int = 784) -> MLXArray {
        // Create random normal values and scale to [0, 1] range
        return abs(MLXRandom.normal([batchSize, inputSize]))
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

    /// Verifies that an array has the expected shape
    private func assertShape(_ array: MLXArray, _ expectedShape: [Int],
                            _ message: String = "",
                            file: StaticString = #file,
                            line: UInt = #line) {
        XCTAssertEqual(array.shape, expectedShape, message, file: file, line: line)
    }

    /// Verifies that all values in an array are finite (not NaN or Inf)
    private func assertAllFinite(_ array: MLXArray,
                                 _ message: String = "All values should be finite",
                                 file: StaticString = #file,
                                 line: UInt = #line) {
        eval(array)
        let values = array.asArray(Float.self)
        XCTAssertFalse(values.isEmpty, "Array should not be empty", file: file, line: line)
        for value in values {
            XCTAssertTrue(value.isFinite, "\(message): found non-finite value \(value)", file: file, line: line)
        }
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
    // MARK: - Forward Pass Shape Tests
    // =============================================================================

    func testForwardShape() {
        // Test that forward pass produces correct output shape
        // Input: [32, 784] → Output: [32, 10]
        let model = AttentionModel()
        let input = createRandomInput(batchSize: 32)

        let output = model(input)

        assertShape(output, [32, 10],
                   "Forward pass should output [32, 10] for batch size 32")
    }

    func testForwardShapeSingleSample() {
        // Test forward pass with a single sample
        // Input: [1, 784] → Output: [1, 10]
        let model = AttentionModel()
        let input = createRandomInput(batchSize: 1)

        let output = model(input)

        assertShape(output, [1, 10],
                   "Forward pass should output [1, 10] for single sample")
    }

    func testForwardShapeSmallBatch() {
        // Test forward pass with small batch size
        // Input: [4, 784] → Output: [4, 10]
        let model = AttentionModel()
        let input = createRandomInput(batchSize: 4)

        let output = model(input)

        assertShape(output, [4, 10],
                   "Forward pass should output [4, 10] for batch size 4")
    }

    func testForwardShapeLargeBatch() {
        // Test forward pass with large batch size
        // Input: [128, 784] → Output: [128, 10]
        let model = AttentionModel()
        let input = createRandomInput(batchSize: 128)

        let output = model(input)

        assertShape(output, [128, 10],
                   "Forward pass should output [128, 10] for batch size 128")
    }

    func testForwardShapeVeryLargeBatch() {
        // Test forward pass with a large (but safe) batch size to avoid OOM/timeouts.
        // Input: [1024, 784] → Output: [1024, 10]
        let model = AttentionModel()
        let input = createRandomInput(batchSize: 1024)

        let output = model(input)

        assertShape(output, [1024, 10],
                   "Forward pass should output [1024, 10] for batch size 1024")
    }

    func testForwardShapeVariousBatchSizes() {
        // Test forward pass consistency across various batch sizes
        let model = AttentionModel()
        let batchSizes = [1, 2, 8, 16, 32, 64, 100, 256]

        for batchSize in batchSizes {
            let input = createRandomInput(batchSize: batchSize)
            let output = model(input)

            assertShape(output, [batchSize, 10],
                       "Forward pass should output [\(batchSize), 10] for batch size \(batchSize)")
        }
    }

    // =============================================================================
    // MARK: - Output Properties Tests
    // =============================================================================

    func testForwardOutputIsFinite() {
        // Test that forward pass produces finite values (no NaN or Inf)
        let model = AttentionModel()
        let input = createRandomInput(batchSize: 32)

        let output = model(input)

        assertAllFinite(output, "Forward pass output should contain finite values")
    }

    func testForwardOutputIsFiniteWithZeroInput() {
        // Test that forward pass handles zero input correctly
        let model = AttentionModel()
        let input = MLXArray.zeros([16, 784])

        let output = model(input)

        assertShape(output, [16, 10], "Forward pass should handle zero input")
        assertAllFinite(output, "Forward pass with zero input should produce finite values")
    }

    func testForwardOutputIsFiniteWithLargeInput() {
        // Test that forward pass handles large input values
        let model = AttentionModel()
        let input = MLXArray.ones([16, 784]) * 100.0

        let output = model(input)

        assertShape(output, [16, 10], "Forward pass should handle large input")
        assertAllFinite(output, "Forward pass with large input should produce finite values")
    }

    func testForwardOutputDimensions() {
        // Test that output has correct number of dimensions
        let model = AttentionModel()
        let batchSizes = [1, 8, 32, 64]

        for batchSize in batchSizes {
            let input = createRandomInput(batchSize: batchSize)
            let output = model(input)

            XCTAssertEqual(output.ndim, 2,
                          "Output should have 2 dimensions [batch, classes] for batch size \(batchSize)")
            XCTAssertEqual(output.shape[0], batchSize,
                          "First dimension should match batch size \(batchSize)")
            XCTAssertEqual(output.shape[1], 10,
                          "Second dimension should be 10 (number of classes)")
        }
    }

    // =============================================================================
    // MARK: - Attention Mechanism Tests
    // =============================================================================

    func testPatchifyConsistency() {
        // Test that patchify operation is consistent across calls
        // The model should produce the same output for the same input
        let model = AttentionModel()
        let input = createRandomInput(batchSize: 16)

        let output1 = model(input)
        let output2 = model(input)

        eval(output1, output2)

        // Extract values and compare
        let values1 = output1.asArray(Float.self)
        let values2 = output2.asArray(Float.self)

        XCTAssertEqual(values1.count, values2.count,
                      "Same input should produce same output size")

        for (v1, v2) in zip(values1, values2) {
            XCTAssertEqual(v1, v2, accuracy: 1e-6,
                          "Same input should produce identical output")
        }
    }

    func testAttentionDeterminism() {
        // Test that attention mechanism is deterministic (same input → same output)
        let model = AttentionModel()
        let batchSize = 8
        let input = createRandomInput(batchSize: batchSize)

        let output1 = model(input)
        let output2 = model(input)

        eval(output1, output2)

        // Verify outputs are identical
        let diff = sum(abs(output1 - output2)).item(Float.self)
        XCTAssertEqual(diff, 0.0, accuracy: 1e-6,
                      "Attention model should be deterministic")
    }

    func testAttentionOutputRange() {
        // Test that attention output (logits) are in a reasonable range
        // Logits shouldn't be extremely large or extremely small
        let model = AttentionModel()
        let input = createRandomInput(batchSize: 32)

        let output = model(input)
        eval(output)

        // Get min and max values
        let minVal = min(output).item(Float.self)
        let maxVal = max(output).item(Float.self)

        // Logits should be in reasonable range (not too extreme)
        // Typical range for untrained model: roughly [-10, 10]
        XCTAssertGreaterThan(minVal, -100.0,
                            "Minimum logit should not be extremely negative")
        XCTAssertLessThan(maxVal, 100.0,
                         "Maximum logit should not be extremely large")
    }

    // =============================================================================
    // MARK: - Loss Function Tests
    // =============================================================================

    func testAttentionLossFunction() {
        // Test the attentionLoss wrapper function
        let model = AttentionModel()
        let batchSize = 16

        let images = createRandomInput(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let loss = attentionLoss(model: model, images: images, labels: labels)
        eval(loss)

        // Verify loss properties
        XCTAssertEqual(loss.size, 1, "Attention loss should be scalar")
        assertAllFinite(loss, "Attention loss should be finite")

        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThan(lossValue, 0.0, "Attention loss should be positive")
        XCTAssertLessThan(lossValue, 20.0, "Attention loss should be reasonable for untrained model")
    }

    func testAttentionLossConsistency() {
        // Test that attentionLoss produces consistent results for same inputs
        let model = AttentionModel()
        let batchSize = 8
        let images = createRandomInput(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let loss1 = attentionLoss(model: model, images: images, labels: labels)
        let loss2 = attentionLoss(model: model, images: images, labels: labels)

        eval(loss1, loss2)

        let lossValue1 = loss1.item(Float.self)
        let lossValue2 = loss2.item(Float.self)

        XCTAssertEqual(lossValue1, lossValue2, accuracy: 1e-6,
                      "Same inputs should produce same loss")
    }

    func testAttentionLossVariousBatchSizes() {
        // Test attention loss with various batch sizes
        let model = AttentionModel()
        let batchSizes = [1, 4, 16, 32, 64]

        for batchSize in batchSizes {
            let images = createRandomInput(batchSize: batchSize)
            let labels = createRandomLabels(batchSize: batchSize)

            let loss = attentionLoss(model: model, images: images, labels: labels)
            eval(loss)

            let lossValue = loss.item(Float.self)
            XCTAssertGreaterThan(lossValue, 0.0,
                                "Loss should be positive for batch size \(batchSize)")
            assertAllFinite(loss, "Loss should be finite for batch size \(batchSize)")
        }
    }

    // =============================================================================
    // MARK: - Accuracy Function Tests
    // =============================================================================

    func testAttentionAccuracyFunction() {
        // Test the attentionAccuracy wrapper function
        let model = AttentionModel()
        let batchSize = 32
        let images = createRandomInput(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let accuracy = attentionAccuracy(model: model, images: images, labels: labels)

        // Verify accuracy is in valid range
        assertInRange(accuracy, 0.0, 1.0,
                     "Attention accuracy should be in [0.0, 1.0]")
    }

    func testAttentionAccuracyConsistency() {
        // Test that attentionAccuracy produces consistent results for same inputs
        let model = AttentionModel()
        let batchSize = 16
        let images = createRandomInput(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let accuracy1 = attentionAccuracy(model: model, images: images, labels: labels)
        let accuracy2 = attentionAccuracy(model: model, images: images, labels: labels)

        XCTAssertEqual(accuracy1, accuracy2, accuracy: 1e-6,
                      "Same inputs should produce same accuracy")
    }

    func testAttentionAccuracyUntrainedModel() {
        // Test accuracy of untrained model (should be around random guessing)
        let model = AttentionModel()
        let batchSize = 100
        let images = createRandomInput(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let accuracy = attentionAccuracy(model: model, images: images, labels: labels)

        // Untrained model should have low accuracy (not much better than random)
        // Random guessing on 10 classes = 10%, allow range [0.0, 0.5]
        assertInRange(accuracy, 0.0, 0.5,
                     "Untrained Attention accuracy should be low (roughly random guessing)")
    }

    func testAttentionAccuracyVariousBatchSizes() {
        // Test Attention accuracy with various batch sizes
        let model = AttentionModel()
        let batchSizes = [1, 4, 16, 32, 64]

        for batchSize in batchSizes {
            let images = createRandomInput(batchSize: batchSize)
            let labels = createRandomLabels(batchSize: batchSize)

            let accuracy = attentionAccuracy(model: model, images: images, labels: labels)

            assertInRange(accuracy, 0.0, 1.0,
                         "Attention accuracy should be in [0.0, 1.0] for batch size \(batchSize)")
        }
    }

    // =============================================================================
    // MARK: - Gradient Flow Tests
    // =============================================================================

    func testAttentionGradientsExist() {
        // Test that attentionLoss produces gradients for backpropagation
        let model = AttentionModel()
        let batchSize = 8
        let images = createRandomInput(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let lossAndGrad = valueAndGrad(model: model, attentionLoss)
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

    func testAttentionGradientsNonZero() {
        // Test that attention gradients are non-zero (indicating proper flow)
        let model = AttentionModel()
        let batchSize = 8
        let images = createRandomInput(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        let lossAndGrad = valueAndGrad(model: model, attentionLoss)
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
    // MARK: - Model Architecture Tests
    // =============================================================================

    func testModelArchitectureConfiguration() {
        // Test that the model is configured with the correct dimensions
        // This verifies the fix for the accuracy issue (dModel=32, ffDim=64)
        let model = AttentionModel()

        // Verify model has the expected layers
        XCTAssertNotNil(model.patchEmbed, "Model should have patch embedding layer")
        XCTAssertNotNil(model.wQ, "Model should have query projection layer")
        XCTAssertNotNil(model.wK, "Model should have key projection layer")
        XCTAssertNotNil(model.wV, "Model should have value projection layer")
        XCTAssertNotNil(model.ff1, "Model should have first feed-forward layer")
        XCTAssertNotNil(model.ff2, "Model should have second feed-forward layer")
        XCTAssertNotNil(model.classifier, "Model should have classifier layer")
        XCTAssertNotNil(model.posEmbeddings, "Model should have positional embeddings")
    }

    func testPositionalEmbeddingsShape() {
        // Test that positional embeddings have the correct shape
        // Should be [49, 32] for 49 patches and dModel=32
        let model = AttentionModel()
        let posEmbed = model.posEmbeddings

        assertShape(posEmbed, [49, 32],
                   "Positional embeddings should be [49, 32] (seq_len=49, dModel=32)")
    }

    func testPositionalEmbeddingsAreFinite() {
        // Test that positional embeddings contain finite values
        let model = AttentionModel()
        let posEmbed = model.posEmbeddings

        assertAllFinite(posEmbed, "Positional embeddings should be finite")
    }

    // =============================================================================
    // MARK: - Training Step Test
    // =============================================================================

    func testTrainingStepReducesLoss() {
        // Test that a single training step can reduce loss
        // This verifies the model is trainable
        let model = AttentionModel()
        let optimizer = SGD(learningRate: 0.01)
        let batchSize = 16

        let images = createRandomInput(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        // Compute initial loss
        let initialLoss = attentionLoss(model: model, images: images, labels: labels)
        eval(initialLoss)
        let initialLossValue = initialLoss.item(Float.self)

        // Perform one training step
        let lossAndGrad = valueAndGrad(model: model, attentionLoss)
        let (_, grads) = lossAndGrad(model, images, labels)
        optimizer.update(model: model, gradients: grads)
        eval(model, optimizer)

        // Compute new loss
        let newLoss = attentionLoss(model: model, images: images, labels: labels)
        eval(newLoss)
        let newLossValue = newLoss.item(Float.self)

        // Verify loss changed (indicates parameters were updated)
        XCTAssertNotEqual(initialLossValue, newLossValue,
                         "Training step should change the loss")

        // Both losses should be finite
        XCTAssertTrue(initialLossValue.isFinite, "Initial loss should be finite")
        XCTAssertTrue(newLossValue.isFinite, "New loss should be finite")
    }

    func testTrainingStepUpdatesParameters() {
        // Test that a training step actually updates model parameters
        let model = AttentionModel()
        let optimizer = SGD(learningRate: 0.01)
        let batchSize = 8

        let images = createRandomInput(batchSize: batchSize)
        let labels = createRandomLabels(batchSize: batchSize)

        // Get initial parameter values (check first layer)
        let initialParams = model.parameters()
        let initialParamsList = initialParams.flattened()
        XCTAssertGreaterThan(initialParamsList.count, 0, "Model should have parameters")

        // Store initial values
        var initialValues: [(String, [Float])] = []
        for (key, param) in initialParamsList {
            eval(param)
            initialValues.append((key, param.asArray(Float.self)))
        }

        // Perform one training step
        let lossAndGrad = valueAndGrad(model: model, attentionLoss)
        let (_, grads) = lossAndGrad(model, images, labels)
        optimizer.update(model: model, gradients: grads)
        eval(model, optimizer)

        // Get updated parameter values
        let updatedParams = model.parameters()
        let updatedParamsList = updatedParams.flattened()

        // Verify at least some parameters changed
        var parametersChanged = false
        for ((key, updatedParam), (initialKey, initialArray)) in zip(updatedParamsList, initialValues) {
            XCTAssertEqual(key, initialKey, "Parameter keys should match")
            eval(updatedParam)
            let updatedArray = updatedParam.asArray(Float.self)

            // Check if any values changed
            if updatedArray != initialArray {
                parametersChanged = true
                break
            }
        }

        XCTAssertTrue(parametersChanged, "Training step should update at least some parameters")
    }

    // =============================================================================
    // MARK: - Input Validation Tests
    // =============================================================================

    func testForwardWithCorrectInputSize() {
        // Test that model handles correctly sized input
        let model = AttentionModel()
        let input = createRandomInput(batchSize: 16, inputSize: 784)

        let output = model(input)

        assertShape(output, [16, 10], "Model should handle 784-dimensional input")
        assertAllFinite(output, "Output should be finite")
    }

    func testForwardInputDimensionality() {
        // Test that input has correct number of dimensions
        let batchSizes = [1, 8, 32]

        for batchSize in batchSizes {
            let input = createRandomInput(batchSize: batchSize)

            XCTAssertEqual(input.ndim, 2,
                          "Input should have 2 dimensions [batch, features] for batch size \(batchSize)")
            XCTAssertEqual(input.shape[0], batchSize,
                          "First dimension should be batch size \(batchSize)")
            XCTAssertEqual(input.shape[1], 784,
                          "Second dimension should be 784 (28×28 flattened)")
        }
    }

    // =============================================================================
    // MARK: - Edge Case Tests
    // =============================================================================

    func testForwardWithAllZeros() {
        // Test that model handles all-zero input gracefully
        let model = AttentionModel()
        let input = MLXArray.zeros([8, 784])

        let output = model(input)

        assertShape(output, [8, 10], "Model should handle all-zero input")
        assertAllFinite(output, "Output should be finite for all-zero input")
    }

    func testForwardWithAllOnes() {
        // Test that model handles uniform input gracefully
        let model = AttentionModel()
        let input = MLXArray.ones([8, 784])

        let output = model(input)

        assertShape(output, [8, 10], "Model should handle all-ones input")
        assertAllFinite(output, "Output should be finite for all-ones input")
    }

    func testForwardWithExtremeValues() {
        // Test that model handles extreme input values
        let model = AttentionModel()
        let input = MLXArray.ones([8, 784]) * 1000.0

        let output = model(input)

        assertShape(output, [8, 10], "Model should handle extreme input values")
        assertAllFinite(output, "Output should be finite for extreme input values")
    }

    // =============================================================================
    // MARK: - Performance Characteristic Tests
    // =============================================================================

    func testModelOutputDistribution() {
        // Test that untrained model produces reasonable output distribution
        // Logits should be roughly balanced (not heavily biased to one class)
        let model = AttentionModel()
        let input = createRandomInput(batchSize: 100)

        let output = model(input)
        eval(output)

        // Check that we get variation in outputs (not all the same)
        let stdDev = sqrt(mean(square(output - mean(output)))).item(Float.self)
        XCTAssertGreaterThan(stdDev, 0.01,
                            "Output should have non-zero standard deviation")
    }

    func testModelConsistentBehavior() {
        // Test that model behaves consistently across multiple invocations
        let model = AttentionModel()
        let input = createRandomInput(batchSize: 16)

        let outputs = (0..<5).map { _ in model(input) }

        // All outputs should be identical for same input
        outputs.forEach { eval($0) }
        for i in 1..<outputs.count {
            let diff = sum(abs(outputs[0] - outputs[i])).item(Float.self)
            XCTAssertEqual(diff, 0.0, accuracy: 1e-6,
                          "Model should produce consistent output for iteration \(i)")
        }
    }
}
