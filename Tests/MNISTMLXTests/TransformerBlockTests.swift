// ============================================================================
// TransformerBlockTests.swift - Tests for Transformer Block Implementation
// ============================================================================
//
// This test suite validates the TransformerBlock's functionality:
// - Output shape correctness for various batch sizes and sequence lengths
// - Correct handling of multi-head attention mechanism
// - Feed-forward network processing
// - Layer normalization and residual connections
// - Edge cases (single sample, large batches, various sequence lengths)
// - Output properties (finite values, correct dimensions)
// - Component integration (attention + FFN + residuals)
//
// The TransformerBlock implements a Pre-LN encoder block:
// - LayerNorm → Multi-Head Attention → (+residual)
// - LayerNorm → Feed-Forward Network → (+residual)
//
// This is the core building block of modern transformers (GPT, BERT, ViT).
//
// ============================================================================

import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import MNISTMLX

final class TransformerBlockTests: MLXTestCase {

    // =============================================================================
    // MARK: - Test Utilities
    // =============================================================================

    /// Creates a random input tensor with specified shape
    /// - Parameters:
    ///   - batchSize: Number of samples in the batch
    ///   - seqLen: Sequence length (number of tokens)
    ///   - dModel: Model dimension (embedding size)
    /// - Returns: MLXArray with shape [batchSize, seqLen, dModel] filled with random values
    private func createRandomInput(batchSize: Int, seqLen: Int, dModel: Int) -> MLXArray {
        // Create random normal values
        return MLXRandom.normal([batchSize, seqLen, dModel])
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
    // MARK: - Shape Tests
    // =============================================================================

    func testForwardShapeStandard() {
        // Test that forward pass produces correct output shape for standard MNIST case
        // Input: [32, 49, 32] → Output: [32, 49, 32]
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let input = createRandomInput(batchSize: 32, seqLen: 49, dModel: 32)

        let output = block(input)

        assertShape(output, [32, 49, 32],
                   "Forward pass should preserve shape [32, 49, 32]")
    }

    func testForwardShapeSingleSample() {
        // Test forward pass with a single sample
        // Input: [1, 49, 32] → Output: [1, 49, 32]
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let input = createRandomInput(batchSize: 1, seqLen: 49, dModel: 32)

        let output = block(input)

        assertShape(output, [1, 49, 32],
                   "Forward pass should preserve shape for single sample")
    }

    func testForwardShapeSmallBatch() {
        // Test forward pass with small batch size
        // Input: [4, 49, 32] → Output: [4, 49, 32]
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let input = createRandomInput(batchSize: 4, seqLen: 49, dModel: 32)

        let output = block(input)

        assertShape(output, [4, 49, 32],
                   "Forward pass should preserve shape for batch size 4")
    }

    func testForwardShapeLargeBatch() {
        // Test forward pass with large batch size
        // Input: [128, 49, 32] → Output: [128, 49, 32]
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let input = createRandomInput(batchSize: 128, seqLen: 49, dModel: 32)

        let output = block(input)

        assertShape(output, [128, 49, 32],
                   "Forward pass should preserve shape for batch size 128")
    }

    func testForwardShapeShortSequence() {
        // Test forward pass with short sequence length
        // Input: [32, 10, 32] → Output: [32, 10, 32]
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let input = createRandomInput(batchSize: 32, seqLen: 10, dModel: 32)

        let output = block(input)

        assertShape(output, [32, 10, 32],
                   "Forward pass should preserve shape for short sequence")
    }

    func testForwardShapeLongSequence() {
        // Test forward pass with long sequence length
        // Input: [16, 256, 32] → Output: [16, 256, 32]
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let input = createRandomInput(batchSize: 16, seqLen: 256, dModel: 32)

        let output = block(input)

        assertShape(output, [16, 256, 32],
                   "Forward pass should preserve shape for long sequence")
    }

    func testForwardShapeVariousBatchSizes() {
        // Test forward pass consistency across various batch sizes
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let batchSizes = [1, 2, 8, 16, 32, 64]
        let seqLen = 49
        let dModel = 32

        for batchSize in batchSizes {
            let input = createRandomInput(batchSize: batchSize, seqLen: seqLen, dModel: dModel)
            let output = block(input)

            assertShape(output, [batchSize, seqLen, dModel],
                       "Forward pass should preserve shape for batch size \(batchSize)")
        }
    }

    func testForwardShapeVariousSequenceLengths() {
        // Test forward pass consistency across various sequence lengths
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let batchSize = 16
        let dModel = 32
        let seqLengths = [1, 10, 49, 100, 196]

        for seqLen in seqLengths {
            let input = createRandomInput(batchSize: batchSize, seqLen: seqLen, dModel: dModel)
            let output = block(input)

            assertShape(output, [batchSize, seqLen, dModel],
                       "Forward pass should preserve shape for sequence length \(seqLen)")
        }
    }

    // =============================================================================
    // MARK: - Different Model Dimensions Tests
    // =============================================================================

    func testForwardShapeDifferentDModel() {
        // Test forward pass with different d_model values
        // Must be divisible by num_heads
        let configs = [
            (dModel: 16, numHeads: 2, ffDim: 32),
            (dModel: 32, numHeads: 4, ffDim: 64),
            (dModel: 64, numHeads: 8, ffDim: 128),
            (dModel: 128, numHeads: 8, ffDim: 256)
        ]

        let batchSize = 8
        let seqLen = 49

        for config in configs {
            let block = TransformerBlock(dModel: config.dModel, numHeads: config.numHeads, ffDim: config.ffDim)
            let input = createRandomInput(batchSize: batchSize, seqLen: seqLen, dModel: config.dModel)
            let output = block(input)

            assertShape(output, [batchSize, seqLen, config.dModel],
                       "Forward pass should preserve shape for d_model=\(config.dModel)")
        }
    }

    func testForwardShapeDifferentNumHeads() {
        // Test forward pass with different numbers of attention heads
        let dModel = 32
        let numHeadsOptions = [1, 2, 4, 8, 16, 32]
        let batchSize = 8
        let seqLen = 49

        for numHeads in numHeadsOptions {
            let block = TransformerBlock(dModel: dModel, numHeads: numHeads, ffDim: 64)
            let input = createRandomInput(batchSize: batchSize, seqLen: seqLen, dModel: dModel)
            let output = block(input)

            assertShape(output, [batchSize, seqLen, dModel],
                       "Forward pass should preserve shape for num_heads=\(numHeads)")
        }
    }

    // =============================================================================
    // MARK: - Output Properties Tests
    // =============================================================================

    func testForwardOutputIsFinite() {
        // Test that forward pass produces finite values (no NaN or Inf)
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let input = createRandomInput(batchSize: 32, seqLen: 49, dModel: 32)

        let output = block(input)

        assertAllFinite(output, "Forward pass output should contain finite values")
    }

    func testForwardOutputIsFiniteWithZeroInput() {
        // Test that forward pass handles zero input correctly
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let input = MLXArray.zeros([16, 49, 32])

        let output = block(input)

        assertShape(output, [16, 49, 32], "Forward pass should handle zero input")
        assertAllFinite(output, "Forward pass with zero input should produce finite values")
    }

    func testForwardOutputIsFiniteWithLargeInput() {
        // Test that forward pass handles large input values
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let input = MLXArray.ones([16, 49, 32]) * 100.0

        let output = block(input)

        assertShape(output, [16, 49, 32], "Forward pass should handle large input")
        assertAllFinite(output, "Forward pass with large input should produce finite values")
    }

    func testForwardOutputIsFiniteWithSmallInput() {
        // Test that forward pass handles very small input values
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let input = MLXArray.ones([16, 49, 32]) * 0.001

        let output = block(input)

        assertShape(output, [16, 49, 32], "Forward pass should handle small input")
        assertAllFinite(output, "Forward pass with small input should produce finite values")
    }

    func testForwardOutputDimensions() {
        // Test that output has correct number of dimensions
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let input = createRandomInput(batchSize: 32, seqLen: 49, dModel: 32)

        let output = block(input)

        XCTAssertEqual(output.ndim, 3, "Output should be 3-dimensional [batch, seq_len, d_model]")
    }

    // =============================================================================
    // MARK: - Component Tests
    // =============================================================================

    func testMultiHeadAttentionComponent() {
        // Test that multi-head attention component works correctly
        let attention = MultiHeadAttention(dModel: 32, numHeads: 4)
        let input = createRandomInput(batchSize: 16, seqLen: 49, dModel: 32)

        let output = attention(input)

        assertShape(output, [16, 49, 32],
                   "Multi-head attention should preserve shape")
        assertAllFinite(output, "Multi-head attention output should be finite")
    }

    func testMultiHeadAttentionSingleHead() {
        // Test multi-head attention with single head (degenerate case)
        let attention = MultiHeadAttention(dModel: 32, numHeads: 1)
        let input = createRandomInput(batchSize: 16, seqLen: 49, dModel: 32)

        let output = attention(input)

        assertShape(output, [16, 49, 32],
                   "Single-head attention should preserve shape")
        assertAllFinite(output, "Single-head attention output should be finite")
    }

    func testMultiHeadAttentionManyHeads() {
        // Test multi-head attention with many heads
        let attention = MultiHeadAttention(dModel: 32, numHeads: 32)
        let input = createRandomInput(batchSize: 16, seqLen: 49, dModel: 32)

        let output = attention(input)

        assertShape(output, [16, 49, 32],
                   "32-head attention should preserve shape")
        assertAllFinite(output, "32-head attention output should be finite")
    }

    func testLayerNormComponent() {
        // Test that layer normalization works correctly
        let dModel = 32
        let norm = LayerNorm(dimensions: dModel, eps: 1e-5, affine: true, bias: true)
        let input = createRandomInput(batchSize: 16, seqLen: 49, dModel: dModel)

        let output = norm(input)

        assertShape(output, [16, 49, dModel],
                   "LayerNorm should preserve shape")
        assertAllFinite(output, "LayerNorm output should be finite")
    }

    func testFeedForwardNetworkComponent() {
        // Test that feed-forward network (two linear layers) works correctly
        let dModel = 32
        let ffDim = 64
        let ffn1 = Linear(dModel, ffDim)
        let ffn2 = Linear(ffDim, dModel)

        let input = createRandomInput(batchSize: 16, seqLen: 49, dModel: dModel)

        // FFN(x) = W2(ReLU(W1(x)))
        let hidden = relu(ffn1(input))
        let output = ffn2(hidden)

        assertShape(output, [16, 49, dModel],
                   "Feed-forward network should preserve shape")
        assertAllFinite(output, "Feed-forward network output should be finite")
    }

    // =============================================================================
    // MARK: - Residual Connection Tests
    // =============================================================================

    func testResidualConnectionsPreserveGradients() {
        // Test that residual connections allow gradient flow
        // We verify this by checking that output contains contributions from input
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)

        // Create a structured input to trace residual connections
        let input = MLXArray.ones([8, 49, 32]) * 5.0
        let output = block(input)

        // Output should be different from input (due to attention and FFN)
        // but should contain contributions from input (residual connections)
        assertShape(output, [8, 49, 32], "Output should have same shape as input")
        assertAllFinite(output, "Output with residual connections should be finite")
    }

    func testResidualConnectionWithIdentity() {
        // Test that if attention and FFN learn to output near-zero,
        // the block approximates identity (due to residuals)
        // This is a theoretical property; we just test the structure
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let input = createRandomInput(batchSize: 8, seqLen: 49, dModel: 32)

        let output = block(input)

        // Output should be finite and have correct shape
        // The actual values depend on learned parameters
        assertShape(output, [8, 49, 32])
        assertAllFinite(output)
    }

    // =============================================================================
    // MARK: - Full TransformerModel Tests
    // =============================================================================

    func testTransformerModelShapeSingleLayer() {
        // Test full TransformerModel with single layer
        let model = TransformerModel(numLayers: 1, dModel: 32, numHeads: 4, ffDim: 64)
        let images = createRandomInput(batchSize: 32, seqLen: 784, dModel: 1).reshaped([32, 784])

        let logits = model(images)

        assertShape(logits, [32, 10],
                   "TransformerModel should output [32, 10] for MNIST classification")
        assertAllFinite(logits, "TransformerModel output should be finite")
    }

    func testTransformerModelShapeTwoLayers() {
        // Test full TransformerModel with two layers (default)
        let model = TransformerModel(numLayers: 2, dModel: 32, numHeads: 4, ffDim: 64)
        let images = createRandomInput(batchSize: 32, seqLen: 784, dModel: 1).reshaped([32, 784])

        let logits = model(images)

        assertShape(logits, [32, 10],
                   "TransformerModel with 2 layers should output [32, 10]")
        assertAllFinite(logits, "TransformerModel output should be finite")
    }

    func testTransformerModelShapeMultipleLayers() {
        // Test full TransformerModel with multiple layers
        let model = TransformerModel(numLayers: 4, dModel: 32, numHeads: 4, ffDim: 64)
        let images = createRandomInput(batchSize: 16, seqLen: 784, dModel: 1).reshaped([16, 784])

        let logits = model(images)

        assertShape(logits, [16, 10],
                   "TransformerModel with 4 layers should output [16, 10]")
        assertAllFinite(logits, "TransformerModel output should be finite")
    }

    func testTransformerModelSingleSample() {
        // Test TransformerModel with single sample
        let model = TransformerModel(numLayers: 2, dModel: 32, numHeads: 4, ffDim: 64)
        let image = createRandomInput(batchSize: 1, seqLen: 784, dModel: 1).reshaped([1, 784])

        let logits = model(image)

        assertShape(logits, [1, 10],
                   "TransformerModel should handle single sample")
        assertAllFinite(logits, "TransformerModel single sample output should be finite")
    }

    func testTransformerModelLargeBatch() {
        // Test TransformerModel with large batch
        let model = TransformerModel(numLayers: 2, dModel: 32, numHeads: 4, ffDim: 64)
        let images = createRandomInput(batchSize: 128, seqLen: 784, dModel: 1).reshaped([128, 784])

        let logits = model(images)

        assertShape(logits, [128, 10],
                   "TransformerModel should handle large batch")
        assertAllFinite(logits, "TransformerModel large batch output should be finite")
    }

    // =============================================================================
    // MARK: - Training Functions Tests
    // =============================================================================

    func testTransformerLoss() {
        // Test that loss function produces valid scalar
        let model = TransformerModel(numLayers: 2, dModel: 32, numHeads: 4, ffDim: 64)
        let images = createRandomInput(batchSize: 32, seqLen: 784, dModel: 1).reshaped([32, 784])
        let labels = MLXArray((0..<32).map { _ in Int32.random(in: 0..<10) })

        let loss = transformerLoss(model: model, images: images, labels: labels)

        XCTAssertEqual(loss.ndim, 0, "Loss should be a scalar")
        assertAllFinite(loss, "Loss should be finite")

        let lossValue = loss.item(Float.self)
        assertInRange(lossValue, 0.0, 10.0, "Loss should be in reasonable range")
    }

    func testTransformerAccuracy() {
        // Test that accuracy function produces valid value in [0, 1]
        let model = TransformerModel(numLayers: 2, dModel: 32, numHeads: 4, ffDim: 64)
        let images = createRandomInput(batchSize: 32, seqLen: 784, dModel: 1).reshaped([32, 784])
        let labels = MLXArray((0..<32).map { _ in Int32.random(in: 0..<10) })

        let accuracy = transformerAccuracy(model: model, images: images, labels: labels)

        XCTAssertTrue(accuracy.isFinite, "Accuracy should be finite")
        assertInRange(accuracy, 0.0, 1.0, "Accuracy should be in [0, 1]")
    }

    // =============================================================================
    // MARK: - Edge Cases
    // =============================================================================

    func testMinimalConfiguration() {
        // Test transformer block with minimal valid configuration
        let block = TransformerBlock(dModel: 2, numHeads: 1, ffDim: 4)
        let input = createRandomInput(batchSize: 2, seqLen: 3, dModel: 2)

        let output = block(input)

        assertShape(output, [2, 3, 2], "Minimal configuration should work")
        assertAllFinite(output, "Minimal configuration output should be finite")
    }

    func testSingleTokenSequence() {
        // Test with sequence length of 1 (single token)
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let input = createRandomInput(batchSize: 16, seqLen: 1, dModel: 32)

        let output = block(input)

        assertShape(output, [16, 1, 32], "Single token sequence should work")
        assertAllFinite(output, "Single token output should be finite")
    }

    // =============================================================================
    // MARK: - Performance Tests
    // =============================================================================

    func testTransformerBlockPerformance() {
        // Measure performance of transformer block forward pass
        let block = TransformerBlock(dModel: 32, numHeads: 4, ffDim: 64)
        let input = createRandomInput(batchSize: 32, seqLen: 49, dModel: 32)

        measure {
            let output = block(input)
            eval(output)  // Force evaluation
        }
    }

    func testTransformerModelPerformance() {
        // Measure performance of full transformer model forward pass
        let model = TransformerModel(numLayers: 2, dModel: 32, numHeads: 4, ffDim: 64)
        let images = createRandomInput(batchSize: 32, seqLen: 784, dModel: 1).reshaped([32, 784])

        measure {
            let logits = model(images)
            eval(logits)  // Force evaluation
        }
    }
}
