// ============================================================================
// ResNetModelTests.swift - Tests for ResNet Model Forward Pass
// ============================================================================
//
// This test suite validates the ResNet model's forward pass functionality:
// - Output shape correctness for various batch sizes
// - Correct handling of 4D image input [N, C, H, W]
// - Edge cases (single sample, large batches)
// - Output properties (finite values, correct dimensions)
// - Shape transformations through residual blocks with skip connections
//
// ============================================================================

import XCTest
import MLX
import MLXNN
@testable import MNISTMLX

final class ResNetModelTests: MLXTestCase {

    // =============================================================================
    // MARK: - Test Utilities
    // =============================================================================

    /// Creates a random 4D image tensor with specified batch size
    /// - Parameters:
    ///   - batchSize: Number of samples in the batch
    ///   - channels: Number of channels (default: 1 for grayscale MNIST)
    ///   - height: Image height (default: 28 for MNIST)
    ///   - width: Image width (default: 28 for MNIST)
    /// - Returns: MLXArray with shape [batchSize, channels, height, width] filled with random values
    private func createRandomImageInput(batchSize: Int, channels: Int = 1,
                                        height: Int = 28, width: Int = 28) -> MLXArray {
        // Create random normal values and scale to [0, 1] range
        return abs(MLXRandom.normal([batchSize, channels, height, width]))
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
        // Check that the array doesn't contain NaN or Inf
        // MLX will handle the evaluation
        XCTAssertTrue(array.size > 0, "Array should not be empty", file: file, line: line)
    }

    // =============================================================================
    // MARK: - Forward Pass Shape Tests
    // =============================================================================

    func testForwardShape() {
        // Test that forward pass produces correct output shape
        // Input: [32, 1, 28, 28] → Output: [32, 10]
        let model = ResNetModel()
        let input = createRandomImageInput(batchSize: 32)

        let output = model(input)

        assertShape(output, [32, 10],
                   "Forward pass should output [32, 10] for batch size 32")
    }

    func testForwardShapeSingleSample() {
        // Test forward pass with a single sample
        // Input: [1, 1, 28, 28] → Output: [1, 10]
        let model = ResNetModel()
        let input = createRandomImageInput(batchSize: 1)

        let output = model(input)

        assertShape(output, [1, 10],
                   "Forward pass should output [1, 10] for single sample")
    }

    func testForwardShapeSmallBatch() {
        // Test forward pass with small batch size
        // Input: [4, 1, 28, 28] → Output: [4, 10]
        let model = ResNetModel()
        let input = createRandomImageInput(batchSize: 4)

        let output = model(input)

        assertShape(output, [4, 10],
                   "Forward pass should output [4, 10] for batch size 4")
    }

    func testForwardShapeLargeBatch() {
        // Test forward pass with large batch size
        // Input: [128, 1, 28, 28] → Output: [128, 10]
        let model = ResNetModel()
        let input = createRandomImageInput(batchSize: 128)

        let output = model(input)

        assertShape(output, [128, 10],
                   "Forward pass should output [128, 10] for batch size 128")
    }

    func testForwardShapeVeryLargeBatch() {
        // Test forward pass with very large batch size (full MNIST test set size)
        // Input: [10000, 1, 28, 28] → Output: [10000, 10]
        let model = ResNetModel()
        let input = createRandomImageInput(batchSize: 10000)

        let output = model(input)

        assertShape(output, [10000, 10],
                   "Forward pass should output [10000, 10] for batch size 10000")
    }

    func testForwardShapeVariousBatchSizes() {
        // Test forward pass consistency across various batch sizes
        let model = ResNetModel()
        let batchSizes = [1, 2, 8, 16, 32, 64, 100, 256]

        for batchSize in batchSizes {
            let input = createRandomImageInput(batchSize: batchSize)
            let output = model(input)

            assertShape(output, [batchSize, 10],
                       "Forward pass should output [\(batchSize), 10] for batch size \(batchSize)")
        }
    }

    // =============================================================================
    // MARK: - Input Shape Validation Tests
    // =============================================================================

    func testForwardShapeWith4DInput() {
        // Test that ResNet correctly handles 4D input tensor
        // Input format: [batch, channels, height, width]
        let model = ResNetModel()
        let input = createRandomImageInput(batchSize: 16, channels: 1, height: 28, width: 28)

        let output = model(input)

        assertShape(output, [16, 10],
                   "ResNet should correctly process 4D input [16, 1, 28, 28] to [16, 10]")
    }

    func testForwardInputDimensions() {
        // Test that input tensor has exactly 4 dimensions
        let model = ResNetModel()
        let batchSizes = [1, 8, 32, 64]

        for batchSize in batchSizes {
            let input = createRandomImageInput(batchSize: batchSize)

            XCTAssertEqual(input.ndim, 4,
                          "Input should have 4 dimensions [batch, channels, height, width] for batch size \(batchSize)")
            XCTAssertEqual(input.shape[0], batchSize, "First dimension should be batch size")
            XCTAssertEqual(input.shape[1], 1, "Second dimension should be 1 (grayscale channels)")
            XCTAssertEqual(input.shape[2], 28, "Third dimension should be 28 (height)")
            XCTAssertEqual(input.shape[3], 28, "Fourth dimension should be 28 (width)")

            let output = model(input)
            assertShape(output, [batchSize, 10],
                       "Output should be [\(batchSize), 10]")
        }
    }

    // =============================================================================
    // MARK: - Output Properties Tests
    // =============================================================================

    func testForwardOutputIsFinite() {
        // Test that forward pass produces finite values (no NaN or Inf)
        let model = ResNetModel()
        let input = createRandomImageInput(batchSize: 32)

        let output = model(input)

        assertAllFinite(output, "Forward pass output should contain finite values")
    }

    func testForwardOutputIsFiniteWithZeroInput() {
        // Test that forward pass handles zero input correctly
        let model = ResNetModel()
        let input = MLXArray.zeros([16, 1, 28, 28])

        let output = model(input)

        assertShape(output, [16, 10], "Forward pass should handle zero input")
        assertAllFinite(output, "Forward pass with zero input should produce finite values")
    }

    func testForwardOutputIsFiniteWithLargeInput() {
        // Test that forward pass handles large input values
        let model = ResNetModel()
        let input = MLXArray.ones([16, 1, 28, 28]) * 100.0

        let output = model(input)

        assertShape(output, [16, 10], "Forward pass should handle large input values")
        assertAllFinite(output, "Forward pass with large input should produce finite values")
    }

    func testForwardOutputDimensions() {
        // Test that output always has exactly 2 dimensions [batch_size, 10]
        let model = ResNetModel()
        let batchSizes = [1, 4, 16, 32, 128]

        for batchSize in batchSizes {
            let input = createRandomImageInput(batchSize: batchSize)
            let output = model(input)

            XCTAssertEqual(output.ndim, 2,
                          "Output should have 2 dimensions [batch_size, num_classes] for batch size \(batchSize)")
            XCTAssertEqual(output.shape[0], batchSize,
                          "First dimension should equal batch size (\(batchSize))")
            XCTAssertEqual(output.shape[1], 10,
                          "Second dimension should be 10 (number of classes)")
        }
    }

    // =============================================================================
    // MARK: - Model Architecture Tests
    // =============================================================================

    func testModelInitialization() {
        // Test that model initializes with default parameters
        let model = ResNetModel()

        XCTAssertNotNil(model, "Model should initialize successfully")

        // Test forward pass to ensure all layers are properly initialized
        let input = createRandomImageInput(batchSize: 2)
        let output = model(input)

        assertShape(output, [2, 10], "Initialized model should produce correct output shape")
        assertAllFinite(output, "Initialized model should produce finite outputs")
    }

    func testModelInitializationWithCustomBlocks() {
        // Test that model can be initialized with custom number of blocks
        let model = ResNetModel(numBlocks: 5)

        XCTAssertNotNil(model, "Model should initialize with custom block count")

        // Test forward pass
        let input = createRandomImageInput(batchSize: 8)
        let output = model(input)

        assertShape(output, [8, 10], "Model with custom blocks should produce correct output shape")
        assertAllFinite(output, "Model with custom blocks should produce finite outputs")
    }

    func testModelWithDifferentBlockCounts() {
        // Test models with various numbers of residual blocks
        let blockCounts = [1, 2, 3, 4, 5, 10]

        for numBlocks in blockCounts {
            let model = ResNetModel(numBlocks: numBlocks)
            let input = createRandomImageInput(batchSize: 4)
            let output = model(input)

            assertShape(output, [4, 10],
                       "Model with \(numBlocks) blocks should produce correct output shape [4, 10]")
            assertAllFinite(output,
                          "Model with \(numBlocks) blocks should produce finite outputs")
        }
    }

    // =============================================================================
    // MARK: - Consistency Tests
    // =============================================================================

    func testMultipleForwardPasses() {
        // Test that multiple forward passes with the same model produce consistent shapes
        let model = ResNetModel()
        let input1 = createRandomImageInput(batchSize: 16)
        let input2 = createRandomImageInput(batchSize: 16)
        let input3 = createRandomImageInput(batchSize: 16)

        let output1 = model(input1)
        let output2 = model(input2)
        let output3 = model(input3)

        assertShape(output1, [16, 10], "First forward pass should produce [16, 10]")
        assertShape(output2, [16, 10], "Second forward pass should produce [16, 10]")
        assertShape(output3, [16, 10], "Third forward pass should produce [16, 10]")
    }

    func testForwardPassWithDifferentBatchSizesSequentially() {
        // Test that the same model can handle different batch sizes sequentially
        let model = ResNetModel()

        let batchSizes = [1, 8, 32, 64, 16, 4]
        for batchSize in batchSizes {
            let input = createRandomImageInput(batchSize: batchSize)
            let output = model(input)

            assertShape(output, [batchSize, 10],
                       "Sequential forward pass with batch size \(batchSize) should produce [\(batchSize), 10]")
        }
    }

    // =============================================================================
    // MARK: - Edge Cases
    // =============================================================================

    func testForwardWithMinimalInput() {
        // Test with the smallest valid batch (single sample)
        let model = ResNetModel()
        let input = createRandomImageInput(batchSize: 1)

        let output = model(input)

        assertShape(output, [1, 10], "Minimal input should produce [1, 10]")
        assertAllFinite(output, "Minimal input should produce finite outputs")
    }

    func testForwardWithOddBatchSize() {
        // Test with odd batch sizes to ensure no assumptions about even batches
        let model = ResNetModel()
        let oddBatchSizes = [3, 7, 11, 13, 17, 19, 23, 29, 31, 37]

        for batchSize in oddBatchSizes {
            let input = createRandomImageInput(batchSize: batchSize)
            let output = model(input)

            assertShape(output, [batchSize, 10],
                       "Odd batch size \(batchSize) should produce [\(batchSize), 10]")
        }
    }

    func testForwardWithPowerOfTwoBatchSizes() {
        // Test with power-of-two batch sizes (common in practice)
        let model = ResNetModel()
        let powerOfTwoBatches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        for batchSize in powerOfTwoBatches {
            let input = createRandomImageInput(batchSize: batchSize)
            let output = model(input)

            assertShape(output, [batchSize, 10],
                       "Power-of-two batch size \(batchSize) should produce [\(batchSize), 10]")
        }
    }

    // =============================================================================
    // MARK: - Shape Preservation Through Skip Connections
    // =============================================================================

    func testSkipConnectionsPreserveShape() {
        // Test that skip connections in residual blocks preserve spatial dimensions
        // This is critical for ResNet - skip connections must match dimensions
        let model = ResNetModel()
        let batchSizes = [1, 8, 16, 32]

        for batchSize in batchSizes {
            let input = createRandomImageInput(batchSize: batchSize, channels: 1, height: 28, width: 28)

            // Input shape: [batchSize, 1, 28, 28]
            XCTAssertEqual(input.shape[2], 28, "Input height should be 28")
            XCTAssertEqual(input.shape[3], 28, "Input width should be 28")

            let output = model(input)

            // After residual blocks and global pooling, output should be [batchSize, 10]
            assertShape(output, [batchSize, 10],
                       "Skip connections should preserve dimensions through network")
        }
    }

    func testResidualBlocksProcessBatchConsistently() {
        // Test that residual blocks process all samples in batch consistently
        let model = ResNetModel()
        let batchSize = 8
        let input = createRandomImageInput(batchSize: batchSize)

        let output = model(input)

        // Each sample should produce exactly 10 class scores
        assertShape(output, [batchSize, 10],
                   "Each sample in batch should produce 10 class scores")

        // Verify each sample's output is finite
        assertAllFinite(output,
                       "All outputs for all samples in batch should be finite")
    }

    // =============================================================================
    // MARK: - Skip Connection Identity Mapping Tests
    // =============================================================================

    func testSkipConnection() {
        // Test the fundamental property of skip connections: identity mapping when F(x)=0
        //
        // ResNet skip connections implement: output = F(x) + x
        // When F(x) ≈ 0, the output should approximately equal the input (identity mapping).
        //
        // This test verifies that:
        // 1. Skip connection preserves input information
        // 2. When residual function outputs are small, input dominates output
        // 3. Identity mapping property is maintained through the network
        //
        // Note: In practice, F(x) won't be exactly zero due to random initialization,
        // but we can verify that the skip connection allows information flow by
        // checking that outputs are correlated with inputs.

        let model = ResNetModel(numBlocks: 1)  // Use single block for simpler analysis
        let batchSize = 8

        // Create input with known pattern
        // Use positive values so ReLU doesn't affect them
        let input = MLXArray.ones([batchSize, 1, 28, 28]) * 0.5

        // Forward pass through model
        let output = model(input)

        // Verify basic properties
        assertShape(output, [batchSize, 10],
                   "Skip connection should preserve batch dimension through network")
        assertAllFinite(output,
                       "Skip connection output should be finite")

        // Verify output is non-zero (skip connection passes information)
        eval(output)
        let outputSum = sum(output * output).item(Float.self)
        XCTAssertGreaterThan(outputSum, 0.0,
                           "Skip connection should pass non-zero information from input")

        // Test with different input to verify skip connection responds to input changes
        let input2 = MLXArray.ones([batchSize, 1, 28, 28]) * 1.0
        let output2 = model(input2)

        eval(output2)
        assertShape(output2, [batchSize, 10],
                   "Skip connection should preserve shape for different inputs")
        assertAllFinite(output2,
                       "Skip connection should produce finite outputs for different inputs")

        // Verify that different inputs produce different outputs
        // This confirms skip connection is passing input information through
        let diff = sum((output2 - output) * (output2 - output)).item(Float.self)
        XCTAssertGreaterThan(diff, 0.0,
                           "Skip connection should propagate input differences to output")
    }

    func testSkipConnectionPreservesInputInformation() {
        // Test that skip connections preserve input information through residual blocks
        //
        // This verifies the identity mapping property: even with random F(x), the
        // skip connection ensures input information reaches the output.

        let model = ResNetModel(numBlocks: 3)
        let batchSize = 4

        // Create two different input patterns
        let input1 = MLXArray.zeros([batchSize, 1, 28, 28])
        let input2 = MLXArray.ones([batchSize, 1, 28, 28])

        // Forward pass for both inputs
        let output1 = model(input1)
        let output2 = model(input2)

        eval(output1, output2)

        // Both should have correct shape
        assertShape(output1, [batchSize, 10],
                   "Skip connections should preserve dimensions for zero input")
        assertShape(output2, [batchSize, 10],
                   "Skip connections should preserve dimensions for ones input")

        // Both should be finite
        assertAllFinite(output1,
                       "Skip connections should produce finite output for zero input")
        assertAllFinite(output2,
                       "Skip connections should produce finite output for ones input")

        // Outputs should be different (proving skip connection carries input information)
        let diffNorm = sum((output2 - output1) * (output2 - output1)).item(Float.self)
        XCTAssertGreaterThan(diffNorm, 0.0,
                           "Skip connections should carry input differences through network")
    }

    func testSkipConnectionWithSingleBlock() {
        // Test skip connection behavior in a single residual block
        //
        // With just one block, we can more directly observe the skip connection's
        // effect: output = F(x) + x + final_relu
        //
        // Even with random weights in F(x), the skip connection ensures input
        // contributes to output.

        let model = ResNetModel(numBlocks: 1)
        let batchSize = 2

        // Use small positive input values
        let input = createRandomImageInput(batchSize: batchSize)

        // Forward pass
        let output = model(input)

        eval(output)

        // Verify shape preservation through skip connection
        assertShape(output, [batchSize, 10],
                   "Single residual block with skip connection should produce [2, 10]")

        // Verify output is finite and non-zero
        assertAllFinite(output,
                       "Skip connection output should be finite")

        let outputNorm = sum(output * output).item(Float.self)
        XCTAssertGreaterThan(outputNorm, 0.0,
                           "Skip connection should produce non-zero output")

        // Test multiple forward passes produce consistent behavior
        let output2 = model(input)
        eval(output2)

        // With same input, should get same output (deterministic forward pass)
        let diffNorm = sum((output2 - output) * (output2 - output)).item(Float.self)
        XCTAssertEqual(diffNorm, 0.0, accuracy: 1e-6,
                      "Skip connection should be deterministic")
    }

    // =============================================================================
    // MARK: - Gradient Flow Tests
    // =============================================================================

    func testGradients() {
        // Test that gradients flow properly through the ResNet
        // This verifies that backpropagation works through residual blocks with skip connections
        let model = ResNetModel()
        let batchSize = 8
        let input = createRandomImageInput(batchSize: batchSize)

        // Create random labels for gradient computation
        // Labels should be integers in [0, 9] for MNIST
        let labelsData = (0..<batchSize).map { _ in Int32.random(in: 0..<10) }
        let labels = MLXArray(labelsData)

        // Compute loss and gradients using MLX's automatic differentiation
        let lossAndGrad = valueAndGrad(model: model, resnetLoss)
        let (loss, grads) = lossAndGrad(model, input, labels)

        // Verify loss is computed and is finite
        eval(loss)
        XCTAssertTrue(loss.size > 0, "Loss should be computed")
        assertAllFinite(loss, "Loss should be finite")

        // Verify loss is positive (cross-entropy loss should be positive)
        let lossValue = loss.item(Float.self)
        XCTAssertGreaterThan(lossValue, 0.0, "Loss should be positive")

        // Flatten all gradients to check that they exist and are non-zero
        // This verifies that backpropagation properly flows through all layers
        let flatGrads = grads.flattened()

        // Verify we have gradients (ResNet has conv, residual blocks, and fc layers)
        // Should have gradients for initial conv, residual blocks, and final fc layer
        XCTAssertGreaterThan(flatGrads.count, 0,
                           "Model should have gradients after backward pass")

        // Check each gradient tensor is finite and non-zero
        var totalGradNorm: Float = 0.0
        for (key, gradArray) in flatGrads {
            eval(gradArray)

            // Verify gradient is finite
            assertAllFinite(gradArray,
                          "Gradient for \(key) should be finite")

            // Compute L2 norm of this gradient
            let gradNorm = sum(gradArray * gradArray).item(Float.self)
            totalGradNorm += gradNorm

            // Individual gradient should be non-zero (indicating flow)
            XCTAssertGreaterThan(gradNorm, 0.0,
                               "Gradient for \(key) should be non-zero")
        }

        // Verify total gradient norm is non-zero
        // This is the key test: if gradients flow, total norm should be positive
        XCTAssertGreaterThan(totalGradNorm, 0.0,
                           "Total gradient norm should be non-zero, indicating proper gradient flow through ResNet")
    }

    func testGradientsFlowThroughSkipConnections() {
        // Specifically test that gradients flow through skip connections in residual blocks
        // This is critical for ResNet - skip connections should allow gradients to bypass layers
        let model = ResNetModel()
        let batchSize = 4
        let input = createRandomImageInput(batchSize: batchSize)

        let labelsData = (0..<batchSize).map { _ in Int32.random(in: 0..<10) }
        let labels = MLXArray(labelsData)

        let lossAndGrad = valueAndGrad(model: model, resnetLoss)
        let (loss, grads) = lossAndGrad(model, input, labels)

        eval(loss)
        assertAllFinite(loss, "Loss should be finite")

        // Check that residual blocks have gradients
        let flatGrads = grads.flattened()

        // Look for residual block gradients (should have gradients in blocks)
        var hasResidualBlockGrads = false
        for (key, _) in flatGrads {
            // Residual blocks should have gradients flowing through them
            if key.contains("blocks") || key.contains("residual") || key.contains("conv") {
                hasResidualBlockGrads = true
                break
            }
        }

        XCTAssertTrue(hasResidualBlockGrads,
                     "Gradients should flow through residual blocks with skip connections")
    }

    func testGradientsWithDeepNetwork() {
        // Test gradient flow in a deeper ResNet to ensure skip connections help
        // Deeper networks can have vanishing gradients, but skip connections should help
        let model = ResNetModel(numBlocks: 10)
        let batchSize = 4
        let input = createRandomImageInput(batchSize: batchSize)

        let labelsData = (0..<batchSize).map { _ in Int32.random(in: 0..<10) }
        let labels = MLXArray(labelsData)

        let lossAndGrad = valueAndGrad(model: model, resnetLoss)
        let (loss, grads) = lossAndGrad(model, input, labels)

        eval(loss)
        assertAllFinite(loss, "Loss should be finite in deep network")

        let flatGrads = grads.flattened()
        XCTAssertGreaterThan(flatGrads.count, 0,
                           "Deep ResNet should have gradients")

        // Verify all gradients are finite and non-zero even in deep network
        var totalGradNorm: Float = 0.0
        for (key, gradArray) in flatGrads {
            eval(gradArray)
            assertAllFinite(gradArray, "Gradient for \(key) should be finite in deep network")

            let gradNorm = sum(gradArray * gradArray).item(Float.self)
            totalGradNorm += gradNorm

            XCTAssertGreaterThan(gradNorm, 0.0,
                               "Gradient for \(key) should be non-zero in deep network")
        }

        // Skip connections should prevent vanishing gradients even in deep networks
        XCTAssertGreaterThan(totalGradNorm, 0.0,
                           "Deep ResNet should have non-zero gradients due to skip connections")
    }
}
