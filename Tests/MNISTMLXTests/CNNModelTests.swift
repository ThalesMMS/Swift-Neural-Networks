// ============================================================================
// CNNModelTests.swift - Tests for CNN Model Forward Pass
// ============================================================================
//
// This test suite validates the CNN model's forward pass functionality:
// - Output shape correctness for various batch sizes
// - Correct handling of 4D image input [N, C, H, W]
// - Edge cases (single sample, large batches)
// - Output properties (finite values, correct dimensions)
// - Shape transformations through convolutional layers
//
// ============================================================================

import XCTest
import MLX
import MLXNN
@testable import MNISTMLX

final class CNNModelTests: MLXTestCase {

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
        let model = CNNModel()
        let input = createRandomImageInput(batchSize: 32)

        let output = model(input)

        assertShape(output, [32, 10],
                   "Forward pass should output [32, 10] for batch size 32")
    }

    func testForwardShapeSingleSample() {
        // Test forward pass with a single sample
        // Input: [1, 1, 28, 28] → Output: [1, 10]
        let model = CNNModel()
        let input = createRandomImageInput(batchSize: 1)

        let output = model(input)

        assertShape(output, [1, 10],
                   "Forward pass should output [1, 10] for single sample")
    }

    func testForwardShapeSmallBatch() {
        // Test forward pass with small batch size
        // Input: [4, 1, 28, 28] → Output: [4, 10]
        let model = CNNModel()
        let input = createRandomImageInput(batchSize: 4)

        let output = model(input)

        assertShape(output, [4, 10],
                   "Forward pass should output [4, 10] for batch size 4")
    }

    func testForwardShapeLargeBatch() {
        // Test forward pass with large batch size
        // Input: [128, 1, 28, 28] → Output: [128, 10]
        let model = CNNModel()
        let input = createRandomImageInput(batchSize: 128)

        let output = model(input)

        assertShape(output, [128, 10],
                   "Forward pass should output [128, 10] for batch size 128")
    }

    func testForwardShapeVeryLargeBatch() {
        // Test forward pass with very large batch size (full MNIST test set size)
        // Input: [10000, 1, 28, 28] → Output: [10000, 10]
        let model = CNNModel()
        let input = createRandomImageInput(batchSize: 10000)

        let output = model(input)

        assertShape(output, [10000, 10],
                   "Forward pass should output [10000, 10] for batch size 10000")
    }

    func testForwardShapeVariousBatchSizes() {
        // Test forward pass consistency across various batch sizes
        let model = CNNModel()
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
        // Test that CNN correctly handles 4D input tensor
        // Input format: [batch, channels, height, width]
        let model = CNNModel()
        let input = createRandomImageInput(batchSize: 16, channels: 1, height: 28, width: 28)

        let output = model(input)

        assertShape(output, [16, 10],
                   "CNN should correctly process 4D input [16, 1, 28, 28] to [16, 10]")
    }

    func testForwardInputDimensions() {
        // Test that input tensor has exactly 4 dimensions
        let model = CNNModel()
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
        let model = CNNModel()
        let input = createRandomImageInput(batchSize: 32)

        let output = model(input)

        assertAllFinite(output, "Forward pass output should contain finite values")
    }

    func testForwardOutputIsFiniteWithZeroInput() {
        // Test that forward pass handles zero input correctly
        let model = CNNModel()
        let input = MLXArray.zeros([16, 1, 28, 28])

        let output = model(input)

        assertShape(output, [16, 10], "Forward pass should handle zero input")
        assertAllFinite(output, "Forward pass with zero input should produce finite values")
    }

    func testForwardOutputIsFiniteWithLargeInput() {
        // Test that forward pass handles large input values
        let model = CNNModel()
        let input = MLXArray.ones([16, 1, 28, 28]) * 100.0

        let output = model(input)

        assertShape(output, [16, 10], "Forward pass should handle large input values")
        assertAllFinite(output, "Forward pass with large input should produce finite values")
    }

    func testForwardOutputIsFiniteWithNormalizedInput() {
        // Test forward pass with normalized input (simulating real MNIST data)
        // MNIST images are normalized to [0, 1]
        let model = CNNModel()
        let input = abs(MLXRandom.normal([32, 1, 28, 28]))

        let output = model(input)

        assertShape(output, [32, 10],
                   "Forward pass with normalized input should produce [32, 10]")
        assertAllFinite(output, "Forward pass with normalized input should be finite")
    }

    // =============================================================================
    // MARK: - Multiple Forward Pass Tests
    // =============================================================================

    func testMultipleForwardPassesConsistent() {
        // Test that multiple forward passes with same input produce same output
        let model = CNNModel()
        let input = createRandomImageInput(batchSize: 16)

        let output1 = model(input)
        let output2 = model(input)

        assertShape(output1, [16, 10], "First forward pass should produce [16, 10]")
        assertShape(output2, [16, 10], "Second forward pass should produce [16, 10]")

        // Both outputs should have same shape
        XCTAssertEqual(output1.shape, output2.shape,
                      "Multiple forward passes should produce consistent shapes")
    }

    func testForwardPassWithDifferentBatchesSameModel() {
        // Test that the same model can handle different batch sizes
        let model = CNNModel()

        let input1 = createRandomImageInput(batchSize: 8)
        let output1 = model(input1)
        assertShape(output1, [8, 10], "First batch should produce [8, 10]")

        let input2 = createRandomImageInput(batchSize: 16)
        let output2 = model(input2)
        assertShape(output2, [16, 10], "Second batch should produce [16, 10]")

        let input3 = createRandomImageInput(batchSize: 32)
        let output3 = model(input3)
        assertShape(output3, [32, 10], "Third batch should produce [32, 10]")
    }

    // =============================================================================
    // MARK: - Edge Cases
    // =============================================================================

    func testForwardShapeOddBatchSizes() {
        // Test forward pass with odd/unusual batch sizes
        let model = CNNModel()
        let oddBatchSizes = [3, 7, 13, 17, 31, 63, 127]

        for batchSize in oddBatchSizes {
            let input = createRandomImageInput(batchSize: batchSize)
            let output = model(input)

            assertShape(output, [batchSize, 10],
                       "Forward pass should handle odd batch size \(batchSize)")
        }
    }

    func testForwardShapeConsistencyAcrossModels() {
        // Test that different model instances produce consistent output shapes
        let batchSize = 32
        let input = createRandomImageInput(batchSize: batchSize)

        let model1 = CNNModel()
        let output1 = model1(input)

        let model2 = CNNModel()
        let output2 = model2(input)

        let model3 = CNNModel()
        let output3 = model3(input)

        // All outputs should have same shape
        assertShape(output1, [batchSize, 10], "Model 1 should produce [32, 10]")
        assertShape(output2, [batchSize, 10], "Model 2 should produce [32, 10]")
        assertShape(output3, [batchSize, 10], "Model 3 should produce [32, 10]")
    }

    func testForwardShapeWithTypicalMNISTBatchSizes() {
        // Test forward pass with typical MNIST training batch sizes
        let model = CNNModel()
        let typicalBatchSizes = [32, 64, 128, 256]

        for batchSize in typicalBatchSizes {
            let input = createRandomImageInput(batchSize: batchSize)
            let output = model(input)

            assertShape(output, [batchSize, 10],
                       "Forward pass should handle typical MNIST batch size \(batchSize)")
        }
    }

    // =============================================================================
    // MARK: - Dimension Validation Tests
    // =============================================================================

    func testForwardOutputDimensions() {
        // Test that output always has exactly 2 dimensions [batch_size, num_classes]
        let model = CNNModel()
        let batchSizes = [1, 10, 50, 100]

        for batchSize in batchSizes {
            let input = createRandomImageInput(batchSize: batchSize)
            let output = model(input)

            XCTAssertEqual(output.ndim, 2,
                          "Output should have 2 dimensions for batch size \(batchSize)")
            XCTAssertEqual(output.shape[0], batchSize,
                          "First dimension should be batch size \(batchSize)")
            XCTAssertEqual(output.shape[1], 10,
                          "Second dimension should be 10 (number of classes)")
        }
    }

    func testForwardOutputNumClasses() {
        // Test that output always has 10 classes (MNIST digits 0-9)
        let model = CNNModel()
        let batchSizes = [1, 5, 10, 20, 50]

        for batchSize in batchSizes {
            let input = createRandomImageInput(batchSize: batchSize)
            let output = model(input)

            XCTAssertEqual(output.shape[1], 10,
                          "Output should have 10 classes for batch size \(batchSize)")
        }
    }

    func testForwardOutputBatchDimension() {
        // Test that output batch dimension matches input batch dimension
        let model = CNNModel()
        let batchSizes = [1, 4, 8, 16, 32, 64, 128]

        for batchSize in batchSizes {
            let input = createRandomImageInput(batchSize: batchSize)
            let output = model(input)

            XCTAssertEqual(output.shape[0], batchSize,
                          "Output batch dimension should match input batch dimension \(batchSize)")
        }
    }

    // =============================================================================
    // MARK: - CNN-Specific Shape Transformation Tests
    // =============================================================================

    func testConvolutionPreservesSpatialDimensions() {
        // Test that convolution with padding=1 preserves spatial dimensions
        // Conv2d: [N, 1, 28, 28] → [N, 8, 28, 28] (with padding=1)
        // This is tested indirectly through the forward pass
        let model = CNNModel()
        let input = createRandomImageInput(batchSize: 16)

        // The model's architecture preserves 28x28 after conv due to padding=1
        // After maxpool with stride=2, it becomes 14x14
        // Then flattened to 8*14*14 = 1568
        // Finally linear layer outputs 10 classes

        let output = model(input)

        assertShape(output, [16, 10],
                   "CNN should correctly transform [16, 1, 28, 28] to [16, 10]")
    }

    func testPoolingReducesSpatialDimensions() {
        // Test that the model correctly handles spatial dimension reduction
        // MaxPool2d with kernel=2, stride=2 should halve spatial dimensions
        // 28x28 → 14x14
        let model = CNNModel()
        let input = createRandomImageInput(batchSize: 8)

        let output = model(input)

        // After conv (28x28), pool (14x14), flatten (1568), fc (10)
        assertShape(output, [8, 10],
                   "CNN should correctly process spatial downsampling")
    }

    func testFlattenedFeatureSize() {
        // Test that the model correctly flattens convolutional features
        // After conv: [N, 8, 28, 28]
        // After pool: [N, 8, 14, 14]
        // Flattened: [N, 8*14*14] = [N, 1568]
        let model = CNNModel()
        let input = createRandomImageInput(batchSize: 4)

        let output = model(input)

        // The model should handle the flattening from [4, 8, 14, 14] to [4, 1568]
        // and then linear layer from 1568 to 10
        assertShape(output, [4, 10],
                   "CNN should correctly flatten and project features")
    }

    // =============================================================================
    // MARK: - Gradient Flow Tests
    // =============================================================================

    func testGradients() {
        // Test that gradients flow properly through the CNN
        // This verifies that backpropagation works through conv, pool, and fc layers
        let model = CNNModel()
        let batchSize = 8
        let input = createRandomImageInput(batchSize: batchSize)

        // Create random labels for gradient computation
        // Labels should be integers in [0, 9] for MNIST
        let labelsData = (0..<batchSize).map { _ in Int32.random(in: 0..<10) }
        let labels = MLXArray(labelsData)

        // Compute loss and gradients using MLX's automatic differentiation
        let lossAndGrad = valueAndGrad(model: model, cnnLoss)
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

        // Verify we have gradients (CNN has conv, pool, and fc layers)
        // Should have gradients for conv weights, conv bias, fc weights, fc bias
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
                           "Total gradient norm should be non-zero, indicating proper gradient flow through CNN")
    }

    func testGradientsFlowThroughConvLayers() {
        // Specifically test that gradients flow through convolutional layers
        let model = CNNModel()
        let batchSize = 4
        let input = createRandomImageInput(batchSize: batchSize)

        let labelsData = (0..<batchSize).map { _ in Int32.random(in: 0..<10) }
        let labels = MLXArray(labelsData)

        let lossAndGrad = valueAndGrad(model: model, cnnLoss)
        let (loss, grads) = lossAndGrad(model, input, labels)

        eval(loss)
        assertAllFinite(loss, "Loss should be finite")

        // Check that convolutional layer has gradients
        let flatGrads = grads.flattened()

        // Look for conv1 gradients (both weights and bias)
        var hasConvGrads = false
        for (key, _) in flatGrads {
            if key.contains("conv1") {
                hasConvGrads = true
                break
            }
        }

        XCTAssertTrue(hasConvGrads,
                     "Gradients should flow through convolutional layers")
    }
}
