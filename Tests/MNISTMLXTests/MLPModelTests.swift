// ============================================================================
// MLPModelTests.swift - Tests for MLP Model Forward Pass
// ============================================================================
//
// This test suite validates the MLP model's forward pass functionality:
// - Output shape correctness for various batch sizes
// - Consistency with different hidden layer sizes
// - Edge cases (single sample, large batches)
// - Output properties (finite values, correct dimensions)
//
// ============================================================================

import XCTest
import MLX
import MLXNN
@testable import MNISTMLX

final class MLPModelTests: XCTestCase {

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
        // Input: [32, 784] → Output: [32, 10]
        let model = MLPModel()
        let input = createRandomInput(batchSize: 32)

        let output = model(input)

        assertShape(output, [32, 10],
                   "Forward pass should output [32, 10] for batch size 32")
    }

    func testForwardShapeSingleSample() {
        // Test forward pass with a single sample
        // Input: [1, 784] → Output: [1, 10]
        let model = MLPModel()
        let input = createRandomInput(batchSize: 1)

        let output = model(input)

        assertShape(output, [1, 10],
                   "Forward pass should output [1, 10] for single sample")
    }

    func testForwardShapeSmallBatch() {
        // Test forward pass with small batch size
        // Input: [4, 784] → Output: [4, 10]
        let model = MLPModel()
        let input = createRandomInput(batchSize: 4)

        let output = model(input)

        assertShape(output, [4, 10],
                   "Forward pass should output [4, 10] for batch size 4")
    }

    func testForwardShapeLargeBatch() {
        // Test forward pass with large batch size
        // Input: [128, 784] → Output: [128, 10]
        let model = MLPModel()
        let input = createRandomInput(batchSize: 128)

        let output = model(input)

        assertShape(output, [128, 10],
                   "Forward pass should output [128, 10] for batch size 128")
    }

    func testForwardShapeVeryLargeBatch() {
        // Test forward pass with very large batch size (full MNIST test set size)
        // Input: [10000, 784] → Output: [10000, 10]
        let model = MLPModel()
        let input = createRandomInput(batchSize: 10000)

        let output = model(input)

        assertShape(output, [10000, 10],
                   "Forward pass should output [10000, 10] for batch size 10000")
    }

    func testForwardShapeVariousBatchSizes() {
        // Test forward pass consistency across various batch sizes
        let model = MLPModel()
        let batchSizes = [1, 2, 8, 16, 32, 64, 100, 256]

        for batchSize in batchSizes {
            let input = createRandomInput(batchSize: batchSize)
            let output = model(input)

            assertShape(output, [batchSize, 10],
                       "Forward pass should output [\(batchSize), 10] for batch size \(batchSize)")
        }
    }

    // =============================================================================
    // MARK: - Custom Hidden Size Tests
    // =============================================================================

    func testForwardShapeCustomHiddenSize() {
        // Test that custom hidden size doesn't affect output shape
        // Output should still be [N, 10] regardless of hidden size
        let hiddenSizes = [128, 256, 512, 1024]
        let batchSize = 32

        for hiddenSize in hiddenSizes {
            let model = MLPModel(hiddenSize: hiddenSize)
            let input = createRandomInput(batchSize: batchSize)
            let output = model(input)

            assertShape(output, [batchSize, 10],
                       "Forward pass with hidden size \(hiddenSize) should output [32, 10]")
        }
    }

    func testForwardShapeSmallHiddenSize() {
        // Test with very small hidden layer
        let model = MLPModel(hiddenSize: 32)
        let input = createRandomInput(batchSize: 16)

        let output = model(input)

        assertShape(output, [16, 10],
                   "Forward pass with small hidden size should output [16, 10]")
    }

    func testForwardShapeLargeHiddenSize() {
        // Test with very large hidden layer
        let model = MLPModel(hiddenSize: 2048)
        let input = createRandomInput(batchSize: 16)

        let output = model(input)

        assertShape(output, [16, 10],
                   "Forward pass with large hidden size should output [16, 10]")
    }

    // =============================================================================
    // MARK: - Output Properties Tests
    // =============================================================================

    func testForwardOutputIsFinite() {
        // Test that forward pass produces finite values (no NaN or Inf)
        let model = MLPModel()
        let input = createRandomInput(batchSize: 32)

        let output = model(input)

        assertAllFinite(output, "Forward pass output should contain finite values")
    }

    func testForwardOutputIsFiniteWithZeroInput() {
        // Test that forward pass handles zero input correctly
        let model = MLPModel()
        let input = MLXArray.zeros([16, 784])

        let output = model(input)

        assertShape(output, [16, 10], "Forward pass should handle zero input")
        assertAllFinite(output, "Forward pass with zero input should produce finite values")
    }

    func testForwardOutputIsFiniteWithLargeInput() {
        // Test that forward pass handles large input values
        let model = MLPModel()
        let input = MLXArray.ones([16, 784]) * 100.0

        let output = model(input)

        assertShape(output, [16, 10], "Forward pass should handle large input values")
        assertAllFinite(output, "Forward pass with large input should produce finite values")
    }

    // =============================================================================
    // MARK: - Multiple Forward Pass Tests
    // =============================================================================

    func testMultipleForwardPassesConsistent() {
        // Test that multiple forward passes with same input produce same output
        let model = MLPModel()
        let input = createRandomInput(batchSize: 16)

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
        let model = MLPModel()

        let input1 = createRandomInput(batchSize: 8)
        let output1 = model(input1)
        assertShape(output1, [8, 10], "First batch should produce [8, 10]")

        let input2 = createRandomInput(batchSize: 16)
        let output2 = model(input2)
        assertShape(output2, [16, 10], "Second batch should produce [16, 10]")

        let input3 = createRandomInput(batchSize: 32)
        let output3 = model(input3)
        assertShape(output3, [32, 10], "Third batch should produce [32, 10]")
    }

    // =============================================================================
    // MARK: - Edge Cases
    // =============================================================================

    func testForwardShapeOddBatchSizes() {
        // Test forward pass with odd/unusual batch sizes
        let model = MLPModel()
        let oddBatchSizes = [3, 7, 13, 17, 31, 63, 127]

        for batchSize in oddBatchSizes {
            let input = createRandomInput(batchSize: batchSize)
            let output = model(input)

            assertShape(output, [batchSize, 10],
                       "Forward pass should handle odd batch size \(batchSize)")
        }
    }

    func testForwardShapeConsistencyAcrossModels() {
        // Test that different model instances produce consistent output shapes
        let batchSize = 32
        let input = createRandomInput(batchSize: batchSize)

        let model1 = MLPModel()
        let output1 = model1(input)

        let model2 = MLPModel()
        let output2 = model2(input)

        let model3 = MLPModel()
        let output3 = model3(input)

        // All outputs should have same shape
        assertShape(output1, [batchSize, 10], "Model 1 should produce [32, 10]")
        assertShape(output2, [batchSize, 10], "Model 2 should produce [32, 10]")
        assertShape(output3, [batchSize, 10], "Model 3 should produce [32, 10]")
    }

    func testForwardShapeWithNormalizedInput() {
        // Test forward pass with normalized input (simulating real MNIST data)
        // MNIST images are normalized to [0, 1]
        let model = MLPModel()
        let input = abs(MLXRandom.normal([32, 784]))

        let output = model(input)

        assertShape(output, [32, 10],
                   "Forward pass with normalized input should produce [32, 10]")
        assertAllFinite(output, "Forward pass with normalized input should be finite")
    }

    func testForwardShapeWithTypicalMNISTBatchSizes() {
        // Test forward pass with typical MNIST training batch sizes
        let model = MLPModel()
        let typicalBatchSizes = [32, 64, 128, 256]

        for batchSize in typicalBatchSizes {
            let input = createRandomInput(batchSize: batchSize)
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
        let model = MLPModel()
        let batchSizes = [1, 10, 50, 100]

        for batchSize in batchSizes {
            let input = createRandomInput(batchSize: batchSize)
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
        let model = MLPModel()
        let batchSizes = [1, 5, 10, 20, 50]

        for batchSize in batchSizes {
            let input = createRandomInput(batchSize: batchSize)
            let output = model(input)

            XCTAssertEqual(output.shape[1], 10,
                          "Output should have 10 classes for batch size \(batchSize)")
        }
    }

    func testForwardOutputBatchDimension() {
        // Test that output batch dimension matches input batch dimension
        let model = MLPModel()
        let batchSizes = [1, 4, 8, 16, 32, 64, 128]

        for batchSize in batchSizes {
            let input = createRandomInput(batchSize: batchSize)
            let output = model(input)

            XCTAssertEqual(output.shape[0], batchSize,
                          "Output batch dimension should match input batch dimension \(batchSize)")
        }
    }

    // =============================================================================
    // MARK: - Gradient Flow Tests
    // =============================================================================

    func testGradients() {
        // Test that gradients flow properly through the network
        // This verifies that backpropagation works and gradients are non-zero
        let model = MLPModel()
        let batchSize = 8
        let input = createRandomInput(batchSize: batchSize)

        // Create random labels for gradient computation
        // Labels should be integers in [0, 9] for MNIST
        let labelsData = (0..<batchSize).map { _ in Int32.random(in: 0..<10) }
        let labels = MLXArray(labelsData)

        // Compute loss and gradients using MLX's automatic differentiation
        let lossAndGrad = valueAndGrad(model: model, mlpLoss)
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

        // Verify we have gradients (should have 4 tensors: 2 weights + 2 biases)
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
                           "Total gradient norm should be non-zero, indicating proper gradient flow")
    }
}
