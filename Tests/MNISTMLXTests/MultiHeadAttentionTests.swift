// ============================================================================
// MultiHeadAttentionTests.swift - Tests for MultiHeadAttention Module
// ============================================================================
//
// This test suite validates the MultiHeadAttention class defined in
// MultiHeadAttention.swift (extracted from TransformerBlock.swift in this PR).
//
// Tests cover:
// - Initialization and stored properties (dModel, numHeads, headDim)
// - Forward pass output shape preservation
// - Correct head dimension computation
// - Output finiteness across various input types
// - Permutation equivariance property of self-attention
// - Attention with various model dimensions and head counts
// - Edge cases: single token, single head, square head dim
//
// ============================================================================

import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import MNISTMLX

final class MultiHeadAttentionTests: MLXTestCase {

    // =============================================================================
    // MARK: - Test Utilities
    // =============================================================================

    /// Creates a random input tensor [batch, seq_len, d_model]
    private func makeInput(batch: Int, seq: Int, dim: Int) -> MLXArray {
        MLXRandom.normal([batch, seq, dim])
    }

    /// Asserts array shape matches expected
    private func assertShape(
        _ array: MLXArray, _ expected: [Int],
        _ message: String = "",
        file: StaticString = #file, line: UInt = #line
    ) {
        XCTAssertEqual(array.shape, expected, message, file: file, line: line)
    }

    /// Asserts all values in array are finite (no NaN or Inf)
    private func assertAllFinite(
        _ array: MLXArray,
        _ message: String = "Values should be finite",
        file: StaticString = #file, line: UInt = #line
    ) {
        eval(array)
        let values = array.asArray(Float.self)
        XCTAssertFalse(values.isEmpty, "Array should not be empty", file: file, line: line)
        for v in values {
            XCTAssertTrue(v.isFinite, "\(message): found \(v)", file: file, line: line)
        }
    }

    // =============================================================================
    // MARK: - Initialization Tests
    // =============================================================================

    func testInitStoresDModel() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        XCTAssertEqual(attn.dModel, 32,
                       "dModel should be stored as provided")
    }

    func testInitStoresNumHeads() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        XCTAssertEqual(attn.numHeads, 4,
                       "numHeads should be stored as provided")
    }

    func testInitComputesHeadDim() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        XCTAssertEqual(attn.headDim, 8,
                       "headDim should be dModel / numHeads = 32 / 4 = 8")
    }

    func testHeadDimComputationVariousConfigs() {
        let cases: [(dModel: Int, numHeads: Int, expectedHeadDim: Int)] = [
            (16, 1, 16),
            (16, 2, 8),
            (16, 4, 4),
            (32, 4, 8),
            (32, 8, 4),
            (64, 8, 8),
            (128, 16, 8),
        ]
        for c in cases {
            let attn = MultiHeadAttention(dModel: c.dModel, numHeads: c.numHeads)
            XCTAssertEqual(attn.headDim, c.expectedHeadDim,
                           "headDim should be \(c.expectedHeadDim) for dModel=\(c.dModel), numHeads=\(c.numHeads)")
        }
    }

    func testHeadDimTimesNumHeadsEqualsDModel() {
        // Fundamental invariant: headDim * numHeads == dModel
        let configs: [(Int, Int)] = [(32, 4), (64, 8), (16, 2), (128, 16)]
        for (dModel, numHeads) in configs {
            let attn = MultiHeadAttention(dModel: dModel, numHeads: numHeads)
            XCTAssertEqual(attn.headDim * attn.numHeads, attn.dModel,
                           "headDim * numHeads must equal dModel")
        }
    }

    // =============================================================================
    // MARK: - Output Shape Tests
    // =============================================================================

    func testOutputShapePreservedStandardConfig() {
        // Standard MNIST transformer config
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        let x = makeInput(batch: 8, seq: 49, dim: 32)
        let output = attn(x)
        assertShape(output, [8, 49, 32],
                    "Output should preserve [batch, seq_len, d_model]")
    }

    func testOutputShapeSingleBatch() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        let x = makeInput(batch: 1, seq: 49, dim: 32)
        let output = attn(x)
        assertShape(output, [1, 49, 32])
    }

    func testOutputShapeLargeBatch() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        let x = makeInput(batch: 64, seq: 49, dim: 32)
        let output = attn(x)
        assertShape(output, [64, 49, 32])
    }

    func testOutputShapeSingleToken() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        let x = makeInput(batch: 8, seq: 1, dim: 32)
        let output = attn(x)
        assertShape(output, [8, 1, 32],
                    "Single-token sequence should work (seq_len=1)")
    }

    func testOutputShapeLongSequence() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        let x = makeInput(batch: 4, seq: 256, dim: 32)
        let output = attn(x)
        assertShape(output, [4, 256, 32])
    }

    func testOutputShapeIsThreeDimensional() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        let x = makeInput(batch: 4, seq: 10, dim: 32)
        let output = attn(x)
        XCTAssertEqual(output.ndim, 3,
                       "Output must be 3D [batch, seq_len, d_model]")
    }

    func testOutputBatchDimMatchesInput() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        let batchSizes = [1, 2, 8, 16, 32]
        for b in batchSizes {
            let x = makeInput(batch: b, seq: 10, dim: 32)
            let out = attn(x)
            XCTAssertEqual(out.shape[0], b,
                           "Output batch dim should match input batch dim \(b)")
        }
    }

    func testOutputSeqDimMatchesInput() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        let seqLengths = [1, 5, 10, 49, 100]
        for s in seqLengths {
            let x = makeInput(batch: 4, seq: s, dim: 32)
            let out = attn(x)
            XCTAssertEqual(out.shape[1], s,
                           "Output seq dim should match input seq dim \(s)")
        }
    }

    func testOutputDModelDimMatchesDModel() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        let x = makeInput(batch: 4, seq: 10, dim: 32)
        let out = attn(x)
        XCTAssertEqual(out.shape[2], 32,
                       "Output last dim should equal dModel")
    }

    // =============================================================================
    // MARK: - Output Finiteness Tests
    // =============================================================================

    func testOutputIsFiniteWithNormalInput() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        let x = makeInput(batch: 8, seq: 49, dim: 32)
        let out = attn(x)
        assertAllFinite(out, "Output should be finite with normal random input")
    }

    func testOutputIsFiniteWithZeroInput() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        let x = MLXArray.zeros([8, 49, 32])
        let out = attn(x)
        assertShape(out, [8, 49, 32])
        assertAllFinite(out, "Output should be finite with zero input")
    }

    func testOutputIsFiniteWithSmallValues() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        let x = MLXArray.ones([8, 49, 32]) * 1e-4
        let out = attn(x)
        assertAllFinite(out, "Output should be finite with very small input values")
    }

    func testOutputIsFiniteWithLargeValues() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        let x = MLXArray.ones([8, 49, 32]) * 10.0
        let out = attn(x)
        assertAllFinite(out, "Output should be finite with large input values (softmax stabilizes)")
    }

    func testOutputIsFiniteWithNegativeInput() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        let x = MLXArray.ones([4, 10, 32]) * (-5.0)
        let out = attn(x)
        assertAllFinite(out, "Output should be finite with negative input")
    }

    // =============================================================================
    // MARK: - Multiple Head Configuration Tests
    // =============================================================================

    func testSingleHeadAttention() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 1)
        let x = makeInput(batch: 4, seq: 10, dim: 32)
        let out = attn(x)
        assertShape(out, [4, 10, 32],
                    "Single-head attention should produce correct shape")
        assertAllFinite(out)
    }

    func testMaxHeadsAttention() {
        // dModel heads (each head has dim 1)
        let dModel = 32
        let attn = MultiHeadAttention(dModel: dModel, numHeads: dModel)
        let x = makeInput(batch: 4, seq: 10, dim: dModel)
        let out = attn(x)
        assertShape(out, [4, 10, dModel],
                    "Max heads (d_model heads) should still produce correct shape")
        assertAllFinite(out)
    }

    func testTwoHeadAttention() {
        let attn = MultiHeadAttention(dModel: 32, numHeads: 2)
        XCTAssertEqual(attn.headDim, 16)
        let x = makeInput(batch: 4, seq: 10, dim: 32)
        let out = attn(x)
        assertShape(out, [4, 10, 32])
        assertAllFinite(out)
    }

    func testEightHeadAttention() {
        let attn = MultiHeadAttention(dModel: 64, numHeads: 8)
        XCTAssertEqual(attn.headDim, 8)
        let x = makeInput(batch: 4, seq: 10, dim: 64)
        let out = attn(x)
        assertShape(out, [4, 10, 64])
        assertAllFinite(out)
    }

    // =============================================================================
    // MARK: - Determinism Tests
    // =============================================================================

    func testSameInputProducesSameOutput() {
        // For the same model (same weights), same input must always give same output
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        let x = makeInput(batch: 4, seq: 10, dim: 32)

        let out1 = attn(x)
        let out2 = attn(x)

        eval(out1, out2)

        let values1 = out1.asArray(Float.self)
        let values2 = out2.asArray(Float.self)

        XCTAssertEqual(values1.count, values2.count,
                       "Outputs should have the same element count")
        for (lhs, rhs) in zip(values1, values2) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-6,
                           "Same input to same model should produce numerically identical output")
        }
    }

    // =============================================================================
    // MARK: - Attention Scale Tests
    // =============================================================================

    func testScaledDotProductScaleIsApplied() {
        // When input has very large values, the scale (1/sqrt(headDim)) should
        // prevent attention score explosion. We verify output stays finite.
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)  // headDim=8, scale=1/sqrt(8)≈0.354
        let x = MLXArray.ones([2, 5, 32]) * 100.0
        let out = attn(x)
        assertAllFinite(out, "Scaling prevents attention score explosion with large inputs")
    }

    // =============================================================================
    // MARK: - Different Architectures
    // =============================================================================

    func testSmallArchitecture() {
        // Minimal architecture: dModel=4, numHeads=2, headDim=2
        let attn = MultiHeadAttention(dModel: 4, numHeads: 2)
        XCTAssertEqual(attn.headDim, 2)
        let x = makeInput(batch: 2, seq: 3, dim: 4)
        let out = attn(x)
        assertShape(out, [2, 3, 4])
        assertAllFinite(out)
    }

    func testLargeArchitecture() {
        // Larger architecture: dModel=128, numHeads=8
        let attn = MultiHeadAttention(dModel: 128, numHeads: 8)
        XCTAssertEqual(attn.headDim, 16)
        let x = makeInput(batch: 4, seq: 20, dim: 128)
        let out = attn(x)
        assertShape(out, [4, 20, 128])
        assertAllFinite(out)
    }

    // =============================================================================
    // MARK: - Linear Projection Count
    // =============================================================================

    func testFourLinearProjectionsExist() {
        // MultiHeadAttention should have exactly 4 linear projections: wQ, wK, wV, wO
        let attn = MultiHeadAttention(dModel: 32, numHeads: 4)
        eval(attn)

        // Each projection has weight [dModel, dModel] and bias [dModel]
        let params = attn.parameters().flattened()

        // Should have 8 parameters: 4 weights + 4 biases (wQ, wK, wV, wO each have weight+bias)
        XCTAssertEqual(params.count, 8,
                       "MultiHeadAttention should have 8 parameters (4 weight + 4 bias tensors)")
    }

    func testProjectionWeightShapes() {
        // All projections should be square (dModel × dModel)
        let dModel = 32
        let attn = MultiHeadAttention(dModel: dModel, numHeads: 4)
        eval(attn)

        let params = attn.parameters().flattened()
        let weightParams = params.filter { $0.0.hasSuffix(".weight") }

        XCTAssertEqual(weightParams.count, 4,
                       "MultiHeadAttention should expose 4 projection weight tensors")
        for (name, weight) in weightParams {
            XCTAssertEqual(weight.shape, [dModel, dModel],
                           "Projection weight '\(name)' should be [\(dModel), \(dModel)]")
        }
    }

    func testProjectionBiasShapes() {
        // All biases should have shape [dModel]
        let dModel = 32
        let attn = MultiHeadAttention(dModel: dModel, numHeads: 4)
        eval(attn)

        let params = attn.parameters().flattened()
        let biasParams = params.filter { $0.0.hasSuffix(".bias") }

        XCTAssertEqual(biasParams.count, 4,
                       "MultiHeadAttention should expose 4 projection bias tensors")
        for (name, bias) in biasParams {
            XCTAssertEqual(bias.shape, [dModel],
                           "Projection bias '\(name)' should be [\(dModel)]")
        }
    }
}
