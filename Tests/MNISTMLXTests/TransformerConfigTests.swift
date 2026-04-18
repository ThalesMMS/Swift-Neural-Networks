// ============================================================================
// TransformerConfigTests.swift - Tests for Transformer Architecture Constants
// ============================================================================
//
// Tests for the constants defined in TransformerConfig.swift:
// - TRANSFORMER_D_MODEL: Model dimension
// - TRANSFORMER_NUM_HEADS: Number of attention heads
// - TRANSFORMER_FF_DIM: Feed-forward hidden dimension
// - TRANSFORMER_DROPOUT: Dropout rate
// - TRANSFORMER_LAYER_NORM_EPS: Layer normalization epsilon
//
// These constants define the default MNIST transformer architecture and must
// satisfy structural requirements (divisibility, stability bounds).
//
// ============================================================================

import XCTest
@testable import MNISTMLX

final class TransformerConfigTests: XCTestCase {

    // =============================================================================
    // MARK: - Constant Value Tests
    // =============================================================================

    func testDModelValue() {
        XCTAssertEqual(TRANSFORMER_D_MODEL, 32,
                       "TRANSFORMER_D_MODEL should be 32")
    }

    func testNumHeadsValue() {
        XCTAssertEqual(TRANSFORMER_NUM_HEADS, 4,
                       "TRANSFORMER_NUM_HEADS should be 4")
    }

    func testFFDimValue() {
        XCTAssertEqual(TRANSFORMER_FF_DIM, 64,
                       "TRANSFORMER_FF_DIM should be 64")
    }

    func testDropoutValue() {
        XCTAssertEqual(TRANSFORMER_DROPOUT, 0.0,
                       "TRANSFORMER_DROPOUT should be 0.0")
    }

    func testLayerNormEpsValue() {
        XCTAssertEqual(TRANSFORMER_LAYER_NORM_EPS, 1e-5, accuracy: 1e-10,
                       "TRANSFORMER_LAYER_NORM_EPS should be 1e-5")
    }

    // =============================================================================
    // MARK: - Structural Constraint Tests
    // =============================================================================

    func testDModelDivisibleByNumHeads() {
        // Multi-head attention requires d_model divisible by num_heads
        XCTAssertGreaterThan(TRANSFORMER_D_MODEL, 0,
                             "TRANSFORMER_D_MODEL must be positive")
        XCTAssertGreaterThan(TRANSFORMER_NUM_HEADS, 0,
                             "TRANSFORMER_NUM_HEADS must be positive")
        guard TRANSFORMER_D_MODEL > 0, TRANSFORMER_NUM_HEADS > 0 else { return }

        XCTAssertEqual(TRANSFORMER_D_MODEL % TRANSFORMER_NUM_HEADS, 0,
                       "TRANSFORMER_D_MODEL (\(TRANSFORMER_D_MODEL)) must be divisible by TRANSFORMER_NUM_HEADS (\(TRANSFORMER_NUM_HEADS))")
    }

    func testHeadDimIsPositive() {
        // Each head must have positive dimension
        XCTAssertGreaterThan(TRANSFORMER_D_MODEL, 0,
                             "TRANSFORMER_D_MODEL must be positive")
        XCTAssertGreaterThan(TRANSFORMER_NUM_HEADS, 0,
                             "TRANSFORMER_NUM_HEADS must be positive")
        guard TRANSFORMER_D_MODEL > 0, TRANSFORMER_NUM_HEADS > 0 else { return }

        let headDim = TRANSFORMER_D_MODEL / TRANSFORMER_NUM_HEADS
        XCTAssertGreaterThan(headDim, 0,
                             "Head dimension (d_model / num_heads) must be positive")
    }

    func testHeadDimIsCorrect() {
        // With d_model=32 and num_heads=4, each head should have dimension 8
        XCTAssertGreaterThan(TRANSFORMER_D_MODEL, 0,
                             "TRANSFORMER_D_MODEL must be positive")
        XCTAssertGreaterThan(TRANSFORMER_NUM_HEADS, 0,
                             "TRANSFORMER_NUM_HEADS must be positive")
        guard TRANSFORMER_D_MODEL > 0, TRANSFORMER_NUM_HEADS > 0 else { return }

        let expectedHeadDim = 8
        let actualHeadDim = TRANSFORMER_D_MODEL / TRANSFORMER_NUM_HEADS
        XCTAssertEqual(actualHeadDim, expectedHeadDim,
                       "Head dimension should be \(expectedHeadDim) (d_model/num_heads = \(TRANSFORMER_D_MODEL)/\(TRANSFORMER_NUM_HEADS))")
    }

    func testFFDimIsLargerThanDModel() {
        // Standard transformer: ff_dim >= d_model (expansion in FFN)
        XCTAssertGreaterThanOrEqual(TRANSFORMER_FF_DIM, TRANSFORMER_D_MODEL,
                                    "FFN hidden dimension should be >= d_model for expressive capacity")
    }

    func testFFDimExpansionRatio() {
        // TRANSFORMER_FF_DIM should be a multiple of TRANSFORMER_D_MODEL
        // (2x in this lightweight implementation)
        XCTAssertGreaterThan(TRANSFORMER_D_MODEL, 0,
                             "TRANSFORMER_D_MODEL must be positive")
        guard TRANSFORMER_D_MODEL > 0 else { return }

        let ratio = TRANSFORMER_FF_DIM / TRANSFORMER_D_MODEL
        XCTAssertGreaterThanOrEqual(ratio, 1,
                                    "FF expansion ratio should be at least 1x")
    }

    func testDModelIsPositive() {
        XCTAssertGreaterThan(TRANSFORMER_D_MODEL, 0,
                             "TRANSFORMER_D_MODEL must be positive")
    }

    func testNumHeadsIsPositive() {
        XCTAssertGreaterThan(TRANSFORMER_NUM_HEADS, 0,
                             "TRANSFORMER_NUM_HEADS must be positive")
    }

    func testFFDimIsPositive() {
        XCTAssertGreaterThan(TRANSFORMER_FF_DIM, 0,
                             "TRANSFORMER_FF_DIM must be positive")
    }

    func testDropoutIsInValidRange() {
        XCTAssertGreaterThanOrEqual(TRANSFORMER_DROPOUT, 0.0,
                                    "Dropout rate must be >= 0")
        XCTAssertLessThan(TRANSFORMER_DROPOUT, 1.0,
                          "Dropout rate must be < 1")
    }

    func testLayerNormEpsIsPositive() {
        XCTAssertGreaterThan(TRANSFORMER_LAYER_NORM_EPS, 0.0,
                             "Layer norm epsilon must be positive for numerical stability")
    }

    func testLayerNormEpsIsSmall() {
        // Epsilon should be small enough not to distort normalization
        XCTAssertLessThan(TRANSFORMER_LAYER_NORM_EPS, 0.01,
                          "Layer norm epsilon should be small (< 0.01)")
    }

    // =============================================================================
    // MARK: - Consistency Tests
    // =============================================================================

    func testConstantsAreSufficientForMNIST() {
        // For 49 tokens (7x7 patches from 28x28 image with 4x4 patches):
        // d_model should be large enough for meaningful representation
        XCTAssertGreaterThanOrEqual(TRANSFORMER_D_MODEL, 8,
                                    "d_model should be at least 8 for MNIST patch encoding")
    }

    func testNumHeadsIsReasonableForDModel() {
        // Sanity: num_heads should be less than or equal to d_model
        // (each head has at least dimension 1)
        XCTAssertLessThanOrEqual(TRANSFORMER_NUM_HEADS, TRANSFORMER_D_MODEL,
                                 "num_heads should not exceed d_model")
    }

}
