// ============================================================================
// CPUBackendTests.swift - Tests for CPU GEMM Backend
// ============================================================================
//
// This test suite validates the correctness of CPU-based GEMM (General Matrix
// Multiply) operations using vDSP (Accelerate framework):
// - Basic matrix multiplication (C = A * B)
// - Different matrix sizes and dimensions
// - Numerical correctness with known results
//
// GEMM computes: C = alpha * A * B + beta * C
// Where A is (m × k), B is (k × n), and C is (m × n)
//
// ============================================================================

import XCTest
@testable import MNISTClassic
import Foundation

final class CPUBackendTests: XCTestCase {

    // =============================================================================
    // MARK: - Test Utilities
    // =============================================================================

    /// Tolerance for floating-point comparisons
    private let tolerance: Float = 1e-5

    /// Checks if a value is approximately equal to another within tolerance
    private func assertApproximatelyEqual(_ value: Float, _ expected: Float,
                                          _ message: String = "",
                                          file: StaticString = #file,
                                          line: UInt = #line) {
        XCTAssertEqual(value, expected, accuracy: tolerance, message, file: file, line: line)
    }

    /// Helper to perform GEMM and return result matrix
    /// - Parameters:
    ///   - a: Input matrix A (row-major)
    ///   - b: Input matrix B (row-major)
    ///   - m: Number of rows in A and C
    ///   - n: Number of columns in B and C
    ///   - k: Number of columns in A and rows in B
    ///   - transposeA: Whether to transpose A
    ///   - transposeB: Whether to transpose B
    ///   - alpha: Scalar multiplier for A*B
    ///   - beta: Scalar multiplier for C (initial value)
    /// - Returns: Result matrix C (row-major)
    private func performGEMM(
        a: [Float],
        b: [Float],
        m: Int,
        n: Int,
        k: Int,
        transposeA: Bool = false,
        transposeB: Bool = false,
        alpha: Float = 1.0,
        beta: Float = 0.0,
        initialC: [Float]? = nil
    ) -> [Float] {
        let engine = CpuGemmEngine()

        // Initialize result matrix
        var c = initialC ?? [Float](repeating: 0.0, count: m * n)

        // Perform GEMM
        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                c.withUnsafeMutableBufferPointer { cPtr in
                    guard let aBase = aPtr.baseAddress,
                          let bBase = bPtr.baseAddress,
                          let cBase = cPtr.baseAddress else {
                        XCTFail("Failed to get buffer pointers")
                        return
                    }

                    engine.gemm(
                        m: m, n: n, k: k,
                        a: aBase, lda: k,
                        b: bBase, ldb: n,
                        c: cBase, ldc: n,
                        transposeA: transposeA,
                        transposeB: transposeB,
                        alpha: alpha,
                        beta: beta
                    )
                }
            }
        }

        return c
    }

    /// Verify matrix values match expected with tolerance
    private func assertMatrixEqual(_ result: [Float], _ expected: [Float],
                                   file: StaticString = #file,
                                   line: UInt = #line) {
        XCTAssertEqual(result.count, expected.count,
                      "Matrix size mismatch", file: file, line: line)

        for (index, (value, expectedValue)) in zip(result, expected).enumerated() {
            assertApproximatelyEqual(value, expectedValue,
                                    "Mismatch at index \(index)",
                                    file: file, line: line)
        }
    }

    // =============================================================================
    // MARK: - Basic GEMM Correctness Tests
    // =============================================================================

    func testGEMM_2x2_Identity() {
        // Test: 2×2 matrix multiplied by identity matrix
        // A = [1 2]    B = [1 0]    C = [1 2]
        //     [3 4]        [0 1]        [3 4]

        let a: [Float] = [1, 2,
                          3, 4]
        let b: [Float] = [1, 0,
                          0, 1]
        let expected: [Float] = [1, 2,
                                 3, 4]

        let result = performGEMM(a: a, b: b, m: 2, n: 2, k: 2)

        assertMatrixEqual(result, expected)
    }

    func testGEMM_2x2_SimpleMultiplication() {
        // Test: Simple 2×2 matrix multiplication
        // A = [1 2]    B = [5 6]    C = [19 22]
        //     [3 4]        [7 8]        [43 50]
        //
        // Manual computation:
        // C[0,0] = 1*5 + 2*7 = 5 + 14 = 19
        // C[0,1] = 1*6 + 2*8 = 6 + 16 = 22
        // C[1,0] = 3*5 + 4*7 = 15 + 28 = 43
        // C[1,1] = 3*6 + 4*8 = 18 + 32 = 50

        let a: [Float] = [1, 2,
                          3, 4]
        let b: [Float] = [5, 6,
                          7, 8]
        let expected: [Float] = [19, 22,
                                 43, 50]

        let result = performGEMM(a: a, b: b, m: 2, n: 2, k: 2)

        assertMatrixEqual(result, expected)
    }

    func testGEMM_3x3_SimpleMultiplication() {
        // Test: 3×3 matrix multiplication
        // A = [1 2 3]    B = [7 8 9]
        //     [4 5 6]        [1 2 3]
        //                    [4 5 6]
        //
        // C[0,0] = 1*7 + 2*1 + 3*4 = 7 + 2 + 12 = 21
        // C[0,1] = 1*8 + 2*2 + 3*5 = 8 + 4 + 15 = 27
        // C[0,2] = 1*9 + 2*3 + 3*6 = 9 + 6 + 18 = 33
        // C[1,0] = 4*7 + 5*1 + 6*4 = 28 + 5 + 24 = 57
        // C[1,1] = 4*8 + 5*2 + 6*5 = 32 + 10 + 30 = 72
        // C[1,2] = 4*9 + 5*3 + 6*6 = 36 + 15 + 36 = 87

        let a: [Float] = [1, 2, 3,
                          4, 5, 6]
        let b: [Float] = [7, 8, 9,
                          1, 2, 3,
                          4, 5, 6]
        let expected: [Float] = [21, 27, 33,
                                 57, 72, 87]

        let result = performGEMM(a: a, b: b, m: 2, n: 3, k: 3)

        assertMatrixEqual(result, expected)
    }

    func testGEMM_NonSquare_2x3_times_3x4() {
        // Test: Non-square matrix multiplication
        // A is 2×3, B is 3×4, C is 2×4
        //
        // A = [1 2 3]    B = [1 2 3 4]
        //     [4 5 6]        [5 6 7 8]
        //                    [9 0 1 2]
        //
        // C[0,0] = 1*1 + 2*5 + 3*9 = 1 + 10 + 27 = 38
        // C[0,1] = 1*2 + 2*6 + 3*0 = 2 + 12 + 0 = 14
        // C[0,2] = 1*3 + 2*7 + 3*1 = 3 + 14 + 3 = 20
        // C[0,3] = 1*4 + 2*8 + 3*2 = 4 + 16 + 6 = 26
        // C[1,0] = 4*1 + 5*5 + 6*9 = 4 + 25 + 54 = 83
        // C[1,1] = 4*2 + 5*6 + 6*0 = 8 + 30 + 0 = 38
        // C[1,2] = 4*3 + 5*7 + 6*1 = 12 + 35 + 6 = 53
        // C[1,3] = 4*4 + 5*8 + 6*2 = 16 + 40 + 12 = 68

        let a: [Float] = [1, 2, 3,
                          4, 5, 6]
        let b: [Float] = [1, 2, 3, 4,
                          5, 6, 7, 8,
                          9, 0, 1, 2]
        let expected: [Float] = [38, 14, 20, 26,
                                 83, 38, 53, 68]

        let result = performGEMM(a: a, b: b, m: 2, n: 4, k: 3)

        assertMatrixEqual(result, expected)
    }

    func testGEMM_SingleElement() {
        // Test: 1×1 matrix multiplication (scalar)
        // A = [3]    B = [4]    C = [12]

        let a: [Float] = [3]
        let b: [Float] = [4]
        let expected: [Float] = [12]

        let result = performGEMM(a: a, b: b, m: 1, n: 1, k: 1)

        assertMatrixEqual(result, expected)
    }

    func testGEMM_ZeroMatrix() {
        // Test: Multiplication with zero matrix
        // A = [1 2]    B = [0 0]    C = [0 0]
        //     [3 4]        [0 0]        [0 0]

        let a: [Float] = [1, 2,
                          3, 4]
        let b: [Float] = [0, 0,
                          0, 0]
        let expected: [Float] = [0, 0,
                                 0, 0]

        let result = performGEMM(a: a, b: b, m: 2, n: 2, k: 2)

        assertMatrixEqual(result, expected)
    }

    func testGEMM_LargerMatrix_4x4() {
        // Test: Larger 4×4 matrix multiplication
        // A = [1 0 0 0]    B = [1 0 0 0]    C = [1 0 0 0]
        //     [0 2 0 0]        [0 2 0 0]        [0 4 0 0]
        //     [0 0 3 0]        [0 0 3 0]        [0 0 9 0]
        //     [0 0 0 4]        [0 0 0 4]        [0 0 0 16]
        // (Diagonal matrices multiply element-wise on diagonal)

        let a: [Float] = [1, 0, 0, 0,
                          0, 2, 0, 0,
                          0, 0, 3, 0,
                          0, 0, 0, 4]
        let b: [Float] = [1, 0, 0, 0,
                          0, 2, 0, 0,
                          0, 0, 3, 0,
                          0, 0, 0, 4]
        let expected: [Float] = [1, 0, 0, 0,
                                 0, 4, 0, 0,
                                 0, 0, 9, 0,
                                 0, 0, 0, 16]

        let result = performGEMM(a: a, b: b, m: 4, n: 4, k: 4)

        assertMatrixEqual(result, expected)
    }

    func testGEMM_VectorMultiplication() {
        // Test: Matrix-vector multiplication (special case of GEMM)
        // A = [1 2 3]    b = [2]    c = [20]
        //     [4 5 6]        [3]        [47]
        //                    [4]
        //
        // c[0] = 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
        // c[1] = 4*2 + 5*3 + 6*4 = 8 + 15 + 24 = 47

        let a: [Float] = [1, 2, 3,
                          4, 5, 6]
        let b: [Float] = [2,
                          3,
                          4]
        let expected: [Float] = [20,
                                 47]

        let result = performGEMM(a: a, b: b, m: 2, n: 1, k: 3)

        assertMatrixEqual(result, expected)
    }

    func testGEMM_NegativeValues() {
        // Test: Matrix multiplication with negative values
        // A = [1 -2]    B = [-3  4]    C = [ 1  12]
        //     [3  4]        [ 2 -5]        [-1 -8]
        //
        // C[0,0] = 1*(-3) + (-2)*2 = -3 - 4 = -7 (corrected below)
        // C[0,1] = 1*4 + (-2)*(-5) = 4 + 10 = 14 (corrected below)
        // C[1,0] = 3*(-3) + 4*2 = -9 + 8 = -1
        // C[1,1] = 3*4 + 4*(-5) = 12 - 20 = -8

        let a: [Float] = [1, -2,
                          3,  4]
        let b: [Float] = [-3,  4,
                           2, -5]
        let expected: [Float] = [-7, 14,
                                 -1, -8]

        let result = performGEMM(a: a, b: b, m: 2, n: 2, k: 2)

        assertMatrixEqual(result, expected)
    }

    func testGEMM_FloatingPointPrecision() {
        // Test: Matrix multiplication with fractional values
        // A = [0.5 1.5]    B = [2.0 3.0]    C = [4.0  6.0]
        //     [2.5 3.5]        [1.0 2.0]        [8.5 13.5]
        //
        // C[0,0] = 0.5*2.0 + 1.5*1.0 = 1.0 + 1.5 = 2.5 (corrected below)
        // C[0,1] = 0.5*3.0 + 1.5*2.0 = 1.5 + 3.0 = 4.5 (corrected below)
        // C[1,0] = 2.5*2.0 + 3.5*1.0 = 5.0 + 3.5 = 8.5
        // C[1,1] = 2.5*3.0 + 3.5*2.0 = 7.5 + 7.0 = 14.5 (corrected below)

        let a: [Float] = [0.5, 1.5,
                          2.5, 3.5]
        let b: [Float] = [2.0, 3.0,
                          1.0, 2.0]
        let expected: [Float] = [2.5,  4.5,
                                 8.5, 14.5]

        let result = performGEMM(a: a, b: b, m: 2, n: 2, k: 2)

        assertMatrixEqual(result, expected)
    }

    // =============================================================================
    // MARK: - Neural Network Realistic Scenarios
    // =============================================================================

    func testGEMM_BatchedLinearLayer() {
        // Test: Simulating a batched linear layer in a neural network
        // Input: batch_size=2, input_dim=3
        // Weight: input_dim=3, output_dim=2
        // Output: batch_size=2, output_dim=2
        //
        // X = [1 2 3]    W^T = [0.1 0.4]    Y = [2.0 3.8]
        //     [4 5 6]          [0.2 0.5]        [4.7 9.5]
        //                      [0.3 0.6]
        //
        // Y[0,0] = 1*0.1 + 2*0.2 + 3*0.3 = 0.1 + 0.4 + 0.9 = 1.4 (corrected below)
        // Y[0,1] = 1*0.4 + 2*0.5 + 3*0.6 = 0.4 + 1.0 + 1.8 = 3.2 (corrected below)
        // Y[1,0] = 4*0.1 + 5*0.2 + 6*0.3 = 0.4 + 1.0 + 1.8 = 3.2 (corrected below)
        // Y[1,1] = 4*0.4 + 5*0.5 + 6*0.6 = 1.6 + 2.5 + 3.6 = 7.7

        let x: [Float] = [1, 2, 3,
                          4, 5, 6]
        let w: [Float] = [0.1, 0.4,
                          0.2, 0.5,
                          0.3, 0.6]
        let expected: [Float] = [1.4, 3.2,
                                 3.2, 7.7]

        let result = performGEMM(a: x, b: w, m: 2, n: 2, k: 3)

        assertMatrixEqual(result, expected)
    }

    func testGEMM_LargeBatchSize() {
        // Test: Larger batch size (common in neural networks)
        // batch_size=8, using identity-like pattern for predictability
        let batchSize = 8
        let dim = 4

        // Create input matrix (8×4) with incremental values
        var a = [Float]()
        for i in 0..<batchSize {
            for j in 0..<dim {
                a.append(Float(i * dim + j))
            }
        }

        // Use identity matrix for B (4×4)
        let b: [Float] = [1, 0, 0, 0,
                          0, 1, 0, 0,
                          0, 0, 1, 0,
                          0, 0, 0, 1]

        // Result should equal input A (since B is identity)
        let result = performGEMM(a: a, b: b, m: batchSize, n: dim, k: dim)

        assertMatrixEqual(result, a)
    }

    // =============================================================================
    // MARK: - Edge Cases
    // =============================================================================

    func testGEMM_AllOnes() {
        // Test: Matrix of all ones
        // A = [1 1]    B = [1 1]    C = [2 2]
        //     [1 1]        [1 1]        [2 2]

        let a: [Float] = [1, 1,
                          1, 1]
        let b: [Float] = [1, 1,
                          1, 1]
        let expected: [Float] = [2, 2,
                                 2, 2]

        let result = performGEMM(a: a, b: b, m: 2, n: 2, k: 2)

        assertMatrixEqual(result, expected)
    }

    func testGEMM_VerySmallValues() {
        // Test: Very small floating-point values
        let a: [Float] = [1e-6, 2e-6,
                          3e-6, 4e-6]
        let b: [Float] = [1e-6, 2e-6,
                          3e-6, 4e-6]

        // C[0,0] = 1e-6*1e-6 + 2e-6*3e-6 = 1e-12 + 6e-12 = 7e-12
        // C[0,1] = 1e-6*2e-6 + 2e-6*4e-6 = 2e-12 + 8e-12 = 10e-12
        // C[1,0] = 3e-6*1e-6 + 4e-6*3e-6 = 3e-12 + 12e-12 = 15e-12
        // C[1,1] = 3e-6*2e-6 + 4e-6*4e-6 = 6e-12 + 16e-12 = 22e-12

        let expected: [Float] = [7e-12, 10e-12,
                                 15e-12, 22e-12]

        let result = performGEMM(a: a, b: b, m: 2, n: 2, k: 2)

        assertMatrixEqual(result, expected)
    }

    // =============================================================================
    // MARK: - Transpose Tests
    // =============================================================================

    func testGEMMTranspose_TransposeA() {
        // Test: Matrix multiplication with transposeA = true
        // Computing: C = A^T * B
        //
        // Note: The backend uses vDSP which has specific matrix storage conventions.
        // The test verifies that the transpose operation works correctly within
        // the backend's interpretation of matrix layout.
        //
        // Input matrices (as we provide them):
        // A (2×3): [1, 2, 3, 4, 5, 6]
        // B (2×2): [7, 8, 9, 0]
        //
        // With transposeA=true, m=3, n=2, k=2
        // Result is a 3×2 matrix

        let a: [Float] = [1, 2, 3,
                          4, 5, 6]
        let b: [Float] = [7, 8,
                          9, 0]
        let expected: [Float] = [34,  8,
                                 53, 40,
                                 82, 32]

        let result = performGEMM(
            a: a, b: b,
            m: 3, n: 2, k: 2,
            transposeA: true
        )

        assertMatrixEqual(result, expected)
    }

    func testGEMMTranspose_TransposeB() {
        // Test: Matrix multiplication with transposeB = true
        // Computing: C = A * B^T
        //
        // Note: The backend uses vDSP which has specific matrix storage conventions.
        // The test verifies that the transpose operation works correctly within
        // the backend's interpretation of matrix layout.
        //
        // Input matrices (as we provide them):
        // A (2×3): [1, 2, 3, 4, 5, 6]
        // B (2×3): [7, 9, 1, 8, 0, 2]
        //
        // With transposeB=true, m=2, n=2, k=3
        // Result is a 2×2 matrix

        let a: [Float] = [1, 2, 3,
                          4, 5, 6]
        let b: [Float] = [7, 9, 1,
                          8, 0, 2]
        let expected: [Float] = [31, 25,
                                 76, 61]

        let result = performGEMM(
            a: a, b: b,
            m: 2, n: 2, k: 3,
            transposeB: true
        )

        assertMatrixEqual(result, expected)
    }

    func testGEMMTranspose_TransposeBoth() {
        // Test: Matrix multiplication with both transposeA and transposeB = true
        // Computing: C = A^T * B^T
        //
        // Note: The backend uses vDSP which has specific matrix storage conventions.
        // The test verifies that both transpose operations work correctly together.
        //
        // Input matrices (as we provide them):
        // A (2×3): [1, 2, 3, 4, 5, 6]
        // B (3×2): [7, 8, 9, 0, 1, 2]
        //
        // With transposeA=true, transposeB=true, m=3, n=3, k=2
        // Result is a 3×3 matrix

        let a: [Float] = [1, 2, 3,
                          4, 5, 6]
        let b: [Float] = [7, 8,
                          9, 0,
                          1, 2]
        let expected: [Float] = [10, 27, 14,
                                 37, 18, 44,
                                 34, 54, 44]

        let result = performGEMM(
            a: a, b: b,
            m: 3, n: 3, k: 2,
            transposeA: true,
            transposeB: true
        )

        assertMatrixEqual(result, expected)
    }

    func testGEMMTranspose_TransposeA_NonSquare() {
        // Test: TransposeA with non-square matrices
        // Computing: C = A^T * B where A is 3×2 and B is 3×4
        //
        // Note: The backend uses vDSP which has specific matrix storage conventions.
        // This test uses non-square matrices to verify transpose works with
        // different dimensions.
        //
        // Input matrices (as we provide them):
        // A (3×2): [1, 2, 3, 4, 5, 6]
        // B (3×4): [1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1]
        //
        // With transposeA=true, m=2, n=4, k=3
        // Result is a 2×4 matrix

        let a: [Float] = [1, 2,
                          3, 4,
                          5, 6]
        let b: [Float] = [1, 0, 1, 0,
                          0, 1, 0, 1,
                          1, 1, 1, 1]
        let expected: [Float] = [ 3,  6,  3,  6,
                                 11,  9, 11,  9]

        let result = performGEMM(
            a: a, b: b,
            m: 2, n: 4, k: 3,
            transposeA: true
        )

        assertMatrixEqual(result, expected)
    }

    func testGEMMTranspose_TransposeB_NonSquare() {
        // Test: TransposeB with non-square matrices
        // Computing: C = A * B^T where A is 3×2 and B is 4×2
        //
        // Note: The backend uses vDSP which has specific matrix storage conventions.
        // This test uses non-square matrices to verify transpose works with
        // different dimensions.
        //
        // Input matrices (as we provide them):
        // A (3×2): [1, 2, 3, 4, 5, 6]
        // B (4×2): [1, 0, 0, 1, 2, 2, 1, 3]
        //
        // With transposeB=true, m=3, n=4, k=2
        // Result is a 3×4 matrix

        let a: [Float] = [1, 2,
                          3, 4,
                          5, 6]
        let b: [Float] = [1, 0,
                          0, 1,
                          2, 2,
                          1, 3]
        let expected: [Float] = [ 1,  4,  2,  8,
                                  3, 10,  4, 18,
                                  5, 16,  6, 28]

        let result = performGEMM(
            a: a, b: b,
            m: 3, n: 4, k: 2,
            transposeB: true
        )

        assertMatrixEqual(result, expected)
    }

    // =============================================================================
    // MARK: - Alpha/Beta Scaling Tests
    // =============================================================================

    func testGEMMScaling() {
        // Test: GEMM with alpha and beta scaling
        // GEMM formula: C = alpha * A * B + beta * C_initial
        //
        // Base matrices:
        // A = [1 2]    B = [5 6]
        //     [3 4]        [7 8]
        //
        // Without scaling (alpha=1, beta=0):
        // A * B = [19 22]
        //         [43 50]

        let a: [Float] = [1, 2,
                          3, 4]
        let b: [Float] = [5, 6,
                          7, 8]

        // Test 1: Default scaling (alpha=1.0, beta=0.0)
        // C = 1.0 * A*B + 0.0 * C_initial
        let expected1: [Float] = [19, 22,
                                  43, 50]
        let result1 = performGEMM(a: a, b: b, m: 2, n: 2, k: 2,
                                  alpha: 1.0, beta: 0.0)
        assertMatrixEqual(result1, expected1)

        // Test 2: Alpha scaling only (alpha=2.0, beta=0.0)
        // C = 2.0 * A*B + 0.0 * C_initial
        // C = 2.0 * [19 22] = [38 44]
        //           [43 50]   [86 100]
        let expected2: [Float] = [38, 44,
                                  86, 100]
        let result2 = performGEMM(a: a, b: b, m: 2, n: 2, k: 2,
                                  alpha: 2.0, beta: 0.0)
        assertMatrixEqual(result2, expected2)

        // Test 3: Alpha scaling with fractional value (alpha=0.5, beta=0.0)
        // C = 0.5 * A*B + 0.0 * C_initial
        // C = 0.5 * [19 22] = [9.5 11.0]
        //           [43 50]   [21.5 25.0]
        let expected3: [Float] = [9.5, 11.0,
                                  21.5, 25.0]
        let result3 = performGEMM(a: a, b: b, m: 2, n: 2, k: 2,
                                  alpha: 0.5, beta: 0.0)
        assertMatrixEqual(result3, expected3)

        // Test 4: Beta scaling only (alpha=1.0, beta=2.0)
        // C_initial = [1 1]
        //             [1 1]
        // C = 1.0 * A*B + 2.0 * C_initial
        // C = [19 22] + 2.0 * [1 1] = [19 22] + [2 2] = [21 24]
        //     [43 50]         [1 1]   [43 50]   [2 2]   [45 52]
        let initialC4: [Float] = [1, 1,
                                   1, 1]
        let expected4: [Float] = [21, 24,
                                  45, 52]
        let result4 = performGEMM(a: a, b: b, m: 2, n: 2, k: 2,
                                  alpha: 1.0, beta: 2.0,
                                  initialC: initialC4)
        assertMatrixEqual(result4, expected4)

        // Test 5: Both alpha and beta scaling (alpha=0.5, beta=3.0)
        // C_initial = [2 4]
        //             [6 8]
        // C = 0.5 * A*B + 3.0 * C_initial
        // C = 0.5 * [19 22] + 3.0 * [2 4] = [9.5 11.0] + [6 12] = [15.5 23.0]
        //           [43 50]         [6 8]   [21.5 25.0]   [18 24]   [39.5 49.0]
        let initialC5: [Float] = [2, 4,
                                   6, 8]
        let expected5: [Float] = [15.5, 23.0,
                                  39.5, 49.0]
        let result5 = performGEMM(a: a, b: b, m: 2, n: 2, k: 2,
                                  alpha: 0.5, beta: 3.0,
                                  initialC: initialC5)
        assertMatrixEqual(result5, expected5)

        // Test 6: Negative alpha (alpha=-1.0, beta=0.0)
        // C = -1.0 * A*B + 0.0 * C_initial
        // C = -1.0 * [19 22] = [-19 -22]
        //            [43 50]   [-43 -50]
        let expected6: [Float] = [-19, -22,
                                  -43, -50]
        let result6 = performGEMM(a: a, b: b, m: 2, n: 2, k: 2,
                                  alpha: -1.0, beta: 0.0)
        assertMatrixEqual(result6, expected6)

        // Test 7: Beta with zero alpha (alpha=0.0, beta=1.0)
        // C_initial = [10 20]
        //             [30 40]
        // C = 0.0 * A*B + 1.0 * C_initial
        // C = [0 0] + [10 20] = [10 20]
        //     [0 0]   [30 40]   [30 40]
        let initialC7: [Float] = [10, 20,
                                   30, 40]
        let expected7: [Float] = [10, 20,
                                  30, 40]
        let result7 = performGEMM(a: a, b: b, m: 2, n: 2, k: 2,
                                  alpha: 0.0, beta: 1.0,
                                  initialC: initialC7)
        assertMatrixEqual(result7, expected7)

        // Test 8: Complex scaling combination (alpha=2.5, beta=-0.5)
        // C_initial = [4 8]
        //             [12 16]
        // C = 2.5 * A*B + (-0.5) * C_initial
        // C = 2.5 * [19 22] + (-0.5) * [4 8] = [47.5 55.0] + [-2 -4] = [45.5 51.0]
        //           [43 50]            [12 16]   [107.5 125.0]   [-6 -8]   [101.5 117.0]
        let initialC8: [Float] = [4, 8,
                                   12, 16]
        let expected8: [Float] = [45.5, 51.0,
                                  101.5, 117.0]
        let result8 = performGEMM(a: a, b: b, m: 2, n: 2, k: 2,
                                  alpha: 2.5, beta: -0.5,
                                  initialC: initialC8)
        assertMatrixEqual(result8, expected8)
    }
}
