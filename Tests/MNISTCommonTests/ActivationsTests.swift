// ============================================================================
// ActivationsTests.swift - Tests for Activation Functions
// ============================================================================
//
// This test suite validates the mathematical correctness of activation functions:
// - Softmax row-wise normalization
// - Numerical stability with large values
// - Output probability distribution properties
// - Pointer-based and array-based implementations
//
// ============================================================================

import XCTest
@testable import MNISTCommon
import Foundation

final class ActivationsTests: XCTestCase {

    // =============================================================================
    // MARK: - Test Utilities
    // =============================================================================

    /// Tolerance for floating-point comparisons
    private let tolerance: Float = 1e-6

    /// Checks if a value is approximately equal to another within tolerance
    private func assertApproximatelyEqual(_ value: Float, _ expected: Float,
                                          _ message: String = "",
                                          file: StaticString = #file,
                                          line: UInt = #line) {
        XCTAssertEqual(value, expected, accuracy: tolerance, message, file: file, line: line)
    }

    /// Verifies that all values in an array are in the range [0, 1]
    private func assertAllInRange01(_ data: [Float],
                                    file: StaticString = #file,
                                    line: UInt = #line) {
        for (index, value) in data.enumerated() {
            XCTAssertGreaterThanOrEqual(value, 0.0,
                                       "Value at index \(index) should be >= 0",
                                       file: file, line: line)
            XCTAssertLessThanOrEqual(value, 1.0,
                                    "Value at index \(index) should be <= 1",
                                    file: file, line: line)
        }
    }

    /// Calculates the sum of elements in a row
    private func sumRow(_ data: [Float], row: Int, cols: Int) -> Float {
        let base = row * cols
        return (0..<cols).reduce(0.0) { $0 + data[base + $1] }
    }

    // =============================================================================
    // MARK: - Softmax Basic Correctness Tests
    // =============================================================================

    func testSoftmaxSingleRowSumsToOne() {
        // Test that softmax of a single row sums to 1.0
        var data: [Float] = [1.0, 2.0, 3.0, 4.0]
        softmaxRows(&data, rows: 1, cols: 4)

        let sum = data.reduce(0.0, +)
        assertApproximatelyEqual(sum, 1.0, "Softmax output should sum to 1.0")
    }

    func testSoftmaxSingleRowRangeIsValid() {
        // Test that all softmax outputs are in [0, 1]
        var data: [Float] = [1.0, 2.0, 3.0, 4.0]
        softmaxRows(&data, rows: 1, cols: 4)

        assertAllInRange01(data)
    }

    func testSoftmaxMultipleRowsEachSumsToOne() {
        // Test that each row independently sums to 1.0
        var data: [Float] = [
            1.0, 2.0, 3.0,  // Row 0
            4.0, 5.0, 6.0,  // Row 1
            7.0, 8.0, 9.0   // Row 2
        ]
        let rows = 3
        let cols = 3

        softmaxRows(&data, rows: rows, cols: cols)

        for r in 0..<rows {
            let rowSum = sumRow(data, row: r, cols: cols)
            assertApproximatelyEqual(rowSum, 1.0,
                                    "Row \(r) should sum to 1.0")
        }
    }

    func testSoftmaxMultipleRowsAllInRange() {
        // Test that all values across multiple rows are in [0, 1]
        var data: [Float] = [
            1.0, 2.0, 3.0, 4.0,
            -1.0, 0.0, 1.0, 2.0,
            10.0, 20.0, 30.0, 40.0
        ]

        softmaxRows(&data, rows: 3, cols: 4)

        assertAllInRange01(data)
    }

    // =============================================================================
    // MARK: - Softmax Mathematical Properties
    // =============================================================================

    func testSoftmaxMaxElementGetsHighestProbability() {
        // The largest input should get the highest probability output
        var data: [Float] = [1.0, 5.0, 2.0, 3.0]
        softmaxRows(&data, rows: 1, cols: 4)

        let maxIndex = data.indices.max(by: { data[$0] < data[$1] })!
        XCTAssertEqual(maxIndex, 1, "Index 1 (originally 5.0) should have highest probability")
        XCTAssertGreaterThan(data[1], data[0], "Max element should have higher probability")
        XCTAssertGreaterThan(data[1], data[2], "Max element should have higher probability")
        XCTAssertGreaterThan(data[1], data[3], "Max element should have higher probability")
    }

    func testSoftmaxPreservesRelativeOrdering() {
        // Softmax preserves the ordering: if x[i] > x[j], then softmax(x)[i] > softmax(x)[j]
        var data: [Float] = [1.0, 2.0, 3.0, 4.0]
        softmaxRows(&data, rows: 1, cols: 4)

        XCTAssertLessThan(data[0], data[1], "Softmax should preserve ordering")
        XCTAssertLessThan(data[1], data[2], "Softmax should preserve ordering")
        XCTAssertLessThan(data[2], data[3], "Softmax should preserve ordering")
    }

    func testSoftmaxUniformInputProducesUniformOutput() {
        // When all inputs are equal, softmax should produce uniform distribution
        var data: [Float] = [5.0, 5.0, 5.0, 5.0]
        softmaxRows(&data, rows: 1, cols: 4)

        let expectedValue: Float = 0.25  // 1/4
        for i in 0..<4 {
            assertApproximatelyEqual(data[i], expectedValue,
                                    "Uniform input should produce uniform output")
        }
    }

    func testSoftmaxWithZeros() {
        // Test softmax with all zeros (should produce uniform distribution)
        var data: [Float] = [0.0, 0.0, 0.0]
        softmaxRows(&data, rows: 1, cols: 3)

        let expectedValue: Float = 1.0 / 3.0
        for i in 0..<3 {
            assertApproximatelyEqual(data[i], expectedValue,
                                    "All zeros should produce uniform distribution")
        }
    }

    func testSoftmaxWithNegativeValues() {
        // Test softmax with negative values
        var data: [Float] = [-1.0, -2.0, -3.0, -4.0]
        softmaxRows(&data, rows: 1, cols: 4)

        // Should still sum to 1 and be in valid range
        let sum = data.reduce(0.0, +)
        assertApproximatelyEqual(sum, 1.0, "Softmax with negatives should sum to 1.0")
        assertAllInRange01(data)

        // Relative ordering should be preserved (less negative = higher probability)
        XCTAssertGreaterThan(data[0], data[1], "Less negative should have higher probability")
        XCTAssertGreaterThan(data[1], data[2], "Less negative should have higher probability")
        XCTAssertGreaterThan(data[2], data[3], "Less negative should have higher probability")
    }

    // =============================================================================
    // MARK: - Softmax Numerical Stability Tests
    // =============================================================================

    func testSoftmaxNumericalStabilityWithLargeValues() {
        // Test that softmax handles large values without overflow
        // Using max subtraction trick, this should work correctly
        var data: [Float] = [1000.0, 1001.0, 1002.0]
        softmaxRows(&data, rows: 1, cols: 3)

        // Should still sum to 1.0 and be in valid range (no NaN or Inf)
        let sum = data.reduce(0.0, +)
        XCTAssertFalse(sum.isNaN, "Sum should not be NaN")
        XCTAssertFalse(sum.isInfinite, "Sum should not be infinite")
        assertApproximatelyEqual(sum, 1.0, "Softmax with large values should sum to 1.0")
        assertAllInRange01(data)
    }

    func testSoftmaxNumericalStabilityWithVeryLargeValues() {
        // Test with extremely large values
        var data: [Float] = [10000.0, 10001.0, 10002.0, 10003.0]
        softmaxRows(&data, rows: 1, cols: 4)

        // Verify no NaN or Inf values
        for (index, value) in data.enumerated() {
            XCTAssertFalse(value.isNaN, "Value at index \(index) should not be NaN")
            XCTAssertFalse(value.isInfinite, "Value at index \(index) should not be infinite")
        }

        let sum = data.reduce(0.0, +)
        assertApproximatelyEqual(sum, 1.0, "Softmax with very large values should sum to 1.0")
    }

    func testSoftmaxWithLargeNegativeValues() {
        // Test with large negative values
        var data: [Float] = [-1000.0, -1001.0, -1002.0]
        softmaxRows(&data, rows: 1, cols: 3)

        let sum = data.reduce(0.0, +)
        XCTAssertFalse(sum.isNaN, "Sum should not be NaN")
        assertApproximatelyEqual(sum, 1.0, "Softmax with large negatives should sum to 1.0")
        assertAllInRange01(data)
    }

    func testSoftmaxWithMixedLargeValues() {
        // Test with mix of large positive and negative values
        var data: [Float] = [-1000.0, 0.0, 1000.0]
        softmaxRows(&data, rows: 1, cols: 3)

        let sum = data.reduce(0.0, +)
        assertApproximatelyEqual(sum, 1.0, "Softmax with mixed large values should sum to 1.0")
        assertAllInRange01(data)

        // The middle value (1000.0) should dominate
        XCTAssertGreaterThan(data[2], 0.99, "Largest value should get nearly all probability")
    }

    // =============================================================================
    // MARK: - Softmax Edge Cases
    // =============================================================================

    func testSoftmaxSingleElement() {
        // Test with single element (should be 1.0)
        var data: [Float] = [42.0]
        softmaxRows(&data, rows: 1, cols: 1)

        assertApproximatelyEqual(data[0], 1.0, "Single element softmax should be 1.0")
    }

    func testSoftmaxSingleColumn() {
        // Test multiple rows with single column each
        var data: [Float] = [1.0, 2.0, 3.0]
        softmaxRows(&data, rows: 3, cols: 1)

        // Each single-element row should be 1.0
        for i in 0..<3 {
            assertApproximatelyEqual(data[i], 1.0,
                                    "Single column softmax should be 1.0")
        }
    }

    func testSoftmaxLargeNumberOfColumns() {
        // Test with many columns
        let cols = 100
        var data = [Float](repeating: 1.0, count: cols)
        softmaxRows(&data, rows: 1, cols: cols)

        let sum = data.reduce(0.0, +)
        assertApproximatelyEqual(sum, 1.0, "Large row should sum to 1.0")

        let expectedValue: Float = 1.0 / Float(cols)
        for i in 0..<cols {
            assertApproximatelyEqual(data[i], expectedValue,
                                    "Each element should be 1/\(cols)")
        }
    }

    func testSoftmaxIndependenceAcrossRows() {
        // Test that softmax processes each row independently
        var data: [Float] = [
            1.0, 2.0,    // Row 0
            100.0, 101.0 // Row 1 (very different scale)
        ]
        softmaxRows(&data, rows: 2, cols: 2)

        // Each row should sum to 1.0 independently
        let row0Sum = sumRow(data, row: 0, cols: 2)
        let row1Sum = sumRow(data, row: 1, cols: 2)

        assertApproximatelyEqual(row0Sum, 1.0, "Row 0 should sum to 1.0")
        assertApproximatelyEqual(row1Sum, 1.0, "Row 1 should sum to 1.0")

        // Despite different scales, each row should maintain relative probabilities
        XCTAssertGreaterThan(data[1], data[0], "Row 0: second element should be larger")
        XCTAssertGreaterThan(data[3], data[2], "Row 1: second element should be larger")
    }

    // =============================================================================
    // MARK: - Softmax Pointer-Based Implementation Tests
    // =============================================================================

    func testSoftmaxPointerMatchesArrayVersion() {
        // Test that pointer-based version produces same results as array-based
        let originalData: [Float] = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            -1.0, 0.0, 1.0
        ]
        let rows = 3
        let cols = 3

        // Test array-based version
        var arrayData = originalData
        softmaxRows(&arrayData, rows: rows, cols: cols)

        // Test pointer-based version
        var pointerData = originalData
        pointerData.withUnsafeMutableBufferPointer { buffer in
            softmaxRowsPointer(buffer.baseAddress!, rows: rows, cols: cols)
        }

        // Results should be identical
        XCTAssertEqual(arrayData.count, pointerData.count, "Counts should match")
        for i in 0..<arrayData.count {
            assertApproximatelyEqual(pointerData[i], arrayData[i],
                                    "Pointer and array versions should produce same result at index \(i)")
        }
    }

    func testSoftmaxPointerBasicCorrectness() {
        // Test pointer version directly for basic correctness
        var data: [Float] = [1.0, 2.0, 3.0, 4.0]

        data.withUnsafeMutableBufferPointer { buffer in
            softmaxRowsPointer(buffer.baseAddress!, rows: 1, cols: 4)
        }

        let sum = data.reduce(0.0, +)
        assertApproximatelyEqual(sum, 1.0, "Pointer version should sum to 1.0")
        assertAllInRange01(data)
    }

    func testSoftmaxPointerMultipleRows() {
        // Test pointer version with multiple rows
        var data: [Float] = [
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        ]
        let rows = 3
        let cols = 2

        data.withUnsafeMutableBufferPointer { buffer in
            softmaxRowsPointer(buffer.baseAddress!, rows: rows, cols: cols)
        }

        // Each row should sum to 1.0
        for r in 0..<rows {
            let rowSum = sumRow(data, row: r, cols: cols)
            assertApproximatelyEqual(rowSum, 1.0,
                                    "Pointer version row \(r) should sum to 1.0")
        }

        assertAllInRange01(data)
    }

    func testSoftmaxPointerNumericalStability() {
        // Test pointer version with large values
        var data: [Float] = [1000.0, 1001.0, 1002.0]

        data.withUnsafeMutableBufferPointer { buffer in
            softmaxRowsPointer(buffer.baseAddress!, rows: 1, cols: 3)
        }

        let sum = data.reduce(0.0, +)
        XCTAssertFalse(sum.isNaN, "Pointer version should not produce NaN")
        XCTAssertFalse(sum.isInfinite, "Pointer version should not produce Inf")
        assertApproximatelyEqual(sum, 1.0, "Pointer version with large values should sum to 1.0")
    }

    func testSoftmaxPointerEquivalence() {
        // Comprehensive test: pointer-based and array-based versions produce identical results
        // across various input scenarios

        // Test case 1: Single row with simple values
        let testCase1: [Float] = [1.0, 2.0, 3.0, 4.0]
        var arrayResult1 = testCase1
        var pointerResult1 = testCase1
        softmaxRows(&arrayResult1, rows: 1, cols: 4)
        pointerResult1.withUnsafeMutableBufferPointer { buffer in
            softmaxRowsPointer(buffer.baseAddress!, rows: 1, cols: 4)
        }
        for i in 0..<4 {
            assertApproximatelyEqual(pointerResult1[i], arrayResult1[i],
                                    "Test case 1: Results should match at index \(i)")
        }

        // Test case 2: Multiple rows with mixed values
        let testCase2: [Float] = [
            1.0, 2.0, 3.0,      // Row 0
            -1.0, 0.0, 1.0,     // Row 1
            10.0, 20.0, 30.0    // Row 2
        ]
        var arrayResult2 = testCase2
        var pointerResult2 = testCase2
        softmaxRows(&arrayResult2, rows: 3, cols: 3)
        pointerResult2.withUnsafeMutableBufferPointer { buffer in
            softmaxRowsPointer(buffer.baseAddress!, rows: 3, cols: 3)
        }
        for i in 0..<9 {
            assertApproximatelyEqual(pointerResult2[i], arrayResult2[i],
                                    "Test case 2: Results should match at index \(i)")
        }

        // Test case 3: Large values (numerical stability test)
        let testCase3: [Float] = [1000.0, 1001.0, 1002.0, 1003.0]
        var arrayResult3 = testCase3
        var pointerResult3 = testCase3
        softmaxRows(&arrayResult3, rows: 1, cols: 4)
        pointerResult3.withUnsafeMutableBufferPointer { buffer in
            softmaxRowsPointer(buffer.baseAddress!, rows: 1, cols: 4)
        }
        for i in 0..<4 {
            assertApproximatelyEqual(pointerResult3[i], arrayResult3[i],
                                    "Test case 3: Results should match at index \(i)")
        }

        // Test case 4: Negative values
        let testCase4: [Float] = [-10.0, -5.0, -2.0, -1.0]
        var arrayResult4 = testCase4
        var pointerResult4 = testCase4
        softmaxRows(&arrayResult4, rows: 1, cols: 4)
        pointerResult4.withUnsafeMutableBufferPointer { buffer in
            softmaxRowsPointer(buffer.baseAddress!, rows: 1, cols: 4)
        }
        for i in 0..<4 {
            assertApproximatelyEqual(pointerResult4[i], arrayResult4[i],
                                    "Test case 4: Results should match at index \(i)")
        }

        // Test case 5: Uniform values
        let testCase5: [Float] = [5.0, 5.0, 5.0, 5.0, 5.0]
        var arrayResult5 = testCase5
        var pointerResult5 = testCase5
        softmaxRows(&arrayResult5, rows: 1, cols: 5)
        pointerResult5.withUnsafeMutableBufferPointer { buffer in
            softmaxRowsPointer(buffer.baseAddress!, rows: 1, cols: 5)
        }
        for i in 0..<5 {
            assertApproximatelyEqual(pointerResult5[i], arrayResult5[i],
                                    "Test case 5: Results should match at index \(i)")
        }

        // Test case 6: Single element
        let testCase6: [Float] = [42.0]
        var arrayResult6 = testCase6
        var pointerResult6 = testCase6
        softmaxRows(&arrayResult6, rows: 1, cols: 1)
        pointerResult6.withUnsafeMutableBufferPointer { buffer in
            softmaxRowsPointer(buffer.baseAddress!, rows: 1, cols: 1)
        }
        assertApproximatelyEqual(pointerResult6[0], arrayResult6[0],
                                "Test case 6: Single element results should match")

        // Test case 7: Batch processing (multiple rows)
        let testCase7: [Float] = [
            0.1, 0.2, 0.3,      // Row 0
            1.0, 2.0, 3.0,      // Row 1
            -1.0, -0.5, 0.0,    // Row 2
            100.0, 101.0, 102.0 // Row 3
        ]
        var arrayResult7 = testCase7
        var pointerResult7 = testCase7
        softmaxRows(&arrayResult7, rows: 4, cols: 3)
        pointerResult7.withUnsafeMutableBufferPointer { buffer in
            softmaxRowsPointer(buffer.baseAddress!, rows: 4, cols: 3)
        }
        for i in 0..<12 {
            assertApproximatelyEqual(pointerResult7[i], arrayResult7[i],
                                    "Test case 7: Results should match at index \(i)")
        }

        // Test case 8: Zeros
        let testCase8: [Float] = [0.0, 0.0, 0.0, 0.0]
        var arrayResult8 = testCase8
        var pointerResult8 = testCase8
        softmaxRows(&arrayResult8, rows: 1, cols: 4)
        pointerResult8.withUnsafeMutableBufferPointer { buffer in
            softmaxRowsPointer(buffer.baseAddress!, rows: 1, cols: 4)
        }
        for i in 0..<4 {
            assertApproximatelyEqual(pointerResult8[i], arrayResult8[i],
                                    "Test case 8: Results should match at index \(i)")
        }
    }

    // =============================================================================
    // MARK: - Softmax Realistic Scenario Tests
    // =============================================================================

    func testSoftmaxTypicalNeuralNetworkOutput() {
        // Test softmax with values typical of neural network final layer
        // (before softmax) for MNIST classification (10 classes)
        var data: [Float] = [
            2.3, -1.2, 0.5, 3.1, -0.8, 1.7, 0.0, -2.1, 1.1, 0.3
        ]

        softmaxRows(&data, rows: 1, cols: 10)

        // Should produce valid probability distribution
        let sum = data.reduce(0.0, +)
        assertApproximatelyEqual(sum, 1.0, "Neural network output should sum to 1.0")
        assertAllInRange01(data)

        // Class 3 (originally 3.1) should have highest probability
        let maxIndex = data.indices.max(by: { data[$0] < data[$1] })!
        XCTAssertEqual(maxIndex, 3, "Highest logit should get highest probability")
    }

    func testSoftmaxBatchedNeuralNetworkOutputs() {
        // Test softmax with batch of neural network outputs
        // Simulating batch size 4, 10 classes
        var data: [Float] = [
            // Sample 0 - clear winner at class 2
            0.1, 0.2, 5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            // Sample 1 - close competition between classes 4 and 5
            0.0, 0.0, 0.0, 0.0, 2.5, 2.6, 0.0, 0.0, 0.0, 0.0,
            // Sample 2 - all similar (uncertain)
            1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0, 1.0, 1.0, 1.0,
            // Sample 3 - strong negative except one
            -10.0, -10.0, -10.0, -10.0, -10.0, 3.0, -10.0, -10.0, -10.0, -10.0
        ]
        let rows = 4
        let cols = 10

        softmaxRows(&data, rows: rows, cols: cols)

        // Verify each sample
        for r in 0..<rows {
            let rowSum = sumRow(data, row: r, cols: cols)
            assertApproximatelyEqual(rowSum, 1.0, "Batch sample \(r) should sum to 1.0")

            let rowData = (0..<cols).map { data[r * cols + $0] }
            assertAllInRange01(rowData)
        }

        // Sample 0: class 2 should dominate
        XCTAssertGreaterThan(data[2], 0.9, "Clear winner should have >90% probability")

        // Sample 3: class 5 should dominate
        XCTAssertGreaterThan(data[3 * cols + 5], 0.99,
                            "Only positive logit should dominate")
    }
}
