// ============================================================================
// SimpleRngTests.swift - Tests for Simple Random Number Generator
// ============================================================================
//
// This test suite validates the reproducibility and correctness of SimpleRng:
// - Deterministic sequences with same seed
// - Different sequences with different seeds
// - Proper range handling for all random functions
// - Edge cases (zero seed, zero upper bound, etc.)
// - Reproducibility across multiple invocations
//
// ============================================================================

import XCTest
@testable import MNISTCommon
import Foundation

final class SimpleRngTests: XCTestCase {

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

    /// Verifies that a float value is in the range [0, 1)
    private func assertInRange01(_ value: Float,
                                 file: StaticString = #file,
                                 line: UInt = #line) {
        XCTAssertGreaterThanOrEqual(value, 0.0,
                                   "Value should be >= 0",
                                   file: file, line: line)
        XCTAssertLessThan(value, 1.0,
                         "Value should be < 1",
                         file: file, line: line)
    }

    /// Verifies that a float value is in the specified range [low, high)
    private func assertInRange(_ value: Float, _ low: Float, _ high: Float,
                               file: StaticString = #file,
                               line: UInt = #line) {
        XCTAssertGreaterThanOrEqual(value, low,
                                   "Value should be >= \(low)",
                                   file: file, line: line)
        XCTAssertLessThan(value, high,
                         "Value should be < \(high)",
                         file: file, line: line)
    }

    // =============================================================================
    // MARK: - Initialization Tests
    // =============================================================================

    func testInitializationWithNonZeroSeed() {
        // Test that RNG initializes with non-zero seed
        var rng = SimpleRng(seed: 12345)
        let value = rng.nextUInt32()

        // Should produce a value (specific value depends on xorshift algorithm)
        XCTAssertNotEqual(value, 0, "RNG should produce values")
    }

    func testInitializationWithZeroSeed() {
        // Test that zero seed uses fixed value (0x9e3779b97f4a7c15)
        var rng1 = SimpleRng(seed: 0)
        var rng2 = SimpleRng(seed: 0x9e3779b97f4a7c15)

        // Both should produce the same sequence
        for _ in 0..<10 {
            XCTAssertEqual(rng1.nextUInt32(), rng2.nextUInt32(),
                          "Zero seed should use fixed value 0x9e3779b97f4a7c15")
        }
    }

    // =============================================================================
    // MARK: - Core Reproducibility Tests
    // =============================================================================

    func testSameSeedProducesSameSequence() {
        // Test that the same seed always produces the same sequence
        let seed: UInt64 = 42
        var rng1 = SimpleRng(seed: seed)
        var rng2 = SimpleRng(seed: seed)

        // Generate sequence from both RNGs
        for i in 0..<100 {
            let value1 = rng1.nextUInt32()
            let value2 = rng2.nextUInt32()
            XCTAssertEqual(value1, value2,
                          "Same seed should produce same value at position \(i)")
        }
    }

    func testDifferentSeedsProduceDifferentSequences() {
        // Test that different seeds produce different sequences
        var rng1 = SimpleRng(seed: 42)
        var rng2 = SimpleRng(seed: 43)

        // At least one value in the sequence should be different
        var foundDifference = false
        for _ in 0..<10 {
            if rng1.nextUInt32() != rng2.nextUInt32() {
                foundDifference = true
                break
            }
        }

        XCTAssertTrue(foundDifference,
                     "Different seeds should produce different sequences")
    }

    func testSequenceIsFullyDeterministic() {
        // Test that the sequence is fully deterministic across multiple runs
        let seed: UInt64 = 123456789
        var rng = SimpleRng(seed: seed)

        // Generate first sequence
        let firstSequence = (0..<50).map { _ in rng.nextUInt32() }

        // Reset with same seed and generate second sequence
        rng = SimpleRng(seed: seed)
        let secondSequence = (0..<50).map { _ in rng.nextUInt32() }

        // Sequences should be identical
        XCTAssertEqual(firstSequence, secondSequence,
                      "Same seed should always produce identical sequences")
    }

    func testMultipleSeedReproducibility() {
        // Test reproducibility with various seeds
        let seeds: [UInt64] = [1, 42, 100, 999, 123456, 0xFFFFFFFF, 0x123456789ABCDEF]

        for seed in seeds {
            var rng1 = SimpleRng(seed: seed)
            var rng2 = SimpleRng(seed: seed)

            for i in 0..<20 {
                XCTAssertEqual(rng1.nextUInt32(), rng2.nextUInt32(),
                              "Seed \(seed) should produce same sequence at position \(i)")
            }
        }
    }

    // =============================================================================
    // MARK: - nextUInt32() Tests
    // =============================================================================

    func testNextUInt32ProducesValues() {
        // Test that nextUInt32 produces values across the full UInt32 range
        var rng = SimpleRng(seed: 42)

        var foundNonZero = false
        for _ in 0..<100 {
            let value = rng.nextUInt32()
            if value != 0 {
                foundNonZero = true
                break
            }
        }

        XCTAssertTrue(foundNonZero, "Should produce non-zero values")
    }

    func testNextUInt32Reproducibility() {
        // Test that nextUInt32 produces reproducible sequences
        let seed: UInt64 = 9876543210
        var rng1 = SimpleRng(seed: seed)
        var rng2 = SimpleRng(seed: seed)

        let sequence1 = (0..<100).map { _ in rng1.nextUInt32() }
        let sequence2 = (0..<100).map { _ in rng2.nextUInt32() }

        XCTAssertEqual(sequence1, sequence2,
                      "nextUInt32 should produce reproducible sequences")
    }

    func testNextUInt32KnownSequence() {
        // Test against known values from the xorshift algorithm
        var rng = SimpleRng(seed: 1)

        // Generate a few values and verify they're consistent
        let firstValue = rng.nextUInt32()

        // Reset and verify same value
        rng = SimpleRng(seed: 1)
        XCTAssertEqual(rng.nextUInt32(), firstValue,
                      "Should produce same first value with seed 1")
    }

    // =============================================================================
    // MARK: - nextFloat() Tests
    // =============================================================================

    func testNextFloatInRange01() {
        // Test that nextFloat produces values in [0, 1)
        var rng = SimpleRng(seed: 42)

        for _ in 0..<1000 {
            let value = rng.nextFloat()
            assertInRange01(value)
        }
    }

    func testNextFloatReproducibility() {
        // Test that nextFloat produces reproducible sequences
        let seed: UInt64 = 777
        var rng1 = SimpleRng(seed: seed)
        var rng2 = SimpleRng(seed: seed)

        for i in 0..<100 {
            let value1 = rng1.nextFloat()
            let value2 = rng2.nextFloat()
            assertApproximatelyEqual(value1, value2,
                                    "nextFloat should be reproducible at position \(i)")
        }
    }

    func testNextFloatProducesVariedValues() {
        // Test that nextFloat produces varied values (not all the same)
        var rng = SimpleRng(seed: 42)

        let values = (0..<100).map { _ in rng.nextFloat() }
        let uniqueValues = Set(values)

        XCTAssertGreaterThan(uniqueValues.count, 90,
                            "nextFloat should produce varied values")
    }

    func testNextFloatNeverReturnsOne() {
        // Test that nextFloat never returns exactly 1.0 (should be [0, 1))
        var rng = SimpleRng(seed: 42)

        for _ in 0..<1000 {
            let value = rng.nextFloat()
            XCTAssertLessThan(value, 1.0, "nextFloat should never return 1.0")
        }
    }

    func testNextFloatSequenceIndependence() {
        // Test that nextFloat sequences are independent for different seeds
        var rng1 = SimpleRng(seed: 100)
        var rng2 = SimpleRng(seed: 200)

        let sequence1 = (0..<50).map { _ in rng1.nextFloat() }
        let sequence2 = (0..<50).map { _ in rng2.nextFloat() }

        XCTAssertNotEqual(sequence1, sequence2,
                         "Different seeds should produce different float sequences")
    }

    // =============================================================================
    // MARK: - uniform() Tests
    // =============================================================================

    func testUniformInSpecifiedRange() {
        // Test that uniform produces values in [low, high)
        var rng = SimpleRng(seed: 42)
        let low: Float = -10.0
        let high: Float = 10.0

        for _ in 0..<1000 {
            let value = rng.uniform(low, high)
            assertInRange(value, low, high)
        }
    }

    func testUniformWithPositiveRange() {
        // Test uniform with positive range [0, 100)
        var rng = SimpleRng(seed: 123)

        for _ in 0..<500 {
            let value = rng.uniform(0, 100)
            assertInRange(value, 0, 100)
        }
    }

    func testUniformWithNegativeRange() {
        // Test uniform with negative range [-100, -50)
        var rng = SimpleRng(seed: 456)

        for _ in 0..<500 {
            let value = rng.uniform(-100, -50)
            assertInRange(value, -100, -50)
        }
    }

    func testUniformWithSmallRange() {
        // Test uniform with very small range [0, 0.001)
        var rng = SimpleRng(seed: 789)

        for _ in 0..<100 {
            let value = rng.uniform(0, 0.001)
            assertInRange(value, 0, 0.001)
        }
    }

    func testUniformWithLargeRange() {
        // Test uniform with large range [-1000, 1000)
        var rng = SimpleRng(seed: 321)

        for _ in 0..<500 {
            let value = rng.uniform(-1000, 1000)
            assertInRange(value, -1000, 1000)
        }
    }

    func testUniformReproducibility() {
        // Test that uniform produces reproducible sequences
        let seed: UInt64 = 999
        var rng1 = SimpleRng(seed: seed)
        var rng2 = SimpleRng(seed: seed)

        for i in 0..<100 {
            let value1 = rng1.uniform(-5, 5)
            let value2 = rng2.uniform(-5, 5)
            assertApproximatelyEqual(value1, value2,
                                    "uniform should be reproducible at position \(i)")
        }
    }

    func testUniformProducesVariedValues() {
        // Test that uniform produces varied values across the range
        var rng = SimpleRng(seed: 555)

        var foundLow = false
        var foundHigh = false

        for _ in 0..<1000 {
            let value = rng.uniform(0, 100)
            if value < 25 {
                foundLow = true
            }
            if value > 75 {
                foundHigh = true
            }
        }

        XCTAssertTrue(foundLow, "Should produce values in lower quartile")
        XCTAssertTrue(foundHigh, "Should produce values in upper quartile")
    }

    func testUniformNeverReturnsHigh() {
        // Test that uniform never returns exactly high (should be [low, high))
        var rng = SimpleRng(seed: 666)
        let high: Float = 10.0

        for _ in 0..<1000 {
            let value = rng.uniform(0, high)
            XCTAssertLessThan(value, high,
                             "uniform should never return exactly high value")
        }
    }

    // =============================================================================
    // MARK: - nextInt() Tests
    // =============================================================================

    func testNextIntInRange() {
        // Test that nextInt produces values in [0, upper)
        var rng = SimpleRng(seed: 42)
        let upper = 100

        for _ in 0..<1000 {
            let value = rng.nextInt(upper: upper)
            XCTAssertGreaterThanOrEqual(value, 0, "nextInt should be >= 0")
            XCTAssertLessThan(value, upper, "nextInt should be < upper")
        }
    }

    func testNextIntWithZeroUpper() {
        // Test that nextInt with upper=0 returns 0
        var rng = SimpleRng(seed: 42)

        for _ in 0..<10 {
            let value = rng.nextInt(upper: 0)
            XCTAssertEqual(value, 0, "nextInt with upper=0 should return 0")
        }
    }

    func testNextIntWithOneUpper() {
        // Test that nextInt with upper=1 always returns 0
        var rng = SimpleRng(seed: 42)

        for _ in 0..<100 {
            let value = rng.nextInt(upper: 1)
            XCTAssertEqual(value, 0, "nextInt with upper=1 should always return 0")
        }
    }

    func testNextIntWithSmallUpper() {
        // Test nextInt with small upper bound (10)
        var rng = SimpleRng(seed: 123)

        for _ in 0..<200 {
            let value = rng.nextInt(upper: 10)
            XCTAssertGreaterThanOrEqual(value, 0, "Should be >= 0")
            XCTAssertLessThan(value, 10, "Should be < 10")
        }
    }

    func testNextIntWithLargeUpper() {
        // Test nextInt with large upper bound (1,000,000)
        var rng = SimpleRng(seed: 456)

        for _ in 0..<100 {
            let value = rng.nextInt(upper: 1_000_000)
            XCTAssertGreaterThanOrEqual(value, 0, "Should be >= 0")
            XCTAssertLessThan(value, 1_000_000, "Should be < 1,000,000")
        }
    }

    func testNextIntReproducibility() {
        // Test that nextInt produces reproducible sequences
        let seed: UInt64 = 888
        var rng1 = SimpleRng(seed: seed)
        var rng2 = SimpleRng(seed: seed)

        for i in 0..<100 {
            let value1 = rng1.nextInt(upper: 50)
            let value2 = rng2.nextInt(upper: 50)
            XCTAssertEqual(value1, value2,
                          "nextInt should be reproducible at position \(i)")
        }
    }

    func testNextIntProducesVariedValues() {
        // Test that nextInt produces varied values across the range
        var rng = SimpleRng(seed: 111)
        let upper = 10

        var counts = [Int](repeating: 0, count: upper)
        for _ in 0..<1000 {
            let value = rng.nextInt(upper: upper)
            counts[value] += 1
        }

        // All values should appear at least once in 1000 samples
        for i in 0..<upper {
            XCTAssertGreaterThan(counts[i], 0,
                                "Value \(i) should appear at least once")
        }
    }

    func testNextIntNeverReturnsUpper() {
        // Test that nextInt never returns exactly upper (should be [0, upper))
        var rng = SimpleRng(seed: 222)
        let upper = 50

        for _ in 0..<1000 {
            let value = rng.nextInt(upper: upper)
            XCTAssertLessThan(value, upper,
                             "nextInt should never return exactly upper")
        }
    }

    // =============================================================================
    // MARK: - reseedFromTime() Tests
    // =============================================================================

    func testReseedFromTimeChangesState() {
        // Test that reseedFromTime changes the RNG state
        var rng = SimpleRng(seed: 42)
        let valueBeforeReseed = rng.nextUInt32()

        // Reset to same seed
        rng = SimpleRng(seed: 42)
        let valueWithoutReseed = rng.nextUInt32()

        // These should be the same
        XCTAssertEqual(valueBeforeReseed, valueWithoutReseed,
                      "Same seed should produce same value")

        // Now reseed from time and generate value
        rng = SimpleRng(seed: 42)
        rng.reseedFromTime()
        let valueAfterReseed = rng.nextUInt32()

        // This should be different (extremely unlikely to be the same)
        XCTAssertNotEqual(valueAfterReseed, valueBeforeReseed,
                         "reseedFromTime should change the sequence")
    }

    func testReseedFromTimeProducesDifferentSequences() {
        // Test that two reseedFromTime calls produce different sequences
        var rng1 = SimpleRng(seed: 0)
        rng1.reseedFromTime()
        let sequence1 = (0..<10).map { _ in rng1.nextUInt32() }

        // Small delay to ensure different timestamp
        Thread.sleep(forTimeInterval: 0.001)

        var rng2 = SimpleRng(seed: 0)
        rng2.reseedFromTime()
        let sequence2 = (0..<10).map { _ in rng2.nextUInt32() }

        // Sequences should be different (at least one value)
        XCTAssertNotEqual(sequence1, sequence2,
                         "Different reseedFromTime calls should produce different sequences")
    }

    // =============================================================================
    // MARK: - Cross-Method Reproducibility Tests
    // =============================================================================

    func testAllMethodsShareSameState() {
        // Test that all methods affect the same internal state
        var rng1 = SimpleRng(seed: 12345)
        var rng2 = SimpleRng(seed: 12345)

        // Call different methods on both RNGs
        _ = rng1.nextUInt32()
        _ = rng2.nextUInt32()

        // Next calls should still be synchronized
        assertApproximatelyEqual(rng1.nextFloat(), rng2.nextFloat(),
                                "Methods should share same state")

        _ = rng1.uniform(-1, 1)
        _ = rng2.uniform(-1, 1)

        XCTAssertEqual(rng1.nextInt(upper: 100), rng2.nextInt(upper: 100),
                      "Methods should share same state")
    }

    func testMixedMethodCallsReproducibility() {
        // Test reproducibility with mixed method calls
        let seed: UInt64 = 54321
        var rng1 = SimpleRng(seed: seed)
        var rng2 = SimpleRng(seed: seed)

        // Mixed sequence of calls
        for _ in 0..<20 {
            XCTAssertEqual(rng1.nextUInt32(), rng2.nextUInt32(),
                          "nextUInt32 should be reproducible")
            assertApproximatelyEqual(rng1.nextFloat(), rng2.nextFloat(),
                                    "nextFloat should be reproducible")
            assertApproximatelyEqual(rng1.uniform(-10, 10), rng2.uniform(-10, 10),
                                    "uniform should be reproducible")
            XCTAssertEqual(rng1.nextInt(upper: 50), rng2.nextInt(upper: 50),
                          "nextInt should be reproducible")
        }
    }

    // =============================================================================
    // MARK: - Statistical Distribution Tests (Basic)
    // =============================================================================

    func testNextFloatDistributionApproximatelyUniform() {
        // Basic test that nextFloat produces approximately uniform distribution
        var rng = SimpleRng(seed: 42)
        let sampleCount = 10000
        let bucketCount = 10
        var buckets = [Int](repeating: 0, count: bucketCount)

        for _ in 0..<sampleCount {
            let value = rng.nextFloat()
            let bucketIndex = min(Int(value * Float(bucketCount)), bucketCount - 1)
            buckets[bucketIndex] += 1
        }

        // Each bucket should have roughly sampleCount/bucketCount samples
        // Allow 20% deviation
        let expectedPerBucket = sampleCount / bucketCount
        let tolerance = Int(Double(expectedPerBucket) * 0.2)

        for i in 0..<bucketCount {
            XCTAssertGreaterThan(buckets[i], expectedPerBucket - tolerance,
                                "Bucket \(i) should have enough samples")
            XCTAssertLessThan(buckets[i], expectedPerBucket + tolerance,
                             "Bucket \(i) should not have too many samples")
        }
    }

    func testUniformDistributionApproximatelyUniform() {
        // Basic test that uniform produces approximately uniform distribution
        var rng = SimpleRng(seed: 999)
        let sampleCount = 10000
        let bucketCount = 10
        var buckets = [Int](repeating: 0, count: bucketCount)
        let low: Float = 0
        let high: Float = 100

        for _ in 0..<sampleCount {
            let value = rng.uniform(low, high)
            let normalized = (value - low) / (high - low)
            let bucketIndex = min(Int(normalized * Float(bucketCount)), bucketCount - 1)
            buckets[bucketIndex] += 1
        }

        // Each bucket should have roughly sampleCount/bucketCount samples
        // Allow 20% deviation
        let expectedPerBucket = sampleCount / bucketCount
        let tolerance = Int(Double(expectedPerBucket) * 0.2)

        for i in 0..<bucketCount {
            XCTAssertGreaterThan(buckets[i], expectedPerBucket - tolerance,
                                "Bucket \(i) should have enough samples")
            XCTAssertLessThan(buckets[i], expectedPerBucket + tolerance,
                             "Bucket \(i) should not have too many samples")
        }
    }

    func testNextIntDistributionApproximatelyUniform() {
        // Basic test that nextInt produces approximately uniform distribution
        var rng = SimpleRng(seed: 777)
        let upper = 20
        let samplesPerValue = 500
        let sampleCount = upper * samplesPerValue
        var counts = [Int](repeating: 0, count: upper)

        for _ in 0..<sampleCount {
            let value = rng.nextInt(upper: upper)
            counts[value] += 1
        }

        // Each value should appear roughly samplesPerValue times
        // Allow 20% deviation
        let tolerance = Int(Double(samplesPerValue) * 0.2)

        for i in 0..<upper {
            XCTAssertGreaterThan(counts[i], samplesPerValue - tolerance,
                                "Value \(i) should appear enough times")
            XCTAssertLessThan(counts[i], samplesPerValue + tolerance,
                             "Value \(i) should not appear too many times")
        }
    }

    // =============================================================================
    // MARK: - Realistic Scenario Tests
    // =============================================================================

    func testWeightInitializationScenario() {
        // Test typical weight initialization scenario for neural networks
        var rng = SimpleRng(seed: 42)
        let layerSize = 784 * 128  // MNIST input to hidden layer

        // Initialize weights in range [-0.1, 0.1]
        let weights = (0..<layerSize).map { _ in rng.uniform(-0.1, 0.1) }

        // Verify all weights in correct range
        for (i, weight) in weights.enumerated() {
            assertInRange(weight, -0.1, 0.1)
            XCTAssertFalse(weight.isNaN, "Weight at \(i) should not be NaN")
            XCTAssertFalse(weight.isInfinite, "Weight at \(i) should not be infinite")
        }
    }

    func testDataShufflingScenario() {
        // Test typical data shuffling scenario
        var rng = SimpleRng(seed: 123)
        let dataSize = 60000  // MNIST training set size

        // Generate shuffled indices
        var indices = Array(0..<dataSize)
        for i in (0..<dataSize).reversed() {
            let j = rng.nextInt(upper: i + 1)
            indices.swapAt(i, j)
        }

        // Verify all indices present exactly once
        let sortedIndices = indices.sorted()
        XCTAssertEqual(sortedIndices, Array(0..<dataSize),
                      "Shuffled indices should contain all original indices")
    }

    func testReproducibleDataShuffling() {
        // Test that data shuffling is reproducible with same seed
        let seed: UInt64 = 999
        let dataSize = 1000

        // Shuffle with first RNG
        var rng1 = SimpleRng(seed: seed)
        var indices1 = Array(0..<dataSize)
        for i in (0..<dataSize).reversed() {
            let j = rng1.nextInt(upper: i + 1)
            indices1.swapAt(i, j)
        }

        // Shuffle with second RNG (same seed)
        var rng2 = SimpleRng(seed: seed)
        var indices2 = Array(0..<dataSize)
        for i in (0..<dataSize).reversed() {
            let j = rng2.nextInt(upper: i + 1)
            indices2.swapAt(i, j)
        }

        // Shuffles should be identical
        XCTAssertEqual(indices1, indices2,
                      "Same seed should produce identical shuffles")
    }

    func testMinibatchSamplingScenario() {
        // Test typical mini-batch sampling scenario
        var rng = SimpleRng(seed: 456)
        let dataSize = 60000
        let batchSize = 128
        let numBatches = 10

        for _ in 0..<numBatches {
            // Sample batch indices
            let batchIndices = (0..<batchSize).map { _ in rng.nextInt(upper: dataSize) }

            // Verify all indices in valid range
            for index in batchIndices {
                XCTAssertGreaterThanOrEqual(index, 0, "Index should be >= 0")
                XCTAssertLessThan(index, dataSize, "Index should be < dataSize")
            }
        }
    }
}
