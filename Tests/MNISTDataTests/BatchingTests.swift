// ============================================================================
// BatchingTests.swift - Tests for MNIST Batch Creation and Shuffling
// ============================================================================
//
// This test suite validates the batch creation and shuffling functionality:
// - Batch size validation
// - Correct number of batches
// - Data completeness (all samples included)
// - Shuffling behavior
// - Edge cases (partial batches, exact division)
//
// ============================================================================

import XCTest
import MLX
@testable import MNISTData
import Foundation

final class BatchingTests: MLXTestCase {

    // =============================================================================
    // MARK: - Test Utilities
    // =============================================================================

    /// Creates synthetic MNIST-like data for testing
    /// - Parameters:
    ///   - numSamples: Number of samples to create
    ///   - imageSize: Size of each flattened image (default: 784)
    /// - Returns: Tuple of (images, labels) where images are sequential floats and labels are indices mod 10
    private func createSyntheticData(numSamples: Int, imageSize: Int = 784) -> (images: MLXArray, labels: MLXArray) {
        // Create images with sequential values for easy verification
        // Image i has all pixels = Float(i)
        var imageData: [Float] = []
        for i in 0..<numSamples {
            imageData.append(contentsOf: [Float](repeating: Float(i), count: imageSize))
        }

        // Create labels: 0, 1, 2, ..., 9, 0, 1, 2, ...
        let labelData = (0..<numSamples).map { Int32($0 % 10) }

        let images = MLXArray(imageData, [numSamples, imageSize])
        let labels = MLXArray(labelData, [numSamples])

        return (images, labels)
    }

    /// Extracts all samples from batches and returns them as a single array
    /// - Parameter batches: Array of (images, labels) batch tuples
    /// - Returns: Tuple of (all images, all labels) concatenated from batches
    private func flattenBatches(_ batches: [(images: MLXArray, labels: MLXArray)]) -> (images: MLXArray, labels: MLXArray) {
        let allImages = concatenated(batches.map { $0.images }, axis: 0)
        let allLabels = concatenated(batches.map { $0.labels }, axis: 0)
        return (allImages, allLabels)
    }

    // =============================================================================
    // MARK: - Basic Batching Tests
    // =============================================================================

    func testBatchCreationWithoutShuffling() throws {
        // Create a small dataset: 10 samples
        let (images, labels) = createSyntheticData(numSamples: 10)

        // Create batches of size 3 (should get 4 batches: 3, 3, 3, 1)
        let batches = batchDataset(images, labels, batchSize: 3, shuffle: false)

        // Validate number of batches
        XCTAssertEqual(batches.count, 4, "Should create 4 batches (3+3+3+1)")

        // Validate batch sizes
        XCTAssertEqual(batches[0].images.shape[0], 3, "First batch should have 3 samples")
        XCTAssertEqual(batches[1].images.shape[0], 3, "Second batch should have 3 samples")
        XCTAssertEqual(batches[2].images.shape[0], 3, "Third batch should have 3 samples")
        XCTAssertEqual(batches[3].images.shape[0], 1, "Fourth batch should have 1 sample")

        // Validate that data is in order (no shuffling)
        for i in 0..<3 {
            let label = batches[0].labels[i].item(Int32.self)
            XCTAssertEqual(label, Int32(i), "First batch labels should be sequential")
        }
    }

    func testBatchCreationExactDivision() throws {
        // Create a dataset that divides evenly: 12 samples
        let (images, labels) = createSyntheticData(numSamples: 12)

        // Create batches of size 4 (should get exactly 3 batches)
        let batches = batchDataset(images, labels, batchSize: 4, shuffle: false)

        // Validate number of batches
        XCTAssertEqual(batches.count, 3, "Should create exactly 3 batches")

        // Validate all batches have the same size
        for i in 0..<batches.count {
            XCTAssertEqual(batches[i].images.shape[0], 4, "Batch \(i) should have 4 samples")
            XCTAssertEqual(batches[i].labels.shape[0], 4, "Batch \(i) labels should have 4 samples")
        }
    }

    func testBatchCreationSingleBatch() throws {
        // Create a dataset smaller than batch size
        let (images, labels) = createSyntheticData(numSamples: 5)

        // Create batches of size 10 (larger than dataset)
        let batches = batchDataset(images, labels, batchSize: 10, shuffle: false)

        // Should create exactly 1 batch with all 5 samples
        XCTAssertEqual(batches.count, 1, "Should create 1 batch")
        XCTAssertEqual(batches[0].images.shape[0], 5, "Single batch should have all 5 samples")
    }

    func testBatchCreationBatchSizeOne() throws {
        // Test with batch size of 1 (edge case)
        let (images, labels) = createSyntheticData(numSamples: 5)

        let batches = batchDataset(images, labels, batchSize: 1, shuffle: false)

        // Should create 5 batches, each with 1 sample
        XCTAssertEqual(batches.count, 5, "Should create 5 batches")

        for i in 0..<batches.count {
            XCTAssertEqual(batches[i].images.shape[0], 1, "Each batch should have 1 sample")
            XCTAssertEqual(batches[i].labels.shape[0], 1, "Each batch label should have 1 sample")
        }
    }

    // =============================================================================
    // MARK: - Data Completeness Tests
    // =============================================================================

    func testAllSamplesIncludedInBatches() throws {
        // Create a dataset and verify all samples appear in batches
        let (images, labels) = createSyntheticData(numSamples: 25)

        let batches = batchDataset(images, labels, batchSize: 7, shuffle: false)

        // Flatten batches back into single arrays
        let (batchedImages, batchedLabels) = flattenBatches(batches)

        // Verify total count
        XCTAssertEqual(batchedImages.shape[0], 25, "All 25 images should be in batches")
        XCTAssertEqual(batchedLabels.shape[0], 25, "All 25 labels should be in batches")

        // Verify data integrity (without shuffling, should be identical)
        for i in 0..<25 {
            let originalLabel = labels[i].item(Int32.self)
            let batchedLabel = batchedLabels[i].item(Int32.self)
            XCTAssertEqual(originalLabel, batchedLabel, "Label at index \(i) should match")
        }
    }

    func testBatchShapeConsistency() throws {
        // Verify that batch shapes are consistent
        let (images, labels) = createSyntheticData(numSamples: 100)

        let batches = batchDataset(images, labels, batchSize: 32, shuffle: false)

        // All batches except possibly the last should have size 32
        for i in 0..<(batches.count - 1) {
            XCTAssertEqual(batches[i].images.shape, [32, 784],
                          "Batch \(i) images should have shape [32, 784]")
            XCTAssertEqual(batches[i].labels.shape, [32],
                          "Batch \(i) labels should have shape [32]")
        }

        // Last batch should have remaining samples
        let lastBatch = batches[batches.count - 1]
        let expectedLastSize = 100 % 32 == 0 ? 32 : 100 % 32
        XCTAssertEqual(lastBatch.images.shape[0], expectedLastSize,
                      "Last batch should have \(expectedLastSize) samples")
    }

    // =============================================================================
    // MARK: - Shuffling Tests
    // =============================================================================

    func testShufflingChangesOrder() throws {
        // Create a dataset with sequential labels
        let (images, labels) = createSyntheticData(numSamples: 100)

        // Create batches with shuffling
        let batches = batchDataset(images, labels, batchSize: 10, shuffle: true)

        // Flatten batches to get shuffled order
        let (_, shuffledLabels) = flattenBatches(batches)

        // Check if order has changed
        // With 100 samples, it's extremely unlikely that shuffling produces the same order
        var orderChanged = false
        for i in 0..<100 {
            let originalLabel = labels[i].item(Int32.self)
            let shuffledLabel = shuffledLabels[i].item(Int32.self)
            if originalLabel != shuffledLabel {
                orderChanged = true
                break
            }
        }

        XCTAssertTrue(orderChanged, "Shuffling should change the order of samples")
    }

    func testShufflingPreservesAllSamples() throws {
        // Verify that shuffling doesn't lose or duplicate samples
        let (images, labels) = createSyntheticData(numSamples: 50)

        let batches = batchDataset(images, labels, batchSize: 8, shuffle: true)

        // Flatten batches
        let (_, shuffledLabels) = flattenBatches(batches)

        // Count occurrences of each label
        var labelCounts = [Int32: Int]()
        for i in 0..<50 {
            let label = shuffledLabels[i].item(Int32.self)
            labelCounts[label, default: 0] += 1
        }

        // Verify each digit (0-9) appears the correct number of times
        // With 50 samples and labels = i % 10, each digit should appear 5 times
        for digit in 0..<10 {
            XCTAssertEqual(labelCounts[Int32(digit)], 5,
                          "Digit \(digit) should appear 5 times after shuffling")
        }
    }

    func testNoShufflingPreservesOrder() throws {
        // Verify that shuffle=false keeps original order
        let (images, labels) = createSyntheticData(numSamples: 30)

        let batches = batchDataset(images, labels, batchSize: 10, shuffle: false)

        // Flatten batches
        let (_, batchedLabels) = flattenBatches(batches)

        // Verify order is preserved
        for i in 0..<30 {
            let originalLabel = labels[i].item(Int32.self)
            let batchedLabel = batchedLabels[i].item(Int32.self)
            XCTAssertEqual(originalLabel, batchedLabel,
                          "Label at index \(i) should be in original order")
        }
    }

    func testShufflingWithSingleBatch() throws {
        // Test shuffling when there's only one batch
        let (images, labels) = createSyntheticData(numSamples: 10)

        let batches = batchDataset(images, labels, batchSize: 20, shuffle: true)

        // Should have exactly 1 batch
        XCTAssertEqual(batches.count, 1, "Should have 1 batch")
        XCTAssertEqual(batches[0].images.shape[0], 10, "Batch should have all 10 samples")

        // Verify all samples are present (even with shuffling)
        var labelCounts = [Int32: Int]()
        for i in 0..<10 {
            let label = batches[0].labels[i].item(Int32.self)
            labelCounts[label, default: 0] += 1
        }

        for digit in 0..<10 {
            XCTAssertEqual(labelCounts[Int32(digit)], 1,
                          "Each digit should appear exactly once")
        }
    }

    // =============================================================================
    // MARK: - Edge Cases
    // =============================================================================

    func testEmptyDataset() throws {
        // Test with empty dataset
        let images = MLXArray([Float](), [0, 784])
        let labels = MLXArray([Int32](), [0])

        let batches = batchDataset(images, labels, batchSize: 32, shuffle: false)

        // Should create 0 batches
        XCTAssertEqual(batches.count, 0, "Empty dataset should produce 0 batches")
    }

    func testLargeDatasetBatching() throws {
        // Test with a larger dataset to ensure scalability
        let (images, labels) = createSyntheticData(numSamples: 1000)

        let batches = batchDataset(images, labels, batchSize: 64, shuffle: false)

        // 1000 / 64 = 15 full batches + 1 partial batch (40 samples)
        XCTAssertEqual(batches.count, 16, "Should create 16 batches")

        // Verify first 15 batches have 64 samples
        for i in 0..<15 {
            XCTAssertEqual(batches[i].images.shape[0], 64, "Batch \(i) should have 64 samples")
        }

        // Last batch should have 40 samples (1000 - 15*64 = 40)
        XCTAssertEqual(batches[15].images.shape[0], 40, "Last batch should have 40 samples")
    }

    func testBatchSizeEqualToDatasetSize() throws {
        // Test when batch size equals dataset size
        let (images, labels) = createSyntheticData(numSamples: 50)

        let batches = batchDataset(images, labels, batchSize: 50, shuffle: false)

        // Should create exactly 1 batch with all samples
        XCTAssertEqual(batches.count, 1, "Should create 1 batch")
        XCTAssertEqual(batches[0].images.shape[0], 50, "Batch should have all 50 samples")
        XCTAssertEqual(batches[0].labels.shape[0], 50, "Batch should have all 50 labels")
    }

    func testImageLabelCorrespondence() throws {
        // Verify that images and labels stay aligned after batching and shuffling
        let (images, labels) = createSyntheticData(numSamples: 40)

        let batches = batchDataset(images, labels, batchSize: 7, shuffle: true)

        // For each batch, verify that image i corresponds to label i
        // Our synthetic data has image i with all pixels = Float(i) and label = i % 10
        for batch in batches {
            for i in 0..<batch.images.shape[0] {
                // Get the first pixel value (all pixels in synthetic image are the same)
                let firstPixel = batch.images[i, 0].item(Float.self)
                let imageIndex = Int(firstPixel)

                let label = batch.labels[i].item(Int32.self)
                let expectedLabel = Int32(imageIndex % 10)

                XCTAssertEqual(label, expectedLabel,
                              "Image-label correspondence should be maintained")
            }
        }
    }

    // =============================================================================
    // MARK: - Realistic Scenario Tests
    // =============================================================================

    func testTypicalTrainingBatchSize() throws {
        // Test with typical MNIST training parameters
        // Simulate mini-batch training with batch size 32
        let (images, labels) = createSyntheticData(numSamples: 128)

        let batches = batchDataset(images, labels, batchSize: 32, shuffle: true)

        // Should create exactly 4 batches (128 / 32 = 4)
        XCTAssertEqual(batches.count, 4, "Should create 4 batches of 32")

        // All batches should be same size
        for i in 0..<batches.count {
            XCTAssertEqual(batches[i].images.shape[0], 32,
                          "Batch \(i) should have 32 samples")
            XCTAssertEqual(batches[i].labels.shape[0], 32,
                          "Batch \(i) should have 32 labels")
        }

        // Verify total samples
        let totalSamples = batches.reduce(0) { $0 + $1.images.shape[0] }
        XCTAssertEqual(totalSamples, 128, "Total samples should be 128")
    }

    func testMultipleEpochShuffling() throws {
        // Verify that calling batchDataset multiple times with shuffle=true
        // produces different orderings (simulating multiple epochs)
        let (images, labels) = createSyntheticData(numSamples: 50)

        // Create batches for "epoch 1"
        let epoch1Batches = batchDataset(images, labels, batchSize: 10, shuffle: true)
        let (_, epoch1Labels) = flattenBatches(epoch1Batches)

        // Create batches for "epoch 2"
        let epoch2Batches = batchDataset(images, labels, batchSize: 10, shuffle: true)
        let (_, epoch2Labels) = flattenBatches(epoch2Batches)

        // Verify that the orderings are (very likely) different
        var orderingsMatch = true
        for i in 0..<50 {
            let label1 = epoch1Labels[i].item(Int32.self)
            let label2 = epoch2Labels[i].item(Int32.self)
            if label1 != label2 {
                orderingsMatch = false
                break
            }
        }

        // With 50 samples and shuffling, it's astronomically unlikely to get the same order twice
        XCTAssertFalse(orderingsMatch,
                      "Multiple shuffle calls should produce different orderings")
    }
}
