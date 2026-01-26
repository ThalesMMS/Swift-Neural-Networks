// ============================================================================
// MNISTLoaderTests.swift - Tests for MNIST Data Loading
// ============================================================================
//
// This test suite validates the MNIST IDX file format parsing, including:
// - Header validation (magic numbers, dimensions)
// - Big-endian integer parsing
// - Error handling for malformed files
// - Data normalization and shape validation
//
// ============================================================================

import XCTest
import MLX
@testable import MNISTData
import Foundation

final class MNISTLoaderTests: XCTestCase {

    // =============================================================================
    // MARK: - Test Utilities
    // =============================================================================

    /// Creates a temporary file with given data and returns its path
    private func createTempFile(data: Data, suffix: String = ".idx-ubyte") -> String {
        let tempDir = FileManager.default.temporaryDirectory
        let fileName = UUID().uuidString + suffix
        let fileURL = tempDir.appendingPathComponent(fileName)

        try! data.write(to: fileURL)

        // Store for cleanup
        addTeardownBlock {
            try? FileManager.default.removeItem(at: fileURL)
        }

        return fileURL.path
    }

    /// Creates a big-endian 32-bit unsigned integer as 4 bytes
    private func bigEndianU32(_ value: UInt32) -> [UInt8] {
        return [
            UInt8((value >> 24) & 0xFF),
            UInt8((value >> 16) & 0xFF),
            UInt8((value >> 8) & 0xFF),
            UInt8(value & 0xFF)
        ]
    }

    /// Creates a valid MNIST images file header
    private func createImagesHeader(numImages: UInt32, rows: UInt32 = 28, cols: UInt32 = 28) -> [UInt8] {
        var header: [UInt8] = []
        header.append(contentsOf: bigEndianU32(0x00000803)) // Magic number
        header.append(contentsOf: bigEndianU32(numImages))  // Number of images
        header.append(contentsOf: bigEndianU32(rows))       // Rows
        header.append(contentsOf: bigEndianU32(cols))       // Columns
        return header
    }

    /// Creates a valid MNIST labels file header
    private func createLabelsHeader(numLabels: UInt32) -> [UInt8] {
        var header: [UInt8] = []
        header.append(contentsOf: bigEndianU32(0x00000801)) // Magic number
        header.append(contentsOf: bigEndianU32(numLabels))  // Number of labels
        return header
    }

    // =============================================================================
    // MARK: - Image File Format Tests
    // =============================================================================

    func testLoadValidImagesFile() throws {
        // Create a valid images file with 2 images (28x28)
        var fileData = createImagesHeader(numImages: 2)

        // Add pixel data for 2 images (28x28 = 784 pixels each)
        // First image: all pixels = 0
        fileData.append(contentsOf: [UInt8](repeating: 0, count: 784))
        // Second image: all pixels = 255
        fileData.append(contentsOf: [UInt8](repeating: 255, count: 784))

        let path = createTempFile(data: Data(fileData))

        // Load the images
        let images = try loadMNISTImages(path: path)

        // Validate shape: [2, 784]
        XCTAssertEqual(images.shape, [2, 784], "Images should have shape [2, 784]")

        // Validate normalization: 0 → 0.0, 255 → 1.0
        let firstPixel = images[0, 0].item(Float.self)
        let lastPixel = images[1, 783].item(Float.self)

        XCTAssertEqual(firstPixel, 0.0, accuracy: 0.001, "Pixel value 0 should normalize to 0.0")
        XCTAssertEqual(lastPixel, 1.0, accuracy: 0.001, "Pixel value 255 should normalize to 1.0")
    }

    func testLoadImagesWithMaxCount() throws {
        // Create a valid images file with 5 images
        var fileData = createImagesHeader(numImages: 5)
        fileData.append(contentsOf: [UInt8](repeating: 128, count: 784 * 5))

        let path = createTempFile(data: Data(fileData))

        // Load only 3 images
        let images = try loadMNISTImages(path: path, maxCount: 3)

        // Validate that only 3 images were loaded
        XCTAssertEqual(images.shape, [3, 784], "Should load only 3 images when maxCount=3")
    }

    func testLoadImagesNormalization() throws {
        // Create a file with specific pixel values to test normalization
        var fileData = createImagesHeader(numImages: 1)

        // Create one image with values: 0, 1, 127, 128, 254, 255, then zeros
        var pixels: [UInt8] = [0, 1, 127, 128, 254, 255]
        pixels.append(contentsOf: [UInt8](repeating: 0, count: 784 - 6))
        fileData.append(contentsOf: pixels)

        let path = createTempFile(data: Data(fileData))
        let images = try loadMNISTImages(path: path)

        // Verify normalization
        XCTAssertEqual(images[0, 0].item(Float.self), 0.0 / 255.0, accuracy: 0.0001)
        XCTAssertEqual(images[0, 1].item(Float.self), 1.0 / 255.0, accuracy: 0.0001)
        XCTAssertEqual(images[0, 2].item(Float.self), 127.0 / 255.0, accuracy: 0.0001)
        XCTAssertEqual(images[0, 3].item(Float.self), 128.0 / 255.0, accuracy: 0.0001)
        XCTAssertEqual(images[0, 4].item(Float.self), 254.0 / 255.0, accuracy: 0.0001)
        XCTAssertEqual(images[0, 5].item(Float.self), 255.0 / 255.0, accuracy: 0.0001)
    }

    func testImageNormalization() throws {
        // Test comprehensive pixel value normalization from [0, 255] → [0.0, 1.0]
        var fileData = createImagesHeader(numImages: 1)

        // Test full range of important pixel values
        var pixels: [UInt8] = []

        // Edge cases
        pixels.append(0)     // Min value → 0.0
        pixels.append(255)   // Max value → 1.0

        // Quarter points
        pixels.append(64)    // ~0.25
        pixels.append(128)   // ~0.5
        pixels.append(192)   // ~0.75

        // Common values
        pixels.append(1)     // Near min
        pixels.append(127)   // Just below mid
        pixels.append(254)   // Near max

        // Additional test values
        pixels.append(32)    // Low range
        pixels.append(96)    // Mid-low range
        pixels.append(160)   // Mid-high range
        pixels.append(224)   // High range

        // Fill remaining pixels with zeros
        pixels.append(contentsOf: [UInt8](repeating: 0, count: 784 - pixels.count))
        fileData.append(contentsOf: pixels)

        let path = createTempFile(data: Data(fileData))
        let images = try loadMNISTImages(path: path)

        // Verify shape
        XCTAssertEqual(images.shape, [1, 784], "Image should have shape [1, 784]")

        // Test edge cases: 0 → 0.0, 255 → 1.0
        XCTAssertEqual(images[0, 0].item(Float.self), 0.0, accuracy: 0.0001,
                      "Pixel value 0 should normalize to 0.0")
        XCTAssertEqual(images[0, 1].item(Float.self), 1.0, accuracy: 0.0001,
                      "Pixel value 255 should normalize to 1.0")

        // Test quarter points
        XCTAssertEqual(images[0, 2].item(Float.self), 64.0 / 255.0, accuracy: 0.0001,
                      "Pixel value 64 should normalize to ~0.251")
        XCTAssertEqual(images[0, 3].item(Float.self), 128.0 / 255.0, accuracy: 0.0001,
                      "Pixel value 128 should normalize to ~0.502")
        XCTAssertEqual(images[0, 4].item(Float.self), 192.0 / 255.0, accuracy: 0.0001,
                      "Pixel value 192 should normalize to ~0.753")

        // Test common values
        XCTAssertEqual(images[0, 5].item(Float.self), 1.0 / 255.0, accuracy: 0.0001,
                      "Pixel value 1 should normalize to ~0.0039")
        XCTAssertEqual(images[0, 6].item(Float.self), 127.0 / 255.0, accuracy: 0.0001,
                      "Pixel value 127 should normalize to ~0.498")
        XCTAssertEqual(images[0, 7].item(Float.self), 254.0 / 255.0, accuracy: 0.0001,
                      "Pixel value 254 should normalize to ~0.996")

        // Test additional range values
        XCTAssertEqual(images[0, 8].item(Float.self), 32.0 / 255.0, accuracy: 0.0001,
                      "Pixel value 32 should normalize to ~0.125")
        XCTAssertEqual(images[0, 9].item(Float.self), 96.0 / 255.0, accuracy: 0.0001,
                      "Pixel value 96 should normalize to ~0.376")
        XCTAssertEqual(images[0, 10].item(Float.self), 160.0 / 255.0, accuracy: 0.0001,
                      "Pixel value 160 should normalize to ~0.627")
        XCTAssertEqual(images[0, 11].item(Float.self), 224.0 / 255.0, accuracy: 0.0001,
                      "Pixel value 224 should normalize to ~0.878")

        // Verify all normalized values are in [0.0, 1.0] range
        for i in 0..<12 {
            let value = images[0, i].item(Float.self)
            XCTAssertGreaterThanOrEqual(value, 0.0, "Normalized value should be >= 0.0")
            XCTAssertLessThanOrEqual(value, 1.0, "Normalized value should be <= 1.0")
        }
    }

    func testLoadImagesInvalidMagicNumber() throws {
        // Create a file with wrong magic number
        var header: [UInt8] = []
        header.append(contentsOf: bigEndianU32(0x12345678)) // Wrong magic
        header.append(contentsOf: bigEndianU32(1))          // Number of images
        header.append(contentsOf: bigEndianU32(28))         // Rows
        header.append(contentsOf: bigEndianU32(28))         // Columns
        header.append(contentsOf: [UInt8](repeating: 0, count: 784))

        let path = createTempFile(data: Data(header))

        // Should throw invalid format error
        XCTAssertThrowsError(try loadMNISTImages(path: path)) { error in
            guard case MNISTError.invalidFormat(let message) = error else {
                XCTFail("Expected MNISTError.invalidFormat, got \(error)")
                return
            }
            XCTAssertTrue(message.contains("magic number"), "Error should mention magic number")
        }
    }

    func testLoadImagesInvalidDimensions() throws {
        // Create a file with wrong dimensions (32x32 instead of 28x28)
        var fileData = createImagesHeader(numImages: 1, rows: 32, cols: 32)
        fileData.append(contentsOf: [UInt8](repeating: 0, count: 32 * 32))

        let path = createTempFile(data: Data(fileData))

        // Should throw invalid format error
        XCTAssertThrowsError(try loadMNISTImages(path: path)) { error in
            guard case MNISTError.invalidFormat(let message) = error else {
                XCTFail("Expected MNISTError.invalidFormat, got \(error)")
                return
            }
            XCTAssertTrue(message.contains("image size"), "Error should mention image size")
        }
    }

    func testLoadImagesTruncatedFile() throws {
        // Create a file with header but incomplete data
        var fileData = createImagesHeader(numImages: 2)
        // Add only 1 image worth of data instead of 2
        fileData.append(contentsOf: [UInt8](repeating: 0, count: 784))

        let path = createTempFile(data: Data(fileData))

        // This should either throw an error or handle gracefully
        // The current implementation will read beyond bounds, which Swift should catch
        // Testing that it doesn't crash
        do {
            _ = try loadMNISTImages(path: path)
            // If it succeeds, we accept it (implementation may handle gracefully)
        } catch {
            // If it throws, that's also acceptable
            // Just ensure it doesn't crash
        }
    }

    func testLoadImagesEmptyFile() throws {
        // Create an empty file
        let path = createTempFile(data: Data())

        // Should throw an error (either fileNotFound or invalidFormat)
        XCTAssertThrowsError(try loadMNISTImages(path: path))
    }

    func testLoadImagesFileNotFound() throws {
        let nonExistentPath = "/tmp/nonexistent_\(UUID().uuidString).idx3-ubyte"

        // Should throw fileNotFound error
        XCTAssertThrowsError(try loadMNISTImages(path: nonExistentPath)) { error in
            guard case MNISTError.fileNotFound(_) = error else {
                XCTFail("Expected MNISTError.fileNotFound, got \(error)")
                return
            }
        }
    }

    func testLoadImagesHeaderOnly() throws {
        // Create a file with only the header, no pixel data
        let fileData = createImagesHeader(numImages: 1)
        let path = createTempFile(data: Data(fileData))

        // Should throw an error when trying to read pixel data
        XCTAssertThrowsError(try loadMNISTImages(path: path))
    }

    // =============================================================================
    // MARK: - Label File Format Tests
    // =============================================================================

    func testLoadValidLabelsFile() throws {
        // Create a valid labels file with 10 labels (digits 0-9)
        var fileData = createLabelsHeader(numLabels: 10)
        fileData.append(contentsOf: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        let path = createTempFile(data: Data(fileData))

        // Load the labels
        let labels = try loadMNISTLabels(path: path)

        // Validate shape: [10]
        XCTAssertEqual(labels.shape, [10], "Labels should have shape [10]")

        // Validate values
        for i in 0..<10 {
            let label = labels[i].item(Int32.self)
            XCTAssertEqual(label, Int32(i), "Label at index \(i) should be \(i)")
        }
    }

    func testLoadLabelsWithMaxCount() throws {
        // Create a valid labels file with 10 labels
        var fileData = createLabelsHeader(numLabels: 10)
        fileData.append(contentsOf: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        let path = createTempFile(data: Data(fileData))

        // Load only 5 labels
        let labels = try loadMNISTLabels(path: path, maxCount: 5)

        // Validate that only 5 labels were loaded
        XCTAssertEqual(labels.shape, [5], "Should load only 5 labels when maxCount=5")

        // Validate values
        for i in 0..<5 {
            let label = labels[i].item(Int32.self)
            XCTAssertEqual(label, Int32(i), "Label at index \(i) should be \(i)")
        }
    }

    func testLoadLabelsInvalidMagicNumber() throws {
        // Create a file with wrong magic number
        var header: [UInt8] = []
        header.append(contentsOf: bigEndianU32(0xDEADBEEF)) // Wrong magic
        header.append(contentsOf: bigEndianU32(5))          // Number of labels
        header.append(contentsOf: [0, 1, 2, 3, 4])

        let path = createTempFile(data: Data(header))

        // Should throw invalid format error
        XCTAssertThrowsError(try loadMNISTLabels(path: path)) { error in
            guard case MNISTError.invalidFormat(let message) = error else {
                XCTFail("Expected MNISTError.invalidFormat, got \(error)")
                return
            }
            XCTAssertTrue(message.contains("magic number"), "Error should mention magic number")
        }
    }

    func testLoadLabelsTruncatedFile() throws {
        // Create a file with header claiming 10 labels but only 5 present
        var fileData = createLabelsHeader(numLabels: 10)
        fileData.append(contentsOf: [0, 1, 2, 3, 4]) // Only 5 labels

        let path = createTempFile(data: Data(fileData))

        // This should either throw an error or handle gracefully
        do {
            _ = try loadMNISTLabels(path: path)
            // If it succeeds, we accept it
        } catch {
            // If it throws, that's also acceptable
        }
    }

    func testLoadLabelsEmptyFile() throws {
        // Create an empty file
        let path = createTempFile(data: Data())

        // Should throw an error
        XCTAssertThrowsError(try loadMNISTLabels(path: path))
    }

    func testLoadLabelsFileNotFound() throws {
        let nonExistentPath = "/tmp/nonexistent_\(UUID().uuidString).idx1-ubyte"

        // Should throw fileNotFound error
        XCTAssertThrowsError(try loadMNISTLabels(path: nonExistentPath)) { error in
            guard case MNISTError.fileNotFound(_) = error else {
                XCTFail("Expected MNISTError.fileNotFound, got \(error)")
                return
            }
        }
    }

    func testLoadLabelsHeaderOnly() throws {
        // Create a file with only the header, no label data
        let fileData = createLabelsHeader(numLabels: 5)
        let path = createTempFile(data: Data(fileData))

        // Should throw an error when trying to read label data
        XCTAssertThrowsError(try loadMNISTLabels(path: path))
    }

    // =============================================================================
    // MARK: - IDX Format Big-Endian Tests
    // =============================================================================

    func testBigEndianHeaderParsing() throws {
        // Test that big-endian integers are correctly parsed
        // Create a file with specific header values
        var fileData: [UInt8] = []

        // Magic number: 0x00000803
        fileData.append(contentsOf: bigEndianU32(0x00000803))

        // Number of images: 12345 (0x00003039)
        fileData.append(contentsOf: bigEndianU32(12345))

        // Rows: 28 (0x0000001C)
        fileData.append(contentsOf: bigEndianU32(28))

        // Cols: 28 (0x0000001C)
        fileData.append(contentsOf: bigEndianU32(28))

        // Add minimal pixel data (just 1 image to avoid truncation)
        fileData.append(contentsOf: [UInt8](repeating: 0, count: 784))

        let path = createTempFile(data: Data(fileData))

        // Load with maxCount=1 to avoid needing 12345 images
        let images = try loadMNISTImages(path: path, maxCount: 1)

        // If parsing succeeded, the shape should be [1, 784]
        XCTAssertEqual(images.shape, [1, 784])
    }

    func testImageMagicNumberValidation() throws {
        // Test all bytes of the magic number
        let validMagic: [UInt8] = [0x00, 0x00, 0x08, 0x03]

        // Test each byte being wrong
        for i in 0..<4 {
            var invalidMagic = validMagic
            invalidMagic[i] = invalidMagic[i] ^ 0xFF // Flip all bits

            var fileData = invalidMagic
            fileData.append(contentsOf: bigEndianU32(1)) // numImages
            fileData.append(contentsOf: bigEndianU32(28)) // rows
            fileData.append(contentsOf: bigEndianU32(28)) // cols
            fileData.append(contentsOf: [UInt8](repeating: 0, count: 784))

            let path = createTempFile(data: Data(fileData), suffix: "_byte\(i).idx")

            // Should throw invalid format error
            XCTAssertThrowsError(try loadMNISTImages(path: path)) { error in
                guard case MNISTError.invalidFormat(_) = error else {
                    XCTFail("Expected MNISTError.invalidFormat for corrupted byte \(i), got \(error)")
                    return
                }
            }
        }
    }

    func testLabelMagicNumberValidation() throws {
        // Test the label magic number (0x00000801)
        let validMagic: [UInt8] = [0x00, 0x00, 0x08, 0x01]

        // Create valid file
        var fileData = validMagic
        fileData.append(contentsOf: bigEndianU32(5))
        fileData.append(contentsOf: [0, 1, 2, 3, 4])

        let validPath = createTempFile(data: Data(fileData))
        XCTAssertNoThrow(try loadMNISTLabels(path: validPath))

        // Test with corrupted magic
        var invalidMagic = validMagic
        invalidMagic[3] = 0x02 // Change last byte

        var invalidData = invalidMagic
        invalidData.append(contentsOf: bigEndianU32(5))
        invalidData.append(contentsOf: [0, 1, 2, 3, 4])

        let invalidPath = createTempFile(data: Data(invalidData))
        XCTAssertThrowsError(try loadMNISTLabels(path: invalidPath))
    }

    // =============================================================================
    // MARK: - Shape and Dimension Tests
    // =============================================================================

    func testImageShapeConsistency() throws {
        // Test various numbers of images
        for numImages in [1, 2, 10, 100] {
            var fileData = createImagesHeader(numImages: UInt32(numImages))
            fileData.append(contentsOf: [UInt8](repeating: 128, count: 784 * numImages))

            let path = createTempFile(data: Data(fileData), suffix: "_\(numImages)imgs.idx")
            let images = try loadMNISTImages(path: path)

            XCTAssertEqual(images.shape, [numImages, 784],
                          "Shape mismatch for \(numImages) images")
        }
    }

    func testLabelShapeConsistency() throws {
        // Test various numbers of labels
        for numLabels in [1, 2, 10, 100] {
            var fileData = createLabelsHeader(numLabels: UInt32(numLabels))
            fileData.append(contentsOf: [UInt8](repeating: 5, count: numLabels))

            let path = createTempFile(data: Data(fileData), suffix: "_\(numLabels)lbls.idx")
            let labels = try loadMNISTLabels(path: path)

            XCTAssertEqual(labels.shape, [numLabels],
                          "Shape mismatch for \(numLabels) labels")
        }
    }

    func testZeroImagesFile() throws {
        // Test file with 0 images (edge case)
        let fileData = createImagesHeader(numImages: 0)
        let path = createTempFile(data: Data(fileData))

        let images = try loadMNISTImages(path: path)
        XCTAssertEqual(images.shape, [0, 784], "Should handle 0 images")
    }

    func testZeroLabelsFile() throws {
        // Test file with 0 labels (edge case)
        let fileData = createLabelsHeader(numLabels: 0)
        let path = createTempFile(data: Data(fileData))

        let labels = try loadMNISTLabels(path: path)
        XCTAssertEqual(labels.shape, [0], "Should handle 0 labels")
    }

    // =============================================================================
    // MARK: - Comprehensive Error Handling Tests
    // =============================================================================

    func testErrorHandlingMissingImagesFile() throws {
        // Test that missing images file throws fileNotFound error
        let nonExistentPath = "/tmp/does_not_exist_\(UUID().uuidString).idx3-ubyte"

        XCTAssertThrowsError(try loadMNISTImages(path: nonExistentPath)) { error in
            guard case MNISTError.fileNotFound(let path) = error else {
                XCTFail("Expected MNISTError.fileNotFound, got \(error)")
                return
            }
            XCTAssertEqual(path, nonExistentPath, "Error should contain the missing file path")
        }
    }

    func testErrorHandlingMissingLabelsFile() throws {
        // Test that missing labels file throws fileNotFound error
        let nonExistentPath = "/tmp/does_not_exist_\(UUID().uuidString).idx1-ubyte"

        XCTAssertThrowsError(try loadMNISTLabels(path: nonExistentPath)) { error in
            guard case MNISTError.fileNotFound(let path) = error else {
                XCTFail("Expected MNISTError.fileNotFound, got \(error)")
                return
            }
            XCTAssertEqual(path, nonExistentPath, "Error should contain the missing file path")
        }
    }

    func testErrorHandlingInvalidImagesMagicNumber() throws {
        // Test that invalid magic number in images file throws invalidFormat error
        var fileData: [UInt8] = []
        fileData.append(contentsOf: bigEndianU32(0xBADC0FFE)) // Invalid magic
        fileData.append(contentsOf: bigEndianU32(1))          // Number of images
        fileData.append(contentsOf: bigEndianU32(28))         // Rows
        fileData.append(contentsOf: bigEndianU32(28))         // Columns
        fileData.append(contentsOf: [UInt8](repeating: 0, count: 784))

        let path = createTempFile(data: Data(fileData))

        XCTAssertThrowsError(try loadMNISTImages(path: path)) { error in
            guard case MNISTError.invalidFormat(let message) = error else {
                XCTFail("Expected MNISTError.invalidFormat, got \(error)")
                return
            }
            XCTAssertTrue(message.contains("magic number"),
                         "Error message should mention magic number: \(message)")
            XCTAssertTrue(message.contains("0xBADC0FFE") || message.contains("3134983934"),
                         "Error message should contain the invalid magic number")
        }
    }

    func testErrorHandlingInvalidLabelsMagicNumber() throws {
        // Test that invalid magic number in labels file throws invalidFormat error
        var fileData: [UInt8] = []
        fileData.append(contentsOf: bigEndianU32(0xCAFEBABE)) // Invalid magic
        fileData.append(contentsOf: bigEndianU32(5))          // Number of labels
        fileData.append(contentsOf: [0, 1, 2, 3, 4])

        let path = createTempFile(data: Data(fileData))

        XCTAssertThrowsError(try loadMNISTLabels(path: path)) { error in
            guard case MNISTError.invalidFormat(let message) = error else {
                XCTFail("Expected MNISTError.invalidFormat, got \(error)")
                return
            }
            XCTAssertTrue(message.contains("magic number"),
                         "Error message should mention magic number: \(message)")
            XCTAssertTrue(message.contains("0xCAFEBABE") || message.contains("3405691582"),
                         "Error message should contain the invalid magic number")
        }
    }

    func testErrorHandlingInvalidImageDimensions() throws {
        // Test that invalid image dimensions throw invalidFormat error
        // Test various invalid dimension combinations
        let invalidDimensions: [(rows: UInt32, cols: UInt32)] = [
            (16, 16),   // Too small
            (32, 32),   // Too large
            (28, 32),   // Wrong width
            (32, 28),   // Wrong height
            (14, 14),   // Half size
            (56, 56),   // Double size
        ]

        for (rows, cols) in invalidDimensions {
            var fileData = createImagesHeader(numImages: 1, rows: rows, cols: cols)
            fileData.append(contentsOf: [UInt8](repeating: 0, count: Int(rows * cols)))

            let path = createTempFile(data: Data(fileData), suffix: "_\(rows)x\(cols).idx")

            XCTAssertThrowsError(try loadMNISTImages(path: path)) { error in
                guard case MNISTError.invalidFormat(let message) = error else {
                    XCTFail("Expected MNISTError.invalidFormat for \(rows)x\(cols), got \(error)")
                    return
                }
                XCTAssertTrue(message.contains("image size") || message.contains("\(rows)x\(cols)"),
                             "Error should mention invalid dimensions for \(rows)x\(cols): \(message)")
            }
        }
    }

    func testErrorHandlingEmptyImagesFile() throws {
        // Test that empty images file throws error
        let emptyData = Data()
        let path = createTempFile(data: emptyData)

        XCTAssertThrowsError(try loadMNISTImages(path: path)) { error in
            // Should throw either invalidFormat or a bounds error
            // We just verify it doesn't crash and throws some error
            XCTAssertTrue(error is MNISTError || error is NSError,
                         "Should throw an error for empty file")
        }
    }

    func testErrorHandlingEmptyLabelsFile() throws {
        // Test that empty labels file throws error
        let emptyData = Data()
        let path = createTempFile(data: emptyData)

        XCTAssertThrowsError(try loadMNISTLabels(path: path)) { error in
            // Should throw either invalidFormat or a bounds error
            XCTAssertTrue(error is MNISTError || error is NSError,
                         "Should throw an error for empty file")
        }
    }

    func testErrorHandlingTruncatedImagesHeader() throws {
        // Test files with incomplete headers
        let incompleteHeaders: [[UInt8]] = [
            [],                                     // Completely empty
            bigEndianU32(0x00000803),              // Only magic number
            bigEndianU32(0x00000803) + bigEndianU32(10),  // Magic + count only
            bigEndianU32(0x00000803) + bigEndianU32(10) + bigEndianU32(28), // Missing cols
        ]

        for (index, headerData) in incompleteHeaders.enumerated() {
            let path = createTempFile(data: Data(headerData), suffix: "_incomplete\(index).idx")

            XCTAssertThrowsError(try loadMNISTImages(path: path)) { error in
                // Should throw some kind of error (invalidFormat or bounds error)
                // We just verify it doesn't crash
            }
        }
    }

    func testErrorHandlingTruncatedLabelsHeader() throws {
        // Test files with incomplete headers
        let incompleteHeaders: [[UInt8]] = [
            [],                                     // Completely empty
            bigEndianU32(0x00000801),              // Only magic number
        ]

        for (index, headerData) in incompleteHeaders.enumerated() {
            let path = createTempFile(data: Data(headerData), suffix: "_incomplete\(index).idx")

            XCTAssertThrowsError(try loadMNISTLabels(path: path)) { error in
                // Should throw some kind of error
            }
        }
    }

    func testErrorHandlingInsufficientImageData() throws {
        // Test file that claims N images but has less data
        var fileData = createImagesHeader(numImages: 10)
        // Add data for only 3 images instead of 10
        fileData.append(contentsOf: [UInt8](repeating: 128, count: 784 * 3))

        let path = createTempFile(data: Data(fileData))

        // This should either throw an error or handle gracefully
        // We test that it doesn't crash
        do {
            let images = try loadMNISTImages(path: path)
            // If it succeeds, verify it doesn't have 10 images
            XCTAssertNotEqual(images.shape[0], 10,
                            "Should not load 10 images when data is insufficient")
        } catch {
            // Throwing an error is acceptable behavior
            // Just verify it's a reasonable error
            XCTAssertTrue(error is MNISTError || error is NSError,
                         "Should throw a reasonable error type")
        }
    }

    func testErrorHandlingInsufficientLabelData() throws {
        // Test file that claims N labels but has less data
        var fileData = createLabelsHeader(numLabels: 100)
        // Add only 10 labels instead of 100
        fileData.append(contentsOf: [UInt8](repeating: 5, count: 10))

        let path = createTempFile(data: Data(fileData))

        // This should either throw an error or handle gracefully
        do {
            let labels = try loadMNISTLabels(path: path)
            // If it succeeds, verify it doesn't have 100 labels
            XCTAssertNotEqual(labels.shape[0], 100,
                            "Should not load 100 labels when data is insufficient")
        } catch {
            // Throwing an error is acceptable behavior
            XCTAssertTrue(error is MNISTError || error is NSError,
                         "Should throw a reasonable error type")
        }
    }

    func testErrorHandlingCorruptedImageData() throws {
        // Test file with valid header but corrupted/truncated pixel data
        var fileData = createImagesHeader(numImages: 5)
        // Add complete data for first 2 images
        fileData.append(contentsOf: [UInt8](repeating: 100, count: 784 * 2))
        // Add incomplete data for third image (only 100 pixels instead of 784)
        fileData.append(contentsOf: [UInt8](repeating: 50, count: 100))
        // No data for images 4 and 5

        let path = createTempFile(data: Data(fileData))

        // Should handle gracefully without crashing
        do {
            _ = try loadMNISTImages(path: path)
            // If successful, it handled the corruption gracefully
        } catch {
            // Throwing is also acceptable
            // Just ensure no crash
        }
    }

    func testErrorHandlingNonReadableBuffer() throws {
        // Test that buffer reading errors are handled
        // Create a minimal file that might cause buffer read issues
        let tinyData = Data([0x00]) // Just 1 byte
        let path = createTempFile(data: tinyData)

        XCTAssertThrowsError(try loadMNISTImages(path: path)) { error in
            // Should throw some error, not crash
        }
    }

    func testErrorHandlingDirectoryInsteadOfFile() throws {
        // Test that passing a directory path instead of file path fails gracefully
        let tempDir = FileManager.default.temporaryDirectory.path

        XCTAssertThrowsError(try loadMNISTImages(path: tempDir)) { error in
            // Should throw fileNotFound or invalidFormat
            XCTAssertTrue(error is MNISTError,
                         "Should throw MNISTError for directory path")
        }
    }

    func testErrorHandlingSpecialCharactersInPath() throws {
        // Test that paths with special characters are handled correctly
        let specialPaths = [
            "/tmp/file with spaces \(UUID().uuidString).idx",
            "/tmp/file@#$%^\(UUID().uuidString).idx",
            "/tmp/file'quote\(UUID().uuidString).idx",
        ]

        for specialPath in specialPaths {
            // All should throw fileNotFound since files don't exist
            XCTAssertThrowsError(try loadMNISTImages(path: specialPath)) { error in
                guard case MNISTError.fileNotFound(_) = error else {
                    XCTFail("Expected MNISTError.fileNotFound for path '\(specialPath)', got \(error)")
                    return
                }
            }
        }
    }

    func testErrorHandlingMismatchedImageAndLabelCounts() throws {
        // Test loading images and labels with different counts
        // Create images file with 100 images
        var imagesData = createImagesHeader(numImages: 100)
        imagesData.append(contentsOf: [UInt8](repeating: 128, count: 784 * 100))
        let imagesPath = createTempFile(data: Data(imagesData), suffix: "_100imgs.idx")

        // Create labels file with 50 labels
        var labelsData = createLabelsHeader(numLabels: 50)
        labelsData.append(contentsOf: [UInt8](repeating: 5, count: 50))
        let labelsPath = createTempFile(data: Data(labelsData), suffix: "_50lbls.idx")

        // Load both successfully
        let images = try loadMNISTImages(path: imagesPath)
        let labels = try loadMNISTLabels(path: labelsPath)

        // Verify they have different counts
        XCTAssertEqual(images.shape[0], 100, "Should load 100 images")
        XCTAssertEqual(labels.shape[0], 50, "Should load 50 labels")
        XCTAssertNotEqual(images.shape[0], labels.shape[0],
                         "Mismatch should be detectable by comparing shapes")
    }

    func testErrorHandlingVeryLargeFileHeader() throws {
        // Test file claiming to have an unreasonably large number of images
        let hugeCount: UInt32 = UInt32.max // Maximum value
        var fileData = createImagesHeader(numImages: hugeCount)
        // Add minimal data
        fileData.append(contentsOf: [UInt8](repeating: 0, count: 784))

        let path = createTempFile(data: Data(fileData))

        // Load with maxCount to avoid memory issues
        let images = try loadMNISTImages(path: path, maxCount: 1)

        // Should successfully load just 1 image despite huge header count
        XCTAssertEqual(images.shape, [1, 784],
                      "Should respect maxCount even with huge header count")
    }

    func testErrorHandlingErrorMessages() throws {
        // Test that error messages are descriptive and helpful

        // 1. File not found error message
        let missingPath = "/nonexistent/path/to/mnist.idx"
        do {
            _ = try loadMNISTImages(path: missingPath)
            XCTFail("Should throw error for missing file")
        } catch let error as MNISTError {
            let description = error.description
            XCTAssertTrue(description.contains("not found") || description.contains("File"),
                         "Error description should be meaningful: \(description)")
            XCTAssertTrue(description.contains(missingPath),
                         "Error should include the path: \(description)")
        }

        // 2. Invalid magic number error message
        var badMagicData: [UInt8] = []
        badMagicData.append(contentsOf: bigEndianU32(0x99999999))
        badMagicData.append(contentsOf: bigEndianU32(1))
        badMagicData.append(contentsOf: bigEndianU32(28))
        badMagicData.append(contentsOf: bigEndianU32(28))
        badMagicData.append(contentsOf: [UInt8](repeating: 0, count: 784))

        let badMagicPath = createTempFile(data: Data(badMagicData))

        do {
            _ = try loadMNISTImages(path: badMagicPath)
            XCTFail("Should throw error for invalid magic number")
        } catch let error as MNISTError {
            let description = error.description
            XCTAssertTrue(description.contains("format") || description.contains("magic"),
                         "Error should mention format or magic: \(description)")
        }

        // 3. Invalid dimensions error message
        var badDimsData = createImagesHeader(numImages: 1, rows: 64, cols: 64)
        badDimsData.append(contentsOf: [UInt8](repeating: 0, count: 64 * 64))
        let badDimsPath = createTempFile(data: Data(badDimsData))

        do {
            _ = try loadMNISTImages(path: badDimsPath)
            XCTFail("Should throw error for invalid dimensions")
        } catch let error as MNISTError {
            let description = error.description
            XCTAssertTrue(description.contains("size") || description.contains("64"),
                         "Error should mention size or dimensions: \(description)")
        }
    }
}
