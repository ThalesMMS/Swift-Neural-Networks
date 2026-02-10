// ============================================================================
// DataLoading.swift - MNIST Data Loading for Shared Module
// ============================================================================
//
// This module provides simple functions to load the MNIST dataset from IDX
// files. These functions are shared across all implementations.
//
// MNIST DATASET FORMAT:
// The MNIST files use the IDX format with big-endian headers:
//
//   Images (train-images.idx3-ubyte, t10k-images.idx3-ubyte):
//     - 4 bytes: magic number (0x00000803)
//     - 4 bytes: number of images
//     - 4 bytes: number of rows (28)
//     - 4 bytes: number of columns (28)
//     - N * 28 * 28 bytes: pixel values (0-255)
//
//   Labels (train-labels.idx1-ubyte, t10k-labels.idx1-ubyte):
//     - 4 bytes: magic number (0x00000801)
//     - 4 bytes: number of labels
//     - N bytes: label values (0-9)
//
// USAGE:
//   let images = readMnistImages(path: "./data/train-images.idx3-ubyte", count: 60000)
//   let labels = readMnistLabels(path: "./data/train-labels.idx1-ubyte", count: 60000)
//
// ============================================================================

import Foundation

// =============================================================================
// MARK: - Image Loading
// =============================================================================

/// Loads MNIST images from an IDX file
///
/// Reads the IDX format binary file and converts pixel values from [0, 255]
/// to normalized Float values in [0.0, 1.0] for neural network training.
///
/// - Parameters:
///   - path: Path to the IDX images file
///   - count: Maximum number of images to load
/// - Returns: Flattened array of normalized pixel values (count * 784 floats)
///
/// ## Example
/// ```swift
/// let images = readMnistImages(path: "./data/train-images.idx3-ubyte", count: 60000)
/// // Returns 60000 * 784 = 47,040,000 float values
/// ```
public func readMnistImages(path: String, count: Int) -> [Float] {
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url) else {
        ColoredPrint.error("❌ Error: Could not open MNIST image file at '\(path)'")
        print()
        print("Expected MNIST files in the data directory:")
        print("  - train-images.idx3-ubyte  (training images)")
        print("  - train-labels.idx1-ubyte  (training labels)")
        print("  - t10k-images.idx3-ubyte   (test images)")
        print("  - t10k-labels.idx1-ubyte   (test labels)")
        print()
        print("Download the MNIST dataset from:")
        print("  http://yann.lecun.com/exdb/mnist/")
        print()
        print("After downloading, extract the .gz files to your data directory.")
        exit(1)
    }

    return data.withUnsafeBytes { rawBuf in
        guard let base = rawBuf.bindMemory(to: UInt8.self).baseAddress else {
            return []
        }
        var offset = 0

        // Helper to read big-endian 32-bit unsigned integers
        // MNIST IDX uses big-endian integers.
        func readU32BE() -> UInt32 {
            let b0 = UInt32(base[offset]) << 24
            let b1 = UInt32(base[offset + 1]) << 16
            let b2 = UInt32(base[offset + 2]) << 8
            let b3 = UInt32(base[offset + 3])
            offset += 4
            return b0 | b1 | b2 | b3
        }

        // Read header: magic number, count, rows, cols
        _ = readU32BE()
        let total = Int(readU32BE())
        let rows = Int(readU32BE())
        let cols = Int(readU32BE())
        let imageSize = rows * cols
        let actualCount = min(count, total)

        // Read and normalize pixel data
        var images = [Float](repeating: 0.0, count: actualCount * imageSize)
        for i in 0..<actualCount {
            let baseIndex = i * imageSize
            for j in 0..<imageSize {
                images[baseIndex + j] = Float(base[offset]) / 255.0
                offset += 1
            }
        }
        return images
    }
}

// =============================================================================
// MARK: - Label Loading
// =============================================================================

/// Loads MNIST labels from an IDX file
///
/// Reads the IDX format binary file and extracts label values (0-9).
///
/// - Parameters:
///   - path: Path to the IDX labels file
///   - count: Maximum number of labels to load
/// - Returns: Array of label values (0-9)
///
/// ## Example
/// ```swift
/// let labels = readMnistLabels(path: "./data/train-labels.idx1-ubyte", count: 60000)
/// // Returns 60,000 UInt8 values, each in range 0-9
/// ```
public func readMnistLabels(path: String, count: Int) -> [UInt8] {
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url) else {
        ColoredPrint.error("❌ Error: Could not open MNIST label file at '\(path)'")
        print()
        print("Expected MNIST files in the data directory:")
        print("  - train-images.idx3-ubyte  (training images)")
        print("  - train-labels.idx1-ubyte  (training labels)")
        print("  - t10k-images.idx3-ubyte   (test images)")
        print("  - t10k-labels.idx1-ubyte   (test labels)")
        print()
        print("Download the MNIST dataset from:")
        print("  http://yann.lecun.com/exdb/mnist/")
        print()
        print("After downloading, extract the .gz files to your data directory.")
        exit(1)
    }

    return data.withUnsafeBytes { rawBuf in
        guard let base = rawBuf.bindMemory(to: UInt8.self).baseAddress else {
            return []
        }
        var offset = 0

        // Helper to read big-endian 32-bit unsigned integers
        func readU32BE() -> UInt32 {
            let b0 = UInt32(base[offset]) << 24
            let b1 = UInt32(base[offset + 1]) << 16
            let b2 = UInt32(base[offset + 2]) << 8
            let b3 = UInt32(base[offset + 3])
            offset += 4
            return b0 | b1 | b2 | b3
        }

        // Read header: magic number, count
        _ = readU32BE()
        let total = Int(readU32BE())
        let actualCount = min(count, total)

        // Read label data
        var labels = [UInt8](repeating: 0, count: actualCount)
        for i in 0..<actualCount {
            labels[i] = base[offset]
            offset += 1
        }
        return labels
    }
}
