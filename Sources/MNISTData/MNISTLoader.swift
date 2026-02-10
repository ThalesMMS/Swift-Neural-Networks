// ============================================================================
// MNISTLoader.swift - MNIST Data Loading Utilities for MLX Swift
// ============================================================================
//
// This module provides functions to load the MNIST dataset into MLXArray
// tensors for training and testing neural networks.
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
//   let (images, labels) = try loadMNIST(
//       imagesPath: "./data/train-images.idx3-ubyte",
//       labelsPath: "./data/train-labels.idx1-ubyte"
//   )
//
// ============================================================================

import Foundation
import MLX

// =============================================================================
// MARK: - Constants
// =============================================================================

/// MNIST image dimensions (28x28 grayscale images)
public let MNIST_IMAGE_HEIGHT = 28
public let MNIST_IMAGE_WIDTH = 28
public let MNIST_IMAGE_SIZE = MNIST_IMAGE_HEIGHT * MNIST_IMAGE_WIDTH  // 784

/// Number of classes (digits 0-9)
public let MNIST_NUM_CLASSES = 10

/// Dataset sizes
public let MNIST_TRAIN_SIZE = 60_000
public let MNIST_TEST_SIZE = 10_000

// =============================================================================
// MARK: - Errors
// =============================================================================

/// Errors that can occur when loading MNIST data
public enum MNISTError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case invalidFormat(String)
    case unexpectedSize(expected: Int, actual: Int)

    public var description: String {
        switch self {
        case .fileNotFound(let path):
            return """
            File not found: \(path)

            → ACTION: Download the MNIST dataset to this location.
                      You can download it from: http://yann.lecun.com/exdb/mnist/
                      Or ensure the --data path points to the correct directory.
            """
        case .invalidFormat(let message):
            return """
            Invalid MNIST format: \(message)

            → ACTION: The file may be corrupted or not a valid MNIST IDX file.
                      Try re-downloading the dataset from: http://yann.lecun.com/exdb/mnist/
                      Ensure you're using the original .gz files or properly extracted .idx3-ubyte files.
            """
        case .unexpectedSize(let expected, let actual):
            return """
            Unexpected size: expected \(expected), got \(actual)

            → ACTION: The file may be incomplete or corrupted.
                      Verify the file downloaded completely (check file size).
                      Expected sizes: train-images ~47MB, train-labels ~60KB,
                                     test-images ~7.8MB, test-labels ~10KB.
                      Re-download if necessary from: http://yann.lecun.com/exdb/mnist/
            """
        }
    }
}

// =============================================================================
// MARK: - IDX Format Helpers
// =============================================================================

/// Reads a big-endian 32-bit unsigned integer from raw bytes
/// 
/// The IDX format uses big-endian (network byte order) for all integers.
/// This function converts 4 bytes to a UInt32 value.
///
/// - Parameters:
///   - base: Pointer to the byte buffer
///   - offset: Current offset in the buffer (will be advanced by 4)
/// - Returns: The decoded UInt32 value
@inline(__always)
private func readBigEndianU32(from base: UnsafePointer<UInt8>, at offset: inout Int) -> UInt32 {
    // Big-endian: most significant byte first
    // Shift each byte to its position and combine with OR
    let b0 = UInt32(base[offset]) << 24     // Byte 0 → bits 24-31
    let b1 = UInt32(base[offset + 1]) << 16 // Byte 1 → bits 16-23
    let b2 = UInt32(base[offset + 2]) << 8  // Byte 2 → bits 8-15
    let b3 = UInt32(base[offset + 3])       // Byte 3 → bits 0-7
    offset += 4
    return b0 | b1 | b2 | b3
}

// =============================================================================
// MARK: - Image Loading
// =============================================================================

/// Loads MNIST images from an IDX file into an MLXArray
///
/// The images are normalized from [0, 255] to [0.0, 1.0] for neural network
/// training. The returned array has shape [N, 784] where N is the number of
/// images and 784 = 28 * 28 is the flattened image size.
///
/// For CNN models, you can reshape to [N, 1, 28, 28] (NCHW format).
///
/// - Parameters:
///   - path: Path to the IDX images file
///   - maxCount: Maximum number of images to load (nil = load all)
/// - Returns: MLXArray of shape [N, 784] with Float32 values in [0, 1]
/// - Throws: MNISTError if the file cannot be read or has invalid format
///
/// ## Example
/// ```swift
/// let images = try loadMNISTImages(path: "./data/train-images.idx3-ubyte")
/// print(images.shape) // [60000, 784]
///
/// // Reshape for CNN (add channel dimension)
/// let cnnImages = images.reshaped([60000, 1, 28, 28])
/// ```
public func loadMNISTImages(path: String, maxCount: Int? = nil) throws -> MLXArray {
    // -------------------------------------------------------------------------
    // Step 1: Read the raw file data
    // -------------------------------------------------------------------------
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url) else {
        throw MNISTError.fileNotFound(path)
    }
    
    // -------------------------------------------------------------------------
    // Step 2: Parse the IDX header and extract pixel data
    // -------------------------------------------------------------------------
    let images: [Float] = try data.withUnsafeBytes { rawBuf in
        guard let base = rawBuf.bindMemory(to: UInt8.self).baseAddress else {
            throw MNISTError.invalidFormat("Cannot read file buffer")
        }
        
        var offset = 0
        
        // Read and validate the magic number
        let magic = readBigEndianU32(from: base, at: &offset)
        guard magic == 0x00000803 else {
            throw MNISTError.invalidFormat("Invalid magic number: \(magic), expected 0x00000803")
        }
        
        // Read dimensions from header
        let totalImages = Int(readBigEndianU32(from: base, at: &offset))
        let rows = Int(readBigEndianU32(from: base, at: &offset))
        let cols = Int(readBigEndianU32(from: base, at: &offset))
        
        // Validate image dimensions
        guard rows == MNIST_IMAGE_HEIGHT && cols == MNIST_IMAGE_WIDTH else {
            throw MNISTError.invalidFormat("Unexpected image size: \(rows)x\(cols)")
        }
        
        // Determine how many images to load
        let count = min(maxCount ?? totalImages, totalImages)
        let imageSize = rows * cols
        
        // -------------------------------------------------------------------------
        // Step 3: Convert pixels to normalized Float32 values
        // -------------------------------------------------------------------------
        // We normalize from [0, 255] to [0.0, 1.0] for better training.
        // This is a standard preprocessing step for neural networks.
        var images = [Float](repeating: 0, count: count * imageSize)
        
        for i in 0..<count {
            let baseIndex = i * imageSize
            for j in 0..<imageSize {
                // Normalize: divide by 255.0 to get values in [0, 1]
                images[baseIndex + j] = Float(base[offset]) / 255.0
                offset += 1
            }
        }
        
        return images
    }
    
    // -------------------------------------------------------------------------
    // Step 4: Create MLXArray tensor
    // -------------------------------------------------------------------------
    // MLXArray is the fundamental data structure in MLX, similar to numpy.ndarray
    // or torch.Tensor. It supports GPU acceleration on Apple Silicon.
    let count = images.count / MNIST_IMAGE_SIZE
    return MLXArray(images, [count, MNIST_IMAGE_SIZE])
}

// =============================================================================
// MARK: - Label Loading
// =============================================================================

/// Loads MNIST labels from an IDX file into an MLXArray
///
/// The labels are integers from 0-9 representing the digit class.
/// The returned array has shape [N] where N is the number of labels.
///
/// - Parameters:
///   - path: Path to the IDX labels file
///   - maxCount: Maximum number of labels to load (nil = load all)
/// - Returns: MLXArray of shape [N] with Int32 values in [0, 9]
/// - Throws: MNISTError if the file cannot be read or has invalid format
///
/// ## Example
/// ```swift
/// let labels = try loadMNISTLabels(path: "./data/train-labels.idx1-ubyte")
/// print(labels.shape) // [60000]
/// ```
public func loadMNISTLabels(path: String, maxCount: Int? = nil) throws -> MLXArray {
    // -------------------------------------------------------------------------
    // Step 1: Read the raw file data
    // -------------------------------------------------------------------------
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url) else {
        throw MNISTError.fileNotFound(path)
    }
    
    // -------------------------------------------------------------------------
    // Step 2: Parse the IDX header and extract labels
    // -------------------------------------------------------------------------
    let labels: [Int32] = try data.withUnsafeBytes { rawBuf in
        guard let base = rawBuf.bindMemory(to: UInt8.self).baseAddress else {
            throw MNISTError.invalidFormat("Cannot read file buffer")
        }
        
        var offset = 0
        
        // Read and validate the magic number
        let magic = readBigEndianU32(from: base, at: &offset)
        guard magic == 0x00000801 else {
            throw MNISTError.invalidFormat("Invalid magic number: \(magic), expected 0x00000801")
        }
        
        // Read total count from header
        let totalLabels = Int(readBigEndianU32(from: base, at: &offset))
        
        // Determine how many labels to load
        let count = min(maxCount ?? totalLabels, totalLabels)
        
        // -------------------------------------------------------------------------
        // Step 3: Convert labels to Int32 array
        // -------------------------------------------------------------------------
        // We use Int32 because that's what MLX expects for integer classification
        var labels = [Int32](repeating: 0, count: count)
        
        for i in 0..<count {
            labels[i] = Int32(base[offset])
            offset += 1
        }
        
        return labels
    }
    
    // -------------------------------------------------------------------------
    // Step 4: Create MLXArray tensor
    // -------------------------------------------------------------------------
    return MLXArray(labels, [labels.count])
}

// =============================================================================
// MARK: - Convenience Functions
// =============================================================================

/// Loads both MNIST images and labels from a directory
///
/// This is a convenience function that loads both images and labels,
/// assuming the standard MNIST file naming convention.
///
/// - Parameters:
///   - directory: Directory containing the MNIST files
///   - train: If true, load training data; if false, load test data
/// - Returns: Tuple of (images, labels) MLXArrays
/// - Throws: MNISTError if files cannot be read
///
/// ## Example
/// ```swift
/// let (trainImages, trainLabels) = try loadMNIST(directory: "./data", train: true)
/// let (testImages, testLabels) = try loadMNIST(directory: "./data", train: false)
/// ```
public func loadMNIST(directory: String, train: Bool) throws -> (images: MLXArray, labels: MLXArray) {
    let prefix = train ? "train" : "t10k"
    let imagesPath = "\(directory)/\(prefix)-images.idx3-ubyte"
    let labelsPath = "\(directory)/\(prefix)-labels.idx1-ubyte"
    
    let images = try loadMNISTImages(path: imagesPath)
    let labels = try loadMNISTLabels(path: labelsPath)
    
    return (images, labels)
}

// =============================================================================
// MARK: - Batching Utilities
// =============================================================================

/// Creates batches from a dataset with optional shuffling
///
/// This generator yields (images, labels) tuples for each batch.
/// The last batch may be smaller than batchSize if the dataset
/// doesn't divide evenly.
///
/// - Parameters:
///   - images: Images array of shape [N, ...]
///   - labels: Labels array of shape [N]
///   - batchSize: Number of samples per batch
///   - shuffle: If true, shuffle indices before batching
/// - Returns: Array of (batchImages, batchLabels) tuples
///
/// ## Example
/// ```swift
/// for (batchImages, batchLabels) in batchDataset(images, labels, batchSize: 32, shuffle: true) {
///     // Train on this batch
///     let loss = trainStep(model, batchImages, batchLabels)
/// }
/// ```
public func batchDataset(
    _ images: MLXArray,
    _ labels: MLXArray,
    batchSize: Int,
    shuffle: Bool = true
) -> [(images: MLXArray, labels: MLXArray)] {
    let n = images.shape[0]
    
    // Create indices and optionally shuffle
    var indices = Array(0..<n)
    if shuffle {
        indices.shuffle()
    }
    
    // Create batches
    var batches: [(MLXArray, MLXArray)] = []
    var start = 0
    
    while start < n {
        let end = min(start + batchSize, n)
        let batchIndices = Array(indices[start..<end])
        
        // Use MLX's advanced indexing to select batch samples
        let idxArray = MLXArray(batchIndices.map { Int32($0) })
        let batchImages = images[idxArray]
        let batchLabels = labels[idxArray]
        
        batches.append((batchImages, batchLabels))
        start = end
    }
    
    return batches
}
