import Foundation
import Accelerate
import MNISTCommon

#if canImport(Metal)
import Metal
#endif

#if canImport(MetalPerformanceShaders)
import MetalPerformanceShaders
#endif

enum CnnPersistenceError: Error, CustomStringConvertible {
    case openFailed(filename: String, underlying: Error)
    case closeFailed(filename: String, underlying: Error)
    case replaceFailed(filename: String, tempFilename: String, underlying: Error)

    var description: String {
        switch self {
        case let .openFailed(filename, underlying):
            return "Failed to open CNN model file for writing: \(filename) (\(underlying))"
        case let .closeFailed(filename, underlying):
            return "Failed to close CNN model file after writing: \(filename) (\(underlying))"
        case let .replaceFailed(filename, tempFilename, underlying):
            return "Failed to replace CNN model file \(filename) with temporary file \(tempFilename) (\(underlying))"
        }
    }
}

/// Serialize and write a `Cnn` model to a binary file at the given path.
///
/// The file format begins with a 4×`Int32` header (convOut, kernel, fcIn, numClasses) followed
/// by parameters serialized as Double values: convW, convB, fcW, fcB.
/// - Parameters:
///   - model: The `Cnn` instance to persist.
///   - filename: Filesystem path where the binary model will be written.
/// - Throws: `CnnPersistenceError` if the destination cannot be opened or closed.
func saveModel(model: Cnn, filename: String) throws {
    let fileManager = FileManager.default
    let destinationURL = URL(fileURLWithPath: filename)
    let tempURL = destinationURL
        .deletingLastPathComponent()
        .appendingPathComponent(".\(destinationURL.lastPathComponent).tmp.\(UUID().uuidString)")

    _ = fileManager.createFile(atPath: tempURL.path, contents: nil)
    var shouldRemoveTemp = true
    defer {
        if shouldRemoveTemp {
            try? fileManager.removeItem(at: tempURL)
        }
    }

    let handle: FileHandle
    do {
        handle = try FileHandle(forWritingTo: tempURL)
    } catch {
        throw CnnPersistenceError.openFailed(filename: filename, underlying: error)
    }

    func writeInt32(_ value: Int32) {
        var v = value
        handle.write(Data(bytes: &v, count: MemoryLayout<Int32>.size))
    }

    func writeFloatArray(_ arr: [Float]) {
        var data = Data()
        data.reserveCapacity(arr.count * MemoryLayout<Double>.size)

        for v in arr {
            var d = Double(v)
            withUnsafeBytes(of: &d) { bytes in
                data.append(contentsOf: bytes)
            }
        }

        handle.write(data)
    }

    // Write model dimensions (header).
    writeInt32(Int32(convOut))
    writeInt32(Int32(kernel))
    writeInt32(Int32(fcIn))
    writeInt32(Int32(numClasses))

    writeFloatArray(model.convW)
    writeFloatArray(model.convB)
    writeFloatArray(model.fcW)
    writeFloatArray(model.fcB)

    do {
        try handle.close()
    } catch {
        throw CnnPersistenceError.closeFailed(filename: filename, underlying: error)
    }

    do {
        if fileManager.fileExists(atPath: destinationURL.path) {
            _ = try fileManager.replaceItemAt(destinationURL, withItemAt: tempURL)
        } else {
            try fileManager.moveItem(at: tempURL, to: destinationURL)
        }
        shouldRemoveTemp = false
    } catch {
        throw CnnPersistenceError.replaceFailed(filename: filename, tempFilename: tempURL.path, underlying: error)
    }

    print("Model saved to \(filename)")
}

/// Loads a CNN model from a binary file and returns a reconstructed `Cnn` if loading succeeds.
///
/// Validates the header against the current architecture constants and reads all parameter arrays.
/// Returns `nil` with a printed error if the file is missing, corrupted, or has a mismatched architecture.
/// - Parameter filename: Filesystem path to the binary model file.
/// - Returns: A `Cnn` populated with the model parameters on success, `nil` on failure.
func loadModel(filename: String) -> Cnn? {
    guard let handle = try? FileHandle(forReadingFrom: URL(fileURLWithPath: filename)) else {
        print("""

        ERROR: Failed to load CNN model
        ================================
        Model file not found: \(filename)

        Solutions:
          1. Train a new model first: swift run MNISTManualCNN --epochs 3
          2. Verify the file exists: ls -la \(filename)

        """)
        return nil
    }
    defer { try? handle.close() }

    func readInt32() -> Int32? {
        guard let data = try? handle.read(upToCount: MemoryLayout<Int32>.size),
              data.count == MemoryLayout<Int32>.size else { return nil }
        return data.withUnsafeBytes { $0.loadUnaligned(as: Int32.self) }
    }

    func readDouble() -> Double? {
        guard let data = try? handle.read(upToCount: MemoryLayout<Double>.size),
              data.count == MemoryLayout<Double>.size else { return nil }
        return data.withUnsafeBytes { $0.loadUnaligned(as: Double.self) }
    }

    /// Reads `count` Double values and converts them to Float; returns nil on truncation.
    func readFloatArray(count: Int, fieldName: String) -> [Float]? {
        var arr = [Float](repeating: 0, count: count)
        for i in 0..<count {
            guard let val = readDouble() else {
                print("ERROR: Model file truncated reading \(fieldName)[\(i)/\(count)] from \(filename). Retrain with: swift run MNISTManualCNN --epochs 3")
                return nil
            }
            arr[i] = Float(val)
        }
        return arr
    }

    // Read and validate header.
    guard let convOutRead = readInt32(),
          let kernelRead = readInt32(),
          let fcInRead = readInt32(),
          let numClassesRead = readInt32() else {
        print("""

        ERROR: Corrupted model file - header unreadable
        ================================================
        Failed to read model header from: \(filename)

        The file may be truncated, corrupted, or not a valid CNN model file.
        Solution: Retrain with: swift run MNISTManualCNN --epochs 3
        """)
        return nil
    }

    if convOutRead != Int32(convOut) || kernelRead != Int32(kernel) ||
       fcInRead != Int32(fcIn) || numClassesRead != Int32(numClasses) {
        print("""

        ERROR: Model architecture mismatch
        ==================================
        Expected: convOut=\(convOut) kernel=\(kernel) fcIn=\(fcIn) numClasses=\(numClasses)
        Found:    convOut=\(convOutRead) kernel=\(kernelRead) fcIn=\(fcInRead) numClasses=\(numClassesRead)

        Solution: Retrain with current architecture: swift run MNISTManualCNN --epochs 3
        """)
        return nil
    }

    guard let convW = readFloatArray(count: convOut * kernel * kernel, fieldName: "convW"),
          let convB = readFloatArray(count: convOut,                   fieldName: "convB"),
          let fcW   = readFloatArray(count: fcIn * numClasses,         fieldName: "fcW"),
          let fcB   = readFloatArray(count: numClasses,                fieldName: "fcB")
    else { return nil }

    print("Model loaded from \(filename)")
    return Cnn(convW: convW, convB: convB, fcW: fcW, fcB: fcB)
}
