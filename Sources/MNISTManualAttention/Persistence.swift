import Foundation

enum AttentionPersistenceError: Error, CustomStringConvertible {
    case openFailed(filename: String, underlying: Error)
    case readFailed(filename: String, fieldName: String, underlying: Error)
    case truncated(filename: String, fieldName: String)

    var description: String {
        switch self {
        case let .openFailed(filename, underlying):
            return "Failed to open attention model file for writing: \(filename) (\(underlying))"
        case let .readFailed(filename, fieldName, underlying):
            return "Failed to read \(fieldName) from attention model file \(filename): \(underlying.localizedDescription)"
        case let .truncated(filename, fieldName):
            return "Attention model file truncated reading \(fieldName) from \(filename)"
        }
    }
}

/// Serialize the given attention model and write it to a binary file at the specified path.
///
/// The file is created or overwritten and contains a fixed Int32 header describing the model
/// dimensions followed by the model's parameter arrays encoded as IEEE‑754 doubles.
///
/// - Parameters:
///   - model: The `AttnModel` instance to persist.
///   - filename: Filesystem path where the model will be written; existing file will be replaced.
/// - Throws: `AttentionPersistenceError.openFailed` when the destination file cannot be opened, or
///           a `FileHandle` seek, truncate, or write error when model data cannot be written.
func saveModel(model: AttnModel, filename: String) throws {
    _ = FileManager.default.createFile(atPath: filename, contents: nil)
    let handle: FileHandle
    do {
        handle = try FileHandle(forWritingTo: URL(fileURLWithPath: filename))
    } catch {
        throw AttentionPersistenceError.openFailed(filename: filename, underlying: error)
    }
    defer { try? handle.close() }
    try handle.seek(toOffset: 0)
    try handle.truncate(atOffset: 0)

    func writeInt32(_ value: Int32) throws {
        var v = value
        try handle.write(contentsOf: Data(bytes: &v, count: MemoryLayout<Int32>.size))
    }

    func writeFloatArray(_ arr: [Float]) throws {
        for v in arr {
            var d = Double(v)
            try handle.write(contentsOf: Data(bytes: &d, count: MemoryLayout<Double>.size))
        }
    }

    // Write model dimensions.
    try writeInt32(Int32(patchDim))
    try writeInt32(Int32(dModel))
    try writeInt32(Int32(seqLen))
    try writeInt32(Int32(ffDim))
    try writeInt32(Int32(numClasses))

    // Write all weights and biases.
    try writeFloatArray(model.wPatch)
    try writeFloatArray(model.bPatch)
    try writeFloatArray(model.pos)
    try writeFloatArray(model.wQ)
    try writeFloatArray(model.bQ)
    try writeFloatArray(model.wK)
    try writeFloatArray(model.bK)
    try writeFloatArray(model.wV)
    try writeFloatArray(model.bV)
    try writeFloatArray(model.wFf1)
    try writeFloatArray(model.bFf1)
    try writeFloatArray(model.wFf2)
    try writeFloatArray(model.bFf2)
    try writeFloatArray(model.wCls)
    try writeFloatArray(model.bCls)

    print("Model saved to \(filename)")
}

/// Loads an `AttnModel` from a binary file and validates its architecture.
///
/// The function expects a file written by `saveModel` containing a 5-field Int32 header
/// (patchDim, dModel, seqLen, ffDim, numClasses) followed by the model parameters serialized as
/// Double-precision values.
/// - Parameter filename: Filesystem path to the binary model file.
/// - Returns: An `AttnModel` populated with the parameters from the file if successful; `nil` if the
///            file cannot be opened, the header is unreadable, the stored architecture does not match
///            the expected runtime dimensions, or the file is truncated/corrupted.
func loadModel(filename: String) -> AttnModel? {
    let handle: FileHandle
    do {
        handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: filename))
    } catch {
        let exists = FileManager.default.fileExists(atPath: filename)
        let reason = exists
            ? "Permission or I/O error: \(error.localizedDescription)"
            : "Model file not found: \(filename)"
        print("""

        ERROR: Failed to load attention model
        ======================================
        \(reason)

        Solutions:
          1. Train a new model to generate the file:
             swift run MNISTManualAttention --epochs 5

          2. Check if the file exists:
             ls -l \(filename)

          3. Verify you're in the correct directory:
             pwd
        """)
        return nil
    }
    defer { try? handle.close() }

    func printReadError(_ error: Error) {
        switch error {
        case let AttentionPersistenceError.truncated(_, fieldName):
            print("ERROR: Model file truncated reading \(fieldName) from \(filename). Retrain with: swift run MNISTManualAttention --epochs 5")
        case let AttentionPersistenceError.readFailed(_, fieldName, underlying):
            print("ERROR: I/O error reading \(fieldName) from \(filename): \(underlying.localizedDescription)")
        default:
            print("ERROR: Failed to read model file \(filename): \(error.localizedDescription)")
        }
    }

    func readData(byteCount: Int, fieldName: String) throws -> Data {
        do {
            guard let data = try handle.read(upToCount: byteCount),
                  data.count == byteCount else {
                throw AttentionPersistenceError.truncated(filename: filename, fieldName: fieldName)
            }
            return data
        } catch let error as AttentionPersistenceError {
            throw error
        } catch {
            throw AttentionPersistenceError.readFailed(filename: filename, fieldName: fieldName, underlying: error)
        }
    }

    func readInt32(fieldName: String) throws -> Int32 {
        let data = try readData(byteCount: MemoryLayout<Int32>.size, fieldName: fieldName)
        return data.withUnsafeBytes { $0.loadUnaligned(as: Int32.self) }
    }

    func readDouble(fieldName: String) throws -> Double {
        let data = try readData(byteCount: MemoryLayout<Double>.size, fieldName: fieldName)
        return data.withUnsafeBytes { $0.loadUnaligned(as: Double.self) }
    }

    /// Reads `count` Double values from the handle and converts them to Float.
    func readFloatArray(count: Int, fieldName: String) throws -> [Float] {
        var arr = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let val = try readDouble(fieldName: "\(fieldName)[\(i)/\(count)]")
            arr[i] = Float(val)
        }
        return arr
    }

    // Read and validate header.
    let patchDimRead: Int32
    let dModelRead: Int32
    let seqLenRead: Int32
    let ffDimRead: Int32
    let numClassesRead: Int32
    do {
        patchDimRead = try readInt32(fieldName: "patchDim")
        dModelRead = try readInt32(fieldName: "dModel")
        seqLenRead = try readInt32(fieldName: "seqLen")
        ffDimRead = try readInt32(fieldName: "ffDim")
        numClassesRead = try readInt32(fieldName: "numClasses")
    } catch {
        printReadError(error)
        return nil
    }

    if patchDimRead != Int32(patchDim) || dModelRead != Int32(dModel) ||
       seqLenRead != Int32(seqLen) || ffDimRead != Int32(ffDim) ||
       numClassesRead != Int32(numClasses) {
        print("""

        ERROR: Model architecture mismatch
        ==================================
        Expected: patchDim=\(patchDim) dModel=\(dModel) seqLen=\(seqLen) ffDim=\(ffDim) numClasses=\(numClasses)
        Found:    patchDim=\(patchDimRead) dModel=\(dModelRead) seqLen=\(seqLenRead) ffDim=\(ffDimRead) numClasses=\(numClassesRead)

        Solution: Retrain with current architecture: swift run MNISTManualAttention --epochs 5
        """)
        return nil
    }

    // Read all weights and biases.
    let wPatch: [Float]
    let bPatch: [Float]
    let pos: [Float]
    let wQ: [Float]
    let bQ: [Float]
    let wK: [Float]
    let bK: [Float]
    let wV: [Float]
    let bV: [Float]
    let wFf1: [Float]
    let bFf1: [Float]
    let wFf2: [Float]
    let bFf2: [Float]
    let wCls: [Float]
    let bCls: [Float]
    do {
        wPatch = try readFloatArray(count: patchDim * dModel, fieldName: "wPatch")
        bPatch = try readFloatArray(count: dModel,            fieldName: "bPatch")
        pos    = try readFloatArray(count: seqLen * dModel,   fieldName: "pos")
        wQ     = try readFloatArray(count: dModel * dModel,   fieldName: "wQ")
        bQ     = try readFloatArray(count: dModel,            fieldName: "bQ")
        wK     = try readFloatArray(count: dModel * dModel,   fieldName: "wK")
        bK     = try readFloatArray(count: dModel,            fieldName: "bK")
        wV     = try readFloatArray(count: dModel * dModel,   fieldName: "wV")
        bV     = try readFloatArray(count: dModel,            fieldName: "bV")
        wFf1   = try readFloatArray(count: dModel * ffDim,    fieldName: "wFf1")
        bFf1   = try readFloatArray(count: ffDim,             fieldName: "bFf1")
        wFf2   = try readFloatArray(count: ffDim * dModel,    fieldName: "wFf2")
        bFf2   = try readFloatArray(count: dModel,            fieldName: "bFf2")
        wCls   = try readFloatArray(count: dModel * numClasses, fieldName: "wCls")
        bCls   = try readFloatArray(count: numClasses,        fieldName: "bCls")
    } catch {
        printReadError(error)
        return nil
    }

    print("Model loaded from \(filename)")
    return AttnModel(
        wPatch: wPatch, bPatch: bPatch, pos: pos,
        wQ: wQ, bQ: bQ, wK: wK, bK: bK, wV: wV, bV: bV,
        wFf1: wFf1, bFf1: bFf1, wFf2: wFf2, bFf2: bFf2,
        wCls: wCls, bCls: bCls
    )
}
