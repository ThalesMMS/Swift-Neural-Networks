import Foundation

// MARK: - Array-based activation functions

/// Apply ReLU activation in-place (set negative values to 0).
func reluInPlace(_ data: inout [Float], count: Int) {
    for i in 0..<count {
        if data[i] < 0 {
            data[i] = 0
        }
    }
}

// Row-wise softmax (in-place).
func softmaxRows(_ data: inout [Float], rows: Int, cols: Int) {
    for r in 0..<rows {
        let base = r * cols
        var maxVal = data[base]
        for c in 1..<cols {
            let v = data[base + c]
            if v > maxVal {
                maxVal = v
            }
        }

        var sum: Float = 0.0
        for c in 0..<cols {
            let e = expf(data[base + c] - maxVal)
            data[base + c] = e
            sum += e
        }

        let inv = 1.0 / sum
        for c in 0..<cols {
            data[base + c] *= inv
        }
    }
}

// Create output delta (softmax + cross-entropy) and accumulate loss.
func computeDeltaAndLoss(
    outputs: [Float],
    labels: [UInt8],
    rows: Int,
    cols: Int,
    delta: inout [Float],
    totalLoss: inout Float
) {
    for r in 0..<rows {
        let base = r * cols
        for c in 0..<cols {
            delta[base + c] = outputs[base + c]
        }
        let label = Int(labels[r])
        let prob = max(outputs[base + label], 1e-9)
        totalLoss += -logf(prob)
        delta[base + label] -= 1
    }
}

// Sum columns (bias gradients).
func sumRows(_ data: [Float], rows: Int, cols: Int, result: inout [Float]) {
    for c in 0..<cols {
        result[c] = 0
    }
    for r in 0..<rows {
        let base = r * cols
        for c in 0..<cols {
            result[c] += data[base + c]
        }
    }
}

// Copy batch by indices to keep contiguous memory for GEMM.
func gatherBatch(
    images: [Float],
    labels: [UInt8],
    indices: [Int],
    start: Int,
    count: Int,
    inputSize: Int,
    outInputs: inout [Float],
    outLabels: inout [UInt8]
) {
    images.withUnsafeBufferPointer { srcBuf in
        outInputs.withUnsafeMutableBufferPointer { dstBuf in
            guard let srcBase = srcBuf.baseAddress, let dstBase = dstBuf.baseAddress else {
                return
            }
            for i in 0..<count {
                let srcIndex = indices[start + i]
                let srcPtr = srcBase.advanced(by: srcIndex * inputSize)
                let dstPtr = dstBase.advanced(by: i * inputSize)
                dstPtr.update(from: srcPtr, count: inputSize)
                outLabels[i] = labels[srcIndex]
            }
        }
    }
}

// MARK: - Pointer-based versions for MPS shared buffers

/// Add bias to each row using unsafe pointers.
func addBiasPointer(_ data: UnsafeMutablePointer<Float>, rows: Int, cols: Int, bias: [Float]) {
    for r in 0..<rows {
        let base = r * cols
        for c in 0..<cols {
            data[base + c] += bias[c]
        }
    }
}

/// Apply ReLU activation in-place using unsafe pointers.
func reluInPlacePointer(_ data: UnsafeMutablePointer<Float>, count: Int) {
    for i in 0..<count {
        if data[i] < 0 {
            data[i] = 0
        }
    }
}

/// Row-wise softmax using unsafe pointers.
func softmaxRowsPointer(_ data: UnsafeMutablePointer<Float>, rows: Int, cols: Int) {
    for r in 0..<rows {
        let base = r * cols
        var maxVal = data[base]
        for c in 1..<cols {
            let v = data[base + c]
            if v > maxVal {
                maxVal = v
            }
        }

        var sum: Float = 0.0
        for c in 0..<cols {
            let e = expf(data[base + c] - maxVal)
            data[base + c] = e
            sum += e
        }

        let inv = 1.0 / sum
        for c in 0..<cols {
            data[base + c] *= inv
        }
    }
}

/// Create output delta (softmax + cross-entropy) and accumulate loss using unsafe pointers.
func computeDeltaAndLossPointer(
    outputs: UnsafePointer<Float>,
    labels: [UInt8],
    rows: Int,
    cols: Int,
    delta: UnsafeMutablePointer<Float>,
    totalLoss: inout Float
) {
    for r in 0..<rows {
        let base = r * cols
        for c in 0..<cols {
            delta[base + c] = outputs[base + c]
        }
        let label = Int(labels[r])
        let prob = max(outputs[base + label], 1e-9)
        totalLoss += -logf(prob)
        delta[base + label] -= 1
    }
}

/// Sum columns (bias gradients) using unsafe pointers.
func sumRowsPointer(_ data: UnsafePointer<Float>, rows: Int, cols: Int, result: inout [Float]) {
    for c in 0..<cols {
        result[c] = 0
    }
    for r in 0..<rows {
        let base = r * cols
        for c in 0..<cols {
            result[c] += data[base + c]
        }
    }
}
