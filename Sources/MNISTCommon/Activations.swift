import Foundation

// MARK: - Array-based activation functions

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

// MARK: - Pointer-based versions for MPS shared buffers

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
