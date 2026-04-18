import Foundation
import Accelerate
import MNISTCommon

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Computes the softmax over a contiguous slice of `data` and writes the probabilities back into that slice.
/// 
/// The softmax is computed for `length` elements starting at index `base`; results replace the original values in `data`.
/// The computation is performed in a numerically stable way.
/// - Parameters:
///   - data: The array containing the input logits; the slice [base ..< base+length] is overwritten with softmax probabilities.
///   - base: The starting index of the slice within `data`.
///   - length: The number of elements in the slice to normalize.
/// - Note: `base` and `length` must define a valid range in `data`; empty slices are ignored.
func softmaxInPlace1D(_ data: inout [Float], base: Int, length: Int) {
    guard length > 0 else { return }

    var maxv = data[base]
    if length > 1 {
        for i in 1..<length {
            let v = data[base + i]
            if v > maxv { maxv = v }
        }
    }

    var sum: Float = 0
    for i in 0..<length {
        let e = expf(data[base + i] - maxv)
        data[base + i] = e
        sum += e
    }

    let inv = 1.0 / sum
    for i in 0..<length {
        data[base + i] *= inv
    }
}

/// Applies the rectified linear unit (ReLU) activation to a value.
/// - Returns: `x` if `x` is greater than zero, `0` otherwise.
func relu(_ x: Float) -> Float { x > 0 ? x : 0 }
