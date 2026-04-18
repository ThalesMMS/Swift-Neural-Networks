import Foundation
import Accelerate
import MNISTCommon

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

#if canImport(Metal)
import Metal
#endif

#if canImport(MetalPerformanceShaders)
import MetalPerformanceShaders
#endif

// =============================================================================
// MARK: - Core Functions
// =============================================================================

/// Computes a numerically stable softmax over the given row, overwriting it with probabilities that sum to 1.
/// - Parameter row: An array of scores that will be replaced in-place by their softmax probabilities. Empty arrays are ignored.
func softmaxRowInPlace(_ row: inout [Float]) {
    guard !row.isEmpty else { return }

    var maxv = row[0]
    for v in row.dropFirst() { if v > maxv { maxv = v } }

    var sum: Float = 0
    for i in 0..<row.count {
        row[i] = expf(row[i] - maxv)
        sum += row[i]
    }

    guard sum > 0 else { return }

    let inv = 1.0 / sum
    for i in 0..<row.count { row[i] *= inv }
}

// CNN parameters stored in flat arrays for cache-friendly loops.
struct Cnn {
    // Conv: 1 -> convOut, kernel 3x3, pad=1
    var convW: [Float] // [convOut * 3 * 3]
    var convB: [Float] // [convOut]
    // FC: fcIn -> 10
    var fcW: [Float]   // [fcIn * 10]
    var fcB: [Float]   // [10]
}

/// Fills the weight array with uniformly distributed random values in the range [-limit, limit].
/// - Parameters:
///   - limit: Absolute bound for sampled values; values are drawn uniformly between `-limit` and `limit`.
///   - rng: Pseudo-random number generator used to sample values.
///   - w: Array of weights to initialize; elements are modified in place.
func xavierInit(limit: Float, rng: inout SimpleRng, w: inout [Float]) {
    for i in 0..<w.count {
        w[i] = rng.uniform(-limit, limit)
    }
}

/// Create a `Cnn` whose weights are initialized with Xavier/Glorot uniform initialization and whose biases are set to zero.
/// 
/// The Xavier limits for the convolution and fully-connected layers are computed from approximate fan-in/fan-out values derived from `kernel`, `convOut`, `fcIn`, and `numClasses`.
/// - Parameters:
///   - rng: A mutable pseudo-random number generator used to sample uniform values for the weight arrays.
/// - Returns: A `Cnn` with `convW` and `fcW` filled with Xavier-uniform random floats and `convB` and `fcB` initialized to zeros.
func initCnn(rng: inout SimpleRng) -> Cnn {
    // Xavier limits based on approximate fan-in/out.
    let fanIn: Float = Float(kernel * kernel)
    let fanOut: Float = Float(kernel * kernel * convOut)
    let convLimit = sqrtf(6.0 / (fanIn + fanOut))

    var convW = [Float](repeating: 0, count: convOut * kernel * kernel)
    let convB = [Float](repeating: 0, count: convOut)
    xavierInit(limit: convLimit, rng: &rng, w: &convW)

    let fcLimit = sqrtf(6.0 / (Float(fcIn) + Float(numClasses)))
    var fcW = [Float](repeating: 0, count: fcIn * numClasses)
    let fcB = [Float](repeating: 0, count: numClasses)
    xavierInit(limit: fcLimit, rng: &rng, w: &fcW)

    return Cnn(convW: convW, convB: convB, fcW: fcW, fcB: fcB)
}
