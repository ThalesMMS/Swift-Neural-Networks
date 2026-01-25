import Foundation
import Darwin

/// A simple random number generator using xorshift algorithm.
/// This provides fast, deterministic pseudo-random number generation
/// for neural network weight initialization and data shuffling.
struct SimpleRng {
    private var state: UInt64

    // Explicit seed (if zero, use a fixed value).
    init(seed: UInt64) {
        self.state = seed == 0 ? 0x9e3779b97f4a7c15 : seed
    }

    // Reseed based on the current time.
    mutating func reseedFromTime() {
        let nanos = UInt64(Date().timeIntervalSince1970 * 1_000_000_000)
        state = nanos == 0 ? 0x9e3779b97f4a7c15 : nanos
    }

    // Basic xorshift to generate u32.
    mutating func nextUInt32() -> UInt32 {
        var x = state
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        state = x
        return UInt32(truncatingIfNeeded: x >> 32)
    }

    // Convert to [0, 1).
    mutating func nextFloat() -> Float {
        return Float(nextUInt32()) / Float(UInt32.max)
    }

    // Uniform sample in [low, high).
    mutating func uniform(_ low: Float, _ high: Float) -> Float {
        return low + (high - low) * nextFloat()
    }

    // Integer sample in [0, upper).
    mutating func nextInt(upper: Int) -> Int {
        return upper == 0 ? 0 : Int(nextUInt32()) % upper
    }
}
