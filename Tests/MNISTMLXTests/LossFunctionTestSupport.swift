import XCTest
import MLX
import MLXNN
@testable import MNISTMLX

private struct SeededTestRng {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 0x9e3779b97f4a7c15 : seed
    }

    mutating func nextUInt32() -> UInt32 {
        var x = state
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        state = x
        return UInt32(truncatingIfNeeded: x >> 32)
    }

    mutating func nextFloat() -> Float {
        Float(nextUInt32()) / (Float(UInt32.max) + 1.0)
    }

    mutating func uniform(_ low: Float, _ high: Float) -> Float {
        low + (high - low) * nextFloat()
    }

    mutating func nextInt(upper: Int) -> Int {
        Int(nextUInt32()) % upper
    }
}

class LossFunctionTestSupport: MLXTestCase {
    // =============================================================================
    // MARK: - Test Utilities
    // =============================================================================

    /// Creates an MLXArray of shape [batchSize, numClasses] filled with random logits sampled uniformly from -5.0 to 5.0.
    /// - Parameters:
    ///   - batchSize: Number of samples (first dimension).
    ///   - numClasses: Number of classes (second dimension). Defaults to 10.
    ///   - seed: Optional seed for deterministic logits. When nil, MLX global randomness is used.
    /// - Returns: An MLXArray of shape [batchSize, numClasses] with values in the range [-5.0, 5.0].
    func createRandomLogits(batchSize: Int, numClasses: Int = 10, seed: UInt64? = nil) -> MLXArray {
        precondition(batchSize >= 0, "createRandomLogits: batchSize must be >= 0")
        precondition(numClasses > 0, "createRandomLogits: numClasses must be > 0")

        if let seed {
            var rng = SeededTestRng(seed: seed)
            var logitsData: [Float] = []
            logitsData.reserveCapacity(batchSize * numClasses)

            for _ in 0..<(batchSize * numClasses) {
                logitsData.append(rng.uniform(-5.0, 5.0))
            }

            return MLXArray(logitsData, [batchSize, numClasses])
        }

        // Create random logits in a reasonable range [-5, 5]
        return MLXRandom.uniform(low: -5.0, high: 5.0, [batchSize, numClasses])
    }

    /// Creates a 1-D label vector of length `batchSize` containing random class indices.
    /// - Parameters:
    ///   - batchSize: Number of labels to create.
    ///   - numClasses: Number of valid classes. Defaults to 10.
    ///   - seed: Optional seed for deterministic labels. When nil, Swift global randomness is used.
    /// - Returns: An `MLXArray` holding `Int32` labels of length `batchSize`, with each value sampled uniformly at random from `0..<numClasses`.
    func createRandomLabels(batchSize: Int, numClasses: Int = 10, seed: UInt64? = nil) -> MLXArray {
        precondition(batchSize >= 0, "createRandomLabels: batchSize must be >= 0")
        precondition(numClasses > 0, "createRandomLabels: numClasses must be > 0")

        if let seed {
            var rng = SeededTestRng(seed: seed)
            var labelsData: [Int32] = []
            labelsData.reserveCapacity(batchSize)

            for _ in 0..<batchSize {
                labelsData.append(Int32(rng.nextInt(upper: numClasses)))
            }

            return MLXArray(labelsData)
        }

        let labelsData = (0..<batchSize).map { _ in Int32.random(in: 0..<Int32(numClasses)) }
        return MLXArray(labelsData)
    }

    private func validatedClassIndex(_ label: Int32, index: Int, numClasses: Int, caller: String) -> Int {
        let classIndex = Int(label)
        guard classIndex >= 0 && classIndex < numClasses else {
            preconditionFailure("\(caller): label \(label) at index \(index) must be in 0..<\(numClasses)")
        }
        return classIndex
    }

    /// Creates logits that strongly favor the provided labels.
    /// - Parameters:
    ///   - labels: A 1-D `MLXArray` of integer class indices (shape `[batchSize]`).
    ///   - numClasses: The number of classes (default is 10).
    /// - Returns: An `MLXArray` with shape `[batchSize, numClasses]` where each row has `10.0` at the index specified by `labels` and `-10.0` for all other classes.
    func createPerfectLogits(labels: MLXArray, numClasses: Int = 10) -> MLXArray {
        precondition(numClasses > 0, "createPerfectLogits: numClasses must be > 0")

        eval(labels)
        let batchSize = labels.shape[0]

        // Set correct class logit to high value
        let labelsArray = labels.asArray(Int32.self)
        var allLogits: [Float] = []
        allLogits.reserveCapacity(batchSize * numClasses)

        for i in 0..<batchSize {
            let correctClass = validatedClassIndex(labelsArray[i], index: i, numClasses: numClasses, caller: "createPerfectLogits")
            // Create logits for this sample
            for j in 0..<numClasses {
                if j == correctClass {
                    allLogits.append(10.0)
                } else {
                    allLogits.append(-10.0)
                }
            }
        }

        return MLXArray(allLogits, [batchSize, numClasses])
    }

    /// Constructs a logits tensor that strongly favors a wrong class for each sample in `labels`.
    /// 
    /// For each label, the logit for `(correctClass + 1) % numClasses` is set to `10.0` and the logit
    /// for the correct class is set to `-10.0`; all other class logits are `-10.0`.
    /// - Parameters:
    ///   - labels: A 1-D `MLXArray` of integer class indices with length equal to the batch size.
    ///   - numClasses: The number of classes (default is 10). Must be greater than 1.
    /// - Returns: An `MLXArray` shaped `[batchSize, numClasses]` containing the constructed logits.
    func createWorstLogits(labels: MLXArray, numClasses: Int = 10) -> MLXArray {
        precondition(numClasses > 1, "createWorstLogits: numClasses must be > 1")

        eval(labels)
        let batchSize = labels.shape[0]

        // Set wrong class logit to high value
        let labelsArray = labels.asArray(Int32.self)
        var allLogits: [Float] = []
        allLogits.reserveCapacity(batchSize * numClasses)

        for i in 0..<batchSize {
            let correctClass = validatedClassIndex(labelsArray[i], index: i, numClasses: numClasses, caller: "createWorstLogits")
            let wrongClass = (correctClass + 1) % numClasses  // Pick different class

            // Create logits for this sample
            for j in 0..<numClasses {
                if j == wrongClass {
                    allLogits.append(10.0)
                } else {
                    allLogits.append(-10.0)
                }
            }
        }

        return MLXArray(allLogits, [batchSize, numClasses])
    }

    /// Evaluates an `MLXArray` and asserts it is non-empty and every value is finite.
    /// - Parameters:
    ///   - array: The `MLXArray` to evaluate and inspect.
    ///   - message: Failure message used when a materialized element is NaN or infinity. Defaults to `"All values should be finite"`.
    ///   - file: Source file reported for XCTest failures. Defaults to the caller's file.
    ///   - line: Source line reported for XCTest failures. Defaults to the caller's line.
    /// - Failure: Fails the test when the evaluated array is empty or any element is NaN or infinity.
    func assertAllFinite(_ array: MLXArray,
                                 _ message: String = "All values should be finite",
                                 file: StaticString = #file,
                                 line: UInt = #line) {
        eval(array)
        XCTAssertTrue(array.size > 0, "Array should not be empty", file: file, line: line)
        let values = array.asArray(Float.self)
        for (index, value) in values.enumerated() {
            XCTAssertTrue(value.isFinite, "\(message) at index \(index): \(value)", file: file, line: line)
        }
    }

    /// Asserts that a floating-point value lies between the given lower and upper bounds (inclusive).
    /// - Parameters:
    ///   - value: The value to validate.
    ///   - low: The inclusive lower bound.
    ///   - high: The inclusive upper bound.
    ///   - message: Optional failure message to display if the assertion fails.
    ///   - file: The source file to report on failure (defaults to the caller's file).
    ///   - line: The source line to report on failure (defaults to the caller's line).
    func assertInRange(_ value: Float, _ low: Float, _ high: Float,
                              _ message: String = "",
                              file: StaticString = #file,
                              line: UInt = #line) {
        XCTAssertGreaterThanOrEqual(value, low, message, file: file, line: line)
        XCTAssertLessThanOrEqual(value, high, message, file: file, line: line)
    }
}
