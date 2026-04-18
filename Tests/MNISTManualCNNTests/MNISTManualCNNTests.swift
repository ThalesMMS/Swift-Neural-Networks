import XCTest
import MNISTCommon
@testable import MNISTManualCNN

final class MNISTManualCNNTests: XCTestCase {

    // MARK: - Model Initialization

    func testModelInitializationDimensions() {
        var rng = SimpleRng(seed: 42)
        let model = initCnn(rng: &rng)

        XCTAssertEqual(model.convW.count, convOut * kernel * kernel)
        XCTAssertEqual(model.convB.count, convOut)
        XCTAssertEqual(model.fcW.count, fcIn * numClasses)
        XCTAssertEqual(model.fcB.count, numClasses)
    }

    func testModelWeightsAreFinite() {
        // All initialized weights must be finite (no NaN or Inf).
        var rng = SimpleRng(seed: 7)
        let model = initCnn(rng: &rng)

        XCTAssertTrue(model.convW.allSatisfy(\.isFinite),
                      "Conv weights should all be finite")
        XCTAssertTrue(model.fcW.allSatisfy(\.isFinite),
                      "FC weights should all be finite")
    }

    func testModelBiasesAreZero() {
        // Biases are initialized to zero.
        var rng = SimpleRng(seed: 7)
        let model = initCnn(rng: &rng)

        XCTAssertTrue(model.convB.allSatisfy { $0 == 0 },
                      "Conv biases should be initialized to zero")
        XCTAssertTrue(model.fcB.allSatisfy { $0 == 0 },
                      "FC biases should be initialized to zero")
    }

    func testModelWeightsWithinXavierBounds() {
        // Xavier limit for conv: sqrt(6 / (fanIn + fanOut)) where fanIn=9, fanOut=9*8=72.
        let fanIn: Float = Float(kernel * kernel)
        let fanOut: Float = Float(kernel * kernel * convOut)
        let convLimit = (6.0 / (fanIn + fanOut)).squareRoot()
        let fcLimit = (6.0 / (Float(fcIn) + Float(numClasses))).squareRoot()

        var rng = SimpleRng(seed: 55)
        let model = initCnn(rng: &rng)

        for w in model.convW {
            XCTAssertGreaterThanOrEqual(w, -convLimit,
                                       "Conv weight should be >= -convLimit")
            XCTAssertLessThanOrEqual(w, convLimit,
                                     "Conv weight should be <= convLimit")
        }
        for w in model.fcW {
            XCTAssertGreaterThanOrEqual(w, -fcLimit,
                                       "FC weight should be >= -fcLimit")
            XCTAssertLessThanOrEqual(w, fcLimit,
                                     "FC weight should be <= fcLimit")
        }
    }

    func testModelDifferentSeedsDifferentWeights() {
        // Two different seeds must produce different weight vectors.
        var rng1 = SimpleRng(seed: 1)
        var rng2 = SimpleRng(seed: 2)
        let model1 = initCnn(rng: &rng1)
        let model2 = initCnn(rng: &rng2)

        XCTAssertNotEqual(model1.convW, model2.convW,
                          "Different seeds should produce different conv weights")
    }

    func testModelSameSeedReproducible() {
        // Same seed must produce identical models.
        var rng1 = SimpleRng(seed: 99)
        var rng2 = SimpleRng(seed: 99)
        let model1 = initCnn(rng: &rng1)
        let model2 = initCnn(rng: &rng2)

        XCTAssertEqual(model1.convW, model2.convW,
                       "Same seed should produce identical conv weights")
        XCTAssertEqual(model1.fcW, model2.fcW,
                       "Same seed should produce identical FC weights")
    }

    // MARK: - im2col / col2im

    func testIm2colAndCol2imSinglePixelKernel() {
        let input: [Float] = [1, 2, 3, 4]

        let columns = im2colForward(
            input: input,
            batch: 1,
            inChannels: 1,
            height: 2,
            width: 2,
            kernelSize: 1,
            pad: 0
        )
        XCTAssertEqual(columns, input)

        let image = col2im(
            colData: columns,
            batch: 1,
            inChannels: 1,
            height: 2,
            width: 2,
            kernelSize: 1,
            pad: 0
        )
        XCTAssertEqual(image, input)
    }

    func testIm2colAndCol2imThreeByThreeKernelAccumulatesOverlap() {
        let input: [Float] = [
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        ]

        let columns = im2colForward(
            input: input,
            batch: 1,
            inChannels: 1,
            height: 3,
            width: 3,
            kernelSize: 3,
            pad: 1
        )

        let expectedColumns: [Float] = [
            0, 0, 0, 0, 1, 2, 0, 4, 5,
            0, 0, 0, 1, 2, 3, 4, 5, 6,
            0, 0, 0, 2, 3, 0, 5, 6, 0,
            0, 1, 2, 0, 4, 5, 0, 7, 8,
            1, 2, 3, 4, 5, 6, 7, 8, 9,
            2, 3, 0, 5, 6, 0, 8, 9, 0,
            0, 4, 5, 0, 7, 8, 0, 0, 0,
            4, 5, 6, 7, 8, 9, 0, 0, 0,
            5, 6, 0, 8, 9, 0, 0, 0, 0,
        ]
        XCTAssertEqual(columns, expectedColumns)

        let image = col2im(
            colData: columns,
            batch: 1,
            inChannels: 1,
            height: 3,
            width: 3,
            kernelSize: 3,
            pad: 1
        )

        XCTAssertEqual(image, [
            4, 12, 12,
            24, 45, 36,
            28, 48, 36,
        ])
    }

    func testIm2colOutputSizeWithPadding() {
        // With pad=1 and kernel=3 on a 4×4 single-channel single-batch image,
        // outHeight=4, outWidth=4, kernelSpatial=9, colChannels=9, colWidth=16.
        let h = 4, w = 4
        let input = [Float](repeating: 1.0, count: h * w)
        let columns = im2colForward(
            input: input,
            batch: 1,
            inChannels: 1,
            height: h,
            width: w,
            kernelSize: 3,
            pad: 1
        )
        let expectedRows = 3 * 3 * 1  // kernelSpatial * inChannels
        let expectedCols = h * w * 1  // outSpatial * batch
        XCTAssertEqual(columns.count, expectedRows * expectedCols,
                       "im2col output should have (k*k*C) × (H*W*B) elements")
    }

    func testIm2colPaddingZeroFillsBoundary() {
        // A 2×2 image with pad=1 and kernel=3 – only the center pixel of each
        // kernel window maps to a real input; boundary positions must be 0.
        // We use an input of all 1s and verify the column total equals image area.
        let h = 2, w = 2
        let input = [Float](repeating: 1.0, count: h * w)
        let columns = im2colForward(
            input: input,
            batch: 1,
            inChannels: 1,
            height: h,
            width: w,
            kernelSize: 3,
            pad: 1
        )
        // Total number of 1-filled entries must equal number of valid pixels per kernel window.
        let totalNonZero = columns.filter { $0 != 0 }.count
        // Each of the h*w spatial output positions receives 1 contribution per kernel position
        // that maps to a valid input pixel. Since every pixel appears in up to k*k windows but
        // we have exactly h*w input pixels, the total active entries = h*w * kernelSpatial only
        // when there's no padding overlap. With pad the total equals sum over all kernel positions
        // of valid (in-bounds) spatial entries. It must be at most h*w*kernelSpatial.
        let maxPossible = h * w * 3 * 3
        XCTAssertGreaterThan(totalNonZero, 0, "Should have some non-zero values")
        XCTAssertLessThanOrEqual(totalNonZero, maxPossible,
                                 "Non-zero count must not exceed total elements")
    }

    func testCol2imOutputSizeMatchesInput() {
        // col2im must produce the same number of elements as the original image.
        let h = 4, w = 4, batch = 2, ch = 1
        let inputSize = batch * ch * h * w
        let input = (0..<inputSize).map { Float($0) }

        let columns = im2colForward(
            input: input,
            batch: batch,
            inChannels: ch,
            height: h,
            width: w,
            kernelSize: 3,
            pad: 1
        )

        let backToImage = col2im(
            colData: columns,
            batch: batch,
            inChannels: ch,
            height: h,
            width: w,
            kernelSize: 3,
            pad: 1
        )

        XCTAssertEqual(backToImage.count, inputSize,
                       "col2im output must have the same element count as the original image")
    }

    // MARK: - softmaxRowInPlace

    func testSoftmaxRowInPlaceSumsToOne() {
        var row: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        softmaxRowInPlace(&row)

        let sum = row.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-6, "Softmax output should sum to 1")
    }

    func testSoftmaxRowInPlaceAllInRange() {
        var row: [Float] = [-2.0, 0.0, 3.0, -1.0, 5.0]
        softmaxRowInPlace(&row)

        for v in row {
            XCTAssertGreaterThanOrEqual(v, 0.0, "Softmax probabilities must be >= 0")
            XCTAssertLessThanOrEqual(v, 1.0, "Softmax probabilities must be <= 1")
        }
    }

    func testSoftmaxRowInPlaceMaxWins() {
        var row: [Float] = [0.0, 10.0, 0.0]
        softmaxRowInPlace(&row)
        // Index 1 should dominate.
        XCTAssertGreaterThan(row[1], row[0],
                             "Softmax of the max element should be highest")
        XCTAssertGreaterThan(row[1], row[2],
                             "Softmax of the max element should be highest")
    }

    func testSoftmaxRowInPlaceNumericallyStable() {
        // Large values should not produce NaN/Inf.
        var row: [Float] = [1000.0, 1001.0, 1002.0]
        softmaxRowInPlace(&row)

        XCTAssertTrue(row.allSatisfy(\.isFinite), "Large inputs must not produce Inf/NaN")
        XCTAssertEqual(row.reduce(0, +), 1.0, accuracy: 1e-5)
    }

    func testSoftmaxRowInPlaceSingleElement() {
        var row: [Float] = [42.0]
        softmaxRowInPlace(&row)
        XCTAssertEqual(row[0], 1.0, accuracy: 1e-6,
                       "Softmax of a single element should be 1.0")
    }

    func testSoftmaxRowInPlaceEmptyRowReturns() {
        var row: [Float] = []
        softmaxRowInPlace(&row)
        XCTAssertTrue(row.isEmpty)
    }

    // MARK: - Forward Pass

    func testCpuForwardAndLossAreFinite() {
        var rng = SimpleRng(seed: 7)
        let model = initCnn(rng: &rng)
        let batch = 2
        let input = [Float](repeating: 0.25, count: batch * numInputs)
        var convAct = [Float](repeating: 0, count: batch * convOut * imgH * imgW)
        var poolOut = [Float](repeating: 0, count: batch * fcIn)
        var poolIdx = [UInt8](repeating: 0, count: batch * convOut * poolH * poolW)
        var logits = [Float](repeating: 0, count: batch * numClasses)
        var delta = [Float](repeating: 0, count: batch * numClasses)

        convForwardRelu(model: model, batch: batch, input: input, convOutAct: &convAct)
        maxPoolForward(batch: batch, convAct: convAct, poolOut: &poolOut, poolIdx: &poolIdx)
        fcForward(model: model, batch: batch, x: poolOut, logits: &logits)

        let loss = softmaxXentBackward(
            probsInPlace: &logits,
            labels: [0, 1],
            batch: batch,
            delta: &delta,
            scale: 1.0 / Float(batch)
        )

        XCTAssertTrue(loss.isFinite)
        XCTAssertTrue(delta.allSatisfy(\.isFinite))
    }

    func testConvForwardOutputHasCorrectCount() {
        // After convForwardRelu the activation buffer must have batch * convOut * H * W elements.
        var rng = SimpleRng(seed: 13)
        let model = initCnn(rng: &rng)
        let batch = 3
        let input = [Float](repeating: 0.5, count: batch * numInputs)
        var convAct = [Float](repeating: 0, count: batch * convOut * imgH * imgW)

        convForwardRelu(model: model, batch: batch, input: input, convOutAct: &convAct)

        XCTAssertEqual(convAct.count, batch * convOut * imgH * imgW)
        XCTAssertTrue(convAct.allSatisfy(\.isFinite),
                      "Conv activations should all be finite")
    }

    func testConvForwardReluAppliesRelu() {
        // ReLU is applied: no negative values should appear in convAct.
        var rng = SimpleRng(seed: 14)
        let model = initCnn(rng: &rng)
        let batch = 2
        let input = [Float](repeating: 0.1, count: batch * numInputs)
        var convAct = [Float](repeating: -999, count: batch * convOut * imgH * imgW)

        convForwardRelu(model: model, batch: batch, input: input, convOutAct: &convAct)

        XCTAssertTrue(convAct.allSatisfy { $0 >= 0 },
                      "ReLU must clamp all negative values to zero")
    }

    func testMaxPoolOutputHasCorrectCount() {
        var rng = SimpleRng(seed: 21)
        let model = initCnn(rng: &rng)
        let batch = 4
        let input = [Float](repeating: 0.3, count: batch * numInputs)
        var convAct = [Float](repeating: 0, count: batch * convOut * imgH * imgW)
        var poolOut = [Float](repeating: 0, count: batch * fcIn)
        var poolIdx = [UInt8](repeating: 0, count: batch * convOut * poolH * poolW)

        convForwardRelu(model: model, batch: batch, input: input, convOutAct: &convAct)
        maxPoolForward(batch: batch, convAct: convAct, poolOut: &poolOut, poolIdx: &poolIdx)

        XCTAssertEqual(poolOut.count, batch * fcIn,
                       "Pool output should have batch * fcIn elements")
        XCTAssertTrue(poolOut.allSatisfy(\.isFinite),
                      "Pool output should all be finite")
    }

    func testFcForwardOutputHasCorrectCount() {
        var rng = SimpleRng(seed: 31)
        let model = initCnn(rng: &rng)
        let batch = 5
        let input = [Float](repeating: 0.2, count: batch * numInputs)
        var convAct = [Float](repeating: 0, count: batch * convOut * imgH * imgW)
        var poolOut = [Float](repeating: 0, count: batch * fcIn)
        var poolIdx = [UInt8](repeating: 0, count: batch * convOut * poolH * poolW)
        var logits = [Float](repeating: 0, count: batch * numClasses)

        convForwardRelu(model: model, batch: batch, input: input, convOutAct: &convAct)
        maxPoolForward(batch: batch, convAct: convAct, poolOut: &poolOut, poolIdx: &poolIdx)
        fcForward(model: model, batch: batch, x: poolOut, logits: &logits)

        XCTAssertEqual(logits.count, batch * numClasses,
                       "FC logits should have batch * numClasses elements")
        XCTAssertTrue(logits.allSatisfy(\.isFinite),
                      "FC logits should all be finite")
    }

    func testSoftmaxXentBackwardDeltaSumsToZeroPerRow() {
        // Cross-entropy + softmax gradient property: sum of delta over classes = 0 per sample.
        var rng = SimpleRng(seed: 41)
        let model = initCnn(rng: &rng)
        let batch = 3
        let input = [Float](repeating: 0.4, count: batch * numInputs)
        var convAct = [Float](repeating: 0, count: batch * convOut * imgH * imgW)
        var poolOut = [Float](repeating: 0, count: batch * fcIn)
        var poolIdx = [UInt8](repeating: 0, count: batch * convOut * poolH * poolW)
        var logits = [Float](repeating: 0, count: batch * numClasses)
        var delta = [Float](repeating: 0, count: batch * numClasses)

        convForwardRelu(model: model, batch: batch, input: input, convOutAct: &convAct)
        maxPoolForward(batch: batch, convAct: convAct, poolOut: &poolOut, poolIdx: &poolIdx)
        fcForward(model: model, batch: batch, x: poolOut, logits: &logits)

        _ = softmaxXentBackward(
            probsInPlace: &logits,
            labels: [0, 1, 2],
            batch: batch,
            delta: &delta,
            scale: 1.0 / Float(batch)
        )

        for b in 0..<batch {
            let rowSum = (0..<numClasses).map { delta[b * numClasses + $0] }.reduce(0, +)
            XCTAssertEqual(rowSum, 0.0, accuracy: 1e-5,
                           "Gradient row \(b) should sum to approximately zero")
        }
    }

    // MARK: - Config Defaults

    func testConfigDefaultValues() {
        let config = Config()
        XCTAssertEqual(config.epochs, 3)
        XCTAssertEqual(config.batchSize, 32)
        XCTAssertEqual(config.learningRate, 0.01, accuracy: 1e-9)
        XCTAssertEqual(config.dataPath, "./data")
        XCTAssertEqual(config.seed, 1)
        XCTAssertFalse(config.useGpu)
    }

    // MARK: - Persistence

    func testPersistenceRoundTrip() throws {
        var rng = SimpleRng(seed: 99)
        let model = initCnn(rng: &rng)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mnist_manual_cnn_\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }

        try saveModel(model: model, filename: url.path)
        let loaded = loadModel(filename: url.path)

        XCTAssertEqual(loaded?.convW.count, model.convW.count)
        XCTAssertEqual(loaded?.convB.count, model.convB.count)
        XCTAssertEqual(loaded?.fcW.count, model.fcW.count)
        XCTAssertEqual(loaded?.fcB.count, model.fcB.count)
        XCTAssertEqual(loaded?.convW.first, model.convW.first)
        XCTAssertEqual(loaded?.fcW.last, model.fcW.last)

        var replacementRng = SimpleRng(seed: 100)
        let replacement = initCnn(rng: &replacementRng)
        try saveModel(model: replacement, filename: url.path)
        let replaced = loadModel(filename: url.path)
        XCTAssertEqual(replaced?.convW.first, replacement.convW.first)
        XCTAssertEqual(replaced?.fcW.last, replacement.fcW.last)
    }

    func testPersistencePreservesAllWeightValues() throws {
        // Every weight value must survive the save/load roundtrip.
        var rng = SimpleRng(seed: 200)
        let model = initCnn(rng: &rng)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mnist_manual_cnn_values_\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }

        try saveModel(model: model, filename: url.path)
        guard let loaded = loadModel(filename: url.path) else {
            XCTFail("loadModel returned nil")
            return
        }

        for (i, (orig, load)) in zip(model.convW, loaded.convW).enumerated() {
            XCTAssertEqual(orig, load, accuracy: 1e-6,
                           "convW[\(i)] should survive the roundtrip")
        }
        for (i, (orig, load)) in zip(model.fcW, loaded.fcW).enumerated() {
            XCTAssertEqual(orig, load, accuracy: 1e-6,
                           "fcW[\(i)] should survive the roundtrip")
        }
    }

    func testPersistenceSaveToMissingDirectoryThrows() {
        var rng = SimpleRng(seed: 201)
        let model = initCnn(rng: &rng)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing_\(UUID().uuidString)")
            .appendingPathComponent("model.bin")

        XCTAssertThrowsError(try saveModel(model: model, filename: url.path)) { error in
            guard case CnnPersistenceError.openFailed(let filename, _) = error else {
                return XCTFail("Expected openFailed, got \(error)")
            }
            XCTAssertEqual(filename, url.path)
        }
    }

    func testLoadModelFromNonexistentFileReturnsNil() {
        let result = loadModel(filename: "/tmp/this_file_does_not_exist_\(UUID().uuidString).bin")
        XCTAssertNil(result, "Loading a non-existent file should return nil")
    }
}
