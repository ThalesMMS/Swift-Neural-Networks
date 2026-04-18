import XCTest
import MNISTCommon
@testable import MNISTManualAttention

final class MNISTManualAttentionTests: XCTestCase {
    private let sentinel: Float = -12_345.0

    private func assertNoSentinel(
        _ values: [Float],
        _ name: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(values.contains(sentinel), "\(name) still contains sentinel values", file: file, line: line)
    }

    private func assertFloatArraysEqual(
        _ actual: [Float],
        _ expected: [Float],
        name: String,
        accuracy: Float = 1e-6,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, "\(name) count mismatch", file: file, line: line)
        for (index, pair) in zip(actual, expected).enumerated() {
            XCTAssertEqual(pair.0, pair.1, accuracy: accuracy, "\(name)[\(index)] mismatch", file: file, line: line)
        }
    }

    private func zeroModel() -> AttnModel {
        AttnModel(
            wPatch: [Float](repeating: 0, count: patchDim * dModel),
            bPatch: [Float](repeating: 0, count: dModel),
            pos: [Float](repeating: 0, count: seqLen * dModel),
            wQ: [Float](repeating: 0, count: dModel * dModel),
            bQ: [Float](repeating: 0, count: dModel),
            wK: [Float](repeating: 0, count: dModel * dModel),
            bK: [Float](repeating: 0, count: dModel),
            wV: [Float](repeating: 0, count: dModel * dModel),
            bV: [Float](repeating: 0, count: dModel),
            wFf1: [Float](repeating: 0, count: dModel * ffDim),
            bFf1: [Float](repeating: 0, count: ffDim),
            wFf2: [Float](repeating: 0, count: ffDim * dModel),
            bFf2: [Float](repeating: 0, count: dModel),
            wCls: [Float](repeating: 0, count: dModel * numClasses),
            bCls: [Float](repeating: 0, count: numClasses)
        )
    }

    // MARK: - Math: relu

    func test_relu_positive_passthrough() {
        XCTAssertEqual(relu(1.0), 1.0, accuracy: 1e-9)
        XCTAssertEqual(relu(100.0), 100.0, accuracy: 1e-9)
        XCTAssertEqual(relu(0.001), 0.001, accuracy: 1e-9)
    }

    func test_relu_negative_clamped_to_zero() {
        XCTAssertEqual(relu(-1.0), 0.0)
        XCTAssertEqual(relu(-100.0), 0.0)
        XCTAssertEqual(relu(-0.001), 0.0)
    }

    func test_relu_zero_returns_zero() {
        XCTAssertEqual(relu(0.0), 0.0)
    }

    func test_relu_very_large_values() {
        XCTAssertTrue(relu(Float.greatestFiniteMagnitude).isFinite,
                      "relu of very large positive should be finite")
        XCTAssertEqual(relu(-Float.greatestFiniteMagnitude), 0.0,
                       "relu of very large negative should be 0")
    }

    // MARK: - Math: softmaxInPlace1D

    func test_softmax_in_place_1d_sums_to_one() {
        var data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        softmaxInPlace1D(&data, base: 0, length: data.count)

        let sum = data.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-6,
                       "softmaxInPlace1D output should sum to 1")
    }

    func test_softmax_in_place_1d_all_in_range() {
        var data: [Float] = [-3.0, 0.0, 2.0, -1.5, 4.0]
        softmaxInPlace1D(&data, base: 0, length: data.count)

        for v in data {
            XCTAssertGreaterThanOrEqual(v, 0.0, "Softmax values must be >= 0")
            XCTAssertLessThanOrEqual(v, 1.0, "Softmax values must be <= 1")
        }
    }

    func test_softmax_in_place_1d_single_element() {
        var data: [Float] = [0.0, 99.9, 0.0]
        // Apply only to element at index 1 (length=1).
        softmaxInPlace1D(&data, base: 1, length: 1)

        XCTAssertEqual(data[1], 1.0, accuracy: 1e-6,
                       "Single-element softmax should equal 1.0")
        // Surrounding elements must not be touched.
        XCTAssertEqual(data[0], 0.0)
        XCTAssertEqual(data[2], 0.0)
    }

    func test_softmax_in_place_1d_empty_slice_leaves_data_unchanged() {
        var data: [Float] = [1.0, 2.0, 3.0]
        softmaxInPlace1D(&data, base: 0, length: 0)
        XCTAssertEqual(data, [1.0, 2.0, 3.0])
    }

    func test_softmax_in_place_1d_with_non_zero_base() {
        // First three elements are a header that must not be touched.
        var data: [Float] = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0]
        softmaxInPlace1D(&data, base: 3, length: 3)

        // Header untouched.
        XCTAssertEqual(data[0], 10.0)
        XCTAssertEqual(data[1], 20.0)
        XCTAssertEqual(data[2], 30.0)

        // Softmax slice sums to 1.
        let softmaxSlice = Array(data[3...5])
        let sum = softmaxSlice.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-6)
        for v in softmaxSlice {
            XCTAssertGreaterThanOrEqual(v, 0.0)
            XCTAssertLessThanOrEqual(v, 1.0)
        }
    }

    func test_softmax_in_place_1d_numerically_stable() {
        // Large values should not produce NaN or Inf.
        var data: [Float] = [1000.0, 1001.0, 1002.0]
        softmaxInPlace1D(&data, base: 0, length: 3)

        XCTAssertTrue(data.allSatisfy(\.isFinite),
                      "Large inputs must not produce Inf/NaN")
        XCTAssertEqual(data.reduce(0, +), 1.0, accuracy: 1e-5)
    }

    func test_softmax_in_place_1d_max_element_wins() {
        var data: [Float] = [0.0, 10.0, 0.0]
        softmaxInPlace1D(&data, base: 0, length: 3)
        XCTAssertGreaterThan(data[1], data[0],
                             "Max-input element should have highest softmax value")
        XCTAssertGreaterThan(data[1], data[2])
    }

    // MARK: - Patch Extraction

    func test_patch_extraction_uses_expected_order() {
        let image = (0..<numInputs).map(Float.init)
        var patches = [Float](repeating: sentinel, count: seqLen * patchDim)

        extractPatches(batchInputs: image, batchCount: 1, patchesOut: &patches)

        assertNoSentinel(patches, "patches")
        let firstPatch = Array(patches[0..<patchDim])
        XCTAssertEqual(firstPatch, [
            0, 1, 2, 3,
            28, 29, 30, 31,
            56, 57, 58, 59,
            84, 85, 86, 87,
        ])
    }

    func test_patch_extraction_single_image_overwrites_expected_values() {
        let image = (0..<numInputs).map(Float.init)
        var patches = [Float](repeating: sentinel, count: seqLen * patchDim)

        extractPatches(batchInputs: image, batchCount: 1, patchesOut: &patches)

        assertNoSentinel(patches, "single-image patches")
        XCTAssertEqual(Array(patches[0..<patchDim]), [
            0, 1, 2, 3,
            28, 29, 30, 31,
            56, 57, 58, 59,
            84, 85, 86, 87,
        ])
        let lastPatchStart = (seqLen - 1) * patchDim
        XCTAssertEqual(Array(patches[lastPatchStart..<(lastPatchStart + patchDim)]), [
            696, 697, 698, 699,
            724, 725, 726, 727,
            752, 753, 754, 755,
            780, 781, 782, 783,
        ])
    }

    func test_patch_extraction_batch_overwrites_expected_values() {
        let batchCount = 3
        let images = (0..<(batchCount * numInputs)).map { index in
            Float((index / numInputs) * 1_000 + (index % numInputs))
        }
        var patches = [Float](repeating: sentinel, count: batchCount * seqLen * patchDim)

        extractPatches(batchInputs: images, batchCount: batchCount, patchesOut: &patches)

        assertNoSentinel(patches, "batch patches")
        for b in 0..<batchCount {
            let patchBase = b * seqLen * patchDim
            let valueBase = Float(b * 1_000)
            XCTAssertEqual(Array(patches[patchBase..<(patchBase + patchDim)]), [
                valueBase + 0, valueBase + 1, valueBase + 2, valueBase + 3,
                valueBase + 28, valueBase + 29, valueBase + 30, valueBase + 31,
                valueBase + 56, valueBase + 57, valueBase + 58, valueBase + 59,
                valueBase + 84, valueBase + 85, valueBase + 86, valueBase + 87,
            ])
            let lastPatchBase = patchBase + (seqLen - 1) * patchDim
            XCTAssertEqual(Array(patches[lastPatchBase..<(lastPatchBase + patchDim)]), [
                valueBase + 696, valueBase + 697, valueBase + 698, valueBase + 699,
                valueBase + 724, valueBase + 725, valueBase + 726, valueBase + 727,
                valueBase + 752, valueBase + 753, valueBase + 754, valueBase + 755,
                valueBase + 780, valueBase + 781, valueBase + 782, valueBase + 783,
            ])
        }
    }

    func test_patch_extraction_preserves_pixel_values() {
        // Uniform image: all patches should contain the same value.
        let val: Float = 0.75
        let image = [Float](repeating: val, count: numInputs)
        var patches = [Float](repeating: sentinel, count: seqLen * patchDim)
        extractPatches(batchInputs: image, batchCount: 1, patchesOut: &patches)
        assertNoSentinel(patches, "uniform patches")
        XCTAssertTrue(patches.allSatisfy { $0 == val },
                      "Uniform input should produce patches with the same constant value")
    }

    func test_patch_extraction_coverage_is_complete() {
        // Each pixel of a sequential image should appear exactly once across all patches.
        let image = (0..<numInputs).map(Float.init)
        var patches = [Float](repeating: sentinel, count: seqLen * patchDim)
        extractPatches(batchInputs: image, batchCount: 1, patchesOut: &patches)

        assertNoSentinel(patches, "coverage patches")
        XCTAssertEqual(patches.sorted(), image.sorted(),
                       "Patches must contain each source pixel exactly once")
    }

    // MARK: - Token Embeddings and Mean Pooling

    func test_token_and_pooling_writes_expected_means() {
        var rng = SimpleRng(seed: 11)
        let model = initModel(rng: &rng)
        let patches = [Float](repeating: 1, count: seqLen * patchDim)
        var tokens = [Float](repeating: sentinel, count: seqLen * dModel)
        var pooled = [Float](repeating: sentinel, count: dModel)

        makeTokens(model: model, batchCount: 1, patches: patches, tokens: &tokens)
        meanPoolTokens(batchCount: 1, tokens: tokens, pooled: &pooled)

        assertNoSentinel(tokens, "tokens")
        assertNoSentinel(pooled, "pooled")
        XCTAssertTrue(tokens.allSatisfy(\.isFinite))
        XCTAssertTrue(pooled.allSatisfy(\.isFinite))
        XCTAssertFalse(pooled.allSatisfy { $0 == 0 })
        for d in 0..<dModel {
            var expected: Float = 0
            let invSeq: Float = 1.0 / Float(seqLen)
            for t in 0..<seqLen {
                expected += tokens[t * dModel + d] * invSeq
            }
            XCTAssertEqual(pooled[d], expected, accuracy: 1e-6, "pooled[\(d)] must equal the token mean")
        }
    }

    func test_make_tokens_applies_relu() {
        // ReLU is applied inside makeTokens; output values must be >= 0.
        var rng = SimpleRng(seed: 50)
        let model = initModel(rng: &rng)
        let patches = [Float](repeating: 0.5, count: seqLen * patchDim)
        var tokens = [Float](repeating: sentinel, count: seqLen * dModel)

        makeTokens(model: model, batchCount: 1, patches: patches, tokens: &tokens)

        assertNoSentinel(tokens, "tokens")
        XCTAssertTrue(tokens.allSatisfy { $0 >= 0 },
                      "makeTokens applies ReLU so all token values must be >= 0")
    }

    func test_mean_pool_tokens_uniform_input() {
        // If all tokens are identical, mean pool output must equal that token.
        let tokenVal: Float = 0.5
        let tokens = [Float](repeating: tokenVal, count: seqLen * dModel)
        var pooled = [Float](repeating: sentinel, count: dModel)

        meanPoolTokens(batchCount: 1, tokens: tokens, pooled: &pooled)

        assertNoSentinel(pooled, "uniform pooled")
        for (i, v) in pooled.enumerated() {
            XCTAssertEqual(v, tokenVal, accuracy: 1e-6,
                           "pooled[\(i)] should equal tokenVal for uniform input")
        }
    }

    func test_mean_pool_tokens_batch_writes_expected_means() {
        let batchCount = 4
        var tokens = [Float](repeating: 0, count: batchCount * seqLen * dModel)
        for b in 0..<batchCount {
            for t in 0..<seqLen {
                for d in 0..<dModel {
                    tokens[(b * seqLen + t) * dModel + d] = Float(b * 100 + t + d)
                }
            }
        }
        var pooled = [Float](repeating: sentinel, count: batchCount * dModel)

        meanPoolTokens(batchCount: batchCount, tokens: tokens, pooled: &pooled)

        assertNoSentinel(pooled, "batch pooled")
        XCTAssertTrue(pooled.allSatisfy(\.isFinite),
                      "All pooled values must be finite")
        for b in 0..<batchCount {
            for d in 0..<dModel {
                var expected: Float = 0
                let invSeq: Float = 1.0 / Float(seqLen)
                for t in 0..<seqLen {
                    expected += tokens[(b * seqLen + t) * dModel + d] * invSeq
                }
                XCTAssertEqual(pooled[b * dModel + d], expected, accuracy: 1e-5,
                               "pooled[\(b), \(d)] must equal the token mean")
            }
        }
    }

    // MARK: - Classifier Forward

    func test_classifier_produces_probability_rows() {
        var rng = SimpleRng(seed: 12)
        let model = initModel(rng: &rng)
        let pooled = [Float](repeating: 0.5, count: 2 * dModel)
        var logits = [Float](repeating: 0, count: 2 * numClasses)
        var probs = [Float](repeating: 0, count: 2 * numClasses)

        classifierForward(model: model, batchCount: 2, pooled: pooled, logits: &logits, probs: &probs)

        for row in 0..<2 {
            let start = row * numClasses
            let sum = probs[start..<(start + numClasses)].reduce(Float(0), +)
            XCTAssertEqual(sum, 1.0, accuracy: 1e-5)
        }
        XCTAssertTrue(logits.allSatisfy(\.isFinite))
        XCTAssertTrue(probs.allSatisfy(\.isFinite))
    }

    func test_classifier_probabilities_all_in_range() {
        var rng = SimpleRng(seed: 60)
        let model = initModel(rng: &rng)
        let pooled = [Float](repeating: 0.3, count: dModel)
        var logits = [Float](repeating: 0, count: numClasses)
        var probs = [Float](repeating: 0, count: numClasses)

        classifierForward(model: model, batchCount: 1, pooled: pooled, logits: &logits, probs: &probs)

        for p in probs {
            XCTAssertGreaterThanOrEqual(p, 0.0, "Probability must be >= 0")
            XCTAssertLessThanOrEqual(p, 1.0, "Probability must be <= 1")
        }
    }

    func test_classifier_logits_are_different_from_probs() {
        // logits and probs should differ (probs are softmax of logits).
        var rng = SimpleRng(seed: 61)
        let model = initModel(rng: &rng)
        let pooled = [Float](repeating: 1.0, count: dModel)
        var logits = [Float](repeating: 0, count: numClasses)
        var probs = [Float](repeating: 0, count: numClasses)

        classifierForward(model: model, batchCount: 1, pooled: pooled, logits: &logits, probs: &probs)

        XCTAssertNotEqual(logits, probs,
                          "Logits and probabilities should differ (probs are softmax of logits)")
    }

    // MARK: - Self Attention

    func test_self_attention_known_fixture_matches_hand_computed_outputs() {
        let q: [Float] = [
            1, 2, 3,
            4, 5, 6,
        ]
        let k: [Float] = [
            7, 8, 9,
            10, 11, 12,
        ]
        var scores = [Float](repeating: sentinel, count: 4)

        computeAttentionScoresVDSP(q: q, k: k, scores: &scores, batchCount: 1, seqLen: 2, dModel: 3)

        assertFloatArraysEqual(scores, [50, 68, 122, 167], name: "scores")

        var model = zeroModel()
        model.wQ[0 * dModel + 0] = 1
        model.wQ[1 * dModel + 1] = 1
        model.wK[0 * dModel + 0] = 1
        model.wK[1 * dModel + 1] = 1
        model.wV[0 * dModel + 0] = 1
        model.wV[1 * dModel + 1] = 1

        var tokens = [Float](repeating: 0, count: seqLen * dModel)
        tokens[0 * dModel + 0] = 1
        tokens[1 * dModel + 1] = 1

        var query = [Float](repeating: sentinel, count: seqLen * dModel)
        var key = [Float](repeating: sentinel, count: seqLen * dModel)
        var value = [Float](repeating: sentinel, count: seqLen * dModel)
        var attn = [Float](repeating: sentinel, count: seqLen * seqLen)
        var attnOut = [Float](repeating: sentinel, count: seqLen * dModel)

        selfAttention(model: model, batchCount: 1, tokens: tokens, q: &query, k: &key, v: &value, attn: &attn, attnOut: &attnOut)

        assertNoSentinel(query, "query")
        assertNoSentinel(key, "key")
        assertNoSentinel(value, "value")
        assertNoSentinel(attn, "attn")
        assertNoSentinel(attnOut, "attnOut")

        let scaledMatch = Float(exp(1.0 / sqrt(Double(dModel))))
        let matchingDenom = scaledMatch + Float(seqLen - 1)
        let matchingWeight = scaledMatch / matchingDenom
        let nonMatchingWeight = Float(1.0) / matchingDenom
        let uniformWeight = Float(1.0) / Float(seqLen)

        XCTAssertEqual(attn[0 * seqLen + 0], matchingWeight, accuracy: 1e-6)
        XCTAssertEqual(attn[0 * seqLen + 1], nonMatchingWeight, accuracy: 1e-6)
        XCTAssertEqual(attn[1 * seqLen + 0], nonMatchingWeight, accuracy: 1e-6)
        XCTAssertEqual(attn[1 * seqLen + 1], matchingWeight, accuracy: 1e-6)

        XCTAssertEqual(attnOut[0 * dModel + 0], matchingWeight, accuracy: 1e-6)
        XCTAssertEqual(attnOut[0 * dModel + 1], nonMatchingWeight, accuracy: 1e-6)
        XCTAssertEqual(attnOut[1 * dModel + 0], nonMatchingWeight, accuracy: 1e-6)
        XCTAssertEqual(attnOut[1 * dModel + 1], matchingWeight, accuracy: 1e-6)
        XCTAssertEqual(attnOut[2 * dModel + 0], uniformWeight, accuracy: 1e-6)
        XCTAssertEqual(attnOut[2 * dModel + 1], uniformWeight, accuracy: 1e-6)

        for token in 0..<seqLen {
            for dimension in 2..<dModel {
                XCTAssertEqual(attnOut[token * dModel + dimension], 0, accuracy: 1e-6)
            }
        }
    }

    // MARK: - Feed-Forward Network

    func test_feed_forward_batch_writes_expected_outputs() {
        var model = zeroModel()
        model.wFf1[0 * ffDim + 0] = 1.0
        model.wFf1[1 * ffDim + 1] = -1.0
        model.bFf1[0] = 0.5
        model.wFf2[0 * dModel + 0] = 2.0
        model.wFf2[1 * dModel + 0] = -1.0
        model.wFf2[0 * dModel + 1] = 1.0
        model.bFf2[0] = 0.25
        model.bFf2[1] = -0.25

        let batchCount = 2
        var attnOut = [Float](repeating: 0, count: batchCount * seqLen * dModel)
        for b in 0..<batchCount {
            for t in 0..<seqLen {
                let base = (b * seqLen + t) * dModel
                attnOut[base + 0] = Float(b + 1)
                attnOut[base + 1] = -Float(b + 2)
            }
        }
        var ffn1 = [Float](repeating: sentinel, count: batchCount * seqLen * ffDim)
        var ffn2 = [Float](repeating: sentinel, count: batchCount * seqLen * dModel)

        feedForward(model: model, batchCount: batchCount, attnOut: attnOut, ffn1: &ffn1, ffn2: &ffn2)

        assertNoSentinel(ffn1, "ffn1")
        assertNoSentinel(ffn2, "ffn2")
        XCTAssertTrue(ffn1.allSatisfy(\.isFinite), "ffn1 must be finite")
        XCTAssertTrue(ffn2.allSatisfy(\.isFinite), "ffn2 must be finite")
        for b in 0..<batchCount {
            let expectedHidden0 = Float(b + 1) + 0.5
            let expectedHidden1 = Float(b + 2)
            var expectedFfn1 = [Float](repeating: 0, count: ffDim)
            expectedFfn1[0] = expectedHidden0
            expectedFfn1[1] = expectedHidden1

            var expectedFfn2 = [Float](repeating: 0, count: dModel)
            expectedFfn2[0] = 2 * expectedHidden0 - expectedHidden1 + 0.25
            expectedFfn2[1] = expectedHidden0 - 0.25

            for t in 0..<seqLen {
                let f1Base = (b * seqLen + t) * ffDim
                let f2Base = (b * seqLen + t) * dModel
                assertFloatArraysEqual(Array(ffn1[f1Base..<(f1Base + ffDim)]), expectedFfn1, name: "ffn1[\(b), \(t)]")
                assertFloatArraysEqual(Array(ffn2[f2Base..<(f2Base + dModel)]), expectedFfn2, name: "ffn2[\(b), \(t)]")
            }
        }
    }

    func test_feed_forward_single_item_clamps_hidden_layer() {
        // The first hidden layer applies ReLU: no negative values should appear in ffn1.
        var model = zeroModel()
        model.wFf1[0 * ffDim + 0] = -1.0
        let batchCount = 1
        var attnOut = [Float](repeating: 0, count: batchCount * seqLen * dModel)
        for t in 0..<seqLen {
            attnOut[t * dModel] = 2.0
        }
        var ffn1 = [Float](repeating: sentinel, count: batchCount * seqLen * ffDim)
        var ffn2 = [Float](repeating: sentinel, count: batchCount * seqLen * dModel)

        feedForward(model: model, batchCount: batchCount, attnOut: attnOut, ffn1: &ffn1, ffn2: &ffn2)

        assertNoSentinel(ffn1, "single-item ffn1")
        assertNoSentinel(ffn2, "single-item ffn2")
        XCTAssertTrue(ffn1.allSatisfy { $0 >= 0 },
                      "ReLU in feed-forward hidden layer must clamp negatives to zero")
        XCTAssertEqual(ffn1, [Float](repeating: 0, count: seqLen * ffDim))
        XCTAssertEqual(ffn2, [Float](repeating: 0, count: seqLen * dModel))
    }

    // MARK: - Model Initialization

    func test_init_model_dimensions() {
        var rng = SimpleRng(seed: 80)
        let model = initModel(rng: &rng)

        XCTAssertEqual(model.wPatch.count, patchDim * dModel)
        XCTAssertEqual(model.bPatch.count, dModel)
        XCTAssertEqual(model.pos.count, seqLen * dModel)
        XCTAssertEqual(model.wQ.count, dModel * dModel)
        XCTAssertEqual(model.bQ.count, dModel)
        XCTAssertEqual(model.wK.count, dModel * dModel)
        XCTAssertEqual(model.bK.count, dModel)
        XCTAssertEqual(model.wV.count, dModel * dModel)
        XCTAssertEqual(model.bV.count, dModel)
        XCTAssertEqual(model.wFf1.count, dModel * ffDim)
        XCTAssertEqual(model.bFf1.count, ffDim)
        XCTAssertEqual(model.wFf2.count, ffDim * dModel)
        XCTAssertEqual(model.bFf2.count, dModel)
        XCTAssertEqual(model.wCls.count, dModel * numClasses)
        XCTAssertEqual(model.bCls.count, numClasses)
    }

    func test_init_model_weights_are_finite() {
        var rng = SimpleRng(seed: 81)
        let model = initModel(rng: &rng)

        XCTAssertTrue(model.wPatch.allSatisfy(\.isFinite), "wPatch must be finite")
        XCTAssertTrue(model.wQ.allSatisfy(\.isFinite), "wQ must be finite")
        XCTAssertTrue(model.wK.allSatisfy(\.isFinite), "wK must be finite")
        XCTAssertTrue(model.wV.allSatisfy(\.isFinite), "wV must be finite")
        XCTAssertTrue(model.wFf1.allSatisfy(\.isFinite), "wFf1 must be finite")
        XCTAssertTrue(model.wFf2.allSatisfy(\.isFinite), "wFf2 must be finite")
        XCTAssertTrue(model.wCls.allSatisfy(\.isFinite), "wCls must be finite")
        XCTAssertTrue(model.pos.allSatisfy(\.isFinite), "pos must be finite")
    }

    func test_init_model_biases_are_zero() {
        var rng = SimpleRng(seed: 82)
        let model = initModel(rng: &rng)

        XCTAssertTrue(model.bPatch.allSatisfy { $0 == 0 }, "bPatch must be zero-initialized")
        XCTAssertTrue(model.bQ.allSatisfy { $0 == 0 }, "bQ must be zero-initialized")
        XCTAssertTrue(model.bK.allSatisfy { $0 == 0 }, "bK must be zero-initialized")
        XCTAssertTrue(model.bV.allSatisfy { $0 == 0 }, "bV must be zero-initialized")
        XCTAssertTrue(model.bFf1.allSatisfy { $0 == 0 }, "bFf1 must be zero-initialized")
        XCTAssertTrue(model.bFf2.allSatisfy { $0 == 0 }, "bFf2 must be zero-initialized")
        XCTAssertTrue(model.bCls.allSatisfy { $0 == 0 }, "bCls must be zero-initialized")
    }

    func test_init_model_reproducibility() {
        var rng1 = SimpleRng(seed: 83)
        var rng2 = SimpleRng(seed: 83)
        let model1 = initModel(rng: &rng1)
        let model2 = initModel(rng: &rng2)

        XCTAssertEqual(model1.wPatch, model2.wPatch,
                       "Same seed should produce identical wPatch")
        XCTAssertEqual(model1.wQ, model2.wQ,
                       "Same seed should produce identical wQ")
        XCTAssertEqual(model1.wCls, model2.wCls,
                       "Same seed should produce identical wCls")
    }

    func test_init_model_different_seeds() {
        var rng1 = SimpleRng(seed: 1)
        var rng2 = SimpleRng(seed: 2)
        let model1 = initModel(rng: &rng1)
        let model2 = initModel(rng: &rng2)

        XCTAssertNotEqual(model1.wPatch, model2.wPatch,
                          "Different seeds should produce different weights")
    }

    // MARK: - Grads

    func test_grads_zero_initialization() {
        let grads = Grads()

        XCTAssertTrue(grads.wPatch.allSatisfy { $0 == 0 })
        XCTAssertTrue(grads.bPatch.allSatisfy { $0 == 0 })
        XCTAssertTrue(grads.wQ.allSatisfy { $0 == 0 })
        XCTAssertTrue(grads.wCls.allSatisfy { $0 == 0 })
    }

    func test_grads_zero_method_clears_values() {
        var grads = Grads()
        // Manually set some values.
        for i in 0..<grads.wPatch.count { grads.wPatch[i] = 1.0 }
        for i in 0..<grads.wCls.count { grads.wCls[i] = 2.0 }

        grads.zero()

        XCTAssertTrue(grads.wPatch.allSatisfy { $0 == 0 },
                      "zero() should clear wPatch")
        XCTAssertTrue(grads.wCls.allSatisfy { $0 == 0 },
                      "zero() should clear wCls")
    }

    // MARK: - Persistence

    func test_persistence_round_trip() throws {
        var rng = SimpleRng(seed: 13)
        let model = initModel(rng: &rng)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mnist_manual_attention_\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }

        try saveModel(model: model, filename: url.path)
        guard let loaded = loadModel(filename: url.path) else {
            XCTFail("loadModel returned nil")
            return
        }

        XCTAssertEqual(loaded.wPatch, model.wPatch)
        XCTAssertEqual(loaded.pos, model.pos)
        XCTAssertEqual(loaded.wQ, model.wQ)
        XCTAssertEqual(loaded.wCls, model.wCls)
    }

    func test_persistence_preserves_all_parameter_arrays() throws {
        // Every parameter array must survive the save/load cycle.
        var rng = SimpleRng(seed: 90)
        let model = initModel(rng: &rng)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mnist_attn_full_\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }

        try saveModel(model: model, filename: url.path)
        guard let loaded = loadModel(filename: url.path) else {
            XCTFail("loadModel returned nil")
            return
        }

        assertFloatArraysEqual(loaded.wPatch, model.wPatch, name: "wPatch")
        assertFloatArraysEqual(loaded.bPatch, model.bPatch, name: "bPatch")
        assertFloatArraysEqual(loaded.pos, model.pos, name: "pos")
        assertFloatArraysEqual(loaded.wQ, model.wQ, name: "wQ")
        assertFloatArraysEqual(loaded.bQ, model.bQ, name: "bQ")
        assertFloatArraysEqual(loaded.wK, model.wK, name: "wK")
        assertFloatArraysEqual(loaded.bK, model.bK, name: "bK")
        assertFloatArraysEqual(loaded.wV, model.wV, name: "wV")
        assertFloatArraysEqual(loaded.bV, model.bV, name: "bV")
        assertFloatArraysEqual(loaded.wFf1, model.wFf1, name: "wFf1")
        assertFloatArraysEqual(loaded.bFf1, model.bFf1, name: "bFf1")
        assertFloatArraysEqual(loaded.wFf2, model.wFf2, name: "wFf2")
        assertFloatArraysEqual(loaded.bFf2, model.bFf2, name: "bFf2")
        assertFloatArraysEqual(loaded.wCls, model.wCls, name: "wCls")
        assertFloatArraysEqual(loaded.bCls, model.bCls, name: "bCls")
    }

    func test_persistence_save_to_missing_directory_throws() {
        var rng = SimpleRng(seed: 91)
        let model = initModel(rng: &rng)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing_\(UUID().uuidString)")
            .appendingPathComponent("model.bin")

        XCTAssertThrowsError(try saveModel(model: model, filename: url.path)) { error in
            guard case AttentionPersistenceError.openFailed(let filename, _) = error else {
                return XCTFail("Expected openFailed, got \(error)")
            }
            XCTAssertEqual(filename, url.path)
        }
    }

    func test_load_model_from_nonexistent_file_returns_nil() {
        let result = loadModel(filename: "/tmp/nonexistent_attention_\(UUID().uuidString).bin")
        XCTAssertNil(result, "loadModel from nonexistent path should return nil")
    }

    // MARK: - Config Defaults

    func test_attention_config_default_values() {
        let config = Config()
        XCTAssertEqual(config.learningRate, 0.005, accuracy: 1e-9)
        XCTAssertEqual(config.epochs, 5)
        XCTAssertEqual(config.batchSize, 32)
        XCTAssertEqual(config.dataPath, "./data")
        XCTAssertEqual(config.seed, 1)
    }
}
