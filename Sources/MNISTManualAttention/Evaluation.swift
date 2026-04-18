import Foundation
import MNISTCommon

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Computes classification accuracy of an attention model over a dataset using batched inference.
/// - Parameters:
///   - model: The `AttnModel` to evaluate.
///   - images: Flattened input images, stored contiguously with `numInputs` floats per example.
///   - labels: Ground-truth class indices for each example; length must equal the number of images.
///   - config: Evaluation configuration (uses `config.batchSize` to control batch processing).
/// - Returns: Accuracy as a percentage (0.0 to 100.0) of correctly predicted labels.
func testAccuracy(model: AttnModel, images: [Float], labels: [UInt8], config: Config) -> Float {
    let n = labels.count
    let batchSize = config.batchSize
    precondition(batchSize > 0, "batchSize must be > 0 before the evaluation minibatch loop so start advances by bsz")

    var batchInputs = [Float](repeating: 0, count: batchSize * numInputs)
    var patches = [Float](repeating: 0, count: batchSize * seqLen * patchDim)
    var tokens  = [Float](repeating: 0, count: batchSize * seqLen * dModel)
    var q       = [Float](repeating: 0, count: batchSize * seqLen * dModel)
    var k       = [Float](repeating: 0, count: batchSize * seqLen * dModel)
    var v       = [Float](repeating: 0, count: batchSize * seqLen * dModel)
    var attn    = [Float](repeating: 0, count: batchSize * seqLen * seqLen)
    var attnOut = [Float](repeating: 0, count: batchSize * seqLen * dModel)
    var ffn1    = [Float](repeating: 0, count: batchSize * seqLen * ffDim)
    var ffn2    = [Float](repeating: 0, count: batchSize * seqLen * dModel)
    var pooled  = [Float](repeating: 0, count: batchSize * dModel)
    var logits  = [Float](repeating: 0, count: batchSize * numClasses)
    var probs   = [Float](repeating: 0, count: batchSize * numClasses)

    var correct = 0

    var start = 0
    while start < n {
        let bsz = min(batchSize, n - start)

        // Contiguous batch copy.
        let srcBase = start * numInputs
        let len = bsz * numInputs
        for i in 0..<len { batchInputs[i] = images[srcBase + i] }

        extractPatches(batchInputs: batchInputs, batchCount: bsz, patchesOut: &patches)
        makeTokens(model: model, batchCount: bsz, patches: patches, tokens: &tokens)
        selfAttention(model: model, batchCount: bsz, tokens: tokens, q: &q, k: &k, v: &v, attn: &attn, attnOut: &attnOut)
        feedForward(model: model, batchCount: bsz, attnOut: attnOut, ffn1: &ffn1, ffn2: &ffn2)
        meanPoolTokens(batchCount: bsz, tokens: ffn2, pooled: &pooled)
        classifierForward(model: model, batchCount: bsz, pooled: pooled, logits: &logits, probs: &probs)

        for b in 0..<bsz {
            let base = b * numClasses
            var best = logits[base]
            var arg = 0
            for c in 1..<numClasses {
                let v = logits[base + c]
                if v > best { best = v; arg = c }
            }
            if UInt8(arg) == labels[start + b] { correct += 1 }
        }

        start += bsz
    }

    return 100.0 * Float(correct) / Float(n)
}
