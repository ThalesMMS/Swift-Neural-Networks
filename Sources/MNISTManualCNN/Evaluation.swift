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
// MARK: - Model Evaluation
// =============================================================================

/// Computes the classification accuracy of `model` on the provided dataset.
/// 
/// The `images` array must contain flattened input vectors for all samples (length == `labels.count * numInputs`). Each entry in `labels` is the ground-truth class index as a `UInt8`. The function evaluates the model in batches of up to `batchSize` and compares the predicted class index for each sample against the corresponding label.
/// - Parameters:
///   - model: The convolutional neural network to evaluate.
///   - images: Flattened input features for all samples concatenated sequentially.
///   - labels: Ground-truth class indices for each sample.
///   - batchSize: Maximum number of samples processed per evaluation batch.
/// - Returns: The classification accuracy as a percentage (0.0 to 100.0).
func testAccuracy(model: Cnn, images: [Float], labels: [UInt8], batchSize: Int) -> Float {
    let n = labels.count
    var correct = 0

    var batchInputs = [Float](repeating: 0, count: batchSize * numInputs)
    var convAct = [Float](repeating: 0, count: batchSize * convOut * imgH * imgW)
    var poolOut = [Float](repeating: 0, count: batchSize * fcIn)
    var poolIdx = [UInt8](repeating: 0, count: batchSize * convOut * poolH * poolW)
    var logits = [Float](repeating: 0, count: batchSize * numClasses)

    var start = 0
    while start < n {
        let bsz = min(batchSize, n - start)
        let len = bsz * numInputs
        let srcStart = start * numInputs
        for i in 0..<len {
            batchInputs[i] = images[srcStart + i]
        }

        convForwardRelu(model: model, batch: bsz, input: batchInputs, convOutAct: &convAct)
        maxPoolForward(batch: bsz, convAct: convAct, poolOut: &poolOut, poolIdx: &poolIdx)
        fcForward(model: model, batch: bsz, x: poolOut, logits: &logits)

        for b in 0..<bsz {
            let base = b * numClasses
            var best = logits[base]
            var arg = 0
            for j in 1..<numClasses {
                let v = logits[base + j]
                if v > best { best = v; arg = j }
            }
            if UInt8(arg) == labels[start + b] { correct += 1 }
        }

        start += bsz
    }

    return 100.0 * Float(correct) / Float(n)
}
