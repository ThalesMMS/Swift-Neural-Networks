// ============================================================================
// EDUCATIONAL REFERENCE: Manual self-attention implementation for learning purposes
//
// This is a standalone educational example demonstrating self-attention mechanisms
// with manual backpropagation. It implements a simplified Transformer-style
// architecture for MNIST classification using patch-based tokenization.
//
// For production use, see: swift run MNISTMLX
// For learning progression, see: LEARNING_GUIDE.md
// ============================================================================
//
// mnist_attention_pool.swift
// Self-attention over patch tokens for MNIST (single-head Transformer-style).
//
// Model (educational):
//   1) 4x4 patches => 49 tokens.
//   2) token_t = ReLU(patch_t * Wpatch + bpatch + pos_t)
//   3) Self-attention (1 head): Q/K/V with a 49x49 score matrix per sample.
//   4) Feed-forward MLP per token (D -> FF -> D).
//   5) Mean-pool tokens -> logits -> softmax cross-entropy.
//
// Training: SGD + minibatches.
//
// Notes:
// - Intentionally simple for study.
// - For faster runs, reduce trainSamples/epochs.

import Foundation
import Accelerate

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

// MNIST constants (images are flat 28x28 in row-major order).
let imgH = 28
let imgW = 28
let numInputs = imgH * imgW
let numClasses = 10

// Patch grid and tokenization.
let patch = 4
let grid = imgH / patch          // 7
let seqLen = grid * grid         // 49
let patchDim = patch * patch     // 16
let dModel = 32                  // model dimension (increased for capacity test)
let ffDim = 64                   // feed-forward hidden size (2x dModel)

// Dataset sizes.
let trainSamples = 60_000
let testSamples = 10_000

// =============================================================================
// MARK: - Configuration
// =============================================================================

/// Configuration for training hyperparameters
struct Config {
    var learningRate: Float = 0.005
    var epochs: Int = 5
    var batchSize: Int = 32
    var dataPath: String = "./data"
    var seed: UInt64 = 1

    /// Parses command-line arguments into configuration
    ///
    /// This is a simple hand-rolled parser. For production code,
    /// Parse command-line arguments and produce a Config with any provided overrides.
    /// Recognized options: `--batch`/`-b` <int>, `--epochs`/`-e` <int>, `--lr`/`-l` <float>, `--seed`/`-s` <uint64>, and `--help`/`-h` (prints usage and exits).
    /// - Returns: A `Config` populated with values overridden by the parsed arguments; fields not specified on the command line retain their default values.
    static func parse() -> Config {
        var config = Config()
        let args = CommandLine.arguments
        var i = 1

        while i < args.count {
            let arg = args[i]

            switch arg {
            case "--batch", "-b":
                let valueIndex = i + 1
                guard valueIndex < args.count else {
                    print("Missing value for \(arg)")
                    printUsage()
                    exit(1)
                }
                let token = args[valueIndex]
                guard let val = Int(token), val > 0 else {
                    print("Invalid value for \(arg): \(token)")
                    printUsage()
                    exit(1)
                }
                config.batchSize = val
                i = valueIndex

            case "--epochs", "-e":
                let valueIndex = i + 1
                guard valueIndex < args.count else {
                    print("Missing value for \(arg)")
                    printUsage()
                    exit(1)
                }
                let token = args[valueIndex]
                guard let val = Int(token), val > 0 else {
                    print("Invalid value for \(arg): \(token)")
                    printUsage()
                    exit(1)
                }
                config.epochs = val
                i = valueIndex

            case "--lr", "-l":
                let valueIndex = i + 1
                guard valueIndex < args.count else {
                    print("Missing value for \(arg)")
                    printUsage()
                    exit(1)
                }
                let token = args[valueIndex]
                guard let val = Float(token), val > 0 else {
                    print("Invalid value for \(arg): \(token)")
                    printUsage()
                    exit(1)
                }
                config.learningRate = val
                i = valueIndex

            case "--data", "-d":
                let valueIndex = i + 1
                guard valueIndex < args.count else {
                    print("Missing value for \(arg)")
                    printUsage()
                    exit(1)
                }
                config.dataPath = args[valueIndex]
                i = valueIndex

            case "--seed", "-s":
                let valueIndex = i + 1
                guard valueIndex < args.count else {
                    print("Missing value for \(arg)")
                    printUsage()
                    exit(1)
                }
                let token = args[valueIndex]
                guard let val = UInt64(token) else {
                    print("Invalid value for \(arg): \(token)")
                    printUsage()
                    exit(1)
                }
                config.seed = val
                i = valueIndex

            case "--help", "-h":
                printUsage()
                exit(0)

            default:
                print("Unknown argument: \(arg)")
                printUsage()
                exit(1)
            }

            i += 1
        }

        return config
    }
}

/// Prints the command-line usage, available options, example invocations, and a brief model architecture summary to standard output.
func printUsage() {
    print("""
    MNIST Attention Pool - Self-Attention Model for MNIST
    ======================================================

    USAGE:
      swift mnist_attention_pool.swift [OPTIONS]

    OPTIONS:
      --batch, -b <n>    Batch size (default: 32)
      --epochs, -e <n>   Number of training epochs (default: 5)
      --lr, -l <f>       Learning rate (default: 0.005)
      --data, -d <path>  Path to MNIST data directory (default: ./data)
      --seed, -s <n>     RNG seed for reproducibility (default: 1)
      --help, -h         Show this help message

    EXAMPLES:
      swift mnist_attention_pool.swift --epochs 10
      swift mnist_attention_pool.swift -b 64 -e 5 -l 0.005
      swift mnist_attention_pool.swift --data ./data --seed 42

    MODEL ARCHITECTURE:
      - 4×4 patches → 49 tokens
      - Self-attention with Q/K/V projections
      - Feed-forward MLP per token
      - Mean-pool → logits → softmax
    """)
}


// Numerically stable softmax for a slice.
func softmaxInPlace1D(_ data: inout [Float], base: Int, length: Int) {
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

func relu(_ x: Float) -> Float { x > 0 ? x : 0 }

// Model parameters stored in flat arrays.
struct AttnModel {
    var wPatch: [Float]  // [patchDim * dModel]
    var bPatch: [Float]  // [dModel]
    var pos: [Float]     // [seqLen * dModel]
    var wQ: [Float]      // [dModel * dModel]
    var bQ: [Float]      // [dModel]
    var wK: [Float]      // [dModel * dModel]
    var bK: [Float]      // [dModel]
    var wV: [Float]      // [dModel * dModel]
    var bV: [Float]      // [dModel]
    var wFf1: [Float]    // [dModel * ffDim]
    var bFf1: [Float]    // [ffDim]
    var wFf2: [Float]    // [ffDim * dModel]
    var bFf2: [Float]    // [dModel]
    var wCls: [Float]    // [dModel * numClasses]
    var bCls: [Float]    // [numClasses]
}

struct Grads {
    var wPatch: [Float]
    var bPatch: [Float]
    var pos: [Float]
    var wQ: [Float]
    var bQ: [Float]
    var wK: [Float]
    var bK: [Float]
    var wV: [Float]
    var bV: [Float]
    var wFf1: [Float]
    var bFf1: [Float]
    var wFf2: [Float]
    var bFf2: [Float]
    var wCls: [Float]
    var bCls: [Float]

    init() {
        wPatch = [Float](repeating: 0, count: patchDim * dModel)
        bPatch = [Float](repeating: 0, count: dModel)
        pos = [Float](repeating: 0, count: seqLen * dModel)
        wQ = [Float](repeating: 0, count: dModel * dModel)
        bQ = [Float](repeating: 0, count: dModel)
        wK = [Float](repeating: 0, count: dModel * dModel)
        bK = [Float](repeating: 0, count: dModel)
        wV = [Float](repeating: 0, count: dModel * dModel)
        bV = [Float](repeating: 0, count: dModel)
        wFf1 = [Float](repeating: 0, count: dModel * ffDim)
        bFf1 = [Float](repeating: 0, count: ffDim)
        wFf2 = [Float](repeating: 0, count: ffDim * dModel)
        bFf2 = [Float](repeating: 0, count: dModel)
        wCls = [Float](repeating: 0, count: dModel * numClasses)
        bCls = [Float](repeating: 0, count: numClasses)
    }

    mutating func zero() {
        for i in 0..<wPatch.count { wPatch[i] = 0 }
        for i in 0..<bPatch.count { bPatch[i] = 0 }
        for i in 0..<pos.count { pos[i] = 0 }
        for i in 0..<wQ.count { wQ[i] = 0 }
        for i in 0..<bQ.count { bQ[i] = 0 }
        for i in 0..<wK.count { wK[i] = 0 }
        for i in 0..<bK.count { bK[i] = 0 }
        for i in 0..<wV.count { wV[i] = 0 }
        for i in 0..<bV.count { bV[i] = 0 }
        for i in 0..<wFf1.count { wFf1[i] = 0 }
        for i in 0..<bFf1.count { bFf1[i] = 0 }
        for i in 0..<wFf2.count { wFf2[i] = 0 }
        for i in 0..<bFf2.count { bFf2[i] = 0 }
        for i in 0..<wCls.count { wCls[i] = 0 }
        for i in 0..<bCls.count { bCls[i] = 0 }
    }
}

func initModel(rng: inout SimpleRng) -> AttnModel {
    // Xavier init for patch projection.
    let limitPatch = sqrtf(6.0 / Float(patchDim + dModel))
    var wPatch = [Float](repeating: 0, count: patchDim * dModel)
    for i in 0..<wPatch.count { wPatch[i] = rng.uniform(-limitPatch, limitPatch) }

    let bPatch = [Float](repeating: 0, count: dModel)

    var pos = [Float](repeating: 0, count: seqLen * dModel)
    let s: Float = 0.1
    for i in 0..<pos.count { pos[i] = rng.uniform(-s, s) }

    let limitAttn = sqrtf(6.0 / Float(dModel + dModel))
    var wQ = [Float](repeating: 0, count: dModel * dModel)
    var wK = [Float](repeating: 0, count: dModel * dModel)
    var wV = [Float](repeating: 0, count: dModel * dModel)
    for i in 0..<wQ.count { wQ[i] = rng.uniform(-limitAttn, limitAttn) }
    for i in 0..<wK.count { wK[i] = rng.uniform(-limitAttn, limitAttn) }
    for i in 0..<wV.count { wV[i] = rng.uniform(-limitAttn, limitAttn) }
    let bQ = [Float](repeating: 0, count: dModel)
    let bK = [Float](repeating: 0, count: dModel)
    let bV = [Float](repeating: 0, count: dModel)

    let limitFf1 = sqrtf(6.0 / Float(dModel + ffDim))
    var wFf1 = [Float](repeating: 0, count: dModel * ffDim)
    for i in 0..<wFf1.count { wFf1[i] = rng.uniform(-limitFf1, limitFf1) }
    let bFf1 = [Float](repeating: 0, count: ffDim)

    let limitFf2 = sqrtf(6.0 / Float(ffDim + dModel))
    var wFf2 = [Float](repeating: 0, count: ffDim * dModel)
    for i in 0..<wFf2.count { wFf2[i] = rng.uniform(-limitFf2, limitFf2) }
    let bFf2 = [Float](repeating: 0, count: dModel)

    let limitCls = sqrtf(6.0 / Float(dModel + numClasses))
    var wCls = [Float](repeating: 0, count: dModel * numClasses)
    for i in 0..<wCls.count { wCls[i] = rng.uniform(-limitCls, limitCls) }
    let bCls = [Float](repeating: 0, count: numClasses)

    return AttnModel(
        wPatch: wPatch,
        bPatch: bPatch,
        pos: pos,
        wQ: wQ,
        bQ: bQ,
        wK: wK,
        bK: bK,
        wV: wV,
        bV: bV,
        wFf1: wFf1,
        bFf1: bFf1,
        wFf2: wFf2,
        bFf2: bFf2,
        wCls: wCls,
        bCls: bCls
    )
}

// Extract 4x4 patches from a contiguous batch of images.
func extractPatches(batchInputs: [Float], batchCount: Int, patchesOut: inout [Float]) {
    // patchesOut: [batchSize * seqLen * patchDim]
    for b in 0..<batchCount {
        let imgBase = b * numInputs
        for py in 0..<grid {
            for px in 0..<grid {
                let t = py * grid + px
                let pBase = (b * seqLen + t) * patchDim
                for dy in 0..<patch {
                    for dx in 0..<patch {
                        let iy = py * patch + dy
                        let ix = px * patch + dx
                        let src = imgBase + iy * imgW + ix
                        let j = dy * patch + dx
                        patchesOut[pBase + j] = batchInputs[src]
                    }
                }
            }
        }
    }
}

// Forward: build post-ReLU tokens.
func makeTokens(model: AttnModel, batchCount: Int, patches: [Float], tokens: inout [Float]) {
    // tokens: [batchSize * seqLen * dModel]
    for b in 0..<batchCount {
        for t in 0..<seqLen {
            let pBase = (b * seqLen + t) * patchDim
            let tokBase = (b * seqLen + t) * dModel
            let posBase = t * dModel
            for d in 0..<dModel {
                var sum = model.bPatch[d] + model.pos[posBase + d]
                // Linear patch projection.
                for j in 0..<patchDim {
                    sum += patches[pBase + j] * model.wPatch[j * dModel + d]
                }
                tokens[tokBase + d] = relu(sum)
            }
        }
    }
}

// Compute Q * K^T for a batch using vDSP (Accelerate).
// Q: [batchCount, seqLen, dModel]
// K: [batchCount, seqLen, dModel]
// scores: [batchCount, seqLen, seqLen] (output)
func computeAttentionScoresVDSP(
    q: [Float],
    k: [Float],
    scores: inout [Float],
    batchCount: Int,
    seqLen: Int,
    dModel: Int
) {
    // For each sample in batch, compute Q_b * K_b^T
    for b in 0..<batchCount {
        let qOffset = b * seqLen * dModel
        let kOffset = b * seqLen * dModel
        let scoresOffset = b * seqLen * seqLen

        // Transpose K from [seqLen, dModel] to [dModel, seqLen]
        var kTransposed = [Float](repeating: 0.0, count: seqLen * dModel)
        q.withUnsafeBufferPointer { qBuf in
            k.withUnsafeBufferPointer { kBuf in
                kTransposed.withUnsafeMutableBufferPointer { ktBuf in
                    guard let qPtr = qBuf.baseAddress,
                          let kPtr = kBuf.baseAddress,
                          let ktPtr = ktBuf.baseAddress else { return }

                    // Transpose: K is [seqLen, dModel] row-major -> K^T is [dModel, seqLen] row-major
                    vDSP_mtrans(
                        kPtr.advanced(by: kOffset),
                        1,
                        ktPtr,
                        1,
                        vDSP_Length(dModel),
                        vDSP_Length(seqLen)
                    )

                    // Matrix multiply: Q [seqLen, dModel] * K^T [dModel, seqLen] = scores [seqLen, seqLen]
                    scores.withUnsafeMutableBufferPointer { scoresBuf in
                        guard let scoresPtr = scoresBuf.baseAddress else { return }
                        vDSP_mmul(
                            qPtr.advanced(by: qOffset),
                            1,
                            ktPtr,
                            1,
                            scoresPtr.advanced(by: scoresOffset),
                            1,
                            vDSP_Length(seqLen),
                            vDSP_Length(seqLen),
                            vDSP_Length(dModel)
                        )
                    }
                }
            }
        }
    }
}

// Self-attention: Q/K/V -> softmax scores -> weighted sum.
func selfAttention(
    model: AttnModel,
    batchCount: Int,
    tokens: [Float],
    q: inout [Float],
    k: inout [Float],
    v: inout [Float],
    attn: inout [Float],
    attnOut: inout [Float]
) {
    let invSqrtD: Float = 1.0 / sqrtf(Float(dModel))

    for b in 0..<batchCount {
        for t in 0..<seqLen {
            let tokBase = (b * seqLen + t) * dModel
            for dOut in 0..<dModel {
                var sumQ = model.bQ[dOut]
                var sumK = model.bK[dOut]
                var sumV = model.bV[dOut]
                for dIn in 0..<dModel {
                    let x = tokens[tokBase + dIn]
                    sumQ += x * model.wQ[dIn * dModel + dOut]
                    sumK += x * model.wK[dIn * dModel + dOut]
                    sumV += x * model.wV[dIn * dModel + dOut]
                }
                q[tokBase + dOut] = sumQ
                k[tokBase + dOut] = sumK
                v[tokBase + dOut] = sumV
            }
        }
    }

    // Compute Q * K^T using vDSP for all batches
    computeAttentionScoresVDSP(
        q: q,
        k: k,
        scores: &attn,
        batchCount: batchCount,
        seqLen: seqLen,
        dModel: dModel
    )

    // Scale scores by 1/sqrt(dModel)
    var invSqrtDVar = invSqrtD
    let totalScores = batchCount * seqLen * seqLen
    vDSP_vsmul(attn, 1, &invSqrtDVar, &attn, 1, vDSP_Length(totalScores))

    // Apply softmax to each attention row
    for b in 0..<batchCount {
        for i in 0..<seqLen {
            let rowBase = (b * seqLen + i) * seqLen
            softmaxInPlace1D(&attn, base: rowBase, length: seqLen)

            let outBase = (b * seqLen + i) * dModel
            for d in 0..<dModel { attnOut[outBase + d] = 0 }
            for j in 0..<seqLen {
                let a = attn[rowBase + j]
                let vBase = (b * seqLen + j) * dModel
                for d in 0..<dModel {
                    attnOut[outBase + d] += a * v[vBase + d]
                }
            }
        }
    }
}

// Feed-forward MLP per token: D -> FF -> D.
func feedForward(
    model: AttnModel,
    batchCount: Int,
    attnOut: [Float],
    ffn1: inout [Float],
    ffn2: inout [Float]
) {
    for b in 0..<batchCount {
        for t in 0..<seqLen {
            let attnBase = (b * seqLen + t) * dModel
            let f1Base = (b * seqLen + t) * ffDim
            let f2Base = (b * seqLen + t) * dModel

            for h in 0..<ffDim {
                var sum = model.bFf1[h]
                for d in 0..<dModel {
                    sum += attnOut[attnBase + d] * model.wFf1[d * ffDim + h]
                }
                ffn1[f1Base + h] = relu(sum)
            }

            for d in 0..<dModel {
                var sum = model.bFf2[d]
                for h in 0..<ffDim {
                    sum += ffn1[f1Base + h] * model.wFf2[h * dModel + d]
                }
                ffn2[f2Base + d] = sum
            }
        }
    }
}

// Mean-pool tokens into a single vector per sample.
func meanPoolTokens(batchCount: Int, tokens: [Float], pooled: inout [Float]) {
    let invSeq: Float = 1.0 / Float(seqLen)
    for b in 0..<batchCount {
        let pBase = b * dModel
        for d in 0..<dModel { pooled[pBase + d] = 0 }
        for t in 0..<seqLen {
            let tokBase = (b * seqLen + t) * dModel
            for d in 0..<dModel {
                pooled[pBase + d] += tokens[tokBase + d] * invSeq
            }
        }
    }
}

func classifierForward(model: AttnModel, batchCount: Int, pooled: [Float], logits: inout [Float], probs: inout [Float]) {
    for b in 0..<batchCount {
        let pBase = b * dModel
        let lBase = b * numClasses
        for c in 0..<numClasses {
            var sum = model.bCls[c]
            for d in 0..<dModel {
                sum += pooled[pBase + d] * model.wCls[d * numClasses + c]
            }
            logits[lBase + c] = sum
            probs[lBase + c] = sum
        }
        // Softmax in-place on probs (row).
        softmaxInPlace1D(&probs, base: lBase, length: numClasses)
    }
}

/// Performs a single training epoch over the provided dataset and updates `model` in place using SGD.
/// - Parameters:
///   - model: The attention model to be updated.
///   - images: Flat array of input images (numSamples * numInputs).
///   - labels: Array of labels for each image.
///   - indices: Array of sample indices; will be shuffled in-place to randomize minibatch order.
///   - rng: Random number generator used for shuffling.
///   - config: Training configuration (controls batch size and learning rate).
/// - Returns: The average cross-entropy loss across all samples for this epoch.
func trainEpoch(
    model: inout AttnModel,
    images: [Float],
    labels: [UInt8],
    indices: inout [Int],
    rng: inout SimpleRng,
    config: Config
) -> Float {
    rng.shuffle(&indices)

    var grads = Grads()

    let batchSize = config.batchSize
    let learningRate = config.learningRate

    // Reusable buffers to avoid per-batch allocations.
    var batchInputs = [Float](repeating: 0, count: batchSize * numInputs)
    var batchLabels = [UInt8](repeating: 0, count: batchSize)

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

    // Backward buffers.
    var dlogits = [Float](repeating: 0, count: batchSize * numClasses)
    var dpooled = [Float](repeating: 0, count: batchSize * dModel)
    var dffn2   = [Float](repeating: 0, count: batchSize * seqLen * dModel)
    var dffn1   = [Float](repeating: 0, count: batchSize * seqLen * ffDim)
    var dattn   = [Float](repeating: 0, count: batchSize * seqLen * dModel)
    var dalpha  = [Float](repeating: 0, count: batchSize * seqLen * seqLen)
    var dscores = [Float](repeating: 0, count: batchSize * seqLen * seqLen)
    var dQ      = [Float](repeating: 0, count: batchSize * seqLen * dModel)
    var dK      = [Float](repeating: 0, count: batchSize * seqLen * dModel)
    var dV      = [Float](repeating: 0, count: batchSize * seqLen * dModel)
    var dtokens = [Float](repeating: 0, count: batchSize * seqLen * dModel)

    let n = indices.count
    var totalLoss: Float = 0

    var start = 0
    while start < n {
        let bsz = min(batchSize, n - start)
        let scale: Float = 1.0 / Float(bsz)

        // Gather mini-batch into contiguous buffers.
        for i in 0..<bsz {
            let idx = indices[start + i]
            let srcBase = idx * numInputs
            let dstBase = i * numInputs
            for j in 0..<numInputs {
                batchInputs[dstBase + j] = images[srcBase + j]
            }
            batchLabels[i] = labels[idx]
        }

        // Forward pass.
        extractPatches(batchInputs: batchInputs, batchCount: bsz, patchesOut: &patches)
        makeTokens(model: model, batchCount: bsz, patches: patches, tokens: &tokens)
        selfAttention(model: model, batchCount: bsz, tokens: tokens, q: &q, k: &k, v: &v, attn: &attn, attnOut: &attnOut)
        feedForward(model: model, batchCount: bsz, attnOut: attnOut, ffn1: &ffn1, ffn2: &ffn2)
        meanPoolTokens(batchCount: bsz, tokens: ffn2, pooled: &pooled)
        classifierForward(model: model, batchCount: bsz, pooled: pooled, logits: &logits, probs: &probs)

        // Loss + dlogits.
        for i in 0..<(bsz * numClasses) { dlogits[i] = 0 }
        for b in 0..<bsz {
            let base = b * numClasses
            let y = Int(batchLabels[b])
            let p = max(probs[base + y], 1e-9)
            totalLoss += -logf(p)
            for c in 0..<numClasses {
                var d = probs[base + c]
                if c == y { d -= 1 }
                dlogits[base + c] = d * scale
            }
        }

        // Backward: zero grads.
        grads.zero()
        for i in 0..<(bsz * dModel) { dpooled[i] = 0 }
        for i in 0..<(bsz * seqLen * dModel) {
            dffn2[i] = 0
            dattn[i] = 0
            dQ[i] = 0
            dK[i] = 0
            dV[i] = 0
            dtokens[i] = 0
        }
        for i in 0..<(bsz * seqLen * ffDim) { dffn1[i] = 0 }
        for i in 0..<(bsz * seqLen * seqLen) { dalpha[i] = 0; dscores[i] = 0 }

        // 1) grad Wcls, bCls, dpooled.
        for b in 0..<bsz {
            let lBase = b * numClasses
            let pBase = b * dModel

            for c in 0..<numClasses {
                let dl = dlogits[lBase + c]
                grads.bCls[c] += dl
            }

            for d in 0..<dModel {
                let pv = pooled[pBase + d]
                let wRow = d * numClasses
                var acc: Float = 0
                for c in 0..<numClasses {
                    let dl = dlogits[lBase + c]
                    grads.wCls[wRow + c] += pv * dl
                    acc += dl * model.wCls[wRow + c]
                }
                dpooled[pBase + d] = acc
            }
        }

        // 2) Mean pool backward -> dffn2.
        let invSeq: Float = 1.0 / Float(seqLen)
        for b in 0..<bsz {
            let pBase = b * dModel
            for t in 0..<seqLen {
                let tokBase = (b * seqLen + t) * dModel
                for d in 0..<dModel {
                    dffn2[tokBase + d] = dpooled[pBase + d] * invSeq
                }
            }
        }

        // 3) FFN2 grads and dffn1.
        for b in 0..<bsz {
            for t in 0..<seqLen {
                let tokBase = (b * seqLen + t) * dModel
                let f1Base = (b * seqLen + t) * ffDim

                for d in 0..<dModel {
                    grads.bFf2[d] += dffn2[tokBase + d]
                }

                for h in 0..<ffDim {
                    let hval = ffn1[f1Base + h]
                    let wRow = h * dModel
                    for d in 0..<dModel {
                        grads.wFf2[wRow + d] += hval * dffn2[tokBase + d]
                    }
                }

                for h in 0..<ffDim {
                    let wRow = h * dModel
                    var sum: Float = 0
                    for d in 0..<dModel {
                        sum += dffn2[tokBase + d] * model.wFf2[wRow + d]
                    }
                    dffn1[f1Base + h] = sum
                }
            }
        }

        // 4) ReLU backward for FFN1.
        for i in 0..<(bsz * seqLen * ffDim) {
            if ffn1[i] <= 0 { dffn1[i] = 0 }
        }

        // 5) FFN1 grads and dattn.
        for b in 0..<bsz {
            for t in 0..<seqLen {
                let attnBase = (b * seqLen + t) * dModel
                let f1Base = (b * seqLen + t) * ffDim

                for h in 0..<ffDim {
                    grads.bFf1[h] += dffn1[f1Base + h]
                }

                for d in 0..<dModel {
                    let wRow = d * ffDim
                    var acc: Float = 0
                    for h in 0..<ffDim {
                        let dh = dffn1[f1Base + h]
                        grads.wFf1[wRow + h] += attnOut[attnBase + d] * dh
                        acc += dh * model.wFf1[wRow + h]
                    }
                    dattn[attnBase + d] = acc
                }
            }
        }

        // 6) Attention backward: dalpha and dV.
        for b in 0..<bsz {
            for i in 0..<seqLen {
                let rowBase = (b * seqLen + i) * seqLen
                let dBase = (b * seqLen + i) * dModel

                for j in 0..<seqLen {
                    let vBase = (b * seqLen + j) * dModel
                    var dot: Float = 0
                    for d in 0..<dModel {
                        dot += dattn[dBase + d] * v[vBase + d]
                    }
                    dalpha[rowBase + j] = dot
                }

                for j in 0..<seqLen {
                    let a = attn[rowBase + j]
                    let vBase = (b * seqLen + j) * dModel
                    for d in 0..<dModel {
                        dV[vBase + d] += a * dattn[dBase + d]
                    }
                }

                var sum: Float = 0
                for j in 0..<seqLen {
                    sum += dalpha[rowBase + j] * attn[rowBase + j]
                }
                for j in 0..<seqLen {
                    let a = attn[rowBase + j]
                    dscores[rowBase + j] = a * (dalpha[rowBase + j] - sum)
                }
            }
        }

        // 7) dscores -> dQ and dK.
        let invSqrtD: Float = 1.0 / sqrtf(Float(dModel))
        for b in 0..<bsz {
            for i in 0..<seqLen {
                let rowBase = (b * seqLen + i) * seqLen
                let qBase = (b * seqLen + i) * dModel
                for j in 0..<seqLen {
                    let kBase = (b * seqLen + j) * dModel
                    let ds = dscores[rowBase + j] * invSqrtD
                    for d in 0..<dModel {
                        dQ[qBase + d] += ds * k[kBase + d]
                        dK[kBase + d] += ds * q[qBase + d]
                    }
                }
            }
        }

        // 8) Q/K/V projection grads and dtokens.
        for b in 0..<bsz {
            for t in 0..<seqLen {
                let tokBase = (b * seqLen + t) * dModel

                for dOut in 0..<dModel {
                    grads.bQ[dOut] += dQ[tokBase + dOut]
                    grads.bK[dOut] += dK[tokBase + dOut]
                    grads.bV[dOut] += dV[tokBase + dOut]
                }

                for dIn in 0..<dModel {
                    let x = tokens[tokBase + dIn]
                    let wRow = dIn * dModel
                    var acc: Float = 0
                    for dOut in 0..<dModel {
                        let dq = dQ[tokBase + dOut]
                        let dk = dK[tokBase + dOut]
                        let dv = dV[tokBase + dOut]
                        grads.wQ[wRow + dOut] += x * dq
                        grads.wK[wRow + dOut] += x * dk
                        grads.wV[wRow + dOut] += x * dv
                        acc += dq * model.wQ[wRow + dOut]
                        acc += dk * model.wK[wRow + dOut]
                        acc += dv * model.wV[wRow + dOut]
                    }
                    dtokens[tokBase + dIn] = acc
                }
            }
        }

        // 9) ReLU backward for tokens.
        for i in 0..<(bsz * seqLen * dModel) {
            if tokens[i] <= 0 { dtokens[i] = 0 }
        }

        // 10) pos, bPatch, wPatch grads.
        for b in 0..<bsz {
            for t in 0..<seqLen {
                let tokBase = (b * seqLen + t) * dModel
                let posBase = t * dModel
                let pBase = (b * seqLen + t) * patchDim

                for d in 0..<dModel {
                    let gdt = dtokens[tokBase + d]
                    grads.pos[posBase + d] += gdt
                    grads.bPatch[d] += gdt
                }

                for j in 0..<patchDim {
                    let x = patches[pBase + j]
                    let wBase = j * dModel
                    for d in 0..<dModel {
                        grads.wPatch[wBase + d] += x * dtokens[tokBase + d]
                    }
                }
            }
        }

        // SGD update (no momentum, no weight decay).
        for i in 0..<model.wPatch.count { model.wPatch[i] -= learningRate * grads.wPatch[i] }
        for i in 0..<model.bPatch.count { model.bPatch[i] -= learningRate * grads.bPatch[i] }
        for i in 0..<model.pos.count    { model.pos[i]    -= learningRate * grads.pos[i] }
        for i in 0..<model.wQ.count     { model.wQ[i]     -= learningRate * grads.wQ[i] }
        for i in 0..<model.bQ.count     { model.bQ[i]     -= learningRate * grads.bQ[i] }
        for i in 0..<model.wK.count     { model.wK[i]     -= learningRate * grads.wK[i] }
        for i in 0..<model.bK.count     { model.bK[i]     -= learningRate * grads.bK[i] }
        for i in 0..<model.wV.count     { model.wV[i]     -= learningRate * grads.wV[i] }
        for i in 0..<model.bV.count     { model.bV[i]     -= learningRate * grads.bV[i] }
        for i in 0..<model.wFf1.count   { model.wFf1[i]   -= learningRate * grads.wFf1[i] }
        for i in 0..<model.bFf1.count   { model.bFf1[i]   -= learningRate * grads.bFf1[i] }
        for i in 0..<model.wFf2.count   { model.wFf2[i]   -= learningRate * grads.wFf2[i] }
        for i in 0..<model.bFf2.count   { model.bFf2[i]   -= learningRate * grads.bFf2[i] }
        for i in 0..<model.wCls.count   { model.wCls[i]   -= learningRate * grads.wCls[i] }
        for i in 0..<model.bCls.count   { model.bCls[i]   -= learningRate * grads.bCls[i] }

        start += bsz
    }

    return totalLoss / Float(n)
}

/// Compute the model's classification accuracy on the provided dataset.
/// - Parameters:
///   - model: The attention model used for inference.
///   - images: Flattened image data where each sample occupies `numInputs` consecutive floats (length must be samples * `numInputs`).
///   - labels: Ground-truth class labels (one `UInt8` per sample).
///   - config: Configuration providing evaluation batch size.
/// - Returns: Accuracy as a percentage between 0.0 and 100.0.
func testAccuracy(model: AttnModel, images: [Float], labels: [UInt8], config: Config) -> Float {
    let n = labels.count
    let batchSize = config.batchSize

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

        // Logits.
        for b in 0..<bsz {
            let pBase = b * dModel
            let lBase = b * numClasses
            for c in 0..<numClasses {
                var sum = model.bCls[c]
                for d in 0..<dModel {
                    sum += pooled[pBase + d] * model.wCls[d * numClasses + c]
                }
                logits[lBase + c] = sum
            }
        }

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

/// Saves the attention model to a binary file in native endianness format.
/// - Parameters:
///   - model: The attention model to save.
///   - filename: Path to the output file where the model will be written.
///
/// The file format stores model dimensions as Int32 values followed by all weights and biases as Double values (converted from Float).
/// The dimensions saved are: patchDim, dModel, seqLen, ffDim, and numClasses.
func saveModel(model: AttnModel, filename: String) {
    FileManager.default.createFile(atPath: filename, contents: nil)
    guard let handle = try? FileHandle(forWritingTo: URL(fileURLWithPath: filename)) else {
        print("""

        ERROR: Failed to save attention model
        ======================================
        Could not open file for writing: \(filename)

        Possible causes:
          - Insufficient permissions to write to this directory
          - Disk is full or write-protected
          - Path contains invalid characters

        Solutions:
          1. Check directory permissions:
             ls -ld \(NSString(string: filename).deletingLastPathComponent)

          2. Verify disk space:
             df -h .

          3. Try saving to a different location:
             swift mnist_attention_pool.swift  # saves to ./attention_model.bin by default
        """)
        exit(1)
    }
    defer { try? handle.close() }

    func writeInt32(_ value: Int32) {
        var v = value
        handle.write(Data(bytes: &v, count: MemoryLayout<Int32>.size))
    }

    func writeDouble(_ value: Double) {
        var v = value
        handle.write(Data(bytes: &v, count: MemoryLayout<Double>.size))
    }

    // Write model dimensions.
    writeInt32(Int32(patchDim))
    writeInt32(Int32(dModel))
    writeInt32(Int32(seqLen))
    writeInt32(Int32(ffDim))
    writeInt32(Int32(numClasses))

    // Write all weights and biases.
    for w in model.wPatch { writeDouble(Double(w)) }
    for b in model.bPatch { writeDouble(Double(b)) }
    for p in model.pos { writeDouble(Double(p)) }
    for w in model.wQ { writeDouble(Double(w)) }
    for b in model.bQ { writeDouble(Double(b)) }
    for w in model.wK { writeDouble(Double(w)) }
    for b in model.bK { writeDouble(Double(b)) }
    for w in model.wV { writeDouble(Double(w)) }
    for b in model.bV { writeDouble(Double(b)) }
    for w in model.wFf1 { writeDouble(Double(w)) }
    for b in model.bFf1 { writeDouble(Double(b)) }
    for w in model.wFf2 { writeDouble(Double(w)) }
    for b in model.bFf2 { writeDouble(Double(b)) }
    for w in model.wCls { writeDouble(Double(w)) }
    for b in model.bCls { writeDouble(Double(b)) }

    print("Model saved to \(filename)")
}

/// Loads the attention model from a binary file in native endianness format.
/// - Parameter filename: Path to the input file where the model is stored.
/// - Returns: An `AttnModel` instance if the file was successfully loaded and model dimensions match, otherwise `nil`.
///
/// The file format reads model dimensions as Int32 values followed by all weights and biases as Double values (converted to Float).
/// The dimensions expected are: patchDim, dModel, seqLen, ffDim, and numClasses.
func loadModel(filename: String) -> AttnModel? {
    guard let handle = try? FileHandle(forReadingFrom: URL(fileURLWithPath: filename)) else {
        print("""

        ERROR: Failed to load attention model
        ======================================
        Model file not found: \(filename)

        This error occurs when trying to load a saved model that doesn't exist.

        Solutions:
          1. Train a new model to generate the file:
             swift mnist_attention_pool.swift --epochs 5

          2. Check if the file exists:
             ls -l \(filename)

          3. Verify you're in the correct directory:
             pwd
        """)
        return nil
    }
    defer { try? handle.close() }

    func readInt32() -> Int32? {
        guard let data = try? handle.read(upToCount: MemoryLayout<Int32>.size),
              data.count == MemoryLayout<Int32>.size else {
            return nil
        }
        return data.withUnsafeBytes { $0.load(as: Int32.self) }
    }

    func readDouble() -> Double? {
        guard let data = try? handle.read(upToCount: MemoryLayout<Double>.size),
              data.count == MemoryLayout<Double>.size else {
            return nil
        }
        return data.withUnsafeBytes { $0.load(as: Double.self) }
    }

    // Read and validate header.
    guard let patchDimRead = readInt32(),
          let dModelRead = readInt32(),
          let seqLenRead = readInt32(),
          let ffDimRead = readInt32(),
          let numClassesRead = readInt32() else {
        print("""

        ERROR: Corrupted model file - header unreadable
        ================================================
        Failed to read model header from: \(filename)

        This error indicates the model file is corrupted or incomplete.
        The model header contains critical architecture information:
          - patchDim (patch dimension)
          - dModel (model embedding dimension)
          - seqLen (sequence length)
          - ffDim (feed-forward dimension)
          - numClasses (output classes)

        Possible causes:
          - File was truncated during save
          - Disk write error occurred
          - File was corrupted during transfer
          - Attempting to load a non-model file

        Solutions:
          1. Retrain the model to regenerate:
             swift mnist_attention_pool.swift --epochs 5

          2. Verify file size (should be >100KB):
             ls -lh \(filename)

          3. Check for disk errors in system logs
        """)
        return nil
    }

    if patchDimRead != Int32(patchDim) || dModelRead != Int32(dModel) ||
       seqLenRead != Int32(seqLen) || ffDimRead != Int32(ffDim) ||
       numClassesRead != Int32(numClasses) {
        print("""

        ERROR: Model architecture mismatch
        ==================================
        The saved model has different dimensions than expected.

        Expected architecture (current code):
          - patchDim   = \(patchDim)
          - dModel     = \(dModel)
          - seqLen     = \(seqLen)
          - ffDim      = \(ffDim)
          - numClasses = \(numClasses)

        Model file architecture:
          - patchDim   = \(patchDimRead)
          - dModel     = \(dModelRead)
          - seqLen     = \(seqLenRead)
          - ffDim      = \(ffDimRead)
          - numClasses = \(numClassesRead)

        This occurs when the model file was created with different hyperparameters.

        Solutions:
          1. Retrain with current architecture:
             swift mnist_attention_pool.swift --epochs 5

          2. Or update code constants to match saved model
             (edit lines 24-27 in this file)

          3. Verify you're loading the correct model file
        """)
        return nil
    }

    // Read all weights and biases.
    var wPatch = [Float](repeating: 0, count: patchDim * dModel)
    for i in 0..<wPatch.count {
        guard let val = readDouble() else {
            print("""

            ERROR: Incomplete model file - patch projection weights corrupted
            ==================================================================
            Failed to read patch projection weight parameter [\(i)/\(wPatch.count)] from: \(filename)

            The model file appears to be truncated or corrupted during the
            patch embedding weight section (wPatch).

            Progress: \(i) of \(wPatch.count) weights read before failure
            Location: After header, in patch projection weights

            Solutions:
              1. Retrain the model:
                 swift mnist_attention_pool.swift --epochs 5

              2. Check file integrity:
                 ls -lh \(filename)  # should be >100KB

              3. Verify disk space during training
            """)
            return nil
        }
        wPatch[i] = Float(val)
    }

    var bPatch = [Float](repeating: 0, count: dModel)
    for i in 0..<bPatch.count {
        guard let val = readDouble() else {
            print("""

            ERROR: Incomplete model file - patch projection biases corrupted
            =================================================================
            Failed to read patch projection bias parameter [\(i)/\(bPatch.count)] from: \(filename)

            The model file appears to be truncated or corrupted during the
            patch embedding bias section (bPatch).

            Progress: \(i) of \(bPatch.count) biases read before failure
            Location: After wPatch, in patch projection biases

            Solutions:
              1. Retrain the model:
                 swift mnist_attention_pool.swift --epochs 5

              2. Check for disk errors during model save

              3. Verify file wasn't interrupted during write
            """)
            return nil
        }
        bPatch[i] = Float(val)
    }

    var pos = [Float](repeating: 0, count: seqLen * dModel)
    for i in 0..<pos.count {
        guard let val = readDouble() else {
            print("""

            ERROR: Incomplete model file - positional encodings corrupted
            ==============================================================
            Failed to read positional encoding parameter [\(i)/\(pos.count)] from: \(filename)

            The model file appears to be truncated or corrupted during the
            positional encoding section (pos).

            Progress: \(i) of \(pos.count) positional encodings read before failure
            Location: After bPatch, in positional encodings

            Solutions:
              1. Retrain the model:
                 swift mnist_attention_pool.swift --epochs 5

              2. Verify complete model was saved:
                 Expected size: >100KB for full attention model

              3. Check system logs for I/O errors
            """)
            return nil
        }
        pos[i] = Float(val)
    }

    var wQ = [Float](repeating: 0, count: dModel * dModel)
    for i in 0..<wQ.count {
        guard let val = readDouble() else {
            print("""

            ERROR: Incomplete model file - Query projection weights corrupted
            ==================================================================
            Failed to read Query (Q) projection weight [\(i)/\(wQ.count)] from: \(filename)

            The model file appears to be truncated or corrupted during the
            self-attention Query weight section (wQ).

            Progress: \(i) of \(wQ.count) Q weights read before failure
            Location: After positional encodings, in attention Q projection

            Solutions:
              1. Retrain the model:
                 swift mnist_attention_pool.swift --epochs 5

              2. Model may have been interrupted during save

              3. Verify disk wasn't full during training
            """)
            return nil
        }
        wQ[i] = Float(val)
    }

    var bQ = [Float](repeating: 0, count: dModel)
    for i in 0..<bQ.count {
        guard let val = readDouble() else {
            print("""

            ERROR: Incomplete model file - Query projection biases corrupted
            =================================================================
            Failed to read Query (Q) projection bias [\(i)/\(bQ.count)] from: \(filename)

            The model file appears to be truncated or corrupted during the
            self-attention Query bias section (bQ).

            Progress: \(i) of \(bQ.count) Q biases read before failure
            Location: After wQ, in attention Q bias

            Solutions:
              1. Retrain the model:
                 swift mnist_attention_pool.swift --epochs 5

              2. Check file size matches expected for complete model

              3. Verify no interruptions occurred during save
            """)
            return nil
        }
        bQ[i] = Float(val)
    }

    var wK = [Float](repeating: 0, count: dModel * dModel)
    for i in 0..<wK.count {
        guard let val = readDouble() else {
            print("""

            ERROR: Incomplete model file - Key projection weights corrupted
            ================================================================
            Failed to read Key (K) projection weight [\(i)/\(wK.count)] from: \(filename)

            The model file appears to be truncated or corrupted during the
            self-attention Key weight section (wK).

            Progress: \(i) of \(wK.count) K weights read before failure
            Location: After bQ, in attention K projection

            Solutions:
              1. Retrain the model:
                 swift mnist_attention_pool.swift --epochs 5

              2. Verify model file wasn't corrupted during transfer

              3. Check for filesystem errors
            """)
            return nil
        }
        wK[i] = Float(val)
    }

    var bK = [Float](repeating: 0, count: dModel)
    for i in 0..<bK.count {
        guard let val = readDouble() else {
            print("""

            ERROR: Incomplete model file - Key projection biases corrupted
            ===============================================================
            Failed to read Key (K) projection bias [\(i)/\(bK.count)] from: \(filename)

            The model file appears to be truncated or corrupted during the
            self-attention Key bias section (bK).

            Progress: \(i) of \(bK.count) K biases read before failure
            Location: After wK, in attention K bias

            Solutions:
              1. Retrain the model:
                 swift mnist_attention_pool.swift --epochs 5

              2. Check available disk space during training

              3. Verify complete save before loading
            """)
            return nil
        }
        bK[i] = Float(val)
    }

    var wV = [Float](repeating: 0, count: dModel * dModel)
    for i in 0..<wV.count {
        guard let val = readDouble() else {
            print("""

            ERROR: Incomplete model file - Value projection weights corrupted
            ==================================================================
            Failed to read Value (V) projection weight [\(i)/\(wV.count)] from: \(filename)

            The model file appears to be truncated or corrupted during the
            self-attention Value weight section (wV).

            Progress: \(i) of \(wV.count) V weights read before failure
            Location: After bK, in attention V projection

            Solutions:
              1. Retrain the model:
                 swift mnist_attention_pool.swift --epochs 5

              2. Model file may be incomplete - check save logs

              3. Verify filesystem integrity
            """)
            return nil
        }
        wV[i] = Float(val)
    }

    var bV = [Float](repeating: 0, count: dModel)
    for i in 0..<bV.count {
        guard let val = readDouble() else {
            print("""

            ERROR: Incomplete model file - Value projection biases corrupted
            =================================================================
            Failed to read Value (V) projection bias [\(i)/\(bV.count)] from: \(filename)

            The model file appears to be truncated or corrupted during the
            self-attention Value bias section (bV).

            Progress: \(i) of \(bV.count) V biases read before failure
            Location: After wV, in attention V bias

            Solutions:
              1. Retrain the model:
                 swift mnist_attention_pool.swift --epochs 5

              2. Check for interruptions during model save

              3. Ensure sufficient disk space
            """)
            return nil
        }
        bV[i] = Float(val)
    }

    var wFf1 = [Float](repeating: 0, count: dModel * ffDim)
    for i in 0..<wFf1.count {
        guard let val = readDouble() else {
            print("""

            ERROR: Incomplete model file - Feed-forward layer 1 weights corrupted
            ======================================================================
            Failed to read feed-forward layer 1 weight [\(i)/\(wFf1.count)] from: \(filename)

            The model file appears to be truncated or corrupted during the
            first feed-forward layer weight section (wFf1).

            Progress: \(i) of \(wFf1.count) FF1 weights read before failure
            Location: After bV, in feed-forward layer 1 weights

            Solutions:
              1. Retrain the model:
                 swift mnist_attention_pool.swift --epochs 5

              2. Verify model file size is complete

              3. Check for disk write errors in system logs
            """)
            return nil
        }
        wFf1[i] = Float(val)
    }

    var bFf1 = [Float](repeating: 0, count: ffDim)
    for i in 0..<bFf1.count {
        guard let val = readDouble() else {
            print("""

            ERROR: Incomplete model file - Feed-forward layer 1 biases corrupted
            =====================================================================
            Failed to read feed-forward layer 1 bias [\(i)/\(bFf1.count)] from: \(filename)

            The model file appears to be truncated or corrupted during the
            first feed-forward layer bias section (bFf1).

            Progress: \(i) of \(bFf1.count) FF1 biases read before failure
            Location: After wFf1, in feed-forward layer 1 biases

            Solutions:
              1. Retrain the model:
                 swift mnist_attention_pool.swift --epochs 5

              2. Model save may have been interrupted

              3. Verify disk space was available during training
            """)
            return nil
        }
        bFf1[i] = Float(val)
    }

    var wFf2 = [Float](repeating: 0, count: ffDim * dModel)
    for i in 0..<wFf2.count {
        guard let val = readDouble() else {
            print("""

            ERROR: Incomplete model file - Feed-forward layer 2 weights corrupted
            ======================================================================
            Failed to read feed-forward layer 2 weight [\(i)/\(wFf2.count)] from: \(filename)

            The model file appears to be truncated or corrupted during the
            second feed-forward layer weight section (wFf2).

            Progress: \(i) of \(wFf2.count) FF2 weights read before failure
            Location: After bFf1, in feed-forward layer 2 weights

            Solutions:
              1. Retrain the model:
                 swift mnist_attention_pool.swift --epochs 5

              2. Check for file corruption during transfer

              3. Verify filesystem errors didn't occur
            """)
            return nil
        }
        wFf2[i] = Float(val)
    }

    var bFf2 = [Float](repeating: 0, count: dModel)
    for i in 0..<bFf2.count {
        guard let val = readDouble() else {
            print("""

            ERROR: Incomplete model file - Feed-forward layer 2 biases corrupted
            =====================================================================
            Failed to read feed-forward layer 2 bias [\(i)/\(bFf2.count)] from: \(filename)

            The model file appears to be truncated or corrupted during the
            second feed-forward layer bias section (bFf2).

            Progress: \(i) of \(bFf2.count) FF2 biases read before failure
            Location: After wFf2, in feed-forward layer 2 biases

            Solutions:
              1. Retrain the model:
                 swift mnist_attention_pool.swift --epochs 5

              2. Verify complete model save before loading

              3. Check for interruptions during write
            """)
            return nil
        }
        bFf2[i] = Float(val)
    }

    var wCls = [Float](repeating: 0, count: dModel * numClasses)
    for i in 0..<wCls.count {
        guard let val = readDouble() else {
            print("""

            ERROR: Incomplete model file - Classification layer weights corrupted
            ======================================================================
            Failed to read classification weight [\(i)/\(wCls.count)] from: \(filename)

            The model file appears to be truncated or corrupted during the
            final classification layer weight section (wCls).

            Progress: \(i) of \(wCls.count) classifier weights read before failure
            Location: After bFf2, in classification layer weights

            Solutions:
              1. Retrain the model:
                 swift mnist_attention_pool.swift --epochs 5

              2. Model file is almost complete - likely write interruption

              3. Check disk space and I/O logs
            """)
            return nil
        }
        wCls[i] = Float(val)
    }

    var bCls = [Float](repeating: 0, count: numClasses)
    for i in 0..<bCls.count {
        guard let val = readDouble() else {
            print("""

            ERROR: Incomplete model file - Classification layer biases corrupted
            =====================================================================
            Failed to read classification bias [\(i)/\(bCls.count)] from: \(filename)

            The model file appears to be truncated at the very end.
            All other parameters loaded successfully.

            Progress: \(i) of \(bCls.count) classifier biases read before failure
            Location: Final section of model file (classification biases)

            Solutions:
              1. Retrain the model:
                 swift mnist_attention_pool.swift --epochs 5

              2. Model save was interrupted at the last step

              3. Verify no disk space issues during training
            """)
            return nil
        }
        bCls[i] = Float(val)
    }

    print("Model loaded from \(filename)")
    return AttnModel(
        wPatch: wPatch,
        bPatch: bPatch,
        pos: pos,
        wQ: wQ,
        bQ: bQ,
        wK: wK,
        bK: bK,
        wV: wV,
        bV: bV,
        wFf1: wFf1,
        bFf1: bFf1,
        wFf2: wFf2,
        bFf2: bFf2,
        wCls: wCls,
        bCls: bCls
    )
}

// =============================================================================
// MARK: - Random Number Generator
// =============================================================================

struct SimpleRng {
    private var state: UInt64

    // Explicit seed (if zero, use a fixed value).
    init(seed: UInt64) {
        self.state = seed == 0 ? 0x9e3779b97f4a7c15 : seed
    }

    /// Reseeds the RNG state using the current wall-clock time in nanoseconds.
    /// If the computed timestamp is zero, sets the state to the fallback constant 0x9e3779b97f4a7c15.
    mutating func reseedFromTime() {
        let nanos = UInt64(Date().timeIntervalSince1970 * 1_000_000_000)
        state = nanos == 0 ? 0x9e3779b97f4a7c15 : nanos
    }

    /// Advances the RNG state and produces a 32-bit pseudorandom unsigned integer.
    /// - Returns: A pseudorandom `UInt32` generated from the updated internal state.
    mutating func nextUInt32() -> UInt32 {
        var x = state
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        state = x
        return UInt32(truncatingIfNeeded: x >> 32)
    }

    /// Produces a pseudo-random value uniformly sampled between 0.0 and 1.0.
    /// - Returns: A pseudo-random Float in the range 0.0 through 1.0 (inclusive).
    mutating func nextFloat() -> Float {
        return Float(nextUInt32()) / Float(UInt32.max)
    }

    /// Returns a random Float sampled uniformly from the range [low, high).
    /// - Parameters:
    ///   - low: Lower bound of the range (inclusive).
    ///   - high: Upper bound of the range (exclusive).
    /// - Returns: A `Float` uniformly distributed between `low` (inclusive) and `high` (exclusive).
    mutating func uniform(_ low: Float, _ high: Float) -> Float {
        return low + (high - low) * nextFloat()
    }

    /// Returns a uniformly distributed integer in the range 0..<upper (exclusive).
    /// - Parameter upper: Exclusive upper bound; if `upper` is 0 the function returns 0.
    /// - Returns: An `Int` in `0..<upper`, or `0` when `upper` is `0`.
    mutating func nextInt(upper: Int) -> Int {
        return upper == 0 ? 0 : Int(nextUInt32()) % upper
    }

    /// Randomly permutes the elements of `array` in place using this RNG.
    /// - Parameter array: The integer array to be shuffled in place.
    mutating func shuffle(_ array: inout [Int]) {
        let n = array.count
        for i in stride(from: n - 1, through: 1, by: -1) {
            let j = nextInt(upper: i + 1)
            array.swapAt(i, j)
        }
    }
}

// =============================================================================
// MARK: - MNIST Data Loading
// =============================================================================

/// Reads an MNIST IDX image file and returns a flat array of normalized pixel values.
///
/// The file is parsed as big-endian IDX (magic, count, rows, cols). Pixel values are converted to
/// Float in the range 0.0–1.0. If the file cannot be opened the process exits with code 1.
/// - Parameters:
///   - path: Filesystem path to the MNIST images IDX file.
///   - count: Maximum number of images to read.
/// - Returns: A flat array of Float values representing `min(count, totalImagesInFile)` grayscale images, each with `rows * cols` pixels normalized to [0.0, 1.0].
func readMnistImages(path: String, count: Int) -> [Float] {
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url) else {
        print("Could not open file \(path)")
        exit(1)
    }

    return data.withUnsafeBytes { rawBuf in
        guard let base = rawBuf.bindMemory(to: UInt8.self).baseAddress else {
            return []
        }
        var offset = 0

        /// Read four bytes from `base` at the current `offset`, interpret them as a big-endian integer, and advance `offset` by four.
        /// - Returns: The 32-bit unsigned integer constructed from the four big-endian bytes.
        func readU32BE() -> UInt32 {
            let b0 = UInt32(base[offset]) << 24
            let b1 = UInt32(base[offset + 1]) << 16
            let b2 = UInt32(base[offset + 2]) << 8
            let b3 = UInt32(base[offset + 3])
            offset += 4
            return b0 | b1 | b2 | b3
        }

        _ = readU32BE()
        let total = Int(readU32BE())
        let rows = Int(readU32BE())
        let cols = Int(readU32BE())
        let imageSize = rows * cols
        let actualCount = min(count, total)

        var images = [Float](repeating: 0.0, count: actualCount * imageSize)
        for i in 0..<actualCount {
            let baseIndex = i * imageSize
            for j in 0..<imageSize {
                images[baseIndex + j] = Float(base[offset]) / 255.0
                offset += 1
            }
        }
        return images
    }
}

/// Reads MNIST label data from an IDX file and returns up to `count` label bytes.
///
/// The function opens the file at `path`, parses the IDX header (big-endian 32-bit fields),
/// and extracts up to `count` label values. If the file cannot be opened the process exits with code 1.
/// - Parameters:
///   - path: Filesystem path to the IDX label file.
///   - count: Maximum number of labels to read.
/// - Returns: An array of `UInt8` label values, length is `min(count, numberOfLabelsInFile)`.
func readMnistLabels(path: String, count: Int) -> [UInt8] {
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url) else {
        print("Could not open file \(path)")
        exit(1)
    }

    return data.withUnsafeBytes { rawBuf in
        guard let base = rawBuf.bindMemory(to: UInt8.self).baseAddress else {
            return []
        }
        var offset = 0

        /// Read four bytes from `base` at the current `offset`, interpret them as a big-endian integer, and advance `offset` by four.
        /// - Returns: The 32-bit unsigned integer constructed from the four big-endian bytes.
        func readU32BE() -> UInt32 {
            let b0 = UInt32(base[offset]) << 24
            let b1 = UInt32(base[offset + 1]) << 16
            let b2 = UInt32(base[offset + 2]) << 8
            let b3 = UInt32(base[offset + 3])
            offset += 4
            return b0 | b1 | b2 | b3
        }

        _ = readU32BE()
        let total = Int(readU32BE())
        let actualCount = min(count, total)

        var labels = [UInt8](repeating: 0, count: actualCount)
        for i in 0..<actualCount {
            labels[i] = base[offset]
            offset += 1
        }
        return labels
    }
}

/// Entry point that runs the full training and evaluation pipeline for the compact Transformer-style MNIST model.
///
/// Parses command-line options, loads MNIST data, initializes RNG and model parameters, runs training for the configured number of epochs while logging per-epoch loss and test accuracy, evaluates final test accuracy, and prints timing summaries. Performs file I/O for reading the dataset and writing a training loss log.
func main() {
    // Parse command-line configuration.
    let config = Config.parse()

    let programStart = Date()

    print("Loading MNIST...")
    let loadStart = Date()
    let trainImages = readMnistImages(path: "./data/train-images.idx3-ubyte", count: trainSamples)
    let trainLabels = readMnistLabels(path: "./data/train-labels.idx1-ubyte", count: trainSamples)
    let testImages  = readMnistImages(path: "./data/t10k-images.idx3-ubyte", count: testSamples)
    let testLabels  = readMnistLabels(path: "./data/t10k-labels.idx1-ubyte", count: testSamples)
    let loadTime = Date().timeIntervalSince(loadStart)
    print(String(format: "Data loading time: %.2f seconds", loadTime))

    print("Config: patch=\(patch)x\(patch) tokens=\(seqLen) d=\(dModel) ff=\(ffDim) batch=\(config.batchSize) epochs=\(config.epochs) lr=\(config.learningRate) seed=\(config.seed)")

    var rng = SimpleRng(seed: config.seed)
    if config.seed == 0 {
        rng.reseedFromTime()
    }
    var model = initModel(rng: &rng)

    // Training log file.
    try? FileManager.default.createDirectory(atPath: "./logs", withIntermediateDirectories: true)
    FileManager.default.createFile(atPath: "./logs/training_loss_attention_mnist.txt", contents: nil)
    let logHandle = try? FileHandle(forWritingTo: URL(fileURLWithPath: "./logs/training_loss_attention_mnist.txt"))
    defer { try? logHandle?.close() }

    let trainN = min(trainLabels.count, trainSamples)
    var indices = Array(0..<trainN)

    print("Training...")
    let trainStart = Date()
    for e in 0..<config.epochs {
        let t0 = Date()
        let avgLoss = trainEpoch(model: &model, images: trainImages, labels: trainLabels, indices: &indices, rng: &rng, config: config)
        let dt = Float(Date().timeIntervalSince(t0))

        let acc = testAccuracy(model: model, images: testImages, labels: testLabels, config: config)
        print(String(format: "Epoch %d | loss=%.6f | time=%.3fs | test_acc=%.2f%%", e + 1, avgLoss, dt, acc))

        if let h = logHandle {
            let line = "\(e + 1),\(avgLoss),\(dt),\(acc)\n"
            h.write(Data(line.utf8))
        }
    }
    let trainTime = Date().timeIntervalSince(trainStart)

    let finalAcc = testAccuracy(model: model, images: testImages, labels: testLabels, config: config)
    print(String(format: "Final Test Accuracy: %.2f%%", finalAcc))

    print("Saving model...")
    saveModel(model: model, filename: "mnist_attention_model.bin")

    let totalTime = Date().timeIntervalSince(programStart)
    print("\n=== Summary ===")
    print(String(format: "Load: %.2fs", loadTime))
    print(String(format: "Train: %.2fs", trainTime))
    print(String(format: "Total: %.2fs", totalTime))
    print("=============")
}

main()
