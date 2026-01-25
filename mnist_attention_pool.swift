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
let dModel = 16                  // keep small for pure CPU
let ffDim = 32                   // feed-forward hidden size

// Dataset sizes.
let trainSamples = 60_000
let testSamples = 10_000

// =============================================================================
// MARK: - Configuration
// =============================================================================

/// Configuration for training hyperparameters
struct Config {
    var learningRate: Float = 0.01
    var epochs: Int = 5
    var batchSize: Int = 32
    var rngSeed: UInt64 = 1

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
                i += 1
                if i < args.count, let val = Int(args[i]), val > 0 {
                    config.batchSize = val
                }

            case "--epochs", "-e":
                i += 1
                if i < args.count, let val = Int(args[i]), val > 0 {
                    config.epochs = val
                }

            case "--lr", "-l":
                i += 1
                if i < args.count, let val = Float(args[i]), val > 0 {
                    config.learningRate = val
                }

            case "--seed", "-s":
                i += 1
                if i < args.count, let val = UInt64(args[i]) {
                    config.rngSeed = val
                }

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
      --lr, -l <f>       Learning rate (default: 0.01)
      --seed, -s <n>     RNG seed for reproducibility (default: 1)
      --help, -h         Show this help message

    EXAMPLES:
      swift mnist_attention_pool.swift --epochs 10
      swift mnist_attention_pool.swift -b 64 -e 5 -l 0.005
      swift mnist_attention_pool.swift --seed 42

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

    for b in 0..<batchCount {
        for i in 0..<seqLen {
            let rowBase = (b * seqLen + i) * seqLen
            let qBase = (b * seqLen + i) * dModel

            for j in 0..<seqLen {
                let kBase = (b * seqLen + j) * dModel
                var score: Float = 0
                for d in 0..<dModel {
                    score += q[qBase + d] * k[kBase + d]
                }
                attn[rowBase + j] = score * invSqrtD
            }
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

    // Extract config values for local use.
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
///   - config: Configuration supplying evaluation batch size.
/// - Returns: Accuracy as a percentage (0.0 to 100.0) of correctly predicted samples.
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

    print("Config: patch=\(patch)x\(patch) tokens=\(seqLen) d=\(dModel) ff=\(ffDim) batch=\(config.batchSize) epochs=\(config.epochs) lr=\(config.learningRate) seed=\(config.rngSeed)")

    var rng = SimpleRng(seed: config.rngSeed)
    if config.rngSeed == 0 {
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

    let totalTime = Date().timeIntervalSince(programStart)
    print("\n=== Summary ===")
    print(String(format: "Load: %.2fs", loadTime))
    print(String(format: "Train: %.2fs", trainTime))
    print(String(format: "Total: %.2fs", totalTime))
    print("=============")
}

main()
