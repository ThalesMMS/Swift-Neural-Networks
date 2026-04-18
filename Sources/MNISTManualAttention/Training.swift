import Foundation
import MNISTCommon

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Performs one training epoch over the provided dataset and updates `model` in place using SGD.
/// - Parameters:
///   - model: The attention model to train; its weights and biases are updated in place.
///   - images: Flattened input samples; each sample occupies `numInputs` consecutive `Float` values.
///   - labels: One label per sample, stored as `UInt8` class indices.
///   - indices: Array of sample indices to iterate; the order is shuffled in place before training.
///   - rng: Random number generator used to shuffle `indices`; it may be mutated.
///   - config: Training configuration containing `batchSize` and `learningRate`.
/// - Returns: The average softmax cross-entropy loss per sample computed over the epoch.
func trainEpoch(
    model: inout AttnModel,
    images: [Float],
    labels: [UInt8],
    indices: inout [Int],
    rng: inout SimpleRng,
    config: Config
) -> Float {
    let batchSize = config.batchSize
    precondition(batchSize > 0, "batchSize must be > 0 before the minibatch loop so start advances by bsz")

    rng.shuffle(&indices)

    var grads = Grads()

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
