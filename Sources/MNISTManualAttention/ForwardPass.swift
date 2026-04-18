import Foundation
import Accelerate
import MNISTCommon

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Extracts non-overlapping square patches from flattened batch images and writes them into a flattened patch tensor.
///
/// The input images are assumed to be laid out back-to-back in `batchInputs` in row-major order (each image of width `imgW` and height `imgH`). Patches are taken on a `grid x grid` layout with patch size `patch x patch`. For each batch sample and patch position the patch's pixels are written in row-major order into `patchesOut` at index `(b * seqLen + t) * patchDim + j`, where `t` is the patch token index and `j` is the flattened pixel offset within the patch.
/// - Parameters:
///   - batchInputs: Flattened input images for the whole batch, length `batchCount * numInputs` (each image `numInputs = imgW * imgH`), row-major per image.
///   - batchCount: Number of images (samples) in the batch.
///   - patchesOut: Output buffer (inout) receiving flattened patches, length `batchCount * seqLen * patchDim`; layout is sample-major, then token index `t`, then patch pixels in row-major order.
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

/// Builds post-ReLU token embeddings from flattened image patches and writes them into `tokens`.
/// 
/// For each sample and token position, computes a patch projection plus a learned patch bias and the positional embedding, applies ReLU, and stores the result in `tokens`.
/// - Parameters:
///   - model: The `AttnModel` containing `bPatch`, `pos`, and `wPatch` parameters used to form token embeddings.
///   - batchCount: Number of samples in the batch.
///   - patches: Flattened patch values with layout [batchCount * seqLen * patchDim].
///   - tokens: Output buffer written in-place with shape [batchCount * seqLen * dModel]; receives the post-ReLU token embeddings.
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
/// Computes batched attention score matrices by multiplying Q and K^T for each sample and writes them into `scores`.
/// - Parameters:
///   - q: Flattened Q tensors in row-major order with shape [batchCount, seqLen, dModel].
///   - k: Flattened K tensors in row-major order with shape [batchCount, seqLen, dModel].
///   - scores: Mutable output buffer; will be filled row-major with shape [batchCount, seqLen, seqLen], where each block is Q_b * K_b^T for a batch sample.
///   - batchCount: Number of samples in the batch.
///   - seqLen: Sequence length (number of tokens) per sample.
///   - dModel: Embedding dimension (width of Q/K vectors).
func computeAttentionScoresVDSP(
    q: [Float],
    k: [Float],
    scores: inout [Float],
    batchCount: Int,
    seqLen: Int,
    dModel: Int
) {
    // For each sample in batch, compute Q_b * K_b^T
    var kTransposed = [Float](repeating: 0.0, count: seqLen * dModel)

    for b in 0..<batchCount {
        let qOffset = b * seqLen * dModel
        let kOffset = b * seqLen * dModel
        let scoresOffset = b * seqLen * seqLen

        // Transpose K from [seqLen, dModel] to [dModel, seqLen]
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

/// Performs a full self-attention forward pass: projects input `tokens` into Q/K/V, computes scaled dot‑product attention with softmax, and writes the attention-weighted outputs.
/// 
/// - Parameters:
///   - model: Learned parameters containing projection weights and biases (wQ/wK/wV and bQ/bK/bV) and positional/patch parameters used for the projections.
///   - batchCount: Number of samples in the batch.
///   - tokens: Input token embeddings laid out row-major as [batch, seqLen, dModel].
///   - q: Output buffer for query projections; written in-place with shape [batch, seqLen, dModel].
///   - k: Output buffer for key projections; written in-place with shape [batch, seqLen, dModel].
///   - v: Output buffer for value projections; written in-place with shape [batch, seqLen, dModel].
///   - attn: Working buffer used to store attention score matrices per batch; on return contains the row-normalized attention weights with shape [batch, seqLen, seqLen].
///   - attnOut: Output buffer for the attention-weighted token representations; written in-place with shape [batch, seqLen, dModel].
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

/// Applies a two-layer feed-forward network to each token: a linear layer with ReLU (D -> FF), followed by a second linear layer (FF -> D).
/// - Parameters:
///   - model: The model containing feed-forward weights and biases (`wFf1`, `bFf1`, `wFf2`, `bFf2`) and dimensionality constants.
///   - batchCount: Number of samples in the batch.
///   - attnOut: Input token embeddings shaped as [batchCount * seqLen * dModel].
///   - ffn1: Preallocated output buffer for the hidden layer activations shaped as [batchCount * seqLen * ffDim]; updated in place.
///   - ffn2: Preallocated output buffer for the final layer outputs shaped as [batchCount * seqLen * dModel]; updated in place.
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

/// Averages token embeddings over the sequence dimension, producing one dModel-length vector per batch sample.
/// 
/// For each sample b, the function computes the elementwise mean across the seqLen token vectors and writes the result into `pooled` at index range `b * dModel ..< (b+1) * dModel`, overwriting any previous contents.
/// - Parameters:
///   - batchCount: Number of samples in the batch (leading dimension of `tokens` and `pooled`).
///   - tokens: Flattened token tensor with shape (batchCount, seqLen, dModel) laid out so that the element for (b, t, d) is at index `(b * seqLen + t) * dModel + d`.
///   - pooled: Preallocated output buffer with length `batchCount * dModel`; on return each contiguous block of length `dModel` contains the mean token embedding for the corresponding sample.
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

/// Compute per-sample class logits and softmax probabilities from pooled embeddings.
///
/// - Parameters:
///   - model: The attention model containing classifier weights `wCls` and biases `bCls`.
///   - batchCount: Number of samples in the batch.
///   - pooled: Flattened pooled embeddings with length `batchCount * dModel` (row-major per sample).
///   - logits: Output buffer written with raw logits; must have length `batchCount * numClasses`.
///   - probs: Output buffer written with softmax probabilities; must have length `batchCount * numClasses`.
///
/// This function writes class logits into `logits` and then computes in-place softmax into `probs` for each sample row.
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
