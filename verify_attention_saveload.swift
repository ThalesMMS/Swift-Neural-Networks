#!/usr/bin/env swift

// verify_attention_saveload.swift
// Verification script for Attention save/load round-trip
// Tests that saved models can be loaded and produce identical predictions

import Foundation

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

// =============================================================================
// MARK: - Colored Print Utility
// =============================================================================

/// Simple colored console output utility
struct ColoredPrint {
    static func error(_ message: String) {
        if ProcessInfo.processInfo.environment["ANSI_COLORS"] == "1" {
            print("\u{001B}[31m\(message)\u{001B}[0m")  // Red
        } else {
            print("❌ ERROR: \(message)")
        }
    }

    static func warning(_ message: String) {
        if ProcessInfo.processInfo.environment["ANSI_COLORS"] == "1" {
            print("\u{001B}[33m\(message)\u{001B}[0m")  // Yellow
        } else {
            print("⚠️  WARNING: \(message)")
        }
    }

    static func success(_ message: String) {
        if ProcessInfo.processInfo.environment["ANSI_COLORS"] == "1" {
            print("\u{001B}[32m\(message)\u{001B}[0m")  // Green
        } else {
            print("✅ \(message)")
        }
    }

    static func info(_ message: String) {
        if ProcessInfo.processInfo.environment["ANSI_COLORS"] == "1" {
            print("\u{001B}[36m\(message)\u{001B}[0m")  // Cyan
        } else {
            print("ℹ️  \(message)")
        }
    }
}

// =============================================================================
// MARK: - Simple Random Number Generator
// =============================================================================

struct SimpleRng {
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
        return Float(nextUInt32()) / Float(UInt32.max)
    }

    mutating func uniform(_ low: Float, _ high: Float) -> Float {
        return low + (high - low) * nextFloat()
    }

    mutating func nextInt(upper: Int) -> Int {
        return upper == 0 ? 0 : Int(nextUInt32()) % upper
    }

    mutating func shuffle(_ array: inout [Int]) {
        let n = array.count
        if n > 1 {
            for i in stride(from: n - 1, through: 1, by: -1) {
                let j = nextInt(upper: i + 1)
                array.swapAt(i, j)
            }
        }
    }
}

// =============================================================================
// MARK: - MNIST Data Loading
// =============================================================================

func readMnistImages(path: String, count: Int) -> [Float] {
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url) else {
        ColoredPrint.error("Failed to load MNIST images from '\(path)'")
        ColoredPrint.info("→ Download MNIST data: run './download_mnist.sh' from project root")
        ColoredPrint.info("→ Or ensure MNIST data exists in ./data/")
        ColoredPrint.info("→ Expected file format: IDX3-ubyte (MNIST image format)")
        exit(1)
    }

    return data.withUnsafeBytes { rawBuf in
        guard let base = rawBuf.bindMemory(to: UInt8.self).baseAddress else {
            return []
        }
        var offset = 0

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

func readMnistLabels(path: String, count: Int) -> [UInt8] {
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url) else {
        ColoredPrint.error("Failed to load MNIST labels from '\(path)'")
        ColoredPrint.info("→ Download MNIST data: run './download_mnist.sh' from project root")
        ColoredPrint.info("→ Or ensure MNIST data exists in ./data/")
        ColoredPrint.info("→ Expected file format: IDX1-ubyte (MNIST label format)")
        exit(1)
    }

    return data.withUnsafeBytes { rawBuf in
        guard let base = rawBuf.bindMemory(to: UInt8.self).baseAddress else {
            return []
        }
        var offset = 0

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

// =============================================================================
// MARK: - MNIST Constants and Configuration
// =============================================================================

let imgH = 28
let imgW = 28
let numInputs = imgH * imgW
let numClasses = 10
let trainSamples = 60_000
let testSamples = 10_000

let patch = 4
let grid = imgH / patch
let seqLen = grid * grid
let patchDim = patch * patch
let dModel = 32
let ffDim = 64

// =============================================================================
// MARK: - Model Structure
// =============================================================================

struct AttnModel {
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
}

// =============================================================================
// MARK: - Forward Pass Functions
// =============================================================================

func extractPatches(batchInputs: [Float], batchCount: Int, patchesOut: inout [Float]) {
    for b in 0..<batchCount {
        let imgBase = b * numInputs
        let patchBase = b * seqLen * patchDim

        for r in 0..<grid {
            for c in 0..<grid {
                let tokenIdx = r * grid + c
                let outBase = patchBase + tokenIdx * patchDim

                for py in 0..<patch {
                    for px in 0..<patch {
                        let iy = r * patch + py
                        let ix = c * patch + px
                        let inIdx = imgBase + iy * imgW + ix
                        let outIdx = outBase + py * patch + px
                        patchesOut[outIdx] = batchInputs[inIdx]
                    }
                }
            }
        }
    }
}

func makeTokens(model: AttnModel, batchCount: Int, patches: [Float], tokens: inout [Float]) {
    for b in 0..<batchCount {
        let patchBase = b * seqLen * patchDim
        let tokenBase = b * seqLen * dModel

        for t in 0..<seqLen {
            let pBase = patchBase + t * patchDim
            let tBase = tokenBase + t * dModel
            let posBase = t * dModel

            for d in 0..<dModel {
                var sum = model.bPatch[d] + model.pos[posBase + d]
                for p in 0..<patchDim {
                    sum += patches[pBase + p] * model.wPatch[p * dModel + d]
                }
                tokens[tBase + d] = max(0, sum)
            }
        }
    }
}

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
    let scale = 1.0 / sqrtf(Float(dModel))

    for b in 0..<batchCount {
        let tokBase = b * seqLen * dModel
        let qBase = b * seqLen * dModel
        let kBase = b * seqLen * dModel
        let vBase = b * seqLen * dModel

        for t in 0..<seqLen {
            let tBase = tokBase + t * dModel
            let qOut = qBase + t * dModel
            let kOut = kBase + t * dModel
            let vOut = vBase + t * dModel

            for d in 0..<dModel {
                var qSum = model.bQ[d]
                var kSum = model.bK[d]
                var vSum = model.bV[d]
                for dd in 0..<dModel {
                    let inp = tokens[tBase + dd]
                    qSum += inp * model.wQ[dd * dModel + d]
                    kSum += inp * model.wK[dd * dModel + d]
                    vSum += inp * model.wV[dd * dModel + d]
                }
                q[qOut + d] = qSum
                k[kOut + d] = kSum
                v[vOut + d] = vSum
            }
        }
    }

    for b in 0..<batchCount {
        let qBase = b * seqLen * dModel
        let kBase = b * seqLen * dModel
        let aBase = b * seqLen * seqLen

        for i in 0..<seqLen {
            let qRow = qBase + i * dModel
            let aRow = aBase + i * seqLen

            var maxScore = -Float.greatestFiniteMagnitude
            for j in 0..<seqLen {
                let kRow = kBase + j * dModel
                var dot: Float = 0
                for d in 0..<dModel {
                    dot += q[qRow + d] * k[kRow + d]
                }
                let score = dot * scale
                attn[aRow + j] = score
                if score > maxScore { maxScore = score }
            }

            var sum: Float = 0
            for j in 0..<seqLen {
                let val = expf(attn[aRow + j] - maxScore)
                attn[aRow + j] = val
                sum += val
            }
            let invSum = 1.0 / sum
            for j in 0..<seqLen {
                attn[aRow + j] *= invSum
            }
        }
    }

    for b in 0..<batchCount {
        let vBase = b * seqLen * dModel
        let aBase = b * seqLen * seqLen
        let oBase = b * seqLen * dModel

        for i in 0..<seqLen {
            let aRow = aBase + i * seqLen
            let oRow = oBase + i * dModel

            for d in 0..<dModel {
                var sum: Float = 0
                for j in 0..<seqLen {
                    sum += attn[aRow + j] * v[vBase + j * dModel + d]
                }
                attnOut[oRow + d] = sum
            }
        }
    }
}

func feedForward(
    model: AttnModel,
    batchCount: Int,
    attnOut: [Float],
    ffn1: inout [Float],
    ffn2: inout [Float]
) {
    for b in 0..<batchCount {
        let inBase = b * seqLen * dModel
        let ff1Base = b * seqLen * ffDim

        for t in 0..<seqLen {
            let inRow = inBase + t * dModel
            let ff1Row = ff1Base + t * ffDim

            for f in 0..<ffDim {
                var sum = model.bFf1[f]
                for d in 0..<dModel {
                    sum += attnOut[inRow + d] * model.wFf1[d * ffDim + f]
                }
                ffn1[ff1Row + f] = max(0, sum)
            }
        }
    }

    for b in 0..<batchCount {
        let ff1Base = b * seqLen * ffDim
        let ff2Base = b * seqLen * dModel

        for t in 0..<seqLen {
            let ff1Row = ff1Base + t * ffDim
            let ff2Row = ff2Base + t * dModel

            for d in 0..<dModel {
                var sum = model.bFf2[d]
                for f in 0..<ffDim {
                    sum += ffn1[ff1Row + f] * model.wFf2[f * dModel + d]
                }
                ffn2[ff2Row + d] = sum
            }
        }
    }
}

func meanPoolTokens(batchCount: Int, tokens: [Float], pooled: inout [Float]) {
    let invSeq = 1.0 / Float(seqLen)
    for b in 0..<batchCount {
        let tokBase = b * seqLen * dModel
        let pBase = b * dModel

        for d in 0..<dModel {
            var sum: Float = 0
            for t in 0..<seqLen {
                sum += tokens[tokBase + t * dModel + d]
            }
            pooled[pBase + d] = sum * invSeq
        }
    }
}

// =============================================================================
// MARK: - Loss and Backward Pass Functions
// =============================================================================

func softmaxXentLoss(
    model: AttnModel,
    batchCount: Int,
    pooled: [Float],
    labels: [UInt8],
    logits: inout [Float],
    probs: inout [Float],
    deltaLogits: inout [Float]
) -> Float {
    for b in 0..<batchCount {
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

    var totalLoss: Float = 0
    let eps: Float = 1e-9

    for b in 0..<batchCount {
        let lBase = b * numClasses
        var maxVal = logits[lBase]
        for c in 1..<numClasses {
            if logits[lBase + c] > maxVal { maxVal = logits[lBase + c] }
        }

        var sumExp: Float = 0
        for c in 0..<numClasses {
            let val = expf(logits[lBase + c] - maxVal)
            probs[lBase + c] = val
            sumExp += val
        }
        let invSum = 1.0 / sumExp
        for c in 0..<numClasses {
            probs[lBase + c] *= invSum
        }

        let y = Int(labels[b])
        let p = max(probs[lBase + y], eps)
        totalLoss += -logf(p)

        for c in 0..<numClasses {
            deltaLogits[lBase + c] = probs[lBase + c]
        }
        deltaLogits[lBase + y] -= 1.0
    }

    let scale = 1.0 / Float(batchCount)
    for i in 0..<(batchCount * numClasses) {
        deltaLogits[i] *= scale
    }

    return totalLoss
}

func classifierBackward(
    model: AttnModel,
    batchCount: Int,
    pooled: [Float],
    deltaLogits: [Float],
    grads: inout Grads,
    deltaPooled: inout [Float]
) {
    for i in 0..<grads.wCls.count { grads.wCls[i] = 0 }
    for i in 0..<grads.bCls.count { grads.bCls[i] = 0 }

    for b in 0..<batchCount {
        let pBase = b * dModel
        let dBase = b * numClasses

        for c in 0..<numClasses {
            grads.bCls[c] += deltaLogits[dBase + c]
        }

        for d in 0..<dModel {
            for c in 0..<numClasses {
                grads.wCls[d * numClasses + c] += pooled[pBase + d] * deltaLogits[dBase + c]
            }
        }
    }

    for i in 0..<(batchCount * dModel) { deltaPooled[i] = 0 }

    for b in 0..<batchCount {
        let pBase = b * dModel
        let dBase = b * numClasses

        for d in 0..<dModel {
            var sum: Float = 0
            for c in 0..<numClasses {
                sum += deltaLogits[dBase + c] * model.wCls[d * numClasses + c]
            }
            deltaPooled[pBase + d] = sum
        }
    }
}

func meanPoolBackward(batchCount: Int, deltaPooled: [Float], deltaTokens: inout [Float]) {
    let val = 1.0 / Float(seqLen)
    for b in 0..<batchCount {
        let dPoolBase = b * dModel
        let dTokBase = b * seqLen * dModel

        for d in 0..<dModel {
            let grad = deltaPooled[dPoolBase + d] * val
            for t in 0..<seqLen {
                deltaTokens[dTokBase + t * dModel + d] = grad
            }
        }
    }
}

func feedForwardBackward(
    model: AttnModel,
    batchCount: Int,
    attnOut: [Float],
    ffn1: [Float],
    deltaFfn2: [Float],
    grads: inout Grads,
    deltaAttnOut: inout [Float]
) {
    for i in 0..<grads.wFf2.count { grads.wFf2[i] = 0 }
    for i in 0..<grads.bFf2.count { grads.bFf2[i] = 0 }

    for b in 0..<batchCount {
        let ff1Base = b * seqLen * ffDim
        let dBase = b * seqLen * dModel

        for t in 0..<seqLen {
            let ff1Row = ff1Base + t * ffDim
            let dRow = dBase + t * dModel

            for d in 0..<dModel {
                grads.bFf2[d] += deltaFfn2[dRow + d]
            }

            for f in 0..<ffDim {
                for d in 0..<dModel {
                    grads.wFf2[f * dModel + d] += ffn1[ff1Row + f] * deltaFfn2[dRow + d]
                }
            }
        }
    }

    var deltaFfn1 = [Float](repeating: 0, count: batchCount * seqLen * ffDim)

    for b in 0..<batchCount {
        let dBase = b * seqLen * dModel
        let dff1Base = b * seqLen * ffDim

        for t in 0..<seqLen {
            let dRow = dBase + t * dModel
            let dff1Row = dff1Base + t * ffDim

            for f in 0..<ffDim {
                var sum: Float = 0
                for d in 0..<dModel {
                    sum += deltaFfn2[dRow + d] * model.wFf2[f * dModel + d]
                }
                deltaFfn1[dff1Row + f] = sum
            }
        }
    }

    for i in 0..<grads.wFf1.count { grads.wFf1[i] = 0 }
    for i in 0..<grads.bFf1.count { grads.bFf1[i] = 0 }

    for b in 0..<batchCount {
        let inBase = b * seqLen * dModel
        let ff1Base = b * seqLen * ffDim
        let dff1Base = b * seqLen * ffDim

        for t in 0..<seqLen {
            let inRow = inBase + t * dModel
            let ff1Row = ff1Base + t * ffDim
            let dff1Row = dff1Base + t * ffDim

            for f in 0..<ffDim {
                let g = (ffn1[ff1Row + f] > 0) ? deltaFfn1[dff1Row + f] : 0
                grads.bFf1[f] += g
                for d in 0..<dModel {
                    grads.wFf1[d * ffDim + f] += attnOut[inRow + d] * g
                }
            }
        }
    }

    for i in 0..<(batchCount * seqLen * dModel) { deltaAttnOut[i] = 0 }

    for b in 0..<batchCount {
        let inBase = b * seqLen * dModel
        let ff1Base = b * seqLen * ffDim
        let dff1Base = b * seqLen * ffDim

        for t in 0..<seqLen {
            let inRow = inBase + t * dModel
            let ff1Row = ff1Base + t * ffDim
            let dff1Row = dff1Base + t * ffDim

            for d in 0..<dModel {
                var sum: Float = 0
                for f in 0..<ffDim {
                    let g = (ffn1[ff1Row + f] > 0) ? deltaFfn1[dff1Row + f] : 0
                    sum += g * model.wFf1[d * ffDim + f]
                }
                deltaAttnOut[inRow + d] = sum
            }
        }
    }
}

func selfAttentionBackward(
    model: AttnModel,
    batchCount: Int,
    tokens: [Float],
    q: [Float],
    k: [Float],
    v: [Float],
    attn: [Float],
    deltaAttnOut: [Float],
    grads: inout Grads,
    deltaTokens: inout [Float]
) {
    let scale = 1.0 / sqrtf(Float(dModel))

    var deltaV = [Float](repeating: 0, count: batchCount * seqLen * dModel)
    var deltaAttn = [Float](repeating: 0, count: batchCount * seqLen * seqLen)

    for b in 0..<batchCount {
        let vBase = b * seqLen * dModel
        let aBase = b * seqLen * seqLen
        let dOutBase = b * seqLen * dModel
        let dVBase = b * seqLen * dModel
        let dABase = b * seqLen * seqLen

        for i in 0..<seqLen {
            let dOutRow = dOutBase + i * dModel
            let aRow = aBase + i * seqLen
            let dARow = dABase + i * seqLen

            for d in 0..<dModel {
                let grad = deltaAttnOut[dOutRow + d]
                for j in 0..<seqLen {
                    deltaV[dVBase + j * dModel + d] += grad * attn[aRow + j]
                    deltaAttn[dARow + j] += grad * v[vBase + j * dModel + d]
                }
            }
        }
    }

    var deltaQ = [Float](repeating: 0, count: batchCount * seqLen * dModel)
    var deltaK = [Float](repeating: 0, count: batchCount * seqLen * dModel)

    for b in 0..<batchCount {
        let qBase = b * seqLen * dModel
        let kBase = b * seqLen * dModel
        let aBase = b * seqLen * seqLen
        let dABase = b * seqLen * seqLen
        let dQBase = b * seqLen * dModel
        let dKBase = b * seqLen * dModel

        for i in 0..<seqLen {
            let aRow = aBase + i * seqLen
            let dARow = dABase + i * seqLen
            let qRow = qBase + i * dModel
            let dQRow = dQBase + i * dModel

            for j in 0..<seqLen {
                let dSoft = deltaAttn[dARow + j]
                var sum: Float = 0
                for jj in 0..<seqLen {
                    sum += deltaAttn[dARow + jj] * attn[aRow + jj]
                }
                let dScore = attn[aRow + j] * (dSoft - sum)

                let kRow = kBase + j * dModel
                let dKRow = dKBase + j * dModel

                for d in 0..<dModel {
                    deltaQ[dQRow + d] += dScore * k[kRow + d] * scale
                    deltaK[dKRow + d] += dScore * q[qRow + d] * scale
                }
            }
        }
    }

    for i in 0..<grads.wV.count { grads.wV[i] = 0 }
    for i in 0..<grads.bV.count { grads.bV[i] = 0 }
    for i in 0..<grads.wK.count { grads.wK[i] = 0 }
    for i in 0..<grads.bK.count { grads.bK[i] = 0 }
    for i in 0..<grads.wQ.count { grads.wQ[i] = 0 }
    for i in 0..<grads.bQ.count { grads.bQ[i] = 0 }

    for b in 0..<batchCount {
        let tokBase = b * seqLen * dModel
        let dVBase = b * seqLen * dModel
        let dKBase = b * seqLen * dModel
        let dQBase = b * seqLen * dModel

        for t in 0..<seqLen {
            let tRow = tokBase + t * dModel
            let dVRow = dVBase + t * dModel
            let dKRow = dKBase + t * dModel
            let dQRow = dQBase + t * dModel

            for d in 0..<dModel {
                grads.bV[d] += deltaV[dVRow + d]
                grads.bK[d] += deltaK[dKRow + d]
                grads.bQ[d] += deltaQ[dQRow + d]

                for dd in 0..<dModel {
                    grads.wV[dd * dModel + d] += tokens[tRow + dd] * deltaV[dVRow + d]
                    grads.wK[dd * dModel + d] += tokens[tRow + dd] * deltaK[dKRow + d]
                    grads.wQ[dd * dModel + d] += tokens[tRow + dd] * deltaQ[dQRow + d]
                }
            }
        }
    }

    for i in 0..<deltaTokens.count { deltaTokens[i] = 0 }

    for b in 0..<batchCount {
        let dVBase = b * seqLen * dModel
        let dKBase = b * seqLen * dModel
        let dQBase = b * seqLen * dModel
        let dTokBase = b * seqLen * dModel

        for t in 0..<seqLen {
            let dVRow = dVBase + t * dModel
            let dKRow = dKBase + t * dModel
            let dQRow = dQBase + t * dModel
            let dTokRow = dTokBase + t * dModel

            for dd in 0..<dModel {
                var sum: Float = 0
                for d in 0..<dModel {
                    sum += deltaV[dVRow + d] * model.wV[dd * dModel + d]
                    sum += deltaK[dKRow + d] * model.wK[dd * dModel + d]
                    sum += deltaQ[dQRow + d] * model.wQ[dd * dModel + d]
                }
                deltaTokens[dTokRow + dd] += sum
            }
        }
    }
}

func makeTokensBackward(
    model: AttnModel,
    batchCount: Int,
    patches: [Float],
    tokens: [Float],
    deltaTokens: [Float],
    grads: inout Grads
) {
    for i in 0..<grads.wPatch.count { grads.wPatch[i] = 0 }
    for i in 0..<grads.bPatch.count { grads.bPatch[i] = 0 }
    for i in 0..<grads.pos.count { grads.pos[i] = 0 }

    for b in 0..<batchCount {
        let patchBase = b * seqLen * patchDim
        let tokenBase = b * seqLen * dModel
        let dTokBase = b * seqLen * dModel

        for t in 0..<seqLen {
            let pBase = patchBase + t * patchDim
            let tBase = tokenBase + t * dModel
            let dBase = dTokBase + t * dModel
            let posBase = t * dModel

            for d in 0..<dModel {
                let grad = (tokens[tBase + d] > 0) ? deltaTokens[dBase + d] : 0

                grads.bPatch[d] += grad
                grads.pos[posBase + d] += grad

                for p in 0..<patchDim {
                    grads.wPatch[p * dModel + d] += patches[pBase + p] * grad
                }
            }
        }
    }
}

// =============================================================================
// MARK: - Training Functions
// =============================================================================

func trainEpoch(
    model: inout AttnModel,
    images: [Float],
    labels: [UInt8],
    indices: inout [Int],
    rng: inout SimpleRng,
    batchSize: Int,
    learningRate: Float
) -> Float {
    rng.shuffle(&indices)

    var grads = Grads()

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
    var deltaLogits = [Float](repeating: 0, count: batchSize * numClasses)
    var deltaPooled = [Float](repeating: 0, count: batchSize * dModel)
    var deltaFfn2   = [Float](repeating: 0, count: batchSize * seqLen * dModel)
    var deltaAttnOut = [Float](repeating: 0, count: batchSize * seqLen * dModel)
    var deltaTokens  = [Float](repeating: 0, count: batchSize * seqLen * dModel)

    var totalLoss: Float = 0
    var start = 0

    while start < indices.count {
        let bsz = min(batchSize, indices.count - start)

        for i in 0..<bsz {
            let srcIndex = indices[start + i]
            let srcBase = srcIndex * numInputs
            let dstBase = i * numInputs
            for j in 0..<numInputs {
                batchInputs[dstBase + j] = images[srcBase + j]
            }
            batchLabels[i] = labels[srcIndex]
        }

        extractPatches(batchInputs: batchInputs, batchCount: bsz, patchesOut: &patches)
        makeTokens(model: model, batchCount: bsz, patches: patches, tokens: &tokens)
        selfAttention(model: model, batchCount: bsz, tokens: tokens, q: &q, k: &k, v: &v, attn: &attn, attnOut: &attnOut)
        feedForward(model: model, batchCount: bsz, attnOut: attnOut, ffn1: &ffn1, ffn2: &ffn2)
        meanPoolTokens(batchCount: bsz, tokens: ffn2, pooled: &pooled)

        let loss = softmaxXentLoss(
            model: model, batchCount: bsz, pooled: pooled, labels: batchLabels,
            logits: &logits, probs: &probs, deltaLogits: &deltaLogits
        )
        totalLoss += loss

        classifierBackward(model: model, batchCount: bsz, pooled: pooled, deltaLogits: deltaLogits, grads: &grads, deltaPooled: &deltaPooled)
        meanPoolBackward(batchCount: bsz, deltaPooled: deltaPooled, deltaTokens: &deltaFfn2)
        feedForwardBackward(model: model, batchCount: bsz, attnOut: attnOut, ffn1: ffn1, deltaFfn2: deltaFfn2, grads: &grads, deltaAttnOut: &deltaAttnOut)
        selfAttentionBackward(model: model, batchCount: bsz, tokens: tokens, q: q, k: k, v: v, attn: attn, deltaAttnOut: deltaAttnOut, grads: &grads, deltaTokens: &deltaTokens)
        makeTokensBackward(model: model, batchCount: bsz, patches: patches, tokens: tokens, deltaTokens: deltaTokens, grads: &grads)

        for i in 0..<model.wPatch.count { model.wPatch[i] -= learningRate * grads.wPatch[i] }
        for i in 0..<model.bPatch.count { model.bPatch[i] -= learningRate * grads.bPatch[i] }
        for i in 0..<model.pos.count { model.pos[i] -= learningRate * grads.pos[i] }
        for i in 0..<model.wQ.count { model.wQ[i] -= learningRate * grads.wQ[i] }
        for i in 0..<model.bQ.count { model.bQ[i] -= learningRate * grads.bQ[i] }
        for i in 0..<model.wK.count { model.wK[i] -= learningRate * grads.wK[i] }
        for i in 0..<model.bK.count { model.bK[i] -= learningRate * grads.bK[i] }
        for i in 0..<model.wV.count { model.wV[i] -= learningRate * grads.wV[i] }
        for i in 0..<model.bV.count { model.bV[i] -= learningRate * grads.bV[i] }
        for i in 0..<model.wFf1.count { model.wFf1[i] -= learningRate * grads.wFf1[i] }
        for i in 0..<model.bFf1.count { model.bFf1[i] -= learningRate * grads.bFf1[i] }
        for i in 0..<model.wFf2.count { model.wFf2[i] -= learningRate * grads.wFf2[i] }
        for i in 0..<model.bFf2.count { model.bFf2[i] -= learningRate * grads.bFf2[i] }
        for i in 0..<model.wCls.count { model.wCls[i] -= learningRate * grads.wCls[i] }
        for i in 0..<model.bCls.count { model.bCls[i] -= learningRate * grads.bCls[i] }

        start += bsz
    }

    return totalLoss / Float(indices.count)
}

// =============================================================================
// MARK: - Model Initialization
// =============================================================================

func initModel(rng: inout SimpleRng) -> AttnModel {
    func xavier(_ fanIn: Int, _ fanOut: Int) -> Float {
        return sqrtf(6.0 / Float(fanIn + fanOut))
    }

    func initArray(count: Int, limit: Float, rng: inout SimpleRng) -> [Float] {
        var arr = [Float](repeating: 0, count: count)
        for i in 0..<count {
            arr[i] = rng.uniform(-limit, limit)
        }
        return arr
    }

    let patchLimit = xavier(patchDim, dModel)
    let wPatch = initArray(count: patchDim * dModel, limit: patchLimit, rng: &rng)
    let bPatch = [Float](repeating: 0, count: dModel)

    let posLimit = xavier(1, dModel)
    let pos = initArray(count: seqLen * dModel, limit: posLimit, rng: &rng)

    let attnLimit = xavier(dModel, dModel)
    let wQ = initArray(count: dModel * dModel, limit: attnLimit, rng: &rng)
    let bQ = [Float](repeating: 0, count: dModel)
    let wK = initArray(count: dModel * dModel, limit: attnLimit, rng: &rng)
    let bK = [Float](repeating: 0, count: dModel)
    let wV = initArray(count: dModel * dModel, limit: attnLimit, rng: &rng)
    let bV = [Float](repeating: 0, count: dModel)

    let ff1Limit = xavier(dModel, ffDim)
    let wFf1 = initArray(count: dModel * ffDim, limit: ff1Limit, rng: &rng)
    let bFf1 = [Float](repeating: 0, count: ffDim)

    let ff2Limit = xavier(ffDim, dModel)
    let wFf2 = initArray(count: ffDim * dModel, limit: ff2Limit, rng: &rng)
    let bFf2 = [Float](repeating: 0, count: dModel)

    let clsLimit = xavier(dModel, numClasses)
    let wCls = initArray(count: dModel * numClasses, limit: clsLimit, rng: &rng)
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

// =============================================================================
// MARK: - Save/Load Functions
// =============================================================================

func saveModel(model: AttnModel, filename: String) {
    FileManager.default.createFile(atPath: filename, contents: nil)
    guard let handle = try? FileHandle(forWritingTo: URL(fileURLWithPath: filename)) else {
        ColoredPrint.error("Failed to open '\(filename)' for writing")
        ColoredPrint.info("→ Check if you have write permissions in the current directory")
        ColoredPrint.info("→ Ensure the path is valid and the disk is not full")
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

    writeInt32(Int32(patchDim))
    writeInt32(Int32(dModel))
    writeInt32(Int32(seqLen))
    writeInt32(Int32(ffDim))
    writeInt32(Int32(numClasses))

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
}

func loadModel(filename: String) -> AttnModel? {
    guard let handle = try? FileHandle(forReadingFrom: URL(fileURLWithPath: filename)) else {
        ColoredPrint.error("Failed to open '\(filename)' for reading")
        ColoredPrint.info("→ Check if the file exists in the current directory")
        ColoredPrint.info("→ Ensure you have read permissions for this file")
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

    guard let patchDimRead = readInt32(),
          let dModelRead = readInt32(),
          let seqLenRead = readInt32(),
          let ffDimRead = readInt32(),
          let numClassesRead = readInt32() else {
        ColoredPrint.error("Failed to read model header from '\(filename)'")
        ColoredPrint.info("→ The file may be corrupted or truncated")
        ColoredPrint.info("→ Ensure the file was saved completely")
        ColoredPrint.info("→ Try re-saving the model")
        return nil
    }

    if patchDimRead != Int32(patchDim) || dModelRead != Int32(dModel) ||
       seqLenRead != Int32(seqLen) || ffDimRead != Int32(ffDim) ||
       numClassesRead != Int32(numClasses) {
        ColoredPrint.error("Model architecture mismatch")
        ColoredPrint.info("→ Expected: patchDim=\(patchDim), dModel=\(dModel), seqLen=\(seqLen), ffDim=\(ffDim), classes=\(numClasses)")
        ColoredPrint.info("→ Found in file: patchDim=\(patchDimRead), dModel=\(dModelRead), seqLen=\(seqLenRead), ffDim=\(ffDimRead), classes=\(numClassesRead)")
        ColoredPrint.info("→ Ensure you're loading an Attention model file, not a different architecture")
        return nil
    }

    var wPatch = [Float](repeating: 0, count: patchDim * dModel)
    for i in 0..<wPatch.count {
        guard let val = readDouble() else { return nil }
        wPatch[i] = Float(val)
    }

    var bPatch = [Float](repeating: 0, count: dModel)
    for i in 0..<bPatch.count {
        guard let val = readDouble() else { return nil }
        bPatch[i] = Float(val)
    }

    var pos = [Float](repeating: 0, count: seqLen * dModel)
    for i in 0..<pos.count {
        guard let val = readDouble() else { return nil }
        pos[i] = Float(val)
    }

    var wQ = [Float](repeating: 0, count: dModel * dModel)
    for i in 0..<wQ.count {
        guard let val = readDouble() else { return nil }
        wQ[i] = Float(val)
    }

    var bQ = [Float](repeating: 0, count: dModel)
    for i in 0..<bQ.count {
        guard let val = readDouble() else { return nil }
        bQ[i] = Float(val)
    }

    var wK = [Float](repeating: 0, count: dModel * dModel)
    for i in 0..<wK.count {
        guard let val = readDouble() else { return nil }
        wK[i] = Float(val)
    }

    var bK = [Float](repeating: 0, count: dModel)
    for i in 0..<bK.count {
        guard let val = readDouble() else { return nil }
        bK[i] = Float(val)
    }

    var wV = [Float](repeating: 0, count: dModel * dModel)
    for i in 0..<wV.count {
        guard let val = readDouble() else { return nil }
        wV[i] = Float(val)
    }

    var bV = [Float](repeating: 0, count: dModel)
    for i in 0..<bV.count {
        guard let val = readDouble() else { return nil }
        bV[i] = Float(val)
    }

    var wFf1 = [Float](repeating: 0, count: dModel * ffDim)
    for i in 0..<wFf1.count {
        guard let val = readDouble() else { return nil }
        wFf1[i] = Float(val)
    }

    var bFf1 = [Float](repeating: 0, count: ffDim)
    for i in 0..<bFf1.count {
        guard let val = readDouble() else { return nil }
        bFf1[i] = Float(val)
    }

    var wFf2 = [Float](repeating: 0, count: ffDim * dModel)
    for i in 0..<wFf2.count {
        guard let val = readDouble() else { return nil }
        wFf2[i] = Float(val)
    }

    var bFf2 = [Float](repeating: 0, count: dModel)
    for i in 0..<bFf2.count {
        guard let val = readDouble() else { return nil }
        bFf2[i] = Float(val)
    }

    var wCls = [Float](repeating: 0, count: dModel * numClasses)
    for i in 0..<wCls.count {
        guard let val = readDouble() else { return nil }
        wCls[i] = Float(val)
    }

    var bCls = [Float](repeating: 0, count: numClasses)
    for i in 0..<bCls.count {
        guard let val = readDouble() else { return nil }
        bCls[i] = Float(val)
    }

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
// MARK: - Prediction Function
// =============================================================================

func predict(model: AttnModel, images: [Float], count: Int) -> [Int] {
    var predictions = [Int]()

    var patches = [Float](repeating: 0, count: seqLen * patchDim)
    var tokens  = [Float](repeating: 0, count: seqLen * dModel)
    var q       = [Float](repeating: 0, count: seqLen * dModel)
    var k       = [Float](repeating: 0, count: seqLen * dModel)
    var v       = [Float](repeating: 0, count: seqLen * dModel)
    var attn    = [Float](repeating: 0, count: seqLen * seqLen)
    var attnOut = [Float](repeating: 0, count: seqLen * dModel)
    var ffn1    = [Float](repeating: 0, count: seqLen * ffDim)
    var ffn2    = [Float](repeating: 0, count: seqLen * dModel)
    var pooled  = [Float](repeating: 0, count: dModel)
    var logits  = [Float](repeating: 0, count: numClasses)

    for i in 0..<count {
        let imgStart = i * numInputs
        let imgEnd = imgStart + numInputs
        let input = Array(images[imgStart..<imgEnd])

        extractPatches(batchInputs: input, batchCount: 1, patchesOut: &patches)
        makeTokens(model: model, batchCount: 1, patches: patches, tokens: &tokens)
        selfAttention(model: model, batchCount: 1, tokens: tokens, q: &q, k: &k, v: &v, attn: &attn, attnOut: &attnOut)
        feedForward(model: model, batchCount: 1, attnOut: attnOut, ffn1: &ffn1, ffn2: &ffn2)
        meanPoolTokens(batchCount: 1, tokens: ffn2, pooled: &pooled)

        for c in 0..<numClasses {
            var sum = model.bCls[c]
            for d in 0..<dModel {
                sum += pooled[d] * model.wCls[d * numClasses + c]
            }
            logits[c] = sum
        }

        var best = logits[0]
        var arg = 0
        for j in 1..<numClasses {
            if logits[j] > best {
                best = logits[j]
                arg = j
            }
        }
        predictions.append(arg)
    }

    return predictions
}

// =============================================================================
// MARK: - Main Verification
// =============================================================================

func main() {
    print("=== Attention Save/Load Round-Trip Verification ===\n")

    let dataPath = "./data"
    let modelFile = "mnist_attention_model_test.bin"
    let batchSize = 32
    let learningRate: Float = 0.005
    let epochs = 1
    let trainSubset = 1000
    let testCount = 100

    print("Loading MNIST data...")
    let trainImages = readMnistImages(path: "\(dataPath)/train-images.idx3-ubyte", count: trainSubset)
    let trainLabels = readMnistLabels(path: "\(dataPath)/train-labels.idx1-ubyte", count: trainSubset)
    let testImages = readMnistImages(path: "\(dataPath)/t10k-images.idx3-ubyte", count: testSamples)
    let testLabels = readMnistLabels(path: "\(dataPath)/t10k-labels.idx1-ubyte", count: testSamples)
    print("Loaded \(trainLabels.count) training samples, \(testLabels.count) test samples\n")

    print("Initializing Attention model...")
    var rng = SimpleRng(seed: 42)
    var model = initModel(rng: &rng)

    print("Training for \(epochs) epoch...\n")

    var indices = Array(0..<trainLabels.count)

    for e in 0..<epochs {
        let avgLoss = trainEpoch(
            model: &model,
            images: trainImages,
            labels: trainLabels,
            indices: &indices,
            rng: &rng,
            batchSize: batchSize,
            learningRate: learningRate
        )
        print(String(format: "Epoch %d completed | loss=%.6f", e + 1, avgLoss))
    }

    print("\nGetting predictions from original trained model...")
    let originalPredictions = predict(model: model, images: testImages, count: testCount)

    print("Saving model to \(modelFile)...")
    saveModel(model: model, filename: modelFile)

    print("Loading model from \(modelFile)...")
    guard let loadedModel = loadModel(filename: modelFile) else {
        ColoredPrint.error("Verification FAILED: Unable to load saved model")
        ColoredPrint.info("→ The save operation may have failed silently")
        ColoredPrint.info("→ Check disk space and file permissions")
        exit(1)
    }

    print("Getting predictions from loaded model...\n")
    let loadedPredictions = predict(model: loadedModel, images: testImages, count: testCount)

    print("=== Verification Results ===")
    var matches = 0
    var mismatches = 0

    for i in 0..<testCount {
        if originalPredictions[i] == loadedPredictions[i] {
            matches += 1
        } else {
            mismatches += 1
            print("Mismatch at index \(i): original=\(originalPredictions[i]), loaded=\(loadedPredictions[i])")
        }
    }

    print("\nResults:")
    print("  Total samples tested: \(testCount)")
    print("  Matching predictions: \(matches)")
    print("  Mismatched predictions: \(mismatches)")

    if mismatches == 0 {
        ColoredPrint.success("\nSUCCESS: All predictions match! Save/load round-trip works correctly.")

        var correct = 0
        for i in 0..<testCount {
            if loadedPredictions[i] == Int(testLabels[i]) {
                correct += 1
            }
        }
        let accuracy = 100.0 * Float(correct) / Float(testCount)
        print(String(format: "  Model accuracy on test samples: %.2f%%", accuracy))

        try? FileManager.default.removeItem(atPath: modelFile)
        print("\nTest model file cleaned up.")
    } else {
        ColoredPrint.error("\nVerification FAILED: Predictions do not match after save/load")
        ColoredPrint.info("→ This indicates the save/load functions have a bug")
        ColoredPrint.info("→ Model weights may not be preserved correctly")
        ColoredPrint.info("→ Check saveModel() and loadModel() implementations")
        exit(1)
    }
}

main()
