import Foundation
import MNISTCommon

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

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

    /// Sets every gradient buffer in the struct to zero, mutating the instance in place.
    mutating func zero() {
        wPatch = [Float](repeating: 0, count: wPatch.count)
        bPatch = [Float](repeating: 0, count: bPatch.count)
        pos    = [Float](repeating: 0, count: pos.count)
        wQ     = [Float](repeating: 0, count: wQ.count)
        bQ     = [Float](repeating: 0, count: bQ.count)
        wK     = [Float](repeating: 0, count: wK.count)
        bK     = [Float](repeating: 0, count: bK.count)
        wV     = [Float](repeating: 0, count: wV.count)
        bV     = [Float](repeating: 0, count: bV.count)
        wFf1   = [Float](repeating: 0, count: wFf1.count)
        bFf1   = [Float](repeating: 0, count: bFf1.count)
        wFf2   = [Float](repeating: 0, count: wFf2.count)
        bFf2   = [Float](repeating: 0, count: bFf2.count)
        wCls   = [Float](repeating: 0, count: wCls.count)
        bCls   = [Float](repeating: 0, count: bCls.count)
    }
}

/// Creates an `AttnModel` with all learnable parameters initialized for training.
/// 
/// Weight matrices are initialized using Xavier-style uniform sampling with layer-specific limits; biases are initialized to zeros. Positional embeddings are sampled uniformly from -0.1 to 0.1.
/// - Parameters:
///   - rng: Pseudo-random number generator used to sample initial parameter values.
/// - Returns: An `AttnModel` whose weight arrays are Xavier-uniform initialized, biases are zeroed, and positional embeddings are uniformly sampled in [-0.1, 0.1].
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
