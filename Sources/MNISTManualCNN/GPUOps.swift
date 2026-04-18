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

#if canImport(MetalPerformanceShaders)
/// Performs a GPU convolution forward pass by running im2col, computing GEMM, then adding bias and applying ReLU.
/// 
/// The function records and commits a Metal command buffer that:
/// 1. Transforms `input` into a column buffer using im2col.
/// 2. Multiplies `convW` by the column buffer to produce intermediate GEMM results.
/// 3. Transposes the GEMM output into NCHW output layout, adds `convB`, and applies ReLU into `convOutAct`.
/// The command buffer is committed and the function waits for completion before returning.
/// - Parameters:
///   - engine: MPS GEMM engine and command queue used to encode GPU work.
///   - kernels: Metal kernels used for im2col and bias/ReLU post-processing.
///   - batch: Number of examples in the batch.
///   - input: Input tensor buffer with layout [batch, 1, imgH, imgW].
///   - convW: Convolution weight buffer with layout [convOut, kernel*kernel*inChannels].
///   - convB: Convolution bias buffer with layout [convOut].
///   - convOutAct: Destination buffer for activated convolution output with layout [batch, convOut, imgH*imgW] (or [batch, convOut, imgH, imgW] depending on consumer layout).
///   - colBuffer: Temporary column buffer for im2col with layout [kernel*kernel*inChannels, batch*imgH*imgW].
///   - gemmTemp: Temporary GEMM output buffer with layout [convOut, batch*imgH*imgW].
func convForwardReluGpu(
    engine: MpsGemmEngine,
    kernels: MpsKernels,
    batch: Int,
    input: MpsBuffer,
    convW: MpsBuffer,
    convB: MpsBuffer,
    convOutAct: MpsBuffer,
    colBuffer: MpsBuffer,
    gemmTemp: MpsBuffer
) throws {
    let commandBuffer = try engine.makeCommandBuffer(operation: "conv forward")

    let colChannels = kernel * kernel * 1  // 9
    let colWidth = imgH * imgW * batch     // 784 * batch
    let spatial = imgH * imgW              // 784
    let outHeight = imgH  // Same as input due to padding
    let outWidth = imgW   // Same as input due to padding

    // Step 1: Transform input using im2col on GPU
    // input: [batch, 1, imgH, imgW] -> colBuffer: [colChannels, colWidth]
    try kernels.encodeIm2col(
        commandBuffer: commandBuffer,
        input: input,
        output: colBuffer,
        batch: batch,
        inChannels: 1,
        inHeight: imgH,
        inWidth: imgW,
        outHeight: outHeight,
        outWidth: outWidth,
        kernelSize: kernel,
        stride: 1,
        padding: pad
    )

    // Step 2: Perform convolution using MPS GEMM
    // result = convW × colBuffer
    // convW: [convOut, colChannels] = [8, 9]
    // colBuffer: [colChannels, colWidth] = [9, 784*batch]
    // gemmTemp: [convOut, colWidth] = [8, 784*batch]
    engine.encodeGemm(
        commandBuffer: commandBuffer,
        m: convOut,
        n: colWidth,
        k: colChannels,
        a: convW,
        b: colBuffer,
        c: gemmTemp,
        transposeA: false,
        transposeB: false,
        alpha: 1.0,
        beta: 0.0
    )

    // Step 3: Transpose from [channels, batch*spatial] to [batch, channels, spatial],
    // add bias, and apply ReLU
    try kernels.encodeConvTransposeBiasRelu(
        commandBuffer: commandBuffer,
        input: gemmTemp,
        output: convOutAct,
        bias: convB,
        batch: batch,
        channels: convOut,
        spatial: spatial
    )

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    try checkMetalCommandBuffer(commandBuffer, operation: "conv forward")
}
#endif

#if canImport(MetalPerformanceShaders)
// GPU version of FC forward using MPS GEMM: logits = X*W + b.
// x: [batch, fcIn] (MpsBuffer)
// fcW: [fcIn, numClasses] (MpsBuffer)
// fcB: [numClasses] (MpsBuffer)
/// Compute the fully-connected layer forward pass producing class logits.
/// - Parameters:
///   - batch: Number of rows (batch size).
///   - x: Input activations with shape [batch, fcIn].
///   - fcW: Weight matrix with shape [fcIn, numClasses].
///   - fcB: Bias vector with length `numClasses`.
///   - logits: Output buffer to receive activations with shape [batch, numClasses].
func fcForwardGpu(
    engine: MpsGemmEngine,
    kernels: MpsKernels,
    batch: Int,
    x: MpsBuffer,
    fcW: MpsBuffer,
    fcB: MpsBuffer,
    logits: MpsBuffer
) throws {
    let commandBuffer = try engine.makeCommandBuffer(operation: "fully-connected forward")

    // Step 1: Matrix multiplication using MPS GEMM
    // logits = x * fcW
    // x: [batch, fcIn]
    // fcW: [fcIn, numClasses]
    // logits: [batch, numClasses]
    engine.encodeGemm(
        commandBuffer: commandBuffer,
        m: batch,
        n: numClasses,
        k: fcIn,
        a: x,
        b: fcW,
        c: logits,
        transposeA: false,
        transposeB: false,
        alpha: 1.0,
        beta: 0.0
    )

    // Step 2: Add bias using Metal kernel
    try kernels.encodeAddBias(
        commandBuffer: commandBuffer,
        data: logits,
        bias: fcB,
        rows: batch,
        cols: numClasses
    )
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    try checkMetalCommandBuffer(commandBuffer, operation: "fully-connected forward")
}

/// Computes fully-connected layer gradients on the GPU.
///
/// The command buffer computes `gradW = x^T * delta`, `gradB = row-wise sum(delta)`, and `dX = delta * fcW^T`, then waits for completion.
/// - Parameters:
///   - engine: MPS GEMM engine and command queue used to encode GPU work.
///   - kernels: Metal kernels used for row-sum reductions.
///   - batch: Number of examples in the batch.
///   - x: Input activations with shape `[batch, fcIn]`.
///   - delta: Upstream gradients (logits gradient) with shape `[batch, numClasses]`.
///   - fcW: Weights with shape `[fcIn, numClasses]`.
///   - gradW: Destination buffer for weight gradients with shape `[fcIn, numClasses]`; overwritten with `x^T * delta`.
///   - gradB: Destination buffer for bias gradients with shape `[numClasses]`; overwritten with row-wise sum of `delta`.
///   - dX: Destination buffer for input gradients with shape `[batch, fcIn]`; overwritten with `delta * fcW^T`.
func fcBackwardGpu(
    engine: MpsGemmEngine,
    kernels: MpsKernels,
    batch: Int,
    x: MpsBuffer,
    delta: MpsBuffer,
    fcW: MpsBuffer,
    gradW: MpsBuffer,
    gradB: MpsBuffer,
    dX: MpsBuffer
) throws {
    let commandBuffer = try engine.makeCommandBuffer(operation: "fully-connected backward")

    // Step 1: Compute weight gradients using MPS GEMM
    // gradW = x^T * delta
    // x^T: [fcIn, batch] (transpose of x: [batch, fcIn])
    // delta: [batch, numClasses]
    // gradW: [fcIn, numClasses]
    engine.encodeGemm(
        commandBuffer: commandBuffer,
        m: fcIn,
        n: numClasses,
        k: batch,
        a: x,
        b: delta,
        c: gradW,
        transposeA: true,
        transposeB: false,
        alpha: 1.0,
        beta: 0.0
    )

    // Step 2: Compute bias gradients by summing delta over batch dimension
    // gradB = sum(delta, axis=0) with scale 1.0
    try kernels.encodeSumRows(
        commandBuffer: commandBuffer,
        data: delta,
        output: gradB,
        rows: batch,
        cols: numClasses,
        scale: 1.0
    )

    // Step 3: Compute input gradients using MPS GEMM
    // dX = delta * W^T
    // delta: [batch, numClasses]
    // W^T: [numClasses, fcIn] (transpose of fcW: [fcIn, numClasses])
    // dX: [batch, fcIn]
    engine.encodeGemm(
        commandBuffer: commandBuffer,
        m: batch,
        n: fcIn,
        k: numClasses,
        a: delta,
        b: fcW,
        c: dX,
        transposeA: false,
        transposeB: true,
        alpha: 1.0,
        beta: 0.0
    )

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    try checkMetalCommandBuffer(commandBuffer, operation: "fully-connected backward")
}

/// Runs 2D max-pooling on the GPU and writes the pooled activations into `output`.
/// 
/// The kernel uses a square pooling window with size and stride equal to `pool`. It expects `input` laid out as `[batch, convOut, imgH, imgW]` and writes `output` with shape `[batch, convOut, poolH, poolW]`, then waits for completion.
/// - Parameters:
///   - engine: MPS GEMM engine and command queue used to encode GPU work.
///   - kernels: Metal kernels used for max pooling.
///   - batch: Number of examples in the batch.
///   - input: Source buffer containing input activations in `[batch, convOut, imgH, imgW]` layout.
///   - output: Destination buffer that will receive pooled activations in `[batch, convOut, poolH, poolW]` layout.
func maxPoolForwardGpu(
    engine: MpsGemmEngine,
    kernels: MpsKernels,
    batch: Int,
    input: MpsBuffer,
    output: MpsBuffer
) throws {
    let commandBuffer = try engine.makeCommandBuffer(operation: "max-pool forward")

    // Perform max pooling using Metal kernel
    try kernels.encodeMaxPoolForward(
        commandBuffer: commandBuffer,
        input: input,
        output: output,
        batch: batch,
        channels: convOut,
        inHeight: imgH,
        inWidth: imgW,
        outHeight: poolH,
        outWidth: poolW,
        poolSize: pool,
        stride: pool
    )

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    try checkMetalCommandBuffer(commandBuffer, operation: "max-pool forward")
}

/// Computes max-pooling backward gradients on the GPU and applies the ReLU mask.
/// 
/// `maxPoolBackwardReluGpu` zeroes `convGrad`, accumulates max-pool backward gradients from `poolGrad` using `convAct` to recover max locations, then zeros entries where `convAct <= 0`. GPU work is recorded and the function waits for completion.
/// - Parameters:
///   - engine: MPS GEMM engine and command queue used to encode GPU work.
///   - kernels: Metal kernels used for max-pool backward and ReLU gradient masking.
///   - batch: Number of examples in the batch.
///   - convAct: Activation buffer from the forward convolution (used to select max indices and ReLU mask).
///   - poolGrad: Upstream gradients from the pooling layer.
///   - convGrad: Destination buffer for the input gradients; it is cleared then written in-place.
func maxPoolBackwardReluGpu(
    engine: MpsGemmEngine,
    kernels: MpsKernels,
    batch: Int,
    convAct: MpsBuffer,
    poolGrad: MpsBuffer,
    convGrad: MpsBuffer
) throws {
    let commandBuffer = try engine.makeCommandBuffer(operation: "max-pool backward")

    // Zero out the gradient buffer first (atomics accumulate)
    memset(convGrad.pointer, 0, convGrad.count * MemoryLayout<Float>.size)

    // Perform max pool backward using Metal kernel
    try kernels.encodeMaxPoolBackward(
        commandBuffer: commandBuffer,
        input: convAct,
        outputGrad: poolGrad,
        inputGrad: convGrad,
        batch: batch,
        channels: convOut,
        inHeight: imgH,
        inWidth: imgW,
        outHeight: poolH,
        outWidth: poolW,
        poolSize: pool,
        stride: pool
    )

    // Apply ReLU gradient: zero out gradients where activation was <= 0
    try kernels.encodeReluGrad(
        commandBuffer: commandBuffer,
        activations: convAct,
        grads: convGrad,
        count: batch * convOut * imgH * imgW
    )

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    try checkMetalCommandBuffer(commandBuffer, operation: "max-pool backward")
}

/// Computes convolution weight and bias gradients on the GPU using im2col, GEMM, and row-sum kernels.
///
/// Mathematical formulation:
/// - colData = im2col(input) → [kernelSize² × inChannels, spatial × batch]
/// - convGrad reshaped → [convOut, spatial × batch]
/// - gradW = convGrad × colData^T → [convOut, kernelSize² × inChannels]
/// - gradB = sum(convGrad) over spatial dimensions → [convOut]
///
/// - Parameters:
///   - engine: GPU GEMM engine and command queue used to run MPS GEMM.
///   - kernels: Metal kernels used for im2col, reshape, row-sum, and related operations.
///   - batch: Number of examples in the batch.
///   - input: Input activations with shape [batch, 1, imgH, imgW].
///   - convGrad: Upstream convolution gradients with shape [batch, convOut, imgH * imgW].
///   - gradW: Destination buffer for weight gradients with shape [convOut, kernel*kernel*1].
///   - gradB: Destination buffer for bias gradients with shape [convOut].
///   - colBuffer: Temporary buffer for the im2col result with shape [kernel*kernel*1, batch*imgH*imgW].
///   - gemmTemp: Temporary buffer for the reshaped conv gradients with shape [convOut, batch*imgH*imgW].
func convBackwardGpu(
    engine: MpsGemmEngine,
    kernels: MpsKernels,
    batch: Int,
    input: MpsBuffer,
    convGrad: MpsBuffer,
    gradW: MpsBuffer,
    gradB: MpsBuffer,
    colBuffer: MpsBuffer,
    gemmTemp: MpsBuffer
) throws {
    let commandBuffer = try engine.makeCommandBuffer(operation: "conv backward")

    let spatial = imgH * imgW
    let colChannels = kernel * kernel * 1  // 9
    let colWidth = spatial * batch         // 784 * batch

    // Step 1: Reshape convGrad from [batch, convOut, spatial] to [convOut, batch*spatial]
    // This matches the CPU implementation's reshape step
    try kernels.encodeReshapeBcsToCbs(
        commandBuffer: commandBuffer,
        input: convGrad,
        output: gemmTemp,
        batch: batch,
        channels: convOut,
        spatial: spatial
    )

    // Step 2: Transform input using im2col on GPU
    // input: [batch, 1, imgH, imgW] -> colBuffer: [colChannels, colWidth]
    try kernels.encodeIm2col(
        commandBuffer: commandBuffer,
        input: input,
        output: colBuffer,
        batch: batch,
        inChannels: 1,
        inHeight: imgH,
        inWidth: imgW,
        outHeight: imgH,
        outWidth: imgW,
        kernelSize: kernel,
        stride: 1,
        padding: pad
    )

    // Step 3: Compute weight gradients using MPS GEMM
    // gradW = reshapedConvGrad × colBuffer^T
    // reshapedConvGrad: [convOut, colWidth] where colWidth = spatial * batch
    // colBuffer^T: [colWidth, colChannels] (transpose of colBuffer: [colChannels, colWidth])
    // gradW: [convOut, colChannels]
    engine.encodeGemm(
        commandBuffer: commandBuffer,
        m: convOut,
        n: colChannels,
        k: colWidth,
        a: gemmTemp,
        b: colBuffer,
        c: gradW,
        transposeA: false,
        transposeB: true,
        alpha: 1.0,
        beta: 0.0
    )

    // Step 4: Compute bias gradients by summing each convOut row of [convOut, batch*spatial].
    try kernels.encodeSumCbsRows(
        commandBuffer: commandBuffer,
        data: gemmTemp,
        output: gradB,
        channels: convOut,
        valuesPerChannel: colWidth,
        scale: 1.0
    )

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    try checkMetalCommandBuffer(commandBuffer, operation: "conv backward")
}
#endif
