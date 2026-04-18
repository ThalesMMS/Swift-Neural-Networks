import Foundation
import Accelerate
import MNISTCommon

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

// =============================================================================
// MARK: - im2col Transformation for vDSP Acceleration
// =============================================================================
//
// OPTIMIZATION: im2col (Image-to-Column) Approach
// ================================================
//
// This implementation uses the im2col algorithm to accelerate convolution operations.
// im2col transforms the convolution operation into a matrix multiplication (GEMM),
// which allows us to leverage highly optimized BLAS routines from Apple's Accelerate
// framework (vDSP).
//
// Traditional Approach (7 nested loops):
//   - Loop over: batch, output channels, input channels, output height, output width,
//     kernel height, kernel width
//   - Results in poor cache utilization and no vectorization
//   - Time complexity: O(batch * C_out * C_in * H * W * K * K)
//
// im2col Approach (matrix multiplication):
//   - Transforms input image into a column matrix where each column represents a
//     receptive field (kernel window)
//   - Reshapes kernel weights into a matrix
//   - Performs a single GEMM: output = weights × im2col(input)
//   - Enables SIMD vectorization via vDSP_mmul
//
// Performance Benefits:
//   - Cache-friendly memory access patterns
//   - SIMD vectorization (processes multiple elements per instruction)
//   - Leverages highly optimized BLAS routines (vDSP_mmul)
//   - Reduces 7 nested loops to 1 matrix multiply + reshape operations
//   - Typical speedup: 3-10x faster than naive nested loops
//
// Trade-offs:
//   - Memory overhead: im2col creates a temporary expanded matrix
//     Size: kernel² * channels * output_spatial * batch
//   - For small images/batches, memory copy overhead may dominate
//   - For production CNNs with large feature maps, the speedup is substantial
//
// Implementation:
//   - im2colForward: Converts image patches to column matrix (forward pass)
//   - col2im: Inverse transformation for gradient accumulation (backward pass)
//   - convForwardRelu: Uses im2col + vDSP_mmul for accelerated convolution
//   - convBackward: Can be further optimized with im2col (currently uses loops)
//

/// Rearranges an NCHW input tensor into column format for convolution.
///
/// Each output column contains one flattened `kernelSize x kernelSize` receptive field for a batch item and output spatial position. Positions outside the padded input image are filled with zero.
/// - Parameters:
///   - input: Flattened input tensor in NCHW layout (batch * inChannels * height * width).
///   - batch: Number of images in the batch.
///   - inChannels: Number of input channels.
///   - height: Input image height (spatial dimension).
///   - width: Input image width (spatial dimension).
///   - kernelSize: Width and height of the square convolution kernel.
///   - pad: Number of zero-padding pixels added to each image border.
/// - Returns: A flattened matrix with shape `(kernelSize * kernelSize * inChannels) x (outHeight * outWidth * batch)`, where `outHeight = height + 2 * pad - kernelSize + 1` and `outWidth = width + 2 * pad - kernelSize + 1`.
func im2colForward(
    input: [Float],
    batch: Int,
    inChannels: Int,
    height: Int,
    width: Int,
    kernelSize: Int,
    pad: Int
) -> [Float] {
    let outHeight = height + 2 * pad - kernelSize + 1
    let outWidth = width + 2 * pad - kernelSize + 1
    let outSpatial = outHeight * outWidth
    let kernelSpatial = kernelSize * kernelSize
    let colChannels = kernelSpatial * inChannels
    let colWidth = outSpatial * batch

    var colData = [Float](repeating: 0.0, count: colChannels * colWidth)

    for b in 0..<batch {
        let batchOffset = b * outSpatial

        for c in 0..<inChannels {
            let channelOffset = c * kernelSpatial
            let inputChannelBase = b * (inChannels * height * width) + c * (height * width)

            for ky in 0..<kernelSize {
                for kx in 0..<kernelSize {
                    let kernelIdx = ky * kernelSize + kx
                    let colRow = channelOffset + kernelIdx

                    for oy in 0..<outHeight {
                        for ox in 0..<outWidth {
                            let iy = oy + ky - pad
                            let ix = ox + kx - pad

                            var value: Float = 0.0
                            if iy >= 0 && iy < height && ix >= 0 && ix < width {
                                let inputIdx = inputChannelBase + iy * width + ix
                                value = input[inputIdx]
                            }

                            let colIdx = colRow * colWidth + batchOffset + oy * outWidth + ox
                            colData[colIdx] = value
                        }
                    }
                }
            }
        }
    }

    return colData
}

/// Accumulates an im2col matrix back into flattened NCHW image layout.
///
/// `colData` is interpreted as `(kernelSize * kernelSize * inChannels) x (outHeight * outWidth * batch)`, matching `im2colForward`. Overlapping receptive-field contributions are summed into the returned image buffer.
/// - Parameters:
///   - colData: Column-formatted patch data produced by `im2colForward`.
///   - batch: Number of images in the batch.
///   - inChannels: Number of input channels.
///   - height: Input image height.
///   - width: Input image width.
///   - kernelSize: Width and height of the square convolution kernel.
///   - pad: Number of zero-padding pixels used for the original im2col transform.
/// - Returns: A flattened image buffer of length `batch * inChannels * height * width` with values accumulated from `colData`.
func col2im(
    colData: [Float],
    batch: Int,
    inChannels: Int,
    height: Int,
    width: Int,
    kernelSize: Int,
    pad: Int
) -> [Float] {
    let outHeight = height + 2 * pad - kernelSize + 1
    let outWidth = width + 2 * pad - kernelSize + 1
    let outSpatial = outHeight * outWidth
    let kernelSpatial = kernelSize * kernelSize
    let colWidth = outSpatial * batch

    var imageData = [Float](repeating: 0.0, count: batch * inChannels * height * width)

    for b in 0..<batch {
        let batchOffset = b * outSpatial

        for c in 0..<inChannels {
            let channelOffset = c * kernelSpatial
            let imageChannelBase = b * (inChannels * height * width) + c * (height * width)

            for ky in 0..<kernelSize {
                for kx in 0..<kernelSize {
                    let kernelIdx = ky * kernelSize + kx
                    let colRow = channelOffset + kernelIdx

                    for oy in 0..<outHeight {
                        for ox in 0..<outWidth {
                            let iy = oy + ky - pad
                            let ix = ox + kx - pad

                            if iy >= 0 && iy < height && ix >= 0 && ix < width {
                                let colIdx = colRow * colWidth + batchOffset + oy * outWidth + ox
                                let imageIdx = imageChannelBase + iy * width + ix
                                // Accumulate gradients from overlapping patches
                                imageData[imageIdx] += colData[colIdx]
                            }
                        }
                    }
                }
            }
        }
    }

    return imageData
}

/// Performs convolution using im2col plus vDSP GEMM, then adds bias and applies ReLU.
/// - Parameters:
///   - model: The CNN model containing convolution weights `convW` and biases `convB`.
///   - batch: Number of images in `input`.
///   - input: Flattened input tensor with layout [batch, inChannels=1, imgH, imgW].
///   - convOutAct: Output buffer to receive activations with layout [batch, convOut, imgH, imgW]; will be overwritten.
func convForwardRelu(model: Cnn, batch: Int, input: [Float], convOutAct: inout [Float]) {
    // Transform input using im2col: [batch * 1 * 28 * 28] -> [9, 784 * batch]
    let colData = im2colForward(
        input: input,
        batch: batch,
        inChannels: 1,
        height: imgH,
        width: imgW,
        kernelSize: kernel,
        pad: pad
    )

    let colChannels = kernel * kernel * 1 // 9
    let colWidth = imgH * imgW * batch    // 784 * batch

    // Weights are [convOut, colChannels] = [8, 9]
    // colData is [colChannels, colWidth] = [9, 784*batch]
    // Result is [convOut, colWidth] = [8, 784*batch]

    var result = [Float](repeating: 0.0, count: convOut * colWidth)

    // Perform matrix multiplication: result = weights × colData
    // vDSP_mmul(A, strideA, B, strideB, C, strideC, M, N, K)
    // Computes C = A × B where A is [M, K], B is [K, N], C is [M, N]
    model.convW.withUnsafeBufferPointer { weightsPtr in
        colData.withUnsafeBufferPointer { colPtr in
            result.withUnsafeMutableBufferPointer { resultPtr in
                guard let wPtr = weightsPtr.baseAddress,
                      let cPtr = colPtr.baseAddress,
                      let rPtr = resultPtr.baseAddress else { return }

                vDSP_mmul(
                    wPtr,           // A: weights [convOut, colChannels]
                    1,              // stride for A
                    cPtr,           // B: colData [colChannels, colWidth]
                    1,              // stride for B
                    rPtr,           // C: result [convOut, colWidth]
                    1,              // stride for C
                    vDSP_Length(convOut),      // M: rows of A
                    vDSP_Length(colWidth),     // N: cols of B
                    vDSP_Length(colChannels)   // K: cols of A / rows of B
                )
            }
        }
    }

    // Add bias and apply ReLU activation
    // Result is [convOut, colWidth], need to reshape to [batch, convOut, imgH, imgW]
    let spatial = imgH * imgW

    for b in 0..<batch {
        let batchOffset = b * spatial
        let outputBatchBase = b * (convOut * spatial)

        for c in 0..<convOut {
            let bias = model.convB[c]
            let outputChannelBase = outputBatchBase + c * spatial

            for s in 0..<spatial {
                // result is stored as [convOut, colWidth] where colWidth = spatial * batch
                let resultIdx = c * colWidth + batchOffset + s
                let outputIdx = outputChannelBase + s

                // Add bias and apply ReLU
                let value = result[resultIdx] + bias
                convOutAct[outputIdx] = (value > 0) ? value : 0
            }
        }
    }
}

/// Performs max pooling on convolutional activations and records the argmax position for each pooled cell.
/// 
/// For each image in the batch and each convolution output channel, this computes the maximum value over each non-overlapping pooling window and writes that max into `poolOut`. Simultaneously records the index of the winning element within the pooling window (row-major offset from 0 to pool*pool-1) into `poolIdx`.
/// - Parameters:
///   - batch: Number of images in the batch.
///   - convAct: Input convolution activations laid out as [batch * convOut * imgH * imgW]. Values are read-only.
///   - poolOut: Output pooled activations laid out as [batch * convOut * (imgH/pool) * (imgW/pool)]. Written in-place.
///   - poolIdx: Output argmax indices for each pooled cell (UInt8), laid out the same as `poolOut`; each entry is the offset within the pooling window (row-major). Written in-place.
func maxPoolForward(batch: Int, convAct: [Float], poolOut: inout [Float], poolIdx: inout [UInt8]) {
    // convAct: [batch*convOut*28*28]
    // poolOut: [batch*convOut*14*14] == [batch*fcIn]
    let convSpatial = imgH * imgW
    let poolSpatial = poolH * poolW

    for b in 0..<batch {
        let convBaseB = b * (convOut * convSpatial)
        let poolBaseB = b * (convOut * poolSpatial)

        for c in 0..<convOut {
            let convBase = convBaseB + c * convSpatial
            let poolBase = poolBaseB + c * poolSpatial

            for py in 0..<poolH {
                for px in 0..<poolW {
                    let iy0 = py * pool
                    let ix0 = px * pool

                    var best = -Float.greatestFiniteMagnitude
                    var bestIdx: UInt8 = 0

                    for dy in 0..<pool {
                        for dx in 0..<pool {
                            let iy = iy0 + dy
                            let ix = ix0 + dx
                            let v = convAct[convBase + iy * imgW + ix]
                            let idx = UInt8(dy * pool + dx)
                            if v > best {
                                best = v
                                bestIdx = idx
                            }
                        }
                    }

                    let outI = poolBase + py * poolW + px
                    poolOut[outI] = best
                    poolIdx[outI] = bestIdx
                }
            }
        }
    }
}

// FC forward: logits = X*W + b.
/// Computes the dense (fully-connected) layer outputs for a batch and writes them into `logits`.
/// - Parameters:
///   - model: The CNN model containing `fcW` (weights) and `fcB` (biases).
///   - batch: Number of examples in `x`.
///   - x: Flattened input activations with layout `[batch, fcIn]` (length `batch * fcIn`).
///   - logits: Output buffer written with layout `[batch, numClasses]` (must have length `batch * numClasses`).
func fcForward(model: Cnn, batch: Int, x: [Float], logits: inout [Float]) {
    for b in 0..<batch {
        let xBase = b * fcIn
        let oBase = b * numClasses
        for j in 0..<numClasses {
            var sum = model.fcB[j]
            for i in 0..<fcIn {
                sum += x[xBase + i] * model.fcW[i * numClasses + j]
            }
            logits[oBase + j] = sum
        }
    }
}

/// Computes softmax probabilities from per-class logits, accumulates cross-entropy loss over the batch, and writes the gradient with respect to the logits.
/// - Parameters:
///   - probsInPlace: On entry, per-sample per-class logits laid out as contiguous rows of length `numClasses`. On exit, those rows are replaced with the corresponding softmax probabilities.
///   - labels: Ground-truth class indices (0-based) for each sample in the batch.
///   - batch: Number of samples in the batch (number of rows in `probsInPlace` / `delta`).
///   - delta: Output buffer (same shape/layout as `probsInPlace`) that will be filled with the gradient of the loss w.r.t. the logits, scaled by `scale`.
///   - scale: Scalar multiplier applied to each gradient value written into `delta`.
/// - Returns: The total cross-entropy loss summed over the batch. The implementation clamps probabilities with a small epsilon (1e-9) before taking the log to avoid numerical issues.
func softmaxXentBackward(probsInPlace: inout [Float], labels: [UInt8], batch: Int, delta: inout [Float], scale: Float) -> Float {
    // probsInPlace holds logits and is overwritten with probs.
    var loss: Float = 0
    let eps: Float = 1e-9

    for b in 0..<batch {
        let base = b * numClasses
        var row = [Float](repeating: 0, count: numClasses)
        for j in 0..<numClasses { row[j] = probsInPlace[base + j] }
        softmaxRowInPlace(&row)
        for j in 0..<numClasses { probsInPlace[base + j] = row[j] }

        let y = Int(labels[b])
        let p = max(row[y], eps)
        loss += -logf(p)

        for j in 0..<numClasses {
            var d = row[j]
            if j == y { d -= 1 }
            delta[base + j] = d * scale
        }
    }

    return loss
}

/// Computes and accumulates gradients for a fully-connected layer over a batch:
/// updates weight gradients (`gradW`), bias gradients (`gradB`), and writes the input gradients into `dX`.
/// - Parameters:
///   - model: The network model containing the current fully-connected weights (`fcW`).
///   - batch: Number of examples in the batch.
///   - x: Input activations for the batch, laid out as consecutive examples (length `batch * fcIn`).
///   - delta: Gradients with respect to the logits for each example (length `batch * numClasses`).
///   - gradW: Output weight gradients (in-out). On entry may be arbitrary; the function zeroes and accumulates into this array. Expected layout: `fcIn * numClasses` where each input feature `i` maps to a contiguous block of `numClasses`.
///   - gradB: Output bias gradients (in-out). On entry may be arbitrary; the function zeroes and accumulates per-class bias gradients (length `numClasses`).
///   - dX: Output gradients with respect to the inputs (in-out). Written as `batch * fcIn`.
func fcBackward(model: Cnn, batch: Int, x: [Float], delta: [Float], gradW: inout [Float], gradB: inout [Float], dX: inout [Float]) {
    // Zero gradients (accumulated over batch).
    for i in 0..<gradW.count { gradW[i] = 0 }
    for i in 0..<gradB.count { gradB[i] = 0 }

    // gradW and gradB.
    for b in 0..<batch {
        let xBase = b * fcIn
        let dBase = b * numClasses

        for j in 0..<numClasses { gradB[j] += delta[dBase + j] }

        for i in 0..<fcIn {
            let xi = x[xBase + i]
            let wRow = i * numClasses
            for j in 0..<numClasses {
                gradW[wRow + j] += xi * delta[dBase + j]
            }
        }
    }

    // dX = delta * W^T.
    for b in 0..<batch {
        let dBase = b * numClasses
        let outBase = b * fcIn
        for i in 0..<fcIn {
            let wRow = i * numClasses
            var sum: Float = 0
            for j in 0..<numClasses {
                sum += delta[dBase + j] * model.fcW[wRow + j]
            }
            dX[outBase + i] = sum
        }
    }
}

/// Scatters max-pool gradients back into the convolution activation layout and applies the ReLU backward mask.
/// 
/// The function routes each pooled output gradient to the activation position recorded in `poolIdx` (index within the pooling window), accumulates into `convGrad`, and then zeros any gradient whose corresponding `convAct` value is <= 0.
/// - Parameters:
///   - batch: Number of examples in the batch.
///   - convAct: Flattened convolution activations with layout [batch, convOut, imgH, imgW]; used to apply the ReLU mask.
///   - poolGrad: Flattened pooled gradients with layout [batch, convOut, poolH, poolW].
///   - poolIdx: Flattened argmax indices (UInt8) for each pooled cell, where each value is the index inside the pooling window (0..pool*pool-1).
///   - convGrad: In-out flattened buffer with layout [batch, convOut, imgH, imgW]; on entry it may contain arbitrary data and on exit contains accumulated gradients after scattering and ReLU masking.
func maxPoolBackwardRelu(batch: Int, convAct: [Float], poolGrad: [Float], poolIdx: [UInt8], convGrad: inout [Float]) {
    let convSpatial = imgH * imgW
    let poolSpatial = poolH * poolW
    let used = batch * convOut * convSpatial

    for i in 0..<used { convGrad[i] = 0 }

    for b in 0..<batch {
        let convBaseB = b * (convOut * convSpatial)
        let poolBaseB = b * (convOut * poolSpatial)

        for c in 0..<convOut {
            let convBase = convBaseB + c * convSpatial
            let poolBase = poolBaseB + c * poolSpatial

            for py in 0..<poolH {
                for px in 0..<poolW {
                    let pI = poolBase + py * poolW + px
                    let g = poolGrad[pI]
                    let a = Int(poolIdx[pI]) // 0..3
                    let dy = a / pool
                    let dx = a % pool

                    let iy = py * pool + dy
                    let ix = px * pool + dx
                    let cI = convBase + iy * imgW + ix
                    convGrad[cI] += g
                }
            }
        }
    }

    // ReLU backward: zero gradients where activation was <= 0.
    for i in 0..<used {
        if convAct[i] <= 0 { convGrad[i] = 0 }
    }
}

/// Computes convolution weight and bias gradients using im2col plus vDSP GEMM.
///
/// Mathematical formulation:
/// - `colData = im2col(input)`, shape `[kernelSize * kernelSize * inChannels, spatial * batch]`
/// - `convGrad` is reshaped to `[convOut, spatial * batch]`
/// - `gradW = convGrad * colData^T`, shape `[convOut, kernelSize * kernelSize * inChannels]`
/// - `gradB` is the sum of `convGrad` over batch and spatial dimensions.
///
/// - Parameters:
///   - model: The CNN model; retained for API consistency with other operations.
///   - batch: Number of examples in the batch.
///   - input: Input activations in `[batch, inChannels=1, imgH, imgW]` layout.
///   - convGrad: Upstream convolution gradients in `[batch, convOut, imgH, imgW]` layout.
///   - gradW: Output buffer for weight gradients; zeroed then overwritten. Must have length `convOut * kernel * kernel`.
///   - gradB: Output buffer for bias gradients; zeroed then overwritten. Must have length `convOut`.
func convBackward(model: Cnn, batch: Int, input: [Float], convGrad: [Float], gradW: inout [Float], gradB: inout [Float]) {
    // Zero gradients
    for i in 0..<gradW.count { gradW[i] = 0 }
    for i in 0..<gradB.count { gradB[i] = 0 }

    let spatial = imgH * imgW
    let colChannels = kernel * kernel * 1  // 9
    let colWidth = spatial * batch         // 784 * batch

    // Step 1: Transform input using im2col: [batch * 1 * 28 * 28] -> [9, 784 * batch]
    let colData = im2colForward(
        input: input,
        batch: batch,
        inChannels: 1,
        height: imgH,
        width: imgW,
        kernelSize: kernel,
        pad: pad
    )

    // Step 2: Reshape convGrad from [batch * convOut * spatial] to [convOut, spatial * batch]
    // convGrad is stored as [batch][convOut][spatial], we need [convOut][batch * spatial]
    var convGradReshaped = [Float](repeating: 0.0, count: convOut * colWidth)
    for b in 0..<batch {
        let batchOffset = b * spatial
        let convGradBatchBase = b * (convOut * spatial)

        for oc in 0..<convOut {
            let convGradChannelBase = convGradBatchBase + oc * spatial
            let reshapedRowBase = oc * colWidth

            for s in 0..<spatial {
                let srcIdx = convGradChannelBase + s
                let dstIdx = reshapedRowBase + batchOffset + s
                convGradReshaped[dstIdx] = convGrad[srcIdx]
            }
        }
    }

    // Step 3: Transpose colData from [colChannels, colWidth] to [colWidth, colChannels]
    // This is needed for the matrix multiplication: gradW = convGradReshaped × colData^T
    var colDataTransposed = [Float](repeating: 0.0, count: colWidth * colChannels)
    colData.withUnsafeBufferPointer { colPtr in
        colDataTransposed.withUnsafeMutableBufferPointer { transPtr in
            guard let cPtr = colPtr.baseAddress,
                  let tPtr = transPtr.baseAddress else { return }
            vDSP_mtrans(
                cPtr,                          // Input matrix
                1,                             // Input stride
                tPtr,                          // Output matrix
                1,                             // Output stride
                vDSP_Length(colChannels),      // Rows of input (becomes cols of output)
                vDSP_Length(colWidth)          // Cols of input (becomes rows of output)
            )
        }
    }

    // Step 4: Compute weight gradients using vDSP_mmul
    // gradW = convGradReshaped × colDataTransposed
    // [convOut, colWidth] × [colWidth, colChannels] → [convOut, colChannels]
    convGradReshaped.withUnsafeBufferPointer { convGradPtr in
        colDataTransposed.withUnsafeBufferPointer { colTransPtr in
            gradW.withUnsafeMutableBufferPointer { gradWPtr in
                guard let cgPtr = convGradPtr.baseAddress,
                      let ctPtr = colTransPtr.baseAddress,
                      let gwPtr = gradWPtr.baseAddress else { return }

                vDSP_mmul(
                    cgPtr,                         // A: convGradReshaped [convOut, colWidth]
                    1,                             // stride for A
                    ctPtr,                         // B: colDataTransposed [colWidth, colChannels]
                    1,                             // stride for B
                    gwPtr,                         // C: gradW [convOut, colChannels]
                    1,                             // stride for C
                    vDSP_Length(convOut),          // M: rows of A
                    vDSP_Length(colChannels),      // N: cols of B
                    vDSP_Length(colWidth)          // K: cols of A / rows of B
                )
            }
        }
    }

    // Step 5: Compute bias gradients by summing convGrad over spatial dimensions
    // gradB[oc] = sum over all spatial locations and batch
    for b in 0..<batch {
        let convGradBatchBase = b * (convOut * spatial)

        for oc in 0..<convOut {
            let convGradChannelBase = convGradBatchBase + oc * spatial

            for s in 0..<spatial {
                gradB[oc] += convGrad[convGradChannelBase + s]
            }
        }
    }
}
