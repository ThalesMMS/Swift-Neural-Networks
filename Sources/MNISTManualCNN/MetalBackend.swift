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
// MARK: - Metal Backend Infrastructure
// =============================================================================

#if canImport(MetalPerformanceShaders)
struct CommandBufferCreationError: Error, CustomStringConvertible {
    let operation: String

    var description: String {
        "Metal command buffer creation failed during \(operation): makeCommandBuffer() returned nil"
    }
}

struct MetalAllocationError: Error, CustomStringConvertible {
    let label: String
    let count: Int
    let byteCount: Int

    var description: String {
        let sizeMB = Double(byteCount) / (1024 * 1024)
        return "Metal buffer allocation failed for \(label): \(count) elements (\(String(format: "%.2f", sizeMB)) MB). Try reducing --batch or closing other GPU-intensive applications."
    }
}

struct MetalEncoderCreationError: Error, CustomStringConvertible {
    let pipeline: String

    var description: String {
        "Metal compute encoder creation failed for \(pipeline): makeComputeCommandEncoder() returned nil"
    }
}

struct MetalCommandBufferExecutionError: Error, CustomStringConvertible {
    let operation: String
    let underlying: Error?

    var description: String {
        if let underlying {
            return "Metal command buffer failed during \(operation): \(underlying)"
        }
        return "Metal command buffer failed during \(operation) with no underlying error"
    }
}

func checkMetalCommandBuffer(_ commandBuffer: MTLCommandBuffer, operation: String) throws {
    if commandBuffer.status == .error {
        throw MetalCommandBufferExecutionError(operation: operation, underlying: commandBuffer.error)
    }
}

/// Metal/MPS backend for GPU-accelerated CNN operations
final class MetalCnnBackend {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              MPSSupportsMTLDevice(device),
              let queue = device.makeCommandQueue() else {
            return nil
        }
        self.device = device
        self.commandQueue = queue
    }
}

// CPU/GPU shared buffer using storageModeShared.
final class MpsBuffer {
    let buffer: MTLBuffer
    let count: Int
    let pointer: UnsafeMutablePointer<Float>

    init(device: MTLDevice, count: Int, label: String, initial: [Float]? = nil) throws {
        let length = count * MemoryLayout<Float>.size
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw MetalAllocationError(label: label, count: count, byteCount: length)
        }
        buffer.label = label
        self.buffer = buffer
        self.count = count
        self.pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        if let initial = initial {
            update(from: initial, count: min(initial.count, count))
        } else {
            memset(pointer, 0, length)
        }
    }

    /// Copies elements from a Swift `Float` array into the buffer's memory.
    /// - Parameters:
    ///   - array: Source `Float` values to copy from.
    ///   - count: Optional number of elements to copy. If omitted, copies up to `min(array.count, self.count)`. If provided, the actual copy length is `min(count, array.count, self.count)`.
    func update(from array: [Float], count: Int? = nil) {
        let n = count ?? min(array.count, self.count)
        array.withUnsafeBufferPointer { buf in
            guard let src = buf.baseAddress else { return }
            pointer.update(from: src, count: n)
        }
    }

    /// Copies elements from the buffer into the provided array.
    /// - Parameters:
    ///   - array: An inout array to receive data; the function writes up to the lesser of `array.count` and the buffer's element count.
    func copy(to array: inout [Float]) {
        let n = min(array.count, count)
        array.withUnsafeMutableBufferPointer { buf in
            guard let dst = buf.baseAddress else { return }
            dst.update(from: pointer, count: n)
        }
    }
}

// Shared buffer for labels (UInt8).
final class MpsBufferU8 {
    let buffer: MTLBuffer
    let count: Int
    let pointer: UnsafeMutablePointer<UInt8>

    init(device: MTLDevice, count: Int, label: String) throws {
        let length = count * MemoryLayout<UInt8>.size
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw MetalAllocationError(label: label, count: count, byteCount: length)
        }
        buffer.label = label
        self.buffer = buffer
        self.count = count
        self.pointer = buffer.contents().bindMemory(to: UInt8.self, capacity: count)
        memset(pointer, 0, length)
    }
}

// GPU backend using MPSMatrixMultiplication with persistent buffers.
final class MpsGemmEngine {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              MPSSupportsMTLDevice(device),
              let queue = device.makeCommandQueue() else {
            return nil
        }
        self.device = device
        self.commandQueue = queue
    }

    /// Creates an `MpsBuffer` backed by this engine's Metal device.
    /// - Parameters:
    ///   - count: Number of `Float` elements the buffer will hold.
    ///   - label: Human-readable label assigned to the underlying `MTLBuffer`.
    ///   - initial: Optional initial contents to copy into the buffer; if `nil` the buffer is zero-initialized.
    /// - Returns: An `MpsBuffer` that wraps a Metal buffer sized for `count` `Float` elements.
    func makeBuffer(count: Int, label: String, initial: [Float]? = nil) throws -> MpsBuffer {
        try MpsBuffer(device: device, count: count, label: label, initial: initial)
    }

    /// Encodes a matrix multiplication operation C = alpha * A * B + beta * C into the given Metal command buffer.
    /// - Parameters:
    ///   - commandBuffer: The `MTLCommandBuffer` to record the MPS matrix-multiplication into.
    ///   - m: Number of rows in the result matrix C (and in A when not transposed).
    ///   - n: Number of columns in the result matrix C (and in B when not transposed).
    ///   - k: The shared inner dimension for the multiplication (columns of A / rows of B when not transposed).
    ///   - a: Source buffer holding matrix A in row-major layout.
    ///   - b: Source buffer holding matrix B in row-major layout.
    ///   - c: Destination buffer holding matrix C in row-major layout; also used as the input C for the fused accumulation.
    ///   - transposeA: If `true`, treat A as transposed (swap A's rows and columns for the operation).
    ///   - transposeB: If `true`, treat B as transposed (swap B's rows and columns for the operation).
    ///   - alpha: Scalar multiplier applied to the product A * B.
    ///   - beta: Scalar multiplier applied to the existing contents of C before accumulation.
    func encodeGemm(
        commandBuffer: MTLCommandBuffer,
        m: Int,
        n: Int,
        k: Int,
        a: MpsBuffer,
        b: MpsBuffer,
        c: MpsBuffer,
        transposeA: Bool,
        transposeB: Bool,
        alpha: Float,
        beta: Float
    ) {
        let stride = MemoryLayout<Float>.size
        let aRows = transposeA ? k : m
        let aCols = transposeA ? m : k
        let bRows = transposeB ? n : k
        let bCols = transposeB ? k : n
        let aDesc = MPSMatrixDescriptor(
            rows: aRows,
            columns: aCols,
            rowBytes: aCols * stride,
            dataType: .float32
        )
        let bDesc = MPSMatrixDescriptor(
            rows: bRows,
            columns: bCols,
            rowBytes: bCols * stride,
            dataType: .float32
        )
        let cDesc = MPSMatrixDescriptor(
            rows: m,
            columns: n,
            rowBytes: n * stride,
            dataType: .float32
        )

        let aMat = MPSMatrix(buffer: a.buffer, descriptor: aDesc)
        let bMat = MPSMatrix(buffer: b.buffer, descriptor: bDesc)
        let cMat = MPSMatrix(buffer: c.buffer, descriptor: cDesc)

        let op = MPSMatrixMultiplication(
            device: device,
            transposeLeft: transposeA,
            transposeRight: transposeB,
            resultRows: m,
            resultColumns: n,
            interiorColumns: k,
            alpha: Double(alpha),
            beta: Double(beta)
        )

        op.encode(commandBuffer: commandBuffer, leftMatrix: aMat, rightMatrix: bMat, resultMatrix: cMat)
    }

    func makeCommandBuffer(operation: String) throws -> MTLCommandBuffer {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            let error = CommandBufferCreationError(operation: operation)
            fputs("\(error.description)\n", stderr)
            throw error
        }
        return commandBuffer
    }

    /// Performs a GPU GEMM operation and writes the result into `c`.
    /// 
    /// Computes C = alpha * A * B + beta * C using an MPS-encoded matrix multiplication and blocks until completion.
    /// - Parameters:
    ///   - m: Number of rows of the resulting matrix C.
    ///   - n: Number of columns of the resulting matrix C.
    ///   - k: The inner dimension shared by A and B (columns of A / rows of B).
    ///   - a: Left-hand input matrix stored in an `MpsBuffer`.
    ///   - b: Right-hand input matrix stored in an `MpsBuffer`.
    ///   - c: Output/input matrix stored in an `MpsBuffer` that receives the result.
    ///   - transposeA: If `true`, treat `a` as transposed (use A^T) when computing the product.
    ///   - transposeB: If `true`, treat `b` as transposed (use B^T) when computing the product.
    ///   - alpha: Scalar multiplier applied to the product A*B.
    ///   - beta: Scalar multiplier applied to the existing contents of C.
    func gemm(
        m: Int,
        n: Int,
        k: Int,
        a: MpsBuffer,
        b: MpsBuffer,
        c: MpsBuffer,
        transposeA: Bool,
        transposeB: Bool,
        alpha: Float,
        beta: Float
    ) throws {
        let commandBuffer = try makeCommandBuffer(operation: "GEMM")
        encodeGemm(
            commandBuffer: commandBuffer,
            m: m,
            n: n,
            k: k,
            a: a,
            b: b,
            c: c,
            transposeA: transposeA,
            transposeB: transposeB,
            alpha: alpha,
            beta: beta
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkMetalCommandBuffer(commandBuffer, operation: "GEMM")
    }
}

// Metal kernels to operate on GPU tensors (ReLU, softmax, reductions, SGD).
final class MpsKernels {
    private let addBiasPSO: MTLComputePipelineState
    private let reluPSO: MTLComputePipelineState
    private let reluGradPSO: MTLComputePipelineState
    private let softmaxPSO: MTLComputePipelineState
    private let sumRowsPSO: MTLComputePipelineState
    private let sumCbsRowsPSO: MTLComputePipelineState
    private let deltaLossPSO: MTLComputePipelineState
    private let sgdPSO: MTLComputePipelineState
    private let maxPoolForwardPSO: MTLComputePipelineState
    private let maxPoolBackwardPSO: MTLComputePipelineState
    private let im2colPSO: MTLComputePipelineState
    private let col2imPSO: MTLComputePipelineState
    private let convAddBiasReluPSO: MTLComputePipelineState
    private let convTransposeBiasReluPSO: MTLComputePipelineState
    private let reshapeBcsToCbsPSO: MTLComputePipelineState

    init?(device: MTLDevice) {
        let library: MTLLibrary
        if let shaderURL = Bundle.module.url(forResource: "MpsKernels", withExtension: "metal", subdirectory: "Shaders") {
            do {
                let source = try String(contentsOf: shaderURL, encoding: .utf8)
                library = try device.makeLibrary(source: source, options: nil)
            } catch {
                print("❌ Metal Library Compilation Failed")
                print("   Source: \(shaderURL.path)")
                print("   Error: \(error)")
                print("")
                print("POSSIBLE CAUSES:")
                print("   • Syntax error in Metal shader code")
                print("   • Incompatible Metal version or GPU")
                print("")
                print("SOLUTIONS:")
                print("   1. Verify your macOS version supports Metal 2.0+")
                print("   2. Try rebuilding: swift build --clean && swift build")
                return nil
            }
        } else if let defaultLibrary = device.makeDefaultLibrary() {
            // Fallback for environments that provide a precompiled default Metal library.
            library = defaultLibrary
        } else {
            print("❌ Metal Shader File Not Found")
            print("   Looking for: Shaders/MpsKernels.metal in the MNISTManualCNN module bundle")
            print("")
            print("SOLUTIONS:")
            print("   1. Rebuild the package: swift build")
            print("   2. Verify the file exists:")
            print("      ls -la Sources/MNISTManualCNN/Shaders/MpsKernels.metal")
            return nil
        }

        /// Creates a compute pipeline state for the given Metal kernel function name.
        /// - Parameters:
        ///   - name: The Metal kernel function name to look up in the loaded library.
        /// - Returns: The `MTLComputePipelineState` for the kernel if successful, `nil` if the function is not found or pipeline creation fails (diagnostic messages are printed on failure).
        func makePSO(_ name: String) -> MTLComputePipelineState? {
            guard let function = library.makeFunction(name: name) else {
                print("❌ Metal Kernel Function Not Found")
                print("   Missing kernel: \(name)")
                print("")
                print("POSSIBLE CAUSES:")
                print("   • Kernel function name mismatch in Metal shader")
                print("   • Metal library compilation partially failed")
                print("   • Corrupted Metal shader source")
                print("")
                print("EXPECTED KERNELS:")
                print("   • add_bias, relu_inplace, relu_grad")
                print("   • softmax_rows, sum_rows, sum_cbs_rows")
                print("   • delta_and_loss, sgd_update")
                print("")
                print("SOLUTIONS:")
                print("   1. Verify MpsKernels.metal contains all required kernels")
                print("   2. Rebuild the project: swift build --clean && swift build")
                print("   3. Restore shader file: git checkout Sources/MNISTManualCNN/Shaders/MpsKernels.metal")
                return nil
            }
            do {
                return try device.makeComputePipelineState(function: function)
            } catch {
                print("❌ Failed to Create Metal Pipeline State")
                print("   Kernel: \(name)")
                print("   Error: \(error)")
                print("")
                print("POSSIBLE CAUSES:")
                print("   • Incompatible GPU or Metal version")
                print("   • Kernel configuration error")
                print("")
                print("SOLUTIONS:")
                print("   1. Verify your Mac supports Metal 2.0+")
                print("   2. Update macOS to the latest version")
                print("   3. Try rebuilding: swift build --clean && swift build")
                return nil
            }
        }

        guard let addBiasPSO = makePSO("add_bias"),
              let reluPSO = makePSO("relu_inplace"),
              let reluGradPSO = makePSO("relu_grad"),
              let softmaxPSO = makePSO("softmax_rows"),
              let sumRowsPSO = makePSO("sum_rows"),
              let sumCbsRowsPSO = makePSO("sum_cbs_rows"),
              let deltaLossPSO = makePSO("delta_and_loss"),
              let sgdPSO = makePSO("sgd_update"),
              let maxPoolForwardPSO = makePSO("max_pool_forward"),
              let maxPoolBackwardPSO = makePSO("max_pool_backward"),
              let im2colPSO = makePSO("im2col"),
              let col2imPSO = makePSO("col2im"),
              let convAddBiasReluPSO = makePSO("conv_add_bias_relu"),
              let convTransposeBiasReluPSO = makePSO("conv_transpose_bias_relu"),
              let reshapeBcsToCbsPSO = makePSO("reshape_bcs_to_cbs") else {
            print("⚠️  Metal Kernel Initialization Failed - Training will use CPU")
            print("   Reason: One or more Metal compute kernels could not be created")
            print("   → The detailed error(s) are shown above")
            print("   → Training will proceed normally on CPU (slower but identical results)")
            print("   → GPU acceleration requires all kernels to initialize successfully")
            return nil
        }

        self.addBiasPSO = addBiasPSO
        self.reluPSO = reluPSO
        self.reluGradPSO = reluGradPSO
        self.softmaxPSO = softmaxPSO
        self.sumRowsPSO = sumRowsPSO
        self.sumCbsRowsPSO = sumCbsRowsPSO
        self.deltaLossPSO = deltaLossPSO
        self.sgdPSO = sgdPSO
        self.maxPoolForwardPSO = maxPoolForwardPSO
        self.maxPoolBackwardPSO = maxPoolBackwardPSO
        self.im2colPSO = im2colPSO
        self.col2imPSO = col2imPSO
        self.convAddBiasReluPSO = convAddBiasReluPSO
        self.convTransposeBiasReluPSO = convTransposeBiasReluPSO
        self.reshapeBcsToCbsPSO = reshapeBcsToCbsPSO
    }

    /// Dispatches a 1D compute kernel using the provided pipeline and encodes resource bindings via the `encode` closure.
    /// - Parameters:
    ///   - commandBuffer: The `MTLCommandBuffer` to record the compute encoder into.
    ///   - pipeline: The `MTLComputePipelineState` to use for the dispatch.
    ///   - count: The total number of threads to launch (treated as a 1D range). If `count <= 0` no work is encoded.
    ///   - encode: A closure that receives the `MTLComputeCommandEncoder` and should bind buffers/bytes/state required by the kernel.
    private func dispatch1D(
        _ commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        count: Int,
        encode: (MTLComputeCommandEncoder) -> Void
    ) throws {
        guard count > 0 else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEncoderCreationError(pipeline: String(describing: pipeline))
        }
        encoder.setComputePipelineState(pipeline)
        encode(encoder)
        let width = pipeline.threadExecutionWidth
        let threads = MTLSize(width: count, height: 1, depth: 1)
        let group = MTLSize(width: min(width, count), height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: group)
        encoder.endEncoding()
    }

    /// Encodes and dispatches the "add bias" compute kernel to add a bias vector to each row of a matrix.
    /// - Parameters:
    ///   - commandBuffer: Command buffer used to record and submit the compute work.
    ///   - data: Buffer containing the matrix values to be updated in row-major layout.
    ///   - bias: Buffer containing the bias vector of length `cols` to add to each row.
    ///   - rows: Number of rows in `data`.
    ///   - cols: Number of columns in `data`.
    func encodeAddBias(commandBuffer: MTLCommandBuffer, data: MpsBuffer, bias: MpsBuffer, rows: Int, cols: Int) throws {
        var rowsU = UInt32(rows)
        var colsU = UInt32(cols)
        try dispatch1D(commandBuffer, pipeline: addBiasPSO, count: rows * cols) { encoder in
            encoder.setBuffer(data.buffer, offset: 0, index: 0)
            encoder.setBuffer(bias.buffer, offset: 0, index: 1)
            encoder.setBytes(&rowsU, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&colsU, length: MemoryLayout<UInt32>.size, index: 3)
        }
    }

    /// Encodes an in-place ReLU compute kernel that applies ReLU to up to `count` elements in `data`.
    /// - Parameters:
    ///   - data: The GPU buffer of `Float` values to modify in place.
    ///   - count: The number of elements in `data` to process. If `count` is less than or equal to zero, no commands are encoded.
    func encodeRelu(commandBuffer: MTLCommandBuffer, data: MpsBuffer, count: Int) throws {
        var countU = UInt32(count)
        try dispatch1D(commandBuffer, pipeline: reluPSO, count: count) { encoder in
            encoder.setBuffer(data.buffer, offset: 0, index: 0)
            encoder.setBytes(&countU, length: MemoryLayout<UInt32>.size, index: 1)
        }
    }

    /// Encodes the ReLU gradient compute kernel into the given command buffer for `count` elements.
    /// - Parameters:
    ///   - commandBuffer: The command buffer to receive the compute encoder and dispatch.
    ///   - activations: Buffer containing the forward ReLU activations (input to the gradient kernel).
    ///   - grads: Buffer containing gradients to be multiplied by the ReLU derivative (in-place output).
    ///   - count: The number of elements to process.
    func encodeReluGrad(commandBuffer: MTLCommandBuffer, activations: MpsBuffer, grads: MpsBuffer, count: Int) throws {
        var countU = UInt32(count)
        try dispatch1D(commandBuffer, pipeline: reluGradPSO, count: count) { encoder in
            encoder.setBuffer(activations.buffer, offset: 0, index: 0)
            encoder.setBuffer(grads.buffer, offset: 0, index: 1)
            encoder.setBytes(&countU, length: MemoryLayout<UInt32>.size, index: 2)
        }
    }

    /// Applies softmax independently to each row of the rows-by-cols matrix stored in `data`.
    /// - Parameters:
    ///   - commandBuffer: The Metal command buffer used to record the GPU work.
    ///   - data: A buffer containing `rows * cols` Float elements (matrix layout, rows × cols).
    ///   - rows: The number of rows in the matrix.
    ///   - cols: The number of columns in the matrix.
    func encodeSoftmax(commandBuffer: MTLCommandBuffer, data: MpsBuffer, rows: Int, cols: Int) throws {
        var rowsU = UInt32(rows)
        var colsU = UInt32(cols)
        try dispatch1D(commandBuffer, pipeline: softmaxPSO, count: rows) { encoder in
            encoder.setBuffer(data.buffer, offset: 0, index: 0)
            encoder.setBytes(&rowsU, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.setBytes(&colsU, length: MemoryLayout<UInt32>.size, index: 2)
        }
    }

    /// Computes the column-wise sum over `rows` for each of `cols` columns, multiplies each sum by `scale`, and writes the results into `output`.
    /// - Parameters:
    ///   - commandBuffer: The command buffer used to encode the compute work.
    ///   - data: Source buffer containing a matrix laid out row-major with `rows * cols` elements.
    ///   - output: Destination buffer which will receive `cols` summed (and scaled) values.
    ///   - rows: Number of rows in `data`.
    ///   - cols: Number of columns in `data` (also the number of output elements).
    ///   - scale: Scalar factor applied to each column sum before storing in `output`.
    func encodeSumRows(
        commandBuffer: MTLCommandBuffer,
        data: MpsBuffer,
        output: MpsBuffer,
        rows: Int,
        cols: Int,
        scale: Float
    ) throws {
        var rowsU = UInt32(rows)
        var colsU = UInt32(cols)
        var scaleVar = scale
        try dispatch1D(commandBuffer, pipeline: sumRowsPSO, count: cols) { encoder in
            encoder.setBuffer(data.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            encoder.setBytes(&rowsU, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&colsU, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&scaleVar, length: MemoryLayout<Float>.size, index: 4)
        }
    }

    /// Sums each channel row in a `[channels, valuesPerChannel]` buffer and writes one scaled value per channel.
    /// - Parameters:
    ///   - commandBuffer: The command buffer used to encode the compute work.
    ///   - data: Source buffer laid out row-major as `[channels, valuesPerChannel]`.
    ///   - output: Destination buffer receiving `channels` values.
    ///   - channels: Number of channel rows in `data`.
    ///   - valuesPerChannel: Number of values to sum for each channel.
    ///   - scale: Scalar factor applied to each channel sum.
    func encodeSumCbsRows(
        commandBuffer: MTLCommandBuffer,
        data: MpsBuffer,
        output: MpsBuffer,
        channels: Int,
        valuesPerChannel: Int,
        scale: Float
    ) throws {
        var channelsU = UInt32(channels)
        var valuesPerChannelU = UInt32(valuesPerChannel)
        var scaleVar = scale
        try dispatch1D(commandBuffer, pipeline: sumCbsRowsPSO, count: channels) { encoder in
            encoder.setBuffer(data.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            encoder.setBytes(&channelsU, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&valuesPerChannelU, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&scaleVar, length: MemoryLayout<Float>.size, index: 4)
        }
    }

    /// Computes per-row softmax cross-entropy loss and its gradient, writing gradients into `delta` and per-row loss values into `loss`.
    /// 
    /// For each row (0..<rows), the kernel interprets `outputs` as unnormalized logits of length `cols`, reads the corresponding label from `labels` (stored as a `UInt8` index), computes the softmax cross-entropy loss for that example, stores the scalar loss into `loss`, and writes the gradient with respect to the logits into `delta`.
    /// - Parameters:
    ///   - commandBuffer: The command buffer used to encode the compute work.
    ///   - outputs: Buffer containing logits arranged row-major as `rows * cols` floats.
    ///   - labels: Buffer of `UInt8` class indices, one per row.
    ///   - delta: Output buffer where per-logit gradients will be written (same shape as `outputs`).
    ///   - loss: Output buffer where one float loss value per row will be written (length `rows`).
    ///   - rows: Number of rows (examples / batch size).
    ///   - cols: Number of columns (classes / logits per example).
    func encodeDeltaAndLoss(
        commandBuffer: MTLCommandBuffer,
        outputs: MpsBuffer,
        labels: MpsBufferU8,
        delta: MpsBuffer,
        loss: MpsBuffer,
        rows: Int,
        cols: Int
    ) throws {
        var rowsU = UInt32(rows)
        var colsU = UInt32(cols)
        try dispatch1D(commandBuffer, pipeline: deltaLossPSO, count: rows) { encoder in
            encoder.setBuffer(outputs.buffer, offset: 0, index: 0)
            encoder.setBuffer(labels.buffer, offset: 0, index: 1)
            encoder.setBuffer(delta.buffer, offset: 0, index: 2)
            encoder.setBuffer(loss.buffer, offset: 0, index: 3)
            encoder.setBytes(&rowsU, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&colsU, length: MemoryLayout<UInt32>.size, index: 5)
        }
    }

    /// Encodes an SGD parameter update kernel into the given command buffer.
    /// - Parameters:
    ///   - commandBuffer: The command buffer to encode into.
    ///   - weights: GPU buffer holding weights to be updated in-place.
    ///   - grads: GPU buffer holding gradients for each weight.
    ///   - count: The number of weight elements to update.
    ///   - learningRate: Scalar learning rate used to scale gradients during the update.
    func encodeSgdUpdate(
        commandBuffer: MTLCommandBuffer,
        weights: MpsBuffer,
        grads: MpsBuffer,
        count: Int,
        learningRate: Float
    ) throws {
        var countU = UInt32(count)
        var lr = learningRate
        try dispatch1D(commandBuffer, pipeline: sgdPSO, count: count) { encoder in
            encoder.setBuffer(weights.buffer, offset: 0, index: 0)
            encoder.setBuffer(grads.buffer, offset: 0, index: 1)
            encoder.setBytes(&countU, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&lr, length: MemoryLayout<Float>.size, index: 3)
        }
    }

    /// Encodes a max-pooling forward pass into the provided command buffer, writing pooled results into `output`.
    /// - Parameters:
    ///   - commandBuffer: The Metal command buffer to record the compute dispatch into.
    ///   - input: Source buffer containing input activations in B×C×H×W (batch, channels, height, width) layout.
    ///   - output: Destination buffer that will receive pooled outputs in B×C×outHeight×outWidth layout.
    ///   - batch: Number of examples in the batch (B).
    ///   - channels: Number of channels (C).
    ///   - inHeight: Input height (H).
    ///   - inWidth: Input width (W).
    ///   - outHeight: Output height after pooling.
    ///   - outWidth: Output width after pooling.
    ///   - poolSize: Size of the pooling kernel (kernel width and height).
    ///   - stride: Stride of the pooling kernel.
    func encodeMaxPoolForward(
        commandBuffer: MTLCommandBuffer,
        input: MpsBuffer,
        output: MpsBuffer,
        batch: Int,
        channels: Int,
        inHeight: Int,
        inWidth: Int,
        outHeight: Int,
        outWidth: Int,
        poolSize: Int,
        stride: Int
    ) throws {
        var batchU = UInt32(batch)
        var channelsU = UInt32(channels)
        var inHeightU = UInt32(inHeight)
        var inWidthU = UInt32(inWidth)
        var outHeightU = UInt32(outHeight)
        var outWidthU = UInt32(outWidth)
        var poolSizeU = UInt32(poolSize)
        var strideU = UInt32(stride)
        let totalOut = batch * channels * outHeight * outWidth
        try dispatch1D(commandBuffer, pipeline: maxPoolForwardPSO, count: totalOut) { encoder in
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            encoder.setBytes(&batchU, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&channelsU, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&inHeightU, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&inWidthU, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&outHeightU, length: MemoryLayout<UInt32>.size, index: 6)
            encoder.setBytes(&outWidthU, length: MemoryLayout<UInt32>.size, index: 7)
            encoder.setBytes(&poolSizeU, length: MemoryLayout<UInt32>.size, index: 8)
            encoder.setBytes(&strideU, length: MemoryLayout<UInt32>.size, index: 9)
        }
    }

    /// Computes input gradients for a max-pooling layer by encoding the backward pass into the provided command buffer.
    /// - Parameters:
    ///   - commandBuffer: The Metal command buffer to encode the kernel into.
    ///   - input: Buffer containing the original input activations (used to determine max locations).  
    ///   - outputGrad: Buffer containing gradients with respect to the pooled output.
    ///   - inputGrad: Buffer to receive gradients with respect to the input (written by the kernel).
    ///   - batch: Number of examples in the batch.
    ///   - channels: Number of feature channels per example.
    ///   - inHeight: Height of the input feature map.
    ///   - inWidth: Width of the input feature map.
    ///   - outHeight: Height of the output (pooled) feature map.
    ///   - outWidth: Width of the output (pooled) feature map.
    ///   - poolSize: Size of the pooling window (assumed square).
    ///   - stride: Stride of the pooling window.
    func encodeMaxPoolBackward(
        commandBuffer: MTLCommandBuffer,
        input: MpsBuffer,
        outputGrad: MpsBuffer,
        inputGrad: MpsBuffer,
        batch: Int,
        channels: Int,
        inHeight: Int,
        inWidth: Int,
        outHeight: Int,
        outWidth: Int,
        poolSize: Int,
        stride: Int
    ) throws {
        var batchU = UInt32(batch)
        var channelsU = UInt32(channels)
        var inHeightU = UInt32(inHeight)
        var inWidthU = UInt32(inWidth)
        var outHeightU = UInt32(outHeight)
        var outWidthU = UInt32(outWidth)
        var poolSizeU = UInt32(poolSize)
        var strideU = UInt32(stride)
        let totalOut = batch * channels * outHeight * outWidth
        try dispatch1D(commandBuffer, pipeline: maxPoolBackwardPSO, count: totalOut) { encoder in
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(outputGrad.buffer, offset: 0, index: 1)
            encoder.setBuffer(inputGrad.buffer, offset: 0, index: 2)
            encoder.setBytes(&batchU, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&channelsU, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&inHeightU, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&inWidthU, length: MemoryLayout<UInt32>.size, index: 6)
            encoder.setBytes(&outHeightU, length: MemoryLayout<UInt32>.size, index: 7)
            encoder.setBytes(&outWidthU, length: MemoryLayout<UInt32>.size, index: 8)
            encoder.setBytes(&poolSizeU, length: MemoryLayout<UInt32>.size, index: 9)
            encoder.setBytes(&strideU, length: MemoryLayout<UInt32>.size, index: 10)
        }
    }

    /// Encodes an im2col transformation into the provided command buffer so the GPU writes rearranged image patches into `output`.
    /// The output matrix has `outputRows = inChannels * (kernelSize * kernelSize)` rows and `outputCols = batch * outHeight * outWidth` columns; each column contains the flattened patch for one output location.
    /// - Parameters:
    ///   - commandBuffer: The Metal command buffer to encode the compute work into.
    ///   - input: Source buffer containing input images in BCHW layout (batch, channels, height, width).
    ///   - output: Destination buffer that will receive the im2col matrix.
    ///   - batch: Number of images in the batch.
    ///   - inChannels: Number of input channels.
    ///   - inHeight: Input height (pixels).
    ///   - inWidth: Input width (pixels).
    ///   - outHeight: Output height after convolution (number of vertical patch positions).
    ///   - outWidth: Output width after convolution (number of horizontal patch positions).
    ///   - kernelSize: Spatial size of the square kernel (kernelSize x kernelSize).
    ///   - stride: Stride between patches.
    ///   - padding: Zero-padding applied to the input on each border.
    func encodeIm2col(
        commandBuffer: MTLCommandBuffer,
        input: MpsBuffer,
        output: MpsBuffer,
        batch: Int,
        inChannels: Int,
        inHeight: Int,
        inWidth: Int,
        outHeight: Int,
        outWidth: Int,
        kernelSize: Int,
        stride: Int,
        padding: Int
    ) throws {
        var batchU = UInt32(batch)
        var inChannelsU = UInt32(inChannels)
        var inHeightU = UInt32(inHeight)
        var inWidthU = UInt32(inWidth)
        var outHeightU = UInt32(outHeight)
        var outWidthU = UInt32(outWidth)
        var kernelSizeU = UInt32(kernelSize)
        var strideU = UInt32(stride)
        var paddingU = UInt32(padding)

        let kernelArea = kernelSize * kernelSize
        let outputCols = batch * outHeight * outWidth
        let outputRows = inChannels * kernelArea
        let totalElements = outputRows * outputCols

        try dispatch1D(commandBuffer, pipeline: im2colPSO, count: totalElements) { encoder in
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            encoder.setBytes(&batchU, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&inChannelsU, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&inHeightU, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&inWidthU, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&outHeightU, length: MemoryLayout<UInt32>.size, index: 6)
            encoder.setBytes(&outWidthU, length: MemoryLayout<UInt32>.size, index: 7)
            encoder.setBytes(&kernelSizeU, length: MemoryLayout<UInt32>.size, index: 8)
            encoder.setBytes(&strideU, length: MemoryLayout<UInt32>.size, index: 9)
            encoder.setBytes(&paddingU, length: MemoryLayout<UInt32>.size, index: 10)
        }
    }

    /// Encodes and dispatches the `col2im` compute kernel to reconstruct image tensors from im2col column data.
    /// - Parameters:
    ///   - commandBuffer: The Metal command buffer to record the compute dispatch into.
    ///   - input: Source buffer containing im2col columns laid out with rows = inChannels * kernelSize * kernelSize and columns = batch * outHeight * outWidth.
    ///   - output: Destination buffer that will receive reconstructed images with layout (batch, inChannels, inHeight, inWidth).
    ///   - batch: Number of images in the batch.
    ///   - inChannels: Number of input channels per image.
    ///   - inHeight: Height of each reconstructed input image.
    ///   - inWidth: Width of each reconstructed input image.
    ///   - outHeight: Output height produced by the corresponding convolution (number of column rows vertically).
    ///   - outWidth: Output width produced by the corresponding convolution (number of column rows horizontally).
    ///   - kernelSize: Spatial size of the convolution kernel (assumed square).
    ///   - stride: Convolution stride used when producing the columns.
    ///   - padding: Padding applied to the input when producing the columns.
    func encodeCol2im(
        commandBuffer: MTLCommandBuffer,
        input: MpsBuffer,
        output: MpsBuffer,
        batch: Int,
        inChannels: Int,
        inHeight: Int,
        inWidth: Int,
        outHeight: Int,
        outWidth: Int,
        kernelSize: Int,
        stride: Int,
        padding: Int
    ) throws {
        var batchU = UInt32(batch)
        var inChannelsU = UInt32(inChannels)
        var inHeightU = UInt32(inHeight)
        var inWidthU = UInt32(inWidth)
        var outHeightU = UInt32(outHeight)
        var outWidthU = UInt32(outWidth)
        var kernelSizeU = UInt32(kernelSize)
        var strideU = UInt32(stride)
        var paddingU = UInt32(padding)

        let kernelArea = kernelSize * kernelSize
        let inputCols = batch * outHeight * outWidth
        let inputRows = inChannels * kernelArea
        let totalElements = inputRows * inputCols

        try dispatch1D(commandBuffer, pipeline: col2imPSO, count: totalElements) { encoder in
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            encoder.setBytes(&batchU, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&inChannelsU, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&inHeightU, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&inWidthU, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&outHeightU, length: MemoryLayout<UInt32>.size, index: 6)
            encoder.setBytes(&outWidthU, length: MemoryLayout<UInt32>.size, index: 7)
            encoder.setBytes(&kernelSizeU, length: MemoryLayout<UInt32>.size, index: 8)
            encoder.setBytes(&strideU, length: MemoryLayout<UInt32>.size, index: 9)
            encoder.setBytes(&paddingU, length: MemoryLayout<UInt32>.size, index: 10)
        }
    }

    /// Adds a per-channel bias to the convolution output stored in `data` and applies an in-place ReLU activation.
    /// 
    /// - Parameters:
    ///   - data: Buffer holding convolution outputs in B x C x H x W layout; modified in-place.
    ///   - bias: Buffer of length `channels` containing per-channel bias values.
    ///   - batch: Number of batches (B).
    ///   - channels: Number of channels (C).
    ///   - height: Spatial height (H).
    ///   - width: Spatial width (W).
    func encodeConvAddBiasRelu(
        commandBuffer: MTLCommandBuffer,
        data: MpsBuffer,
        bias: MpsBuffer,
        batch: Int,
        channels: Int,
        height: Int,
        width: Int
    ) throws {
        var batchU = UInt32(batch)
        var channelsU = UInt32(channels)
        var heightU = UInt32(height)
        var widthU = UInt32(width)
        let totalElements = batch * channels * height * width

        try dispatch1D(commandBuffer, pipeline: convAddBiasReluPSO, count: totalElements) { encoder in
            encoder.setBuffer(data.buffer, offset: 0, index: 0)
            encoder.setBuffer(bias.buffer, offset: 0, index: 1)
            encoder.setBytes(&batchU, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&channelsU, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&heightU, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&widthU, length: MemoryLayout<UInt32>.size, index: 5)
        }
    }

    /// Encodes a convolution-transpose operation followed by bias addition and ReLU into the provided command buffer.
    /// 
    /// Dispatches one thread per output element (batch * channels * spatial) and binds the input, output and bias buffers
    /// together with the integer geometry parameters.
    /// - Parameters:
    ///   - commandBuffer: The command buffer into which the kernel will be encoded.
    ///   - input: GPU buffer containing the input tensor.
    ///   - output: GPU buffer for the output tensor (written in-place by the kernel).
    ///   - bias: GPU buffer holding per-channel bias values.
    ///   - batch: Number of batches.
    ///   - channels: Number of channels.
    ///   - spatial: Spatial size per channel (e.g., height * width).
    func encodeConvTransposeBiasRelu(
        commandBuffer: MTLCommandBuffer,
        input: MpsBuffer,
        output: MpsBuffer,
        bias: MpsBuffer,
        batch: Int,
        channels: Int,
        spatial: Int
    ) throws {
        var batchU = UInt32(batch)
        var channelsU = UInt32(channels)
        var spatialU = UInt32(spatial)
        let totalElements = batch * channels * spatial

        try dispatch1D(commandBuffer, pipeline: convTransposeBiasReluPSO, count: totalElements) { encoder in
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            encoder.setBuffer(bias.buffer, offset: 0, index: 2)
            encoder.setBytes(&batchU, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&channelsU, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&spatialU, length: MemoryLayout<UInt32>.size, index: 5)
        }
    }

    /// Encodes a kernel that reshapes a tensor from BCS (batch, channels, spatial) layout to CBS (channels, batch, spatial) layout into the given command buffer.
    /// 
    /// - Parameters:
    ///   - commandBuffer: The Metal command buffer to encode the compute work into.
    ///   - input: Source buffer containing tensor elements in BCS order.
    ///   - output: Destination buffer to receive tensor elements in CBS order.
    ///   - batch: Number of batches (dimension size for B in BCS).
    ///   - channels: Number of channels (dimension size for C in BCS).
    ///   - spatial: Number of spatial locations per channel (dimension size for S in BCS).
    func encodeReshapeBcsToCbs(
        commandBuffer: MTLCommandBuffer,
        input: MpsBuffer,
        output: MpsBuffer,
        batch: Int,
        channels: Int,
        spatial: Int
    ) throws {
        var batchU = UInt32(batch)
        var channelsU = UInt32(channels)
        var spatialU = UInt32(spatial)
        let totalElements = batch * channels * spatial

        try dispatch1D(commandBuffer, pipeline: reshapeBcsToCbsPSO, count: totalElements) { encoder in
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            encoder.setBytes(&batchU, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&channelsU, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&spatialU, length: MemoryLayout<UInt32>.size, index: 4)
        }
    }
}
#endif
