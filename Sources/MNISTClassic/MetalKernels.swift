import Foundation
import MNISTCommon

#if canImport(Metal)
import Metal
#endif

#if canImport(MetalPerformanceShaders)
// CPU/GPU shared buffer using storageModeShared.
final class MpsBuffer {
    let buffer: MTLBuffer
    let count: Int
    let pointer: UnsafeMutablePointer<Float>

    init(device: MTLDevice, count: Int, label: String, initial: [Float]? = nil) {
        let length = count * MemoryLayout<Float>.size
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            let sizeMB = Double(length) / (1024 * 1024)
            ColoredPrint.error("❌ Metal Buffer Allocation Failed")
            ColoredPrint.error("   Buffer: \(label)")
            ColoredPrint.error("   Size: \(count) elements (\(String(format: "%.2f", sizeMB)) MB)")
            ColoredPrint.error("")
            ColoredPrint.error("POSSIBLE CAUSES:")
            ColoredPrint.error("   • Insufficient GPU memory available")
            ColoredPrint.error("   • Too many applications using GPU resources")
            ColoredPrint.error("   • Batch size or model size too large for available memory")
            ColoredPrint.error("")
            ColoredPrint.error("SOLUTIONS:")
            ColoredPrint.error("   1. Reduce batch size with --batch flag (try 16 or 32)")
            ColoredPrint.error("   2. Close other GPU-intensive applications")
            ColoredPrint.error("   3. Check Activity Monitor (GPU tab) for memory usage")
            ColoredPrint.error("   4. Reduce hidden layer size with --hidden flag")
            ColoredPrint.error("")
            ColoredPrint.error("Example: swift run MNISTClassic --batch 16 --hidden 256")
            exit(1)
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

    func update(from array: [Float], count: Int? = nil) {
        let n = count ?? min(array.count, self.count)
        array.withUnsafeBufferPointer { buf in
            guard let src = buf.baseAddress else { return }
            pointer.update(from: src, count: n)
        }
    }

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

    init(device: MTLDevice, count: Int, label: String) {
        let length = count * MemoryLayout<UInt8>.size
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            let sizeKB = Double(length) / 1024
            ColoredPrint.error("❌ Metal Buffer Allocation Failed")
            ColoredPrint.error("   Buffer: \(label)")
            ColoredPrint.error("   Size: \(count) elements (\(String(format: "%.2f", sizeKB)) KB)")
            ColoredPrint.error("")
            ColoredPrint.error("POSSIBLE CAUSES:")
            ColoredPrint.error("   • Insufficient GPU memory available")
            ColoredPrint.error("   • Too many applications using GPU resources")
            ColoredPrint.error("   • Batch size too large for available memory")
            ColoredPrint.error("")
            ColoredPrint.error("SOLUTIONS:")
            ColoredPrint.error("   1. Reduce batch size with --batch flag (try 16 or 32)")
            ColoredPrint.error("   2. Close other GPU-intensive applications")
            ColoredPrint.error("   3. Check Activity Monitor (GPU tab) for memory usage")
            ColoredPrint.error("")
            ColoredPrint.error("Example: swift run MNISTClassic --batch 16")
            exit(1)
        }
        buffer.label = label
        self.buffer = buffer
        self.count = count
        self.pointer = buffer.contents().bindMemory(to: UInt8.self, capacity: count)
        memset(pointer, 0, length)
    }
}

// Metal kernels to operate on GPU tensors (ReLU, softmax, reductions, SGD).
final class MpsKernels {
    private let addBiasPSO: MTLComputePipelineState
    private let reluPSO: MTLComputePipelineState
    private let reluGradPSO: MTLComputePipelineState
    private let softmaxPSO: MTLComputePipelineState
    private let sumRowsPSO: MTLComputePipelineState
    private let deltaLossPSO: MTLComputePipelineState
    private let sgdPSO: MTLComputePipelineState

    init?(device: MTLDevice) {
        let library: MTLLibrary
        // Try loading from default library first (pre-compiled .metal files)
        if let defaultLibrary = device.makeDefaultLibrary() {
            library = defaultLibrary
        } else {
            // Try loading from .metal file and compiling it
            // SPM packages the .metal file in the module's bundle
            if let url = Bundle.module.url(forResource: "MpsKernels", withExtension: "metal"),
               let source = try? String(contentsOf: url, encoding: .utf8) {
                do {
                    library = try device.makeLibrary(source: source, options: nil)
                } catch {
                    ColoredPrint.error("❌ Metal Library Compilation Failed")
                    ColoredPrint.error("   Source: MpsKernels.metal")
                    ColoredPrint.error("   Error: \(error)")
                    ColoredPrint.error("")
                    ColoredPrint.error("POSSIBLE CAUSES:")
                    ColoredPrint.error("   • Syntax error in Metal shader code")
                    ColoredPrint.error("   • Incompatible Metal version or GPU")
                    ColoredPrint.error("   • Corrupted Metal shader source file")
                    ColoredPrint.error("")
                    ColoredPrint.error("SOLUTIONS:")
                    ColoredPrint.error("   1. Check the shader source at: \(url.path)")
                    ColoredPrint.error("   2. Verify your macOS version supports Metal 2.0+")
                    ColoredPrint.error("   3. Re-clone the repository to get fresh shader files:")
                    ColoredPrint.error("      git checkout Sources/MNISTClassic/MpsKernels.metal")
                    ColoredPrint.error("   4. Try rebuilding: swift build --clean && swift build")
                    return nil
                }
            } else {
                ColoredPrint.error("❌ Metal Shader File Not Found")
                ColoredPrint.error("   Looking for: MpsKernels.metal")
                ColoredPrint.error("   Expected location: Sources/MNISTClassic/")
                ColoredPrint.error("")
                ColoredPrint.error("POSSIBLE CAUSES:")
                ColoredPrint.error("   • Missing or corrupted build artifacts")
                ColoredPrint.error("   • Incomplete project checkout")
                ColoredPrint.error("   • SwiftPM bundle resources not properly configured")
                ColoredPrint.error("")
                ColoredPrint.error("SOLUTIONS:")
                ColoredPrint.error("   1. Verify the file exists:")
                ColoredPrint.error("      ls -la Sources/MNISTClassic/MpsKernels.metal")
                ColoredPrint.error("   2. Clean and rebuild the project:")
                ColoredPrint.error("      swift build --clean && swift build")
                ColoredPrint.error("   3. If missing, re-clone the repository:")
                ColoredPrint.error("      git checkout Sources/MNISTClassic/MpsKernels.metal")
                return nil
            }
        }

        func makePSO(_ name: String) -> MTLComputePipelineState? {
            guard let function = library.makeFunction(name: name) else {
                ColoredPrint.error("❌ Metal Kernel Function Not Found")
                ColoredPrint.error("   Missing kernel: \(name)")
                ColoredPrint.error("")
                ColoredPrint.error("POSSIBLE CAUSES:")
                ColoredPrint.error("   • Kernel function name mismatch in Metal shader")
                ColoredPrint.error("   • Metal library compilation partially failed")
                ColoredPrint.error("   • Corrupted Metal shader source")
                ColoredPrint.error("")
                ColoredPrint.error("EXPECTED KERNELS:")
                ColoredPrint.error("   • add_bias, relu_inplace, relu_grad")
                ColoredPrint.error("   • softmax_rows, sum_rows")
                ColoredPrint.error("   • delta_and_loss, sgd_update")
                ColoredPrint.error("")
                ColoredPrint.error("SOLUTIONS:")
                ColoredPrint.error("   1. Verify MpsKernels.metal contains all required kernels")
                ColoredPrint.error("   2. Rebuild the project: swift build --clean && swift build")
                ColoredPrint.error("   3. Restore shader file: git checkout Sources/MNISTClassic/MpsKernels.metal")
                return nil
            }
            do {
                return try device.makeComputePipelineState(function: function)
            } catch {
                ColoredPrint.error("❌ Failed to Create Metal Pipeline State")
                ColoredPrint.error("   Kernel: \(name)")
                ColoredPrint.error("   Error: \(error)")
                ColoredPrint.error("")
                ColoredPrint.error("POSSIBLE CAUSES:")
                ColoredPrint.error("   • Incompatible GPU or Metal version")
                ColoredPrint.error("   • Kernel configuration error")
                ColoredPrint.error("")
                ColoredPrint.error("SOLUTIONS:")
                ColoredPrint.error("   1. Verify your Mac supports Metal 2.0+")
                ColoredPrint.error("   2. Update macOS to the latest version")
                ColoredPrint.error("   3. Try rebuilding: swift build --clean && swift build")
                return nil
            }
        }

        guard let addBiasPSO = makePSO("add_bias"),
              let reluPSO = makePSO("relu_inplace"),
              let reluGradPSO = makePSO("relu_grad"),
              let softmaxPSO = makePSO("softmax_rows"),
              let sumRowsPSO = makePSO("sum_rows"),
              let deltaLossPSO = makePSO("delta_and_loss"),
              let sgdPSO = makePSO("sgd_update") else {
            return nil
        }

        self.addBiasPSO = addBiasPSO
        self.reluPSO = reluPSO
        self.reluGradPSO = reluGradPSO
        self.softmaxPSO = softmaxPSO
        self.sumRowsPSO = sumRowsPSO
        self.deltaLossPSO = deltaLossPSO
        self.sgdPSO = sgdPSO
    }

    private func dispatch1D(
        _ commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        count: Int,
        encode: (MTLComputeCommandEncoder) -> Void
    ) {
        guard count > 0 else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encode(encoder)
        let width = pipeline.threadExecutionWidth
        let threads = MTLSize(width: count, height: 1, depth: 1)
        let group = MTLSize(width: min(width, count), height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: group)
        encoder.endEncoding()
    }

    func encodeAddBias(commandBuffer: MTLCommandBuffer, data: MpsBuffer, bias: MpsBuffer, rows: Int, cols: Int) {
        var rowsU = UInt32(rows)
        var colsU = UInt32(cols)
        dispatch1D(commandBuffer, pipeline: addBiasPSO, count: rows * cols) { encoder in
            encoder.setBuffer(data.buffer, offset: 0, index: 0)
            encoder.setBuffer(bias.buffer, offset: 0, index: 1)
            encoder.setBytes(&rowsU, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&colsU, length: MemoryLayout<UInt32>.size, index: 3)
        }
    }

    func encodeRelu(commandBuffer: MTLCommandBuffer, data: MpsBuffer, count: Int) {
        var countU = UInt32(count)
        dispatch1D(commandBuffer, pipeline: reluPSO, count: count) { encoder in
            encoder.setBuffer(data.buffer, offset: 0, index: 0)
            encoder.setBytes(&countU, length: MemoryLayout<UInt32>.size, index: 1)
        }
    }

    func encodeReluGrad(commandBuffer: MTLCommandBuffer, activations: MpsBuffer, grads: MpsBuffer, count: Int) {
        var countU = UInt32(count)
        dispatch1D(commandBuffer, pipeline: reluGradPSO, count: count) { encoder in
            encoder.setBuffer(activations.buffer, offset: 0, index: 0)
            encoder.setBuffer(grads.buffer, offset: 0, index: 1)
            encoder.setBytes(&countU, length: MemoryLayout<UInt32>.size, index: 2)
        }
    }

    func encodeSoftmax(commandBuffer: MTLCommandBuffer, data: MpsBuffer, rows: Int, cols: Int) {
        var rowsU = UInt32(rows)
        var colsU = UInt32(cols)
        dispatch1D(commandBuffer, pipeline: softmaxPSO, count: rows) { encoder in
            encoder.setBuffer(data.buffer, offset: 0, index: 0)
            encoder.setBytes(&rowsU, length: MemoryLayout<UInt32>.size, index: 1)
            encoder.setBytes(&colsU, length: MemoryLayout<UInt32>.size, index: 2)
        }
    }

    func encodeSumRows(
        commandBuffer: MTLCommandBuffer,
        data: MpsBuffer,
        output: MpsBuffer,
        rows: Int,
        cols: Int,
        scale: Float
    ) {
        var rowsU = UInt32(rows)
        var colsU = UInt32(cols)
        var scaleVar = scale
        dispatch1D(commandBuffer, pipeline: sumRowsPSO, count: cols) { encoder in
            encoder.setBuffer(data.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            encoder.setBytes(&rowsU, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&colsU, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&scaleVar, length: MemoryLayout<Float>.size, index: 4)
        }
    }

    func encodeDeltaAndLoss(
        commandBuffer: MTLCommandBuffer,
        outputs: MpsBuffer,
        labels: MpsBufferU8,
        delta: MpsBuffer,
        loss: MpsBuffer,
        rows: Int,
        cols: Int
    ) {
        var rowsU = UInt32(rows)
        var colsU = UInt32(cols)
        dispatch1D(commandBuffer, pipeline: deltaLossPSO, count: rows) { encoder in
            encoder.setBuffer(outputs.buffer, offset: 0, index: 0)
            encoder.setBuffer(labels.buffer, offset: 0, index: 1)
            encoder.setBuffer(delta.buffer, offset: 0, index: 2)
            encoder.setBuffer(loss.buffer, offset: 0, index: 3)
            encoder.setBytes(&rowsU, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&colsU, length: MemoryLayout<UInt32>.size, index: 5)
        }
    }

    func encodeSgdUpdate(
        commandBuffer: MTLCommandBuffer,
        weights: MpsBuffer,
        grads: MpsBuffer,
        count: Int,
        learningRate: Float
    ) {
        var countU = UInt32(count)
        var lr = learningRate
        dispatch1D(commandBuffer, pipeline: sgdPSO, count: count) { encoder in
            encoder.setBuffer(weights.buffer, offset: 0, index: 0)
            encoder.setBuffer(grads.buffer, offset: 0, index: 1)
            encoder.setBytes(&countU, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&lr, length: MemoryLayout<Float>.size, index: 3)
        }
    }
}
#endif
