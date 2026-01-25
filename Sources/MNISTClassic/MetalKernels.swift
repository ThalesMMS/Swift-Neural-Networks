import Foundation

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
            print("Failed to allocate MTLBuffer for \(label)")
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
            print("Failed to allocate MTLBuffer for \(label)")
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
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void add_bias(device float* data [[buffer(0)]],
                             device const float* bias [[buffer(1)]],
                             constant uint& rows [[buffer(2)]],
                             constant uint& cols [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
            uint total = rows * cols;
            if (gid >= total) return;
            uint col = gid % cols;
            data[gid] += bias[col];
        }

        kernel void relu_inplace(device float* data [[buffer(0)]],
                                 constant uint& count [[buffer(1)]],
                                 uint gid [[thread_position_in_grid]]) {
            if (gid >= count) return;
            float v = data[gid];
            data[gid] = v > 0.0f ? v : 0.0f;
        }

        kernel void relu_grad(device const float* activations [[buffer(0)]],
                              device float* grads [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
            if (gid >= count) return;
            if (activations[gid] <= 0.0f) {
                grads[gid] = 0.0f;
            }
        }

        kernel void softmax_rows(device float* data [[buffer(0)]],
                                 constant uint& rows [[buffer(1)]],
                                 constant uint& cols [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
            if (gid >= rows) return;
            uint base = gid * cols;
            float maxVal = data[base];
            for (uint c = 1; c < cols; ++c) {
                float v = data[base + c];
                if (v > maxVal) maxVal = v;
            }
            float sum = 0.0f;
            for (uint c = 0; c < cols; ++c) {
                float e = exp(data[base + c] - maxVal);
                data[base + c] = e;
                sum += e;
            }
            float inv = 1.0f / sum;
            for (uint c = 0; c < cols; ++c) {
                data[base + c] *= inv;
            }
        }

        kernel void sum_rows(device const float* data [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& rows [[buffer(2)]],
                             constant uint& cols [[buffer(3)]],
                             constant float& scale [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
            if (gid >= cols) return;
            float acc = 0.0f;
            for (uint r = 0; r < rows; ++r) {
                acc += data[r * cols + gid];
            }
            out[gid] = acc * scale;
        }

        kernel void delta_and_loss(device const float* outputs [[buffer(0)]],
                                   device const uchar* labels [[buffer(1)]],
                                   device float* delta [[buffer(2)]],
                                   device float* loss [[buffer(3)]],
                                   constant uint& rows [[buffer(4)]],
                                   constant uint& cols [[buffer(5)]],
                                   uint gid [[thread_position_in_grid]]) {
            if (gid >= rows) return;
            uint base = gid * cols;
            uint label = labels[gid];
            float prob = outputs[base + label];
            if (prob < 1e-9f) prob = 1e-9f;
            loss[gid] = -log(prob);
            for (uint c = 0; c < cols; ++c) {
                float v = outputs[base + c];
                if (c == label) v -= 1.0f;
                delta[base + c] = v;
            }
        }

        kernel void sgd_update(device float* weights [[buffer(0)]],
                               device const float* grads [[buffer(1)]],
                               constant uint& count [[buffer(2)]],
                               constant float& lr [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
            if (gid >= count) return;
            weights[gid] -= lr * grads[gid];
        }
        """

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            print("Failed to compile Metal kernels: \(error)")
            return nil
        }

        func makePSO(_ name: String) -> MTLComputePipelineState? {
            guard let function = library.makeFunction(name: name) else {
                print("Missing Metal kernel: \(name)")
                return nil
            }
            return try? device.makeComputePipelineState(function: function)
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
