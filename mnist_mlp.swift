// ============================================================================
// DEPRECATED: This monolithic file has been refactored into a modular structure.
//
// Please use the new modular implementation located in:
//   - Sources/NeuralNetwork/
//   - Sources/Layers/
//   - Sources/Training/
//   - Sources/Utils/
//   - Sources/Main/
//
// This file is kept for reference only and will be removed in a future version.
// ============================================================================

import Foundation
import Accelerate
import Darwin

#if canImport(Metal)
import Metal
#endif

#if canImport(MetalPerformanceShadersGraph)
// Training with MPSGraph to keep the whole flow on the GPU.
func trainMpsGraph(
    nn: inout NeuralNetwork,
    images: [Float],
    labels: [UInt8],
    numSamples: Int,
    rng: inout SimpleRng
) {
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        print("MPSGraph device unavailable, falling back to CPU.")
        let cpu = CpuGemmEngine()
        train(nn: &nn, images: images, labels: labels, numSamples: numSamples, engine: cpu, rng: &rng)
        return
    }

    let graph = MPSGraph()
    let inputShape = [NSNumber(value: batchSize), NSNumber(value: numInputs)]
    let labelShape = [NSNumber(value: batchSize), NSNumber(value: numOutputs)]

    let inputTensor = graph.placeholder(shape: inputShape, dataType: .float32, name: "input")
    let labelTensor = graph.placeholder(shape: labelShape, dataType: .float32, name: "labels")

    func dataFromArray(_ array: [Float]) -> Data {
        return array.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    let w1Shape = [NSNumber(value: numInputs), NSNumber(value: numHidden)]
    let b1Shape = [NSNumber(value: numHidden)]
    let w2Shape = [NSNumber(value: numHidden), NSNumber(value: numOutputs)]
    let b2Shape = [NSNumber(value: numOutputs)]

    let w1Var = graph.variable(with: dataFromArray(nn.hidden.weights), shape: w1Shape, dataType: .float32, name: "W1")
    let b1Var = graph.variable(with: dataFromArray(nn.hidden.biases), shape: b1Shape, dataType: .float32, name: "b1")
    let w2Var = graph.variable(with: dataFromArray(nn.output.weights), shape: w2Shape, dataType: .float32, name: "W2")
    let b2Var = graph.variable(with: dataFromArray(nn.output.biases), shape: b2Shape, dataType: .float32, name: "b2")

    let w1Read = graph.read(w1Var, name: "W1_read")
    let b1Read = graph.read(b1Var, name: "b1_read")
    let w2Read = graph.read(w2Var, name: "W2_read")
    let b2Read = graph.read(b2Var, name: "b2_read")

    let hiddenLinear = graph.matrixMultiplication(primary: inputTensor, secondary: w1Read, name: "hidden_mm")
    let hiddenBias = graph.addition(hiddenLinear, b1Read, name: "hidden_bias")
    let hiddenRelu = graph.reLU(with: hiddenBias, name: "hidden_relu")

    let logits = graph.matrixMultiplication(primary: hiddenRelu, secondary: w2Read, name: "output_mm")
    let logitsBias = graph.addition(logits, b2Read, name: "output_bias")

    let loss = graph.softMaxCrossEntropy(
        logitsBias,
        labels: labelTensor,
        axis: 1,
        reuctionType: .mean,
        name: "loss"
    )

    let grads = graph.gradients(of: loss, with: [w1Read, b1Read, w2Read, b2Read], name: "grads")
    guard let gradW1 = grads[w1Read],
          let gradB1 = grads[b1Read],
          let gradW2 = grads[w2Read],
          let gradB2 = grads[b2Read] else {
        print("Failed to build MPSGraph gradients, falling back to CPU.")
        let cpu = CpuGemmEngine()
        train(nn: &nn, images: images, labels: labels, numSamples: numSamples, engine: cpu, rng: &rng)
        return
    }

    let lrTensor = graph.constant(Double(learningRate), dataType: .float32)
    let w1Update = graph.subtraction(w1Read, graph.multiplication(gradW1, lrTensor, name: nil), name: "W1_update")
    let b1Update = graph.subtraction(b1Read, graph.multiplication(gradB1, lrTensor, name: nil), name: "b1_update")
    let w2Update = graph.subtraction(w2Read, graph.multiplication(gradW2, lrTensor, name: nil), name: "W2_update")
    let b2Update = graph.subtraction(b2Read, graph.multiplication(gradB2, lrTensor, name: nil), name: "b2_update")

    let assignW1 = graph.assign(w1Var, tensor: w1Update, name: "assignW1")
    let assignB1 = graph.assign(b1Var, tensor: b1Update, name: "assignB1")
    let assignW2 = graph.assign(w2Var, tensor: w2Update, name: "assignW2")
    let assignB2 = graph.assign(b2Var, tensor: b2Update, name: "assignB2")

    let graphDevice = MPSGraphDevice(mtlDevice: device)
    let inputType = MPSGraphShapedType(shape: inputShape, dataType: .float32)
    let labelType = MPSGraphShapedType(shape: labelShape, dataType: .float32)
    let feeds: [MPSGraphTensor: MPSGraphShapedType] = [
        inputTensor: inputType,
        labelTensor: labelType
    ]

    let executable = graph.compile(
        with: graphDevice,
        feeds: feeds,
        targetTensors: [loss],
        targetOperations: [assignW1, assignB1, assignW2, assignB2],
        compilationDescriptor: nil
    )

    // Input/label buffers to feed the graph.
    let inputBytes = batchSize * numInputs * MemoryLayout<Float>.size
    let labelBytes = batchSize * numOutputs * MemoryLayout<Float>.size
    guard let inputBuffer = device.makeBuffer(length: inputBytes, options: .storageModeShared),
          let labelBuffer = device.makeBuffer(length: labelBytes, options: .storageModeShared) else {
        print("Failed to allocate MPSGraph buffers, falling back to CPU.")
        let cpu = CpuGemmEngine()
        train(nn: &nn, images: images, labels: labels, numSamples: numSamples, engine: cpu, rng: &rng)
        return
    }

    let inputData = MPSGraphTensorData(inputBuffer, shape: inputShape, dataType: .float32)
    let labelData = MPSGraphTensorData(labelBuffer, shape: labelShape, dataType: .float32)

    // Keep a fixed batch size (drop the remainder to simplify the graph).
    let effectiveSamples = (numSamples / batchSize) * batchSize
    if effectiveSamples < numSamples {
        print("MPSGraph: descartando \(numSamples - effectiveSamples) amostras para manter batch fixo.")
    }

    var indices = Array(0..<effectiveSamples)
    let inputPtr = inputBuffer.contents().bindMemory(to: Float.self, capacity: batchSize * numInputs)
    let labelPtr = labelBuffer.contents().bindMemory(to: Float.self, capacity: batchSize * numOutputs)

    images.withUnsafeBufferPointer { imagesBuf in
        labels.withUnsafeBufferPointer { labelsBuf in
            guard let imagesBase = imagesBuf.baseAddress,
                  let labelsBase = labelsBuf.baseAddress else {
                return
            }

            for epoch in 0..<epochs {
                var totalLoss: Float = 0.0
                let startTime = Date()

                // Fisher-Yates shuffle.
                if effectiveSamples > 1 {
                    for i in stride(from: effectiveSamples - 1, through: 1, by: -1) {
                        let j = rng.nextInt(upper: i + 1)
                        indices.swapAt(i, j)
                    }
                }

                for batchStart in stride(from: 0, to: effectiveSamples, by: batchSize) {
                    // Copy inputs into the GPU buffer.
                    for i in 0..<batchSize {
                        let srcIndex = indices[batchStart + i]
                        let srcOffset = srcIndex * numInputs
                        let dstOffset = i * numInputs
                        let srcPtr = imagesBase.advanced(by: srcOffset)
                        inputPtr.advanced(by: dstOffset).update(from: srcPtr, count: numInputs)
                    }

                    // One-hot labels into the GPU buffer.
                    memset(labelPtr, 0, labelBytes)
                    for i in 0..<batchSize {
                        let label = Int(labelsBase[indices[batchStart + i]])
                        labelPtr[i * numOutputs + label] = 1.0
                    }

                    let results: [MPSGraphTensorData]? = nil
                    let execDesc: MPSGraphExecutableExecutionDescriptor? = nil
                    let outputs = executable.run(
                        with: queue,
                        inputs: [inputData, labelData],
                        results: results,
                        executionDescriptor: execDesc
                    )

                    if let lossData = outputs.first {
                        let ndarray = lossData.mpsndarray()
                        var lossValue: Float = 0.0
                        ndarray.readBytes(&lossValue, strideBytes: nil)
                        totalLoss += lossValue * Float(batchSize)
                    }
                }

                let duration = Float(Date().timeIntervalSince(startTime))
                let avgLoss = totalLoss / Float(effectiveSamples)
                print("Epoch \(epoch + 1), Loss: \(String(format: "%.6f", avgLoss)) Time: \(String(format: "%.6f", duration))")
            }
        }
    }

    // Read final weights to save and test on CPU.
    let feedDict: [MPSGraphTensor: MPSGraphTensorData] = [
        inputTensor: inputData,
        labelTensor: labelData
    ]
    let readResults = graph.run(
        with: queue,
        feeds: feedDict,
        targetTensors: [w1Read, b1Read, w2Read, b2Read],
        targetOperations: nil
    )

    func readTensor(_ data: MPSGraphTensorData, count: Int) -> [Float] {
        var values = [Float](repeating: 0.0, count: count)
        let ndarray = data.mpsndarray()
        values.withUnsafeMutableBufferPointer { buf in
            guard let ptr = buf.baseAddress else { return }
            ndarray.readBytes(ptr, strideBytes: nil)
        }
        return values
    }

    if let w1Data = readResults[w1Read],
       let b1Data = readResults[b1Read],
       let w2Data = readResults[w2Read],
       let b2Data = readResults[b2Read] {
        nn.hidden.weights = readTensor(w1Data, count: numInputs * numHidden)
        nn.hidden.biases = readTensor(b1Data, count: numHidden)
        nn.output.weights = readTensor(w2Data, count: numHidden * numOutputs)
        nn.output.biases = readTensor(b2Data, count: numOutputs)
    } else {
        print("Warning: failed to read weights from MPSGraph, keeping previous values.")
    }
}

// Test using MPSGraph (GPU inference).
func testMpsGraph(
    nn: NeuralNetwork,
    images: [Float],
    labels: [UInt8],
    numSamples: Int
) {
    guard let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        print("MPSGraph device unavailable, falling back to CPU test.")
        test(nn: nn, images: images, labels: labels, numSamples: numSamples)
        return
    }

    let graph = MPSGraph()
    let inputShape = [NSNumber(value: batchSize), NSNumber(value: numInputs)]
    let inputTensor = graph.placeholder(shape: inputShape, dataType: .float32, name: "input")

    func dataFromArray(_ array: [Float]) -> Data {
        return array.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    let w1Shape = [NSNumber(value: numInputs), NSNumber(value: numHidden)]
    let b1Shape = [NSNumber(value: numHidden)]
    let w2Shape = [NSNumber(value: numHidden), NSNumber(value: numOutputs)]
    let b2Shape = [NSNumber(value: numOutputs)]

    let w1Var = graph.variable(with: dataFromArray(nn.hidden.weights), shape: w1Shape, dataType: .float32, name: "W1_test")
    let b1Var = graph.variable(with: dataFromArray(nn.hidden.biases), shape: b1Shape, dataType: .float32, name: "b1_test")
    let w2Var = graph.variable(with: dataFromArray(nn.output.weights), shape: w2Shape, dataType: .float32, name: "W2_test")
    let b2Var = graph.variable(with: dataFromArray(nn.output.biases), shape: b2Shape, dataType: .float32, name: "b2_test")

    let w1Read = graph.read(w1Var, name: "W1_read_test")
    let b1Read = graph.read(b1Var, name: "b1_read_test")
    let w2Read = graph.read(w2Var, name: "W2_read_test")
    let b2Read = graph.read(b2Var, name: "b2_read_test")

    let hiddenLinear = graph.matrixMultiplication(primary: inputTensor, secondary: w1Read, name: "hidden_mm_test")
    let hiddenBias = graph.addition(hiddenLinear, b1Read, name: "hidden_bias_test")
    let hiddenRelu = graph.reLU(with: hiddenBias, name: "hidden_relu_test")

    let logits = graph.matrixMultiplication(primary: hiddenRelu, secondary: w2Read, name: "output_mm_test")
    let logitsBias = graph.addition(logits, b2Read, name: "logits_test")

    let graphDevice = MPSGraphDevice(mtlDevice: device)
    let inputType = MPSGraphShapedType(shape: inputShape, dataType: .float32)
    let feeds: [MPSGraphTensor: MPSGraphShapedType] = [
        inputTensor: inputType
    ]

    let executable = graph.compile(
        with: graphDevice,
        feeds: feeds,
        targetTensors: [logitsBias],
        targetOperations: nil,
        compilationDescriptor: nil
    )

    let inputBytes = batchSize * numInputs * MemoryLayout<Float>.size
    guard let inputBuffer = device.makeBuffer(length: inputBytes, options: .storageModeShared) else {
        print("Failed to allocate MPSGraph test buffers, falling back to CPU test.")
        test(nn: nn, images: images, labels: labels, numSamples: numSamples)
        return
    }
    let inputData = MPSGraphTensorData(inputBuffer, shape: inputShape, dataType: .float32)
    let inputPtr = inputBuffer.contents().bindMemory(to: Float.self, capacity: batchSize * numInputs)

    let fullBatches = numSamples / batchSize
    let remainder = numSamples - fullBatches * batchSize
    var correct = 0
    var logitsHost = [Float](repeating: 0.0, count: batchSize * numOutputs)

    images.withUnsafeBufferPointer { imagesBuf in
        guard let imagesBase = imagesBuf.baseAddress else { return }
        for batch in 0..<fullBatches {
            let start = batch * batchSize
            let src = imagesBase.advanced(by: start * numInputs)
            inputPtr.update(from: src, count: batchSize * numInputs)

            let outputs = executable.run(
                with: queue,
                inputs: [inputData],
                results: nil,
                executionDescriptor: nil
            )

            guard let logitsData = outputs.first else { continue }
            let ndarray = logitsData.mpsndarray()
            logitsHost.withUnsafeMutableBufferPointer { buf in
                guard let ptr = buf.baseAddress else { return }
                ndarray.readBytes(ptr, strideBytes: nil)
            }

            for i in 0..<batchSize {
                let base = i * numOutputs
                var maxVal = logitsHost[base]
                var maxIdx = 0
                for o in 1..<numOutputs {
                    let v = logitsHost[base + o]
                    if v > maxVal {
                        maxVal = v
                        maxIdx = o
                    }
                }
                if UInt8(maxIdx) == labels[start + i] {
                    correct += 1
                }
            }
        }
    }

    if remainder > 0 {
        correct += testCpuRange(
            nn: nn,
            images: images,
            labels: labels,
            start: fullBatches * batchSize,
            count: remainder
        )
    }

    let accuracy = Float(correct) / Float(numSamples) * 100.0
    print(String(format: "Test Accuracy: %.2f%%", accuracy))
}
#endif

#if canImport(MetalPerformanceShaders)
import MetalPerformanceShaders
#endif

#if canImport(MetalPerformanceShadersGraph)
import MetalPerformanceShadersGraph
#endif

// Sequential MLP for MNIST (Swift port for study and optimization).
let numInputs = 784
var numHidden = 512
let numOutputs = 10
let trainSamples = 60_000
let testSamples = 10_000
var learningRate: Float = 0.01
var epochs = 10
var batchSize = 64
var rngSeed: UInt64 = 1
// Activation types used in the network.
enum ActivationType {
    case sigmoid
    case relu
    case softmax
}

// Dense layer: weights (input x output), biases, and activation.
struct DenseLayer {
    let inputSize: Int
    let outputSize: Int
    var weights: [Float] // row-major
    var biases: [Float]
    let activation: ActivationType
}

// Network with one hidden layer and one output layer.
struct NeuralNetwork {
    var hidden: DenseLayer
    var output: DenseLayer
}

// GEMM interface (matrix multiplication).
protocol GemmEngine {
    func gemm(
        m: Int,
        n: Int,
        k: Int,
        a: UnsafePointer<Float>,
        lda: Int,
        b: UnsafePointer<Float>,
        ldb: Int,
        c: UnsafeMutablePointer<Float>,
        ldc: Int,
        transposeA: Bool,
        transposeB: Bool,
        alpha: Float,
        beta: Float
    )
}

// CPU backend using vDSP (Accelerate).
final class CpuGemmEngine: GemmEngine {
    func gemm(
        m: Int,
        n: Int,
        k: Int,
        a: UnsafePointer<Float>,
        lda: Int,
        b: UnsafePointer<Float>,
        ldb: Int,
        c: UnsafeMutablePointer<Float>,
        ldc: Int,
        transposeA: Bool,
        transposeB: Bool,
        alpha: Float,
        beta: Float
    ) {
        let _ = lda
        let _ = ldb
        let _ = ldc

        let aRows = transposeA ? k : m
        let aCols = transposeA ? m : k
        let bRows = transposeB ? n : k
        let bCols = transposeB ? k : n

        var aTransposed: [Float]? = nil
        var bTransposed: [Float]? = nil

        if transposeA {
            var buffer = [Float](repeating: 0.0, count: m * k)
            buffer.withUnsafeMutableBufferPointer { outBuf in
                guard let outPtr = outBuf.baseAddress else { return }
                vDSP_mtrans(a, 1, outPtr, 1, vDSP_Length(aRows), vDSP_Length(aCols))
            }
            aTransposed = buffer
        }

        if transposeB {
            var buffer = [Float](repeating: 0.0, count: k * n)
            buffer.withUnsafeMutableBufferPointer { outBuf in
                guard let outPtr = outBuf.baseAddress else { return }
                vDSP_mtrans(b, 1, outPtr, 1, vDSP_Length(bRows), vDSP_Length(bCols))
            }
            bTransposed = buffer
        }

        func withMatrixPointer<T>(
            _ matrix: [Float]?,
            fallback: UnsafePointer<Float>,
            _ body: (UnsafePointer<Float>) -> T
        ) -> T {
            if let matrix = matrix {
                return matrix.withUnsafeBufferPointer { buf in
                    guard let ptr = buf.baseAddress else {
                        return body(fallback)
                    }
                    return body(ptr)
                }
            }
            return body(fallback)
        }

        withMatrixPointer(aTransposed, fallback: a) { aPtr in
            withMatrixPointer(bTransposed, fallback: b) { bPtr in
                let count = m * n
                if alpha == 1.0 && beta == 0.0 {
                    vDSP_mmul(
                        aPtr,
                        1,
                        bPtr,
                        1,
                        c,
                        1,
                        vDSP_Length(m),
                        vDSP_Length(n),
                        vDSP_Length(k)
                    )
                } else {
                    var temp = [Float](repeating: 0.0, count: count)
                    temp.withUnsafeMutableBufferPointer { tempBuf in
                        guard let tempPtr = tempBuf.baseAddress else { return }
                        vDSP_mmul(
                            aPtr,
                            1,
                            bPtr,
                            1,
                            tempPtr,
                            1,
                            vDSP_Length(m),
                            vDSP_Length(n),
                            vDSP_Length(k)
                        )

                        var alphaVar = alpha
                        vDSP_vsmul(tempPtr, 1, &alphaVar, tempPtr, 1, vDSP_Length(count))

                        if beta != 0.0 {
                            var betaVar = beta
                            vDSP_vsma(c, 1, &betaVar, tempPtr, 1, tempPtr, 1, vDSP_Length(count))
                        }

                        c.update(from: tempPtr, count: count)
                    }
                }
            }
        }
    }
}

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

    func makeBuffer(count: Int, label: String, initial: [Float]? = nil) -> MpsBuffer {
        return MpsBuffer(device: device, count: count, label: label, initial: initial)
    }

    // GEMM GPU: C = alpha * A * B + beta * C (encode only).
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

    // GEMM GPU: C = alpha * A * B + beta * C.
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
    ) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
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
    }
}
#endif

// Backend selected at runtime.
enum GemmBackend {
    case cpu(CpuGemmEngine)
    #if canImport(MetalPerformanceShaders)
    case mps(MpsGemmEngine)
    #endif
}

// Select backend: --mps tries GPU, otherwise use CPU.
func selectGemmBackend(useMPS: Bool) -> GemmBackend {
    if useMPS {
        #if canImport(MetalPerformanceShaders)
        if let engine = MpsGemmEngine() {
            print("Using MPS GEMM backend (shared buffers).")
            return .mps(engine)
        }
        #endif
        print("MPS not available, falling back to CPU.")
    }
    return .cpu(CpuGemmEngine())
}

// Wrapper for GEMM with Swift arrays.
func gemm(
    engine: GemmEngine,
    m: Int,
    n: Int,
    k: Int,
    a: [Float],
    lda: Int,
    b: [Float],
    ldb: Int,
    c: inout [Float],
    ldc: Int,
    transposeA: Bool = false,
    transposeB: Bool = false,
    alpha: Float = 1.0,
    beta: Float = 0.0
) {
    a.withUnsafeBufferPointer { aBuf in
        b.withUnsafeBufferPointer { bBuf in
            c.withUnsafeMutableBufferPointer { cBuf in
                guard let aPtr = aBuf.baseAddress,
                      let bPtr = bBuf.baseAddress,
                      let cPtr = cBuf.baseAddress else {
                    return
                }
                engine.gemm(
                    m: m,
                    n: n,
                    k: k,
                    a: aPtr,
                    lda: lda,
                    b: bPtr,
                    ldb: ldb,
                    c: cPtr,
                    ldc: ldc,
                    transposeA: transposeA,
                    transposeB: transposeB,
                    alpha: alpha,
                    beta: beta
                )
            }
        }
    }
}

// Initialize a layer with Xavier and zero biases.
func initializeLayer(
    inputSize: Int,
    outputSize: Int,
    activation: ActivationType,
    rng: inout SimpleRng
) -> DenseLayer {
    let limit = sqrtf(6.0 / Float(inputSize + outputSize))
    var weights = [Float](repeating: 0.0, count: inputSize * outputSize)
    for i in 0..<weights.count {
        weights[i] = rng.uniform(-limit, limit)
    }
    let biases = [Float](repeating: 0.0, count: outputSize)
    return DenseLayer(
        inputSize: inputSize,
        outputSize: outputSize,
        weights: weights,
        biases: biases,
        activation: activation
    )
}

// Network construction 784 -> 512 -> 10.
func initializeNetwork(rng: inout SimpleRng) -> NeuralNetwork {
    rng.reseedFromTime()
    let hidden = initializeLayer(
        inputSize: numInputs,
        outputSize: numHidden,
        activation: .relu,
        rng: &rng
    )
    let output = initializeLayer(
        inputSize: numHidden,
        outputSize: numOutputs,
        activation: .softmax,
        rng: &rng
    )
    return NeuralNetwork(hidden: hidden, output: output)
}

// Add the bias to each matrix row.
func addBias(_ data: inout [Float], rows: Int, cols: Int, bias: [Float]) {
    for r in 0..<rows {
        let base = r * cols
        for c in 0..<cols {
            data[base + c] += bias[c]
        }
    }
}

// ReLU in-place.
func reluInPlace(_ data: inout [Float], count: Int) {
    for i in 0..<count {
        if data[i] < 0 {
            data[i] = 0
        }
    }
}

// Create output delta (softmax + cross-entropy) and accumulate loss.
func computeDeltaAndLoss(
    outputs: [Float],
    labels: [UInt8],
    rows: Int,
    cols: Int,
    delta: inout [Float],
    totalLoss: inout Float
) {
    for r in 0..<rows {
        let base = r * cols
        for c in 0..<cols {
            delta[base + c] = outputs[base + c]
        }
        let label = Int(labels[r])
        let prob = max(outputs[base + label], 1e-9)
        totalLoss += -logf(prob)
        delta[base + label] -= 1
    }
}

// Sum columns (bias gradients).
func sumRows(_ data: [Float], rows: Int, cols: Int, result: inout [Float]) {
    for c in 0..<cols {
        result[c] = 0
    }
    for r in 0..<rows {
        let base = r * cols
        for c in 0..<cols {
            result[c] += data[base + c]
        }
    }
}

// Copy batch by indices to keep contiguous memory for GEMM.
func gatherBatch(
    images: [Float],
    labels: [UInt8],
    indices: [Int],
    start: Int,
    count: Int,
    inputSize: Int,
    outInputs: inout [Float],
    outLabels: inout [UInt8]
) {
    images.withUnsafeBufferPointer { srcBuf in
        outInputs.withUnsafeMutableBufferPointer { dstBuf in
            guard let srcBase = srcBuf.baseAddress, let dstBase = dstBuf.baseAddress else {
                return
            }
            for i in 0..<count {
                let srcIndex = indices[start + i]
                let srcPtr = srcBase.advanced(by: srcIndex * inputSize)
                let dstPtr = dstBase.advanced(by: i * inputSize)
                dstPtr.update(from: srcPtr, count: inputSize)
                outLabels[i] = labels[srcIndex]
            }
        }
    }
}

// Pointer-based versions for MPS shared buffers.
func addBiasPointer(_ data: UnsafeMutablePointer<Float>, rows: Int, cols: Int, bias: [Float]) {
    for r in 0..<rows {
        let base = r * cols
        for c in 0..<cols {
            data[base + c] += bias[c]
        }
    }
}

func reluInPlacePointer(_ data: UnsafeMutablePointer<Float>, count: Int) {
    for i in 0..<count {
        if data[i] < 0 {
            data[i] = 0
        }
    }
}

func computeDeltaAndLossPointer(
    outputs: UnsafePointer<Float>,
    labels: [UInt8],
    rows: Int,
    cols: Int,
    delta: UnsafeMutablePointer<Float>,
    totalLoss: inout Float
) {
    for r in 0..<rows {
        let base = r * cols
        for c in 0..<cols {
            delta[base + c] = outputs[base + c]
        }
        let label = Int(labels[r])
        let prob = max(outputs[base + label], 1e-9)
        totalLoss += -logf(prob)
        delta[base + label] -= 1
    }
}

func sumRowsPointer(_ data: UnsafePointer<Float>, rows: Int, cols: Int, result: inout [Float]) {
    for c in 0..<cols {
        result[c] = 0
    }
    for r in 0..<rows {
        let base = r * cols
        for c in 0..<cols {
            result[c] += data[base + c]
        }
    }
}

func gatherBatchToPointer(
    images: [Float],
    labels: [UInt8],
    indices: [Int],
    start: Int,
    count: Int,
    inputSize: Int,
    outInputs: UnsafeMutablePointer<Float>,
    outLabels: UnsafeMutablePointer<UInt8>
) {
    images.withUnsafeBufferPointer { srcBuf in
        guard let srcBase = srcBuf.baseAddress else { return }
        for i in 0..<count {
            let srcIndex = indices[start + i]
            let srcPtr = srcBase.advanced(by: srcIndex * inputSize)
            let dstPtr = outInputs.advanced(by: i * inputSize)
            dstPtr.update(from: srcPtr, count: inputSize)
            outLabels[i] = labels[srcIndex]
        }
    }
}

// Copy a contiguous block (no shuffle) into the input buffer.
func copyContiguousBatchToPointer(
    images: [Float],
    start: Int,
    count: Int,
    inputSize: Int,
    outInputs: UnsafeMutablePointer<Float>
) {
    let total = count * inputSize
    images.withUnsafeBufferPointer { srcBuf in
        guard let srcBase = srcBuf.baseAddress else { return }
        let srcPtr = srcBase.advanced(by: start * inputSize)
        outInputs.update(from: srcPtr, count: total)
    }
}

// Vectorized SGD update: weights = weights - lr * grads.
func applySgdUpdate(
    weights: UnsafeMutablePointer<Float>,
    grads: UnsafePointer<Float>,
    count: Int,
    learningRate: Float
) {
    var negLr = -learningRate
    vDSP_vsma(grads, 1, &negLr, weights, 1, weights, 1, vDSP_Length(count))
}

// Training with shuffling and minibatches.
func train(
    nn: inout NeuralNetwork,
    images: [Float],
    labels: [UInt8],
    numSamples: Int,
    engine: GemmEngine,
    rng: inout SimpleRng
) {
    let logsPath = "./logs"
    let logFile = "\(logsPath)/training_loss_c.txt"
    try? FileManager.default.createDirectory(atPath: logsPath, withIntermediateDirectories: true)
    FileManager.default.createFile(atPath: logFile, contents: nil)
    let logHandle = try? FileHandle(forWritingTo: URL(fileURLWithPath: logFile))
    defer { try? logHandle?.close() }

    // Buffers reused to avoid per-batch allocations.
    var batchInputs = [Float](repeating: 0.0, count: batchSize * numInputs)
    var batchLabels = [UInt8](repeating: 0, count: batchSize)
    var a1 = [Float](repeating: 0.0, count: batchSize * numHidden)
    var a2 = [Float](repeating: 0.0, count: batchSize * numOutputs)
    var dZ2 = [Float](repeating: 0.0, count: batchSize * numOutputs)
    var dZ1 = [Float](repeating: 0.0, count: batchSize * numHidden)
    var gradW1 = [Float](repeating: 0.0, count: numInputs * numHidden)
    var gradW2 = [Float](repeating: 0.0, count: numHidden * numOutputs)
    var gradB1 = [Float](repeating: 0.0, count: numHidden)
    var gradB2 = [Float](repeating: 0.0, count: numOutputs)

    var indices = Array(0..<numSamples)

    for epoch in 0..<epochs {
        var totalLoss: Float = 0.0
        let startTime = Date()

        // Fisher-Yates shuffle.
        if numSamples > 1 {
            for i in stride(from: numSamples - 1, through: 1, by: -1) {
                let j = rng.nextInt(upper: i + 1)
                indices.swapAt(i, j)
            }
        }

        for batchStart in stride(from: 0, to: numSamples, by: batchSize) {
            let batchCount = min(batchSize, numSamples - batchStart)
            let scale = 1.0 / Float(batchCount)

            // Copy batch into a contiguous buffer.
            gatherBatch(
                images: images,
                labels: labels,
                indices: indices,
                start: batchStart,
                count: batchCount,
                inputSize: numInputs,
                outInputs: &batchInputs,
                outLabels: &batchLabels
            )

            // Forward: hidden layer.
            gemm(
                engine: engine,
                m: batchCount,
                n: numHidden,
                k: numInputs,
                a: batchInputs,
                lda: numInputs,
                b: nn.hidden.weights,
                ldb: numHidden,
                c: &a1,
                ldc: numHidden
            )
            addBias(&a1, rows: batchCount, cols: numHidden, bias: nn.hidden.biases)
            reluInPlace(&a1, count: batchCount * numHidden)

            // Forward: output layer.
            gemm(
                engine: engine,
                m: batchCount,
                n: numOutputs,
                k: numHidden,
                a: a1,
                lda: numHidden,
                b: nn.output.weights,
                ldb: numOutputs,
                c: &a2,
                ldc: numOutputs
            )
            addBias(&a2, rows: batchCount, cols: numOutputs, bias: nn.output.biases)
            softmaxRows(&a2, rows: batchCount, cols: numOutputs)

            // Output delta and loss.
            computeDeltaAndLoss(
                outputs: a2,
                labels: batchLabels,
                rows: batchCount,
                cols: numOutputs,
                delta: &dZ2,
                totalLoss: &totalLoss
            )

            // Output-layer gradients: dW2 = A1^T * dZ2.
            gemm(
                engine: engine,
                m: numHidden,
                n: numOutputs,
                k: batchCount,
                a: a1,
                lda: numHidden,
                b: dZ2,
                ldb: numOutputs,
                c: &gradW2,
                ldc: numOutputs,
                transposeA: true,
                transposeB: false,
                alpha: scale
            )
            sumRows(dZ2, rows: batchCount, cols: numOutputs, result: &gradB2)
            for i in 0..<gradB2.count {
                gradB2[i] *= scale
            }

            // Hidden-layer gradient: dZ1 = dZ2 * W2^T.
            gemm(
                engine: engine,
                m: batchCount,
                n: numHidden,
                k: numOutputs,
                a: dZ2,
                lda: numOutputs,
                b: nn.output.weights,
                ldb: numOutputs,
                c: &dZ1,
                ldc: numHidden,
                transposeA: false,
                transposeB: true
            )
            for i in 0..<(batchCount * numHidden) {
                if a1[i] <= 0 {
                    dZ1[i] = 0
                }
            }

            // Hidden-layer gradients: dW1 = X^T * dZ1.
            gemm(
                engine: engine,
                m: numInputs,
                n: numHidden,
                k: batchCount,
                a: batchInputs,
                lda: numInputs,
                b: dZ1,
                ldb: numHidden,
                c: &gradW1,
                ldc: numHidden,
                transposeA: true,
                transposeB: false,
                alpha: scale
            )
            sumRows(dZ1, rows: batchCount, cols: numHidden, result: &gradB1)
            for i in 0..<gradB1.count {
                gradB1[i] *= scale
            }

            // Update weights and biases (SGD).
            for i in 0..<nn.output.weights.count {
                nn.output.weights[i] -= learningRate * gradW2[i]
            }
            for i in 0..<nn.output.biases.count {
                nn.output.biases[i] -= learningRate * gradB2[i]
            }
            for i in 0..<nn.hidden.weights.count {
                nn.hidden.weights[i] -= learningRate * gradW1[i]
            }
            for i in 0..<nn.hidden.biases.count {
                nn.hidden.biases[i] -= learningRate * gradB1[i]
            }
        }

        let duration = Float(Date().timeIntervalSince(startTime))
        let avgLoss = totalLoss / Float(numSamples)
        print("Epoch \(epoch + 1), Loss: \(String(format: "%.6f", avgLoss)) Time: \(String(format: "%.6f", duration))")
        if let handle = logHandle {
            let line = "\(epoch + 1),\(avgLoss),\(duration)\n"
            handle.write(Data(line.utf8))
        }
    }
}

#if canImport(MetalPerformanceShaders)
// Optimized training using MPS with shared CPU/GPU buffers.
func trainMps(
    nn: inout NeuralNetwork,
    images: [Float],
    labels: [UInt8],
    numSamples: Int,
    engine: MpsGemmEngine,
    rng: inout SimpleRng
) {
    let logsPath = "./logs"
    let logFile = "\(logsPath)/training_loss_c.txt"
    try? FileManager.default.createDirectory(atPath: logsPath, withIntermediateDirectories: true)
    FileManager.default.createFile(atPath: logFile, contents: nil)
    let logHandle = try? FileHandle(forWritingTo: URL(fileURLWithPath: logFile))
    defer { try? logHandle?.close() }

    guard let kernels = MpsKernels(device: engine.device) else {
        print("Metal kernels unavailable, falling back to CPU training.")
        let cpu = CpuGemmEngine()
        train(nn: &nn, images: images, labels: labels, numSamples: numSamples, engine: cpu, rng: &rng)
        return
    }

    // Persistent buffers to avoid per-batch allocation.
    let batchInputs = engine.makeBuffer(count: batchSize * numInputs, label: "batchInputs")
    let batchLabels = MpsBufferU8(device: engine.device, count: batchSize, label: "batchLabels")
    let loss = engine.makeBuffer(count: batchSize, label: "loss")

    let a1 = engine.makeBuffer(count: batchSize * numHidden, label: "a1")
    let a2 = engine.makeBuffer(count: batchSize * numOutputs, label: "a2")
    let dZ2 = engine.makeBuffer(count: batchSize * numOutputs, label: "dZ2")
    let dZ1 = engine.makeBuffer(count: batchSize * numHidden, label: "dZ1")
    let gradW1 = engine.makeBuffer(count: numInputs * numHidden, label: "gradW1")
    let gradW2 = engine.makeBuffer(count: numHidden * numOutputs, label: "gradW2")
    let gradB1 = engine.makeBuffer(count: numHidden, label: "gradB1")
    let gradB2 = engine.makeBuffer(count: numOutputs, label: "gradB2")

    let w1 = engine.makeBuffer(count: nn.hidden.weights.count, label: "W1", initial: nn.hidden.weights)
    let b1 = engine.makeBuffer(count: nn.hidden.biases.count, label: "b1", initial: nn.hidden.biases)
    let w2 = engine.makeBuffer(count: nn.output.weights.count, label: "W2", initial: nn.output.weights)
    let b2 = engine.makeBuffer(count: nn.output.biases.count, label: "b2", initial: nn.output.biases)

    var indices = Array(0..<numSamples)

    for epoch in 0..<epochs {
        var totalLoss: Float = 0.0
        let startTime = Date()

        // Fisher-Yates shuffle.
        if numSamples > 1 {
            for i in stride(from: numSamples - 1, through: 1, by: -1) {
                let j = rng.nextInt(upper: i + 1)
                indices.swapAt(i, j)
            }
        }

        for batchStart in stride(from: 0, to: numSamples, by: batchSize) {
            let batchCount = min(batchSize, numSamples - batchStart)
            let scale = 1.0 / Float(batchCount)

            // Copy batch into a contiguous buffer.
            gatherBatchToPointer(
                images: images,
                labels: labels,
                indices: indices,
                start: batchStart,
                count: batchCount,
                inputSize: numInputs,
                outInputs: batchInputs.pointer,
                outLabels: batchLabels.pointer
            )

            guard let commandBuffer = engine.commandQueue.makeCommandBuffer() else {
                continue
            }

            // Forward: hidden layer.
            engine.encodeGemm(
                commandBuffer: commandBuffer,
                m: batchCount,
                n: numHidden,
                k: numInputs,
                a: batchInputs,
                b: w1,
                c: a1,
                transposeA: false,
                transposeB: false,
                alpha: 1.0,
                beta: 0.0
            )
            kernels.encodeAddBias(commandBuffer: commandBuffer, data: a1, bias: b1, rows: batchCount, cols: numHidden)
            kernels.encodeRelu(commandBuffer: commandBuffer, data: a1, count: batchCount * numHidden)

            // Forward: output layer.
            engine.encodeGemm(
                commandBuffer: commandBuffer,
                m: batchCount,
                n: numOutputs,
                k: numHidden,
                a: a1,
                b: w2,
                c: a2,
                transposeA: false,
                transposeB: false,
                alpha: 1.0,
                beta: 0.0
            )
            kernels.encodeAddBias(commandBuffer: commandBuffer, data: a2, bias: b2, rows: batchCount, cols: numOutputs)
            kernels.encodeSoftmax(commandBuffer: commandBuffer, data: a2, rows: batchCount, cols: numOutputs)

            // Output delta and loss.
            kernels.encodeDeltaAndLoss(
                commandBuffer: commandBuffer,
                outputs: a2,
                labels: batchLabels,
                delta: dZ2,
                loss: loss,
                rows: batchCount,
                cols: numOutputs
            )

            // Output-layer gradients: dW2 = A1^T * dZ2.
            engine.encodeGemm(
                commandBuffer: commandBuffer,
                m: numHidden,
                n: numOutputs,
                k: batchCount,
                a: a1,
                b: dZ2,
                c: gradW2,
                transposeA: true,
                transposeB: false,
                alpha: scale,
                beta: 0.0
            )
            kernels.encodeSumRows(
                commandBuffer: commandBuffer,
                data: dZ2,
                output: gradB2,
                rows: batchCount,
                cols: numOutputs,
                scale: scale
            )

            // Hidden-layer gradient: dZ1 = dZ2 * W2^T.
            engine.encodeGemm(
                commandBuffer: commandBuffer,
                m: batchCount,
                n: numHidden,
                k: numOutputs,
                a: dZ2,
                b: w2,
                c: dZ1,
                transposeA: false,
                transposeB: true,
                alpha: 1.0,
                beta: 0.0
            )
            kernels.encodeReluGrad(commandBuffer: commandBuffer, activations: a1, grads: dZ1, count: batchCount * numHidden)

            // Hidden-layer gradients: dW1 = X^T * dZ1.
            engine.encodeGemm(
                commandBuffer: commandBuffer,
                m: numInputs,
                n: numHidden,
                k: batchCount,
                a: batchInputs,
                b: dZ1,
                c: gradW1,
                transposeA: true,
                transposeB: false,
                alpha: scale,
                beta: 0.0
            )
            kernels.encodeSumRows(
                commandBuffer: commandBuffer,
                data: dZ1,
                output: gradB1,
                rows: batchCount,
                cols: numHidden,
                scale: scale
            )

            // Update weights and biases (SGD).
            kernels.encodeSgdUpdate(
                commandBuffer: commandBuffer,
                weights: w2,
                grads: gradW2,
                count: w2.count,
                learningRate: learningRate
            )
            kernels.encodeSgdUpdate(
                commandBuffer: commandBuffer,
                weights: b2,
                grads: gradB2,
                count: b2.count,
                learningRate: learningRate
            )
            kernels.encodeSgdUpdate(
                commandBuffer: commandBuffer,
                weights: w1,
                grads: gradW1,
                count: w1.count,
                learningRate: learningRate
            )
            kernels.encodeSgdUpdate(
                commandBuffer: commandBuffer,
                weights: b1,
                grads: gradB1,
                count: b1.count,
                learningRate: learningRate
            )

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            // Sum batch loss (shared buffer).
            var batchLoss: Float = 0.0
            let lossPtr = loss.pointer
            for i in 0..<batchCount {
                batchLoss += lossPtr[i]
            }
            totalLoss += batchLoss
        }

        let duration = Float(Date().timeIntervalSince(startTime))
        let avgLoss = totalLoss / Float(numSamples)
        print("Epoch \(epoch + 1), Loss: \(String(format: "%.6f", avgLoss)) Time: \(String(format: "%.6f", duration))")
        if let handle = logHandle {
            let line = "\(epoch + 1),\(avgLoss),\(duration)\n"
            handle.write(Data(line.utf8))
        }
    }

    // Copy updated weights back into the network struct.
    var hiddenWeights = nn.hidden.weights
    var hiddenBiases = nn.hidden.biases
    var outputWeights = nn.output.weights
    var outputBiases = nn.output.biases

    w1.copy(to: &hiddenWeights)
    b1.copy(to: &hiddenBiases)
    w2.copy(to: &outputWeights)
    b2.copy(to: &outputBiases)

    nn.hidden.weights = hiddenWeights
    nn.hidden.biases = hiddenBiases
    nn.output.weights = outputWeights
    nn.output.biases = outputBiases
}

// Optimized test using MPS (GPU inference).
func testMps(
    nn: NeuralNetwork,
    images: [Float],
    labels: [UInt8],
    numSamples: Int,
    engine: MpsGemmEngine
) {
    guard let kernels = MpsKernels(device: engine.device) else {
        print("Metal kernels unavailable, falling back to CPU test.")
        test(nn: nn, images: images, labels: labels, numSamples: numSamples)
        return
    }

    let batchInputs = engine.makeBuffer(count: batchSize * numInputs, label: "testInputs")
    let a1 = engine.makeBuffer(count: batchSize * numHidden, label: "testA1")
    let a2 = engine.makeBuffer(count: batchSize * numOutputs, label: "testA2")

    let w1 = engine.makeBuffer(count: nn.hidden.weights.count, label: "testW1", initial: nn.hidden.weights)
    let b1 = engine.makeBuffer(count: nn.hidden.biases.count, label: "testB1", initial: nn.hidden.biases)
    let w2 = engine.makeBuffer(count: nn.output.weights.count, label: "testW2", initial: nn.output.weights)
    let b2 = engine.makeBuffer(count: nn.output.biases.count, label: "testB2", initial: nn.output.biases)

    var correct = 0
    images.withUnsafeBufferPointer { imagesBuf in
        guard let imagesBase = imagesBuf.baseAddress else { return }
        for batchStart in stride(from: 0, to: numSamples, by: batchSize) {
            let batchCount = min(batchSize, numSamples - batchStart)
            let src = imagesBase.advanced(by: batchStart * numInputs)
            batchInputs.pointer.update(from: src, count: batchCount * numInputs)

            guard let commandBuffer = engine.commandQueue.makeCommandBuffer() else {
                continue
            }

            engine.encodeGemm(
                commandBuffer: commandBuffer,
                m: batchCount,
                n: numHidden,
                k: numInputs,
                a: batchInputs,
                b: w1,
                c: a1,
                transposeA: false,
                transposeB: false,
                alpha: 1.0,
                beta: 0.0
            )
            kernels.encodeAddBias(commandBuffer: commandBuffer, data: a1, bias: b1, rows: batchCount, cols: numHidden)
            kernels.encodeRelu(commandBuffer: commandBuffer, data: a1, count: batchCount * numHidden)

            engine.encodeGemm(
                commandBuffer: commandBuffer,
                m: batchCount,
                n: numOutputs,
                k: numHidden,
                a: a1,
                b: w2,
                c: a2,
                transposeA: false,
                transposeB: false,
                alpha: 1.0,
                beta: 0.0
            )
            kernels.encodeAddBias(commandBuffer: commandBuffer, data: a2, bias: b2, rows: batchCount, cols: numOutputs)

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let logitsPtr = a2.pointer
            for i in 0..<batchCount {
                let base = i * numOutputs
                var maxVal = logitsPtr[base]
                var maxIdx = 0
                for o in 1..<numOutputs {
                    let v = logitsPtr[base + o]
                    if v > maxVal {
                        maxVal = v
                        maxIdx = o
                    }
                }
                if UInt8(maxIdx) == labels[batchStart + i] {
                    correct += 1
                }
            }
        }
    }

    let accuracy = Float(correct) / Float(numSamples) * 100.0
    print(String(format: "Test Accuracy: %.2f%%", accuracy))
}
#endif


// Evaluate accuracy on the test set (CPU) and return correct count.
func testCpuRange(
    nn: NeuralNetwork,
    images: [Float],
    labels: [UInt8],
    start: Int,
    count: Int
) -> Int {
    var correct = 0
    var hidden = [Float](repeating: 0.0, count: numHidden)
    var output = [Float](repeating: 0.0, count: numOutputs)

    for i in 0..<count {
        let index = start + i
        let base = index * numInputs

        // Forward hidden layer.
        for h in 0..<numHidden {
            var sum = nn.hidden.biases[h]
            let wBase = h
            for j in 0..<numInputs {
                sum += images[base + j] * nn.hidden.weights[wBase + j * numHidden]
            }
            hidden[h] = max(0.0, sum)
        }

        // Forward output layer.
        for o in 0..<numOutputs {
            var sum = nn.output.biases[o]
            let wBase = o
            for h in 0..<numHidden {
                sum += hidden[h] * nn.output.weights[wBase + h * numOutputs]
            }
            output[o] = sum
        }

        // Argmax over logits (equivalent to softmax).
        var maxVal = output[0]
        var maxIdx = 0
        for o in 1..<numOutputs {
            if output[o] > maxVal {
                maxVal = output[o]
                maxIdx = o
            }
        }

        if UInt8(maxIdx) == labels[index] {
            correct += 1
        }
    }

    return correct
}

// Evaluate accuracy on the test set (CPU).
func test(nn: NeuralNetwork, images: [Float], labels: [UInt8], numSamples: Int) {
    let correct = testCpuRange(nn: nn, images: images, labels: labels, start: 0, count: numSamples)
    let accuracy = Float(correct) / Float(numSamples) * 100.0
    print(String(format: "Test Accuracy: %.2f%%", accuracy))
}

// Save the model in binary (int + double, native endianness).
func saveModel(nn: NeuralNetwork, filename: String) {
    FileManager.default.createFile(atPath: filename, contents: nil)
    guard let handle = try? FileHandle(forWritingTo: URL(fileURLWithPath: filename)) else {
        print("Could not open file \(filename) for writing model")
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

    writeInt32(Int32(nn.hidden.inputSize))
    writeInt32(Int32(nn.hidden.outputSize))
    writeInt32(Int32(nn.output.outputSize))

    for w in nn.hidden.weights {
        writeDouble(Double(w))
    }
    for b in nn.hidden.biases {
        writeDouble(Double(b))
    }
    for w in nn.output.weights {
        writeDouble(Double(w))
    }
    for b in nn.output.biases {
        writeDouble(Double(b))
    }

    print("Model saved to \(filename)")
}


func applyCliOverrides() {
    let args = CommandLine.arguments
    var i = 1
    while i < args.count {
        let arg = args[i]
        switch arg {
        case "--batch":
            guard i + 1 < args.count, let value = Int(args[i + 1]), value > 0 else {
                print("Invalid value for --batch")
                exit(1)
            }
            batchSize = value
            i += 1
        case "--hidden":
            guard i + 1 < args.count, let value = Int(args[i + 1]), value > 0 else {
                print("Invalid value for --hidden")
                exit(1)
            }
            numHidden = value
            i += 1
        case "--epochs":
            guard i + 1 < args.count, let value = Int(args[i + 1]), value > 0 else {
                print("Invalid value for --epochs")
                exit(1)
            }
            epochs = value
            i += 1
        case "--lr":
            guard i + 1 < args.count, let value = Float(args[i + 1]), value > 0 else {
                print("Invalid value for --lr")
                exit(1)
            }
            learningRate = value
            i += 1
        case "--seed":
            guard i + 1 < args.count, let value = UInt64(args[i + 1]) else {
                print("Invalid value for --seed")
                exit(1)
            }
            rngSeed = value
            i += 1
        case "--help":
            print("""
Usage: mnist_mlp_swift [--mps] [--mpsgraph] [--batch N] [--hidden N] [--epochs N] [--lr F] [--seed N]
""")
            exit(0)
        default:
            break
        }
        i += 1
    }
}

func main() {
    applyCliOverrides()

    let useMpsGraph = CommandLine.arguments.contains("--mpsgraph")
    let useMPS = CommandLine.arguments.contains("--mps") || useMpsGraph

    let programStart = Date()

    print("Loading training data...")
    let loadStart = Date()
    let trainImages = readMnistImages(path: "./data/train-images.idx3-ubyte", count: trainSamples)
    let trainLabels = readMnistLabels(path: "./data/train-labels.idx1-ubyte", count: trainSamples)

    print("Loading test data...")
    let testImages = readMnistImages(path: "./data/t10k-images.idx3-ubyte", count: testSamples)
    let testLabels = readMnistLabels(path: "./data/t10k-labels.idx1-ubyte", count: testSamples)
    let loadTime = Date().timeIntervalSince(loadStart)
    print(String(format: "Data loading time: %.2f seconds", loadTime))

    print("Initializing neural network...")
    print("Config: hidden=\(numHidden) batch=\(batchSize) epochs=\(epochs) lr=\(learningRate) seed=\(rngSeed)")
    var rng = SimpleRng(seed: rngSeed)
    var nn = initializeNetwork(rng: &rng)

    print("Training neural network...")
    let trainStart = Date()
    var usedGraph = false
    if useMpsGraph {
        #if canImport(MetalPerformanceShadersGraph)
        print("Using MPSGraph backend.")
        trainMpsGraph(
            nn: &nn,
            images: trainImages,
            labels: trainLabels,
            numSamples: trainImages.count / numInputs,
            rng: &rng
        )
        usedGraph = true
        #else
        print("MPSGraph not available, falling back to MPS kernels.")
        #endif
    }

    if !usedGraph {
        let backend = selectGemmBackend(useMPS: useMPS)
        switch backend {
        case .cpu(let cpu):
            train(
                nn: &nn,
                images: trainImages,
                labels: trainLabels,
                numSamples: trainImages.count / numInputs,
                engine: cpu,
                rng: &rng
            )
        #if canImport(MetalPerformanceShaders)
        case .mps(let mps):
            trainMps(
                nn: &nn,
                images: trainImages,
                labels: trainLabels,
                numSamples: trainImages.count / numInputs,
                engine: mps,
                rng: &rng
            )
        #endif
        }
    }
    let trainTime = Date().timeIntervalSince(trainStart)
    print(String(format: "Total training time: %.2f seconds", trainTime))

    print("Testing neural network...")
    let testStart = Date()
    var testedOnGPU = false

    #if canImport(MetalPerformanceShadersGraph)
    if useMpsGraph {
        print("Testing with MPSGraph backend.")
        testMpsGraph(
            nn: nn,
            images: testImages,
            labels: testLabels,
            numSamples: testImages.count / numInputs
        )
        testedOnGPU = true
    }
    #endif

    if !testedOnGPU && useMPS {
        #if canImport(MetalPerformanceShaders)
        if let engine = MpsGemmEngine() {
            print("Testing with MPS GEMM backend.")
            testMps(
                nn: nn,
                images: testImages,
                labels: testLabels,
                numSamples: testImages.count / numInputs,
                engine: engine
            )
            testedOnGPU = true
        }
        #endif
    }

    if !testedOnGPU {
        test(
            nn: nn,
            images: testImages,
            labels: testLabels,
            numSamples: testImages.count / numInputs
        )
    }

    let testTime = Date().timeIntervalSince(testStart)
    print(String(format: "Testing time: %.2f seconds", testTime))

    print("Saving model...")
    saveModel(nn: nn, filename: "mnist_model.bin")

    let totalTime = Date().timeIntervalSince(programStart)
    print("\n=== Performance Summary ===")
    print(String(format: "Data loading time: %.2f seconds", loadTime))
    print(String(format: "Total training time: %.2f seconds", trainTime))
    print(String(format: "Testing time: %.2f seconds", testTime))
    print(String(format: "Total program time: %.2f seconds", totalTime))
    print("========================")
}

main()
