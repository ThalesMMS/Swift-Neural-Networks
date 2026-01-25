import Foundation

#if canImport(MetalPerformanceShadersGraph)
import Metal
import MetalPerformanceShadersGraph

// MARK: - Global Training Parameters
// These are shared across modules and defined in Training.swift.

let numInputs = 784
var numHidden = 512
let numOutputs = 10
var learningRate: Float = 0.01
var epochs = 10
var batchSize = 64

// MARK: - MPSGraph Training Functions

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
