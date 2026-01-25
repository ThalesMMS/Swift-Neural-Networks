import Foundation
#if canImport(MetalPerformanceShaders)
import Metal
import MetalPerformanceShaders
#endif

// MARK: - Global Training Parameters
// These will be overridden by CLI arguments if provided.

let numInputs = 784
var numHidden = 512
let numOutputs = 10

// Training hyperparameters.
var learningRate: Float = 0.01
var epochs = 10
var batchSize = 64

// MARK: - Helper Functions

/// Wrapper for GEMM operations using the GemmEngine protocol.
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

/// Add the bias to each matrix row.
func addBias(_ data: inout [Float], rows: Int, cols: Int, bias: [Float]) {
    for r in 0..<rows {
        let base = r * cols
        for c in 0..<cols {
            data[base + c] += bias[c]
        }
    }
}

/// Copy batch by indices to keep contiguous memory for GEMM (pointer version).
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

// MARK: - Model Initialization

/// Initialize a dense layer with Xavier/Glorot initialization.
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

/// Network construction 784 -> 512 -> 10.
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

// MARK: - Model Persistence

/// Save the trained model to a binary file.
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

// MARK: - CPU Training

/// Training with shuffling and minibatches.
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
// MARK: - GPU Training

/// Optimized training using MPS with shared CPU/GPU buffers.
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
#endif
