import Foundation
#if canImport(MetalPerformanceShaders)
import Metal
import MetalPerformanceShaders
#endif

// MARK: - Testing Functions
// Note: Global testing parameters (numInputs, numHidden, batchSize, etc.) are defined in Training.swift

#if canImport(MetalPerformanceShaders)
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
