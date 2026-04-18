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

/// Entrypoint that trains a MNIST convolutional neural network and saves the trained model.
/// 
/// Parses command-line configuration, loads the MNIST dataset, and trains the CNN for the configured
/// number of epochs using either the Metal Performance Shaders GPU path (when requested and available)
/// or a CPU fallback. Training progress is logged to a file when possible; after training the routine
/// evaluates test accuracy and persists the trained model to disk.
func main() {
    // Parse command-line arguments
    let config = Config.parse()

    // Check if GPU training requested
    let useMPS = config.useGpu

    // Initialize Metal backend if requested and available
    var useGPU = false
    #if canImport(MetalPerformanceShaders)
    var mpsEngine: MpsGemmEngine?
    var mpsKernels: MpsKernels?

    if useMPS {
        if let engine = MpsGemmEngine(), let kernels = MpsKernels(device: engine.device) {
            mpsEngine = engine
            mpsKernels = kernels
            useGPU = true
            print("✓ Metal GPU backend initialized: \(engine.device.name)")
        } else {
            print("⚠️  Metal GPU Not Available - Training will use CPU")
            print("   Reason: No Metal-compatible GPU device found or initialization failed")
            print("   → This is expected on non-Apple Silicon Macs or in virtual machines")
            print("   → Performance will be slower but training will proceed normally")
            print("   → To enable GPU: Ensure you're running on Apple Silicon (M1/M2/M3) hardware")
            useGPU = false
        }
    }
    #else
    if useMPS {
        print("⚠️  Metal Performance Shaders Not Available - Training will use CPU")
        print("   Reason: MetalPerformanceShaders framework not available on this platform")
        print("   → This is expected on non-macOS platforms")
        print("   → Training will proceed normally on CPU")
    }
    #endif

    print("Loading MNIST...")
    let trainImages = readMnistImages(path: "\(config.dataPath)/train-images.idx3-ubyte", count: trainSamples)
    let trainLabels = readMnistLabels(path: "\(config.dataPath)/train-labels.idx1-ubyte", count: trainSamples)
    let testImages  = readMnistImages(path: "\(config.dataPath)/t10k-images.idx3-ubyte", count: testSamples)
    let testLabels  = readMnistLabels(path: "\(config.dataPath)/t10k-labels.idx1-ubyte", count: testSamples)

    print("Train: \(trainLabels.count) | Test: \(testLabels.count)")

    var rng = SimpleRng(seed: config.seed)
    var model = initCnn(rng: &rng)

    let logURL = URL(fileURLWithPath: "./logs/training_loss_cnn.txt")
    var logHandle: FileHandle?
    do {
        try FileManager.default.createDirectory(atPath: "./logs", withIntermediateDirectories: true)
        if !FileManager.default.createFile(atPath: logURL.path, contents: nil),
           !FileManager.default.fileExists(atPath: logURL.path) {
            fputs("Warning: failed to create \(logURL.path); training will proceed without file logging.\n", stderr)
            logHandle = nil
        } else {
            logHandle = try FileHandle(forWritingTo: logURL)
            try logHandle?.truncate(atOffset: 0)
        }
    } catch {
        fputs("Warning: failed to open \(logURL.path): \(error); training will proceed without file logging.\n", stderr)
        logHandle = nil
    }
    defer { try? logHandle?.close() }

    // Training buffers (reused each batch to avoid allocations).
    var batchInputs = [Float](repeating: 0, count: config.batchSize * numInputs)
    var batchLabels = [UInt8](repeating: 0, count: config.batchSize)

    var convAct = [Float](repeating: 0, count: config.batchSize * convOut * imgH * imgW)
    var poolOut = [Float](repeating: 0, count: config.batchSize * fcIn)
    var poolIdx = [UInt8](repeating: 0, count: config.batchSize * convOut * poolH * poolW)
    var logits = [Float](repeating: 0, count: config.batchSize * numClasses)
    var delta  = [Float](repeating: 0, count: config.batchSize * numClasses)

    var dPool = [Float](repeating: 0, count: config.batchSize * fcIn)
    var dConv = [Float](repeating: 0, count: config.batchSize * convOut * imgH * imgW)

    var gradFcW = [Float](repeating: 0, count: fcIn * numClasses)
    var gradFcB = [Float](repeating: 0, count: numClasses)
    var gradConvW = [Float](repeating: 0, count: convOut * kernel * kernel)
    var gradConvB = [Float](repeating: 0, count: convOut)

    var indices = Array(0..<trainLabels.count)

    // GPU buffers (allocated only if useGPU is true)
    #if canImport(MetalPerformanceShaders)
    var gpuBatchInputs: MpsBuffer?
    var gpuBatchLabels: MpsBufferU8?
    var gpuConvAct: MpsBuffer?
    var gpuPoolOut: MpsBuffer?
    var gpuLogits: MpsBuffer?
    var gpuDelta: MpsBuffer?
    var gpuDPool: MpsBuffer?
    var gpuDConv: MpsBuffer?
    var gpuConvW: MpsBuffer?
    var gpuConvB: MpsBuffer?
    var gpuFcW: MpsBuffer?
    var gpuFcB: MpsBuffer?
    var gpuGradConvW: MpsBuffer?
    var gpuGradConvB: MpsBuffer?
    var gpuGradFcW: MpsBuffer?
    var gpuGradFcB: MpsBuffer?
    var gpuColBuffer: MpsBuffer?
    var gpuConvGemm: MpsBuffer?

    if useGPU, let engine = mpsEngine, let _ = mpsKernels {
        do {
            // Allocate GPU buffers for training
            gpuBatchInputs = try engine.makeBuffer(count: config.batchSize * numInputs, label: "batchInputs")
            gpuBatchLabels = try MpsBufferU8(device: engine.device, count: config.batchSize, label: "batchLabels")
            gpuConvAct = try engine.makeBuffer(count: config.batchSize * convOut * imgH * imgW, label: "convAct")
            gpuPoolOut = try engine.makeBuffer(count: config.batchSize * fcIn, label: "poolOut")
            gpuLogits = try engine.makeBuffer(count: config.batchSize * numClasses, label: "logits")
            gpuDelta = try engine.makeBuffer(count: config.batchSize * numClasses, label: "delta")
            gpuDPool = try engine.makeBuffer(count: config.batchSize * fcIn, label: "dPool")
            gpuDConv = try engine.makeBuffer(count: config.batchSize * convOut * imgH * imgW, label: "dConv")

            // Model weights on GPU
            gpuConvW = try engine.makeBuffer(count: convOut * kernel * kernel, label: "convW", initial: model.convW)
            gpuConvB = try engine.makeBuffer(count: convOut, label: "convB", initial: model.convB)
            gpuFcW = try engine.makeBuffer(count: fcIn * numClasses, label: "fcW", initial: model.fcW)
            gpuFcB = try engine.makeBuffer(count: numClasses, label: "fcB", initial: model.fcB)

            // Gradient buffers
            gpuGradConvW = try engine.makeBuffer(count: convOut * kernel * kernel, label: "gradConvW")
            gpuGradConvB = try engine.makeBuffer(count: convOut, label: "gradConvB")
            gpuGradFcW = try engine.makeBuffer(count: fcIn * numClasses, label: "gradFcW")
            gpuGradFcB = try engine.makeBuffer(count: numClasses, label: "gradFcB")

            // Im2col buffer
            gpuColBuffer = try engine.makeBuffer(count: kernel * kernel * imgH * imgW * config.batchSize, label: "colBuffer")

            // Temporary buffer for GEMM output before transposition
            gpuConvGemm = try engine.makeBuffer(count: convOut * imgH * imgW * config.batchSize, label: "convGemm")
        } catch {
            fputs("Warning: Metal GPU buffer allocation failed; falling back to CPU: \(error)\n", stderr)
            useGPU = false
        }
    }
    #endif

    if useGPU {
        print("Training CNN on GPU: epochs=\(config.epochs) batch=\(config.batchSize) lr=\(config.learningRate)")
    } else {
        print("Training CNN on CPU: epochs=\(config.epochs) batch=\(config.batchSize) lr=\(config.learningRate)")
    }

    for e in 0..<config.epochs {
        let t0 = Date()
        rng.shuffle(&indices)

        var totalLoss: Float = 0
        var start = 0
        while start < indices.count {
            let bsz = min(config.batchSize, indices.count - start)
            let scale = 1.0 / Float(bsz)

            // Gather a random mini-batch into contiguous buffers.
            for i in 0..<bsz {
                let srcIndex = indices[start + i]
                let srcBase = srcIndex * numInputs
                let dstBase = i * numInputs
                for j in 0..<numInputs {
                    batchInputs[dstBase + j] = trainImages[srcBase + j]
                }
                batchLabels[i] = trainLabels[srcIndex]
            }

            #if canImport(MetalPerformanceShaders)
            if useGPU,
               let engine = mpsEngine,
               let kernels = mpsKernels,
               let gpuInput = gpuBatchInputs,
               let gpuLabels = gpuBatchLabels,
               let gpuConv = gpuConvAct,
               let gpuPool = gpuPoolOut,
               let gpuLog = gpuLogits,
               let gpuDel = gpuDelta,
               let gpuDP = gpuDPool,
               let gpuDC = gpuDConv,
               let gpuCW = gpuConvW,
               let gpuCB = gpuConvB,
               let gpuFW = gpuFcW,
               let gpuFB = gpuFcB,
               let gpuGCW = gpuGradConvW,
               let gpuGCB = gpuGradConvB,
               let gpuGFW = gpuGradFcW,
               let gpuGFB = gpuGradFcB,
               let gpuCol = gpuColBuffer,
               let gpuConvGemmTemp = gpuConvGemm {

                do {
                    // Copy batch data to GPU
                    gpuInput.update(from: batchInputs, count: bsz * numInputs)
                    gpuLabels.pointer.update(from: batchLabels, count: bsz)

                    // Forward: conv -> pool -> FC -> logits on GPU
                    try convForwardReluGpu(engine: engine, kernels: kernels, batch: bsz, input: gpuInput,
                                           convW: gpuCW, convB: gpuCB, convOutAct: gpuConv, colBuffer: gpuCol, gemmTemp: gpuConvGemmTemp)
                    try maxPoolForwardGpu(engine: engine, kernels: kernels, batch: bsz, input: gpuConv, output: gpuPool)
                    try fcForwardGpu(engine: engine, kernels: kernels, batch: bsz, x: gpuPool, fcW: gpuFW, fcB: gpuFB, logits: gpuLog)

                    // Copy logits back to compute loss on CPU
                    gpuLog.copy(to: &logits)
                    let batchLoss = softmaxXentBackward(probsInPlace: &logits, labels: batchLabels, batch: bsz, delta: &delta, scale: scale)
                    gpuDel.update(from: delta, count: bsz * numClasses)

                    // Backward: FC -> pool -> conv on GPU
                    try fcBackwardGpu(engine: engine, kernels: kernels, batch: bsz, x: gpuPool, delta: gpuDel,
                                      fcW: gpuFW, gradW: gpuGFW, gradB: gpuGFB, dX: gpuDP)
                    try maxPoolBackwardReluGpu(engine: engine, kernels: kernels, batch: bsz, convAct: gpuConv,
                                               poolGrad: gpuDP, convGrad: gpuDC)
                    try convBackwardGpu(engine: engine, kernels: kernels, batch: bsz, input: gpuInput, convGrad: gpuDC,
                                        gradW: gpuGCW, gradB: gpuGCB, colBuffer: gpuCol, gemmTemp: gpuConvGemmTemp)

                    // SGD update on GPU using Metal kernels
                    let cmdBuf = try engine.makeCommandBuffer(operation: "SGD update")
                    try kernels.encodeSgdUpdate(commandBuffer: cmdBuf, weights: gpuCW, grads: gpuGCW, count: convOut * kernel * kernel, learningRate: config.learningRate)
                    try kernels.encodeSgdUpdate(commandBuffer: cmdBuf, weights: gpuCB, grads: gpuGCB, count: convOut, learningRate: config.learningRate)
                    try kernels.encodeSgdUpdate(commandBuffer: cmdBuf, weights: gpuFW, grads: gpuGFW, count: fcIn * numClasses, learningRate: config.learningRate)
                    try kernels.encodeSgdUpdate(commandBuffer: cmdBuf, weights: gpuFB, grads: gpuGFB, count: numClasses, learningRate: config.learningRate)
                    cmdBuf.commit()
                    cmdBuf.waitUntilCompleted()
                    try checkMetalCommandBuffer(cmdBuf, operation: "SGD update")
                    totalLoss += batchLoss
                } catch {
                    fputs("Warning: Metal GPU operation failed; falling back to CPU for this and subsequent batches: \(error)\n", stderr)
                    gpuCW.copy(to: &model.convW)
                    gpuCB.copy(to: &model.convB)
                    gpuFW.copy(to: &model.fcW)
                    gpuFB.copy(to: &model.fcB)
                    useGPU = false
                    gpuBatchInputs = nil
                    gpuBatchLabels = nil
                    gpuConvAct = nil
                    gpuPoolOut = nil
                    gpuLogits = nil
                    gpuDelta = nil
                    gpuDPool = nil
                    gpuDConv = nil
                    gpuConvW = nil
                    gpuConvB = nil
                    gpuFcW = nil
                    gpuFcB = nil
                    gpuGradConvW = nil
                    gpuGradConvB = nil
                    gpuGradFcW = nil
                    gpuGradFcB = nil
                    gpuColBuffer = nil
                    gpuConvGemm = nil
                }
            }
            #endif

            if !useGPU {
                // CPU training path
                // Forward: conv -> pool -> FC -> logits.
                convForwardRelu(model: model, batch: bsz, input: batchInputs, convOutAct: &convAct)
                maxPoolForward(batch: bsz, convAct: convAct, poolOut: &poolOut, poolIdx: &poolIdx)
                fcForward(model: model, batch: bsz, x: poolOut, logits: &logits)

                // Softmax + loss + gradient at logits.
                totalLoss += softmaxXentBackward(probsInPlace: &logits, labels: batchLabels, batch: bsz, delta: &delta, scale: scale)

                // Backward: FC -> pool -> conv.
                fcBackward(model: model, batch: bsz, x: poolOut, delta: delta, gradW: &gradFcW, gradB: &gradFcB, dX: &dPool)
                maxPoolBackwardRelu(batch: bsz, convAct: convAct, poolGrad: dPool, poolIdx: poolIdx, convGrad: &dConv)
                convBackward(model: model, batch: bsz, input: batchInputs, convGrad: dConv, gradW: &gradConvW, gradB: &gradConvB)

                // SGD update (no momentum, no weight decay).
                for i in 0..<model.fcW.count { model.fcW[i] -= config.learningRate * gradFcW[i] }
                for i in 0..<model.fcB.count { model.fcB[i] -= config.learningRate * gradFcB[i] }
                for i in 0..<model.convW.count { model.convW[i] -= config.learningRate * gradConvW[i] }
                for i in 0..<model.convB.count { model.convB[i] -= config.learningRate * gradConvB[i] }
            }

            start += bsz
        }

        #if canImport(MetalPerformanceShaders)
        if useGPU,
           let gpuCW = gpuConvW,
           let gpuCB = gpuConvB,
           let gpuFW = gpuFcW,
           let gpuFB = gpuFcB {
            gpuCW.copy(to: &model.convW)
            gpuCB.copy(to: &model.convB)
            gpuFW.copy(to: &model.fcW)
            gpuFB.copy(to: &model.fcB)
        }
        #endif

        let dt = Float(Date().timeIntervalSince(t0))
        let avgLoss = totalLoss / Float(trainLabels.count)
        print(String(format: "Epoch %d | loss=%.6f | time=%.3fs", e + 1, avgLoss, dt))
        if let h = logHandle {
            let line = "\(e + 1),\(avgLoss),\(dt)\n"
            h.write(Data(line.utf8))
        }
    }

    print("Testing...")
    let acc = testAccuracy(model: model, images: testImages, labels: testLabels, batchSize: config.batchSize)
    print(String(format: "Test Accuracy: %.2f%%", acc))

    print("Saving model...")
    do {
        try saveModel(model: model, filename: "mnist_cnn_model.bin")
    } catch {
        fputs("ERROR: Failed to save CNN model: \(error)\n", stderr)
        exit(1)
    }
}

main()
