// mnist_cnn.swift
// Minimal CNN for MNIST on CPU using explicit loops (no deps beyond Foundation).
// Expected files:
//   ./data/train-images.idx3-ubyte
//   ./data/train-labels.idx1-ubyte
//   ./data/t10k-images.idx3-ubyte
//   ./data/t10k-labels.idx1-ubyte
//
// Output:
//   - logs/training_loss_cnn.txt (epoch,loss,time)
//   - prints test accuracy

import Foundation
import Darwin

// =============================================================================
// MARK: - Simple Random Number Generator
// =============================================================================

/// A simple random number generator using xorshift algorithm.
/// This provides fast, deterministic pseudo-random number generation
/// for neural network weight initialization and data shuffling.
struct SimpleRng {
    private var state: UInt64

    // Explicit seed (if zero, use a fixed value).
    init(seed: UInt64) {
        self.state = seed == 0 ? 0x9e3779b97f4a7c15 : seed
    }

    // Reseed based on the current time.
    mutating func reseedFromTime() {
        let nanos = UInt64(Date().timeIntervalSince1970 * 1_000_000_000)
        state = nanos == 0 ? 0x9e3779b97f4a7c15 : nanos
    }

    // Basic xorshift to generate u32.
    mutating func nextUInt32() -> UInt32 {
        var x = state
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        state = x
        return UInt32(truncatingIfNeeded: x >> 32)
    }

    // Convert to [0, 1).
    mutating func nextFloat() -> Float {
        return Float(nextUInt32()) / Float(UInt32.max)
    }

    // Uniform sample in [low, high).
    mutating func uniform(_ low: Float, _ high: Float) -> Float {
        return low + (high - low) * nextFloat()
    }

    // Integer sample in [0, upper).
    mutating func nextInt(upper: Int) -> Int {
        return upper == 0 ? 0 : Int(nextUInt32()) % upper
    }

    /// Fisher-Yates shuffle for an array of Int.
    mutating func shuffle(_ array: inout [Int]) {
        let n = array.count
        if n > 1 {
            for i in stride(from: n - 1, through: 1, by: -1) {
                let j = nextInt(upper: i + 1)
                array.swapAt(i, j)
            }
        }
    }
}

// =============================================================================
// MARK: - MNIST Constants and Configuration
// =============================================================================

// MNIST constants (images are flat 28x28 in row-major order).
let imgH = 28
let imgW = 28
let numInputs = imgH * imgW // 784
let numClasses = 10
let trainSamples = 60_000
let testSamples  = 10_000

// CNN topology: 1x28x28 -> conv -> ReLU -> 2x2 maxpool -> FC(10).
let convOut = 8
let kernel = 3
let pad = 1
let pool = 2

let poolH = imgH / pool
let poolW = imgW / pool
let fcIn = convOut * poolH * poolW // 1568

// NOTE: SimpleRng has been extracted to Sources/MNISTCommon/SimpleRng.swift
// To use this file as a standalone script, you'll need the SimpleRng implementation.
// For package-based builds: import MNISTCommon

// =============================================================================
// MARK: - Command-Line Argument Parsing
// =============================================================================

/// Configuration parsed from command-line arguments
struct Config {
    var epochs: Int = 3
    var batchSize: Int = 32
    var learningRate: Float = 0.01
    var dataPath: String = "./data"
    var seed: UInt64 = 1

    /// Parses command-line arguments into configuration
    ///
    /// This is a simple hand-rolled parser. For production code,
    /// consider using Swift Argument Parser package.
    static func parse() -> Config {
        var config = Config()
        let args = CommandLine.arguments
        var i = 1

        while i < args.count {
            let arg = args[i]

            switch arg {
            case "--epochs", "-e":
                i += 1
                if i < args.count, let val = Int(args[i]) {
                    config.epochs = val
                }

            case "--batch", "-b":
                i += 1
                if i < args.count, let val = Int(args[i]) {
                    config.batchSize = val
                }

            case "--lr", "-l":
                i += 1
                if i < args.count, let val = Float(args[i]) {
                    config.learningRate = val
                }

            case "--data", "-d":
                i += 1
                if i < args.count {
                    config.dataPath = args[i]
                }

            case "--seed", "-s":
                i += 1
                if i < args.count, let val = UInt64(args[i]) {
                    config.seed = val
                }

            case "--help", "-h":
                printUsage()
                exit(0)

            default:
                print("Unknown argument: \(arg)")
                printUsage()
                exit(1)
            }

            i += 1
        }

        return config
    }
}

/// Prints usage information
func printUsage() {
    print("""
    MNIST CNN - Convolutional Neural Network for MNIST
    ===================================================

    USAGE:
      swift mnist_cnn.swift [OPTIONS]

    OPTIONS:
      --epochs, -e <n>      Number of training epochs (default: 3)
      --batch, -b <n>       Batch size (default: 32)
      --lr, -l <f>          Learning rate (default: 0.01)
      --data, -d <path>     Path to MNIST data directory (default: ./data)
      --seed, -s <n>        Random seed for reproducibility (default: 1)
      --help, -h            Show this help message

    EXAMPLES:
      swift mnist_cnn.swift --epochs 5
      swift mnist_cnn.swift -e 10 -b 64 -l 0.005
      swift mnist_cnn.swift --seed 42

    MODEL ARCHITECTURE:
      Input:  28×28 grayscale images (784 pixels)
      Conv:   3×3 kernel, 8 filters, ReLU activation
      Pool:   2×2 max pooling
      FC:     Fully connected layer to 10 classes
      Output: 10-class softmax (digits 0-9)

    EXPECTED DATA FILES:
      <data-path>/train-images.idx3-ubyte
      <data-path>/train-labels.idx1-ubyte
      <data-path>/t10k-images.idx3-ubyte
      <data-path>/t10k-labels.idx1-ubyte

    OUTPUT:
      logs/training_loss_cnn.txt - Training loss per epoch
    """)
}

// =============================================================================
// MARK: - Core Functions
// =============================================================================

// Stable softmax for a single row.
func softmaxRowInPlace(_ row: inout [Float]) {
    var maxv = row[0]
    for v in row.dropFirst() { if v > maxv { maxv = v } }

    var sum: Float = 0
    for i in 0..<row.count {
        row[i] = expf(row[i] - maxv)
        sum += row[i]
    }

    let inv = 1.0 / sum
    for i in 0..<row.count { row[i] *= inv }
}

// CNN parameters stored in flat arrays for cache-friendly loops.
struct Cnn {
    // Conv: 1 -> convOut, kernel 3x3, pad=1
    var convW: [Float] // [convOut * 3 * 3]
    var convB: [Float] // [convOut]
    // FC: fcIn -> 10
    var fcW: [Float]   // [fcIn * 10]
    var fcB: [Float]   // [10]
}

// Xavier/Glorot uniform init for stable activations.
func xavierInit(limit: Float, rng: inout SimpleRng, w: inout [Float]) {
    for i in 0..<w.count {
        w[i] = rng.uniform(-limit, limit)
    }
}

func initCnn(rng: inout SimpleRng) -> Cnn {
    // Xavier limits based on approximate fan-in/out.
    let fanIn: Float = Float(kernel * kernel)
    let fanOut: Float = Float(kernel * kernel * convOut)
    let convLimit = sqrtf(6.0 / (fanIn + fanOut))

    var convW = [Float](repeating: 0, count: convOut * kernel * kernel)
    let convB = [Float](repeating: 0, count: convOut)
    xavierInit(limit: convLimit, rng: &rng, w: &convW)

    let fcLimit = sqrtf(6.0 / (Float(fcIn) + Float(numClasses)))
    var fcW = [Float](repeating: 0, count: fcIn * numClasses)
    let fcB = [Float](repeating: 0, count: numClasses)
    xavierInit(limit: fcLimit, rng: &rng, w: &fcW)

    return Cnn(convW: convW, convB: convB, fcW: fcW, fcB: fcB)
}

// Forward conv + ReLU.
// input: [batch * 784], convOutAct: [batch * convOut * 28 * 28]
func convForwardRelu(model: Cnn, batch: Int, input: [Float], convOutAct: inout [Float]) {
    let spatial = imgH * imgW
    for b in 0..<batch {
        let inBase = b * numInputs
        let outBaseB = b * (convOut * spatial)

        for oc in 0..<convOut {
            let wBase = oc * (kernel * kernel)
            let bias = model.convB[oc]
            let outBase = outBaseB + oc * spatial

            // For each output pixel, accumulate a 3x3 window with zero-padding.
            for oy in 0..<imgH {
                for ox in 0..<imgW {
                    var sum = bias
                    for ky in 0..<kernel {
                        for kx in 0..<kernel {
                            let iy = oy + ky - pad
                            let ix = ox + kx - pad
                            if iy >= 0 && iy < imgH && ix >= 0 && ix < imgW {
                                let inIdx = inBase + iy * imgW + ix
                                let wIdx = wBase + ky * kernel + kx
                                sum += input[inIdx] * model.convW[wIdx]
                            }
                        }
                    }
                    let outIdx = outBase + oy * imgW + ox
                    convOutAct[outIdx] = (sum > 0) ? sum : 0 // ReLU activation
                }
            }
        }
    }
}

// MaxPool 2x2 stride 2. Stores argmax indices for backprop.
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
// x: [batch*fcIn], logits: [batch*10]
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

// Softmax + cross-entropy: returns summed loss and writes delta = (probs - onehot) * scale.
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

// FC backward: compute gradW, gradB and dX.
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

// MaxPool backward: scatter grads to argmax positions, then apply ReLU mask.
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

// Conv backward: gradW and gradB (no dInput since this is the first layer).
func convBackward(model: Cnn, batch: Int, input: [Float], convGrad: [Float], gradW: inout [Float], gradB: inout [Float]) {
    for i in 0..<gradW.count { gradW[i] = 0 }
    for i in 0..<gradB.count { gradB[i] = 0 }

    let spatial = imgH * imgW

    for b in 0..<batch {
        let inBase = b * numInputs
        let gBaseB = b * (convOut * spatial)

        for oc in 0..<convOut {
            let wBase = oc * (kernel * kernel)
            let gBase = gBaseB + oc * spatial

            for oy in 0..<imgH {
                for ox in 0..<imgW {
                    let g = convGrad[gBase + oy * imgW + ox]
                    gradB[oc] += g

                    for ky in 0..<kernel {
                        for kx in 0..<kernel {
                            let iy = oy + ky - pad
                            let ix = ox + kx - pad
                            if iy >= 0 && iy < imgH && ix >= 0 && ix < imgW {
                                let inIdx = inBase + iy * imgW + ix
                                let wIdx = wBase + ky * kernel + kx
                                gradW[wIdx] += g * input[inIdx]
                            }
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// MARK: - MNIST Data Loading
// =============================================================================

// MNIST IDX file readers (big-endian format).
func readMnistImages(path: String, count: Int) -> [Float] {
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url) else {
        print("Could not open file \(path)")
        exit(1)
    }

    return data.withUnsafeBytes { rawBuf in
        guard let base = rawBuf.bindMemory(to: UInt8.self).baseAddress else {
            return []
        }
        var offset = 0

        func readU32BE() -> UInt32 {
            let b0 = UInt32(base[offset]) << 24
            let b1 = UInt32(base[offset + 1]) << 16
            let b2 = UInt32(base[offset + 2]) << 8
            let b3 = UInt32(base[offset + 3])
            offset += 4
            return b0 | b1 | b2 | b3
        }

        _ = readU32BE()
        let total = Int(readU32BE())
        let rows = Int(readU32BE())
        let cols = Int(readU32BE())
        let imageSize = rows * cols
        let actualCount = min(count, total)

        var images = [Float](repeating: 0.0, count: actualCount * imageSize)
        for i in 0..<actualCount {
            let baseIndex = i * imageSize
            for j in 0..<imageSize {
                images[baseIndex + j] = Float(base[offset]) / 255.0
                offset += 1
            }
        }
        return images
    }
}

func readMnistLabels(path: String, count: Int) -> [UInt8] {
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url) else {
        print("Could not open file \(path)")
        exit(1)
    }

    return data.withUnsafeBytes { rawBuf in
        guard let base = rawBuf.bindMemory(to: UInt8.self).baseAddress else {
            return []
        }
        var offset = 0

        func readU32BE() -> UInt32 {
            let b0 = UInt32(base[offset]) << 24
            let b1 = UInt32(base[offset + 1]) << 16
            let b2 = UInt32(base[offset + 2]) << 8
            let b3 = UInt32(base[offset + 3])
            offset += 4
            return b0 | b1 | b2 | b3
        }

        _ = readU32BE()
        let total = Int(readU32BE())
        let actualCount = min(count, total)

        var labels = [UInt8](repeating: 0, count: actualCount)
        for i in 0..<actualCount {
            labels[i] = base[offset]
            offset += 1
        }
        return labels
    }
}

// =============================================================================
// MARK: - Model Evaluation
// =============================================================================

// Evaluate accuracy by running forward passes in batches.
func testAccuracy(model: Cnn, images: [Float], labels: [UInt8], batchSize: Int) -> Float {
    let n = labels.count
    var correct = 0

    var batchInputs = [Float](repeating: 0, count: batchSize * numInputs)
    var convAct = [Float](repeating: 0, count: batchSize * convOut * imgH * imgW)
    var poolOut = [Float](repeating: 0, count: batchSize * fcIn)
    var poolIdx = [UInt8](repeating: 0, count: batchSize * convOut * poolH * poolW)
    var logits = [Float](repeating: 0, count: batchSize * numClasses)

    var start = 0
    while start < n {
        let bsz = min(batchSize, n - start)
        let len = bsz * numInputs
        let srcStart = start * numInputs
        for i in 0..<len {
            batchInputs[i] = images[srcStart + i]
        }

        convForwardRelu(model: model, batch: bsz, input: batchInputs, convOutAct: &convAct)
        maxPoolForward(batch: bsz, convAct: convAct, poolOut: &poolOut, poolIdx: &poolIdx)
        fcForward(model: model, batch: bsz, x: poolOut, logits: &logits)

        for b in 0..<bsz {
            let base = b * numClasses
            var best = logits[base]
            var arg = 0
            for j in 1..<numClasses {
                let v = logits[base + j]
                if v > best { best = v; arg = j }
            }
            if UInt8(arg) == labels[start + b] { correct += 1 }
        }

        start += bsz
    }

    return 100.0 * Float(correct) / Float(n)
}

func main() {
    // Parse command-line arguments
    let config = Config.parse()

    print("Loading MNIST...")
    let trainImages = readMnistImages(path: "\(config.dataPath)/train-images.idx3-ubyte", count: trainSamples)
    let trainLabels = readMnistLabels(path: "\(config.dataPath)/train-labels.idx1-ubyte", count: trainSamples)
    let testImages  = readMnistImages(path: "\(config.dataPath)/t10k-images.idx3-ubyte", count: testSamples)
    let testLabels  = readMnistLabels(path: "\(config.dataPath)/t10k-labels.idx1-ubyte", count: testSamples)

    print("Train: \(trainLabels.count) | Test: \(testLabels.count)")

    var rng = SimpleRng(seed: config.seed)
    var model = initCnn(rng: &rng)

    try? FileManager.default.createDirectory(atPath: "./logs", withIntermediateDirectories: true)
    FileManager.default.createFile(atPath: "./logs/training_loss_cnn.txt", contents: nil)
    let logHandle = try? FileHandle(forWritingTo: URL(fileURLWithPath: "./logs/training_loss_cnn.txt"))
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

    print("Training CNN: epochs=\(config.epochs) batch=\(config.batchSize) lr=\(config.learningRate)")

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

            start += bsz
        }

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
}

main()
