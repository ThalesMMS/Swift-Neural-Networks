#!/usr/bin/env swift

// verify_cnn_saveload.swift
// Verification script for CNN save/load round-trip
// Tests that saved models can be loaded and produce identical predictions

import Foundation
import Darwin

// =============================================================================
// MARK: - Colored Print Utility
// =============================================================================

/// Simple colored console output utility
struct ColoredPrint {
    static func error(_ message: String) {
        if ProcessInfo.processInfo.environment["ANSI_COLORS"] == "1" {
            print("\u{001B}[31m\(message)\u{001B}[0m")  // Red
        } else {
            print("❌ ERROR: \(message)")
        }
    }

    static func warning(_ message: String) {
        if ProcessInfo.processInfo.environment["ANSI_COLORS"] == "1" {
            print("\u{001B}[33m\(message)\u{001B}[0m")  // Yellow
        } else {
            print("⚠️  WARNING: \(message)")
        }
    }

    static func success(_ message: String) {
        if ProcessInfo.processInfo.environment["ANSI_COLORS"] == "1" {
            print("\u{001B}[32m\(message)\u{001B}[0m")  // Green
        } else {
            print("✅ \(message)")
        }
    }

    static func info(_ message: String) {
        if ProcessInfo.processInfo.environment["ANSI_COLORS"] == "1" {
            print("\u{001B}[36m\(message)\u{001B}[0m")  // Cyan
        } else {
            print("ℹ️  \(message)")
        }
    }
}

// =============================================================================
// MARK: - Simple Random Number Generator
// =============================================================================

struct SimpleRng {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 0x9e3779b97f4a7c15 : seed
    }

    mutating func nextUInt32() -> UInt32 {
        var x = state
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        state = x
        return UInt32(truncatingIfNeeded: x >> 32)
    }

    mutating func nextFloat() -> Float {
        return Float(nextUInt32()) / Float(UInt32.max)
    }

    mutating func uniform(_ low: Float, _ high: Float) -> Float {
        return low + (high - low) * nextFloat()
    }

    mutating func nextInt(upper: Int) -> Int {
        return upper == 0 ? 0 : Int(nextUInt32()) % upper
    }

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

let imgH = 28
let imgW = 28
let numInputs = imgH * imgW
let numClasses = 10
let trainSamples = 60_000
let testSamples  = 10_000

let convOut = 8
let kernel = 3
let pad = 1
let pool = 2

let poolH = imgH / pool
let poolW = imgW / pool
let fcIn = convOut * poolH * poolW

// =============================================================================
// MARK: - Core Functions
// =============================================================================

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

struct Cnn {
    var convW: [Float]
    var convB: [Float]
    var fcW: [Float]
    var fcB: [Float]
}

func xavierInit(limit: Float, rng: inout SimpleRng, w: inout [Float]) {
    for i in 0..<w.count {
        w[i] = rng.uniform(-limit, limit)
    }
}

func initCnn(rng: inout SimpleRng) -> Cnn {
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

func convForwardRelu(model: Cnn, batch: Int, input: [Float], convOutAct: inout [Float]) {
    let spatial = imgH * imgW
    for b in 0..<batch {
        let inBase = b * numInputs
        let outBaseB = b * (convOut * spatial)

        for oc in 0..<convOut {
            let wBase = oc * (kernel * kernel)
            let bias = model.convB[oc]
            let outBase = outBaseB + oc * spatial

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
                    convOutAct[outIdx] = (sum > 0) ? sum : 0
                }
            }
        }
    }
}

func maxPoolForward(batch: Int, convAct: [Float], poolOut: inout [Float], poolIdx: inout [UInt8]) {
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

func softmaxXentBackward(probsInPlace: inout [Float], labels: [UInt8], batch: Int, delta: inout [Float], scale: Float) -> Float {
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

func fcBackward(model: Cnn, batch: Int, x: [Float], delta: [Float], gradW: inout [Float], gradB: inout [Float], dX: inout [Float]) {
    for i in 0..<gradW.count { gradW[i] = 0 }
    for i in 0..<gradB.count { gradB[i] = 0 }

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
                    let a = Int(poolIdx[pI])
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

    for i in 0..<used {
        if convAct[i] <= 0 { convGrad[i] = 0 }
    }
}

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

func saveModel(model: Cnn, filename: String) {
    FileManager.default.createFile(atPath: filename, contents: nil)
    guard let handle = try? FileHandle(forWritingTo: URL(fileURLWithPath: filename)) else {
        ColoredPrint.error("Failed to open '\(filename)' for writing")
        ColoredPrint.info("→ Check if you have write permissions in the current directory")
        ColoredPrint.info("→ Ensure the path is valid and the disk is not full")
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

    writeInt32(Int32(convOut))
    writeInt32(Int32(kernel))
    writeInt32(Int32(fcIn))
    writeInt32(Int32(numClasses))

    for w in model.convW { writeDouble(Double(w)) }
    for b in model.convB { writeDouble(Double(b)) }
    for w in model.fcW { writeDouble(Double(w)) }
    for b in model.fcB { writeDouble(Double(b)) }
}

func loadModel(filename: String) -> Cnn? {
    guard let handle = try? FileHandle(forReadingFrom: URL(fileURLWithPath: filename)) else {
        ColoredPrint.error("Failed to open '\(filename)' for reading")
        ColoredPrint.info("→ Check if the file exists in the current directory")
        ColoredPrint.info("→ Ensure you have read permissions for this file")
        return nil
    }
    defer { try? handle.close() }

    func readInt32() -> Int32? {
        guard let data = try? handle.read(upToCount: MemoryLayout<Int32>.size),
              data.count == MemoryLayout<Int32>.size else {
            return nil
        }
        return data.withUnsafeBytes { $0.load(as: Int32.self) }
    }

    func readDouble() -> Double? {
        guard let data = try? handle.read(upToCount: MemoryLayout<Double>.size),
              data.count == MemoryLayout<Double>.size else {
            return nil
        }
        return data.withUnsafeBytes { $0.load(as: Double.self) }
    }

    guard let convOutRead = readInt32(),
          let kernelRead = readInt32(),
          let fcInRead = readInt32(),
          let numClassesRead = readInt32() else {
        ColoredPrint.error("Failed to read model header from '\(filename)'")
        ColoredPrint.info("→ The file may be corrupted or truncated")
        ColoredPrint.info("→ Ensure the file was saved completely")
        ColoredPrint.info("→ Try re-saving the model")
        return nil
    }

    if convOutRead != Int32(convOut) || kernelRead != Int32(kernel) ||
       fcInRead != Int32(fcIn) || numClassesRead != Int32(numClasses) {
        ColoredPrint.error("Model architecture mismatch")
        ColoredPrint.info("→ Expected: convOut=\(convOut), kernel=\(kernel), fcIn=\(fcIn), classes=\(numClasses)")
        ColoredPrint.info("→ Found in file: convOut=\(convOutRead), kernel=\(kernelRead), fcIn=\(fcInRead), classes=\(numClassesRead)")
        ColoredPrint.info("→ Ensure you're loading a CNN model file, not a different architecture")
        return nil
    }

    var convW = [Float](repeating: 0, count: convOut * kernel * kernel)
    for i in 0..<convW.count {
        guard let val = readDouble() else { return nil }
        convW[i] = Float(val)
    }

    var convB = [Float](repeating: 0, count: convOut)
    for i in 0..<convB.count {
        guard let val = readDouble() else { return nil }
        convB[i] = Float(val)
    }

    var fcW = [Float](repeating: 0, count: fcIn * numClasses)
    for i in 0..<fcW.count {
        guard let val = readDouble() else { return nil }
        fcW[i] = Float(val)
    }

    var fcB = [Float](repeating: 0, count: numClasses)
    for i in 0..<fcB.count {
        guard let val = readDouble() else { return nil }
        fcB[i] = Float(val)
    }

    return Cnn(convW: convW, convB: convB, fcW: fcW, fcB: fcB)
}

// =============================================================================
// MARK: - MNIST Data Loading
// =============================================================================

func readMnistImages(path: String, count: Int) -> [Float] {
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url) else {
        ColoredPrint.error("Failed to load MNIST images from '\(path)'")
        ColoredPrint.info("→ Download MNIST data: run './download_mnist.sh' from project root")
        ColoredPrint.info("→ Or ensure MNIST data exists in ./data/")
        ColoredPrint.info("→ Expected file format: IDX3-ubyte (MNIST image format)")
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
        ColoredPrint.error("Failed to load MNIST labels from '\(path)'")
        ColoredPrint.info("→ Download MNIST data: run './download_mnist.sh' from project root")
        ColoredPrint.info("→ Or ensure MNIST data exists in ./data/")
        ColoredPrint.info("→ Expected file format: IDX1-ubyte (MNIST label format)")
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
// MARK: - Prediction Functions
// =============================================================================

func predict(model: Cnn, images: [Float], count: Int) -> [Int] {
    var predictions = [Int]()

    var convAct = [Float](repeating: 0, count: convOut * imgH * imgW)
    var poolOut = [Float](repeating: 0, count: fcIn)
    var poolIdx = [UInt8](repeating: 0, count: convOut * poolH * poolW)
    var logits = [Float](repeating: 0, count: numClasses)

    for i in 0..<count {
        let imgStart = i * numInputs
        let imgEnd = imgStart + numInputs
        let input = Array(images[imgStart..<imgEnd])

        convForwardRelu(model: model, batch: 1, input: input, convOutAct: &convAct)
        maxPoolForward(batch: 1, convAct: convAct, poolOut: &poolOut, poolIdx: &poolIdx)
        fcForward(model: model, batch: 1, x: poolOut, logits: &logits)

        var best = logits[0]
        var arg = 0
        for j in 1..<numClasses {
            if logits[j] > best {
                best = logits[j]
                arg = j
            }
        }
        predictions.append(arg)
    }

    return predictions
}

// =============================================================================
// MARK: - Main Verification
// =============================================================================

func main() {
    print("=== CNN Save/Load Round-Trip Verification ===\n")

    let dataPath = "./data"
    let modelFile = "mnist_cnn_model_test.bin"
    let batchSize = 32
    let learningRate: Float = 0.01
    let epochs = 1
    let trainSubset = 1000  // Train on subset for faster verification
    let testCount = 100  // Test on first 100 images

    // Load MNIST data
    print("Loading MNIST data...")
    let trainImages = readMnistImages(path: "\(dataPath)/train-images.idx3-ubyte", count: trainSubset)
    let trainLabels = readMnistLabels(path: "\(dataPath)/train-labels.idx1-ubyte", count: trainSubset)
    let testImages = readMnistImages(path: "\(dataPath)/t10k-images.idx3-ubyte", count: testSamples)
    let testLabels = readMnistLabels(path: "\(dataPath)/t10k-labels.idx1-ubyte", count: testSamples)
    print("Loaded \(trainLabels.count) training samples, \(testLabels.count) test samples\n")

    // Initialize and train model
    print("Initializing CNN model...")
    var rng = SimpleRng(seed: 42)
    var model = initCnn(rng: &rng)

    print("Training for \(epochs) epoch...\n")

    var batchInputs = [Float](repeating: 0, count: batchSize * numInputs)
    var batchLabels = [UInt8](repeating: 0, count: batchSize)
    var convAct = [Float](repeating: 0, count: batchSize * convOut * imgH * imgW)
    var poolOut = [Float](repeating: 0, count: batchSize * fcIn)
    var poolIdx = [UInt8](repeating: 0, count: batchSize * convOut * poolH * poolW)
    var logits = [Float](repeating: 0, count: batchSize * numClasses)
    var delta = [Float](repeating: 0, count: batchSize * numClasses)
    var dPool = [Float](repeating: 0, count: batchSize * fcIn)
    var dConv = [Float](repeating: 0, count: batchSize * convOut * imgH * imgW)
    var gradFcW = [Float](repeating: 0, count: fcIn * numClasses)
    var gradFcB = [Float](repeating: 0, count: numClasses)
    var gradConvW = [Float](repeating: 0, count: convOut * kernel * kernel)
    var gradConvB = [Float](repeating: 0, count: convOut)

    var indices = Array(0..<trainLabels.count)

    for e in 0..<epochs {
        rng.shuffle(&indices)
        var totalLoss: Float = 0
        var start = 0

        while start < indices.count {
            let bsz = min(batchSize, indices.count - start)
            let scale = 1.0 / Float(bsz)

            for i in 0..<bsz {
                let srcIndex = indices[start + i]
                let srcBase = srcIndex * numInputs
                let dstBase = i * numInputs
                for j in 0..<numInputs {
                    batchInputs[dstBase + j] = trainImages[srcBase + j]
                }
                batchLabels[i] = trainLabels[srcIndex]
            }

            convForwardRelu(model: model, batch: bsz, input: batchInputs, convOutAct: &convAct)
            maxPoolForward(batch: bsz, convAct: convAct, poolOut: &poolOut, poolIdx: &poolIdx)
            fcForward(model: model, batch: bsz, x: poolOut, logits: &logits)
            totalLoss += softmaxXentBackward(probsInPlace: &logits, labels: batchLabels, batch: bsz, delta: &delta, scale: scale)

            fcBackward(model: model, batch: bsz, x: poolOut, delta: delta, gradW: &gradFcW, gradB: &gradFcB, dX: &dPool)
            maxPoolBackwardRelu(batch: bsz, convAct: convAct, poolGrad: dPool, poolIdx: poolIdx, convGrad: &dConv)
            convBackward(model: model, batch: bsz, input: batchInputs, convGrad: dConv, gradW: &gradConvW, gradB: &gradConvB)

            for i in 0..<model.fcW.count { model.fcW[i] -= learningRate * gradFcW[i] }
            for i in 0..<model.fcB.count { model.fcB[i] -= learningRate * gradFcB[i] }
            for i in 0..<model.convW.count { model.convW[i] -= learningRate * gradConvW[i] }
            for i in 0..<model.convB.count { model.convB[i] -= learningRate * gradConvB[i] }

            start += bsz
        }

        let avgLoss = totalLoss / Float(trainLabels.count)
        print(String(format: "Epoch %d completed | loss=%.6f", e + 1, avgLoss))
    }

    // Get predictions from original model
    print("\nGetting predictions from original trained model...")
    let originalPredictions = predict(model: model, images: testImages, count: testCount)

    // Save the model
    print("Saving model to \(modelFile)...")
    saveModel(model: model, filename: modelFile)

    // Load the model
    print("Loading model from \(modelFile)...")
    guard let loadedModel = loadModel(filename: modelFile) else {
        ColoredPrint.error("Verification FAILED: Unable to load saved model")
        ColoredPrint.info("→ The save operation may have failed silently")
        ColoredPrint.info("→ Check disk space and file permissions")
        exit(1)
    }

    // Get predictions from loaded model
    print("Getting predictions from loaded model...\n")
    let loadedPredictions = predict(model: loadedModel, images: testImages, count: testCount)

    // Compare predictions
    print("=== Verification Results ===")
    var matches = 0
    var mismatches = 0

    for i in 0..<testCount {
        if originalPredictions[i] == loadedPredictions[i] {
            matches += 1
        } else {
            mismatches += 1
            print("Mismatch at index \(i): original=\(originalPredictions[i]), loaded=\(loadedPredictions[i])")
        }
    }

    print("\nResults:")
    print("  Total samples tested: \(testCount)")
    print("  Matching predictions: \(matches)")
    print("  Mismatched predictions: \(mismatches)")

    if mismatches == 0 {
        ColoredPrint.success("\nSUCCESS: All predictions match! Save/load round-trip works correctly.")

        // Calculate accuracy on test set
        var correct = 0
        for i in 0..<testCount {
            if loadedPredictions[i] == Int(testLabels[i]) {
                correct += 1
            }
        }
        let accuracy = 100.0 * Float(correct) / Float(testCount)
        print(String(format: "  Model accuracy on test samples: %.2f%%", accuracy))

        // Clean up test file
        try? FileManager.default.removeItem(atPath: modelFile)
        print("\nTest model file cleaned up.")
    } else {
        ColoredPrint.error("\nVerification FAILED: Predictions do not match after save/load")
        ColoredPrint.info("→ This indicates the save/load functions have a bug")
        ColoredPrint.info("→ Model weights may not be preserved correctly")
        ColoredPrint.info("→ Check saveModel() and loadModel() implementations")
        exit(1)
    }
}

main()
