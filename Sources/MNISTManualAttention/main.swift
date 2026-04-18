import Foundation
import Accelerate
import MNISTCommon

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Entry point that trains an attention-based model on the MNIST dataset, evaluates it, and saves the trained model.
/// 
/// Loads training and test images/labels from the configured data path, initializes randomness and the model, trains for the configured number of epochs while logging per-epoch loss, duration, and test accuracy, evaluates final test accuracy, writes a training log (if writable), saves the model to "mnist_attention_model.bin", and prints load/train/total timing summaries.
func main() {
    // Parse command-line configuration.
    let config = Config.parse()

    let programStart = Date()

    print("Loading MNIST...")
    let loadStart = Date()
    let trainImages = readMnistImages(path: "\(config.dataPath)/train-images.idx3-ubyte", count: trainSamples)
    let trainLabels = readMnistLabels(path: "\(config.dataPath)/train-labels.idx1-ubyte", count: trainSamples)
    let testImages  = readMnistImages(path: "\(config.dataPath)/t10k-images.idx3-ubyte", count: testSamples)
    let testLabels  = readMnistLabels(path: "\(config.dataPath)/t10k-labels.idx1-ubyte", count: testSamples)
    let loadTime = Date().timeIntervalSince(loadStart)
    print(String(format: "Data loading time: %.2f seconds", loadTime))

    print("Config: patch=\(patch)x\(patch) tokens=\(seqLen) d=\(dModel) ff=\(ffDim) batch=\(config.batchSize) epochs=\(config.epochs) lr=\(config.learningRate) seed=\(config.seed)")

    var rng = SimpleRng(seed: config.seed)
    if config.seed == 0 {
        rng.reseedFromTime()
    }
    var model = initModel(rng: &rng)

    // Training log file.
    try? FileManager.default.createDirectory(atPath: "./logs", withIntermediateDirectories: true)
    FileManager.default.createFile(atPath: "./logs/training_loss_attention_mnist.txt", contents: nil)
    let logHandle = try? FileHandle(forWritingTo: URL(fileURLWithPath: "./logs/training_loss_attention_mnist.txt"))
    defer { try? logHandle?.close() }

    let trainN = min(trainLabels.count, trainSamples)
    var indices = Array(0..<trainN)

    print("Training...")
    let trainStart = Date()
    for e in 0..<config.epochs {
        let t0 = Date()
        let avgLoss = trainEpoch(model: &model, images: trainImages, labels: trainLabels, indices: &indices, rng: &rng, config: config)
        let dt = Float(Date().timeIntervalSince(t0))

        let acc = testAccuracy(model: model, images: testImages, labels: testLabels, config: config)
        print(String(format: "Epoch %d | loss=%.6f | time=%.3fs | test_acc=%.2f%%", e + 1, avgLoss, dt, acc))

        if let h = logHandle {
            let line = "\(e + 1),\(avgLoss),\(dt),\(acc)\n"
            h.write(Data(line.utf8))
        }
    }
    let trainTime = Date().timeIntervalSince(trainStart)

    let finalAcc = testAccuracy(model: model, images: testImages, labels: testLabels, config: config)
    print(String(format: "Final Test Accuracy: %.2f%%", finalAcc))

    print("Saving model...")
    do {
        try saveModel(model: model, filename: "mnist_attention_model.bin")
    } catch {
        fputs("""
        ERROR: Failed to save attention model: \(error)
        Try running the MNISTManualAttention target again:
          swift run MNISTManualAttention
        Default output filename: mnist_attention_model.bin

        """, stderr)
        exit(1)
    }

    let totalTime = Date().timeIntervalSince(programStart)
    print("\n=== Summary ===")
    print(String(format: "Load: %.2fs", loadTime))
    print(String(format: "Train: %.2fs", trainTime))
    print(String(format: "Total: %.2fs", totalTime))
    print("=============")
}

main()
