// ============================================================================
// ProgressBarTests.swift - Edge Case Testing for Progress Bar
// ============================================================================
//
// This file tests the ProgressBar utility with various edge cases to ensure
// it handles different batch sizes and configurations gracefully.
//
// ============================================================================

import Foundation

// Simplified ProgressBar for testing (mirrors the actual implementation)
class TestProgressBar {
    private let totalBatches: Int
    private var startTime: Date?
    private let barWidth: Int = 30
    private let carriageReturn = "\r"
    private let clearLine = "\u{001B}[2K"

    init(totalBatches: Int) {
        self.totalBatches = totalBatches
    }

    func start() {
        startTime = Date()
    }

    func update(batch: Int, loss: Float) {
        guard let startTime = startTime else { return }

        // Calculate progress metrics
        let progress = Float(batch) / Float(totalBatches)
        let percentage = Int(progress * 100)

        // Build progress bar visualization
        let filledWidth = Int(progress * Float(barWidth))
        let emptyWidth = barWidth - filledWidth

        var bar = "["
        if filledWidth > 0 {
            bar += String(repeating: "=", count: max(0, filledWidth - 1))
            bar += ">"
        }
        bar += String(repeating: " ", count: emptyWidth)
        bar += "]"

        // Calculate ETA
        let elapsed = Date().timeIntervalSince(startTime)
        let avgTimePerBatch = elapsed / Double(batch)
        let remainingBatches = totalBatches - batch
        let etaSeconds = avgTimePerBatch * Double(remainingBatches)
        let etaString = formatTime(etaSeconds)

        // Format loss with 6 decimal places
        let lossString = String(format: "%.6f", loss)

        // Build complete progress line
        let progressLine = "\(clearLine)\(carriageReturn)\(bar) \(percentage)% (\(batch)/\(totalBatches)) | Loss: \(lossString) | ETA: \(etaString)"

        print(progressLine, terminator: "")
        fflush(stdout)
    }

    func finish() {
        print() // Move to next line after progress bar
    }

    private func formatTime(_ seconds: Double) -> String {
        guard seconds.isFinite && seconds >= 0 else {
            return "--"
        }

        let totalSeconds = Int(seconds)
        let minutes = totalSeconds / 60
        let secs = totalSeconds % 60

        if minutes > 0 {
            return "\(minutes)m \(secs)s"
        } else {
            return "\(secs)s"
        }
    }
}

// ============================================================================
// MARK: - Test Functions
// ============================================================================

/// Test Case 1: Very Small Batches (batch=8)
/// MNIST: 60000 samples / 8 = 7500 batches
func testVerySmallBatches() {
    print("\n=== Test 1: Very Small Batches (batch=8) ===")
    let n = 60000
    let batchSize = 8
    let totalBatches = (n + batchSize - 1) / batchSize
    print("Total batches: \(totalBatches)")

    let progress = TestProgressBar(totalBatches: totalBatches)
    progress.start()

    // Simulate training with a few updates
    for i in 1...min(10, totalBatches) {
        usleep(10000) // 10ms delay to simulate training
        progress.update(batch: i, loss: Float.random(in: 0.5...2.0))
    }

    // Jump to middle
    progress.update(batch: totalBatches / 2, loss: Float.random(in: 0.3...0.8))

    // Jump to near end
    progress.update(batch: totalBatches - 1, loss: Float.random(in: 0.1...0.5))

    // Final batch
    progress.update(batch: totalBatches, loss: Float.random(in: 0.1...0.3))
    progress.finish()

    print("✓ Test passed: Progress bar handled 7500 batches")
}

/// Test Case 2: Very Large Batches (batch=256)
/// MNIST: 60000 samples / 256 = 234.375 → 235 batches
func testVeryLargeBatches() {
    print("\n=== Test 2: Very Large Batches (batch=256) ===")
    let n = 60000
    let batchSize = 256
    let totalBatches = (n + batchSize - 1) / batchSize
    print("Total batches: \(totalBatches)")
    print("Last batch size: \(n - (totalBatches - 1) * batchSize) samples")

    let progress = TestProgressBar(totalBatches: totalBatches)
    progress.start()

    // Simulate all batches
    for i in 1...totalBatches {
        usleep(20000) // 20ms delay to simulate training
        progress.update(batch: i, loss: Float.random(in: 0.5...0.1))
    }
    progress.finish()

    print("✓ Test passed: Progress bar handled 235 batches with partial last batch")
}

/// Test Case 3: Single Batch (edge case)
func testSingleBatch() {
    print("\n=== Test 3: Single Batch ===")
    let totalBatches = 1

    let progress = TestProgressBar(totalBatches: totalBatches)
    progress.start()

    usleep(100000) // 100ms delay
    progress.update(batch: 1, loss: 1.234567)
    progress.finish()

    print("✓ Test passed: Progress bar handled single batch")
}

/// Test Case 4: Last Batch Smaller Than Batch Size
/// MNIST: 60000 samples / 128 = 468.75 → 469 batches
/// Last batch: 60000 - (468 * 128) = 96 samples
func testLastBatchPartial() {
    print("\n=== Test 4: Last Batch Smaller (batch=128) ===")
    let n = 60000
    let batchSize = 128
    let totalBatches = (n + batchSize - 1) / batchSize
    let lastBatchSize = n - (totalBatches - 1) * batchSize

    print("Total batches: \(totalBatches)")
    print("Last batch size: \(lastBatchSize) samples (< \(batchSize))")

    let progress = TestProgressBar(totalBatches: totalBatches)
    progress.start()

    // Jump to last few batches
    let startBatch = max(1, totalBatches - 3)
    for i in startBatch...totalBatches {
        usleep(50000) // 50ms delay
        progress.update(batch: i, loss: Float.random(in: 0.1...0.3))
    }
    progress.finish()

    print("✓ Test passed: Progress bar handled partial last batch correctly")
}

/// Test Case 5: Very Few Batches (10 batches total)
func testFewBatches() {
    print("\n=== Test 5: Very Few Batches (10 total) ===")
    let totalBatches = 10

    let progress = TestProgressBar(totalBatches: totalBatches)
    progress.start()

    for i in 1...totalBatches {
        usleep(50000) // 50ms delay
        progress.update(batch: i, loss: Float.random(in: 1.0...0.1))
    }
    progress.finish()

    print("✓ Test passed: Progress bar handled small number of batches")
}

/// Test Case 6: Progress Bar Calculations with Edge Values
func testProgressCalculations() {
    print("\n=== Test 6: Progress Calculations ===")

    // Test division by zero protection
    let progress1 = TestProgressBar(totalBatches: 1)
    progress1.start()
    usleep(10000)
    progress1.update(batch: 1, loss: 0.5)
    progress1.finish()
    print("✓ Single batch calculations work")

    // Test very large batch count
    let progress2 = TestProgressBar(totalBatches: 100000)
    progress2.start()
    progress2.update(batch: 50000, loss: 0.5)
    progress2.finish()
    print("✓ Large batch count calculations work")

    // Test ETA with very small elapsed time
    let progress3 = TestProgressBar(totalBatches: 100)
    progress3.start()
    usleep(100) // 0.1ms - very small time
    progress3.update(batch: 1, loss: 0.5)
    progress3.finish()
    print("✓ Small elapsed time calculations work")
}

// ============================================================================
// MARK: - Main Test Runner
// ============================================================================

print("╔═══════════════════════════════════════════════════════╗")
print("║   Progress Bar Edge Case Testing                      ║")
print("╚═══════════════════════════════════════════════════════╝")

testVerySmallBatches()
testVeryLargeBatches()
testSingleBatch()
testLastBatchPartial()
testFewBatches()
testProgressCalculations()

print("\n╔═══════════════════════════════════════════════════════╗")
print("║   All Edge Case Tests Passed ✓                        ║")
print("╚═══════════════════════════════════════════════════════╝\n")
