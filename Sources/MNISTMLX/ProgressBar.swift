// ============================================================================
// ProgressBar.swift - Terminal Progress Bar for Training Visualization
// ============================================================================
//
// This utility class provides a visual progress bar for training epochs,
// showing batch completion, current loss, and estimated time remaining (ETA).
//
// FEATURES:
//   - In-place terminal updates using ANSI escape codes
//   - Real-time loss display
//   - ETA calculation based on average batch time
//   - Clean formatting with percentage and batch counters
//
// USAGE:
//   let progress = ProgressBar(totalBatches: 468)
//   progress.start()
//   for batch in 0..<totalBatches {
//       // ... training step ...
//       progress.update(batch: batch + 1, loss: currentLoss)
//   }
//   progress.finish()
//
// ============================================================================

import Foundation

/// A terminal progress bar for visualizing training progress
///
/// Displays progress in the format:
/// [=====>    ] 50% (234/468) | Loss: 0.234567 | ETA: 2m 15s
class ProgressBar {
    // -------------------------------------------------------------------------
    // MARK: - Properties
    // -------------------------------------------------------------------------

    /// Total number of batches to process
    private let totalBatches: Int

    /// Time when training started (for ETA calculation)
    private var startTime: Date?

    /// Width of the progress bar in characters
    private let barWidth: Int = 30

    // -------------------------------------------------------------------------
    // MARK: - ANSI Escape Codes
    // -------------------------------------------------------------------------

    /// Carriage return - moves cursor to start of line
    private let carriageReturn = "\r"

    /// Clear entire line
    private let clearLine = "\u{001B}[2K"

    // -------------------------------------------------------------------------
    // MARK: - Initialization
    // -------------------------------------------------------------------------

    /// Creates a new progress bar
    ///
    /// - Parameter totalBatches: Total number of batches to process
    init(totalBatches: Int) {
        self.totalBatches = totalBatches
    }

    // -------------------------------------------------------------------------
    // MARK: - Public Methods
    // -------------------------------------------------------------------------

    /// Starts the progress bar timer
    func start() {
        startTime = Date()
    }

    /// Updates the progress bar display
    ///
    /// - Parameters:
    ///   - batch: Current batch number (1-indexed)
    ///   - loss: Current loss value to display
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

        // Print without newline (using Swift's print with terminator)
        print(progressLine, terminator: "")
        fflush(stdout)
    }

    /// Finishes the progress bar and prints a newline
    func finish() {
        print() // Move to next line after progress bar
    }

    // -------------------------------------------------------------------------
    // MARK: - Private Helper Methods
    // -------------------------------------------------------------------------

    /// Formats time in seconds to a human-readable string
    ///
    /// - Parameter seconds: Time in seconds
    /// - Returns: Formatted string like "2m 15s" or "45s"
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
