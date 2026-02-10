import Foundation

/// ANSI color codes for terminal output.
/// Colors are only applied when the ANSI_COLORS environment variable is set to "1".
/// This provides backwards compatibility and allows users to opt-in to colored output.
public enum ANSIColors {
    // ANSI color codes
    public static let red = "\u{001B}[31m"
    public static let yellow = "\u{001B}[33m"
    public static let green = "\u{001B}[32m"
    public static let cyan = "\u{001B}[36m"
    public static let reset = "\u{001B}[0m"

    /// Check if ANSI colors are enabled via environment variable
    public static var isEnabled: Bool {
        return ProcessInfo.processInfo.environment["ANSI_COLORS"] == "1"
    }

    /// Apply color to text if ANSI colors are enabled
    /// - Parameters:
    ///   - text: The text to colorize
    ///   - color: The ANSI color code to apply
    /// - Returns: Colorized text if enabled, otherwise plain text
    public static func colorize(_ text: String, with color: String) -> String {
        return isEnabled ? "\(color)\(text)\(reset)" : text
    }
}

/// Provides convenient functions for printing colored messages to the terminal.
/// All functions check the ANSI_COLORS environment variable and only apply
/// colors when it is set to "1", ensuring backwards compatibility.
public struct ColoredPrint {

    /// Print an error message in red
    /// - Parameter message: The message to print
    public static func error(_ message: String) {
        let colored = ANSIColors.colorize(message, with: ANSIColors.red)
        print(colored)
    }

    /// Print a warning message in yellow
    /// - Parameter message: The message to print
    public static func warning(_ message: String) {
        let colored = ANSIColors.colorize(message, with: ANSIColors.yellow)
        print(colored)
    }

    /// Print a success message in green
    /// - Parameter message: The message to print
    public static func success(_ message: String) {
        let colored = ANSIColors.colorize(message, with: ANSIColors.green)
        print(colored)
    }

    /// Print a progress message in cyan
    /// - Parameter message: The message to print
    public static func progress(_ message: String) {
        let colored = ANSIColors.colorize(message, with: ANSIColors.cyan)
        print(colored)
    }

    /// Print an informational message (no color)
    /// - Parameter message: The message to print
    public static func info(_ message: String) {
        print(message)
    }
}
