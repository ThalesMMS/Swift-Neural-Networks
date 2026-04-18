import Foundation

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
let packagePath = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
process.arguments = ["swift", "run", "--package-path", packagePath, "MNISTManualAttention"] + Array(CommandLine.arguments.dropFirst())

do {
    try process.run()
    process.waitUntilExit()
    exit(process.terminationStatus)
} catch {
    fputs("Failed to launch SwiftPM target MNISTManualAttention: \(error)\n", stderr)
    exit(1)
}
