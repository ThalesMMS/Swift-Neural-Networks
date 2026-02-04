import Foundation
import XCTest

#if canImport(Metal)
import Metal
#endif

enum MLXTestSupport {
    static func skipReason() -> String? {
        if envFlagEnabled("SKIP_MLX_TESTS") {
            return "SKIP_MLX_TESTS is set"
        }
        let wantsRun = envFlagEnabled("RUN_MLX_TESTS")
#if canImport(Metal)
        if MTLCreateSystemDefaultDevice() == nil {
            return "Metal device unavailable"
        }
#else
        return "Metal framework unavailable"
#endif
        if envFlagEnabled("CI") && !wantsRun {
            return "CI environment without RUN_MLX_TESTS=1"
        }
        return nil
    }

    private static func envFlagEnabled(_ key: String) -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[key] else {
            return false
        }
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return value == "1" || value == "true" || value == "yes"
    }
}

class MLXTestCase: XCTestCase {
    override func setUpWithError() throws {
        try super.setUpWithError()
        if let reason = MLXTestSupport.skipReason() {
            throw XCTSkip("Skipping MLX tests: \(reason)")
        }
    }
}
