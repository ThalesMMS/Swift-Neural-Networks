import XCTest
@testable import MNISTCommon
@testable import MNISTData

/// Basic smoke tests to verify the Swift Neural Networks package builds and imports correctly.
/// These tests ensure that the core modules can be imported and basic functionality works.
final class SmokeTests: XCTestCase {

    // MARK: - Module Import Tests

    /// Test that MNISTCommon module imports successfully and SimpleRng works.
    func testMNISTCommonImport() throws {
        // Verify we can instantiate SimpleRng from MNISTCommon
        var rng = SimpleRng(seed: 42)

        // Verify basic RNG functionality
        let randomValue = rng.nextFloat()
        XCTAssertGreaterThanOrEqual(randomValue, 0.0)
        XCTAssertLessThan(randomValue, 1.0)

        // Verify uniform distribution
        let uniformValue = rng.uniform(10.0, 20.0)
        XCTAssertGreaterThanOrEqual(uniformValue, 10.0)
        XCTAssertLessThan(uniformValue, 20.0)
    }

    /// Test that MNISTData module imports successfully.
    func testMNISTDataImport() throws {
        // This test verifies the MNISTData module can be imported.
        // We don't test actual data loading here since that requires MNIST files.
        // The fact that this test compiles and runs proves the module is accessible.
        XCTAssertTrue(true, "MNISTData module imported successfully")
    }

    // MARK: - Basic Functionality Tests

    /// Test SimpleRng produces deterministic results with same seed.
    func testSimpleRngDeterminism() throws {
        var rng1 = SimpleRng(seed: 12345)
        var rng2 = SimpleRng(seed: 12345)

        // Same seed should produce same sequence
        let value1 = rng1.nextUInt32()
        let value2 = rng2.nextUInt32()
        XCTAssertEqual(value1, value2, "Same seed should produce same random sequence")

        // Next values should also match
        let nextValue1 = rng1.nextFloat()
        let nextValue2 = rng2.nextFloat()
        XCTAssertEqual(nextValue1, nextValue2, "RNG sequences should remain synchronized")
    }

    /// Test SimpleRng integer sampling.
    func testSimpleRngIntegerSampling() throws {
        var rng = SimpleRng(seed: 999)

        // Sample integers in range [0, 10)
        for _ in 0..<100 {
            let value = rng.nextInt(upper: 10)
            XCTAssertGreaterThanOrEqual(value, 0)
            XCTAssertLessThan(value, 10)
        }

        // Edge case: upper = 0 should return 0
        let zeroValue = rng.nextInt(upper: 0)
        XCTAssertEqual(zeroValue, 0)
    }

    /// Test that package builds without errors.
    func testPackageBuilds() throws {
        // If we get here, the package has built successfully.
        // This test serves as a canary for build failures.
        XCTAssertTrue(true, "Package built successfully")
    }
}
