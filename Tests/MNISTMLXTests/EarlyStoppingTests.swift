// ============================================================================
// EarlyStoppingTests.swift - Tests for Early Stopping Patience Logic
// ============================================================================

import XCTest
@testable import MNISTMLX

final class EarlyStoppingTests: XCTestCase {

    private func runAccuracyProgression(
        _ accuracies: [Float],
        patience: Int?,
        minDelta: Float
    ) -> (epochsRun: Int, state: EarlyStoppingState) {
        var state = EarlyStoppingState()
        var bestValidationAccuracy: Float = 0.0
        var epochsRun = 0

        for (index, validationAccuracy) in accuracies.enumerated() {
            let epoch = index + 1
            epochsRun = epoch
            let shouldStop = state.update(
                validationAccuracy: validationAccuracy,
                bestValidationAccuracy: bestValidationAccuracy,
                epoch: epoch,
                patience: patience,
                minDelta: minDelta
            )

            if validationAccuracy > bestValidationAccuracy {
                bestValidationAccuracy = validationAccuracy
            }
            if shouldStop {
                break
            }
        }

        return (epochsRun, state)
    }

    func testDefaultBehaviorDoesNotStopWithoutPatience() {
        var state = EarlyStoppingState()

        let shouldStop = state.update(
            validationAccuracy: 0.90,
            bestValidationAccuracy: 0.91,
            epoch: 2,
            patience: nil,
            minDelta: 0.0
        )

        XCTAssertFalse(shouldStop)
        XCTAssertFalse(state.stoppedEarly)
        XCTAssertNil(state.reason)
    }

    func testOmittingEarlyStoppingRunsAllConfiguredEpochs() {
        let result = runAccuracyProgression(
            [0.90, 0.90, 0.89, 0.88],
            patience: nil,
            minDelta: 0.0
        )

        XCTAssertEqual(result.epochsRun, 4)
        XCTAssertFalse(result.state.stoppedEarly)
        XCTAssertNil(result.state.reason)
    }

    func testPatienceCounterResetsOnMeaningfulImprovement() {
        var state = EarlyStoppingState(patienceCounter: 1)

        let shouldStop = state.update(
            validationAccuracy: 0.912,
            bestValidationAccuracy: 0.91,
            epoch: 3,
            patience: 2,
            minDelta: 0.001
        )

        XCTAssertFalse(shouldStop)
        XCTAssertEqual(state.patienceCounter, 0)
        XCTAssertFalse(state.stoppedEarly)
    }

    func testPatienceCounterIncrementsWhenImprovementEqualsMinDelta() {
        var state = EarlyStoppingState()

        let shouldStop = state.update(
            validationAccuracy: 0.75,
            bestValidationAccuracy: 0.50,
            epoch: 2,
            patience: 2,
            minDelta: 0.25
        )

        XCTAssertFalse(shouldStop)
        XCTAssertEqual(state.patienceCounter, 1)
        XCTAssertFalse(state.stoppedEarly)
    }

    func testPatienceCounterIncrementsWhenImprovementIsBelowMinDelta() {
        var state = EarlyStoppingState()

        let shouldStop = state.update(
            validationAccuracy: 0.625,
            bestValidationAccuracy: 0.50,
            epoch: 2,
            patience: 2,
            minDelta: 0.25
        )

        XCTAssertFalse(shouldStop)
        XCTAssertEqual(state.patienceCounter, 1)
        XCTAssertFalse(state.stoppedEarly)
    }

    func testEarlyStopTriggersWhenValidationStagnates() {
        var state = EarlyStoppingState()

        XCTAssertFalse(state.update(
            validationAccuracy: 0.9105,
            bestValidationAccuracy: 0.91,
            epoch: 2,
            patience: 2,
            minDelta: 0.001
        ))

        XCTAssertTrue(state.update(
            validationAccuracy: 0.9106,
            bestValidationAccuracy: 0.91,
            epoch: 3,
            patience: 2,
            minDelta: 0.001
        ))

        XCTAssertTrue(state.stoppedEarly)
        XCTAssertEqual(state.patienceCounter, 2)
        XCTAssertEqual(
            state.reason,
            "Early stopping: no validation improvement for 2 epochs (patience exhausted at epoch 3)"
        )
    }

    func testPatienceOneStopsAfterFirstNonImprovingEpoch() {
        let result = runAccuracyProgression(
            [0.80, 0.80, 0.81],
            patience: 1,
            minDelta: 0.0
        )

        XCTAssertEqual(result.epochsRun, 2)
        XCTAssertTrue(result.state.stoppedEarly)
        XCTAssertEqual(
            result.state.reason,
            "Early stopping: no validation improvement for 1 epoch (patience exhausted at epoch 2)"
        )
    }

    func testMinDeltaZeroResetsOnAnyStrictImprovement() {
        let result = runAccuracyProgression(
            [0.80, 0.8001, 0.8001],
            patience: 1,
            minDelta: 0.0
        )

        XCTAssertEqual(result.epochsRun, 3)
        XCTAssertTrue(result.state.stoppedEarly)
        XCTAssertEqual(result.state.patienceCounter, 1)
    }

    func testVeryHighMinDeltaRequiresLargeImprovementToResetPatience() {
        let result = runAccuracyProgression(
            [0.80, 0.85, 0.90],
            patience: 2,
            minDelta: 0.20
        )

        XCTAssertEqual(result.epochsRun, 3)
        XCTAssertTrue(result.state.stoppedEarly)
        XCTAssertEqual(result.state.patienceCounter, 2)
        XCTAssertEqual(
            result.state.reason,
            "Early stopping: no validation improvement for 2 epochs (patience exhausted at epoch 3)"
        )
    }
}
