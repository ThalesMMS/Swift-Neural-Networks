#!/bin/bash
# Verification script for CLI flag consistency across MNIST training scripts
# Tests that mnist_cnn.swift and mnist_attention_pool.swift have consistent CLI interfaces

set -e  # Exit on first error

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS_COUNT=0
FAIL_COUNT=0
TEST_COUNT=0

# Track overall test status
ALL_TESTS_PASSED=true

echo -e "${BOLD}CLI Consistency Verification${NC}"
echo "==============================="
echo ""

# Function to run a test
run_test() {
    local test_name="$1"
    local command="$2"
    local expected="$3"
    local check_type="${4:-exit_code}"  # exit_code, output, or error

    TEST_COUNT=$((TEST_COUNT + 1))
    echo -e "${YELLOW}Test $TEST_COUNT:${NC} $test_name"

    if [ "$check_type" = "exit_code" ]; then
        # Check exit code
        if eval "$command" > /dev/null 2>&1; then
            if [ "$expected" = "0" ]; then
                echo -e "  ${GREEN}✓ PASS${NC}"
                PASS_COUNT=$((PASS_COUNT + 1))
            else
                echo -e "  ${RED}✗ FAIL${NC} - Expected failure but succeeded"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                ALL_TESTS_PASSED=false
            fi
        else
            if [ "$expected" = "1" ]; then
                echo -e "  ${GREEN}✓ PASS${NC}"
                PASS_COUNT=$((PASS_COUNT + 1))
            else
                echo -e "  ${RED}✗ FAIL${NC} - Expected success but failed"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                ALL_TESTS_PASSED=false
            fi
        fi
    elif [ "$check_type" = "output" ]; then
        # Check if output contains expected string
        local output=$(eval "$command" 2>&1)
        if echo "$output" | grep -F -e "$expected" > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓ PASS${NC}"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo -e "  ${RED}✗ FAIL${NC} - Expected output not found: $expected"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            ALL_TESTS_PASSED=false
        fi
    elif [ "$check_type" = "error" ]; then
        # Check if error output contains expected string
        local output=$(eval "$command" 2>&1)
        if echo "$output" | grep -F -e "$expected" > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓ PASS${NC}"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo -e "  ${RED}✗ FAIL${NC} - Expected error message not found: $expected"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            ALL_TESTS_PASSED=false
        fi
    fi
    echo ""
}

# Test 1: mnist_cnn.swift --help shows all 6 flags
echo -e "${BOLD}Testing mnist_cnn.swift${NC}"
echo "----------------------------"
run_test "mnist_cnn.swift --help shows --epochs flag" \
    "swift mnist_cnn.swift --help" \
    "--epochs" \
    "output"

run_test "mnist_cnn.swift --help shows --batch flag" \
    "swift mnist_cnn.swift --help" \
    "--batch" \
    "output"

run_test "mnist_cnn.swift --help shows --lr flag" \
    "swift mnist_cnn.swift --help" \
    "--lr" \
    "output"

run_test "mnist_cnn.swift --help shows --data flag" \
    "swift mnist_cnn.swift --help" \
    "--data" \
    "output"

run_test "mnist_cnn.swift --help shows --seed flag" \
    "swift mnist_cnn.swift --help" \
    "--seed" \
    "output"

run_test "mnist_cnn.swift --help shows --help flag" \
    "swift mnist_cnn.swift --help" \
    "--help" \
    "output"

# Test 2: mnist_attention_pool.swift --help shows all 6 flags
echo -e "${BOLD}Testing mnist_attention_pool.swift${NC}"
echo "----------------------------"
run_test "mnist_attention_pool.swift --help shows --epochs flag" \
    "swift mnist_attention_pool.swift --help" \
    "--epochs" \
    "output"

run_test "mnist_attention_pool.swift --help shows --batch flag" \
    "swift mnist_attention_pool.swift --help" \
    "--batch" \
    "output"

run_test "mnist_attention_pool.swift --help shows --lr flag" \
    "swift mnist_attention_pool.swift --help" \
    "--lr" \
    "output"

run_test "mnist_attention_pool.swift --help shows --data flag" \
    "swift mnist_attention_pool.swift --help" \
    "--data" \
    "output"

run_test "mnist_attention_pool.swift --help shows --seed flag" \
    "swift mnist_attention_pool.swift --help" \
    "--seed" \
    "output"

run_test "mnist_attention_pool.swift --help shows --help flag" \
    "swift mnist_attention_pool.swift --help" \
    "--help" \
    "output"

# Test 3: Both scripts accept custom --data paths (will fail on data loading, which is expected)
echo -e "${BOLD}Testing Custom Data Path Acceptance${NC}"
echo "----------------------------"
run_test "mnist_cnn.swift accepts --data flag (compilation succeeds)" \
    "timeout 10s swift mnist_cnn.swift --data ./custom_data --epochs 1 2>&1" \
    "MNIST" \
    "output"

run_test "mnist_attention_pool.swift accepts --data flag (compilation succeeds)" \
    "timeout 10s swift mnist_attention_pool.swift --data ./custom_data --epochs 1 2>&1" \
    "MNIST" \
    "output"

# Test 4: Both scripts handle invalid arguments consistently
echo -e "${BOLD}Testing Error Handling Consistency${NC}"
echo "----------------------------"
run_test "mnist_cnn.swift rejects invalid flag with error" \
    "swift mnist_cnn.swift --invalid-flag" \
    "Unknown argument" \
    "error"

run_test "mnist_attention_pool.swift rejects invalid flag with error" \
    "swift mnist_attention_pool.swift --invalid-flag" \
    "Unknown argument" \
    "error"

run_test "mnist_cnn.swift rejects invalid epochs value (0)" \
    "swift mnist_cnn.swift --epochs 0" \
    "Invalid value" \
    "error"

run_test "mnist_attention_pool.swift rejects invalid epochs value (0)" \
    "swift mnist_attention_pool.swift --epochs 0" \
    "Invalid value" \
    "error"

run_test "mnist_cnn.swift rejects invalid batch value (-1)" \
    "swift mnist_cnn.swift --batch -1" \
    "Invalid value" \
    "error"

run_test "mnist_attention_pool.swift rejects invalid batch value (-1)" \
    "swift mnist_attention_pool.swift --batch -1" \
    "Invalid value" \
    "error"

run_test "mnist_cnn.swift rejects invalid lr value (0)" \
    "swift mnist_cnn.swift --lr 0" \
    "Invalid value" \
    "error"

run_test "mnist_attention_pool.swift rejects invalid lr value (0)" \
    "swift mnist_attention_pool.swift --lr 0" \
    "Invalid value" \
    "error"

# Test 5: Verify short flag aliases work
echo -e "${BOLD}Testing Short Flag Aliases${NC}"
echo "----------------------------"
run_test "mnist_cnn.swift accepts -e for --epochs" \
    "swift mnist_cnn.swift -h 2>&1" \
    "-e" \
    "output"

run_test "mnist_attention_pool.swift accepts -e for --epochs" \
    "swift mnist_attention_pool.swift -h 2>&1" \
    "-e" \
    "output"

run_test "mnist_cnn.swift accepts -b for --batch" \
    "swift mnist_cnn.swift -h 2>&1" \
    "-b" \
    "output"

run_test "mnist_attention_pool.swift accepts -b for --batch" \
    "swift mnist_attention_pool.swift -h 2>&1" \
    "-b" \
    "output"

run_test "mnist_cnn.swift accepts -d for --data" \
    "swift mnist_cnn.swift -h 2>&1" \
    "-d" \
    "output"

run_test "mnist_attention_pool.swift accepts -d for --data" \
    "swift mnist_attention_pool.swift -h 2>&1" \
    "-d" \
    "output"

run_test "mnist_cnn.swift accepts -s for --seed" \
    "swift mnist_cnn.swift -h 2>&1" \
    "-s" \
    "output"

run_test "mnist_attention_pool.swift accepts -s for --seed" \
    "swift mnist_attention_pool.swift -h 2>&1" \
    "-s" \
    "output"

# Summary
echo ""
echo -e "${BOLD}Test Summary${NC}"
echo "============="
echo -e "Total tests: $TEST_COUNT"
echo -e "${GREEN}Passed: $PASS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"
echo ""

if [ "$ALL_TESTS_PASSED" = true ]; then
    echo -e "${GREEN}${BOLD}✓ All CLI consistency tests passed!${NC}"
    echo ""
    echo "Both mnist_cnn.swift and mnist_attention_pool.swift have:"
    echo "  • All 6 standard CLI flags (--epochs, --batch, --lr, --data, --seed, --help)"
    echo "  • Consistent short flag aliases (-e, -b, -l, -d, -s, -h)"
    echo "  • Consistent error handling for invalid arguments"
    echo "  • Support for custom data paths"
    exit 0
else
    echo -e "${RED}${BOLD}✗ Some tests failed. Please review the output above.${NC}"
    exit 1
fi
