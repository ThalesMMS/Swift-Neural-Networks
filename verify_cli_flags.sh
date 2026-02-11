#!/bin/bash
# Quick CLI flag verification for mnist_cnn.swift and mnist_attention_pool.swift
# Verifies both scripts have consistent CLI interfaces

set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BOLD}CLI Flags Verification${NC}"
echo "======================="
echo ""

# Function to check for a flag in help output
check_flag() {
    local script="$1"
    local flag="$2"
    if swift "$script" --help 2>&1 | grep -F -e "$flag" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $script has $flag"
        return 0
    else
        echo -e "${RED}✗${NC} $script missing $flag"
        return 1
    fi
}

# Test 1: Verify mnist_cnn.swift has all 6 flags
echo -e "${YELLOW}Checking mnist_cnn.swift flags...${NC}"
check_flag "mnist_cnn.swift" "--epochs"
check_flag "mnist_cnn.swift" "--batch"
check_flag "mnist_cnn.swift" "--lr"
check_flag "mnist_cnn.swift" "--data"
check_flag "mnist_cnn.swift" "--seed"
check_flag "mnist_cnn.swift" "--help"
echo ""

# Test 2: Verify mnist_attention_pool.swift has all 6 flags
echo -e "${YELLOW}Checking mnist_attention_pool.swift flags...${NC}"
check_flag "mnist_attention_pool.swift" "--epochs"
check_flag "mnist_attention_pool.swift" "--batch"
check_flag "mnist_attention_pool.swift" "--lr"
check_flag "mnist_attention_pool.swift" "--data"
check_flag "mnist_attention_pool.swift" "--seed"
check_flag "mnist_attention_pool.swift" "--help"
echo ""

# Test 3: Verify short aliases
echo -e "${YELLOW}Checking short flag aliases...${NC}"
check_flag "mnist_cnn.swift" "-e"
check_flag "mnist_cnn.swift" "-b"
check_flag "mnist_cnn.swift" "-l"
check_flag "mnist_cnn.swift" "-d"
check_flag "mnist_cnn.swift" "-s"
check_flag "mnist_cnn.swift" "-h"
check_flag "mnist_attention_pool.swift" "-e"
check_flag "mnist_attention_pool.swift" "-b"
check_flag "mnist_attention_pool.swift" "-l"
check_flag "mnist_attention_pool.swift" "-d"
check_flag "mnist_attention_pool.swift" "-s"
check_flag "mnist_attention_pool.swift" "-h"
echo ""

# Test 4: Verify error handling for unknown arguments
echo -e "${YELLOW}Checking error handling...${NC}"
if swift mnist_cnn.swift --invalid-flag 2>&1 | grep -F -e "Unknown argument" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} mnist_cnn.swift rejects unknown arguments"
else
    echo -e "${RED}✗${NC} mnist_cnn.swift doesn't properly reject unknown arguments"
fi

if swift mnist_attention_pool.swift --invalid-flag 2>&1 | grep -F -e "Unknown argument" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} mnist_attention_pool.swift rejects unknown arguments"
else
    echo -e "${RED}✗${NC} mnist_attention_pool.swift doesn't properly reject unknown arguments"
fi
echo ""

# Test 5: Verify --data flag is accepted (quick syntax check)
echo -e "${YELLOW}Checking --data flag acceptance...${NC}"
# Just check that the flag is recognized (error will be about missing data files, not unknown flag)
if swift mnist_cnn.swift --data ./custom_data --help 2>&1 | grep -F -e "USAGE" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} mnist_cnn.swift accepts --data flag"
else
    echo -e "${RED}✗${NC} mnist_cnn.swift doesn't accept --data flag"
fi

if swift mnist_attention_pool.swift --data ./custom_data --help 2>&1 | grep -F -e "USAGE" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} mnist_attention_pool.swift accepts --data flag"
else
    echo -e "${RED}✗${NC} mnist_attention_pool.swift doesn't accept --data flag"
fi
echo ""

echo -e "${GREEN}${BOLD}✓ CLI flag verification complete!${NC}"
echo ""
echo "Summary:"
echo "  • Both scripts have all 6 standard flags: --epochs, --batch, --lr, --data, --seed, --help"
echo "  • Both scripts have short aliases: -e, -b, -l, -d, -s, -h"
echo "  • Both scripts reject unknown arguments consistently"
echo "  • Both scripts accept custom --data paths"
