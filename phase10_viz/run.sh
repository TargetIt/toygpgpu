#!/bin/bash
set -e; cd "$(dirname "$0")"

# --trace: run a demo program with trace output (no demo programs in this phase)
if [ "$1" = "--trace" ]; then
    echo "No demo programs available for Phase 10."
    echo "Usage: python3 src/learning_console.py <program.asm> --auto"
    exit 1
fi

echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 10: Visualization & Toolchain         ║"
echo "╚══════════════════════════════════════════════╝"
python3 tests/test_phase10.py
