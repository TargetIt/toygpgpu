#!/bin/bash
# Phase 0: Scalar Processor — 一键运行脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --trace: run a demo program with trace output
if [ "$1" = "--trace" ]; then
    DEMO="${2:-tests/programs/01_basic_arith.asm}"
    echo "--- Trace: $DEMO ---"
    PYTHONIOENCODING=utf-8 python3 src/learning_console.py "$DEMO" --auto --max-cycles 500
    exit $?
fi

echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 0: Scalar Processor Test Suite        ║"
echo "║  toygpgpu — Building a GPGPU Sim in Python   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Run test suite
python3 tests/test_phase0.py

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 0 Complete                             ║"
echo "╚══════════════════════════════════════════════╝"
