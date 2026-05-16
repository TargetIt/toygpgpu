#!/bin/bash
set -e; cd "$(dirname "$0")"

# --trace: run a demo program with trace output
if [ "$1" = "--trace" ]; then
    DEMO="${2:-tests/programs/demo_basic.asm}"
    echo "--- Trace: $DEMO ---"
    PYTHONIOENCODING=utf-8 python3 src/learning_console.py "$DEMO" --auto --max-cycles 500
    exit $?
fi

echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 11: Learning Console Test Suite       ║"
echo "╚══════════════════════════════════════════════╝"
python3 tests/test_phase11.py
echo ""
echo "Interactive demo:"
echo "  python3 src/learning_console.py tests/programs/demo_divergence.asm --warp-size 4"
