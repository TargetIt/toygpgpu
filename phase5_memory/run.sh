#!/bin/bash
set -e; cd "$(dirname "$0")"

# --trace: run a demo program with trace output
if [ "$1" = "--trace" ]; then
    DEMO="${2:-tests/programs/01_shared_mem.asm}"
    echo "--- Trace: $DEMO ---"
    PYTHONIOENCODING=utf-8 python3 src/learning_console.py "$DEMO" --auto --max-cycles 500
    exit $?
fi

echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 5: Memory Hierarchy Test Suite        ║"
echo "╚══════════════════════════════════════════════╝"
python3 tests/test_phase5.py
