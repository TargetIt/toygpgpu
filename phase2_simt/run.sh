#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --trace: run a demo program with trace output
if [ "$1" = "--trace" ]; then
    DEMO="${2:-tests/programs/01_tid_wid.asm}"
    echo "--- Trace: $DEMO ---"
    PYTHONIOENCODING=utf-8 python3 src/learning_console.py "$DEMO" --auto --max-cycles 500
    exit $?
fi

echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 2: SIMT Core (Warp) Test Suite        ║"
echo "╚══════════════════════════════════════════════╝"
python3 tests/test_phase2.py
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 2 Complete                             ║"
echo "╚══════════════════════════════════════════════╝"
