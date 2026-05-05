#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 3: SIMT Stack + Branch Test Suite     ║"
echo "╚══════════════════════════════════════════════╝"
python3 tests/test_phase3.py
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 3 Complete                             ║"
echo "╚══════════════════════════════════════════════╝"
