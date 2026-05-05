#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 2: SIMT Core (Warp) Test Suite        ║"
echo "╚══════════════════════════════════════════════╝"
python3 tests/test_phase2.py
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 2 Complete                             ║"
echo "╚══════════════════════════════════════════════╝"
