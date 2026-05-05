#!/bin/bash
set -e; cd "$(dirname "$0")"
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 11: Learning Console Test Suite       ║"
echo "╚══════════════════════════════════════════════╝"
python3 tests/test_phase11.py
echo ""
echo "Interactive demo:"
echo "  python3 src/learning_console.py tests/programs/demo_divergence.asm --warp-size 4"
