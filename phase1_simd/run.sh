#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 1: SIMD Vector Processor Test Suite   ║"
echo "╚══════════════════════════════════════════════╝"
python3 tests/test_phase1.py
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 1 Complete                             ║"
echo "╚══════════════════════════════════════════════╝"
