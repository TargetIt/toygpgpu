#!/bin/bash
set -e; cd "$(dirname "$0")"
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 10: Visualization & Toolchain         ║"
echo "╚══════════════════════════════════════════════╝"
python3 tests/test_phase10.py
