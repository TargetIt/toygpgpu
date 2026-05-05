#!/bin/bash
set -e; cd "$(dirname "$0")"
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 5: Memory Hierarchy Test Suite        ║"
echo "╚══════════════════════════════════════════════╝"
python3 tests/test_phase5.py
