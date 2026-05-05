#!/bin/bash
set -e; cd "$(dirname "$0")"
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 8: PTX Frontend Test Suite            ║"
echo "╚══════════════════════════════════════════════╝"
python3 tests/test_phase8.py
