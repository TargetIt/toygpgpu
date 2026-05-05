#!/bin/bash
set -e; cd "$(dirname "$0")"
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 7: Pipeline Decouple Test Suite       ║"
echo "╚══════════════════════════════════════════════╝"
python3 tests/test_phase7.py
