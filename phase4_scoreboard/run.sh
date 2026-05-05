#!/bin/bash
set -e; cd "$(dirname "$0")"
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 4: Scoreboard + Pipeline Test Suite   ║"
echo "╚══════════════════════════════════════════════╝"
python3 tests/test_phase4.py
