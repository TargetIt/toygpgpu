#!/bin/bash
set -e; cd "$(dirname "$0")"
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 6: Kernel Launch & Scheduling         ║"
echo "╚══════════════════════════════════════════════╝"
python3 tests/test_phase6.py
