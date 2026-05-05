#!/bin/bash
set -e; cd "$(dirname "$0")"
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 9: Tensor Core MMA Test Suite         ║"
echo "╚══════════════════════════════════════════════╝"
python3 tests/test_phase9.py
