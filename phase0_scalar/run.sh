#!/bin/bash
# Phase 0: Scalar Processor — 一键运行脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 0: Scalar Processor Test Suite        ║"
echo "║  toygpgpu — Building a GPGPU Sim in Python   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Run test suite
python3 tests/test_phase0.py

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Phase 0 Complete                             ║"
echo "╚══════════════════════════════════════════════╝"
