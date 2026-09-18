#!/bin/bash
# build_improved.sh - Canonical improved build (P2-2: single output path).
# Output: build/Release/cga_bfs_improved (same as improved/build.sh).
set -e
echo "═══════════════════════════════════════════════════════"
echo "  CGA-BFS Improved Build | 25-dim features | DTA-Learners"
echo "═══════════════════════════════════════════════════════"

# Delegate to the canonical script so CMake and nvcc paths cannot diverge.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/improved/build.sh"

echo "Done! Improved binary ready at build/Release/cga_bfs_improved"
echo "(legacy improved/build/ directory is no longer used; see P2-2)"
