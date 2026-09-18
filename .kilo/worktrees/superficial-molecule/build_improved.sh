#!/bin/bash
# build_improved.sh - Build script for the Improved CGA-BFS variants

echo "═══════════════════════════════════════════════════════"
echo "  CGA-BFS Improved Build | 25-dim features | DTA-Learners"
echo "═══════════════════════════════════════════════════════"

mkdir -p improved/build
cd improved/build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DIMPROVED_FEATURES=ON \
      -DARCH=sm_86 \
      ../../
make -j$(nproc)

echo "Done! Improved binary ready at improved/build/cga_bfs_improved"
