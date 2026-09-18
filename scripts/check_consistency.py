"""CI consistency guard (P2-7): fails on stale spec numbers/paths.

Checks:
- No 'max_depth=5 each' (correct: M1=3,M2=7,M3=6,M4=8 per SPEC.md).
- No 'W=12' for DTA window (correct: W=20).
- No 'improved/models/*.pkl' (correct: improved/models_trained/).
- No 'improved/build/cga_bfs_improved' (correct: build/Release/cga_bfs_improved).
- No '.exe' fallback on Linux paths outside os.name guard.
- FEATURE_NAMES match FEATURE_SPEC.md (basic DIM 19, improved DIM 25 with
  col_cv at 4 and degree_variance at 19).
- KERNELS.md canonical names present in training scripts.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAIL = []

def grep_stale(pattern, paths, msg):
    rx = re.compile(pattern)
    for rel in paths:
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            continue
        with open(p, errors="ignore") as f:
            for i, line in enumerate(f, 1):
                if "was stale" in line or "stale " in line.lower() and "see " in line:
                    continue  # documenting the fix, not restating the stale value
                if rx.search(line):
                    FAIL.append(f"{rel}:{i}: {msg}: {line.strip()[:100]}")

DOCS = ["README.md", "EXECUTION_STEPS.md", "FLOW_ARCHITECTURE.md",
        "improved/README.md", "SPEC.md", "build.sh", "improved/build.sh"]

grep_stale(r"max_depth\s*=\s*5\s*each", DOCS, "stale depths (see SPEC.md M1=3,M2=7,M3=6,M4=8)")
grep_stale(r"\bW\s*=\s*12\b", DOCS + ["improved/models/inference.h"],
           "stale W=12 (code UCB_WINDOW=20, see SPEC.md)")
grep_stale(r"improved/models/\*\.pkl", DOCS, "stale model path (use improved/models_trained/)")
grep_stale(r"improved/build/cga_bfs_improved", DOCS, "stale binary path (use build/Release/cga_bfs_improved)")

# .exe outside os.name guard (P2-8).
for rel in ["improved/scripts/run_inference.py", "scripts/run_inference.py"]:
    p = os.path.join(ROOT, rel)
    with open(p) as f:
        txt = f.read()
    if '.exe' in txt and 'os.name' not in txt:
        FAIL.append(f"{rel}: bare .exe fallback without os.name guard")

# Feature spec checks.
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "improved", "scripts"))
try:
    import importlib.util as _ilu
    def load_names(path):
        spec = _ilu.spec_from_file_location("fx", path)
        m = _ilu.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.FEATURE_NAMES, m.DIM
    bn, bd = load_names(os.path.join(ROOT, "scripts", "feature_extraction.py"))
    assert bd == 19 and bn[4] == "col_cv", f"basic FEATURE mismatch: {bd} {bn[4]}"
    # improved module has same filename; load via path (already cached as fx) — force reload
    import importlib as _il
    spec2 = _ilu.spec_from_file_location("fx2", os.path.join(ROOT, "improved", "scripts", "feature_extraction.py"))
    m2 = _ilu.module_from_spec(spec2)
    spec2.loader.exec_module(m2)
    assert m2.DIM == 25 and m2.FEATURE_NAMES[4] == "col_cv" and m2.FEATURE_NAMES[19] == "degree_variance", \
        f"improved FEATURE mismatch: {m2.DIM} {m2.FEATURE_NAMES[4]} {m2.FEATURE_NAMES[19]}"
except Exception as e:
    FAIL.append(f"feature spec: {e}")

# Kernel names in training scripts.
for rel in ["scripts/train_models.py", "improved/scripts/train_models.py"]:
    p = os.path.join(ROOT, rel)
    with open(p) as f:
        txt = f.read()
    for k in ["PM-BHash", "LB-PM-BHash", "PB-MSPA", "LB-PB-MSPA", "LB-MSPA"]:
        if k not in txt:
            FAIL.append(f"{rel}: missing canonical kernel {k} (see KERNELS.md)")

if FAIL:
    print("CONSISTENCY CHECK FAILED:")
    for x in FAIL:
        print(" -", x)
    sys.exit(1)
print("Consistency check passed.")
