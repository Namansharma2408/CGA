"""Legacy shim (P2-5): use scripts/plot_basic.py instead. Kept for backward compat."""
import os, sys, warnings
warnings.warn("plot_paper_figures.py is deprecated; use scripts/plot_basic.py", DeprecationWarning)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import runpy
runpy.run_path(os.path.join(SCRIPT_DIR, "plot_basic.py"), run_name="__main__")
