"""Central path contracts (P2-4, P2-6). All scripts resolve from __file__, never bare cwd."""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODELS_TRAINED_DIR = os.path.join(PROJECT_ROOT, "models_trained")
IMPROVED_DIR = os.path.join(PROJECT_ROOT, "improved")
IMPROVED_MODELS_TRAINED_DIR = os.path.join(IMPROVED_DIR, "models_trained")

BASIC_BINARY = os.path.join(PROJECT_ROOT, "build", "Release", "cga_bfs")
BASIC_BINARY_FALLBACK = os.path.join(PROJECT_ROOT, "build", "cga_bfs")
IMPROVED_BINARY = os.path.join(PROJECT_ROOT, "build", "Release", "cga_bfs_improved")

TRAIN_GLOB = os.path.join(PROJECT_ROOT, "data", "train", "*.mtx")
TEST_GLOB = os.path.join(PROJECT_ROOT, "data", "test", "*.mtx")

BASIC_SUMMARY_CSV = os.path.join(PROJECT_ROOT, "bfs_run_summary.csv")
IMPROVED_RESULTS_DIR = os.path.join(IMPROVED_DIR, "results")
IMPROVED_INFERENCE_CSV = os.path.join(IMPROVED_RESULTS_DIR, "inference_results.csv")
