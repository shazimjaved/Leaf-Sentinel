import argparse
from pathlib import Path
import yaml
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.train import run_training

def main():
    parser = argparse.ArgumentParser(description="LeafSentinel Phase 3 Classification Training")
    parser.add_argument("--config", type=str, default="configs/classification.yaml", help="Path to config")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch on subset")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    run_training(config, args.smoke_test)

if __name__ == "__main__":
    main()
