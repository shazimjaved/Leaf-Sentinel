import argparse
from pathlib import Path
import yaml
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.evaluate import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="LeafSentinel Phase 3 Classification Evaluation")
    parser.add_argument("--config", type=str, default="configs/classification.yaml", help="Path to config")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    run_evaluation(config)

if __name__ == "__main__":
    main()
