import argparse
import yaml
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.severity.train import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)

def main():
    parser = argparse.ArgumentParser(description="LeafSentinel Phase 4 Severity Regression Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--smoke-test", action="store_true", help="Run in quick smoke test mode on CPU")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_training(config, smoke_test=args.smoke_test)

if __name__ == "__main__":
    main()
