"""train.py — entry point. Usage: python scripts/train.py --config configs/base_config.yaml"""
import argparse, yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config: {cfg}")

if __name__ == "__main__":
    main()
