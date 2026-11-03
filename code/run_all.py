"""Train first, then evaluate only the explicitly selected run."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--quick", action="store_true", help="One-epoch W1 smoke check, not paper reproduction")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.train and args.predictions is None:
        parser.error("Specify --train or an explicit --predictions file; old outputs are never selected automatically")
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, check=True)
    if args.train:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Training output must be empty to avoid mixing runs")
        command = [sys.executable, str(root / "code/train_lowo.py"), "--output", str(args.output)]
        if args.quick:
            command.extend(["--test-wafer", "W1", "--mae-epochs", "1", "--teacher-epochs", "1", "--student-epochs", "1"])
        subprocess.run(command, cwd=root, check=True)
        predictions = args.output / "predictions_all_folds.csv"
    else:
        predictions = args.predictions
    subprocess.run([sys.executable, str(root / "code/evaluate.py"), "--predictions", str(predictions),
                    "--manifest", str(root / "data/manifest.csv"), "--output", str(args.output / "metrics_per_wafer.csv")],
                   cwd=root, check=True)


if __name__ == "__main__":
    main()
