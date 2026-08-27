"""Command-line entry point for synthetic mobility simulation."""

import argparse
from pathlib import Path
import random

import numpy as np

from mobsim import DEpr, DTEpr, EPR, IPT, Environment


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducible simulations."""
    np.random.seed(seed)
    random.seed(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic individual mobility trajectories.")
    parser.add_argument(
        "--population",
        type=int,
        default=8000,
        help="Population number to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2000,
        help="Location-sequence length per user (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="dtepr",
        choices=["epr", "ipt", "depr", "dtepr"],
        help="Mobility model (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/simulation.yml"),
        help="Simulation YAML file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: outputs/simulation/<model>.csv)",
    )
    parser.add_argument("--seed", type=int, default=49, help="Random seed (default: %(default)s)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_seed(args.seed)

    env = Environment(args.config)
    model_classes = {
        "epr": EPR,
        "ipt": IPT,
        "depr": DEpr,
        "dtepr": DTEpr,
    }
    simulator = model_classes[args.model](env)
    trajectory = simulator.simulate(seq_len=args.sequence_length, pop_num=args.population)
    trajectory.index.name = "index"

    output_path = args.output or Path("outputs/simulation") / f"{args.model}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(output_path)
    print(f"Saved {len(trajectory):,} trajectory rows to {output_path}")


if __name__ == "__main__":
    main()
