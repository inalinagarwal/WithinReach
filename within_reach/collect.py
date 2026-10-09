"""Collect physical robot pushes without rendering (CPU only)."""
import argparse
from pathlib import Path

import numpy as np

from .common import split_for, write_record
from .simulator import CupSimulator, sample_actions, stage_actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--pushes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/processed/sim.jsonl")
    args = parser.parse_args()
    if args.episodes < 1 or args.pushes < 1:
        parser.error("episodes and pushes must be positive")
    rng = np.random.default_rng(args.seed)
    simulator = CupSimulator()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Refuse accidental overwrites of a dataset.
    with open(args.out, "x") as stream:
        for episode in range(args.episodes):
            episode_id = f"sim-{args.seed}-{episode:05d}"
            simulator.reset(*rng.uniform([-0.12, -0.16, -np.pi], [0.12, 0.16, np.pi]),
                            friction=rng.uniform(0.25, 0.45))
            for step in range(args.pushes):
                state = simulator.state()
                if not simulator.upright() or np.any(np.abs(state[:2]) > 0.24):
                    break
                # Try alternatives if a contact/endpoint exceeds the toy arm's reach.
                action = None
                # Cover both stage-specific action families as well as generic pushes.
                mode = rng.choice(["generic", "slide", "orient"])
                goal = np.r_[rng.uniform(-0.15, 0.15, 2), rng.uniform(-np.pi, np.pi)]
                candidates = (sample_actions(state, rng, count=24) if mode == "generic"
                              else stage_actions(state, rng, 24, goal, mode))
                for candidate in candidates:
                    start = state[:2] + candidate[:2]
                    end = start + candidate[2:4]
                    if np.any(np.abs(start) > 0.30) or np.any(np.abs(end) > 0.30):
                        continue
                    try:
                        simulator.ik(start)
                        simulator.ik(end)
                    except ValueError:
                        continue
                    action = candidate
                    break
                if action is None:
                    break
                before, action, after, upright = simulator.push(action)
                write_record(stream, dict(source="sim", episode=episode_id,
                    split=split_for(episode_id), step=step, before=before.tolist(),
                    action=action.tolist(), after=after.tolist(), upright=upright,
                    physics_revision=simulator.physics_revision, action_family=str(mode)))
            if (episode + 1) % 25 == 0:
                print(f"Collected {episode + 1}/{args.episodes} episodes", flush=True)
    simulator.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
