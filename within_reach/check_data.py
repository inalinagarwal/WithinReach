"""Inspect dataset balance, motion, action ranges and episode split integrity."""
import argparse
from collections import Counter

import numpy as np

from .common import read_records, target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path")
    args = p.parse_args()
    records = read_records(args.path)
    if not records:
        raise ValueError("Dataset is empty")
    assignments = {}
    ids = set()
    for r in records:
        if r["episode"] in assignments and assignments[r["episode"]] != r["split"]:
            raise ValueError(f"Episode leaks across splits: {r['episode']}")
        assignments[r["episode"]] = r["split"]
        identity = (r["episode"], r["step"])
        if identity in ids:
            raise ValueError(f"Duplicate transition: {identity}")
        ids.add(identity)
    delta = np.stack([target(r) for r in records])
    actions = np.asarray([r["action"] for r in records])
    displacement = np.linalg.norm(delta[:, :2], axis=1) * 1000
    angles = np.abs(np.rad2deg(delta[:, 2]))
    print(f"Transitions: {len(records)}; episodes: {len(assignments)}")
    print("Transitions by split:", dict(Counter(r["split"] for r in records)))
    print("Episodes by split:", dict(Counter(assignments.values())))
    print("Sources:", dict(Counter(r["source"] for r in records)))
    print(f"Upright outcomes: {np.mean([r['upright'] for r in records]):.1%}")
    print(f"Median / p90 displacement: {np.median(displacement):.1f} / {np.quantile(displacement, .9):.1f} mm")
    print(f"Median / p90 rotation: {np.median(angles):.1f} / {np.quantile(angles, .9):.1f} deg")
    print(f"Push duration min/max: {actions[:, 4].min():.2f}/{actions[:, 4].max():.2f} seconds")
    if np.mean(displacement < 1) > 0.5:
        print("WARNING: >50% of pushes move less than 1 mm. Inspect contact/controller before scaling up.")
    if np.quantile(angles, .9) < 3:
        print("WARNING: few rotations. Collect off-centre pushes before claiming orientation learning.")
    for split in ("train", "val", "test"):
        if split not in assignments.values():
            print(f"WARNING: missing {split} episodes. Collect more or assign whole videos deliberately.")


if __name__ == "__main__":
    main()
