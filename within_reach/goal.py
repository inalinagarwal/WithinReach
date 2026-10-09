"""Estimate a preferred placement from marked TRAIN demonstrations only."""
import argparse
import numpy as np

from .common import read_records, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--human", default="data/processed/human.jsonl")
    p.add_argument("--out", default="outputs/goal.json")
    p.add_argument("--half-width", type=float, default=0.07)
    p.add_argument("--half-depth", type=float, default=0.07)
    p.add_argument("--angle-tolerance-deg", type=float, default=20)
    args = p.parse_args()
    if args.half_width <= 0 or args.half_depth <= 0 or not 0 < args.angle_tolerance_deg <= 180:
        p.error("Zone dimensions must be positive and angular tolerance in (0, 180]")
    records = [r for r in read_records(args.human)
               if r["split"] == "train" and r.get("preferred") and r["upright"]]
    if not records:
        raise ValueError("No preferred TRAIN examples. Annotate final placements with --preferred --split train.")
    states = np.asarray([r["after"] for r in records])
    mean_angle_vector = states[:, 2:4].mean(0)
    if np.linalg.norm(mean_angle_vector) < 0.5:
        raise ValueError("Preferred handle angles conflict. Use one person's one preference per experiment.")
    save_json(args.out, dict(x=float(states[:, 0].mean()), y=float(states[:, 1].mean()),
        yaw=float(np.arctan2(mean_angle_vector[1], mean_angle_vector[0])),
        half_width=args.half_width, half_depth=args.half_depth,
        angle_tolerance_deg=args.angle_tolerance_deg, source="human preferred training placements",
        n=len(records), episodes=sorted(set(r["episode"] for r in records))))
    print(f"Saved goal to {args.out}")


if __name__ == "__main__":
    main()
