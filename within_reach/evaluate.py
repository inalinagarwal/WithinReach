"""One-step and open-loop state prediction errors on held-out episodes."""
import argparse
from collections import defaultdict

import numpy as np

from .common import read_records, target, advance, wrap, yaw, save_json
from .model import load_predictor


def analytic(state, actions):
    actions = np.atleast_2d(actions)
    # Explicit simple reference, not calibrated against test outcomes.
    return np.c_[0.5 * actions[:, 2:4], np.zeros(len(actions))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/processed/sim.jsonl")
    p.add_argument("--models", nargs="*", default=["outputs/sim.pt"])
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--out", default="outputs/prediction_metrics.json")
    args = p.parse_args()
    records = [r for r in read_records(args.data) if r["split"] == args.split and r["upright"]]
    if not records:
        raise ValueError(f"No upright {args.split} records; collect more episodes")
    predictors = {"no_motion": lambda state, actions: np.zeros((len(np.atleast_2d(actions)), 3)),
                  "half_travel_no_rotation": analytic}
    predictors.update({path: load_predictor(path) for path in args.models})
    results = dict(split=args.split, transitions=len(records),
                   episodes=len(set(r["episode"] for r in records)), models={})
    groups = defaultdict(list)
    for r in records:
        groups[r["episode"]].append(r)
    for name, predict in predictors.items():
        errors = []
        drift = []
        for r in records:
            prediction = predict(np.asarray(r["before"]), [r["action"]])[0]
            truth = target(r)
            errors.append([np.linalg.norm(prediction[:2] - truth[:2]) * 1000,
                           np.rad2deg(abs(wrap(prediction[2] - truth[2])))])
        for group in groups.values():
            group.sort(key=lambda r: r["step"])
            # Human clips can contain separate pushes/repositioning; only sim is continuous.
            if group[0]["source"] != "sim":
                continue
            predicted_state = np.asarray(group[0]["before"])
            last_step = group[0]["step"] - 1
            for r in group:
                if r["step"] != last_step + 1:
                    predicted_state = np.asarray(r["before"])
                delta = predict(predicted_state, [r["action"]])[0]
                predicted_state = advance(predicted_state, delta)
                last_step = r["step"]
            drift.append([np.linalg.norm(predicted_state[:2] - group[-1]["after"][:2]) * 1000,
                          np.rad2deg(abs(wrap(yaw(predicted_state) - yaw(group[-1]["after"]))))])
        values = np.asarray(errors)
        metrics = dict(position_mae_mm=float(values[:, 0].mean()),
                       angle_mae_deg=float(values[:, 1].mean()),
                       position_p90_mm=float(np.quantile(values[:, 0], 0.9)))
        if drift:
            metrics.update(open_loop_final_position_mm=float(np.mean(drift, axis=0)[0]),
                           open_loop_final_angle_deg=float(np.mean(drift, axis=0)[1]))
        results["models"][name] = metrics
        print(name, metrics)
    save_json(args.out, results)


if __name__ == "__main__":
    main()
