"""Choose candidate pushes with predicted outcomes and execute them in the robot simulator."""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import advance, placement_cost, wrap, yaw, save_json
from .evaluate import analytic
from .model import load_predictor
from .simulator import CupSimulator, sample_actions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="Omit for half-travel/no-rotation analytical baseline")
    p.add_argument("--goal", help="Human-derived goal JSON; omit for synthetic smoke-test goal")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-pushes", type=int, default=12)
    p.add_argument("--candidates", type=int, default=96)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--video", help="Optional MP4 for first episode only")
    p.add_argument("--out", default="outputs/task_metrics.json")
    args = p.parse_args()
    if min(args.episodes, args.max_pushes, args.candidates) < 1:
        p.error("episodes, max-pushes and candidates must be positive")
    config = json.loads(Path(args.goal).read_text()) if args.goal else dict(
        x=0.12, y=-0.12, yaw=0, half_width=0.07, half_depth=0.07,
        angle_tolerance_deg=20, source="SYNTHETIC smoke-test goal; no human data")
    goal = np.array([config["x"], config["y"], config["yaw"]])
    if np.any(np.abs(goal[:2]) > 0.22):
        raise ValueError("Goal outside supported workspace. Keep zone centre within +/-0.22 metres.")
    predict = load_predictor(args.model) if args.model else analytic
    simulator = CupSimulator()
    initial_rng = np.random.default_rng(args.seed)
    traces = []

    def success(state, upright):
        return bool(upright and abs(state[0] - goal[0]) <= config["half_width"] and
                    abs(state[1] - goal[1]) <= config["half_depth"] and
                    abs(wrap(yaw(state) - goal[2])) <= np.deg2rad(config["angle_tolerance_deg"]))

    try:
        for episode in range(args.episodes):
            # Initial conditions remain identical across model/baseline runs with same seed.
            start = initial_rng.uniform([-0.12, -0.12, -np.pi], [0.08, 0.12, np.pi])
            simulator.reset(*start)
            if episode == 0 and args.video:
                simulator.start_video(args.video)
            candidate_rng = np.random.default_rng(args.seed + episode + 10000)
            trace = dict(episode=episode, initial=simulator.state().tolist(), pushes=[])
            for step in range(args.max_pushes):
                state = simulator.state()
                if success(state, simulator.upright()) or not simulator.upright() or np.any(np.abs(state[:2]) > 0.27):
                    break
                candidates = sample_actions(state, candidate_rng, args.candidates, goal)
                valid = []
                for action in candidates:
                    contact, end = state[:2] + action[:2], state[:2] + action[:2] + action[2:4]
                    if np.any(np.abs(contact) > 0.30) or np.any(np.abs(end) > 0.30):
                        continue
                    try:
                        simulator.ik(contact)
                        simulator.ik(end)
                    except ValueError:
                        continue
                    valid.append(action)
                if not valid:
                    break
                candidates = np.asarray(valid)
                deltas = predict(state, candidates)
                outcomes = [advance(state, delta) for delta in deltas]
                scores = [placement_cost(outcome, goal) for outcome in outcomes]
                index = int(np.argmin(scores))
                before, action, after, upright = simulator.push(candidates[index])
                trace["pushes"].append(dict(before=before.tolist(), action=action.tolist(),
                    predicted_after=outcomes[index].tolist(), after=after.tolist(), upright=upright))
            final = simulator.state()
            trace.update(final=final.tolist(), upright=simulator.upright(),
                success=success(final, simulator.upright()),
                final_distance_mm=float(np.linalg.norm(final[:2] - goal[:2]) * 1000),
                final_angle_error_deg=float(np.rad2deg(abs(wrap(yaw(final) - goal[2])))))
            traces.append(trace)
            print(f"Episode {episode+1}: success={trace['success']}, pushes={len(trace['pushes'])}, "
                  f"distance={trace['final_distance_mm']:.1f} mm, angle={trace['final_angle_error_deg']:.1f} deg", flush=True)
            if episode == 0 and args.video:
                simulator.close()
                simulator.video = simulator.renderer = None
        save_json(args.out, dict(model=args.model or "analytical baseline", seed=args.seed,
            goal=config, episodes=traces, success_rate=float(np.mean([t["success"] for t in traces])),
            upright_rate=float(np.mean([t["upright"] for t in traces])),
            average_pushes=float(np.mean([len(t["pushes"]) for t in traces])),
            mean_final_distance_mm=float(np.mean([t["final_distance_mm"] for t in traces])),
            mean_final_angle_error_deg=float(np.mean([t["final_angle_error_deg"] for t in traces]))))
        print(f"Saved metrics and state traces to {args.out}")
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
