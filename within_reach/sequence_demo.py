"""Execute video-derived coarse skill sequences using the existing learned dynamics."""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import advance, placement_cost, save_json, wrap, yaw
from .evaluate import analytic
from .model import load_predictor
from .simulator import CupSimulator, stage_actions


def stage_cost(predicted_state, goal, mode, slide_yaw, half_width, half_depth):
    """Slide preserves the stage-entry angle; orient strongly penalises zone exit."""
    position_error = np.linalg.norm(predicted_state[:2] - goal[:2])
    if mode == 'slide':
        return float(position_error + 0.08 * abs(wrap(yaw(predicted_state) - slide_yaw)))
    angle_error = abs(wrap(yaw(predicted_state) - goal[2]))
    if mode == 'orient':
        outside = np.maximum(np.abs(predicted_state[:2] - goal[:2]) -
                             np.array([half_width, half_depth]), 0).sum()
        return float(2 * position_error + 0.06 * angle_error + 8 * outside)
    return placement_cost(predicted_state, goal)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--model', help='Omit to use the analytical predictor baseline')
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--max-pushes-per-stage', type=int, default=12)
    p.add_argument('--candidates', type=int, default=96)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--video', help='First episode only; optional MP4')
    p.add_argument('--out', default='outputs/human_sequence_metrics.json')
    args = p.parse_args()
    if min(args.episodes, args.max_pushes_per_stage, args.candidates) < 1:
        p.error('Episode/push/candidate counts must be positive')
    plan = json.loads(Path(args.plan).read_text())
    predict = load_predictor(args.model) if args.model else analytic
    sim = CupSimulator()
    traces = []
    initial_rng = np.random.default_rng(args.seed)

    def reached(state, stage):
        goal = np.asarray(stage['goal'])
        position_ok = (abs(state[0] - goal[0]) <= plan['half_width'] and
                       abs(state[1] - goal[1]) <= plan['half_depth'])
        angle_ok = not stage['require_angle'] or abs(wrap(yaw(state) - goal[2])) <= np.deg2rad(plan['angle_tolerance_deg'])
        return bool(sim.upright() and position_ok and angle_ok)

    try:
        for episode in range(args.episodes):
            start = np.asarray(plan['initial']) + initial_rng.uniform([-.015, -.015, -.12], [.015, .015, .12])
            sim.reset(*start)
            if episode == 0 and args.video:
                sim.start_video(args.video)
            rng = np.random.default_rng(args.seed + 10000 + episode)
            trace = dict(episode=episode, initial=sim.state().tolist(), stages=[])
            for stage in plan['stages']:
                goal = np.asarray(stage['goal'])
                mode = stage['name'] if stage['name'] in ('slide', 'orient') else 'joint'
                slide_yaw = yaw(sim.state())
                entry = dict(name=stage['name'], goal=goal.tolist(),
                             initial_yaw_rad=float(slide_yaw), pushes=[])
                for _ in range(args.max_pushes_per_stage):
                    state = sim.state()
                    if reached(state, stage) or not sim.upright() or np.any(np.abs(state[:2]) > .27):
                        break
                    valid = []
                    for action in stage_actions(state, rng, args.candidates, goal, mode):
                        start_xy = state[:2] + action[:2]
                        end_xy = start_xy + action[2:4]
                        if np.any(np.abs(start_xy) > .30) or np.any(np.abs(end_xy) > .30):
                            continue
                        try:
                            sim.ik(start_xy)
                            sim.ik(end_xy)
                        except ValueError:
                            continue
                        valid.append(action)
                    if not valid:
                        break
                    actions = np.asarray(valid)
                    deltas = predict(state, actions)
                    predicted = [advance(state, delta) for delta in deltas]
                    costs = [stage_cost(s, goal, mode, slide_yaw, plan['half_width'],
                                        plan['half_depth']) for s in predicted]
                    selected = int(np.argmin(costs))
                    before, action, after, upright = sim.push(actions[selected])
                    entry['pushes'].append(dict(before=before.tolist(), action=action.tolist(),
                        predicted_after=predicted[selected].tolist(), after=after.tolist(), upright=upright,
                        reposition_distance_mm=float(np.linalg.norm(before[:2] - state[:2]) * 1000),
                        angle_change_deg=float(np.rad2deg(abs(wrap(yaw(after) - yaw(before)))))))
                final_state = sim.state()
                entry.update(success=reached(final_state, stage), final=final_state.tolist(),
                    final_angle_drift_deg=float(np.rad2deg(abs(wrap(yaw(final_state) - slide_yaw)))))
                trace['stages'].append(entry)
                print(f"Episode {episode+1} {stage['name']}: success={entry['success']}, "
                      f"pushes={len(entry['pushes'])}", flush=True)
                # Do not claim a chained sequence completed if an earlier stage failed.
                if not entry['success']:
                    break
            trace.update(success=len(trace['stages']) == len(plan['stages']) and
                         all(s['success'] for s in trace['stages']),
                         final=sim.state().tolist(), upright=sim.upright())
            traces.append(trace)
            if episode == 0 and args.video:
                sim.close()
                sim.video = sim.renderer = None
        save_json(args.out, dict(plan=plan, model=args.model or 'analytical baseline', seed=args.seed,
            physics_revision=sim.physics_revision,
            success_rate=float(np.mean([t['success'] for t in traces])),
            upright_rate=float(np.mean([t['upright'] for t in traces])), episodes=traces,
            scope='Applicant-video goal/sequence retargeting with simulation-trained dynamics; no VLA or human dynamics pretraining.'))
        print(f'Saved {args.out}')
    finally:
        sim.close()


if __name__ == '__main__':
    main()
