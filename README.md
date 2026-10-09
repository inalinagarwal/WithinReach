# Project

A robot that pushes a cup into a target area and turns its handle toward a chosen direction.

# Why this task?

I wanted to explore applications of assistance for older adults: bringing a cup closer and leaving its handle facing their preferred hand. Next stage of the project could be to test it for actual old humans.

## What I built

I recorded five videos using my phone: three of pushing and rotating a cup, one of tipping it, and one of pushing it off the table. I used the successful videos to define coarse robot goals and the order of actions: slide first, then orient, or do both together.

The camera moved during recording, so I couldn't reliably recover physical trajectories. Instead, the target distance is a chosen simulator parameter, not a measurement from the video.

A small PyTorch model learns cup motion from MuJoCo pushes. Given the cup pose and a proposed push, it predicts the change in position and angle. The controller scores candidate pushes, executes one, observes the cup, and repeats.

# The videos supply the goals and skill order. Simulation data trains the dynamics model. This is a state world model, not a VLA or a video-generation model.

- MuJoCo: hollow cup and a simple arm with two rotating joints and a lifting pusher.
- Predictor: 9 inputs, two 128-unit hidden layers, 3 outputs; 18,179 parameters.
- Inputs: cup position/orientation, contact offset, push displacement and duration.
- Outputs: change in x, y and yaw.
- CPU training: 450 upright transitions from 150 simulation episodes, 150 epochs, seed 0.

The physical cup is 6.5 cm wide and 8 cm tall. The simulated cup is approximately 7 cm wide and 9.5 cm tall.

## Results

Prediction errors on **84 upright transitions from 28 episodes**:

| Predictor | Position error | Angle error |
|---|---:|---:|
| No motion | 37.55 mm | 22.10° |
| Half-travel, no rotation | 20.48 mm | 22.10° |
| Learned model | **14.60 mm** | **17.09°** |

The learned model reduced position error by 29% and angle error by 23% against the analytical baseline.

Task results on **50 matching starting conditions**, seed 456:

| Outcome | Learned | Analytical |
|---|---:|---:|
| Complete slide-and-orient task | **49/50** | 45/50 |
| Reached target area | 50/50 | 50/50 |
| Stayed upright | 50/50 | 49/50 |
| Average pushes | 9.14 | **8.28** |

Both methods use the same plan, 96 candidate proposals and up to 12 pushes per stage. Success means the cup centre is in a 10 × 10 cm area, handle angle is within 30° of the goal, and the cup remains upright.

The observed success advantage is uncertain: paired exact test p = 0.219. These trials cover small starting-pose changes around one task. In the earlier three-run pilot, the learned controller succeeded once and the baseline three times.

[Prediction metrics](outputs/elevated_prediction_metrics.json) · [Learned trials](outputs/learned_50.json) · [Baseline trials](outputs/baseline_50.json)

## What didn't work

The cup still rotates during sliding: average drift was 44° for the learned controller. Keeping translation and rotation separate remains unresolved. Early arm geometry also caused unintended cup movement; I raised the links and regenerated the training data.

The tipping and falling videos are failure examples, not training for a safety model. There is no physical robot test, obstacle handling or human reach estimation. Only one task and one trained model were evaluated.

## Run

Use Python 3.10+ from the repository root. Training runs on CPU.

```bash
python -m pip install -e .
python -m within_reach.retarget
python -m within_reach.sequence_demo \
  --plan outputs/human_goals/pushing_thenrotating_plan.json \
  --model outputs/sim_elevated.pt --episodes 1 --seed 456 \
  --video outputs/submission_demo.mp4 --out outputs/submission_demo.json
```

To repeat the 50-run comparison, use `--episodes 50`, omit `--video`, and run again without `--model` for the analytical baseline. Keep the same seed and plan.

To collect fresh data, retrain and evaluate:

```bash
python -m within_reach.collect --episodes 300 --pushes 3 --seed 42 --out data/processed/repro.jsonl
python -m within_reach.train --sim data/processed/repro.jsonl --mode sim --sim-episodes 150 --seed 0 --out outputs/repro.pt
python -m within_reach.evaluate --data data/processed/repro.jsonl --models outputs/repro.pt --split test --out outputs/repro_metrics.json
```

Tested with Python 3.10.20, PyTorch 2.11.0, MuJoCo 3.6.0, NumPy 2.2.6 and OpenCV 4.13.0.92 on Apple Silicon. Exact outcomes may vary by version.

Recordings are in `data/raw/`; reviewed annotations and plans in `outputs/human_video_review/` and `outputs/human_goals/`. The main code is `simulator.py`, `model.py`, `train.py`, `retarget.py` and `sequence_demo.py` inside `within_reach/`.

The recordings, annotations, final checkpoint, simulator dataset, pilot results and 50-run results are included in this folder. `MANIFEST.sha256` lists their checksums. The local `.gitignore` excludes caches and new reproduction outputs, not the supplied experiment artifacts.
