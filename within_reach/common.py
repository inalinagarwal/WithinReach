"""Shared data schema, episode splits and geometry (metres and radians)."""
import hashlib
import json
from pathlib import Path

import numpy as np


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pose(x, y, yaw):
    return np.array([x, y, np.cos(yaw), np.sin(yaw)], dtype=np.float64)


def yaw(state):
    return np.arctan2(state[3], state[2])


def split_for(episode):
    # All pushes from a video/episode share a split, preventing frame leakage.
    value = int(hashlib.sha256(episode.encode()).hexdigest()[:8], 16) % 10
    return "test" if value == 0 else "val" if value == 1 else "train"


def validate(record):
    for key, size in (("before", 4), ("after", 4), ("action", 5)):
        values = np.asarray(record[key], dtype=float)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {key}: expected {size} finite values")
    if record["action"][4] <= 0:
        raise ValueError("Push duration must be positive")
    for key in ("before", "after"):
        if not np.isclose(np.linalg.norm(record[key][2:4]), 1, atol=1e-3):
            raise ValueError(f"{key} angle must be encoded as unit cos/sin")
    if record["source"] not in ("human", "sim"):
        raise ValueError("source must be human or sim")
    if record["split"] not in ("train", "val", "test"):
        raise ValueError("Unknown split")
    return record


def read_records(path):
    with open(path) as stream:
        return [validate(json.loads(line)) for line in stream if line.strip()]


def write_record(stream, record):
    stream.write(json.dumps(validate(record)) + "\n")
    stream.flush()


def features(record):
    return np.r_[record["before"], record["action"]].astype(np.float32)


def target(record):
    before, after = np.asarray(record["before"]), np.asarray(record["after"])
    return np.array([*(after[:2] - before[:2]), wrap(yaw(after) - yaw(before))], dtype=np.float32)


def advance(state, delta):
    return pose(state[0] + delta[0], state[1] + delta[1], yaw(state) + delta[2])


def placement_cost(state, goal, angle_weight=0.04):
    return float(np.linalg.norm(np.asarray(state)[:2] - goal[:2]) +
                 angle_weight * abs(wrap(yaw(state) - goal[2])))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
