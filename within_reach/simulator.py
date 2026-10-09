"""Physical cup and a toy 3-DOF articulated pusher; no object teleporting during pushes."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .common import pose, yaw, wrap


class CupSimulator:
    physics_revision = "elevated_arm_v1"
    def __init__(self):
        root = ET.parse(Path(__file__).parent / "assets/scene.xml").getroot()
        cup = root.find(".//body[@name='cup']")
        for i in range(20):
            theta = i * 2 * np.pi / 20
            ET.SubElement(cup, "geom", name=f"wall_{i}", type="box",
                          pos=f"{0.033 * np.cos(theta)} {0.033 * np.sin(theta)} 0.046",
                          size="0.003 0.0055 0.046", euler=f"0 0 {theta}",
                          rgba="0.1 0.65 0.7 1")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.cup_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cup_free")
        self.cup_q = self.model.jnt_qposadr[joint]
        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        self.renderer = None
        self.video = None
        self.frame_count = 0

    def ik(self, xy):
        dx, dy = np.asarray(xy) - np.array([-0.45, 0])
        c2 = (dx * dx + dy * dy - 2 * 0.4**2) / (2 * 0.4**2)
        if abs(c2) > 1:
            raise ValueError(f"Pusher target outside arm reach: {xy}")
        q2 = np.arccos(np.clip(c2, -1, 1))
        q1 = np.arctan2(dy, dx) - np.arctan2(0.4 * np.sin(q2), 0.4 + 0.4 * np.cos(q2))
        return np.array([q1, q2])

    def reset(self, x=0, y=0, angle=0, friction=0.35):
        mujoco.mj_resetData(self.model, self.data)
        self.model.geom_friction[:, 0] = friction
        self.data.qpos[:2] = self.ik([-0.15, -0.25])
        self.data.qpos[2] = 0.15
        q = self.cup_q
        self.data.qpos[q:q+3] = [x, y, 0.004]
        self.data.qpos[q+3:q+7] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        self.data.ctrl[:] = self.data.qpos[:3]
        mujoco.mj_forward(self.model, self.data)
        self.hold(0.3)

    def state(self):
        p = self.data.xpos[self.cup_id]
        rotation = self.data.xmat[self.cup_id].reshape(3, 3)
        angle = np.arctan2(rotation[1, 0], rotation[0, 0])
        return pose(p[0], p[1], angle)

    def upright(self):
        return bool(self.data.xmat[self.cup_id].reshape(3, 3)[2, 2] > np.cos(np.deg2rad(20)))

    def step(self):
        mujoco.mj_step(self.model, self.data)
        self.frame_count += 1
        if self.video is not None and self.frame_count % 12 == 0:
            self.renderer.update_scene(self.data, camera="overview")
            self.video.write(self.renderer.render()[:, :, ::-1])

    def hold(self, seconds):
        for _ in range(max(1, int(seconds / self.model.opt.timestep))):
            self.step()

    def move(self, xy, lift, seconds):
        start = self.data.site_xpos[self.tip_id, :2].copy()
        start_lift = float(self.data.qpos[2])
        steps = max(1, int(seconds / self.model.opt.timestep))
        for i in range(steps):
            t = (i + 1) / steps
            self.data.ctrl[:2] = self.ik(start * (1 - t) + np.asarray(xy) * t)
            self.data.ctrl[2] = start_lift * (1 - t) + lift * t
            self.step()
        self.hold(0.12)

    def push(self, action):
        # action: contact offset x/y, commanded travel x/y, duration seconds.
        action = np.asarray(action, dtype=float)
        start = self.state()[:2] + action[:2]
        self.move(self.data.site_xpos[self.tip_id, :2].copy(), 0.15, 0.2)
        self.move(start, 0.15, 0.45)
        self.move(start, 0, 0.25)
        self.hold(0.15)
        before = self.state()
        # Store measured relative starting contact, including actuator tracking error.
        measured_action = action.copy()
        measured_action[:2] = self.data.site_xpos[self.tip_id, :2] - before[:2]
        commanded_end = self.data.site_xpos[self.tip_id, :2].copy() + action[2:4]
        self.move(commanded_end, 0, action[4])
        self.move(self.data.site_xpos[self.tip_id, :2].copy(), 0.15, 0.2)
        self.hold(0.3)
        return before, measured_action, self.state(), self.upright()

    def start_video(self, path):
        import cv2
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 480))
        if not self.video.isOpened():
            raise RuntimeError(f"Could not create video: {path}")

    def close(self):
        if self.video is not None:
            self.video.release()
        if self.renderer is not None:
            self.renderer.close()


def sample_actions(state, rng, count=24, goal=None):
    actions = []
    for _ in range(count):
        contact_angle = rng.uniform(-np.pi, np.pi)
        if goal is not None and rng.random() < 0.65:
            travel_angle = np.arctan2(goal[1] - state[1], goal[0] - state[0]) + rng.normal(0, 0.45)
            contact_angle = wrap(travel_angle + np.pi + rng.normal(0, 0.35))
        else:
            travel_angle = contact_angle + np.pi + rng.uniform(-0.7, 0.7)
        # Allow extra clearance near the handle.
        near_handle = abs(wrap(contact_angle - yaw(state))) < 0.5
        radius = 0.083 if near_handle else 0.054
        contact = radius * np.array([np.cos(contact_angle), np.sin(contact_angle)])
        travel = rng.uniform(0.025, 0.085) * np.array([np.cos(travel_angle), np.sin(travel_angle)])
        actions.append(np.r_[contact, travel, rng.uniform(0.4, 0.8)])
    return np.asarray(actions)


def stage_actions(state, rng, count, goal, mode):
    """Centred approach for sliding; tangent/inward contacts for turning.

    These are candidate generators, not guaranteed no-slip/no-rotation skills.
    The learned predictor still scores the resulting motions.
    """
    if mode not in ("slide", "orient"):
        return sample_actions(state, rng, count, goal)
    actions = []
    toward = np.asarray(goal[:2]) - np.asarray(state[:2])
    distance = np.linalg.norm(toward)
    for i in range(count):
        # Some orientation candidates restore position if the cup drifts.
        centre_push = mode == "slide" or (i % 4 == 0 and distance > 0.015)
        if centre_push:
            direction = np.arctan2(toward[1], toward[0]) + rng.uniform(-0.10, 0.10)
            travel_direction = np.array([np.cos(direction), np.sin(direction)])
            normal = -travel_direction
            contact_angle = np.arctan2(normal[1], normal[0])
            near_handle = abs(wrap(contact_angle - yaw(state))) < 0.5
            radius = 0.083 if near_handle else 0.054
            contact = radius * normal
            # Travel and centre-to-contact vector are collinear (zero ideal moment).
            length = rng.uniform(0.025, min(0.075, max(0.026, distance + 0.012)))
            travel = length * travel_direction
        else:
            contact_angle = (yaw(state) + rng.uniform(-0.35, 0.35)
                             if i % 2 else rng.uniform(-np.pi, np.pi))
            normal = np.array([np.cos(contact_angle), np.sin(contact_angle)])
            near_handle = abs(wrap(contact_angle - yaw(state))) < 0.5
            radius = 0.083 if near_handle else 0.054
            contact = radius * normal
            tangent = np.array([-normal[1], normal[0]]) * rng.choice([-1, 1])
            direction = tangent - rng.uniform(0.35, 0.65) * normal
            direction /= np.linalg.norm(direction)
            travel = rng.uniform(0.025, 0.05) * direction
        actions.append(np.r_[contact, travel, rng.uniform(0.5, 0.9)])
    return np.asarray(actions)
