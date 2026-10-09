"""Click tabletop corners, cup centre/handle and fingertip in before/after video frames."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .common import pose, split_for, write_record, read_records, save_json


WINDOW = "Within Reach annotation"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video")
    p.add_argument("--width", type=float, default=0.50, help="Measured workspace width, metres")
    p.add_argument("--depth", type=float, default=0.40, help="Measured workspace depth, metres")
    p.add_argument("--calibration", default="data/processed/calibration.json")
    p.add_argument("--out", default="data/processed/human.jsonl")
    p.add_argument("--split", choices=["train", "val", "test"], help="Override per-video split")
    p.add_argument("--episode", help="Unique video ID; default human-<filename stem>")
    p.add_argument("--preferred", action="store_true", help="Saved after states are preferred placements")
    args = p.parse_args()
    if args.width <= 0 or args.depth <= 0:
        p.error("Workspace dimensions must be positive")
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise ValueError(f"Could not open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 1:
        raise ValueError("Video must have valid frame count and FPS")
    ok, first = capture.read()
    if not ok:
        raise ValueError("Cannot read first frame")
    scale = min(1, 1200 / first.shape[1], 750 / first.shape[0])
    episode = args.episode or f"human-{Path(args.video).stem}"
    existing = read_records(args.out) if Path(args.out).exists() else []
    previous = [r for r in existing if r["episode"] == episode]
    split = args.split or (previous[0]["split"] if previous else split_for(episode))
    if previous and any(r["split"] != split for r in previous):
        raise ValueError("All annotations of one video must use the same split")
    next_step = max((r["step"] for r in previous), default=-1) + 1
    cv2.namedWindow(WINDOW)
    clicks = []

    def collect_corner(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append([x / scale, y / scale])

    calibration_path = Path(args.calibration)
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        if calibration["image_size"] != [first.shape[1], first.shape[0]]:
            raise ValueError("Video resolution changed. Use a new --calibration path.")
        H = np.asarray(calibration["homography"], dtype=float)
        print("Reusing calibration. This is valid only if camera/table have not moved.")
    else:
        cv2.setMouseCallback(WINDOW, collect_corner)
        print("Click workspace corners in IMAGE order: top-left, top-right, bottom-right, bottom-left.")
        while True:
            frame = cv2.resize(first, None, fx=scale, fy=scale)
            for i, point in enumerate(clicks):
                pixel = tuple(np.round(np.asarray(point) * scale).astype(int))
                cv2.circle(frame, pixel, 5, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "4 corners TL TR BR BL; Enter accept; r reset; q quit",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(30) & 255
            if key == ord("q"):
                capture.release()
                cv2.destroyAllWindows()
                return
            if key == ord("r"):
                clicks.clear()
            if key in (10, 13) and len(clicks) == 4:
                break
        w, d = args.width / 2, args.depth / 2
        destination = np.float32([[-w, d], [w, d], [w, -d], [-w, -d]])
        H = cv2.getPerspectiveTransform(np.float32(clicks), destination)
        if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
            raise ValueError("Degenerate corner calibration; retry with distinct table corners")
        save_json(calibration_path, dict(homography=H.tolist(), width=args.width,
                  depth=args.depth, image_size=[first.shape[1], first.shape[0]],
                  corners=clicks, coordinates="x image-right, y image-up; origin workspace centre"))

    selected = {"before": None, "after": None}
    current_index = 0
    active = None

    def get_index():
        return cv2.getTrackbarPos("frame", WINDOW)

    def click_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and active is not None:
            item = selected[active]
            if item is not None and get_index() == item["frame"] and len(item["points"]) < 3:
                item["points"].append([x / scale, y / scale])

    cv2.createTrackbar("frame", WINDOW, 0, total - 1, lambda value: None)
    cv2.setMouseCallback(WINDOW, click_point)
    print(f"Episode {episode}; split={split}. All pushes from this video remain in that split.")
    print("Use slider or j/l (one frame), J/L (one second). b: before; a: after.")
    print("After b/a, click: 1 cup base centre, 2 handle direction, 3 fingertip centre.")
    print("s saves a push; r clears selection; q quits. Mark AFTER before fingertip lifts away.")

    def metric_points(points):
        return cv2.perspectiveTransform(np.float32(points).reshape(1, -1, 2), H)[0]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            current_index = get_index()
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_index)
            ok, original = capture.read()
            if not ok:
                raise ValueError(f"Could not read frame {current_index}")
            display = cv2.resize(original, None, fx=scale, fy=scale)
            for name, item in selected.items():
                if item is not None and item["frame"] == current_index:
                    for i, point in enumerate(item["points"]):
                        pixel = tuple(np.round(np.asarray(point) * scale).astype(int))
                        cv2.circle(display, pixel, 5, (0, 0, 255), -1)
                        cv2.putText(display, f"{name}:{i+1}", pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(display, f"Frame {current_index} | b before / a after / s save / q quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(30) & 255
            if key == ord("q"):
                break
            if key in (ord("j"), ord("l"), ord("J"), ord("L")):
                amount = int(fps) if key in (ord("J"), ord("L")) else 1
                amount *= -1 if key in (ord("j"), ord("J")) else 1
                cv2.setTrackbarPos("frame", WINDOW, int(np.clip(current_index + amount, 0, total - 1)))
            if key in (ord("b"), ord("a")):
                active = "before" if key == ord("b") else "after"
                selected[active] = dict(frame=current_index, points=[])
                print(f"Mark {active}: cup base centre, handle direction, fingertip centre")
            if key == ord("r"):
                selected = {"before": None, "after": None}
                active = None
            if key == ord("s"):
                if any(item is None or len(item["points"]) != 3 for item in selected.values()):
                    print("Select both frames and three points in each first.")
                    continue
                b, a = selected["before"], selected["after"]
                duration = (a["frame"] - b["frame"]) / fps
                if duration <= 0:
                    print("After must occur later than before.")
                    continue
                if any(r.get("before_frame") == b["frame"] and r.get("after_frame") == a["frame"] for r in previous):
                    print("That frame pair was already saved; choose a different push.")
                    continue
                bp, ap = metric_points(b["points"]), metric_points(a["points"])
                before = pose(*bp[0], np.arctan2(*(bp[1] - bp[0])[::-1]))
                after = pose(*ap[0], np.arctan2(*(ap[1] - ap[0])[::-1]))
                action = np.r_[bp[2] - bp[0], ap[2] - bp[2], duration]
                record = dict(source="human", episode=episode, split=split, step=next_step,
                    before=before.tolist(), after=after.tolist(), action=action.tolist(),
                    upright=True, preferred=args.preferred, video=str(Path(args.video)),
                    before_frame=b["frame"], after_frame=a["frame"],
                    calibration=str(calibration_path), points_before=b["points"], points_after=a["points"])
                with open(args.out, "a") as stream:
                    write_record(stream, record)
                previous.append(record)
                next_step += 1
                selected = {"before": None, "after": None}
                active = None
                print(f"Saved push {next_step}; duration={duration:.2f}s to {args.out}")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
