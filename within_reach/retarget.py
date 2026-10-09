"""Turn reviewed human video landmarks into explicitly coarse simulator goals.

This is qualitative retargeting, not metric trajectory reconstruction or VLA training.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', default='outputs/human_video_review/video_annotations_pixels.json')
    p.add_argument('--out-dir', default='outputs/human_goals')
    p.add_argument('--cup-diameter', type=float, default=0.065)
    p.add_argument('--cup-height', type=float, default=0.080)
    p.add_argument('--distance-in-diameters', type=float, default=1.5,
                   help='Designed simulator displacement, NOT a measured video displacement')
    args = p.parse_args()
    if min(args.cup_diameter, args.cup_height, args.distance_in_diameters) <= 0:
        p.error('Dimensions and designed distance must be positive')
    source = Path(args.annotations)
    payload = json.loads(source.read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    distance = args.cup_diameter * args.distance_in_diameters
    if distance > 0.18:
        p.error('Designed displacement must be <=0.18 m for this workspace')
    manifest = []
    for video in payload['videos']:
        if video['outcome'] != 'upright':
            print(f"Excluded {video['video']}: {video['outcome']} (no failure-trained model)")
            continue
        # Coarse screen-side of handle relative to cup rim. Avoid interpreting the
        # vertical projection as calibrated world yaw: rim/handle have different heights.
        usable = [f for f in video['frames'] if f['landmarks']['rim_center_px'] is not None
                  and f['landmarks']['handle_point_px'] is not None]
        first, last = usable[0], usable[-1]
        first_x = first['landmarks']['rim_center_px'][0]
        last_x = last['landmarks']['rim_center_px'][0]
        dx = last_x - first_x
        if abs(dx) < 40:
            raise ValueError(f"Ambiguous gross slide direction for {video['video']}")
        handle_dx = last['landmarks']['handle_point_px'][0] - last_x
        if abs(handle_dx) < 30:
            raise ValueError(f"Ambiguous final handle screen-side for {video['video']}")
        direction = -1 if dx < 0 else 1
        final_yaw = math.pi if handle_dx < 0 else 0.0
        target = [direction * distance, 0.0, final_yaw]
        sequential = not any('coupled' in s['label'] for s in video['segments'])
        provenance = dict(video=video['video'], episode=video['episode'],
            annotations=str(source), annotations_sha256=digest,
            first_frame=first['frame'], final_frame=last['frame'],
            gross_slide_screen_direction='left' if direction < 0 else 'right',
            final_handle_screen_side='left' if handle_dx < 0 else 'right',
            confidence='coarse manually reviewed screen cues; camera motion is a confound',
            transfer='image-left/right assigned to simulator -x/+x; no metric reconstruction',
            designed_slide_distance_m=distance, human_cup_diameter_m=args.cup_diameter,
            human_cup_height_m=args.cup_height,
            limitations=['Camera motion prevents calibrated physical translation/yaw recovery.',
                        'Simulator displacement is designed at 1.5 cup diameters by default.',
                        'Screen-side orientation is reduced to left/right, losing depth direction.',
                        'No demonstrated reachable region or clinically validated accessibility.'])
        goal = dict(x=target[0], y=target[1], yaw=target[2], half_width=0.05,
                    half_depth=0.05, angle_tolerance_deg=30,
                    source='coarse applicant-video goal retargeting', provenance=provenance)
        stages = []
        if sequential:
            stages.append(dict(name='slide', goal=target, require_angle=False))
            stages.append(dict(name='orient', goal=target, require_angle=True))
        else:
            stages.append(dict(name='slide_and_orient', goal=target, require_angle=True))
        # Robot starts on the opposite side of the target; handle starts image-right,
        # matching the reviewed first frames. Jitter is introduced by the runner.
        plan = dict(video=video['video'], source='coarse applicant-video skill sequence',
                    initial=[-direction * 0.08, 0, 0], stages=stages,
                    half_width=0.05, half_depth=0.05, angle_tolerance_deg=30,
                    provenance=provenance)
        stem = Path(video['video']).stem
        (out / f'{stem}_goal.json').write_text(json.dumps(goal, indent=2) + '\n')
        (out / f'{stem}_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        manifest.append(dict(video=video['video'], goal=f'{stem}_goal.json', plan=f'{stem}_plan.json'))
        print(f"{video['video']}: {len(stages)} stages, target x={target[0]:.4f}, handle="
              f"{provenance['final_handle_screen_side']}")
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {len(manifest)} plans to {out}. No human dynamics training has been performed.')


if __name__ == '__main__':
    main()
