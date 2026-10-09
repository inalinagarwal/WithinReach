# Video annotation review

All five original MOV files were inspected. The three upright manipulation videos contain readable cup/handle motion. The other two show tipping and leaving the table.

`video_annotations_pixels.json` records approximate frame intervals and manually reviewed landmarks in the original 1920x1080 pixel coordinates. The `*_review.jpg` files show the selected annotated frames. Green is rim centre, cyan is estimated base projection, magenta is handle, yellow is approximate hand contact. The magenta arrow is an image-space orientation cue, not a calibrated physical yaw angle.

## Usable content

- `pushing_thenrotating.MOV`: sliding at roughly 1.83–3.67 s, rotation at 4.43–6.00 s, final visible pose from 6.73 s.
- `pushing_rotatinginotherdirection.MOV`: sliding at roughly 1.60–3.30 s, rotation at 5.23–5.97 s, final visible pose from 6.57 s.
- `pushing_and_rotating_sametime.MOV`: coupled motion at roughly 1.67–2.67 s and 4.33–5.33 s; additional final adjustment. Reserve this entire video for an illustrative held-out check.
- `pushing_off_table.MOV`: cup at the edge around 4.53 s and out of view by 5.10 s.
- `tipping_on_table.MOV`: upright at 3.87 s, tipping by 4.53 s, lying on its side by 5.17 s.

These intervals are selected from sampled frames, not precise onset detection. Review before using them as ground truth.

## Calibration limitation

The camera moves, the perspective is oblique, and a measured four-corner tabletop rectangle is not visible. The existing fixed-homography annotation workflow cannot directly produce reliable metric transitions from these recordings. Cup dimensions can provide a scale reference, but diameter alone does not solve camera motion and perspective. Do not feed raw pixel distances into a model trained on metres.

Two defensible next steps are: record a short fixed-camera calibrated clip if feasible; or use these videos to derive explicit qualitative skill/orientation goals and retarget those goals in simulation, disclosing that the trajectories are not metrically reconstructed. The second option requires a goal/skill bridge; the current pixel annotations alone do not train the existing dynamics model.

The two failure videos cannot train the current planar upright dynamics predictor without changing its state/action labels and outputs. Include them as qualitative failure analysis, not evidence of a learned safety model.
