# Week 6: Improving Results

**Change one thing, measure, compare.** I took my Week 5 counter ([yolo-people-counter](https://github.com/Moiz-Mukhtar/yolo-people-counter): line IN/OUT and restricted zone) and changed one setting at a time on the same video, measuring the count error and FPS after every change.

## How to run

```bash
git clone https://github.com/Moiz-Mukhtar/yolo-counter-tuning.git
cd yolo-counter-tuning
pip install -r requirements.txt

python experiments.py --table            # show the results table from results.csv (instant)
python experiments.py baseline imgsz_960 # rerun only the experiments named (names are in experiments.py)
```

To rerun **everything** (about 15–30 minutes on a laptop CPU), delete `results.csv` first, then run `python experiments.py`. The script skips any experiment already in `results.csv`, so an interrupted run carries on where it stopped. Run it from inside the folder, because it uses `input.mp4`, `setup.json` and `trackers/` from there.

| File | What it is |
|---|---|
| `experiments.py` | Runs the Week 5 counter with each setting and measures count error and FPS |
| `input.mp4`, `setup.json` | The Week 5 video, and the same line and zone as Week 5 |
| `trackers/` | Copies of `bytetrack.yaml`, each with one value changed |
| `results.csv` | Every number from every run |
| `results.md` | The results table, made by the script |

The model weights (`yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`) download automatically the first time. The results were made with Ultralytics 8.4.194 and Supervision 0.31.5; other versions may give slightly different counts.

## Setup

- **Video:** my hardest video, the Week 5 hall clip (13.6 s, 1920×1080, 341 frames). It has a crowd, a tight group of 4 people in the zone, and a missed crossing caused by an ID switch.
- **Manual count** (checked frame by frame in Week 5): **IN 7, OUT 11, zone 4 people.**
- **Same line and zone for every run** (`setup.json` from Week 5).
- **Count error** (slide 10) = |model count − manual count| ÷ manual count. For one number per run, I add up the mistakes in all three counts and divide by everything there was to count:
  `(|IN − 7| + |OUT − 11| + |zone − 4|) ÷ 22`
- **FPS** = frames per second for detection + tracking + counting. It was measured on a 2-core cloud CPU, so a laptop will be faster, but every run used the same machine, so they compare fairly. The first 10 frames are not timed because they include loading the model.

## Before and after

**Baseline = Week 5 settings:** `yolo11n`, imgsz 640, conf default (0.1 for tracking), NMS iou 0.7, default `bytetrack.yaml`.

| Version (one change from the baseline) | IN (7) | OUT (11) | Zone (4) | Count error | FPS | Keep? |
|---|---|---|---|---|---|---|
| **Baseline** | 7 | 9 | 3 | **13.6%** | 16.6 | – |
| conf 0.25 | 7 | 9 | 3 | 13.6% | 16.9 | No: no change |
| conf 0.4 | 7 | 10 | 2 | 13.6% | 17.2 | No: misses the group |
| NMS iou 0.5 | 7 | 9 | 3 | 13.6% | 17.7 | No: no change |
| NMS iou 0.85 | 7 | 9 | 3 | 13.6% | 16.9 | No: no change |
| imgsz 960 | 7 | 10 | 9 | 27.3% | 8.5 | No: worse and 2× slower |
| imgsz 1280 | 7 | 10 | 17 | 63.6% | 4.8 | No: much worse and 3.5× slower |
| yolo11s | 7 | 10 | 15 | 54.5% | 6.7 | No: worse and 2.5× slower |
| yolo11m | 7 | 10 | 7 | 18.2% | 2.6 | No: worse and 6× slower |
| track_buffer 60 | 7 | 9 | 3 | 13.6% | 16.4 | No: no change |
| **match_thresh 0.9** | 7 | **10** | 3 | **9.1%** | 16.9 | **Yes** |
| **fuse_score off** | 7 | **10** | 3 | **9.1%** | 16.4 | **Yes (same effect)** |

**Best: match_thresh 0.9.** Count error went from **13.6% to 9.1%**, with no loss in speed.

### Combining changes

| Version | IN (7) | OUT (11) | Zone (4) | Count error | FPS |
|---|---|---|---|---|---|
| match_thresh 0.9 + fuse_score off | **8** | 10 | 3 | 13.6% | 17.1 |
| tracker fixes + yolo11s | 6 | 9 | 7 | 27.3% | 7.1 |
| tracker fixes + yolo11m | 7 | 10 | 7 | 18.2% | 2.6 |
| tracker fixes + imgsz 960 | 7 | **11** | 7 | 13.6% | 8.3 |

### Ablation (slide 4)

Starting from both tracker changes together, I removed one at a time:

| Setup | Count error |
|---|---|
| match_thresh 0.9 + fuse_score off | 13.6% |
| remove fuse_score off (match_thresh 0.9 only) | **9.1%** |
| remove match_thresh 0.9 (fuse_score off only) | 9.1% |
| remove both (baseline) | 13.6% |

Each change helps **on its own**, but together they are worse: they fix the same mistake, and doing both makes matching too loose (see below). So I keep only **match_thresh 0.9**.

## What helped most

**The tracker settings helped most, not a bigger model.**

- **match_thresh 0.9** (or **fuse_score off**) rescued the crossing I found missing in Week 5. The man with the grey jacket (ID 9) got a new ID while standing on the line, so his crossing was never counted. With either change, ByteTrack accepts a match it rejected before and keeps his ID, so the crossing is counted at frame 84. OUT went from 9 to 10, and count error from 13.6% to 9.1%, at the same speed.
- **Both together went too far.** They also rescued that crossing, but added a false IN at frame 320, where several people walk close together (IN = 8). Most likely the looser matching swapped IDs between two of them, which made one ID look like it crossed the line.

**Bigger models and larger images made it worse, which surprised me.** They do see more people, and the line count got better: OUT went from 9 to 10 with each of them (and to 11 with the tracker fixes at imgsz 960). But in the tight group of 4, they detect each person on and off, and the tracker gives them new IDs over and over. The zone counts **IDs**, not people, so every new ID counts as another person entering: 7 to 17 instead of 4. With imgsz 960, imgsz 1280 and yolo11s, the total unique IDs in the video also rose from 67 to 103–132, which shows the extra ID switches. On top of that, they ran 2–6× slower. Better detection only helps if the tracker can keep the IDs.

**Confidence, NMS iou and track_buffer did not change the count error** on this video:
- conf 0.25 changed nothing. conf 0.4 fixed one crossing (OUT 10) but removed weak boxes in the group, so the zone dropped to 2 people. The two effects cancel out, and the error stayed at 13.6%.
- NMS iou changes how overlapping boxes are merged. It changed the number of unique IDs (64–96) but not the counts.
- track_buffer 60 changed nothing. The ID switch on the line is not caused by a person being forgotten too early. The tracker still remembers the old ID, but does not match the person back to it. That is why match_thresh and fuse_score help, and track_buffer does not.

**Speed:** no setting reached real-time (25–30 FPS, slide 7) on this 2-core CPU. The baseline and the tracker changes ran at about 17 FPS, which a laptop CPU or a GPU should push higher. The bigger models and images ran at 2.6–8.5 FPS, far from real-time.

**Best result for the line alone:** tracker fixes + imgsz 960 counted IN and OUT exactly right (7 and 11). But it's half as fast and over-counts the zone, so it's not the best overall.

## Next steps

- **Count people in the zone, not IDs:** for example, the most people in the zone at the same moment, which ID switches cannot inflate. `results.csv` records it (`zone_peak`): the baseline only ever saw 3 at once, yolo11m saw exactly **4**, and yolo11s / imgsz 960 saw 6–7, so they also add some duplicate boxes. A bigger model plus this zone count could fix the zone without the ID-switch problem.
- **Fine-tuning** (slide 9) on labeled frames of tight groups seen from above, so the detector separates people standing close together.
- **Check on a second video**, so the settings are not just tuned to this one clip.
