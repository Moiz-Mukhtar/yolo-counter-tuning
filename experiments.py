"""
Week 6: Improving Results
Change one thing, measure, compare.

Runs my Week 5 counter (line IN/OUT + restricted zone) on the same video with different
settings and measures, for each run:
  - IN, OUT and zone counts, compared with my manual count
  - count error = |model count - manual count| / manual count
  - FPS (frames processed per second, detection + tracking)

    python experiments.py              run every experiment that is not in results.csv yet
    python experiments.py baseline     run only the experiments named
    python experiments.py --table      only rebuild results.md from results.csv

Results go to results.csv (one row per run) and results.md (the before and after table).
"""

import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

VIDEO = "input.mp4"
SETUP_JSON = "setup.json"              # the same line and zone as Week 5
RESULTS_CSV = "results.csv"
RESULTS_MD = "results.md"

# Manual count from Week 5 (checked frame by frame)
MANUAL = {"in": 7, "out": 11, "zone": 4}
WARMUP_FRAMES = 10                     # not timed: the first frames include one-time model loading

# Week 5 settings = the baseline. Every experiment changes ONE thing from here.
BASELINE = dict(model="yolo11n.pt", imgsz=640, conf=None, iou=0.7, tracker="bytetrack.yaml")
# conf=None means the Ultralytics tracking default: boxes down to 0.1 go to the tracker

EXPERIMENTS = {
    # name:            (what changed,                    settings changed from the baseline)
    "baseline":        ("Baseline (Week 5 settings)",    {}),
    # Confidence
    "conf_0.25":       ("conf 0.25",                     {"conf": 0.25}),
    "conf_0.4":        ("conf 0.4",                      {"conf": 0.4}),
    # NMS overlap
    "iou_0.5":         ("NMS iou 0.5",                   {"iou": 0.5}),
    "iou_0.85":        ("NMS iou 0.85",                  {"iou": 0.85}),
    # Image size
    "imgsz_960":       ("imgsz 960",                     {"imgsz": 960}),
    "imgsz_1280":      ("imgsz 1280",                    {"imgsz": 1280}),
    # Model size
    "yolo11s":         ("yolo11s",                       {"model": "yolo11s.pt"}),
    "yolo11m":         ("yolo11m",                       {"model": "yolo11m.pt"}),
    # Tracker settings (copies of bytetrack.yaml with one value changed, in trackers/)
    "track_buffer_60": ("track_buffer 60",               {"tracker": "trackers/bytetrack_buffer60.yaml"}),
    "match_thresh_0.9":("match_thresh 0.9",              {"tracker": "trackers/bytetrack_match09.yaml"}),
    "fuse_score_off":  ("fuse_score off",                {"tracker": "trackers/bytetrack_fuse_off.yaml"}),
    # Combining what helped (more than one change from the baseline)
    "tracker_best":    ("match_thresh 0.9 + fuse_score off", {"tracker": "trackers/bytetrack_best.yaml"}),
    "tracker_best_11s":("tracker fixes + yolo11s",       {"tracker": "trackers/bytetrack_best.yaml", "model": "yolo11s.pt"}),
    "tracker_best_11m":("tracker fixes + yolo11m",       {"tracker": "trackers/bytetrack_best.yaml", "model": "yolo11m.pt"}),
    "tracker_best_960":("tracker fixes + imgsz 960",     {"tracker": "trackers/bytetrack_best.yaml", "imgsz": 960}),
}

FIELDS = ["name", "change", "model", "imgsz", "conf", "iou", "tracker",
          "in", "out", "zone", "zone_peak", "err_in", "err_out", "err_zone", "count_error", "fps",
          "unique_ids"]


def count_error(model_count, manual_count):
    return abs(model_count - manual_count) / manual_count


def run(settings):
    """Week 5 counting logic, without drawing, on the whole video. Returns counts and FPS."""
    with open(SETUP_JSON) as f:
        setup = json.load(f)
    line = sv.LineZone(start=sv.Point(*setup["line"][0]), end=sv.Point(*setup["line"][1]))
    zone = sv.PolygonZone(polygon=np.array(setup["zone"], dtype=np.int32),
                          triggering_anchors=(sv.Position.BOTTOM_CENTER,))

    model = YOLO(settings["model"])                 # a fresh model = a fresh tracker
    kwargs = dict(persist=True, classes=[0], imgsz=settings["imgsz"], iou=settings["iou"],
                  tracker=settings["tracker"], verbose=False)
    if settings["conf"] is not None:
        kwargs["conf"] = settings["conf"]

    cap = cv2.VideoCapture(VIDEO)
    zone_ids, all_ids = set(), set()
    zone_peak = 0                                   # most people in the zone at the same moment
    frames, timed, busy = 0, 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        t0 = time.perf_counter()
        result = model.track(frame, **kwargs)[0]
        detections = sv.Detections.from_ultralytics(result)
        if detections.tracker_id is None:
            detections = sv.Detections.empty()
            detections.tracker_id = np.empty(0, dtype=int)
        line.trigger(detections)
        inside = zone.trigger(detections)
        if frames > WARMUP_FRAMES:                  # the first frames include one-time model loading
            busy += time.perf_counter() - t0
            timed += 1
        all_ids.update(int(t) for t in detections.tracker_id)
        zone_ids.update(int(t) for t, z in zip(detections.tracker_id, inside) if z)
        zone_peak = max(zone_peak, int(np.sum(inside)))
    cap.release()
    return {"in": line.in_count, "out": line.out_count, "zone": len(zone_ids),
            "zone_peak": zone_peak, "fps": timed / busy, "unique_ids": len(all_ids)}


def load_results():
    if not os.path.exists(RESULTS_CSV):
        return {}
    with open(RESULTS_CSV) as f:
        return {r["name"]: r for r in csv.DictReader(f)}


def save_results(rows):
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for name in rows:
            w.writerow(rows[name])


def pct(x):
    return f"{100 * float(x):.1f}%"


def write_table(rows):
    """results.md: the before and after table, in the order of EXPERIMENTS."""
    base = rows.get("baseline")
    lines = ["| Version | IN (7) | OUT (11) | Zone IDs (4) | Count error | FPS | Better than baseline? "
             "| Most in zone at once (4) | Unique IDs |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in list(EXPERIMENTS) + [n for n in rows if n not in EXPERIMENTS]:
        if name not in rows:
            continue
        r = rows[name]
        if base is None or name == "baseline":
            verdict = "–"
        else:
            d = float(r["count_error"]) - float(base["count_error"])
            verdict = "Yes" if d < -1e-9 else ("No (worse)" if d > 1e-9 else "No change")
        lines.append(f"| {r['change']} | {r['in']} | {r['out']} | {r['zone']} | "
                     f"{pct(r['count_error'])} | {float(r['fps']):.1f} | {verdict} | "
                     f"{r.get('zone_peak', '')} | {r['unique_ids']} |")
    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    rows = load_results()
    if "--table" in sys.argv:
        write_table(rows)
        return
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or [n for n in EXPERIMENTS if n not in rows]
    for name in names:
        change, diff = EXPERIMENTS[name]
        settings = {**BASELINE, **diff}
        print(f"[RUN] {name}: {change} ...", flush=True)
        res = run(settings)
        e_in = count_error(res["in"], MANUAL["in"])
        e_out = count_error(res["out"], MANUAL["out"])
        e_zone = count_error(res["zone"], MANUAL["zone"])
        # One number per run: all counting mistakes together, divided by everything there was to count
        total_err = (abs(res["in"] - MANUAL["in"]) + abs(res["out"] - MANUAL["out"])
                     + abs(res["zone"] - MANUAL["zone"])) / sum(MANUAL.values())
        rows[name] = {"name": name, "change": change, "model": settings["model"],
                      "imgsz": settings["imgsz"], "conf": settings["conf"] or "default (0.1)",
                      "iou": settings["iou"], "tracker": settings["tracker"],
                      "in": res["in"], "out": res["out"], "zone": res["zone"],
                      "zone_peak": res["zone_peak"],
                      "err_in": f"{e_in:.4f}", "err_out": f"{e_out:.4f}", "err_zone": f"{e_zone:.4f}",
                      "count_error": f"{total_err:.4f}", "fps": f"{res['fps']:.2f}",
                      "unique_ids": res["unique_ids"]}
        save_results(rows)
        print(f"      IN {res['in']}  OUT {res['out']}  zone {res['zone']}  "
              f"count error {pct(total_err)}  FPS {res['fps']:.1f}", flush=True)
    write_table(rows)


if __name__ == "__main__":
    main()
