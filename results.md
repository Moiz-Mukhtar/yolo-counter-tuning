| Version | IN (7) | OUT (11) | Zone IDs (4) | Count error | FPS | Better than baseline? | Most in zone at once (4) | Unique IDs |
|---|---|---|---|---|---|---|---|---|
| Baseline (Week 5 settings) | 7 | 9 | 3 | 13.6% | 16.6 | – | 3 | 67 |
| conf 0.25 | 7 | 9 | 3 | 13.6% | 16.9 | No change | 3 | 70 |
| conf 0.4 | 7 | 10 | 2 | 13.6% | 17.2 | No change | 1 | 52 |
| NMS iou 0.5 | 7 | 9 | 3 | 13.6% | 17.7 | No change | 3 | 64 |
| NMS iou 0.85 | 7 | 9 | 3 | 13.6% | 16.9 | No change | 3 | 96 |
| imgsz 960 | 7 | 10 | 9 | 27.3% | 8.5 | No (worse) | 6 | 103 |
| imgsz 1280 | 7 | 10 | 17 | 63.6% | 4.8 | No (worse) | 6 | 132 |
| yolo11s | 7 | 10 | 15 | 54.5% | 6.7 | No (worse) | 7 | 103 |
| yolo11m | 7 | 10 | 7 | 18.2% | 2.6 | No (worse) | 4 | 64 |
| track_buffer 60 | 7 | 9 | 3 | 13.6% | 16.4 | No change | 3 | 67 |
| match_thresh 0.9 | 7 | 10 | 3 | 9.1% | 16.9 | Yes | 3 | 60 |
| fuse_score off | 7 | 10 | 3 | 9.1% | 16.4 | Yes | 3 | 67 |
| match_thresh 0.9 + fuse_score off | 8 | 10 | 3 | 13.6% | 17.1 | No change | 3 | 66 |
| tracker fixes + yolo11s | 6 | 9 | 7 | 27.3% | 7.1 | No (worse) | 6 | 94 |
| tracker fixes + yolo11m | 7 | 10 | 7 | 18.2% | 2.6 | No (worse) | 5 | 63 |
| tracker fixes + imgsz 960 | 7 | 11 | 7 | 13.6% | 8.3 | No change | 6 | 91 |
