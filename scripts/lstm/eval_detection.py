"""
eval_detection.py  –  End-to-end check of shot detection + made/missed
=====================================================================
Runs the full pipeline (cached YOLO boxes → shot detector → predictor) on the
videos that have labeled shots and reports, per video and in total:

  found    labeled shots the detector matched (±FRAME_TOL frames overlap)
  missed   labeled shots it never detected
  false    detections with no labeled shot nearby
  acc      made/missed accuracy on the matched shots

Labels come from the labeler CSVs and the app (app_labeled.csv). Note: the app
labels only contain shots an earlier detector found, so "false" on that video
can include real shots nobody labeled.

    python scripts/lstm/eval_detection.py                          # current settings
    python scripts/lstm/eval_detection.py --ball-conf 0.75 --conf 0.75   # old behaviour
    python scripts/lstm/eval_detection.py --sweep                   # threshold grid

The first run per video fills the detection cache (YOLO on the GPU); later runs
take seconds.
"""

import io
import sys
import json
import time
import argparse
import contextlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RAW_VIDEO_DIR, ANNOTATIONS_DIR, EVAL_DIR, PREDICTIONS_DIR, YOLO_MODEL, RIM_RULE_CONFIG  # noqa: E402
import lstm_infer as infer  # noqa: E402
from lstm_eval import match_shots  # noqa: E402

FRAME_TOL = 5

# (video file, labels csv, source_video filter or None)
EVAL_SETS = [
    ("20250711_171647.mp4", ANNOTATIONS_DIR / "basketball_shot_data_labeled_3103.csv", None),
    ("app_20261017_182226_f30ea479-39d4-4d77-8e9c-29a54a7480a6.mp4",
     ANNOTATIONS_DIR / "app_labeled.csv", "app_20261017_182226_f30ea479-39d4-4d77-8e9c-29a54a7480a6.mp4"),
]


def load_labels(csv_path: Path, source_video: str | None) -> pd.DataFrame:
    head = csv_path.read_text(encoding="utf-8", errors="ignore")[:500]
    df = pd.read_csv(csv_path, sep=';' if head.count(';') > head.count(',') else ',')
    if source_video is not None:
        df = df[df['source_video'] == source_video]
    gt = (df.groupby('shot_number')
            .agg(start_frame=('frame_count', 'min'), end_frame=('frame_count', 'max'),
                 shot_result=('shot_result', 'first'))
            .reset_index())
    return gt[gt['shot_result'].isin([0, 1])].reset_index(drop=True)


def run(predictor, video: Path, gt: pd.DataFrame) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        _, stats = predictor.process_video(str(video), display=False, save_video=False, verbose=False)
    y_true, y_pred, matched, ghosts, missed = match_shots(stats['shots'], gt, tol=FRAME_TOL)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return {'labeled': len(gt), 'found': len(matched), 'missed': len(missed), 'false': len(ghosts),
            'correct': correct, 'acc': correct / len(matched) if matched else None,
            'detected': len(stats['shots'])}


def evaluate(predictor, ball_conf: float, conf: float, label: str = "") -> dict:
    infer.BALL_CONF, infer.YOLO_CONF = ball_conf, conf
    per_video, total = {}, {'labeled': 0, 'found': 0, 'missed': 0, 'false': 0, 'correct': 0}
    for video_name, csv_path, source in EVAL_SETS:
        video = RAW_VIDEO_DIR / video_name
        if not video.exists() or not csv_path.exists():
            continue
        r = run(predictor, video, load_labels(csv_path, source))
        per_video[video_name[:28]] = r
        for k in total:
            total[k] += r[k]
    total['acc'] = total['correct'] / total['found'] if total['found'] else None
    total['recall'] = total['found'] / total['labeled'] if total['labeled'] else None
    print(f"  {label:<26} ball≥{ball_conf:.2f} other≥{conf:.2f} | "
          + " | ".join(f"{v}: {r['found']}/{r['labeled']} found, {r['false']} false, "
                       f"acc {r['acc']:.3f}" if r['acc'] is not None else f"{v}: none found"
                       for v, r in per_video.items())
          + f" || TOTAL recall {total['recall']:.3f}, false {total['false']}, acc {total['acc']:.3f}")
    return {'ball_conf': ball_conf, 'conf': conf, 'label': label, 'videos': per_video, 'total': total}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ball-conf", type=float, default=infer.BALL_CONF)
    ap.add_argument("--conf", type=float, default=infer.YOLO_CONF, help="basket / player threshold")
    ap.add_argument("--config", default=str(RIM_RULE_CONFIG), help="predictor config")
    ap.add_argument("--sweep", action="store_true", help="evaluate a grid of thresholds")
    ap.add_argument("--trigger", choices=["new", "old"], default="new",
                    help="new: hoop zone + aim check (HPAI-20); old: ball above the players")
    ap.add_argument("--tag", default="", help="name for the saved results file")
    args = ap.parse_args()

    infer.NEW_TRIGGER = args.trigger == "new"
    cfg = Path(args.config)
    base = cfg.name.replace('_config.pkl', '')
    with contextlib.redirect_stdout(io.StringIO()):
        predictor = infer.RealTimeShotPredictor(
            str(YOLO_MODEL), str(cfg.parent / f"{base}_best.keras"),
            str(cfg.parent / f"{base}_scaler.pkl"), str(cfg), str(PREDICTIONS_DIR))

    print(f"Predictor: {cfg.name} | trigger: {args.trigger}")
    results = []
    if args.sweep:
        results.append(evaluate(predictor, 0.75, 0.75, "old behaviour"))
        for ball in (0.25, 0.35, 0.5):
            for other in (0.5, 0.6, 0.75):
                results.append(evaluate(predictor, ball, other))
    else:
        results.append(evaluate(predictor, args.ball_conf, args.conf, "current"))

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"detection_eval_{time.strftime('%Y%m%d_%H%M%S')}{'_' + args.tag if args.tag else ''}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved → {out}")
