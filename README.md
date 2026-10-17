# HooperAI 🏀

Basketball shot analytics from video. A YOLO detector finds the **ball, basket and
players** in every frame; a shot-detection state machine cuts each shot into a
sequence of per-frame features; a predictor decides whether the shot was made or
missed. A React Native (Expo) companion app lives in `HooperAI/`.

```
video ──► YOLO (ball / basket / player) ──► shot detector ──► basket-relative features ──► rim-pass rule (or LSTM) ──► MADE / MISSED
```

The default predictor is a **rim-pass rule**: once the ball has been above the rim,
a shot is *made* if the ball drops back through rim height within `rim_threshold`
basket widths of the rim centre. Measured leave-one-video-out on the 184 labeled
shots (`python scripts/lstm/lstm_train.py`):

| Predictor | Unseen video | Same videos (5-fold) |
|---|---|---|
| Rim-pass rule (1 parameter, default) | **92.9%** | 94.0% |
| LSTM v7 – residual on the rule, 3-seed ensemble | 92.4% | **95.1%** |
| LSTM v6 – sequence only | 63.0% | 91.3% |
| LSTM, old raw-pixel features | 53.8% | 87.0% |

LSTM v7 learns a correction on top of the rule (`logit = a·rule + b + correction`,
correction from a BiLSTM + attention over the frames, L2-penalised so it stays near
the rule until labeled data says otherwise). It ties the rule on unseen videos and
beats it within known videos; with more labeled videos it is the one expected to
improve. Models v1–v5 reported 97–100% only because augmented copies of test shots
were in the training set.

## Project structure

```
HooperAI/                  repo root
├── scripts/
│   ├── paths.py           shared paths – every script takes its defaults from here
│   ├── shot_features.py   shared feature transform + shot-detection constants (train = infer)
│   ├── labeling/
│   │   └── shot_labeler.py    play a video, auto-detect shots, label made/missed → CSV
│   ├── lstm/
│   │   ├── lstm_train.py      cross-validate + fit the rim-pass rule and the LSTM from labeled CSVs
│   │   ├── lstm_infer.py      YOLO + predictor on a video (annotated video, CSV, JSON)
│   │   └── lstm_eval.py       run inference and compare against a labeled CSV (confusion matrix)
│   ├── yolo/
│   │   ├── yolo_eval.py       per-class precision / recall / mAP on a YOLO test set
│   │   ├── compare_models.py  compare two YOLO models on one video
│   │   ├── show_detections.py preview YOLO boxes on a video
│   │   ├── plot_training.py   plot an Ultralytics results.csv
│   │   └── export_tflite.py   export the detector to INT8 TFLite
│   ├── app/
│   │   ├── server.py          laptop test bench: local web UI + JSON API (FastAPI)
│   │   └── static/            the UI (index.html, app.js, app.css)
│   └── ui/
│       └── infer_ui.py        Tkinter launcher for lstm_infer.py
├── HooperAI/              Expo mobile app (UI prototype, not yet connected to the model)
├── notebooks/experiments/ Colab notebook for training the YOLO detector
├── docs/                  Colab notebook setup guide
├── env.example            Roboflow settings for the notebook (copy to .env)
├── data/                  local only, except labeled CSVs (see below)
├── archives/              older models and zipped datasets/runs (local only)
├── runs/                  Ultralytics training output (local only)
└── requirements.txt
```

### Data layout

`data/` is git-ignored except `data/processed/annotations/*.csv`.

| Folder | Contents |
|---|---|
| `data/raw/videos/` | source videos |
| `data/raw/model_dataset/` | YOLO dataset (`data.yaml`, train/valid/test) |
| `data/processed/annotations/` | labeled shot CSVs – the LSTM training data (tracked) |
| `data/model/` | YOLO weights, `basketball_shot_rim_rule_config.pkl`, LSTM models (`basketball_shot_lstm_v7_best.keras`, `*_scaler.pkl`, `*_config.pkl`) |
| `data/processed/detections/` | cached per-frame YOLO boxes (speeds up re-evaluation) |
| `data/processed/predictions/` | output of `lstm_infer.py` |
| `data/processed/eval/` | output of `lstm_eval.py` |

## Setup

Python 3.10+ (developed on 3.13).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

For GPU detection, install a CUDA build of PyTorch that your NVIDIA driver supports
(<https://pytorch.org/get-started/locally/>). `nvidia-smi` shows the highest CUDA version
the driver supports; with driver 546 (CUDA 12.3) use the `cu126` wheels:

```bash
pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Check it with `python -c "import torch; print(torch.cuda.is_available())"`. `lstm_infer.py`
prints the device it uses; on CPU, YOLO is the bottleneck by far.

Model weights and videos are not in git. Copy them into `data/model/` and `data/raw/videos/`.

## Workflow

Run everything from the repo root. Each script has a configuration block near the
top; the default paths come from `scripts/paths.py`.

1. **Label shots:** set `VIDEO_PATH` / `OUTPUT_CSV` in `shot_labeler.py`, then
   ```bash
   python scripts/labeling/shot_labeler.py
   ```
   Keys: `1` made, `0` missed, `2` discard, `x` cancel current shot, `q` quit. The CSV is
   appended to, so labeling can span sessions.
2. **Train:** add the CSV to `DATA_PATHS` and set `MODEL_NAME` in `lstm_train.py`, then
   ```bash
   python scripts/lstm/lstm_train.py
   ```
   It cross-validates the rim-pass rule and the LSTM (leave-one-video-out and 5-fold by
   shot, split before augmentation, thresholds tuned on training/validation data only),
   then saves `basketball_shot_rim_rule_config.pkl` and the LSTM.
   The leave-one-video-out score is the one that predicts performance on new footage.
   `--cv video|shot|none`, `--compare-legacy` (old raw-pixel features), `--no-save`.
3. **Predict on a video:**
   ```bash
   python scripts/lstm/lstm_infer.py
   python scripts/ui/infer_ui.py      # same, with a GUI
   ```
4. **Evaluate against labels:**
   ```bash
   python scripts/lstm/lstm_eval.py --video data/raw/videos/<video>.mp4 --gt data/processed/annotations/<labels>.csv
   ```
   Headless runs cache the per-frame YOLO boxes in `data/processed/detections/`, so
   re-evaluating a new predictor on the same video skips detection and takes seconds.
   To use the LSTM instead of the rule, pass `--config data/model/<lstm>_config.pkl`.
   Only evaluate on a video whose CSV was **not** in `DATA_PATHS`.

### Laptop test bench

Test the newest pipeline on the laptop before integrating it into the mobile app:

```bash
python scripts/app/server.py
```

Open <http://127.0.0.1:8000>, pick a video (or upload one), choose a predictor and press
**Analyze video**. You get the video with live YOLO boxes, a timeline of detected shots
and a shot list. Label each shot **Made** / **Missed** (keys `M` / `X`, `J` / `K` to move
between shots). Labels are saved with the shot's frames to
`data/processed/annotations/app_labeled.csv`, in the same format as the labeler CSVs:
add it to `DATA_PATHS` in `lstm_train.py` to retrain on your corrections.

The UI only uses the JSON API (`/api/analyze`, `/api/jobs/{id}`, `/api/feedback`, …,
listed in `server.py`), so the mobile app can call the same endpoints. Set
`HOOPER_HOST=0.0.0.0` to reach it from a phone on the same network.

### Mobile app

```bash
cd HooperAI
npm install
npx expo start
```

See `HooperAI/README.md` for connecting it to the inference pipeline.

## Known issues

- 145 of the 184 labeled shots come from one video; more labeled videos are the
  biggest remaining improvement, especially for the LSTM.
- The rim-pass rule needs the ball tracked near the rim; long occlusions there
  (> 5 frames) count as a miss.
- Models v1–v5 still load in `lstm_infer.py` (legacy feature path).

## Branches

`main` is stable; do day-to-day work on `development` or feature branches and open a PR into `main`.

## License

Datasets from [Roboflow Universe](https://universe.roboflow.com/) are CC BY 4.0.
Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and TensorFlow/Keras.
