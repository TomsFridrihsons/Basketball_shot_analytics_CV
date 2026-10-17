# HooperAI 🏀

Basketball shot analytics from video. A YOLO detector finds the **ball, basket and
players** in every frame; a shot-detection state machine cuts each shot into a
sequence of per-frame features; an **LSTM** predicts whether the shot was made or
missed. A React Native (Expo) companion app lives in `HooperAI/`.

```
video ──► YOLO (ball / basket / player) ──► shot detector ──► 18 features per frame ──► LSTM ──► MADE / MISSED
```

## Project structure

```
HooperAI/                  repo root
├── scripts/
│   ├── paths.py           shared paths – every script takes its defaults from here
│   ├── labeling/
│   │   └── shot_labeler.py    play a video, auto-detect shots, label made/missed → CSV
│   ├── lstm/
│   │   ├── lstm_train.py      train the shot-outcome LSTM from labeled CSVs
│   │   ├── lstm_infer.py      YOLO + LSTM inference on a video (annotated video, CSV, JSON)
│   │   └── lstm_eval.py       run inference and compare against a labeled CSV (confusion matrix)
│   ├── yolo/
│   │   ├── yolo_eval.py       per-class precision / recall / mAP on a YOLO test set
│   │   ├── compare_models.py  compare two YOLO models on one video
│   │   ├── show_detections.py preview YOLO boxes on a video
│   │   ├── plot_training.py   plot an Ultralytics results.csv
│   │   └── export_tflite.py   export the detector to INT8 TFLite
│   └── ui/
│       └── infer_ui.py        Tkinter launcher for lstm_infer.py
├── HooperAI/              Expo mobile app (UI prototype, not yet connected to the model)
├── docs/                  extra guides (Colab training)
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
| `data/model/` | current YOLO weights and LSTM models (`*_best.keras`, `*_scaler.pkl`, `*_config.pkl`) |
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

For GPU detection, install the CUDA build of PyTorch from <https://pytorch.org/get-started/locally/>.
Check it with `python -c "import torch; print(torch.cuda.is_available())"`.

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
3. **Predict on a video:**
   ```bash
   python scripts/lstm/lstm_infer.py
   python scripts/ui/infer_ui.py      # same, with a GUI
   ```
4. **Evaluate against labels:**
   ```bash
   python scripts/lstm/lstm_eval.py --video data/raw/videos/<video>.mp4 --gt data/processed/annotations/<labels>.csv
   ```

### Mobile app

```bash
cd HooperAI
npm install
npx expo start
```

See `HooperAI/README.md` for connecting it to the inference pipeline.

## Known issues

- `lstm_train.py` augments shots **before** the train/test split, so copies of a shot
  land in both sets and test scores are inflated. Split by shot first.
- The optimal threshold is picked on the test set; use a validation set instead.
- Features are raw pixel coordinates, so models don't transfer well between camera setups.
- Shot confirmation differs between `shot_labeler.py` (`CONFIRM_FRAMES = 10`) and
  `lstm_infer.py` (23 frames).

## Branches

`main` is stable; do day-to-day work on `development` or feature branches and open a PR into `main`.

## License

Datasets from [Roboflow Universe](https://universe.roboflow.com/) are CC BY 4.0.
Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and TensorFlow/Keras.
