# HooperAI – Progress & Decisions (source of truth)

Living record of what the project is, what was decided and why, what is done, and
what is next. Update it in the same commit as any change that alters a decision,
a result or the status below. Last updated: **2026-10-17** (HPAI-15).

---

## 1. What HooperAI is

Basketball shot analytics from video: a phone records (or picks) a clip, the laptop
detects shots and decides **made / missed**, and the user reviews and labels each
shot. Labels become training data.

```
phone (Expo app) ──upload, original quality──► laptop server (FastAPI, scripts/app/server.py)
                                                 │  YOLO (ball / basket / player, GPU)
                                                 │  shot detector (state machine)
                                                 │  predictor: rim-pass rule (default) or LSTM v7
phone ◄──────── shots, review, labels ───────────┘──► Firestore users/{uid}/sessions (mirror)
```

| Part | Where |
|---|---|
| Detector, features, predictors, training | `scripts/` (`shot_features.py`, `lstm/`, `yolo/`, `labeling/`) |
| Laptop server + browser test bench | `scripts/app/` (`server.py`, `auth.py`, `firestore_sync.py`, `static/`) |
| Mobile app (Expo SDK 57) | `HooperAI/` |
| Setup guides | `README.md`, `HooperAI/README.md`, `docs/MOBILE_SETUP.md` |
| Data (local only, except labeled CSVs) | `data/` |

---

## 2. Constraints and preferences (from the project owner)

- **No paid services for now.** All processing runs on the owner's laptop GPU
  (RTX 3060 Laptop, driver 546 / CUDA 12.3). Firebase stays on the **Spark** plan
  (Auth + Firestore only – no Storage, no Cloud Functions, no backups). Remote access
  via Tailscale personal (free). Electricity is not a concern.
- **Videos keep original quality** end to end (no downscaled uploads).
- Data files (videos, images, CSVs, models) are never deleted during clean-ups.
- Work is tracked as **HPAI-XX tickets** (section 8).

---

## 3. Key decisions (and why)

| # | Decision | Why | Ticket |
|---|---|---|---|
| D1 | Repo reorganised into `scripts/{lstm,yolo,labeling,ui,app}` with shared `scripts/paths.py`; no absolute paths | Duplicate docs/scripts and hard-coded `C:\...` paths made the repo non-portable | HPAI-1 |
| D2 | `main` = production, `development` = integration; `origin/main` merged into `development` keeping only the Colab notebook + its guide from main's scaffolding | Branches had diverged (19 vs 6 commits) | HPAI-2 |
| D3 | Evaluation is **leave-one-video-out**, split **before** augmentation, thresholds fit on training data only | v1–v5 LSTMs reported 97–100% only because augmented copies of test shots were in training; honest LOVO was ~54% (chance) | HPAI-3 |
| D4 | Features v2: basket-relative, scale-free, gap interpolation, smoothed velocity, last 64 frames, plus rim-pass features; one shared module (`scripts/shot_features.py`) for training and inference | Raw pixels tied the model to one camera position; train/infer code had drifted (confirm frames 10 vs 23) | HPAI-3 |
| D5 | **Default predictor = rim-pass rule**: made if the ball drops through rim height within `rim_threshold` (0.45) basket widths of the rim centre | One parameter, **92.9% on unseen videos** vs 63% for the sequence LSTM | HPAI-3 |
| D6 | LSTM v7 = **residual on the rule** (logit = a·rule + b + L2-penalised BiLSTM/attention correction), 3-seed ensemble, threshold 0.5 | Ties the rule on unseen video (92.4%), beats it within known videos (95.1%); expected to improve as labeled videos grow | HPAI-4 |
| D7 | YOLO on CUDA (cu126 PyTorch wheels for driver 546) with FP16; per-frame detections cached in `data/processed/detections/` | CPU ran at ~4 fps; GPU ~24 fps; re-analysis from cache takes seconds instead of minutes | HPAI-5 |
| D8 | **Server processing (laptop) first, on-device later** | API already existed, full-size model, zero cost for one user; on-device needs a smaller YOLO + TS port of shot logic (planned) | HPAI-6/8 |
| D9 | Server auth: localhost trusted; API key or Firebase ID tokens off-laptop; refuses to listen on the network without auth | Safe to expose over Wi-Fi/Tailscale | HPAI-7 |
| D10 | Laptop is the **source of truth** for sessions and labels; the phone caches and syncs | Labels can come from the phone, the browser bench or another device | HPAI-12 |
| D11 | Shot ends when the ball is lost for 0.5 s (15 frames) after confirmation; a shot still open at video end is still predicted | After the rim the ball is often lost in the net/blur – such shots were silently dropped | HPAI-13 |
| D12 | Labels: 1 made, 0 missed, **-1 not a shot** (→ `app_not_shots.csv`); users can **add missed shots** (frames rebuilt from cached detections) | False detections must not pollute made/missed training data; missed shots should still become training data | HPAI-14 |
| D13 | Mobile app on **Expo SDK 57** (React Native 0.86, React 19, React Navigation 7, Firebase 13, expo-video) | Play Store Expo Go only runs the current SDK | HPAI-10 |
| D14 | Portrait video is supported as-is | OpenCV auto-rotates; the earlier "no shots" was the lost-ball bug (D11), not orientation | HPAI-13 |

---

## 4. Results (measured)

Leave-one-video-out on 184 labeled shots from 4 videos (`python scripts/lstm/lstm_train.py`):

| Predictor | Unseen video | Same videos (5-fold) |
|---|---|---|
| Rim-pass rule (default) | **92.9%** | 94.0% |
| LSTM v7 residual ensemble | 92.4% | **95.1%** |
| LSTM v6 sequence only | 63.0% | 91.3% |
| LSTM, old raw-pixel features | 53.8% | 87.0% |

Speed: YOLO on GPU ~24 fps (1080p), CPU ~4 fps; cached re-analysis of a 6-min video
~27 s (vs 445 s). 145 of the 184 training shots come from a single video – more
labeled videos are the biggest remaining lever.

---

## 5. Current state (2026-10-17)

**Working end to end on a real phone (Android, Expo Go SDK 57):** sign-in (Firebase),
record/pick → upload → GPU analysis → review with video → label made / missed /
not a shot → add missed shots → session history and stats (synced with the laptop).

- Firebase project `hooperai-4f9cc` (Spark): Email/Password auth, Firestore `eur3`
  with `firestore.rules` (users read only their own sessions), web app config in
  `HooperAI/.env`, service account at `secrets/firebase-adminsdk.json`,
  `secrets/server.env` → server runs in Firebase mode with Firestore sync. All
  secrets are git-ignored.
- Python env: `.venv` (Python 3.13) with torch 2.11 cu126, TensorFlow 2.21 (CPU –
  no GPU TF on native Windows), ultralytics 8.4, FastAPI, firebase-admin.
- Tests: `python -m pytest tests -q` → 11 passing (server API with fakes).
- Labels collected in the app so far: 18 shots (9 made) in
  `data/processed/annotations/app_labeled.csv`.

**Run it**
```bash
.venv\Scripts\python.exe scripts\app\server.py      # laptop server (reads secrets/server.env)
cd HooperAI && npx expo start                      # app; Expo Go → exp://<laptop-ip>:8081
```

---

## 6. Known issues / limitations

- Uploads stop if the app goes to the background during the upload itself
  (analysis polling survives app switching).
- In-progress jobs are lost if the laptop server restarts (finished sessions persist).
- Ball detection is the weak link (e.g. 19% of frames on a portrait clip); the shot
  detector needs ~23 frames of the ball above the players.
- Shot zones are not detected yet (the UI hides the section).
- Windows Firewall must allow `.venv\Scripts\python.exe` on the active network
  (the owner's Wi-Fi is set as Public).
- Two old January stashes hold a 2.4 GB video blob in the local `.git` (not on GitHub);
  dropping them + `git gc` is pending the owner's OK.

---

## 7. Next steps (candidates for tickets)

1. Retrain with `app_labeled.csv` added to `DATA_PATHS` once more videos are labeled;
   compare rule vs LSTM v7 again (LOVO).
2. Use `app_not_shots.csv` to tune/learn the shot detector (false positives).
3. Train yolo26n/s, export TFLite INT8, measure accuracy loss (on-device phase).
4. Port `shot_features.py` + shot detector to TypeScript with golden-vector tests;
   on-device live predictions (Expo dev build, vision-camera + fast-tflite).
5. Background-safe uploads (resumable / chunked).

---

## 8. Ticket log (HPAI-XX)

Next free ticket: **HPAI-16**. Workflow: `docs/WORKFLOW.md`.

| Ticket | Type | Title | Status |
|---|---|---|---|
| HPAI-1 | chore | Project clean-up: remove duplicate docs/scripts, reorganise `scripts/`, shared paths, README | Done |
| HPAI-2 | chore | Fix GitHub branches: merge `origin/main` into `development`, sync `main`/`testing` | Done |
| HPAI-3 | feature | Honest evaluation (LOVO), features v2, rim-pass rule predictor | Done |
| HPAI-4 | feature | LSTM v7 residual-on-rule ensemble trainer | Done |
| HPAI-5 | feature | GPU inference (cu126), FP16, detection cache | Done |
| HPAI-6 | feature | Laptop test bench (browser UI + JSON API) | Done |
| HPAI-7 | feature | Server auth (API key / Firebase), persisted sessions, Firestore mirror, tests | Done |
| HPAI-8 | feature | Mobile app ↔ laptop: record/pick, upload, analyze, label | Done |
| HPAI-9 | chore | Firebase project setup (Auth, Firestore eur3 + rules, web app, service account) | Done |
| HPAI-10 | chore | Upgrade mobile app to Expo SDK 57 | Done |
| HPAI-11 | feature | Session review with video playback + session history | Done |
| HPAI-12 | bug | Labels not reflected on Home/Stats; analysis stuck after app switch (sync + polling) | Done |
| HPAI-13 | bug | Shots dropped when the ball is lost after the rim (portrait clip) | Done |
| HPAI-14 | feature | "Not a shot" label and manually added shots | Done |
| HPAI-15 | chore | `progress_context.md`, HPAI ticket workflow, release to `main` | Done |
