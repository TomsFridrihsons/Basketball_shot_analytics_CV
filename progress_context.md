# HooperAI – Progress & Decisions (source of truth)

Living record of what the project is, what was decided and why, what is done, and
what is next. Update it in the same commit as any change that alters a decision,
a result or the status below. Last updated: **2026-10-17** (HPAI-24).

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
- **Planned:** paid tiers, automatic video deletion after 14 / 30 days, and collecting
  data for further training (section 9). The free-only rule applies until then.
- Keep the old `testing` and `colab-training` branches for now (owner, 2026-10-17).

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
| D15 | **Per-class thresholds: ball ≥ 0.35, basket/player ≥ 0.75**; YOLO cache stores all boxes ≥ 0.25 | Ball at 0.75 missed 20% of real shots; 0.25–0.50 all find 98% end to end | HPAI-18/24 |
| D16 | **Default predictor = rim-pass rule on features v3** (track cleaning + parabola gap fill ≤ 1 s, threshold 0.40) | 96.7% unseen-video vs 92.9% for v2; trajectory judge gives identical end-to-end results, kept optional | HPAI-19/21/24 |
| D17 | **Old start trigger (ball above players) stays default**; hoop-zone trigger + aim check behind `NEW_TRIGGER` | New trigger: 99.4% shots found but 17 false vs 3 – more total errors; needs a stricter aim/false-shot filter first | HPAI-20/24 |
| D18 | Shot metrics (speed, release/entry angle, arc) shown only when `reliable` (gravity check 0.6–1.6 g) | Camera behind the shooter makes depth unknown; 80/184 shots pass | HPAI-22 |

---

## 4. Results (measured)

Leave-one-video-out on 184 labeled shots from 4 videos (`python scripts/lstm/lstm_train.py`):

| Predictor | Unseen video | Same videos (5-fold) |
|---|---|---|
| Rim-pass rule (default) | **92.9%** | 94.0% |
| LSTM v7 residual ensemble | 92.4% | **95.1%** |
| LSTM v6 sequence only | 63.0% | 91.3% |
| LSTM, old raw-pixel features | 53.8% | 87.0% |

With features v3 (HPAI-19) the rule reaches **96.7%** on unseen video.

End to end (YOLO → shot detector → made/missed) on `20250711_171647` (145 labeled shots)
+ app session 18:22 (18), `python scripts/lstm/eval_detection.py` (HPAI-24):

| Setup | Shots found | False shots | Made/missed correct |
|---|---|---|---|
| Before (all ≥ 0.75, old trigger, rule v2) | 78.5% | 3 | 79.7% |
| **Default now** (ball 0.35, old trigger, rule v3) | 95.1% | 3 | **91.6%** |
| New trigger + aim check (rule v3 or judge) | **99.4%** | 17 | 92.0% |

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

Next free ticket: **HPAI-26**. Workflow: `docs/WORKFLOW.md`.

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
| HPAI-16 | chore | Plan: paid tiers, video retention, training-data collection (section 9) | Done |
| HPAI-17 | chore | Detection roadmap (section 10): selected work + parked items | Done |
| HPAI-18 | feature | Per-object confidence thresholds; detection cache keeps confidences; baseline evaluation | Done |
| HPAI-19 | feature | Better ball tracking: outlier cleaning, parabola gap filling, time-based (fps-independent) detector | Done |
| HPAI-20 | feature | New shot start trigger (hoop zone / leaves shooter) + aim check | Done |
| HPAI-21 | feature | Made/missed upgrades: parabola at rim height, ball hidden by the net, rim bounces | Done |
| HPAI-22 | feature | Measurement: rim-width scale, release speed, arc height, entry angle | Done |
| HPAI-23 | feature | Home screen redesign (Apple HIG): "Today" hero with makes/attempts + shot-log dots, review prompt, labelled tab bar, WCAG-contrast text colours | Done |
| HPAI-24 | fix | Detection defaults from the end-to-end eval: ball 0.35 / others 0.75, rule v3, old trigger | Done |
| HPAI-25 | feature | Record + Session screens redesign (Apple HIG): full-bleed camera, one panel per state (permission, ready, offline, upload/analyze steps, done, error), review on the Session screen, 11 pt minimum text, red only for recording | Done |

---

## 9. Future plan: paid tiers, video retention, training data

Status: **planning only, nothing implemented.** Owner requirements are marked (owner);
everything else is a suggestion to decide on. Open decisions are in 9.6.

### 9.1 Paid tiers (suggested)

| | **Free** | **Pro** (~€4.99/mo or ~€39/yr) | **Coach / Team** (~€14.99/mo) |
|---|---|---|---|
| Analysis | 3–5 sessions or ~60 min video / month | ~10 h video / month (fair use) | per player, shared quota |
| Video kept (owner) | **14 days** | **30 days** (option: 90) | 30–90 days |
| Predictor | rim-pass rule | rule + LSTM v7, faster queue | as Pro |
| Review & labels | yes | yes | coach can review and label players' shots |
| Stats | session FG% | trends over time, shot zones / heat map, weekly report | team dashboard, player comparison |
| Extras | – | CSV/PDF export, made-shot highlight reel | roster, shared sessions, web dashboard |
| Ads | none | none | none |

Other options: **session packs** (one-off, e.g. €2.99 for 5 sessions) for occasional
users; a **free Pro trial** (7–14 days); **labeling credits** (label N shots, get extra
free minutes – also feeds 9.3).

Feature ideas for paid tiers, roughly by value / effort:
1. Shot zones + heat map (needs court / player-position mapping) – Pro.
2. Progress over time: FG% by week, by zone, streaks – Pro.
3. Made-shot highlight clip export – Pro (cheap: cut from the original video).
4. Live on-device predictions during practice (on-device roadmap, section 7) – Pro.
5. Coach accounts: assign players, review their sessions, comments – Team.
6. Shooting-form feedback (pose estimation: release point, arc) – later, premium.
7. Challenges / leaderboards between friends – engagement, any tier.

### 9.2 Video retention and deletion (owner: after 14 or 30 days)

- **Deleted after the retention period:** the uploaded video (`data/raw/videos/app_*`)
  and its per-frame detection cache. **Kept:** the session result (shots, labels,
  stats), the per-shot feature rows used for training (if the user consented, 9.3) and
  the Firestore summary.
- Retention by tier, counted from upload: Free 14 days, Pro 30 (or 90) days. Store
  `uploaded_at` and the user's tier on each session (server `app_sessions/*.json` and
  Firestore) so the rule can be applied per user.
- **Implementation on the laptop server:** a daily sweep (at start-up and every 24 h in
  a background thread, or via Windows Task Scheduler) deletes expired files, marks the
  session `video_expired: true` and logs what it removed. The app then shows "video
  deleted – results kept" instead of the player, and "video expires in N days" on the
  review screen.
- **Once in the cloud:** object-storage lifecycle rules (Cloudflare R2 / S3) delete
  objects after N days at no extra cost – one prefix per tier.
- **User controls:** "delete this video now" and "delete my account" (videos, sessions,
  labels linked to the user, Firestore docs, Firebase Auth user). Required under GDPR
  (right to erasure); the owner is in the EU (Latvia).
- Videos stored on the phone are the user's own files; the app never deletes them.

### 9.3 Collecting data for further training (owner)

What is worth keeping, most useful first:
1. **Labeled shot feature rows** – already collected: `app_labeled.csv` (made / missed),
   `app_not_shots.csv` (false detections), manually added shots. Small, no images.
2. **Short frame crops around each labeled shot** (the 2–3 s window, downscaled) – what
   YOLO needs to detect the ball better (the weak link, section 6). Keep them beyond the
   video retention only with consent.
3. **Low-confidence shots and model mistakes** – the most valuable labels (active
   learning: ask users to label those first).

Rules:
- **Opt-in consent** in the app ("Help improve HooperAI: keep anonymous shot data and
  short clips for training"), off by default for frames; asked at sign-up and in
  Settings; can be withdrawn (data then deleted).
- Privacy policy and terms before any public release. Frames show people, so treat them
  as personal data (GDPR: consent as the lawful basis, data minimisation, retention
  limits, list processors such as Firebase and hosting).
- **Pseudonymise** training exports: replace the uid with a random id, drop file names;
  keep the source video only as an opaque id so leave-one-video-out evaluation still
  works.
- Version datasets as dated snapshots, and always evaluate per video / user (D3).

### 9.4 Payments and accounts

- Digital subscriptions in mobile apps must use **Google Play Billing / Apple In-App
  Purchase** (store fee 15% for small developers on subscriptions). **RevenueCat**
  wraps both and is free up to a revenue threshold; Stripe only for a web dashboard.
- Entitlements: store the tier on the user (Firebase custom claims or a Firestore
  `users/{uid}` document written by the server / payment webhook). Enforce quotas
  (minutes per month, retention) **on the server**; the app only mirrors them in the UI.
- Paid tiers at scale may need Firebase **Blaze** (pay-as-you-go; free quotas still
  apply), e.g. for payment webhooks in Cloud Functions.

### 9.5 Cost reality (why tiers and on-device matter)

- Paying users can't depend on the owner's laptop: processing has to move to a hosted
  GPU or onto the phone. Estimate: ~€0.003–0.01 per minute of video on a serverless GPU
  (based on the measured 24 fps on an RTX 3060).
- A heavy Pro user (10 h / month ≈ 600 min) costs ~€2–6 / month in GPU time at full
  price – close to the subscription. So: cap minutes per tier, use cheaper settings
  (yolo26s, cached detections) and prioritise the **on-device** roadmap (section 7,
  items 3–4), which makes analysis free to run.
- Storage: 1080p ≈ 100–130 MB per minute; retention (9.2) keeps storage bounded.

### 9.6 Open decisions

- [ ] Retention: Free 14 days / Pro 30 days – confirm, or 30 / 90?
- [ ] Prices and quotas per tier; trial length.
- [ ] Training data beyond retention: feature rows only, or also frame crops?
- [ ] Where paid users' processing runs: hosted GPU or on-device first.
- [ ] Payments: RevenueCat (+ Play Billing) or direct.

### 9.7 Ticket candidates (not created yet)

| Area | Work |
|---|---|
| Retention | `uploaded_at` + tier on sessions; daily deletion sweep; "expires in N days" / "video deleted" UI; delete-now and delete-account |
| Data | consent flag + Settings toggle; pseudonymised dataset export script; frame-crop extraction around labeled shots |
| Tiers | entitlement model + server-side quotas; RevenueCat / Play Billing; paywall screen |
| Pro features | trends screen; highlight clip export; shot zones |
| Legal | privacy policy, terms, in-app consent texts |

---

## 10. Detection roadmap (2026-10-17, from the research in the conversation)

Sources: avishah3/AI-Basketball-Shot-Detection-Tracker (hoop-zone trigger, line at rim
height, track cleaning), Roboflow "AI basketball shot evaluator" / aarnavshah12/shot-tracker
(release from the wrist, parabola aim check, net-occlusion fall speed, rim bounces,
rim-width scale), Roboflow player identification (tracking, team clustering, jersey
numbers), court-keypoint homography papers (KaliCalib, Basketball-SORT, BasketTracking),
pose-based form analysis (mc261670164/AI-basketball-analysis), TrackNet.

### 10.1 Selected – done, merged to development (owner, 2026-10-17)

| Ticket | Improvement |
|---|---|
| HPAI-18 | **Separate confidence thresholds per object** (ball lower than basket/player); cache all boxes ≥ 0.25 with their confidence so thresholds can be tuned without re-running YOLO; baseline numbers on the labeled videos |
| HPAI-19 | **Better ball tracking**: drop jumpy / wrongly shaped ball boxes, fill gaps with a fitted parabola, express detector timings in seconds (60 fps videos) |
| HPAI-20 | **New start trigger** (ball enters the zone above the hoop, or leaves the shooter going up) + **aim check** (fitted parabola must point at the rim) |
| HPAI-21 | **Made/missed upgrades**: parabola through several points at rim height; **ball hidden by the net** (fall speed when it reappears); **rim bounces** |
| HPAI-22 | **Measurement**: rim width (45.7 cm) → metres; release speed, arc height, entry angle |

Every step is measured before it is kept: (a) made/missed on the 184 labeled shots
(leave-one-video-out), (b) end to end on `20250711_171647` (145 labeled shots) and the
18:22 app session (18 labeled shots): shots found / missed / false, made-missed accuracy.
After HPAI-18…22: **revisit the order of the remaining items** (owner).
Open from this round: cut the new trigger's false shots (14 extra on the long video) so it
can become the default; more labeled phone videos (only 18 shots today).

### 10.2 Parked – do not forget (owner: "leave B and C for later")

**B. Telling players apart**
- B1. Track players across frames (Ultralytics BoT-SORT, `model.track`).
- B2. Who shot: tracked player nearest the ball at release / whose wrist it left (pose).
- B3. "Which one is you?": tap a player once in the review screen, then match the rest of
  the session by appearance (OSNet re-ID or SigLIP embeddings + clustering).
- B4. Teams: unsupervised appearance clustering into two teams → per-player / per-team
  FG% (Coach tier).
- Avoid face recognition (biometric data under GDPR).

**C. New stats and features**
- C1. Shot metrics shown in the app (release angle, arc, entry angle, speed) – the
  measurement itself is HPAI-22; presentation / Pro packaging is parked.
- C2. Shot zones + shot chart via a one-time 4-tap court calibration (free-throw lane
  corners → homography → 9 zones).
- C3. Shooting form from pose (YOLO11-pose): elbow / knee angles, release height,
  consistency, made vs missed form.
- C4. Detection quality metrics from app labels (missed shots added, "not a shot")
  tracked over time.
- C5. Retrain YOLO with the University of Arizona "Basketball Shooting Robot" dataset
  (~9.8k images) + frame crops from app-labeled shots; optional "ball in net" class.
- C6. Second YOLO pass on a full-resolution crop around the hoop (small ball).
