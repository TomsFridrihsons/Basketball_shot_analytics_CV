"""
server.py  –  HooperAI test bench + mobile backend (local web UI + JSON API)
===========================================================================
Runs the newest pipeline on the laptop. The browser UI (static/) and the
mobile app (HooperAI/) use the same JSON API.

    python scripts/app/server.py                     # → http://127.0.0.1:8000 (laptop only)
    HOOPER_HOST=0.0.0.0 HOOPER_API_KEY=... python scripts/app/server.py   # phone on Wi-Fi / Tailscale

Authentication: see auth.py (none on localhost, API key, or Firebase ID tokens).
Sessions are saved in data/processed/app_sessions/ so results survive restarts;
in Firebase mode they are also mirrored to Firestore (firestore_sync.py).

API (all JSON; everything except /api/health needs credentials off-laptop)
  GET  /api/health                 server up, auth mode (no auth)
  GET  /api/status                 device, predictors, YOLO model
  GET  /api/videos                 video library (data/raw/videos)
  POST /api/videos                 upload (multipart field "file") → {name}
  GET  /api/videos/{name}/stream   video file with HTTP range support
  POST /api/analyze                {video, config?, yolo_conf?} → {job_id}
  GET  /api/jobs/{job_id}          progress + result
  GET  /api/sessions               your analyzed sessions (newest first)
  GET  /api/detections/{job_id}    per-frame boxes for the overlay
  POST /api/feedback               {job_id, shot_number, label}
  GET  /api/stats                  totals over app_labeled.csv
"""

import os
import sys
import json
import time
import uuid
import pickle
import threading
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

def _load_env_file(path: Path):
    """KEY=VALUE lines from secrets/server.env (git-ignored); real env vars win."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not line.lstrip().startswith("#") and key.strip() not in os.environ:
                os.environ[key.strip()] = value.strip().strip('"')


_load_env_file(Path(__file__).resolve().parents[2] / "secrets" / "server.env")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lstm"))
from paths import (DATA_DIR, MODEL_DIR, RAW_VIDEO_DIR, ANNOTATIONS_DIR,  # noqa: E402
                   PREDICTIONS_DIR, YOLO_MODEL, RIM_RULE_CONFIG)
import shot_features as sf  # noqa: E402
import lstm_infer as infer  # noqa: E402
from auth import Auth, User  # noqa: E402
from firestore_sync import FirestoreSync  # noqa: E402

STATIC_DIR   = Path(__file__).resolve().parent / "static"
FEEDBACK_CSV = ANNOTATIONS_DIR / "app_labeled.csv"
SESSIONS_DIR = DATA_DIR / "processed" / "app_sessions"
VIDEO_EXTS   = {".mp4", ".mov", ".avi", ".mkv", ".3gp", ".m4v"}
API_VERSION  = 2

app = FastAPI(title="HooperAI test bench")
AUTH = Auth()
FIRESTORE = FirestoreSync(AUTH.firebase_app)
JOBS: dict = {}             # job_id → in-memory job (also persisted when done)
_PREDICTORS: dict = {}      # (config, yolo conf) → RealTimeShotPredictor (load once)
_LOCK = threading.Lock()    # one analysis at a time (GPU memory, shared predictor state)

current_user = AUTH.user    # FastAPI dependency


# ── helpers ──────────────────────────────────────────────────────────────────

def _video_path(name: str) -> Path:
    path = (RAW_VIDEO_DIR / name).resolve()
    if path.parent != RAW_VIDEO_DIR.resolve() or not path.exists():
        raise HTTPException(404, f"video not found: {name}")
    return path


def _predictor_configs():
    """Every loadable predictor config in data/model, rim rule first."""
    out = []
    for cfg in sorted(MODEL_DIR.glob("*_config.pkl"), key=lambda p: p.name):
        try:
            with open(cfg, 'rb') as f:
                c = pickle.load(f)
        except Exception:
            continue
        kind = c.get('model_type', 'lstm')
        if kind == 'lstm' and c.get('feature_version', 1) < 2:
            continue   # legacy v1–v5 models: leaked evaluation, not worth testing
        cv = c.get('cv_metrics', {})
        video_cv = cv.get('video_rim_rule' if kind == 'rim_rule' else 'video', {})
        out.append({
            'config': cfg.name,
            'label': 'Rim-pass rule' if kind == 'rim_rule'
                     else cfg.name.replace('basketball_shot_', '').replace('_config.pkl', '').upper(),
            'kind': kind,
            'unseen_video_accuracy': video_cv.get('accuracy'),
        })
    out.sort(key=lambda p: (p['kind'] != 'rim_rule', p['config']))
    return out


def _get_predictor(config_name: str, yolo_conf: float):
    key = (config_name, round(yolo_conf, 2))
    if key not in _PREDICTORS:
        cfg = MODEL_DIR / config_name
        base = cfg.name.replace('_config.pkl', '')
        infer.YOLO_CONF = yolo_conf
        _PREDICTORS[key] = infer.RealTimeShotPredictor(
            yolo_path   = str(YOLO_MODEL),
            lstm_path   = str(MODEL_DIR / f"{base}_best.keras"),
            scaler_path = str(MODEL_DIR / f"{base}_scaler.pkl"),
            config_path = str(cfg),
            output_dir  = str(PREDICTIONS_DIR),
        )
    infer.YOLO_CONF = yolo_conf
    return _PREDICTORS[key]


def _load_feedback() -> pd.DataFrame:
    if FEEDBACK_CSV.exists():
        return pd.read_csv(FEEDBACK_CSV)
    return pd.DataFrame()


def _shot_labels(video: str) -> dict:
    """{start_frame: label} for shots of `video` already labeled in the app."""
    df = _load_feedback()
    if df.empty:
        return {}
    df = df[df['source_video'] == video]
    shots = df.groupby('shot_number').agg(start=('frame_count', 'min'), label=('shot_result', 'first'))
    return {int(r.start): int(r.label) for r in shots.itertuples()}


# ── session persistence ──────────────────────────────────────────────────────

def _save_session(job: dict):
    """Result as JSON (listing) + shot frames as pickle (needed for labels)."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    public = {k: job[k] for k in ('id', 'uid', 'video', 'config', 'created', 'status', 'result') if k in job}
    (SESSIONS_DIR / f"{job['id']}.json").write_text(json.dumps(public), encoding="utf-8")
    with open(SESSIONS_DIR / f"{job['id']}_frames.pkl", 'wb') as f:
        pickle.dump({'shot_frames': job.get('shot_frames', {}), 'cache_path': job.get('cache_path')}, f)


def _load_session(job_id: str) -> dict | None:
    path = SESSIONS_DIR / f"{job_id}.json"
    if not path.exists():
        return None
    job = json.loads(path.read_text(encoding="utf-8"))
    job['progress'] = 1.0
    frames = SESSIONS_DIR / f"{job_id}_frames.pkl"
    if frames.exists():
        with open(frames, 'rb') as f:
            job.update(pickle.load(f))
    JOBS[job_id] = job
    return job


def _job(job_id: str, user: User) -> dict:
    if not job_id.isalnum():
        raise HTTPException(404, "unknown job")
    job = JOBS.get(job_id) or _load_session(job_id)
    if job is None or (user.mode != "local" and job.get('uid') != user.uid):
        raise HTTPException(404, "unknown job")   # 404, not 403: don't reveal other users' jobs
    return job


# ── analysis job ─────────────────────────────────────────────────────────────

def _run_job(job_id: str, video: Path, config_name: str, yolo_conf: float):
    job = JOBS[job_id]
    try:
        with _LOCK:
            job['status'] = 'loading models'
            pred = _get_predictor(config_name, yolo_conf)

            def progress(i, n):
                job['status'] = 'analyzing'
                job['progress'] = i / max(n, 1)

            job['status'] = 'analyzing'
            _, stats = pred.process_video(str(video), display=False, save_video=False,
                                          verbose=False, progress_cb=progress)
            job['shot_frames'] = dict(pred.shot_frames)
            job['cache_path'] = str(pred._det_cache_path(str(video)))

        labels = _shot_labels(video.name)
        shots = []
        for s in stats['shots']:
            frames = job['shot_frames'].get(s['shot_number'], [])
            feats = sf.shot_to_features(np.array(frames, dtype=np.float64)) if frames else None
            s = dict(s)
            s['rim_offset'] = round(sf.rim_pass_offset(feats), 3) if feats is not None else None
            s['label'] = labels.get(int(s['start_frame']))
            shots.append(s)

        info = stats['video_info']
        job.update(status='done', progress=1.0, result={
            'video': video.name, 'fps': info['fps'], 'total_frames': info['total_frames'],
            'width': info['width'], 'height': info['height'],
            'threshold': stats['model_info']['optimal_threshold'],
            'shots': shots,
        })
        _save_session(job)
        try:
            FIRESTORE.save_session(job['uid'], job)
        except Exception:
            traceback.print_exc()   # results are safe on disk; sync is best-effort
    except Exception as e:
        traceback.print_exc()
        job.update(status='error', error=f"{type(e).__name__}: {e}")


# ── API ──────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {'ok': True, 'auth_mode': AUTH.mode, 'firestore': FIRESTORE.enabled, 'api_version': API_VERSION}


@app.get("/api/status")
def status(user: User = Depends(current_user)):
    import torch
    return {
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'gpu': torch.cuda.is_available(),
        'yolo_model': YOLO_MODEL.name,
        'yolo_found': YOLO_MODEL.exists(),
        'predictors': _predictor_configs(),
        'default_predictor': RIM_RULE_CONFIG.name,
        'user': {'uid': user.uid, 'mode': user.mode, 'email': user.email},
    }


@app.get("/api/videos")
def videos(user: User = Depends(current_user)):
    out = []
    for p in sorted(RAW_VIDEO_DIR.iterdir()) if RAW_VIDEO_DIR.exists() else []:
        if p.suffix.lower() in VIDEO_EXTS:
            st = p.stat()
            out.append({'name': p.name, 'size_mb': round(st.st_size / 1e6, 1), 'mtime': st.st_mtime})
    out.sort(key=lambda v: -v['mtime'])
    return out


@app.post("/api/videos")
async def upload_video(file: UploadFile = File(...), user: User = Depends(current_user)):
    """Store an uploaded clip as data/raw/videos/app_<time>_<name> (original quality)."""
    original = Path(file.filename or "upload.mp4").name
    if Path(original).suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(400, "unsupported video type")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in original)
    name = f"app_{time.strftime('%Y%m%d_%H%M%S')}_{safe}"
    RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_VIDEO_DIR / name
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, 'wb') as f:
        while chunk := await file.read(8 << 20):
            f.write(chunk)
    tmp.replace(dest)   # never leave a half-written video under its final name
    return {'name': name, 'size_mb': round(dest.stat().st_size / 1e6, 1)}


@app.get("/api/videos/{name}/stream")
def stream_video(name: str, request: Request, user: User = Depends(current_user)):
    """Serve the video with HTTP range support so the browser can seek."""
    path = _video_path(name)
    size = path.stat().st_size
    rng = request.headers.get('range')
    if not rng:
        return FileResponse(path, media_type='video/mp4', headers={'Accept-Ranges': 'bytes'})
    start_s, _, end_s = rng.replace('bytes=', '').partition('-')
    start = int(start_s or 0)
    end = min(int(end_s) if end_s else start + (4 << 20) - 1, size - 1)

    def chunks():
        with open(path, 'rb') as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                data = f.read(min(1 << 20, left))
                if not data:
                    break
                left -= len(data)
                yield data

    return StreamingResponse(chunks(), status_code=206, media_type='video/mp4', headers={
        'Content-Range': f'bytes {start}-{end}/{size}',
        'Accept-Ranges': 'bytes',
        'Content-Length': str(end - start + 1),
    })


class AnalyzeRequest(BaseModel):
    video: str
    config: str = RIM_RULE_CONFIG.name
    yolo_conf: float = 0.75


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, user: User = Depends(current_user)):
    video = _video_path(req.video)
    if Path(req.config).name != req.config or not (MODEL_DIR / req.config).exists():
        raise HTTPException(404, f"predictor not found: {req.config}")
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {'id': job_id, 'uid': user.uid, 'status': 'queued', 'progress': 0.0,
                    'video': video.name, 'config': req.config,
                    'created': time.strftime('%Y-%m-%dT%H:%M:%S')}
    threading.Thread(target=_run_job, args=(job_id, video, req.config, req.yolo_conf),
                     daemon=True).start()
    return {'job_id': job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, user: User = Depends(current_user)):
    job = _job(job_id, user)
    return {k: v for k, v in job.items()
            if k in ('id', 'status', 'progress', 'video', 'config', 'created', 'error', 'result')}


@app.get("/api/sessions")
def sessions(user: User = Depends(current_user)):
    out = []
    for path in SESSIONS_DIR.glob("*.json") if SESSIONS_DIR.exists() else []:
        try:
            s = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if user.mode != "local" and s.get('uid') != user.uid:
            continue
        shots = s.get('result', {}).get('shots', [])
        out.append({
            'id': s['id'], 'video': s['video'], 'config': s['config'], 'created': s['created'],
            'total': len(shots),
            'made': sum(1 for x in shots if (x['label'] if x.get('label') is not None else x['prediction']) == 1),
            'labeled': sum(1 for x in shots if x.get('label') is not None),
        })
    out.sort(key=lambda s: s['created'], reverse=True)
    return out


@app.get("/api/detections/{job_id}")
def detections(job_id: str, user: User = Depends(current_user)):
    """Boxes per frame as [frame, cls, x1, y1, x2, y2] rows (from the detection cache)."""
    job = _job(job_id, user)
    if job.get('status') != 'done' or not Path(job.get('cache_path') or '').exists():
        raise HTTPException(409, "detections not ready")
    boxes = np.load(job['cache_path'])['boxes']
    return {'boxes': np.round(boxes).astype(int).tolist(),
            'classes': {infer.RealTimeShotPredictor.CLS_BALL: 'ball',
                        infer.RealTimeShotPredictor.CLS_BASKET: 'basket',
                        infer.RealTimeShotPredictor.CLS_PLAYER: 'player'}}


class Feedback(BaseModel):
    job_id: str
    shot_number: int
    label: int      # 1 made, 0 missed


@app.post("/api/feedback")
def feedback(fb: Feedback, user: User = Depends(current_user)):
    """Save the shot's raw frames with the human label, in labeler-CSV format."""
    if fb.label not in (0, 1):
        raise HTTPException(400, "label must be 0 (missed) or 1 (made)")
    job = _job(fb.job_id, user)
    frames = job.get('shot_frames', {}).get(fb.shot_number)
    shot = next((s for s in (job.get('result') or {}).get('shots', [])
                 if s['shot_number'] == fb.shot_number), None)
    if frames is None or shot is None:
        raise HTTPException(404, "shot not found in this analysis")

    df = pd.DataFrame(frames, columns=sf.RAW_COLS)
    start = int(shot['start_frame'])
    df['frame_count'] = start + np.arange(len(df))
    df['frame_within_shot'] = np.arange(len(df))
    df['shot_result'] = int(fb.label)
    df['model_pred'] = int(shot['prediction'])
    df['model_prob'] = shot['probability']
    df['predictor'] = job['config']
    df['source_video'] = job['video']

    old = _load_feedback()
    if not old.empty:
        # relabeling the same shot replaces the previous rows
        same = (old['source_video'] == job['video']) & (old['frame_count'].between(start, start + len(df) - 1))
        old_shots = old.loc[same, 'shot_number'].unique()
        old = old[~old['shot_number'].isin(old_shots)]
    next_id = int(old['shot_number'].max()) + 1 if not old.empty else 1
    df['shot_number'] = next_id
    out = pd.concat([old, df], ignore_index=True) if not old.empty else df
    FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FEEDBACK_CSV, index=False)

    shot['label'] = int(fb.label)
    _save_session(job)
    try:
        FIRESTORE.update_label(job['uid'], job)
    except Exception:
        traceback.print_exc()
    return {'saved': True, 'labeled_shots': int(out['shot_number'].nunique())}


@app.get("/api/stats")
def stats(user: User = Depends(current_user)):
    df = _load_feedback()
    if df.empty:
        return {'labeled_shots': 0, 'videos': 0}
    shots = df.groupby('shot_number').agg(label=('shot_result', 'first'), pred=('model_pred', 'first'),
                                          video=('source_video', 'first'))
    return {
        'labeled_shots': int(len(shots)),
        'videos': int(shots['video'].nunique()),
        'made': int(shots['label'].sum()),
        'fg_pct': float(shots['label'].mean()),
        'model_accuracy': float((shots['label'] == shots['pred']).mean()),
        'csv': str(FEEDBACK_CSV),
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    host = os.environ.get("HOOPER_HOST", "127.0.0.1")   # 0.0.0.0 to reach it from a phone
    port = int(os.environ.get("HOOPER_PORT", "8000"))
    AUTH.check_host(host)
    print(f"HooperAI server → http://{host}:{port}   (auth: {AUTH.mode}, "
          f"firestore sync: {'on' if FIRESTORE.enabled else 'off'})")
    uvicorn.run(app, host=host, port=port)
