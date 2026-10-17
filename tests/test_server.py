"""
API tests for scripts/app/server.py without YOLO, GPU or Firebase:
the predictor and Firestore are replaced by fakes, all paths point to tmp dirs.

    python -m pytest tests -q
"""

import json
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "app"))
import server  # noqa: E402
import shot_features as sf  # noqa: E402
from auth import User  # noqa: E402

LABELED_CSV = ROOT / "data" / "processed" / "annotations" / "basketball_shot_data_labeled_04_05_02.csv"


def _real_shots():
    """Two real labeled shots (raw frames) so rim offsets are computed for real."""
    head = LABELED_CSV.read_text()[:500]
    df = pd.read_csv(LABELED_CSV, sep=';' if head.count(';') > head.count(',') else ',')
    groups = [g for _, g in df.groupby('shot_number')][:2]
    return [g[sf.RAW_COLS].to_numpy().tolist() for g in groups]


class FakePredictor:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.shot_frames = {}

    def _det_cache_path(self, video_path):
        return self.cache_path

    def process_video(self, video_path, display, save_video, verbose, progress_cb=None):
        frames = _real_shots()
        self.shot_frames = {1: frames[0], 2: frames[1]}
        if progress_cb:
            progress_cb(150, 300)
        shots = [
            {'shot_number': 1, 'prediction': 1, 'prediction_text': 'MADE', 'probability': 0.9,
             'confidence': 'HIGH', 'start_frame': 10, 'end_frame': 10 + len(frames[0]), 'n_frames': len(frames[0])},
            {'shot_number': 2, 'prediction': 0, 'prediction_text': 'MISSED', 'probability': 0.1,
             'confidence': 'HIGH', 'start_frame': 200, 'end_frame': 200 + len(frames[1]), 'n_frames': len(frames[1])},
        ]
        return None, {'shots': shots,
                      'video_info': {'fps': 30.0, 'total_frames': 300, 'width': 1920, 'height': 1080},
                      'model_info': {'optimal_threshold': 0.5}}


class FakeFirestore:
    enabled = True

    def __init__(self):
        self.saved = []

    def save_session(self, uid, session):
        self.saved.append((uid, session['id'], [s.get('label') for s in session['result']['shots']]))

    update_label = save_session


@pytest.fixture
def env(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    models = tmp_path / "model"
    videos.mkdir()
    models.mkdir()
    (videos / "clip.mp4").write_bytes(b"\x00" * 1024)
    with open(models / "basketball_shot_rim_rule_config.pkl", "wb") as f:
        pickle.dump({'model_type': 'rim_rule', 'rim_threshold': 0.45}, f)
    cache = tmp_path / "det.npz"
    np.savez_compressed(cache, boxes=np.array([[1, 0, 10, 10, 20, 20]], dtype=np.float32))

    monkeypatch.setattr(server, "RAW_VIDEO_DIR", videos)
    monkeypatch.setattr(server, "MODEL_DIR", models)
    monkeypatch.setattr(server, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(server, "FEEDBACK_CSV", tmp_path / "app_labeled.csv")
    monkeypatch.setattr(server, "NOT_SHOTS_CSV", tmp_path / "app_not_shots.csv")
    monkeypatch.setattr(server, "_get_predictor", lambda config, conf: FakePredictor(cache))
    fs = FakeFirestore()
    monkeypatch.setattr(server, "FIRESTORE", fs)
    monkeypatch.setattr(server.AUTH, "mode", "api_key")
    monkeypatch.setattr(server.AUTH, "api_key", "secret")
    monkeypatch.setattr(server.AUTH, "trust_localhost", True)
    server.JOBS.clear()
    return {'tmp': tmp_path, 'firestore': fs, 'client': TestClient(server.app)}


KEY = {"X-API-Key": "secret"}


def _analyze(client, headers=KEY, video="clip.mp4"):
    r = client.post("/api/analyze", json={"video": video}, headers=headers)
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    for _ in range(600):   # up to 30 s: the laptop may be busy (Expo bundling, live server)
        job = client.get(f"/api/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("done", "error"):
            return job_id, job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health_needs_no_auth(env):
    r = env['client'].get("/api/health")
    assert r.status_code == 200 and r.json()["auth_mode"] == "api_key"


def test_api_key_required(env):
    c = env['client']
    assert c.get("/api/videos").status_code == 401
    assert c.get("/api/videos", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/api/videos", headers=KEY).status_code == 200
    assert c.get("/api/videos", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_api_key_mode_accepts_app_sending_firebase_token_too(env):
    # The app sends its Firebase ID token and the API key; an API-key server uses the key
    both = {"Authorization": "Bearer some-firebase-id-token", "X-API-Key": "secret"}
    assert env['client'].get("/api/videos", headers=both).status_code == 200
    wrong = {"Authorization": "Bearer some-firebase-id-token", "X-API-Key": "nope"}
    assert env['client'].get("/api/videos", headers=wrong).status_code == 401


def test_analyze_persists_session(env):
    c = env['client']
    job_id, job = _analyze(c)
    assert job["status"] == "done", job
    shots = job["result"]["shots"]
    assert [s["shot_number"] for s in shots] == [1, 2]
    assert all(s["rim_offset"] is not None for s in shots)
    assert (env['tmp'] / "sessions" / f"{job_id}.json").exists()

    # survives a restart: drop the in-memory job and read it back from disk
    server.JOBS.clear()
    assert c.get(f"/api/jobs/{job_id}", headers=KEY).json()["result"]["shots"][0]["shot_number"] == 1
    listed = c.get("/api/sessions", headers=KEY).json()
    assert listed[0]["id"] == job_id and listed[0]["total"] == 2


def test_feedback_writes_training_rows_and_relabels(env):
    c = env['client']
    job_id, _ = _analyze(c)
    r = c.post("/api/feedback", json={"job_id": job_id, "shot_number": 1, "label": 1}, headers=KEY)
    assert r.status_code == 200, r.text
    df = pd.read_csv(env['tmp'] / "app_labeled.csv")
    assert set(sf.RAW_COLS) <= set(df.columns)
    assert {'shot_number', 'frame_count', 'shot_result', 'source_video'} <= set(df.columns)
    assert df['shot_result'].unique().tolist() == [1]

    # relabel the same shot → replaced, not duplicated
    c.post("/api/feedback", json={"job_id": job_id, "shot_number": 1, "label": 0}, headers=KEY)
    df = pd.read_csv(env['tmp'] / "app_labeled.csv")
    assert df['shot_number'].nunique() == 1 and df['shot_result'].unique().tolist() == [0]

    # label survives restart and is reported on the session
    server.JOBS.clear()
    assert c.get(f"/api/jobs/{job_id}", headers=KEY).json()["result"]["shots"][0]["label"] == 0
    assert c.post("/api/feedback", json={"job_id": job_id, "shot_number": 1, "label": 5},
                  headers=KEY).status_code == 400


def test_not_a_shot_moves_rows_out_of_training_data(env):
    c = env['client']
    job_id, _ = _analyze(c)
    c.post("/api/feedback", json={"job_id": job_id, "shot_number": 1, "label": 1}, headers=KEY)
    r = c.post("/api/feedback", json={"job_id": job_id, "shot_number": 1, "label": -1}, headers=KEY)
    assert r.status_code == 200, r.text
    # removed from the made/missed file, stored as a false detection instead
    labeled = pd.read_csv(env['tmp'] / "app_labeled.csv")
    assert labeled.empty or 1 not in labeled['shot_result'].tolist()
    not_shots = pd.read_csv(env['tmp'] / "app_not_shots.csv")
    assert not_shots['shot_result'].unique().tolist() == [-1]
    # excluded from totals, but counted as reviewed
    assert r.json()['summary'] == {'total': 1, 'made': 0, 'labeled': 1}
    s = [x for x in c.get("/api/sessions", headers=KEY).json() if x['id'] == job_id][0]
    assert (s['total'], s['labeled']) == (1, 1)
    # changing your mind moves it back
    c.post("/api/feedback", json={"job_id": job_id, "shot_number": 1, "label": 0}, headers=KEY)
    assert pd.read_csv(env['tmp'] / "app_not_shots.csv").empty
    assert pd.read_csv(env['tmp'] / "app_labeled.csv")['shot_result'].tolist()[0] == 0


def test_manual_shot_added_with_frames_from_cache(env):
    c = env['client']
    job_id, _ = _analyze(c)
    r = c.post("/api/manual_shot", json={"job_id": job_id, "time_s": 4.0, "label": 1}, headers=KEY)
    assert r.status_code == 200, r.text
    body = r.json()
    shot = body['shot']
    assert shot['manual'] and shot['prediction'] is None and shot['label'] == 1
    # 2.0 s before and 0.7 s after the marked moment (frame 121 at 30 fps)
    assert (shot['start_frame'], shot['end_frame']) == (61, 142)
    assert [s['shot_number'] for s in body['result']['shots']] == [1, 3, 2]   # sorted by time
    assert body['summary'] == {'total': 3, 'made': 2, 'labeled': 1}
    rows = pd.read_csv(env['tmp'] / "app_labeled.csv")
    assert len(rows) == 142 - 61 + 1 and rows['model_pred'].unique().tolist() == [-1]
    # survives a restart and can be relabeled like any shot
    server.JOBS.clear()
    assert c.post("/api/feedback", json={"job_id": job_id, "shot_number": 3, "label": 0},
                  headers=KEY).status_code == 200
    assert c.post("/api/manual_shot", json={"job_id": job_id, "time_s": 1.0, "label": 2},
                  headers=KEY).status_code == 400


def test_firebase_users_are_isolated(env, monkeypatch):
    monkeypatch.setattr(server.AUTH, "mode", "firebase")
    monkeypatch.setattr(server.AUTH, "verify_firebase",
                        lambda token: User(uid=token.replace("tok-", ""), mode="firebase"))
    alice = {"Authorization": "Bearer tok-alice"}
    bob = {"Authorization": "Bearer tok-bob"}
    c = env['client']

    name = c.post("/api/videos", files={"file": ("a.mp4", b"abc" * 100, "video/mp4")},
                  headers=alice).json()["name"]
    job_id, job = _analyze(c, alice, video=name)
    assert job["status"] == "done"
    assert env['firestore'].saved[-1][:2] == ("alice", job_id)
    assert c.get(f"/api/jobs/{job_id}", headers=bob).status_code == 404
    assert c.get("/api/sessions", headers=bob).json() == []
    assert c.post("/api/feedback", json={"job_id": job_id, "shot_number": 2, "label": 0},
                  headers=bob).status_code == 404

    c.post("/api/feedback", json={"job_id": job_id, "shot_number": 2, "label": 0}, headers=alice)
    assert env['firestore'].saved[-1] == ("alice", job_id, [None, 0])


def _firebase(monkeypatch):
    monkeypatch.setattr(server.AUTH, "mode", "firebase")
    monkeypatch.setattr(server.AUTH, "verify_firebase",
                        lambda token: User(uid=token.replace("tok-", ""), mode="firebase"))
    return {"Authorization": "Bearer tok-alice"}, {"Authorization": "Bearer tok-bob"}


def test_firebase_users_only_see_their_own_videos(env, monkeypatch):
    alice, bob = _firebase(monkeypatch)
    c = env['client']
    name = c.post("/api/videos", files={"file": ("a.mp4", b"abc" * 100, "video/mp4")},
                  headers=alice).json()["name"]

    assert [v["name"] for v in c.get("/api/videos", headers=alice).json()] == [name]
    assert c.get("/api/videos", headers=bob).json() == []
    assert c.get(f"/api/videos/{name}/stream", headers=bob).status_code == 404
    assert c.post("/api/analyze", json={"video": name}, headers=bob).status_code == 404
    # videos copied onto the laptop by hand belong to nobody: not for Firebase users
    assert c.post("/api/analyze", json={"video": "clip.mp4"}, headers=bob).status_code == 404
    assert c.get("/api/stats", headers=bob).status_code == 403

    job_id, job = _analyze(c, alice, video=name)
    assert job["status"] == "done"


def test_upload_keeps_original_and_prefixes_name(env):
    r = env['client'].post("/api/videos", files={"file": ("my shot.mp4", b"abc" * 100, "video/mp4")},
                           headers=KEY)
    assert r.status_code == 200, r.text
    name = r.json()["name"]
    assert name.startswith("app_") and name.endswith("my_shot.mp4")
    assert (env['tmp'] / "videos" / name).read_bytes() == b"abc" * 100
    assert env['client'].post("/api/videos", files={"file": ("x.exe", b"1", "application/octet-stream")},
                              headers=KEY).status_code == 400


def test_path_traversal_rejected(env):
    c = env['client']
    assert c.post("/api/analyze", json={"video": "../secret.mp4"}, headers=KEY).status_code == 404
    assert c.post("/api/analyze", json={"video": "clip.mp4", "config": "../x_config.pkl"},
                  headers=KEY).status_code == 404
    assert c.get("/api/jobs/..%2Fetc", headers=KEY).status_code == 404


def test_refuses_open_network_without_auth(monkeypatch):
    from auth import Auth
    monkeypatch.delenv("FIREBASE_CREDENTIALS", raising=False)
    monkeypatch.delenv("HOOPER_API_KEY", raising=False)
    monkeypatch.delenv("HOOPER_INSECURE", raising=False)
    a = Auth()
    a.check_host("127.0.0.1")          # fine
    with pytest.raises(SystemExit):
        a.check_host("0.0.0.0")


def test_earlier_uploads_get_the_owner_of_their_session(env, monkeypatch):
    alice, bob = _firebase(monkeypatch)
    sessions = env['tmp'] / "sessions"
    sessions.mkdir(exist_ok=True)
    (env['tmp'] / "videos" / "app_old.mp4").write_bytes(b"\x00" * 10)
    (sessions / "abc.json").write_text(json.dumps(
        {'id': 'abc', 'uid': 'alice', 'video': 'app_old.mp4', 'config': 'x', 'created': '2026-10-01T10:00:00',
         'status': 'done', 'result': {'shots': []}}), encoding="utf-8")

    assert server.backfill_video_owners() == 1
    c = env['client']
    assert [v["name"] for v in c.get("/api/videos", headers=alice).json()] == ["app_old.mp4"]
    assert c.get("/api/videos", headers=bob).json() == []
    assert [s["id"] for s in c.get("/api/sessions", headers=alice).json()] == ["abc"]
