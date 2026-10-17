"""
firestore_sync.py  –  Mirror session results to Firestore (Firebase mode only)
=============================================================================
Writes one small document per analyzed session and updates it when a shot is
labeled, so the mobile app can show history even when the laptop is off.
Stays far inside the free Spark quotas (a session doc is a few KB; one write
per analysis plus one per label). Videos are never uploaded to Firebase.

Layout:  users/{uid}/sessions/{session_id}
         { video, predictor, created, fps, total_frames,
           shots: [{shot_number, start_s, end_s, prediction, probability,
                    confidence, rim_offset, label}], made, total }
"""


class FirestoreSync:
    def __init__(self, firebase_app=None):
        self.db = None
        if firebase_app is not None:
            from firebase_admin import firestore
            self.db = firestore.client(app=firebase_app)

    @property
    def enabled(self) -> bool:
        return self.db is not None

    def _doc(self, uid: str, session_id: str):
        return self.db.collection("users").document(uid).collection("sessions").document(session_id)

    @staticmethod
    def _shot(s: dict, fps: float) -> dict:
        return {
            'shot_number': int(s['shot_number']),
            'start_s':     round((s['start_frame'] - 1) / fps, 2) if fps else None,
            'end_s':       round((s['end_frame'] - 1) / fps, 2) if fps else None,
            'prediction':  None if s.get('prediction') is None else int(s['prediction']),   # None: added by you
            'probability': None if s.get('probability') is None else float(s['probability']),
            'confidence':  s['confidence'],
            'rim_offset':  s.get('rim_offset'),
            'label':       s.get('label'),           # 1 made, 0 missed, -1 not a shot
            'manual':      bool(s.get('manual')),
        }

    def save_session(self, uid: str, session: dict):
        if not self.enabled or uid in ("local", "api_key"):
            return
        r = session['result']
        shots = [self._shot(s, r['fps']) for s in r['shots']]
        self._doc(uid, session['id']).set({
            'video':        r['video'],
            'predictor':    session['config'],
            'created':      session['created'],
            'fps':          r['fps'],
            'total_frames': r['total_frames'],
            'shots':        shots,
            'total':        sum(1 for s in shots if s['label'] != -1),
            'made':         sum(1 for s in shots if s['label'] != -1
                                and (s['label'] if s['label'] is not None else s['prediction']) == 1),
        })

    def update_label(self, uid: str, session: dict):
        # Rewrites the shots array (small) – simpler than per-element updates
        self.save_session(uid, session)
