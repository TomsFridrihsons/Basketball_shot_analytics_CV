"""
auth.py  –  Who is calling the test-bench API
=============================================
The mode is picked from environment variables when the server starts:

  firebase  FIREBASE_CREDENTIALS=<service-account.json>
            Clients send "Authorization: Bearer <Firebase ID token>"; the token is
            verified with firebase-admin (free) and its uid owns the sessions.
  api_key   HOOPER_API_KEY=<secret>
            Clients send "X-API-Key: <secret>" (or Bearer <secret>). One shared user.
  none      Neither set. Only allowed when the server listens on localhost,
            unless HOOPER_INSECURE=1.

Requests from the laptop itself (127.0.0.1 / ::1) are trusted without
credentials so the browser test bench keeps working, unless
HOOPER_TRUST_LOCALHOST=0 (set that when a tunnel such as cloudflared forwards
traffic from localhost; Tailscale traffic does not come from localhost).
"""

import os
import hmac
from dataclasses import dataclass

from fastapi import HTTPException, Request

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


@dataclass
class User:
    uid: str
    mode: str          # 'firebase' | 'api_key' | 'local'
    email: str | None = None


class Auth:
    def __init__(self):
        self.credentials = os.environ.get("FIREBASE_CREDENTIALS")
        self.api_key = os.environ.get("HOOPER_API_KEY")
        self.trust_localhost = os.environ.get("HOOPER_TRUST_LOCALHOST", "1") != "0"
        self.firebase_app = None
        if self.credentials:
            import firebase_admin
            from firebase_admin import credentials
            self.firebase_app = firebase_admin.initialize_app(credentials.Certificate(self.credentials))
            self.mode = "firebase"
        elif self.api_key:
            self.mode = "api_key"
        else:
            self.mode = "none"

    def check_host(self, host: str):
        """Refuse to expose an unauthenticated API beyond this laptop."""
        if self.mode == "none" and host not in LOCAL_HOSTS and os.environ.get("HOOPER_INSECURE") != "1":
            raise SystemExit(
                f"Refusing to listen on {host} without authentication.\n"
                "Set HOOPER_API_KEY=<secret> (or FIREBASE_CREDENTIALS=<service-account.json>),\n"
                "or HOOPER_INSECURE=1 if you really want an open server on your network."
            )

    def verify_firebase(self, token: str) -> User:
        from firebase_admin import auth as fb_auth
        claims = fb_auth.verify_id_token(token, app=self.firebase_app)
        return User(uid=claims["uid"], mode="firebase", email=claims.get("email"))

    def user(self, request: Request) -> User:
        """FastAPI dependency: the calling user, or 401."""
        client = request.client.host if request.client else ""
        if self.mode == "none" or (self.trust_localhost and client in LOCAL_HOSTS):
            return User(uid="local", mode="local")

        header = request.headers.get("authorization", "")
        bearer = header[7:].strip() if header.lower().startswith("bearer ") else None
        api_key = request.headers.get("x-api-key")
        # <video src> cannot send headers, so the stream endpoint also accepts ?token=
        query = request.query_params.get("token")
        # The app may send both a Firebase token and an API key (while the server
        # has no service account yet); use the one this mode understands.
        supplied = (api_key or bearer or query) if self.mode == "api_key" else (bearer or query)
        if not supplied:
            raise HTTPException(401, "missing credentials")

        if self.mode == "firebase":
            try:
                return self.verify_firebase(supplied)
            except Exception as e:
                raise HTTPException(401, f"invalid Firebase ID token: {type(e).__name__}")
        if hmac.compare_digest(supplied, self.api_key):
            return User(uid="api_key", mode="api_key")
        raise HTTPException(401, "invalid API key")
