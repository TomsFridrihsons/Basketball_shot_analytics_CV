# Mobile app ↔ laptop server setup

The phone records (or picks) a video in original quality and uploads it to the laptop,
which runs YOLO + shot detection + the predictor on its GPU and sends back the shots.
You label each shot made/missed in the app; labels become training data.

Everything here is free: the laptop does the processing, Tailscale's personal plan
connects the phone away from home, and Firebase stays on the free **Spark** plan
(Authentication + Firestore only, no Storage, no Cloud Functions).

```
phone (Expo app) ──upload video──► laptop: scripts/app/server.py ──► YOLO (GPU) + rim-pass rule
       ▲                                │
       └──────── shots / labels ────────┘──► Firestore users/{uid}/sessions (Firebase mode only)
```

## 1. Run it now, without Firebase (API-key mode)

1. **Server secret.** Copy `secrets/server.env.example` to `secrets/server.env` and set
   `HOOPER_API_KEY` to a random value:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(24))"
   ```
2. **Start the server** on the laptop (from the repo root):
   ```bash
   .venv\Scripts\python.exe scripts\app\server.py
   ```
   It prints `auth: api_key`. On first start Windows asks about the firewall: allow
   **Private networks** only.
3. **Laptop address.** `ipconfig` → the Wi-Fi adapter's IPv4 address, e.g. `192.168.1.23`.
4. **App.** Install **Expo Go** on the phone, then on the laptop:
   ```bash
   cd HooperAI
   npm install
   npx expo start
   ```
   Scan the QR code. In the app open **Settings** (gear icon) → address
   `http://192.168.1.23:8000`, API key from step 1 → **Test connection** → Save.
5. **Use it.** Record tab → record or pick a video from the library → **Analyze on laptop**
   → label each shot **Made** / **Missed**. Labels land in
   `data/processed/annotations/app_labeled.csv` (add it to `DATA_PATHS` in
   `scripts/lstm/lstm_train.py` to retrain).

Optional: copy `HooperAI/.env.example` to `HooperAI/.env` and set
`EXPO_PUBLIC_SERVER_URL` so new installs start with the right address.

## 2. Away from home: Tailscale (free)

1. Install Tailscale on the laptop and the phone, sign in with the same account
   (personal plan, free).
2. Use the laptop's Tailscale IP (`100.x.y.z`, shown in the Tailscale app) as the server
   address in the app's Settings.

Traffic goes directly between your devices and is encrypted; nothing is exposed to the
internet. (If you use a tunnel that forwards from localhost instead, such as cloudflared,
set `HOOPER_TRUST_LOCALHOST=0` in `secrets/server.env`.)

## 3. Firebase accounts (still free)

**Status (2026-10-17), project `hooperai-4f9cc`:** steps 1–5 are done: Email/Password
sign-in enabled, Firestore created in `eur3` (production mode) with `firestore.rules`
published, web app "HooperAI mobile" registered and its config written to
`HooperAI/.env` (git-ignored). **Remaining: step 6 (service-account key) and step 7.**
Until step 6 the server stays in API-key mode; the app sends both its Firebase token and
the API key from Settings, so keep the API key filled in.

1. [console.firebase.google.com](https://console.firebase.google.com) → **Add project**.
   Google Analytics is not needed. **Stay on the Spark (free) plan; do not upgrade.**
2. **Build → Authentication → Get started → Email/Password → Enable.**
3. **Build → Firestore Database → Create database**, production mode, a European location
   (e.g. `eur3`). Then **Rules** → paste the contents of [`firestore.rules`](../firestore.rules)
   → Publish.
4. **Do not enable Storage** (videos never go to Firebase; new Storage buckets require the
   paid Blaze plan).
5. **Project settings → General → Your apps → Web app (`</>`)** → register (no hosting).
   Copy the config values into `HooperAI/.env` (see `HooperAI/.env.example`):
   `EXPO_PUBLIC_FIREBASE_API_KEY`, `..._AUTH_DOMAIN`, `..._PROJECT_ID`,
   `..._MESSAGING_SENDER_ID`, `..._APP_ID`.
6. **Project settings → Service accounts → Generate new private key.** Save the file as
   `secrets/firebase-adminsdk.json` (git-ignored, keep it private) and set in
   `secrets/server.env`:
   ```
   FIREBASE_CREDENTIALS=C:\Users\fridr\Documents\HooperAI\secrets\firebase-adminsdk.json
   ```
7. Restart the server (it prints `auth: firebase, firestore sync: on`) and
   `npx expo start --clear` (`--clear` is needed after editing `.env`; otherwise Metro
   can serve a cached bundle without the new values). The app now opens with sign-in; create
   your account there. Each analyzed session is mirrored to
   `users/{your uid}/sessions/{id}` in Firestore.

### What stays free

| Service | Use here | Free (Spark) limit |
|---|---|---|
| Authentication (email/password) | sign-in | free for email/password |
| Firestore | ~1 small doc per session + 1 write per label | 1 GiB, 50k reads / 20k writes per day |
| Tailscale personal | phone ↔ laptop | free |
| Laptop GPU | all video processing | your electricity |

## Troubleshooting

| Symptom | Fix |
|---|---|
| App: "laptop offline" | Server running with `HOOPER_HOST=0.0.0.0`? Phone on the same Wi-Fi / Tailscale? Firewall allowed for private networks? |
| `Not authorized` | API key in the app differs from `HOOPER_API_KEY`, or (Firebase mode) sign in again |
| Server exits with "Refusing to listen on 0.0.0.0 without authentication" | Set `HOOPER_API_KEY` or `FIREBASE_CREDENTIALS` in `secrets/server.env` |
| Upload slow over Tailscale away from home | Limited by your home internet's upload speed; on the same Wi-Fi it is much faster |
| No shots detected | Keep the hoop and the ball's flight in frame; try the Library option with a known clip |

## Development notes

- API: see the docstring at the top of `scripts/app/server.py`; the browser test bench
  (`http://127.0.0.1:8000` on the laptop) uses the same endpoints.
- Server tests (no GPU / Firebase needed): `python -m pytest tests -q`.
- App bundle check without a phone: `cd HooperAI && npx expo export --platform android`.
