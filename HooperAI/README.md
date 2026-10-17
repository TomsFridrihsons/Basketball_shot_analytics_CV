# HooperAI Mobile App

React Native (Expo SDK 51) app for HooperAI. It records or picks a basketball video,
sends it in original quality to the laptop server (`scripts/app/server.py`), and shows
the detected shots with made/missed predictions. You label each shot; labels go back to
the laptop as training data.

Full setup (server, Wi-Fi / Tailscale, Firebase): [`docs/MOBILE_SETUP.md`](../docs/MOBILE_SETUP.md).

## Run

```bash
npm install
npx expo start
```

Scan the QR code with Expo Go, open **Settings** (gear icon), enter the laptop server
address and API key, and press **Test connection**.

## Structure

```
App.js                       root: navigation container
src/
  config.js                  server URL default + Firebase web config (from .env)
  navigation/
    RootNavigator.js         sign-in gate (only when Firebase is configured) + stack
    TabNavigator.js          Home · Record · Stats tabs
  screens/
    HomeScreen.js            today's numbers + shot log
    RecordScreen.js          record / pick → upload → analyze → label shots
    StatsScreen.js           FG%, how often the model agreed with your labels
    SettingsScreen.js        server address, API key, predictor, account
    LoginScreen.js           email/password sign-in (Firebase)
  services/
    api.js                   laptop server client (upload, analyze, poll, feedback)
    auth.js                  Firebase Auth (inactive until configured)
    settings.js              AsyncStorage-backed settings
  components/index.js        shared UI pieces
  utils/
    theme.js                 colors, spacing, typography
    storage.js               local shot history + stats
```

## Configuration

Copy `.env.example` to `.env` (git-ignored) and restart `npx expo start`:

- `EXPO_PUBLIC_SERVER_URL`: initial server address (can be changed in Settings).
- `EXPO_PUBLIC_FIREBASE_*`: Firebase web app config. While these are empty the app
  runs without sign-in and authenticates to the server with the API key from Settings.

## Notes

- Videos are uploaded as recorded (1080p, no re-encoding); only results come back.
- Recording is muted, so no microphone permission is needed.
- Check that the app bundles without a phone: `npx expo export --platform android`.
- A store build needs EAS (`npx eas build --platform android`); the free tier is enough
  for one developer.
