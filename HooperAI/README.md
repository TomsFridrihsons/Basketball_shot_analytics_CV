# HooperAI Mobile App

React Native (Expo) companion app for the HooperAI basketball shot prediction system.

## Setup

```bash
npm install
npx expo start
```

Press `a` to open on Android, or scan the QR code with the Expo Go app.

## Project structure

```
src/
  screens/
    HomeScreen.js     – session summary + shot log
    RecordScreen.js   – live camera + prediction UI + feedback buttons
    StatsScreen.js    – FG% ring, shot zone breakdown, model info
  components/
    index.js          – shared UI components (cards, badges, buttons)
  navigation/
    TabNavigator.js   – bottom tab bar with custom record button
  utils/
    theme.js          – colors, spacing, typography tokens
    storage.js        – AsyncStorage helpers + stat calculations
```

## Connecting to the LSTM pipeline

RecordScreen.js has a `DEMO_SHOTS` array for UI testing.
Replace the `simulateShot()` function with a real connection to your backend:

### Option A — REST API (recommended for MVP)
Run a small Flask/FastAPI server on the same network that wraps `lstm_infer.py`:

```python
# server.py (add alongside lstm_infer.py)
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    frame_data = request.json  # receive frame features
    result = infer.predict_from_features(frame_data)
    return jsonify(result)  # { outcome, confidence, zone }
```

Then in RecordScreen.js:
```js
const res = await fetch('http://YOUR_PC_IP:5000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ features: currentFrameFeatures }),
});
const prediction = await res.json();
setPrediction(prediction);
```

### Option B — WebSocket (for lower latency)
Use a WebSocket connection so the PC pushes predictions as they happen,
rather than the app polling.

### Feedback labels
Each call to `handleFeedback()` calls `saveShot()` which stores the shot in
AsyncStorage. To export to `feedback_labels.csv` (compatible with your
existing training pipeline), add an export button in StatsScreen:

```js
import * as Sharing from 'expo-sharing';
import * as FileSystem from 'expo-file-system';

async function exportCSV() {
  const shots = await getShots();
  const csv = ['shot_number,frame,shot_result,feedback']
    .concat(shots.map((s, i) => `${i+1},0,${s.outcome === 'made' ? 1 : 0},${s.outcome}`))
    .join('\n');
  const path = FileSystem.documentDirectory + 'feedback_labels.csv';
  await FileSystem.writeAsStringAsync(path, csv);
  await Sharing.shareAsync(path);
}
```

## Shot zones
Zone classification in the app matches the 9 Ballogy-style zones from
`shot_location.py`. Make sure the zone string your backend returns matches
one of: Left corner 3, Top of key 3, Right corner 3, Left wing, Paint,
Right wing, Left elbow, Mid-range, Right elbow.

## Building for Android

```bash
npx expo build:android
# or with EAS:
npx eas build --platform android
```
