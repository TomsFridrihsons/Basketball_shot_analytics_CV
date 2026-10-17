import AsyncStorage from '@react-native-async-storage/async-storage';

const SHOTS_KEY = 'hooperai_shots';
const SESSION_KEY = 'hooperai_sessions';

export async function saveShot(shot) {
  try {
    const raw = await AsyncStorage.getItem(SHOTS_KEY);
    const shots = raw ? JSON.parse(raw) : [];
    shots.unshift({ ...shot, id: Date.now(), timestamp: new Date().toISOString() });
    await AsyncStorage.setItem(SHOTS_KEY, JSON.stringify(shots));
    return shots;
  } catch (e) {
    console.error('saveShot error', e);
    return [];
  }
}

export async function getShots() {
  try {
    const raw = await AsyncStorage.getItem(SHOTS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    return [];
  }
}

export async function clearShots() {
  await AsyncStorage.removeItem(SHOTS_KEY);
}

// ── Analyzed sessions (from the laptop server) ──────────────────────────────

/** Shot record for the Home/Stats screens from a server shot. */
function toShot(sessionId, s, fps) {
  const labeled = s.label != null;
  const predicted = s.prediction === 1 ? 'made' : 'missed';
  const outcome = labeled ? (s.label === 1 ? 'made' : 'missed') : predicted;
  return {
    id: `${sessionId}-${s.shot_number}`,
    sessionId,
    shotNumber: s.shot_number,
    outcome,
    predicted,
    labeled,
    correct: labeled ? outcome === predicted : null,
    // certainty of the predicted outcome, in %
    confidence: Math.round(Math.max(s.probability, 1 - s.probability) * 100),
    videoTime: fps ? (s.start_frame - 1) / fps : null,
    zone: null,
    timestamp: new Date().toISOString(),
  };
}

/** Store an analyzed session and its shots (replaces earlier copies of it).
 *  extra.localUri: the original clip on this phone, for review playback. */
export async function saveSession(sessionId, result, extra = {}) {
  try {
    const shots = (await getShots()).filter(s => s.sessionId !== sessionId);
    const fresh = result.shots.map(s => toShot(sessionId, s, result.fps)).reverse();
    const next = [...fresh, ...shots];
    await AsyncStorage.setItem(SHOTS_KEY, JSON.stringify(next));

    const sessions = (await getSessions()).filter(s => s.id !== sessionId);
    const previous = (await getSessions()).find(s => s.id === sessionId);
    sessions.unshift({
      id: sessionId,
      video: result.video,               // name on the laptop (for streaming it back)
      localUri: extra.localUri ?? previous?.localUri ?? null,   // original clip on this phone
      result,                            // full server result: shots, fps, labels
      timestamp: previous?.timestamp ?? new Date().toISOString(),
    });
    await AsyncStorage.setItem(SESSION_KEY, JSON.stringify(sessions.slice(0, 100)));
    return next;
  } catch (e) {
    console.error('saveSession error', e);
    return [];
  }
}

export async function getSessions() {
  try {
    const raw = await AsyncStorage.getItem(SESSION_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    return [];
  }
}

export async function getSession(sessionId) {
  return (await getSessions()).find(s => s.id === sessionId) || null;
}

/** Summary numbers for a stored session (labels win over predictions). */
export function sessionSummary(session) {
  const shots = session?.result?.shots || [];
  const made = shots.filter(s => (s.label ?? s.prediction) === 1).length;
  const labeled = shots.filter(s => s.label != null).length;
  return { total: shots.length, made, labeled, pct: shots.length ? Math.round((made / shots.length) * 100) : null };
}

/** Apply your made/missed label to a stored shot and its session. */
export async function labelShot(sessionId, shotNumber, label) {
  const shots = await getShots();
  const next = shots.map(s => {
    if (s.sessionId !== sessionId || s.shotNumber !== shotNumber) return s;
    const outcome = label === 1 ? 'made' : 'missed';
    return { ...s, outcome, labeled: true, correct: outcome === s.predicted };
  });
  await AsyncStorage.setItem(SHOTS_KEY, JSON.stringify(next));

  const sessions = await getSessions();
  const updated = sessions.map(s => (s.id !== sessionId ? s : {
    ...s,
    result: { ...s.result, shots: s.result.shots.map(x => (x.shot_number === shotNumber ? { ...x, label } : x)) },
  }));
  await AsyncStorage.setItem(SESSION_KEY, JSON.stringify(updated));
  return next;
}

/** How often the model agreed with your labels. */
export function calcModelStats(shots) {
  const labeled = shots.filter(s => s.labeled);
  const right = labeled.filter(s => s.correct).length;
  return { labeled: labeled.length, right, pct: labeled.length ? Math.round((right / labeled.length) * 100) : null };
}

export function formatTimeAgo(isoString) {
  const diff = Date.now() - new Date(isoString).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export function calcStats(shots) {
  if (!shots.length) return { total: 0, made: 0, missed: 0, pct: 0 };
  const made = shots.filter(s => s.outcome === 'made').length;
  return {
    total:  shots.length,
    made,
    missed: shots.length - made,
    pct:    Math.round((made / shots.length) * 100),
  };
}

export function calcZoneStats(shots) {
  const zones = {};
  shots.forEach(s => {
    if (!s.zone) return;
    if (!zones[s.zone]) zones[s.zone] = { made: 0, total: 0 };
    zones[s.zone].total++;
    if (s.outcome === 'made') zones[s.zone].made++;
  });
  return Object.entries(zones).map(([name, d]) => ({
    name,
    pct:  Math.round((d.made / d.total) * 100),
    att:  d.total,
    made: d.made,
  })).sort((a, b) => b.att - a.att);
}
