import AsyncStorage from '@react-native-async-storage/async-storage';

// Every account on this phone gets its own cache: keys end in the Firebase uid
// ('local' when accounts are off). Set by RootNavigator before any screen loads.
let owner = 'local';
const LEGACY_KEYS = ['hooperai_shots', 'hooperai_sessions', 'hooperai_pending_job'];

const scoped = (base) => `${base}_${owner}`;
const shotsKey = () => scoped('hooperai_shots');
const sessionsKey = () => scoped('hooperai_sessions');
export const pendingJobKey = () => scoped('hooperai_pending_job');

/** Switch the cache to this account. The old shared cache can't be told apart by
 *  account, so it is dropped once – the laptop re-syncs each account's sessions. */
export async function setStorageOwner(uid) {
  owner = uid || 'local';
  try { await AsyncStorage.multiRemove(LEGACY_KEYS); } catch (_) { /* nothing to drop */ }
}
export const NOT_A_SHOT = -1;   // label for detections that were not real shots

export async function saveShot(shot) {
  try {
    const raw = await AsyncStorage.getItem(shotsKey());
    const shots = raw ? JSON.parse(raw) : [];
    shots.unshift({ ...shot, id: Date.now(), timestamp: new Date().toISOString() });
    await AsyncStorage.setItem(shotsKey(), JSON.stringify(shots));
    return shots;
  } catch (e) {
    console.error('saveShot error', e);
    return [];
  }
}

export async function getShots() {
  try {
    const raw = await AsyncStorage.getItem(shotsKey());
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    return [];
  }
}

export async function clearShots() {
  await AsyncStorage.removeItem(shotsKey());
}

// ── Analyzed sessions (from the laptop server) ──────────────────────────────

/** Shot record for the Home/Stats screens from a server shot. */
function toShot(sessionId, s, fps, timestamp) {
  const labeled = s.label != null;
  // shots you added have no prediction; treat your label as the model's for stats
  const predicted = s.prediction == null ? (s.label === 1 ? 'made' : 'missed')
    : s.prediction === 1 ? 'made' : 'missed';
  const outcome = labeled ? (s.label === 1 ? 'made' : 'missed') : predicted;
  return {
    id: `${sessionId}-${s.shot_number}`,
    sessionId,
    shotNumber: s.shot_number,
    outcome,
    predicted,
    labeled,
    manual: !!s.manual,
    correct: labeled && s.prediction != null ? outcome === predicted : null,
    // certainty of the predicted outcome, in %
    confidence: s.probability == null ? null : Math.round(Math.max(s.probability, 1 - s.probability) * 100),
    videoTime: fps ? (s.start_frame - 1) / fps : null,
    zone: null,
    timestamp,
  };
}

/** Store an analyzed session and its shots (replaces earlier copies of it).
 *  extra.localUri: the original clip on this phone, for review playback.
 *  extra.timestamp: when the session was analyzed (default: now / previous). */
export async function saveSession(sessionId, result, extra = {}) {
  try {
    const all = await getSessions();
    const previous = all.find(s => s.id === sessionId);
    const timestamp = extra.timestamp ?? previous?.timestamp ?? new Date().toISOString();

    const shots = (await getShots()).filter(s => s.sessionId !== sessionId);
    const fresh = result.shots.filter(s => s.label !== NOT_A_SHOT)
      .map(s => toShot(sessionId, s, result.fps, timestamp)).reverse();
    const next = [...fresh, ...shots]
      .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    await AsyncStorage.setItem(shotsKey(), JSON.stringify(next));

    const sessions = all.filter(s => s.id !== sessionId);
    sessions.push({
      id: sessionId,
      video: result.video,               // name on the laptop (for streaming it back)
      localUri: extra.localUri ?? previous?.localUri ?? null,   // original clip on this phone
      result,                            // full server result: shots, fps, labels
      timestamp,
    });
    sessions.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    await AsyncStorage.setItem(sessionsKey(), JSON.stringify(sessions.slice(0, 100)));
    return next;
  } catch (e) {
    console.error('saveSession error', e);
    return [];
  }
}

export async function getSessions() {
  try {
    const raw = await AsyncStorage.getItem(sessionsKey());
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
  return summarizeShots(session?.result?.shots || []);
}

/** Same counts as the laptop's /api/sessions: "not a shot" detections are left
 *  out of total/made but count as reviewed. */
export function summarizeShots(shots) {
  const real = shots.filter(s => s.label !== NOT_A_SHOT);
  const made = real.filter(s => (s.label ?? s.prediction) === 1).length;
  const labeled = shots.filter(s => s.label != null).length;
  return { total: real.length, made, labeled, pct: real.length ? Math.round((made / real.length) * 100) : null };
}

/** Apply your label (1 made, 0 missed, -1 not a shot) to a stored session. */
export async function labelShot(sessionId, shotNumber, label) {
  const session = await getSession(sessionId);
  if (!session) return getShots();
  const result = {
    ...session.result,
    shots: session.result.shots.map(x => (x.shot_number === shotNumber ? { ...x, label } : x)),
  };
  return saveSession(sessionId, result);   // rebuilds this session's shots for Home/Stats
}

/** How often the model agreed with your labels. */
export function calcModelStats(shots) {
  // shots you added yourself had no prediction to agree with
  const labeled = shots.filter(s => s.labeled && s.correct != null);
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
