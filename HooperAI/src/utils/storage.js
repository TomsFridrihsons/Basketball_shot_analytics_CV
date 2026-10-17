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
