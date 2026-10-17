import AsyncStorage from '@react-native-async-storage/async-storage';
import { DEFAULT_SERVER_URL } from '../config';

const KEY = 'hooperai_settings';

const DEFAULTS = {
  serverUrl: DEFAULT_SERVER_URL,
  apiKey: '',
  predictor: null,        // config file name; null = server default (rim-pass rule)
  predictorLabel: null,
  predictorAccuracy: null,
};

export async function getSettings() {
  try {
    const raw = await AsyncStorage.getItem(KEY);
    return { ...DEFAULTS, ...(raw ? JSON.parse(raw) : {}) };
  } catch (e) {
    return { ...DEFAULTS };
  }
}

export async function saveSettings(patch) {
  const next = { ...(await getSettings()), ...patch };
  if (next.serverUrl) next.serverUrl = next.serverUrl.trim().replace(/\/+$/, '');
  await AsyncStorage.setItem(KEY, JSON.stringify(next));
  return next;
}
