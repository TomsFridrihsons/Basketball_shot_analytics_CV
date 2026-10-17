// Client for the laptop server (scripts/app/server.py). Every call reads the
// server URL from Settings and authenticates with a Firebase ID token when
// signed in, otherwise with the API key from Settings (if any).
// Upload with progress is part of the legacy API in SDK 54+ (still supported)
import * as FileSystem from 'expo-file-system/legacy';
import { getSettings } from './settings';
import { getIdToken } from './auth';

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

// Sends both when available: a Firebase-mode server uses the ID token, an
// API-key-mode server (no service account configured yet) uses the key.
async function authHeaders() {
  const headers = {};
  const token = await getIdToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const { apiKey } = await getSettings();
  if (apiKey) headers['X-API-Key'] = apiKey;
  return headers;
}

async function request(path, { method = 'GET', body, timeoutMs = 15000 } = {}) {
  const { serverUrl } = await getSettings();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(`${serverUrl}${path}`, {
      method,
      headers: { ...(body ? { 'Content-Type': 'application/json' } : {}), ...(await authHeaders()) },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
  } catch (e) {
    throw new ApiError(
      e.name === 'AbortError'
        ? `Server did not answer (${serverUrl}). Is the laptop on and the server running?`
        : `Cannot reach ${serverUrl}. Check Wi-Fi / Tailscale and the server address in Settings.`,
      0,
    );
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not JSON */ }
    if (res.status === 401) detail = `Not authorized: ${detail}. Check the API key or sign in again.`;
    throw new ApiError(detail, res.status);
  }
  return res.json();
}

export const health = () => request('/api/health', { timeoutMs: 5000 });
export const status = () => request('/api/status');
export const sessions = () => request('/api/sessions');
export const job = (jobId) => request(`/api/jobs/${jobId}`);
export const feedback = (jobId, shotNumber, label) =>
  request('/api/feedback', { method: 'POST', body: { job_id: jobId, shot_number: shotNumber, label } });

export async function analyze(video) {
  const { predictor } = await getSettings();
  return request('/api/analyze', { method: 'POST', body: predictor ? { video, config: predictor } : { video } });
}

/** expo-video source that streams an uploaded video back from the laptop. */
export async function videoSource(videoName) {
  const { serverUrl } = await getSettings();
  return { uri: `${serverUrl}/api/videos/${encodeURIComponent(videoName)}/stream`, headers: await authHeaders() };
}

/** Upload a local video file (original quality). onProgress(0..1). → {name, size_mb} */
export async function uploadVideo(uri, onProgress) {
  const { serverUrl } = await getSettings();
  const task = FileSystem.createUploadTask(
    `${serverUrl}/api/videos`,
    uri,
    {
      httpMethod: 'POST',
      uploadType: FileSystem.FileSystemUploadType.MULTIPART,
      fieldName: 'file',
      mimeType: 'video/mp4',
      headers: await authHeaders(),
    },
    ({ totalBytesSent, totalBytesExpectedToSend }) => {
      if (onProgress && totalBytesExpectedToSend > 0) onProgress(totalBytesSent / totalBytesExpectedToSend);
    },
  );
  let res;
  try {
    res = await task.uploadAsync();
  } catch (e) {
    throw new ApiError(`Upload failed: ${e.message}`, 0);
  }
  if (!res || res.status !== 200) {
    let detail = res?.body;
    try { detail = JSON.parse(res.body).detail; } catch (_) { /* keep raw body */ }
    throw new ApiError(`Upload failed (${res?.status}): ${detail}`, res?.status ?? 0);
  }
  return JSON.parse(res.body);
}

/** Poll a job until done. onProgress(fraction, statusText). Resolves with the result. */
export async function waitForJob(jobId, onProgress, isCancelled = () => false) {
  for (;;) {
    if (isCancelled()) throw new ApiError('cancelled', 0);
    const j = await job(jobId);
    if (j.status === 'done') return j.result;
    if (j.status === 'error') throw new ApiError(j.error || 'Analysis failed', 500);
    onProgress?.(j.progress || 0, j.status);
    await new Promise((r) => setTimeout(r, 1000));
  }
}
