// App configuration. Values come from HooperAI/.env (EXPO_PUBLIC_* variables are
// inlined by the Expo CLI at build time); see .env.example. Nothing here is secret:
// Firebase web config is public by design, access is enforced by Firebase rules.

export const DEFAULT_SERVER_URL =
  process.env.EXPO_PUBLIC_SERVER_URL || 'http://192.168.1.100:8000';

export const FIREBASE_CONFIG = {
  apiKey:            process.env.EXPO_PUBLIC_FIREBASE_API_KEY,
  authDomain:        process.env.EXPO_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId:         process.env.EXPO_PUBLIC_FIREBASE_PROJECT_ID,
  messagingSenderId: process.env.EXPO_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId:             process.env.EXPO_PUBLIC_FIREBASE_APP_ID,
};

// Until the Firebase project exists the app runs without sign-in and talks to
// the laptop server with the API key from Settings (or no key on a trusted LAN).
export const isFirebaseConfigured = Boolean(
  FIREBASE_CONFIG.apiKey && FIREBASE_CONFIG.projectId && FIREBASE_CONFIG.appId
);
