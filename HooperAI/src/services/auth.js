// Firebase Authentication (email + password). Inactive until the Firebase web
// config is set in .env – see isFirebaseConfigured. Uses only the free tier.
import AsyncStorage from '@react-native-async-storage/async-storage';
import { initializeApp, getApps } from 'firebase/app';
import {
  initializeAuth, getAuth, getReactNativePersistence, onAuthStateChanged,
  signInWithEmailAndPassword, createUserWithEmailAndPassword,
  sendPasswordResetEmail, signOut as fbSignOut, deleteUser,
  EmailAuthProvider, reauthenticateWithCredential,
} from 'firebase/auth';
import { FIREBASE_CONFIG, isFirebaseConfigured } from '../config';

let auth = null;

function getAuthInstance() {
  if (!isFirebaseConfigured) return null;
  if (!auth) {
    const app = getApps()[0] || initializeApp(FIREBASE_CONFIG);
    try {
      // Keeps the user signed in between app launches
      auth = initializeAuth(app, { persistence: getReactNativePersistence(AsyncStorage) });
    } catch (e) {
      // A hot reload re-runs this module while Firebase keeps its auth instance
      if (e?.code !== 'auth/already-initialized') throw e;
      auth = getAuth(app);
    }
  }
  return auth;
}

export function onUserChanged(callback) {
  const a = getAuthInstance();
  if (!a) {
    callback(null);
    return () => {};
  }
  return onAuthStateChanged(a, callback);
}

export function currentUser() {
  return getAuthInstance()?.currentUser ?? null;
}

/** Firebase ID token for the laptop server, or null when Firebase is off. */
export async function getIdToken() {
  const user = currentUser();
  return user ? user.getIdToken() : null;   // refreshed automatically when expired
}

export const signIn = (email, password) =>
  signInWithEmailAndPassword(getAuthInstance(), email.trim(), password);

export const signUp = (email, password) =>
  createUserWithEmailAndPassword(getAuthInstance(), email.trim(), password);

export const resetPassword = (email) => sendPasswordResetEmail(getAuthInstance(), email.trim());

export const signOut = () => (getAuthInstance() ? fbSignOut(getAuthInstance()) : Promise.resolve());

/** Delete the signed-in account. Firebase only deletes after a recent sign-in, so
 *  confirm the password first instead of making people sign out and back in. */
export async function deleteAccount(password) {
  const user = currentUser();
  await reauthenticateWithCredential(user, EmailAuthProvider.credential(user.email, password));
  await deleteUser(user);
}

/** Human-readable message for Firebase auth error codes. */
export function authErrorMessage(e) {
  const code = e?.code || '';
  const map = {
    'auth/invalid-email': 'That email address is not valid.',
    'auth/invalid-credential': 'Wrong email or password.',
    'auth/user-not-found': 'No account with that email.',
    'auth/wrong-password': 'Wrong email or password.',
    'auth/email-already-in-use': 'An account with that email already exists.',
    'auth/weak-password': 'Use at least 6 characters.',
    'auth/network-request-failed': 'No internet connection.',
    'auth/too-many-requests': 'Too many attempts – try again later.',
    'auth/requires-recent-login': 'For your security, sign out and sign in again, then delete the account.',
    'auth/missing-password': 'Enter your password.',
  };
  return map[code] || e?.message || 'Something went wrong.';
}
