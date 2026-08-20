import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Device from 'expo-device';
import { Platform } from 'react-native';

const STORAGE_KEY = 'abjadi.apiBaseUrl';
const API_PORT = 3500;

/** Simulator / same-machine dev — LAN IPs often fail from iOS Simulator. */
const SIMULATOR_API_URL = `http://127.0.0.1:${API_PORT}`;
const DEFAULT_API_URL = `http://localhost:${API_PORT}`;

let cachedUrl: string | null = null;

function isSimulator(): boolean {
  return Platform.OS === 'ios' && !Device.isDevice;
}

function normalizeApiUrl(raw: string): string {
  let url = raw.trim().replace(/\/$/, '');
  if (!url) return preferredDefault();
  if (!/^https?:\/\//i.test(url)) {
    url = `http://${url}`;
  }
  return url;
}

function preferredDefault(): string {
  const fromEnv = process.env.EXPO_PUBLIC_API_URL?.trim();
  if (fromEnv && fromEnv.length > 0) {
    return normalizeApiUrl(fromEnv);
  }
  // No env set — local API on this Mac.
  if (isSimulator()) return SIMULATOR_API_URL;
  return DEFAULT_API_URL;
}

function hostFromUrl(url: string): string | null {
  try {
    return new URL(url).hostname;
  } catch {
    return null;
  }
}

/** On simulator, migrate stale LAN URLs saved from a physical-device session. */
function migrateStoredUrl(stored: string): string {
  const host = hostFromUrl(stored);
  if (!host) return preferredDefault();

  const fromEnv = process.env.EXPO_PUBLIC_API_URL?.trim();
  const envUrl = fromEnv && fromEnv.length > 0 ? normalizeApiUrl(fromEnv) : null;
  const envHost = envUrl ? hostFromUrl(envUrl) : null;

  // If .env points at a remote API (e.g. Fly) but AsyncStorage still has a
  // leftover local/simulator URL, prefer the env — that was the old bug.
  if (envUrl && envHost && envHost !== '127.0.0.1' && envHost !== 'localhost') {
    const storedIsLocal =
      host === '127.0.0.1' ||
      host === 'localhost' ||
      /^192\.168\./.test(host) ||
      /^10\./.test(host) ||
      /^172\.(1[6-9]|2\d|3[01])\./.test(host);
    if (storedIsLocal) return envUrl;
  }

  if (isSimulator()) {
    const isLan =
      /^192\.168\./.test(host) ||
      /^10\./.test(host) ||
      /^172\.(1[6-9]|2\d|3[01])\./.test(host);
    if (isLan || host === 'localhost') {
      return SIMULATOR_API_URL;
    }
  }

  return stored;
}

/** Load persisted URL (call once on app start). */
export async function loadApiConfig(): Promise<string> {
  if (cachedUrl) return cachedUrl;

  try {
    const stored = await AsyncStorage.getItem(STORAGE_KEY);
    if (stored) {
      const normalized = normalizeApiUrl(stored);
      const migrated = migrateStoredUrl(normalized);
      cachedUrl = migrated;
      if (migrated !== normalized) {
        await AsyncStorage.setItem(STORAGE_KEY, migrated);
      }
    } else {
      cachedUrl = preferredDefault();
    }
  } catch {
    cachedUrl = preferredDefault();
  }

  return cachedUrl;
}

export function getApiBaseUrl(): string {
  return cachedUrl ?? preferredDefault();
}

export async function setApiBaseUrl(raw: string): Promise<string> {
  const normalized = normalizeApiUrl(raw);
  cachedUrl = normalized;
  await AsyncStorage.setItem(STORAGE_KEY, normalized);
  return normalized;
}

export async function clearApiBaseUrl(): Promise<string> {
  await AsyncStorage.removeItem(STORAGE_KEY);
  cachedUrl = preferredDefault();
  return cachedUrl;
}

export function isLikelyNetworkError(message: string): boolean {
  return (
    message.includes('Network') ||
    message.includes('Failed to fetch') ||
    message.includes('fetch failed') ||
    message.includes('Host unreachable') ||
    message.includes('NoRouteToHost') ||
    message.includes('CLEARTEXT') ||
    message.includes('timeout') ||
    message.includes('Could not connect') ||
    message.includes('Connection refused')
  );
}

export function isDevSimulator(): boolean {
  return isSimulator();
}

export function apiUrlHint(): string {
  const fromEnv = process.env.EXPO_PUBLIC_API_URL?.trim();
  if (fromEnv) {
    return `من .env: ${fromEnv}`;
  }
  if (isSimulator()) {
    return 'المحاكي: استخدم http://127.0.0.1:3500 — شغّل bun dev في apps/api';
  }
  return 'Mac: ipconfig getifaddr en0 — نفس شبكة Wi‑Fi، بدون USB';
}
