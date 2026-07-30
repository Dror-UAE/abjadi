import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = 'abjadi.apiBaseUrl';
const DEFAULT_API_URL = 'http://localhost:3001';

let cachedUrl: string | null = null;

function normalizeApiUrl(raw: string): string {
  let url = raw.trim().replace(/\/$/, '');
  if (!url) return DEFAULT_API_URL;
  if (!/^https?:\/\//i.test(url)) {
    url = `http://${url}`;
  }
  return url;
}

function envDefault(): string {
  const fromEnv = process.env.EXPO_PUBLIC_API_URL?.trim();
  return normalizeApiUrl(fromEnv && fromEnv.length > 0 ? fromEnv : DEFAULT_API_URL);
}

/** Load persisted URL (call once on app start). */
export async function loadApiConfig(): Promise<string> {
  if (cachedUrl) return cachedUrl;
  try {
    const stored = await AsyncStorage.getItem(STORAGE_KEY);
    cachedUrl = stored ? normalizeApiUrl(stored) : envDefault();
  } catch {
    cachedUrl = envDefault();
  }
  return cachedUrl;
}

export function getApiBaseUrl(): string {
  return cachedUrl ?? envDefault();
}

export async function setApiBaseUrl(raw: string): Promise<string> {
  const normalized = normalizeApiUrl(raw);
  cachedUrl = normalized;
  await AsyncStorage.setItem(STORAGE_KEY, normalized);
  return normalized;
}

export async function clearApiBaseUrl(): Promise<string> {
  await AsyncStorage.removeItem(STORAGE_KEY);
  cachedUrl = envDefault();
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
    message.includes('timeout')
  );
}
