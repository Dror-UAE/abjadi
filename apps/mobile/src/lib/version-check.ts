import Constants from 'expo-constants';
import { Platform } from 'react-native';

import { getApiBaseUrl } from './api-config';

const VERSION_CHECK_TIMEOUT_MS = 5_000;

export type VersionCheckStatus =
  | 'supported'
  | 'optional-update'
  | 'force-update'
  | 'unknown';

export type MobilePlatformPolicy = {
  minimumSupportedVersion: string;
  latestVersion: string;
  storeUrl: string;
};

export type MobileConfigResponse = {
  ios: MobilePlatformPolicy;
  android: MobilePlatformPolicy;
  updateMessage: {
    ar: string;
    en: string;
  };
};

export type VersionCheckResult = {
  status: VersionCheckStatus;
  currentVersion: string;
  minimumSupportedVersion: string | null;
  latestVersion: string | null;
  platform: 'ios' | 'android' | 'web' | 'unknown';
  storeUrl: string | null;
  updateMessage: {
    ar: string;
    en: string;
  } | null;
};

const FALLBACK_MESSAGE = {
  ar: 'يتوفر إصدار جديد من أبجدي. يرجى تحديث التطبيق للمتابعة.',
  en: 'A new version of Abjadi is available. Please update to continue.',
} as const;

/**
 * Marketing / CFBundleShortVersionString / versionName — not buildNumber/versionCode.
 * In Expo Go, nativeApplicationVersion is Expo Go itself, so use app config there.
 */
export function getInstalledAppVersion(): string {
  if (Constants.appOwnership === 'expo') {
    return Constants.expoConfig?.version?.trim() || '0.0.0';
  }

  return (
    Constants.nativeApplicationVersion?.trim() ||
    Constants.expoConfig?.version?.trim() ||
    '0.0.0'
  );
}

export function getAppPlatform(): VersionCheckResult['platform'] {
  if (Platform.OS === 'ios') return 'ios';
  if (Platform.OS === 'android') return 'android';
  if (Platform.OS === 'web') return 'web';
  return 'unknown';
}

/** Parse "1.2.3" / "1.2.3-beta" → [1,2,3]. Non-numeric segments become 0. */
export function parseSemver(version: string): [number, number, number] {
  const core = version.trim().split(/[+-]/)[0] ?? '';
  const parts = core.split('.').slice(0, 3);
  const nums: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < 3; i += 1) {
    const n = Number.parseInt(parts[i] ?? '0', 10);
    nums[i] = Number.isFinite(n) && n >= 0 ? n : 0;
  }
  return nums;
}

/** Negative if a < b, 0 if equal, positive if a > b. */
export function compareSemver(a: string, b: string): number {
  const left = parseSemver(a);
  const right = parseSemver(b);
  for (let i = 0; i < 3; i += 1) {
    if (left[i] !== right[i]) return left[i] - right[i];
  }
  return 0;
}

function unknownResult(currentVersion: string): VersionCheckResult {
  return {
    status: 'unknown',
    currentVersion,
    minimumSupportedVersion: null,
    latestVersion: null,
    platform: getAppPlatform(),
    storeUrl: null,
    updateMessage: null,
  };
}

function evaluatePolicy(
  currentVersion: string,
  policy: MobilePlatformPolicy,
  updateMessage: MobileConfigResponse['updateMessage']
): VersionCheckResult {
  const platform = getAppPlatform();
  const min = policy.minimumSupportedVersion.trim();
  const latest = policy.latestVersion.trim();
  const storeUrl = policy.storeUrl.trim() || null;

  if (!min) {
    return unknownResult(currentVersion);
  }

  let status: VersionCheckStatus = 'supported';
  if (compareSemver(currentVersion, min) < 0) {
    status = 'force-update';
  } else if (latest && compareSemver(currentVersion, latest) < 0) {
    status = 'optional-update';
  }

  return {
    status,
    currentVersion,
    minimumSupportedVersion: min,
    latestVersion: latest || null,
    platform,
    storeUrl,
    updateMessage: {
      ar: updateMessage.ar?.trim() || FALLBACK_MESSAGE.ar,
      en: updateMessage.en?.trim() || FALLBACK_MESSAGE.en,
    },
  };
}

/**
 * Fetch remote binary compatibility policy and classify this install.
 * Fail open: network/parse errors → `unknown` (allow app).
 * Successful unsupported response → `force-update`.
 */
export async function checkAppVersion(signal?: AbortSignal): Promise<VersionCheckResult> {
  const currentVersion = getInstalledAppVersion();
  const platform = getAppPlatform();

  if (platform !== 'ios' && platform !== 'android') {
    return {
      ...unknownResult(currentVersion),
      status: 'supported',
      platform,
    };
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), VERSION_CHECK_TIMEOUT_MS);

  const onAbort = () => controller.abort();
  signal?.addEventListener('abort', onAbort);

  try {
    const response = await fetch(`${getApiBaseUrl()}/mobile/config`, {
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    });

    if (!response.ok) {
      return unknownResult(currentVersion);
    }

    const data = (await response.json()) as MobileConfigResponse;
    const policy = platform === 'ios' ? data.ios : data.android;

    if (!policy?.minimumSupportedVersion) {
      return unknownResult(currentVersion);
    }

    return evaluatePolicy(currentVersion, policy, data.updateMessage ?? FALLBACK_MESSAGE);
  } catch {
    return unknownResult(currentVersion);
  } finally {
    clearTimeout(timeoutId);
    signal?.removeEventListener('abort', onAbort);
  }
}
