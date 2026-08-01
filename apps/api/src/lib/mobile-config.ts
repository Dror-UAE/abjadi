import { getSupabase, isSupabaseConfigured } from "./supabase.js";

export type MobileConfigResponse = {
  ios: {
    minimumSupportedVersion: string;
    latestVersion: string;
    storeUrl: string;
  };
  android: {
    minimumSupportedVersion: string;
    latestVersion: string;
    storeUrl: string;
  };
  updateMessage: {
    ar: string;
    en: string;
  };
};

type MobileVersionPolicyRow = {
  ios_minimum_supported_version: string;
  ios_latest_version: string;
  ios_store_url: string | null;
  android_minimum_supported_version: string;
  android_latest_version: string;
  android_store_url: string | null;
  update_message_ar: string;
  update_message_en: string;
};

function mapRow(row: MobileVersionPolicyRow): MobileConfigResponse {
  return {
    ios: {
      minimumSupportedVersion: row.ios_minimum_supported_version,
      latestVersion: row.ios_latest_version,
      storeUrl: row.ios_store_url?.trim() ?? "",
    },
    android: {
      minimumSupportedVersion: row.android_minimum_supported_version,
      latestVersion: row.android_latest_version,
      storeUrl: row.android_store_url?.trim() ?? "",
    },
    updateMessage: {
      ar: row.update_message_ar,
      en: row.update_message_en,
    },
  };
}

/**
 * Active store-binary compatibility policy from Postgres.
 * Returns null when Supabase is unset, the query fails, or no active row exists
 * so `/mobile/config` can respond 503 and the mobile client can fail open.
 */
export async function getMobileConfig(): Promise<MobileConfigResponse | null> {
  if (!isSupabaseConfigured()) {
    console.error("[mobile-config] Supabase is not configured");
    return null;
  }

  const supabase = getSupabase();
  if (!supabase) {
    console.error("[mobile-config] Supabase client unavailable");
    return null;
  }

  const { data, error } = await supabase
    .from("mobile_version_policies")
    .select(
      "ios_minimum_supported_version, ios_latest_version, ios_store_url, android_minimum_supported_version, android_latest_version, android_store_url, update_message_ar, update_message_en"
    )
    .eq("is_active", true)
    .maybeSingle<MobileVersionPolicyRow>();

  if (error) {
    console.error("[mobile-config] failed to load policy:", error.message);
    return null;
  }

  if (!data) {
    console.error("[mobile-config] no active mobile_version_policies row");
    return null;
  }

  return mapRow(data);
}
