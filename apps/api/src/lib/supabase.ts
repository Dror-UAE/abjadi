import { createClient, type SupabaseClient } from "@supabase/supabase-js";

let client: SupabaseClient | null | undefined;

function getSupabaseUrl(): string | undefined {
  return process.env.SUPABASE_URL?.trim();
}

/** Server key: SUPABASE_SECRET_KEY (new) or SUPABASE_SERVICE_ROLE_KEY (legacy JWT). */
function getSupabaseSecretKey(): string | undefined {
  return (
    process.env.SUPABASE_SECRET_KEY?.trim() ??
    process.env.SUPABASE_SERVICE_ROLE_KEY?.trim()
  );
}

export function isSupabaseConfigured(): boolean {
  return Boolean(getSupabaseUrl() && getSupabaseSecretKey());
}

/** Secret-key client (server only). Returns null if not configured. */
export function getSupabase(): SupabaseClient | null {
  if (client !== undefined) return client;

  const url = getSupabaseUrl();
  const key = getSupabaseSecretKey();
  if (!url || !key) {
    client = null;
    return client;
  }

  client = createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
  return client;
}

export function makePublicId(prefix = "ABJ"): string {
  const stamp = Date.now().toString().slice(-8);
  const rand = Math.random().toString(36).slice(2, 6).toUpperCase();
  return `${prefix}-${stamp}${rand}`;
}
