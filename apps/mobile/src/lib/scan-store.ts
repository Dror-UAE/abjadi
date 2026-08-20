import AsyncStorage from '@react-native-async-storage/async-storage';

import type { OcrGlyph, OcrLine, OcrSuccess, ScanRecord } from './ocr-types';

const STORAGE_KEY = 'abjadi.scans.v1';
const MAX_STORED_SCANS = 40;

type StoredScanRecord = Omit<ScanRecord, 'result'> & {
  result: Omit<OcrSuccess, 'overlayBase64'>;
};

const scans = new Map<string, ScanRecord>();
let loaded = false;
let loadPromise: Promise<void> | null = null;

function stripForStorage(record: ScanRecord): StoredScanRecord {
  const { overlayBase64: _overlay, ...result } = record.result;
  return { ...record, result };
}

function reviveRecord(stored: StoredScanRecord): ScanRecord {
  return {
    ...stored,
    result: {
      ...stored.result,
      ok: true,
    },
  };
}

async function persistToDisk(): Promise<void> {
  const records = listScansSync()
    .slice(0, MAX_STORED_SCANS)
    .map(stripForStorage);
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(records));
}

export async function loadScans(): Promise<void> {
  if (loaded) return;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    try {
      const raw = await AsyncStorage.getItem(STORAGE_KEY);
      if (raw) {
        const records = JSON.parse(raw) as StoredScanRecord[];
        scans.clear();
        for (const stored of records) {
          const record = reviveRecord(stored);
          scans.set(record.id, record);
        }
      }
    } catch (err) {
      console.warn('[scan-store] load failed', err);
    } finally {
      loaded = true;
    }
  })();

  return loadPromise;
}

function listScansSync(): ScanRecord[] {
  return Array.from(scans.values()).sort((a, b) => b.createdAt - a.createdAt);
}

export function listScans(): ScanRecord[] {
  return listScansSync();
}

export async function saveScan(record: ScanRecord): Promise<void> {
  await loadScans();
  scans.set(record.id, record);
  await persistToDisk();
}

export function getScan(id: string | undefined | null): ScanRecord | undefined {
  if (!id) return undefined;
  return scans.get(id);
}

export function getScanByServerId(serverId: string | undefined | null): ScanRecord | undefined {
  if (!serverId) return undefined;
  return listScansSync().find((s) => s.serverScanId === serverId || s.result.scanId === serverId);
}

export async function upsertScan(record: ScanRecord): Promise<void> {
  await saveScan(record);
}

export async function clearScan(id: string): Promise<void> {
  await loadScans();
  scans.delete(id);
  await persistToDisk();
}

/** Wipe all locally cached scans (AsyncStorage + memory). */
export async function clearAllScans(): Promise<void> {
  scans.clear();
  loaded = true;
  loadPromise = null;
  await AsyncStorage.removeItem(STORAGE_KEY);
}


export async function setScanDocumentationTitle(
  localScanId: string | undefined,
  title: string
): Promise<void> {
  if (!localScanId) return;
  await loadScans();
  const scan = scans.get(localScanId);
  if (!scan) return;

  scans.set(localScanId, {
    ...scan,
    documentationTitle: title.trim() || scan.documentationTitle,
  });
  await persistToDisk();
}

/** Pull documentation titles from server summaries onto matching local scans. */
export async function mergeRemoteDocumentationTitles(
  remoteScans: Array<{
    id: string;
    documentationTitle?: string;
    sourceImageUrl?: string;
    overlayImageUrl?: string;
  }>
): Promise<void> {
  await loadScans();
  let changed = false;

  for (const remote of remoteScans) {
    const title = remote.documentationTitle?.trim();
    const local = getScanByServerId(remote.id);
    if (!local) continue;

    const nextTitle = title || local.documentationTitle;
    const nextSource = remote.sourceImageUrl || local.sourceImageUrl;
    const nextOverlay = remote.overlayImageUrl || local.overlayImageUrl;

    // If cached imageUri is an old Supabase signed URL, refresh it from latest signed URL.
    const isSignedSupabaseUrl =
      local.imageUri.startsWith("https://") &&
      local.imageUri.includes(".supabase.co/storage/v1/object/sign/");
    const nextImageUri = isSignedSupabaseUrl
      ? nextOverlay || nextSource || local.imageUri
      : local.imageUri || nextOverlay || nextSource || "";

    const hasChanged =
      nextTitle !== local.documentationTitle ||
      nextSource !== local.sourceImageUrl ||
      nextOverlay !== local.overlayImageUrl ||
      nextImageUri !== local.imageUri;

    if (!hasChanged) continue;

    scans.set(local.id, {
      ...local,
      documentationTitle: nextTitle,
      sourceImageUrl: nextSource,
      overlayImageUrl: nextOverlay,
      imageUri: nextImageUri,
    });
    changed = true;
  }

  if (changed) await persistToDisk();
}

export function recordFromServerDetail(detail: {
  id: string;
  publicId: string;
  status?: string;
  documentationTitle?: string;
  sourceImageUrl?: string;
  overlayImageUrl?: string;
  result: {
    text: string;
    nLines: number;
    nGlyphs: number;
    lines: OcrLine[];
    glyphs: OcrGlyph[];
    device: string;
    mode: 'paper' | 'stone';
  };
}): ScanRecord {
  const existing = getScanByServerId(detail.id);
  const id = existing?.id ?? `server-${detail.id}`;
  const remoteTitle = detail.documentationTitle?.trim();
  const documentedFallback =
    !remoteTitle && detail.status === 'documented' ? 'موثّق' : undefined;

  return {
    id,
    imageUri: existing?.imageUri ?? detail.sourceImageUrl ?? '',
    sourceImageUrl: detail.sourceImageUrl,
    overlayImageUrl: detail.overlayImageUrl,
    serverScanId: detail.id,
    publicId: detail.publicId,
    documentationTitle:
      remoteTitle || documentedFallback || existing?.documentationTitle,
    createdAt: existing?.createdAt ?? Date.now(),
    result: {
      ok: true,
      text: detail.result.text,
      nLines: detail.result.nLines,
      nGlyphs: detail.result.nGlyphs,
      lines: detail.result.lines,
      glyphs: detail.result.glyphs,
      device: detail.result.device,
      mode: detail.result.mode,
      scanId: detail.id,
      publicId: detail.publicId,
      persisted: true,
    },
  };
}
