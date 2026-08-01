import { router } from 'expo-router';

import type { ScanRecord, ScanSummary } from './ocr-types';

const DEFAULT_SCAN_TITLE = 'نص محلَّل';
const TITLE_MAX_LEN = 56;

export function scanDisplayTitle(text: string | undefined | null): string {
  const trimmed = text?.trim();
  if (!trimmed) return DEFAULT_SCAN_TITLE;
  if (trimmed.length <= TITLE_MAX_LEN) return trimmed;
  return `${trimmed.slice(0, TITLE_MAX_LEN)}…`;
}

/** Prefer documentation form title; never show raw OCR / public ids as the card title. */
export function listItemTitle(opts: {
  documentationTitle?: string | null;
}): string {
  const docTitle = opts.documentationTitle?.trim();
  if (docTitle) return scanDisplayTitle(docTitle);
  return DEFAULT_SCAN_TITLE;
}

export type HistoryListItem = {
  key: string;
  localScanId?: string;
  serverScanId?: string;
  title: string;
  time: string;
  preview: string;
  publicId?: string;
  imageUri: string;
  status?: string;
  confidence?: number | null;
  sortTs: number;
};

function parseTime(value: string | number): number {
  if (typeof value === 'number') return value;
  const ts = Date.parse(value);
  return Number.isNaN(ts) ? 0 : ts;
}

export function formatHistoryTime(value: string | number): string {
  const date = new Date(parseTime(value));
  if (Number.isNaN(date.getTime())) return '—';

  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startOfThatDay = new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
  const diffDays = Math.round((startOfToday - startOfThatDay) / 86_400_000);

  const time = date.toLocaleTimeString('ar-SA', { hour: 'numeric', minute: '2-digit' });
  if (diffDays === 0) return `اليوم، ${time}`;
  if (diffDays === 1) return `أمس، ${time}`;
  return date.toLocaleDateString('ar-SA', { day: 'numeric', month: 'short' });
}

export function buildHistoryItems(
  localScans: ScanRecord[],
  remoteScans: ScanSummary[]
): HistoryListItem[] {
  const remoteById = new Map(remoteScans.map((s) => [s.id, s]));
  const linkedServerIds = new Set<string>();
  const items: HistoryListItem[] = [];

  for (const local of localScans) {
    if (local.serverScanId) linkedServerIds.add(local.serverScanId);

    const remote = local.serverScanId ? remoteById.get(local.serverScanId) : undefined;
    const documentationTitle =
      local.documentationTitle?.trim() || remote?.documentationTitle?.trim() || undefined;
    const status =
      documentationTitle || remote?.status === 'documented'
        ? 'documented'
        : local.serverScanId
          ? 'analyzed'
          : 'local';

    items.push({
      key: local.id,
      localScanId: local.id,
      serverScanId: local.serverScanId,
      title: listItemTitle({ documentationTitle }),
      time: formatHistoryTime(local.createdAt),
      preview: local.result.text?.trim().slice(0, 72) || remote?.previewText || '—',
      publicId: local.publicId ?? remote?.publicId,
      imageUri:
        local.imageUri ||
        local.overlayImageUrl ||
        local.sourceImageUrl ||
        remote?.overlayImageUrl ||
        remote?.sourceImageUrl ||
        '',
      status,
      confidence: remote?.avgConfidence,
      sortTs: local.createdAt,
    });
  }

  for (const remote of remoteScans) {
    if (linkedServerIds.has(remote.id)) continue;

    items.push({
      key: `server-${remote.id}`,
      serverScanId: remote.id,
      title: listItemTitle({ documentationTitle: remote.documentationTitle }),
      time: formatHistoryTime(remote.createdAt),
      preview: remote.previewText,
      publicId: remote.publicId,
      imageUri: remote.overlayImageUrl || remote.sourceImageUrl || '',
      status: remote.documentationTitle || remote.status === 'documented' ? 'documented' : remote.status,
      confidence: remote.avgConfidence,
      sortTs: parseTime(remote.createdAt),
    });
  }

  return items.sort((a, b) => b.sortTs - a.sortTs);
}

export function syncStatusLabel(item: Pick<HistoryListItem, 'serverScanId' | 'status'>): string {
  if (item.status === 'documented') return 'موثّق';
  if (item.serverScanId) return 'المحفوظات';
  return 'على الجهاز';
}

export function openHistoryItem(item: HistoryListItem): void {
  if (item.localScanId) {
    router.push({
      pathname: '/result',
      params: {
        scanId: item.localScanId,
        uri: item.imageUri ? encodeURIComponent(item.imageUri) : 'mock',
      },
    });
    return;
  }

  if (item.serverScanId) {
    router.push({
      pathname: '/result',
      params: {
        serverScanId: item.serverScanId,
        uri: item.imageUri ? encodeURIComponent(item.imageUri) : 'mock',
      },
    });
  }
}
