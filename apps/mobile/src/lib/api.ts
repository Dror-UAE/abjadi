import type {
  DocumentationPayload,
  DocumentationResponse,
  OcrResponse,
  ScanDetailResponse,
  ScanListResponse,
} from './ocr-types';
import { getApiBaseUrl } from './api-config';

function guessMime(uri: string): string {
  const lower = uri.toLowerCase();
  if (lower.includes('.png')) return 'image/png';
  if (lower.includes('.webp')) return 'image/webp';
  return 'image/jpeg';
}

function guessName(uri: string): string {
  const lower = uri.toLowerCase();
  if (lower.includes('.png')) return 'scan.png';
  if (lower.includes('.webp')) return 'scan.webp';
  return 'scan.jpg';
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

export class ApiError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export { getApiBaseUrl, setApiBaseUrl, loadApiConfig } from './api-config';

async function imageUriToBase64Payload(imageUri: string): Promise<{
  imageBase64: string;
  filename: string;
  mimeType: string;
}> {
  const response = await fetch(imageUri);
  if (!response.ok) {
    throw new ApiError('تعذر قراءة الصورة من الجهاز');
  }

  const buffer = await response.arrayBuffer();
  if (buffer.byteLength === 0) {
    throw new ApiError('الصورة فارغة');
  }

  return {
    imageBase64: arrayBufferToBase64(buffer),
    filename: guessName(imageUri),
    mimeType: response.headers.get('content-type') || guessMime(imageUri),
  };
}

export async function uriToBase64Image(
  imageUri: string
): Promise<{ base64: string; filename: string; mimeType: string }> {
  const payload = await imageUriToBase64Payload(imageUri);
  return {
    base64: payload.imageBase64,
    filename: payload.filename,
    mimeType: payload.mimeType,
  };
}

export async function checkApiHealth(signal?: AbortSignal): Promise<boolean> {
  const response = await fetch(`${getApiBaseUrl()}/health`, { signal });
  if (!response.ok) return false;
  const data = (await response.json()) as { modelReady?: boolean };
  return Boolean(data.modelReady);
}

export async function uploadOcr(imageUri: string, signal?: AbortSignal): Promise<OcrResponse> {
  const payload = await imageUriToBase64Payload(imageUri);

  const response = await fetch(`${getApiBaseUrl()}/ocr`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });

  let data: OcrResponse | null = null;
  try {
    data = (await response.json()) as OcrResponse;
  } catch {
    throw new ApiError('تعذر قراءة رد الخادم', response.status);
  }

  if (!response.ok) {
    const detail =
      data && !data.ok ? data.detail || data.error : `HTTP ${response.status}`;
    throw new ApiError(detail || 'فشل التحليل', response.status);
  }

  return data;
}

export async function submitDocumentation(
  payload: DocumentationPayload,
  signal?: AbortSignal
): Promise<DocumentationResponse> {
  const response = await fetch(`${getApiBaseUrl()}/documentations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });

  let data: DocumentationResponse | null = null;
  try {
    data = (await response.json()) as DocumentationResponse;
  } catch {
    throw new ApiError('تعذر قراءة رد الخادم', response.status);
  }

  if (!response.ok || !data.ok) {
    const detail = !data.ok ? data.detail || data.error : `HTTP ${response.status}`;
    throw new ApiError(detail || 'فشل حفظ التوثيق', response.status);
  }

  return data;
}

export async function fetchScanHistory(limit = 50, signal?: AbortSignal): Promise<ScanListResponse> {
  const response = await fetch(`${getApiBaseUrl()}/scans?limit=${limit}`, { signal });

  let data: ScanListResponse | null = null;
  try {
    data = (await response.json()) as ScanListResponse;
  } catch {
    throw new ApiError('تعذر قراءة سجل الخادم', response.status);
  }

  if (!response.ok || !data.ok) {
    const detail = !data.ok ? data.detail || data.error : `HTTP ${response.status}`;
    throw new ApiError(detail || 'تعذر تحميل السجل', response.status);
  }

  return data;
}

export async function fetchScanById(
  scanId: string,
  signal?: AbortSignal
): Promise<ScanDetailResponse> {
  const response = await fetch(`${getApiBaseUrl()}/scans/${encodeURIComponent(scanId)}`, {
    signal,
  });

  let data: ScanDetailResponse | null = null;
  try {
    data = (await response.json()) as ScanDetailResponse;
  } catch {
    throw new ApiError('تعذر قراءة تفاصيل التحليل', response.status);
  }

  if (!response.ok || !data.ok) {
    const detail = !data.ok ? data.detail || data.error : `HTTP ${response.status}`;
    throw new ApiError(detail || 'تعذر تحميل التحليل', response.status);
  }

  return data;
}
