import type {
  DocumentationPayload,
  DocumentationResponse,
  OcrResponse,
  OcrSuccess,
  ScanDetailResponse,
  ScanListResponse,
} from './ocr-types';
import * as ImageManipulator from 'expo-image-manipulator';
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
  // Reduce upload size for remote APIs (Fly) to avoid request failures on large camera images.
  let uploadUri = imageUri;
  try {
    const optimized = await ImageManipulator.manipulateAsync(
      imageUri,
      [{ resize: { width: 1280 } }],
      { compress: 0.65, format: ImageManipulator.SaveFormat.JPEG }
    );
    uploadUri = optimized.uri;
  } catch {
    // Fall back to original URI if manipulation fails.
  }

  let response: Response;
  try {
    response = await fetch(uploadUri);
  } catch {
    throw new ApiError('تعذر تجهيز الصورة للرفع. جرّب إعادة التصوير أو اختيار صورة أصغر.');
  }
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

type OcrStartResponse = {
  ok?: boolean;
  async?: boolean;
  jobId?: string;
  error?: string;
  detail?: string;
  text?: string;
  status?: string;
  pending?: boolean;
} & Partial<OcrSuccess>;

type OcrPollResponse = OcrResponse & {
  status?: string;
  pending?: boolean;
  jobId?: string;
};

export async function uploadOcr(imageUri: string, signal?: AbortSignal): Promise<OcrResponse> {
  const payload = await imageUriToBase64Payload(imageUri);

  let startResponse: Response;
  try {
    startResponse = await fetch(`${getApiBaseUrl()}/ocr?async=1`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-abjadi-ocr-async': '1',
      },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (err) {
    if (signal?.aborted || (err instanceof Error && err.name === 'AbortError')) {
      throw err;
    }
    throw new ApiError('فشل رفع الصورة إلى الخادم. تحقق من الإنترنت أو جرّب صورة أصغر.');
  }

  let startData: OcrStartResponse | null = null;
  try {
    startData = (await startResponse.json()) as OcrStartResponse;
  } catch {
    throw new ApiError('تعذر قراءة رد الخادم', startResponse.status);
  }

  // Older/sync servers may still return full OCR immediately.
  if (startResponse.ok && startData.ok && !startData.async && startData.text) {
    return startData as OcrResponse;
  }

  if ((!startResponse.ok && startResponse.status !== 202) || !startData.jobId) {
    const detail = startData.detail || startData.error || `HTTP ${startResponse.status}`;
    throw new ApiError(detail || 'فشل بدء التحليل', startResponse.status);
  }

  const jobId = startData.jobId;
  const startedAt = Date.now();
  const maxWaitMs = 180_000;

  while (Date.now() - startedAt < maxWaitMs) {
    if (signal?.aborted) {
      throw new DOMException('Aborted', 'AbortError');
    }

    await new Promise((resolve) => setTimeout(resolve, 1500));

    let pollResponse: Response;
    try {
      pollResponse = await fetch(`${getApiBaseUrl()}/ocr/jobs/${encodeURIComponent(jobId)}`, {
        signal,
      });
    } catch (err) {
      if (signal?.aborted || (err instanceof Error && err.name === 'AbortError')) {
        throw err;
      }
      // Transient network blip — keep polling.
      continue;
    }

    let pollData: OcrPollResponse | null = null;
    try {
      pollData = (await pollResponse.json()) as OcrPollResponse;
    } catch {
      continue;
    }

    if (!pollData) continue;

    if (pollData.ok && pollData.status === 'succeeded') {
      return pollData;
    }

    if (
      pollData.ok &&
      (pollData.status === 'queued' || pollData.status === 'running' || pollData.pending)
    ) {
      continue;
    }

    if (!pollData.ok) {
      throw new ApiError(pollData.detail || pollData.error || 'فشل التحليل', pollResponse.status);
    }
  }

  throw new ApiError('انتهت مهلة التحليل على الخادم. أعد المحاولة.');
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
