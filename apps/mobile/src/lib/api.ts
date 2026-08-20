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

async function prepareUploadImage(imageUri: string): Promise<{
  uri: string;
  filename: string;
  mimeType: string;
  imageBase64?: string;
}> {
  // Resize for remote APIs. Prefer keeping a local file URI for multipart upload
  // (smaller + more reliable than JSON base64 on mobile networks).
  try {
    const optimized = await ImageManipulator.manipulateAsync(
      imageUri,
      [{ resize: { width: 1024 } }],
      {
        compress: 0.55,
        format: ImageManipulator.SaveFormat.JPEG,
        base64: true,
      }
    );
    return {
      uri: optimized.uri,
      filename: 'scan.jpg',
      mimeType: 'image/jpeg',
      imageBase64: optimized.base64,
    };
  } catch {
    return {
      uri: imageUri,
      filename: guessName(imageUri),
      mimeType: guessMime(imageUri),
    };
  }
}

async function imageUriToBase64Payload(imageUri: string): Promise<{
  imageBase64: string;
  filename: string;
  mimeType: string;
}> {
  const prepared = await prepareUploadImage(imageUri);
  if (prepared.imageBase64) {
    return {
      imageBase64: prepared.imageBase64,
      filename: prepared.filename,
      mimeType: prepared.mimeType,
    };
  }

  let response: Response;
  try {
    response = await fetch(prepared.uri);
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
    filename: prepared.filename,
    mimeType: response.headers.get('content-type') || prepared.mimeType,
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

/** Read any local file URI as base64 (PDFs / docs — no image compression). */
export async function uriToBase64File(
  fileUri: string,
  opts?: { filename?: string; mimeType?: string }
): Promise<{ base64: string; filename: string; mimeType: string }> {
  let response: Response;
  try {
    response = await fetch(fileUri);
  } catch {
    throw new ApiError('تعذر قراءة الملف من الجهاز');
  }
  if (!response.ok) {
    throw new ApiError('تعذر قراءة الملف من الجهاز');
  }

  const buffer = await response.arrayBuffer();
  if (buffer.byteLength === 0) {
    throw new ApiError('الملف فارغ');
  }
  if (buffer.byteLength > 12 * 1024 * 1024) {
    throw new ApiError('حجم الملف أكبر من المسموح (12 ميجابايت)');
  }

  const headerType = response.headers.get('content-type')?.split(';')[0]?.trim();
  const mimeType = opts?.mimeType?.trim() || headerType || 'application/octet-stream';
  const fromUri = fileUri.split('/').pop()?.split('?')[0];
  const filename = opts?.filename?.trim() || fromUri || 'document.bin';

  return {
    base64: arrayBufferToBase64(buffer),
    filename,
    mimeType,
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

type OcrPollResponse = {
  ok?: boolean;
  status?: string;
  pending?: boolean;
  jobId?: string;
  hasOverlay?: boolean;
  overlayBase64?: string;
  error?: string;
  detail?: string;
  text?: string;
  arabicText?: string;
  arabicLines?: string[];
  nLines?: number;
  nGlyphs?: number;
  lines?: OcrSuccess['lines'];
  glyphs?: OcrSuccess['glyphs'];
  device?: string;
  mode?: OcrSuccess['mode'];
  scanId?: string;
  publicId?: string;
  persisted?: boolean;
};

function toOcrSuccess(poll: OcrPollResponse, overlayBase64?: string): OcrSuccess {
  return {
    ok: true,
    text: poll.text ?? '',
    arabicText: poll.arabicText,
    arabicLines: poll.arabicLines,
    nLines: poll.nLines ?? poll.lines?.length ?? 0,
    nGlyphs: poll.nGlyphs ?? poll.glyphs?.length ?? 0,
    lines: poll.lines ?? [],
    glyphs: poll.glyphs ?? [],
    device: poll.device ?? 'cpu',
    mode: poll.mode ?? 'stone',
    overlayBase64: overlayBase64 ?? poll.overlayBase64,
    scanId: poll.scanId,
    publicId: poll.publicId,
    persisted: poll.persisted,
  };
}

export async function uploadOcr(
  imageUri: string,
  signal?: AbortSignal,
  mode: 'paper' | 'stone' = 'stone'
): Promise<OcrResponse> {
  const prepared = await prepareUploadImage(imageUri);

  // Multipart file upload — avoids huge JSON base64 bodies that get reset on Fly.
  const form = new FormData();
  form.append('mode', mode);
  form.append('image', {
    uri: prepared.uri,
    name: prepared.filename,
    type: prepared.mimeType,
  } as unknown as Blob);

  let startResponse: Response;
  try {
    startResponse = await fetch(`${getApiBaseUrl()}/ocr?async=1&mode=${mode}`, {
      method: 'POST',
      headers: {
        'x-abjadi-ocr-async': '1',
        'x-abjadi-ocr-mode': mode,
      },
      body: form,
      signal,
    });
  } catch (err) {
    if (signal?.aborted || (err instanceof Error && err.name === 'AbortError')) {
      throw err;
    }
    // Fallback: JSON base64 if multipart fails on this platform.
    try {
      const payload = await imageUriToBase64Payload(imageUri);
      startResponse = await fetch(`${getApiBaseUrl()}/ocr?async=1&mode=${mode}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-abjadi-ocr-async': '1',
          'x-abjadi-ocr-mode': mode,
        },
        body: JSON.stringify({ ...payload, mode }),
        signal,
      });
    } catch (fallbackErr) {
      if (
        signal?.aborted ||
        (fallbackErr instanceof Error && fallbackErr.name === 'AbortError')
      ) {
        throw fallbackErr;
      }
      throw new ApiError('فشل رفع الصورة إلى الخادم. تحقق من الإنترنت أو جرّب صورة أصغر.');
    }
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

    if (pollData.status === 'succeeded' && (pollData.ok === true || typeof pollData.text === 'string')) {
      let success = toOcrSuccess(pollData);

      // Overlay is served separately so poll stays small/fast.
      if (pollData.hasOverlay && !success.overlayBase64) {
        try {
          const overlayRes = await fetch(
            `${getApiBaseUrl()}/ocr/jobs/${encodeURIComponent(jobId)}/overlay`,
            { signal }
          );
          if (overlayRes.ok) {
            const overlayJson = (await overlayRes.json()) as {
              ok?: boolean;
              overlayBase64?: string;
            };
            if (overlayJson.overlayBase64) {
              success = { ...success, overlayBase64: overlayJson.overlayBase64 };
            }
          }
        } catch {
          // Text-only result is still usable without overlay.
        }
      }

      // scanId is attached after background Supabase persist — wait briefly so
      // local history can link to the server row (avoids duplicate "على الجهاز" + "المحفوظات").
      if (!success.scanId) {
        for (let i = 0; i < 12; i++) {
          if (signal?.aborted) break;
          await new Promise((resolve) => setTimeout(resolve, 1000));
          try {
            const againRes = await fetch(
              `${getApiBaseUrl()}/ocr/jobs/${encodeURIComponent(jobId)}`,
              { signal }
            );
            if (!againRes.ok) continue;
            const again = (await againRes.json()) as OcrPollResponse;
            if (again.scanId || again.publicId) {
              success = {
                ...success,
                scanId: again.scanId ?? success.scanId,
                publicId: again.publicId ?? success.publicId,
                persisted: again.persisted ?? success.persisted,
              };
              break;
            }
          } catch {
            // keep waiting
          }
        }
      }

      return success;
    }

    if (
      pollData.ok &&
      (pollData.status === 'queued' || pollData.status === 'running' || pollData.pending)
    ) {
      continue;
    }

    if (pollData.ok === false) {
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

export async function deleteScan(
  scanId: string,
  signal?: AbortSignal
): Promise<{ ok: true; id: string }> {
  const response = await fetch(`${getApiBaseUrl()}/scans/${encodeURIComponent(scanId)}`, {
    method: 'DELETE',
    signal,
  });

  let data: { ok?: boolean; id?: string; error?: string; detail?: string } | null = null;
  try {
    data = (await response.json()) as {
      ok?: boolean;
      id?: string;
      error?: string;
      detail?: string;
    };
  } catch {
    throw new ApiError('تعذر قراءة رد الحذف', response.status);
  }

  if (!response.ok || !data?.ok || !data.id) {
    // Already gone on server is fine — local delete can still proceed.
    if (response.status === 404) {
      return { ok: true, id: scanId };
    }
    throw new ApiError(data?.detail || data?.error || 'فشل حذف التحليل', response.status);
  }

  return { ok: true, id: data.id };
}
