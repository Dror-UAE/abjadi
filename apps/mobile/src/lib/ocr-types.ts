export type OcrGlyph = {
  character: string;
  display?: string;
  name?: string;
  arabicName?: string;
  /** Arabic script equivalent (ا، ل، …) */
  arabicLetter?: string;
  confidence: number;
  trusted?: boolean;
  isSeparator?: boolean;
};

export type OcrLine = {
  text: string;
  glyphs: OcrGlyph[];
};

export type OcrSuccess = {
  ok: true;
  text: string;
  /** Transliteration of Musnad into Arabic script */
  arabicText?: string;
  arabicLines?: string[];
  nLines: number;
  nGlyphs: number;
  lines: OcrLine[];
  glyphs: OcrGlyph[];
  device: string;
  mode: 'paper';
  overlayBase64?: string;
  /** Supabase scan row id (when API persistence is enabled) */
  scanId?: string;
  publicId?: string;
  persisted?: boolean;
};

export type OcrFailure = {
  ok: false;
  error: string;
  detail?: string;
};

export type OcrResponse = OcrSuccess | OcrFailure;

export type ScanRecord = {
  id: string;
  imageUri: string;
  result: OcrSuccess;
  createdAt: number;
  /** Server scan UUID for documentation submit */
  serverScanId?: string;
  publicId?: string;
  /** Signed URLs when loaded from server history */
  sourceImageUrl?: string;
  overlayImageUrl?: string;
  /** User-facing title after documentation submit */
  documentationTitle?: string;
};

export type DocumentationPayload = {
  scanId: string;
  title: string;
  scriptType: string;
  language: string;
  region: string;
  country: string;
  description: string;
  condition: string;
  imageSource: string;
  ocrTextEdited: string;
  notes: string;
  confidence: number;
  extraImages?: Array<{ base64: string; filename?: string; mimeType?: string }>;
  extraDocuments?: Array<{ base64: string; filename?: string; mimeType?: string }>;
};

export type DocumentationResponse =
  | { ok: true; id: string; publicId: string; status: string }
  | { ok: false; error: string; detail?: string };

export type ScanSummary = {
  id: string;
  publicId: string;
  status: string;
  avgConfidence: number | null;
  createdAt: string;
  previewText: string;
  documentationTitle?: string;
  sourceImageUrl?: string;
  overlayImageUrl?: string;
};

export type ScanListResponse =
  | { ok: true; scans: ScanSummary[] }
  | { ok: false; error: string; detail?: string };

export type ScanDetailResponse =
  | {
      ok: true;
      scan: ScanSummary & {
        result: {
          ok: true;
          text: string;
          nLines: number;
          nGlyphs: number;
          lines: OcrLine[];
          glyphs: OcrGlyph[];
          device: string;
          mode: 'paper';
        };
      };
    }
  | { ok: false; error: string; detail?: string };
