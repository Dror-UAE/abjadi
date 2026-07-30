import type { OcrGlyph, OcrLine } from './ocr-types';

/** Musnad letter → Arabic script (from model labels.json transliteration). */
const MUSNAD_TO_ARABIC: Record<string, string> = {
  '𐩠': 'ه',
  '𐩡': 'ل',
  '𐩢': 'ح',
  '𐩣': 'م',
  '𐩤': 'ق',
  '𐩥': 'و',
  '𐩦': 'ش',
  '𐩧': 'ر',
  '𐩨': 'ب',
  '𐩩': 'ت',
  '𐩪': 'س',
  '𐩫': 'ك',
  '𐩬': 'ن',
  '𐩭': 'خ',
  '𐩮': 'ص',
  '𐩯': 'س',
  '𐩰': 'ف',
  '𐩱': 'ا',
  '𐩲': 'ع',
  '𐩳': 'ض',
  '𐩴': 'ج',
  '𐩵': 'د',
  '𐩶': 'غ',
  '𐩷': 'ط',
  '𐩸': 'ز',
  '𐩹': 'ذ',
  '𐩺': 'ي',
  '𐩻': 'ث',
  '𐩼': 'ظ',
};

const NUMERAL_TO_ARABIC: Record<string, string> = {
  NUM_1: '١',
  NUM_2: '٢',
  NUM_3: '٣',
  NUM_4: '٤',
  NUM_5: '٥',
  NUM_6: '٦',
  NUM_10: '١٠',
  NUM_50: '٥٠',
  NUM_100: '١٠٠',
  NUM_1000: '١٠٠٠',
};

export function musnadGlyphToArabic(glyph: OcrGlyph): string {
  if (glyph.isSeparator) {
    const display = glyph.display?.trim();
    if (display === '|') return ' ';
    if (display === '\n') return '\n';
    return display || ' ';
  }

  const character = glyph.character ?? '';
  if (character.startsWith('NUM_')) {
    return NUMERAL_TO_ARABIC[character] ?? glyph.display ?? character.replace('NUM_', '');
  }

  return MUSNAD_TO_ARABIC[character] ?? glyph.display ?? '';
}

export function translateMusnadGlyphs(glyphs: OcrGlyph[]): string {
  return glyphs.map(musnadGlyphToArabic).join('').replace(/[ \t]+/g, ' ').trim();
}

export function translateMusnadLines(lines: OcrLine[]): string {
  return lines
    .map((line) => translateMusnadGlyphs(line.glyphs))
    .filter(Boolean)
    .join('\n');
}

export function translateMusnadText(text: string): string {
  let out = '';
  for (const char of text) {
    if (char === '|') {
      out += ' ';
      continue;
    }
    if (char === '\n') {
      out += '\n';
      continue;
    }
    out += MUSNAD_TO_ARABIC[char] ?? char;
  }
  return out.replace(/[ \t]+/g, ' ').trim();
}

export function getArabicLetterForGlyph(glyph: OcrGlyph): string | undefined {
  const letter = musnadGlyphToArabic(glyph).trim();
  return letter || undefined;
}
