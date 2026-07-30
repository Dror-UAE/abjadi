/**
 * App color tokens for light and dark mode.
 * Use via `useTheme()` — do not hardcode colors in screens.
 */

import { Platform } from 'react-native';

/** Brand palette */
export const Palette = {
  deepSandBrown: '#593A23',
  ochreClay: '#C18A3B',
  sandstone: '#D6B693',
  desertSage: '#6F8F84',
  limestone: '#F3EFE6',
  duneBeige: '#F7F4EC',
} as const;

export const Colors = {
  light: {
    text: Palette.deepSandBrown,
    background: Palette.duneBeige,
    backgroundElement: Palette.limestone,
    backgroundSelected: Palette.sandstone,
    textSecondary: Palette.desertSage,
    tint: Palette.ochreClay,
    border: Palette.sandstone,
    icon: Palette.desertSage,
  },
  dark: {
    text: Palette.duneBeige,
    background: Palette.deepSandBrown,
    backgroundElement: '#5C4634',
    backgroundSelected: '#6B5340',
    textSecondary: Palette.sandstone,
    tint: Palette.ochreClay,
    border: '#6B5340',
    icon: Palette.desertSage,
  },
} as const;

export type ColorScheme = keyof typeof Colors;
export type ThemeColor = keyof typeof Colors.light & keyof typeof Colors.dark;

export const Fonts = {
  ...Platform.select({
    ios: {
      /** iOS `UIFontDescriptorSystemDesignDefault` */
      sans: 'system-ui',
      /** iOS `UIFontDescriptorSystemDesignSerif` */
      serif: 'ui-serif',
      /** iOS `UIFontDescriptorSystemDesignRounded` */
      rounded: 'ui-rounded',
      /** iOS `UIFontDescriptorSystemDesignMonospaced` */
      mono: 'ui-monospace',
    },
    default: {
      sans: 'normal',
      serif: 'serif',
      rounded: 'normal',
      mono: 'monospace',
    },
    web: {
      sans: 'system-ui',
      serif: 'serif',
      rounded: 'system-ui',
      mono: 'monospace',
    },
  })!,
  /** Noto Naskh Arabic — use for all Arabic UI copy */
  arabic: {
    regular: 'NotoNaskhArabic_400Regular',
    medium: 'NotoNaskhArabic_500Medium',
    semiBold: 'NotoNaskhArabic_600SemiBold',
    bold: 'NotoNaskhArabic_700Bold',
  },
  /** Special script font for detected historic glyphs when available */
  script: 'NotoSansOldSouthArabian_400Regular',
} as const;

export const Spacing = {
  half: 2,
  one: 4,
  two: 8,
  three: 16,
  four: 24,
  five: 32,
  six: 64,
} as const;

export const BottomTabInset = Platform.select({ ios: 50, android: 88 }) ?? 0;
export const MaxContentWidth = 800;

/**
 * Chrome for capture / analyzing / result flows.
 * Keeps the archaeological AI look in both light and dark.
 */
export const FlowChrome = {
  light: {
    background: Palette.duneBeige,
    backgroundDeep: '#EFE8DC',
    gradient: [Palette.limestone, Palette.duneBeige, '#E8DFD0'] as const,
    text: Palette.deepSandBrown,
    textSecondary: 'rgba(89, 58, 35, 0.58)',
    textMuted: 'rgba(89, 58, 35, 0.4)',
    card: 'rgba(255, 252, 247, 0.92)',
    cardBorder: 'rgba(89, 58, 35, 0.12)',
    progressTrack: 'rgba(89, 58, 35, 0.1)',
    progressLabel: 'rgba(89, 58, 35, 0.5)',
    iconBtn: 'rgba(89, 58, 35, 0.07)',
    iconBtnBorder: 'rgba(89, 58, 35, 0.12)',
    icon: Palette.deepSandBrown,
    stepPending: 'rgba(89, 58, 35, 0.28)',
    stepDone: Palette.ochreClay,
    stepActive: Palette.ochreClay,
    stepLabelPending: 'rgba(89, 58, 35, 0.34)',
    stepLabelActive: Palette.deepSandBrown,
    stepLabelDone: 'rgba(89, 58, 35, 0.7)',
    glow: 'rgba(193, 138, 59, 0.22)',
    ambient: 'rgba(214, 182, 147, 0.45)',
    ringSoft: 'rgba(89, 58, 35, 0.1)',
    ringAccent: Palette.sandstone,
    cardInner: Palette.limestone,
    badgeBg: 'rgba(247, 244, 236, 0.92)',
    statusBar: 'dark' as const,
  },
  dark: {
    background: '#050403',
    backgroundDeep: '#020201',
    gradient: ['#050403', '#050403', '#050403'] as const,
    text: Palette.duneBeige,
    textSecondary: 'rgba(247, 244, 236, 0.62)',
    textMuted: 'rgba(247, 244, 236, 0.38)',
    card: '#050403',
    cardBorder: 'rgba(214, 182, 147, 0.18)',
    progressTrack: 'rgba(247, 244, 236, 0.08)',
    progressLabel: 'rgba(247, 244, 236, 0.55)',
    iconBtn: 'rgba(255, 255, 255, 0.06)',
    iconBtnBorder: 'rgba(247, 244, 236, 0.12)',
    icon: Palette.duneBeige,
    stepPending: 'rgba(247, 244, 236, 0.28)',
    stepDone: Palette.sandstone,
    stepActive: Palette.ochreClay,
    stepLabelPending: 'rgba(247, 244, 236, 0.34)',
    stepLabelActive: Palette.duneBeige,
    stepLabelDone: 'rgba(247, 244, 236, 0.72)',
    glow: 'rgba(193, 138, 59, 0.12)',
    ambient: 'rgba(193, 138, 59, 0.08)',
    ringSoft: 'rgba(247, 244, 236, 0.05)',
    ringAccent: Palette.sandstone,
    cardInner: '#050403',
    badgeBg: 'rgba(5, 4, 3, 0.85)',
    statusBar: 'light' as const,
  },
} as const;
