import { LinearGradient } from 'expo-linear-gradient';
import { router, useLocalSearchParams } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SymbolView } from 'expo-symbols';
import { useEffect, useMemo, useState } from 'react';
import {
  Image,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  type ImageSourcePropType,
} from 'react-native';
import Animated, { FadeIn, FadeInDown, FadeInUp } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { ArabicText } from '@/components/arabic-text';
import { Fonts, Palette, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { fetchScanById } from '@/lib/api';
import type { OcrGlyph } from '@/lib/ocr-types';
import type { ScanRecord } from '@/lib/ocr-types';
import {
  getScan,
  getScanByServerId,
  recordFromServerDetail,
  upsertScan,
} from '@/lib/scan-store';
import { translateMusnadLines, getArabicLetterForGlyph } from '@/lib/musnad-translate';

const FALLBACK_IMAGE = require('../../assets/images/scanned-image.jpg');

/** Mock detected characters when opened without a live scan (e.g. History). */
const MOCK_CHARS = [
  { glyph: '𐩱', nameAr: 'ألف', conf: 97 },
  { glyph: '𐩡', nameAr: 'لام', conf: 96 },
  { glyph: '𐩭', nameAr: 'خاء', conf: 94 },
  { glyph: '𐩸', nameAr: 'زاي', conf: 95 },
  { glyph: '𐩣', nameAr: 'ميم', conf: 98 },
  { glyph: '𐩫', nameAr: 'كاف', conf: 96 },
  { glyph: '𐩡', nameAr: 'لام', conf: 97 },
] as const;

const MOCK_LINE = '𐩱𐩡 𐩭𐩸𐩣 𐩫𐩡';

function resolveImageSource(uri?: string | string[]): ImageSourcePropType {
  const value = Array.isArray(uri) ? uri[0] : uri;
  if (value && value !== 'mock') {
    try {
      return { uri: decodeURIComponent(value) };
    } catch {
      return { uri: value };
    }
  }
  return FALLBACK_IMAGE;
}

function displayGlyph(g: OcrGlyph): string {
  if (g.isSeparator) return g.display || '|';
  if (g.character?.startsWith('NUM_')) return g.display || g.character.replace('NUM_', '');
  return g.character || '?';
}

function glyphNameAr(g: OcrGlyph): string {
  if (g.isSeparator) return 'فاصل';
  return g.arabicName || g.name || '—';
}

export function ResultScreen() {
  const insets = useSafeAreaInsets();
  const { flow } = useTheme();
  const { uri, scanId, serverScanId } = useLocalSearchParams<{
    uri?: string;
    scanId?: string;
    serverScanId?: string;
  }>();
  const [scan, setScan] = useState<ScanRecord | undefined>(() => getScan(scanId));
  const [hydrating, setHydrating] = useState(false);
  const [showOriginal, setShowOriginal] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function hydrate() {
      const local = scanId ? getScan(scanId) : undefined;
      if (local) {
        setScan(local);
        return;
      }

      const remoteId = serverScanId?.trim();
      if (!remoteId) return;

      const cached = getScanByServerId(remoteId);
      if (cached) {
        setScan(cached);
        return;
      }

      setHydrating(true);
      try {
        const response = await fetchScanById(remoteId);
        if (cancelled || !response.ok) return;
        const record = recordFromServerDetail(response.scan);
        await upsertScan(record);
        if (!cancelled) setScan(record);
      } catch {
        // Fall back to mock preview when server fetch fails.
      } finally {
        if (!cancelled) setHydrating(false);
      }
    }

    void hydrate();
    return () => {
      cancelled = true;
    };
  }, [scanId, serverScanId]);

  const originalSource = useMemo((): ImageSourcePropType => {
    if (uri && uri !== 'mock') return resolveImageSource(uri);
    if (scan?.imageUri) return { uri: scan.imageUri };
    if (scan?.sourceImageUrl) return { uri: scan.sourceImageUrl };
    return FALLBACK_IMAGE;
  }, [uri, scan]);

  const overlaySource = useMemo((): ImageSourcePropType | null => {
    const b64 = scan?.result.overlayBase64;
    if (b64) return { uri: `data:image/png;base64,${b64}` };
    if (scan?.overlayImageUrl) return { uri: scan.overlayImageUrl };
    return null;
  }, [scan]);

  const previewSource =
    !showOriginal && overlaySource ? overlaySource : originalSource;
  const hasOverlay = Boolean(overlaySource);

  const detectedLine = scan?.result.text?.trim() || MOCK_LINE;
  const arabicTranslation = useMemo(() => {
    if (!scan) return 'ملك مملكة سبأ';
    if (scan.result.arabicText?.trim()) return scan.result.arabicText.trim();
    if (scan.result.lines?.length) return translateMusnadLines(scan.result.lines);
    return '';
  }, [scan]);

  const detectedChars = useMemo(() => {
    if (!scan) {
      return MOCK_CHARS.map((c) => ({
        glyph: c.glyph,
        nameAr: c.nameAr,
        arabicLetter: c.nameAr,
        conf: c.conf,
      }));
    }

    return scan.result.glyphs
      .filter((g) => !g.isSeparator && g.character && g.character !== '?')
      .slice(0, 24)
      .map((g) => ({
        glyph: displayGlyph(g),
        nameAr: glyphNameAr(g),
        arabicLetter: g.arabicLetter || getArabicLetterForGlyph(g) || '—',
        conf: Math.round(Math.max(0, Math.min(1, g.confidence)) * 100),
      }));
  }, [scan]);

  const avgConf = useMemo(() => {
    if (!detectedChars.length) return 0;
    const sum = detectedChars.reduce((acc, c) => acc + c.conf, 0);
    return Math.round(sum / detectedChars.length);
  }, [detectedChars]);

  const goHome = () => {
    router.replace('/home');
  };

  const close = () => {
    if (router.canGoBack()) {
      router.back();
    } else {
      goHome();
    }
  };

  return (
    <View style={[styles.root, { backgroundColor: flow.background }]}>
      <StatusBar style={flow.statusBar} />
      <LinearGradient colors={[...flow.gradient]} style={StyleSheet.absoluteFill} />

      <View style={[styles.topBar, { paddingTop: insets.top + Spacing.two }]}>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Close"
          onPress={close}
          style={({ pressed }) => [
            styles.iconBtn,
            {
              backgroundColor: flow.iconBtn,
              borderColor: flow.iconBtnBorder,
            },
            pressed && styles.pressed,
          ]}
        >
          <SymbolView
            name={{ ios: 'xmark', android: 'close', web: 'close' }}
            size={14}
            tintColor={flow.icon}
          />
        </Pressable>

        <ArabicText weight="semiBold" style={[styles.topTitle, { color: flow.text }]}>
          نتيجة التحليل
        </ArabicText>

        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Done"
          onPress={goHome}
          style={({ pressed }) => [
            styles.iconBtn,
            {
              backgroundColor: flow.iconBtn,
              borderColor: flow.iconBtnBorder,
            },
            pressed && styles.pressed,
          ]}
        >
          <SymbolView
            name={{ ios: 'checkmark', android: 'check', web: 'check' }}
            size={14}
            tintColor={Palette.sandstone}
          />
        </Pressable>
      </View>

      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={[
          styles.scroll,
          { paddingBottom: Math.max(insets.bottom, Spacing.three) + Spacing.four },
        ]}
      >
        <Animated.View entering={FadeIn.duration(450)} style={styles.previewCard}>
          <LinearGradient
            colors={[
              'rgba(214, 182, 147, 0.55)',
              'rgba(193, 138, 59, 0.25)',
              'rgba(89, 58, 35, 0.2)',
            ]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.previewBorder}
          >
            <Image source={previewSource} style={styles.previewImage} resizeMode="contain" />
          </LinearGradient>
          {hasOverlay ? (
            <Pressable
              accessibilityRole="button"
              accessibilityLabel={showOriginal ? 'Show overlay' : 'Show original'}
              onPress={() => setShowOriginal((v) => !v)}
              style={({ pressed }) => [
                styles.toggleBtn,
                {
                  backgroundColor: flow.badgeBg,
                  borderColor: flow.cardBorder,
                },
                pressed && styles.pressed,
              ]}
            >
              <SymbolView
                name={{
                  ios: showOriginal ? 'square.on.square' : 'viewfinder',
                  android: showOriginal ? 'layers' : 'center_focus_strong',
                  web: showOriginal ? 'layers' : 'center_focus_strong',
                }}
                size={10}
                tintColor={Palette.ochreClay}
              />
              <ArabicText weight="medium" style={styles.badgeText}>
                {showOriginal ? 'الصورة الأصلية' : 'تحليل الحروف'}
              </ArabicText>
            </Pressable>
          ) : null}
          <View
            style={[
              styles.badge,
              {
                backgroundColor: flow.badgeBg,
                borderColor: flow.cardBorder,
              },
            ]}
          >
            <SymbolView
              name={{ ios: 'sparkles', android: 'auto_awesome', web: 'auto_awesome' }}
              size={10}
              tintColor={Palette.ochreClay}
            />
            <ArabicText weight="medium" style={styles.badgeText}>
              {scan ? `${scan.result.nGlyphs} حرف` : 'تم التعرف'}
            </ArabicText>
          </View>
        </Animated.View>

        <Animated.View
          entering={FadeInDown.delay(120).duration(480)}
          style={styles.section}
        >
          <ArabicText weight="bold" style={[styles.sectionTitle, { color: flow.text }]}>
            النص المكتشف
          </ArabicText>

          <View
            style={[
              styles.scriptBlock,
              {
                backgroundColor: flow.card,
                borderColor: flow.cardBorder,
              },
            ]}
          >
            <Text style={styles.scriptLine}>{detectedLine}</Text>
          </View>
        </Animated.View>

        <Animated.View
          entering={FadeInDown.delay(200).duration(480)}
          style={styles.section}
        >
          <ArabicText weight="bold" style={[styles.sectionTitle, { color: flow.text }]}>
            الحروف المكتشفة
          </ArabicText>

          <View style={styles.charGrid}>
            {detectedChars.map((char, index) => (
              <Animated.View
                key={`${char.glyph}-${index}`}
                entering={FadeInUp.delay(220 + index * 40).duration(380)}
                style={[
                  styles.charCard,
                  {
                    backgroundColor: flow.card,
                    borderColor: flow.cardBorder,
                  },
                ]}
              >
                <Text style={[styles.charGlyph, { color: flow.text }]}>{char.glyph}</Text>
                <ArabicText style={[styles.charArabicLetter, { color: Palette.ochreClay }]}>
                  {char.arabicLetter}
                </ArabicText>
                <ArabicText style={[styles.charName, { color: flow.progressLabel }]}>
                  {char.nameAr}
                </ArabicText>
                <ArabicText style={[styles.charConf, { color: flow.textSecondary }]}>
                  {char.conf}%
                </ArabicText>
              </Animated.View>
            ))}
          </View>
        </Animated.View>

        <Animated.View
          entering={FadeInDown.delay(320).duration(480)}
          style={styles.section}
        >
          <ArabicText weight="bold" style={[styles.sectionTitle, { color: flow.text }]}>
            الترجمة العربية
          </ArabicText>

          <View
            style={[
              styles.meaningCard,
              {
                backgroundColor: flow.card,
                borderColor: flow.cardBorder,
              },
            ]}
          >
            <ArabicText
              weight="semiBold"
              style={[styles.meaningValue, { color: flow.text, textAlign: 'right' }]}
            >
              {arabicTranslation || '—'}
            </ArabicText>
            <ArabicText style={[styles.meaningHint, { color: flow.textSecondary }]}>
              تحويل حرفي من المسند إلى العربية
            </ArabicText>
          </View>
        </Animated.View>

        <Animated.View
          entering={FadeInDown.delay(400).duration(480)}
          style={styles.actions}
        >
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="وثّق النقش"
            onPress={() => {
              const imageUri = Array.isArray(uri) ? uri[0] : uri ?? 'mock';
              router.push({
                pathname: '/documentation',
                params: {
                  uri: imageUri,
                  confidence: String(avgConf || 96),
                  scanId: typeof scanId === 'string' ? scanId : Array.isArray(scanId) ? scanId[0] : '',
                },
              });
            }}
            style={({ pressed }) => [styles.documentBtn, pressed && styles.pressed]}
          >
            <SymbolView
              name={{
                ios: 'archivebox.fill',
                android: 'inventory_2',
                web: 'inventory_2',
              }}
              size={18}
              tintColor={Palette.duneBeige}
            />
            <ArabicText weight="bold" style={styles.documentBtnLabel}>
              وثّق النقش
            </ArabicText>
          </Pressable>

          <Pressable
            accessibilityRole="button"
            onPress={() => router.replace('/capture')}
            style={({ pressed }) => [
              styles.secondaryBtn,
              {
                backgroundColor: flow.card,
                borderColor: flow.cardBorder,
              },
              pressed && styles.pressed,
            ]}
          >
            <ArabicText
              weight="semiBold"
              style={[styles.secondaryBtnLabel, { color: flow.text }]}
            >
              تصوير نص آخر
            </ArabicText>
          </Pressable>

          <Pressable
            accessibilityRole="button"
            onPress={goHome}
            style={({ pressed }) => [styles.primaryBtn, pressed && styles.pressed]}
          >
            <ArabicText weight="semiBold" style={styles.primaryBtnLabel}>
              العودة للرئيسية
            </ArabicText>
          </Pressable>
        </Animated.View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
  },

  topBar: {
    paddingHorizontal: Spacing.three,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: Spacing.one,
  },

  topTitle: {
    fontSize: 15,
  },

  iconBtn: {
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: StyleSheet.hairlineWidth,
  },

  scroll: {
    paddingHorizontal: Spacing.three,
    gap: Spacing.three,
  },

  previewCard: {
    alignSelf: 'center',
    width: '100%',
    maxWidth: 420,
    aspectRatio: 1.55,
    borderRadius: 16,
    ...Platform.select({
      ios: {
        shadowColor: Palette.ochreClay,
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.16,
        shadowRadius: 12,
      },
      android: { elevation: 4 },
      default: {},
    }),
  },

  previewBorder: {
    flex: 1,
    borderRadius: 16,
    padding: 1,
  },

  previewImage: {
    flex: 1,
    borderRadius: 15,
    width: '100%',
    backgroundColor: 'rgba(18, 12, 8, 0.35)',
  },

  badge: {
    position: 'absolute',
    top: 8,
    left: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 999,
    borderWidth: StyleSheet.hairlineWidth,
  },

  toggleBtn: {
    position: 'absolute',
    top: 8,
    right: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 999,
    borderWidth: StyleSheet.hairlineWidth,
  },

  badgeText: {
    color: Palette.sandstone,
    fontSize: 10,
  },

  section: {
    gap: 6,
  },

  sectionTitle: {
    fontSize: 14,
    textAlign: 'right',
  },

  scriptBlock: {
    borderRadius: 14,
    paddingVertical: Spacing.two,
    paddingHorizontal: Spacing.two,
    alignItems: 'center',
    borderWidth: StyleSheet.hairlineWidth,
  },

  scriptLine: {
    fontFamily: Fonts.script,
    fontSize: 20,
    lineHeight: 30,
    color: Palette.sandstone,
    letterSpacing: 2,
    textAlign: 'center',
    writingDirection: 'rtl',
  },

  charGrid: {
    flexDirection: 'row-reverse',
    flexWrap: 'wrap',
    gap: 6,
    justifyContent: 'flex-start',
  },

  charCard: {
    width: '18%',
    minWidth: 54,
    flexGrow: 1,
    maxWidth: 64,
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 4,
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    gap: 1,
  },

  charGlyph: {
    fontFamily: Fonts.script,
    fontSize: 20,
    lineHeight: 26,
  },

  charArabicLetter: {
    fontSize: 16,
    lineHeight: 22,
  },

  charName: {
    fontSize: 10,
  },

  charConf: {
    fontSize: 9,
    marginTop: 1,
  },

  meaningCard: {
    borderRadius: 14,
    paddingVertical: Spacing.two,
    paddingHorizontal: Spacing.three,
    borderWidth: StyleSheet.hairlineWidth,
  },

  meaningValue: {
    fontSize: 18,
    lineHeight: 28,
  },

  meaningHint: {
    fontSize: 11,
    marginTop: 6,
    textAlign: 'right',
  },

  actions: {
    gap: Spacing.two,
    marginTop: Spacing.half,
  },

  documentBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    paddingVertical: 15,
    borderRadius: 999,
    backgroundColor: Palette.ochreClay,
    ...Platform.select({
      ios: {
        shadowColor: Palette.ochreClay,
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.28,
        shadowRadius: 12,
      },
      android: { elevation: 4 },
      default: {},
    }),
  },

  documentBtnLabel: {
    color: Palette.duneBeige,
    fontSize: 16,
  },

  primaryBtn: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 999,
    backgroundColor: Palette.sandstone,
  },

  primaryBtnLabel: {
    color: Palette.deepSandBrown,
    fontSize: 14,
  },

  secondaryBtn: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 11,
    borderRadius: 999,
    borderWidth: StyleSheet.hairlineWidth,
  },

  secondaryBtnLabel: {
    fontSize: 13,
  },

  pressed: {
    opacity: 0.82,
    transform: [{ scale: 0.98 }],
  },
});
