import { LinearGradient } from 'expo-linear-gradient';
import { router, useLocalSearchParams } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SymbolView } from 'expo-symbols';
import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Clipboard,
  Image,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  type ImageSourcePropType,
} from 'react-native';
import Animated, { FadeIn, FadeInDown } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { ArabicText } from '@/components/arabic-text';
import { Fonts, Palette, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { fetchScanById, deleteScan } from '@/lib/api';
import type { ScanRecord } from '@/lib/ocr-types';
import {
  clearScan,
  getScan,
  getScanByServerId,
  recordFromServerDetail,
  upsertScan,
} from '@/lib/scan-store';
import { translateMusnadLines } from '@/lib/musnad-translate';

const FALLBACK_IMAGE = require('../../assets/images/scanned-image.jpg');

const MOCK_LINE = '𐩱𐩡 𐩭𐩸𐩣 𐩫𐩡';
const MOCK_AVG_CONF = 96;

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

  const avgConf = useMemo(() => {
    if (!scan?.result.glyphs?.length) return MOCK_AVG_CONF;
    const glyphs = scan.result.glyphs.filter(
      (g) => !g.isSeparator && g.character && g.character !== '?'
    );
    if (!glyphs.length) return MOCK_AVG_CONF;
    const sum = glyphs.reduce(
      (acc, g) => acc + Math.max(0, Math.min(1, g.confidence)) * 100,
      0
    );
    return Math.round(sum / glyphs.length);
  }, [scan]);

  /** Local capture — can document. */
  const isLocalCapture = Boolean(scan && !scan.id.startsWith('server-'));

  /** Can delete local captures and server history items. */
  const canDelete = Boolean(scan);

  /** Own scan, not already documented, and saved on the server. */
  const canDocument = useMemo(() => {
    if (!scan || !isLocalCapture) return false;
    const alreadyDocumented = Boolean(scan.documentationTitle?.trim());
    const hasServerScan = Boolean(scan.serverScanId || scan.result.scanId);
    return !alreadyDocumented && hasServerScan;
  }, [scan, isLocalCapture]);

  const copyText = (text: string, label: string) => {
    Clipboard.setString(text);
    Alert.alert('تم النسخ', `تم نسخ ${label} إلى الحافظة.`);
  };

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

  const confirmDelete = () => {
    if (!scan || !canDelete) return;

    Alert.alert(
      'حذف التحليل',
      'سيتم حذف هذا التحليل من جهازك ومن الخادم. هل تريد المتابعة؟',
      [
        { text: 'إلغاء', style: 'cancel' },
        {
          text: 'حذف',
          style: 'destructive',
          onPress: () => {
            void (async () => {
              const serverId = scan.serverScanId || scan.result.scanId;
              try {
                if (serverId) {
                  await deleteScan(serverId);
                }
              } catch (err) {
                const message =
                  err instanceof Error ? err.message : 'فشل حذف التحليل من الخادم';
                Alert.alert('تعذر الحذف', message);
                return;
              }

              await clearScan(scan.id);
              goHome();
            })();
          },
        },
      ]
    );
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
              {avgConf}% ثقة
            </ArabicText>
          </View>
        </Animated.View>

        <Animated.View
          entering={FadeInDown.delay(120).duration(480)}
          style={styles.section}
        >
          <View style={styles.sectionHeader}>
            <Pressable
              onPress={() => copyText(detectedLine, 'النص المستخرج')}
              style={({ pressed }) => [styles.copyBtn, pressed && styles.pressed]}
              accessibilityRole="button"
              accessibilityLabel="نسخ النص المستخرج"
            >
              <SymbolView
                name={{ ios: 'doc.on.doc', android: 'content_copy', web: 'content_copy' }}
                size={13}
                tintColor={Palette.ochreClay}
              />
              <ArabicText weight="medium" style={styles.copyBtnLabel}>نسخ</ArabicText>
            </Pressable>
            <ArabicText weight="bold" style={[styles.sectionTitle, { color: flow.text }]}>
              النص المستخرج
            </ArabicText>
          </View>

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
          entering={FadeInDown.delay(220).duration(480)}
          style={styles.section}
        >
          <View style={styles.sectionHeader}>
            <Pressable
              onPress={() => copyText(arabicTranslation || '—', 'الترجمة العربية')}
              style={({ pressed }) => [styles.copyBtn, pressed && styles.pressed]}
              accessibilityRole="button"
              accessibilityLabel="نسخ الترجمة العربية"
            >
              <SymbolView
                name={{ ios: 'doc.on.doc', android: 'content_copy', web: 'content_copy' }}
                size={13}
                tintColor={Palette.ochreClay}
              />
              <ArabicText weight="medium" style={styles.copyBtnLabel}>نسخ</ArabicText>
            </Pressable>
            <ArabicText weight="bold" style={[styles.sectionTitle, { color: flow.text }]}>
              الترجمة العربية
            </ArabicText>
          </View>

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
          {canDocument ? (
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="وثّق النقش"
              onPress={() => {
                const imageUri = Array.isArray(uri) ? uri[0] : uri ?? 'mock';
                const localId =
                  typeof scanId === 'string'
                    ? scanId
                    : Array.isArray(scanId)
                      ? scanId[0]
                      : scan?.id ?? '';
                router.push({
                  pathname: '/documentation',
                  params: {
                    uri: imageUri,
                    confidence: String(avgConf || 96),
                    scanId: localId,
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
          ) : null}

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

          {canDelete ? (
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="حذف التحليل"
              onPress={confirmDelete}
              style={({ pressed }) => [
                styles.deleteBtn,
                {
                  borderColor: 'rgba(166, 48, 36, 0.35)',
                  backgroundColor: 'rgba(166, 48, 36, 0.08)',
                },
                pressed && styles.pressed,
              ]}
            >
              <SymbolView
                name={{ ios: 'trash', android: 'delete', web: 'delete' }}
                size={16}
                tintColor="#A63024"
              />
              <ArabicText weight="semiBold" style={styles.deleteBtnLabel}>
                حذف التحليل
              </ArabicText>
            </Pressable>
          ) : null}
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

  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },

  sectionTitle: {
    fontSize: 14,
    textAlign: 'right',
  },

  copyBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
    backgroundColor: 'rgba(193, 138, 59, 0.1)',
  },

  copyBtnLabel: {
    fontSize: 12,
    color: Palette.ochreClay,
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

  deleteBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 12,
    borderRadius: 999,
    borderWidth: StyleSheet.hairlineWidth,
  },

  deleteBtnLabel: {
    color: '#A63024',
    fontSize: 14,
  },

  pressed: {
    opacity: 0.82,
    transform: [{ scale: 0.98 }],
  },
});
