import { LinearGradient } from 'expo-linear-gradient';
import { router, useLocalSearchParams } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SymbolView } from 'expo-symbols';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Dimensions,
  Image,
  Platform,
  Pressable,
  StyleSheet,
  TextInput,
  View,
  type ImageSourcePropType,
} from 'react-native';
import Animated, {
  Easing,
  FadeIn,
  FadeInDown,
  interpolate,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withTiming,
  type SharedValue,
} from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { ArabicText } from '@/components/arabic-text';
import { FlowChrome, Palette, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { ApiError, checkApiHealth, getApiBaseUrl, uploadOcr } from '@/lib/api';
import { isLikelyNetworkError, loadApiConfig, setApiBaseUrl } from '@/lib/api-config';
import { saveScan } from '@/lib/scan-store';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const CARD_SIZE = Math.min(SCREEN_WIDTH * 0.72, 300);
const RING_SIZE = CARD_SIZE + 36;
const FALLBACK_IMAGE = require('../../assets/images/scanned-image.jpg');

type FlowTokens = (typeof FlowChrome)[keyof typeof FlowChrome];
type StepStatus = 'done' | 'active' | 'pending';

type Step = {
  id: string;
  label: string;
  status: StepStatus;
};

const STEP_LABELS = [
  'تجهيز الصورة',
  'تحديد منطقة النص',
  'التعرف على الحروف',
  'ترجمة النص',
] as const;

function resolveImageUri(uri?: string | string[]): string | undefined {
  const value = Array.isArray(uri) ? uri[0] : uri;
  if (!value || value === 'mock') return undefined;
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function resolveImageSource(uri?: string | string[]): ImageSourcePropType {
  const decoded = resolveImageUri(uri);
  if (decoded) return { uri: decoded };
  return FALLBACK_IMAGE;
}

function StepRow({
  label,
  status,
  index,
  flow,
}: {
  label: string;
  status: StepStatus;
  index: number;
  flow: FlowTokens;
}) {
  const icon =
    status === 'done'
      ? { ios: 'checkmark.circle.fill' as const, android: 'check_circle' as const }
      : status === 'active'
        ? { ios: 'circle.dotted' as const, android: 'radio_button_checked' as const }
        : { ios: 'circle' as const, android: 'radio_button_unchecked' as const };

  const tint =
    status === 'done'
      ? flow.stepDone
      : status === 'active'
        ? flow.stepActive
        : flow.stepPending;

  const labelColor =
    status === 'pending'
      ? flow.stepLabelPending
      : status === 'active'
        ? flow.stepLabelActive
        : flow.stepLabelDone;

  return (
    <Animated.View
      entering={FadeInDown.delay(120 + index * 70).duration(420)}
      style={styles.stepRow}
    >
      <ArabicText
        weight={status === 'active' ? 'semiBold' : 'regular'}
        style={[styles.stepLabel, { color: labelColor }]}
      >
        {label}
      </ArabicText>
      <SymbolView
        name={{ ios: icon.ios, android: icon.android, web: icon.android }}
        size={18}
        tintColor={tint}
      />
    </Animated.View>
  );
}

function AnalyzingOrb({
  imageSource,
  progress,
  flow,
}: {
  imageSource: ImageSourcePropType;
  progress: SharedValue<number>;
  flow: FlowTokens;
}) {
  const spin = useSharedValue(0);
  const scan = useSharedValue(0);
  const glow = useSharedValue(0.35);

  useEffect(() => {
    spin.value = withRepeat(
      withTiming(1, { duration: 4200, easing: Easing.linear }),
      -1,
      false
    );
    scan.value = withRepeat(
      withTiming(1, { duration: 2400, easing: Easing.inOut(Easing.quad) }),
      -1,
      true
    );
    glow.value = withRepeat(
      withTiming(1, { duration: 1800, easing: Easing.inOut(Easing.sin) }),
      -1,
      true
    );
  }, [spin, scan, glow]);

  const ringStyle = useAnimatedStyle(() => ({
    transform: [{ rotate: `${spin.value * 360}deg` }],
  }));

  const glowStyle = useAnimatedStyle(() => ({
    opacity: interpolate(glow.value, [0, 1], [0.28, 0.62]),
    transform: [{ scale: interpolate(glow.value, [0, 1], [0.98, 1.04]) }],
  }));

  const scanStyle = useAnimatedStyle(() => ({
    transform: [
      {
        translateY: interpolate(scan.value, [0, 1], [-CARD_SIZE * 0.42, CARD_SIZE * 0.42]),
      },
    ],
    opacity: interpolate(scan.value, [0, 0.15, 0.85, 1], [0, 1, 1, 0]),
  }));

  const arcStyle = useAnimatedStyle(() => ({
    transform: [{ rotate: `${-90 + progress.value * 360}deg` }],
  }));

  return (
    <View style={styles.orbWrap}>
      <Animated.View
        style={[styles.glowHalo, { backgroundColor: flow.glow }, glowStyle]}
      />

      <Animated.View
        style={[
          styles.spinRing,
          {
            borderColor: flow.ringSoft,
            borderTopColor: Palette.sandstone,
            borderRightColor: 'rgba(193, 138, 59, 0.55)',
          },
          ringStyle,
        ]}
      />

      <Animated.View
        style={[
          styles.progressArc,
          {
            borderColor: flow.ringSoft,
            borderTopColor: Palette.ochreClay,
            borderRightColor: 'rgba(214, 182, 147, 0.35)',
          },
          arcStyle,
        ]}
      />

      <View style={styles.cardOuter}>
        <LinearGradient
          colors={[
            'rgba(214, 182, 147, 0.7)',
            'rgba(193, 138, 59, 0.4)',
            'rgba(89, 58, 35, 0.25)',
            'rgba(214, 182, 147, 0.55)',
          ]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.cardGlowBorder}
        >
          <View style={[styles.cardInner, { backgroundColor: flow.cardInner }]}>
            <Image source={imageSource} style={styles.cardImage} resizeMode="contain" />

            <Animated.View pointerEvents="none" style={[styles.scanBand, scanStyle]}>
              <LinearGradient
                colors={[
                  'transparent',
                  'rgba(214, 182, 147, 0.1)',
                  'rgba(247, 244, 236, 0.65)',
                  'rgba(214, 182, 147, 0.14)',
                  'transparent',
                ]}
                start={{ x: 0.5, y: 0 }}
                end={{ x: 0.5, y: 1 }}
                style={styles.scanGradient}
              />
            </Animated.View>

            <LinearGradient
              colors={['rgba(10,7,5,0.16)', 'transparent', 'rgba(10,7,5,0.3)']}
              style={StyleSheet.absoluteFill}
            />
          </View>
        </LinearGradient>
      </View>
    </View>
  );
}

export function AnalyzingScreen() {
  const insets = useSafeAreaInsets();
  const { flow } = useTheme();
  const { uri } = useLocalSearchParams<{ uri?: string }>();
  const imageUri = useMemo(() => resolveImageUri(uri), [uri]);
  const imageSource = useMemo(() => resolveImageSource(uri), [uri]);

  const progress = useSharedValue(0.08);
  const [stepIndex, setStepIndex] = useState(0);
  const [percent, setPercent] = useState(8);
  const [error, setError] = useState<string | null>(null);
  const [retryKey, setRetryKey] = useState(0);
  const [apiUrlDraft, setApiUrlDraft] = useState('');
  const [savingServer, setSavingServer] = useState(false);
  const finishedRef = useRef(false);

  useEffect(() => {
    void loadApiConfig().then((url) => setApiUrlDraft(url));
  }, []);

  useEffect(() => {
    finishedRef.current = false;
    setError(null);
    setStepIndex(0);
    setPercent(8);
    progress.value = 0.08;
    progress.value = withTiming(0.85, {
      duration: 14000,
      easing: Easing.out(Easing.cubic),
    });

    const stepTimers = [
      setTimeout(() => setStepIndex(1), 400),
      setTimeout(() => setStepIndex(2), 1200),
      setTimeout(() => setStepIndex(3), 2800),
    ];

    const tick = setInterval(() => {
      setPercent((current) => {
        if (current >= 90) return current;
        return current + 1;
      });
    }, 120);

    const controller = new AbortController();

    async function run() {
      try {
        if (!imageUri) {
          throw new ApiError(
            'لا توجد صورة للتحليل. أعد التصوير أو اختر صورة من المعرض.'
          );
        }

        const response = await uploadOcr(imageUri, controller.signal);
        if (controller.signal.aborted || finishedRef.current) return;

        if (!response.ok) {
          throw new ApiError(response.detail || response.error);
        }

        const scanId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        await saveScan({
          id: scanId,
          imageUri,
          result: response,
          createdAt: Date.now(),
          serverScanId: response.scanId,
          publicId: response.publicId,
        });

        finishedRef.current = true;
        setStepIndex(4);
        setPercent(100);
        progress.value = withTiming(1, { duration: 350 });

        setTimeout(() => {
          router.replace({
            pathname: '/result',
            params: {
              uri: encodeURIComponent(imageUri),
              scanId,
            },
          });
        }, 450);
      } catch (err) {
        if (controller.signal.aborted || finishedRef.current) return;
        const message =
          err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : 'فشل الاتصال بخادم التحليل';
        setError(
          isLikelyNetworkError(message)
            ? 'تعذر الاتصال بخادم التحليل. عدّل عنوان الخادم أدناه.'
            : message
        );
        setStepIndex(0);
        progress.value = withTiming(0.12, { duration: 300 });
        setPercent(0);
      }
    }

    void run();

    return () => {
      controller.abort();
      stepTimers.forEach(clearTimeout);
      clearInterval(tick);
    };
  }, [imageUri, progress, retryKey]);

  const steps: Step[] = STEP_LABELS.map((label, index) => ({
    id: String(index),
    label,
    status:
      index < stepIndex ? 'done' : index === stepIndex ? 'active' : 'pending',
  }));

  if (stepIndex >= 4) {
    steps.forEach((step) => {
      step.status = 'done';
    });
  }

  const barStyle = useAnimatedStyle(() => ({
    width: `${interpolate(progress.value, [0, 1], [6, 100])}%`,
  }));

  const close = () => {
    if (router.canGoBack()) {
      router.back();
    } else {
      router.replace('/home');
    }
  };

  const saveServerAndRetry = async () => {
    setSavingServer(true);
    try {
      const url = await setApiBaseUrl(apiUrlDraft);
      setApiUrlDraft(url);
      const ok = await checkApiHealth();
      if (!ok) {
        setError('الخادم لا يستجيب. تأكد أن bun dev يعمل في apps/api');
        return;
      }
      setError(null);
      setRetryKey((k) => k + 1);
    } catch {
      setError(`تعذر الوصول إلى ${getApiBaseUrl()}`);
    } finally {
      setSavingServer(false);
    }
  };

  return (
    <View style={[styles.root, { backgroundColor: flow.background }]}>
      <StatusBar style={flow.statusBar} />

      <LinearGradient colors={[...flow.gradient]} style={StyleSheet.absoluteFill} />

      <View
        pointerEvents="none"
        style={[styles.ambientGlow, { top: insets.top + 48 }]}
      >
        <LinearGradient
          colors={[flow.ambient, 'transparent']}
          style={styles.ambientFill}
        />
      </View>

      <View style={[styles.topBar, { paddingTop: insets.top + Spacing.two }]}>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Close"
          onPress={close}
          style={({ pressed }) => [
            styles.closeBtn,
            {
              backgroundColor: flow.iconBtn,
              borderColor: flow.iconBtnBorder,
            },
            pressed && styles.pressed,
          ]}
        >
          <SymbolView
            name={{ ios: 'xmark', android: 'close', web: 'close' }}
            size={16}
            tintColor={flow.icon}
          />
        </Pressable>
      </View>

      <View style={styles.content}>
        <Animated.View entering={FadeIn.duration(500)} style={styles.orbSection}>
          <AnalyzingOrb imageSource={imageSource} progress={progress} flow={flow} />
        </Animated.View>

        <Animated.View
          entering={FadeInDown.delay(180).duration(500)}
          style={styles.copyBlock}
        >
          <ArabicText weight="bold" style={[styles.title, { color: flow.text }]}>
            {error
              ? 'تعذر إكمال التحليل'
              : stepIndex >= 4
                ? 'اكتمل التحليل'
                : 'جاري تحليل النص...'}
          </ArabicText>
          {error ? (
            <ArabicText style={[styles.errorCopy, { color: flow.textSecondary }]}>
              {error}
            </ArabicText>
          ) : null}
        </Animated.View>

        {error ? (
          <View style={styles.errorPanel}>
            <ArabicText style={[styles.serverLabel, { color: flow.textSecondary }]}>
              عنوان خادم التحليل (Mac + Wi‑Fi)
            </ArabicText>
            <TextInput
              value={apiUrlDraft}
              onChangeText={setApiUrlDraft}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
              placeholder="http://192.168.x.x:3001"
              placeholderTextColor={flow.textMuted}
              style={[
                styles.serverInput,
                {
                  color: flow.text,
                  borderColor: flow.cardBorder,
                  backgroundColor: flow.card,
                },
              ]}
            />
            <ArabicText style={[styles.serverHint, { color: flow.textMuted }]}>
              Mac: ipconfig getifaddr en0 — نفس شبكة Wi‑Fi، بدون USB
            </ArabicText>
            <Pressable
              accessibilityRole="button"
              disabled={savingServer}
              onPress={() => void saveServerAndRetry()}
              style={({ pressed }) => [
                styles.retryBtn,
                savingServer && styles.retryBtnDisabled,
                pressed && !savingServer && styles.pressed,
              ]}
            >
              <ArabicText weight="semiBold" style={styles.retryLabel}>
                {savingServer ? 'جاري الاتصال...' : 'حفظ وإعادة المحاولة'}
              </ArabicText>
            </Pressable>
          </View>
        ) : (
          <View
            style={[
              styles.stepsCard,
              {
                backgroundColor: flow.card,
                borderColor: flow.cardBorder,
              },
            ]}
          >
            {steps.map((step, index) => (
              <StepRow
                key={step.id}
                label={step.label}
                status={step.status}
                index={index}
                flow={flow}
              />
            ))}
          </View>
        )}
      </View>

      <View
        style={[
          styles.bottom,
          { paddingBottom: Math.max(insets.bottom, Spacing.three) + Spacing.two },
        ]}
      >
        <View style={styles.progressMeta}>
          <ArabicText
            weight="medium"
            style={[styles.progressLabel, { color: flow.progressLabel }]}
          >
            معالجة الذكاء الاصطناعي
          </ArabicText>
          <ArabicText
            weight="semiBold"
            style={[styles.progressPercent, { color: Palette.ochreClay }]}
          >
            {percent}%
          </ArabicText>
        </View>

        <View style={[styles.progressTrack, { backgroundColor: flow.progressTrack }]}>
          <Animated.View style={[styles.progressFill, barStyle]}>
            <LinearGradient
              colors={[Palette.ochreClay, Palette.sandstone, Palette.limestone]}
              start={{ x: 0, y: 0.5 }}
              end={{ x: 1, y: 0.5 }}
              style={StyleSheet.absoluteFill}
            />
          </Animated.View>
        </View>

        <ArabicText style={[styles.footerHint, { color: flow.textMuted }]}>
          يرجى الانتظار بينما يقرأ أبجدي النص
        </ArabicText>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
  },

  ambientGlow: {
    position: 'absolute',
    alignSelf: 'center',
    width: RING_SIZE + 80,
    height: RING_SIZE + 80,
    borderRadius: (RING_SIZE + 80) / 2,
    overflow: 'hidden',
  },

  ambientFill: {
    flex: 1,
  },

  topBar: {
    paddingHorizontal: Spacing.three,
    zIndex: 2,
  },

  closeBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: StyleSheet.hairlineWidth,
  },

  content: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: Spacing.four,
    gap: Spacing.four,
    marginTop: -Spacing.three,
  },

  orbSection: {
    alignItems: 'center',
    justifyContent: 'center',
  },

  orbWrap: {
    width: RING_SIZE,
    height: RING_SIZE,
    alignItems: 'center',
    justifyContent: 'center',
  },

  glowHalo: {
    position: 'absolute',
    width: RING_SIZE + 28,
    height: RING_SIZE + 28,
    borderRadius: (RING_SIZE + 28) / 2,
    ...Platform.select({
      ios: {
        shadowColor: Palette.ochreClay,
        shadowOffset: { width: 0, height: 0 },
        shadowOpacity: 0.55,
        shadowRadius: 28,
      },
      default: {},
    }),
  },

  spinRing: {
    position: 'absolute',
    width: RING_SIZE,
    height: RING_SIZE,
    borderRadius: RING_SIZE / 2,
    borderWidth: 1.5,
  },

  progressArc: {
    position: 'absolute',
    width: RING_SIZE - 10,
    height: RING_SIZE - 10,
    borderRadius: (RING_SIZE - 10) / 2,
    borderWidth: 2.5,
  },

  cardOuter: {
    width: CARD_SIZE,
    height: CARD_SIZE,
    borderRadius: 28,
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 18 },
        shadowOpacity: 0.45,
        shadowRadius: 28,
      },
      android: {
        elevation: 12,
      },
      default: {},
    }),
  },

  cardGlowBorder: {
    flex: 1,
    borderRadius: 28,
    padding: 1.5,
  },

  cardInner: {
    flex: 1,
    borderRadius: 26.5,
    overflow: 'hidden',
  },

  cardImage: {
    width: '100%',
    height: '100%',
  },

  scanBand: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 56,
  },

  scanGradient: {
    flex: 1,
  },

  copyBlock: {
    alignItems: 'center',
    gap: Spacing.one,
    paddingHorizontal: Spacing.two,
  },

  title: {
    fontSize: 24,
    lineHeight: 34,
    textAlign: 'center',
  },

  errorCopy: {
    fontSize: 13,
    lineHeight: 20,
    textAlign: 'center',
    marginTop: Spacing.one,
  },

  retryBtn: {
    paddingVertical: 12,
    paddingHorizontal: Spacing.four,
    borderRadius: 999,
    backgroundColor: Palette.ochreClay,
    alignItems: 'center',
  },

  retryBtnDisabled: {
    opacity: 0.6,
  },

  retryLabel: {
    color: Palette.duneBeige,
    fontSize: 15,
  },

  errorPanel: {
    width: '100%',
    maxWidth: 340,
    gap: Spacing.two,
  },

  serverLabel: {
    fontSize: 12,
    textAlign: 'right',
  },

  serverInput: {
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: 12,
    paddingHorizontal: Spacing.two,
    paddingVertical: Platform.OS === 'ios' ? 12 : 10,
    fontSize: 14,
    writingDirection: 'ltr',
    textAlign: 'left',
  },

  serverHint: {
    fontSize: 11,
    lineHeight: 16,
    textAlign: 'right',
  },

  stepsCard: {
    width: '100%',
    maxWidth: 340,
    borderRadius: 22,
    paddingVertical: Spacing.three,
    paddingHorizontal: Spacing.three,
    gap: 12,
    borderWidth: StyleSheet.hairlineWidth,
  },

  stepRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.two,
  },

  stepLabel: {
    flex: 1,
    fontSize: 14,
    lineHeight: 22,
    textAlign: 'right',
  },

  bottom: {
    paddingHorizontal: Spacing.four,
    gap: Spacing.two,
  },

  progressMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },

  progressLabel: {
    fontSize: 12,
  },

  progressPercent: {
    fontSize: 13,
  },

  progressTrack: {
    height: 4,
    borderRadius: 999,
    overflow: 'hidden',
  },

  progressFill: {
    height: '100%',
    borderRadius: 999,
    overflow: 'hidden',
  },

  footerHint: {
    fontSize: 12,
    textAlign: 'center',
    marginTop: Spacing.one,
  },

  pressed: {
    opacity: 0.75,
    transform: [{ scale: 0.96 }],
  },
});
