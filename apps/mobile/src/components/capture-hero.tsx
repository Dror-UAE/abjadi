import { Image } from 'expo-image';
import { LinearGradient } from 'expo-linear-gradient';
import { SymbolView } from 'expo-symbols';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  FlatList,
  Platform,
  Pressable,
  StyleSheet,
  View,
  type LayoutChangeEvent,
  type ListRenderItemInfo,
  type NativeScrollEvent,
  type NativeSyntheticEvent,
  type StyleProp,
  type ViewStyle,
} from 'react-native';
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';

import { ArabicText } from '@/components/arabic-text';
import { Palette, Spacing } from '@/constants/theme';

const HERO_IMAGES = [
  require('../../assets/images/musnad-hero.jpg'),
  require('../../assets/images/carsol-1.jpg'),
  require('../../assets/images/scanned-image.jpg'),
  require('../../assets/images/home-mock-1.jpeg'),
] as const;

const AUTO_ADVANCE_MS = 4200;

type CaptureHeroProps = {
  onCapture?: () => void;
  onLibrary?: () => void;
  style?: StyleProp<ViewStyle>;
};

export function CaptureHero({ onCapture, onLibrary, style }: CaptureHeroProps) {
  const listRef = useRef<FlatList<(typeof HERO_IMAGES)[number]>>(null);
  const indexRef = useRef(0);
  const [width, setWidth] = useState(0);
  const [activeIndex, setActiveIndex] = useState(0);

  const enter = useSharedValue(0);

  useEffect(() => {
    enter.value = withTiming(1, {
      duration: 720,
      easing: Easing.out(Easing.cubic),
    });
  }, [enter]);

  const fadeStyle = useAnimatedStyle(() => ({
    opacity: enter.value,
    transform: [{ translateY: (1 - enter.value) * 10 }],
  }));

  const onLayout = useCallback((event: LayoutChangeEvent) => {
    const next = Math.round(event.nativeEvent.layout.width);
    if (next > 0) setWidth(next);
  }, []);

  const goToIndex = useCallback(
    (next: number, animated = true) => {
      if (width <= 0) return;
      indexRef.current = next;
      setActiveIndex(next);
      listRef.current?.scrollToOffset({
        offset: next * width,
        animated,
      });
    },
    [width]
  );

  useEffect(() => {
    if (width <= 0) return;

    const timer = setInterval(() => {
      const next = (indexRef.current + 1) % HERO_IMAGES.length;
      goToIndex(next, true);
    }, AUTO_ADVANCE_MS);

    return () => clearInterval(timer);
  }, [goToIndex, width]);

  const onScrollEnd = useCallback(
    (event: NativeSyntheticEvent<NativeScrollEvent>) => {
      if (width <= 0) return;
      const next = Math.round(event.nativeEvent.contentOffset.x / width);
      const clamped = Math.max(0, Math.min(HERO_IMAGES.length - 1, next));
      indexRef.current = clamped;
      setActiveIndex(clamped);
    },
    [width]
  );

  const renderItem = useCallback(
    ({ item }: ListRenderItemInfo<(typeof HERO_IMAGES)[number]>) => (
      <View style={{ width: width || 1, height: '100%' }}>
        <Image
          source={item}
          style={StyleSheet.absoluteFill}
          contentFit="cover"
          accessibilityIgnoresInvertColors
        />
      </View>
    ),
    [width]
  );

  return (
    <Animated.View style={[styles.shadowWrap, fadeStyle, style]} onLayout={onLayout}>
      <View style={styles.card}>
        {width > 0 ? (
          <FlatList
            ref={listRef}
            data={[...HERO_IMAGES]}
            keyExtractor={(_, i) => `hero-${i}`}
            renderItem={renderItem}
            horizontal
            pagingEnabled
            bounces={false}
            showsHorizontalScrollIndicator={false}
            onMomentumScrollEnd={onScrollEnd}
            style={StyleSheet.absoluteFill}
            getItemLayout={(_, index) => ({
              length: width,
              offset: width * index,
              index,
            })}
          />
        ) : (
          <Image
            source={HERO_IMAGES[0]}
            style={StyleSheet.absoluteFill}
            contentFit="cover"
            accessibilityIgnoresInvertColors
          />
        )}

        <LinearGradient
          pointerEvents="none"
          colors={[
            'rgba(35, 22, 12, 0.35)',
            'rgba(35, 22, 12, 0.18)',
            'rgba(25, 15, 8, 0.78)',
          ]}
          locations={[0, 0.42, 1]}
          style={StyleSheet.absoluteFill}
        />

        <View pointerEvents="none" style={styles.innerBevel} />

        <View style={styles.content} pointerEvents="box-none">
          <View style={styles.topRow} pointerEvents="box-none">
            <View style={styles.brandRow}>
              <Image
                source={require('../../assets/images/logo.png')}
                style={styles.logo}
                contentFit="contain"
                accessibilityLabel="أبجدي"
              />
              <ArabicText weight="bold" style={styles.brandName}>
                أبجدي
              </ArabicText>
            </View>

            <View style={styles.dots} pointerEvents="none">
              {HERO_IMAGES.map((_, i) => (
                <View
                  key={`dot-${i}`}
                  style={[styles.dot, i === activeIndex && styles.dotActive]}
                />
              ))}
            </View>
          </View>

          <View style={styles.copyBlock} pointerEvents="none">
            <ArabicText weight="bold" style={styles.headline}>
              اقرأ ما كتبته الأحجار
            </ArabicText>
            <ArabicText style={styles.subhead}>
              حلّل النقوش والكتابات القديمة بصورة واحدة
            </ArabicText>
          </View>

          <View style={styles.ctaRow}>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="من المكتبة"
              onPress={onLibrary}
              style={({ pressed }) => [
                styles.secondaryCta,
                pressed && styles.pressedSoft,
              ]}
            >
              <SymbolView
                name={{
                  ios: 'photo.on.rectangle',
                  android: 'photo_library',
                  web: 'photo_library',
                }}
                size={18}
                tintColor={Palette.duneBeige}
              />
              <ArabicText weight="medium" style={styles.secondaryLabel}>
                المكتبة
              </ArabicText>
            </Pressable>

            <Pressable
              accessibilityRole="button"
              accessibilityLabel="ابدأ التحليل"
              onPress={onCapture}
              style={({ pressed }) => [
                styles.primaryCta,
                pressed && styles.pressedPrimary,
              ]}
            >
              <SymbolView
                name={{
                  ios: 'camera.fill',
                  android: 'photo_camera',
                  web: 'photo_camera',
                }}
                size={18}
                tintColor={Palette.deepSandBrown}
              />
              <ArabicText weight="bold" style={styles.primaryLabel}>
                ابدأ التحليل
              </ArabicText>
            </Pressable>
          </View>
        </View>
      </View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  shadowWrap: {
    borderRadius: 26,
    ...Platform.select({
      ios: {
        shadowColor: '#2A1810',
        shadowOffset: { width: 0, height: 14 },
        shadowOpacity: 0.28,
        shadowRadius: 22,
      },
      android: { elevation: 10 },
      default: {},
    }),
  },

  card: {
    borderRadius: 26,
    overflow: 'hidden',
    minHeight: 268,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(214, 182, 147, 0.35)',
  },

  innerBevel: {
    ...StyleSheet.absoluteFill,
    borderRadius: 26,
    borderWidth: 1,
    borderColor: 'rgba(255, 248, 235, 0.14)',
  },

  content: {
    flex: 1,
    justifyContent: 'space-between',
    paddingTop: Spacing.three,
    paddingBottom: Spacing.three,
    paddingHorizontal: Spacing.three,
    minHeight: 268,
  },

  topRow: {
    flexDirection: 'row-reverse',
    alignItems: 'center',
    justifyContent: 'space-between',
  },

  brandRow: {
    flexDirection: 'row-reverse',
    alignItems: 'center',
    gap: Spacing.two,
  },

  logo: {
    width: 36,
    height: 36,
  },

  brandName: {
    color: Palette.duneBeige,
    fontSize: 20,
    letterSpacing: 0.3,
    textShadowColor: 'rgba(20, 12, 6, 0.45)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 4,
  },

  dots: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },

  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: 'rgba(247, 244, 236, 0.35)',
  },

  dotActive: {
    width: 16,
    backgroundColor: Palette.duneBeige,
  },

  copyBlock: {
    alignItems: 'flex-end',
    gap: Spacing.one,
    paddingHorizontal: Spacing.one,
    marginTop: Spacing.four,
  },

  headline: {
    color: Palette.duneBeige,
    fontSize: 26,
    lineHeight: 36,
    textAlign: 'right',
    textShadowColor: 'rgba(20, 12, 6, 0.5)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 6,
  },

  subhead: {
    color: 'rgba(247, 244, 236, 0.86)',
    fontSize: 14,
    lineHeight: 22,
    textAlign: 'right',
  },

  ctaRow: {
    flexDirection: 'row-reverse',
    alignItems: 'center',
    gap: Spacing.two,
    marginTop: Spacing.three,
  },

  primaryCta: {
    flex: 1.35,
    minHeight: 52,
    borderRadius: 16,
    backgroundColor: Palette.duneBeige,
    flexDirection: 'row-reverse',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingHorizontal: Spacing.three,
  },

  primaryLabel: {
    color: Palette.deepSandBrown,
    fontSize: 15,
  },

  secondaryCta: {
    flex: 1,
    minHeight: 52,
    borderRadius: 16,
    backgroundColor: 'rgba(247, 244, 236, 0.14)',
    borderWidth: 1,
    borderColor: 'rgba(247, 244, 236, 0.28)',
    flexDirection: 'row-reverse',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingHorizontal: Spacing.two,
  },

  secondaryLabel: {
    color: Palette.duneBeige,
    fontSize: 14,
  },

  pressedPrimary: {
    opacity: 0.92,
    transform: [{ scale: 0.985 }],
  },

  pressedSoft: {
    opacity: 0.88,
  },
});
