import { Image } from 'expo-image';
import * as Linking from 'expo-linking';
import { StatusBar } from 'expo-status-bar';
import { useCallback, useEffect } from 'react';
import { BackHandler, Pressable, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { ArabicText } from '@/components/arabic-text';
import { Palette, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import type { VersionCheckResult } from '@/lib/version-check';

type ForceUpdateScreenProps = {
  result: VersionCheckResult;
};

export function ForceUpdateScreen({ result }: ForceUpdateScreenProps) {
  const insets = useSafeAreaInsets();
  const { colors, isDark } = useTheme();
  const storeUrl = result.storeUrl?.trim() ?? '';
  const canOpenStore = storeUrl.length > 0;

  const messageAr = result.updateMessage?.ar ?? 'يرجى تحديث التطبيق للمتابعة.';
  const messageEn =
    result.updateMessage?.en ?? 'Please update the app to continue.';

  useEffect(() => {
    const sub = BackHandler.addEventListener('hardwareBackPress', () => true);
    return () => sub.remove();
  }, []);

  const openStore = useCallback(() => {
    if (!canOpenStore) return;
    void Linking.openURL(storeUrl);
  }, [canOpenStore, storeUrl]);

  return (
    <View
      style={[
        styles.root,
        {
          backgroundColor: colors.background,
          paddingTop: insets.top + Spacing.five,
          paddingBottom: Math.max(insets.bottom, Spacing.three) + Spacing.three,
        },
      ]}
    >
      <StatusBar style={isDark ? 'light' : 'dark'} />

      <View style={styles.content}>
        <Image
          source={require('../../assets/images/logo.png')}
          style={styles.logo}
          contentFit="contain"
          accessibilityLabel="Abjadi"
        />

        <ArabicText weight="bold" style={[styles.title, { color: colors.text }]}>
          تحديث مطلوب
        </ArabicText>

        <ArabicText style={[styles.messageAr, { color: colors.textSecondary }]}>
          {messageAr}
        </ArabicText>

        <Text style={[styles.messageEn, { color: colors.textSecondary }]}>
          {messageEn}
        </Text>
      </View>

      <View style={styles.footer}>
        <Pressable
          accessibilityRole="button"
          accessibilityState={{ disabled: !canOpenStore }}
          disabled={!canOpenStore}
          onPress={openStore}
          style={({ pressed }) => [
            styles.cta,
            !canOpenStore && styles.ctaDisabled,
            pressed && canOpenStore && styles.ctaPressed,
          ]}
        >
          <ArabicText weight="medium" style={styles.ctaLabel}>
            حدّث
          </ArabicText>
          <Text style={styles.ctaSecondary}>Update</Text>
        </Pressable>

        {!canOpenStore ? (
          <ArabicText style={[styles.storeHint, { color: colors.textSecondary }]}>
            رابط المتجر غير متاح حالياً. حدّث التطبيق من App Store أو Google Play عند
            توفره.
          </ArabicText>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    paddingHorizontal: Spacing.four,
  },

  content: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },

  logo: {
    width: 128,
    height: 128,
    marginBottom: Spacing.four,
  },

  title: {
    fontSize: 26,
    textAlign: 'center',
  },

  messageAr: {
    marginTop: Spacing.three,
    fontSize: 16,
    lineHeight: 26,
    textAlign: 'center',
  },

  messageEn: {
    marginTop: Spacing.two,
    fontSize: 14,
    lineHeight: 22,
    textAlign: 'center',
    opacity: 0.85,
    writingDirection: 'ltr',
  },

  footer: {
    width: '100%',
    gap: Spacing.three,
  },

  cta: {
    alignSelf: 'stretch',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 60,
    backgroundColor: Palette.deepSandBrown,
    borderRadius: 18,
    overflow: 'hidden',
  },

  ctaDisabled: {
    opacity: 0.45,
  },

  ctaPressed: {
    opacity: 0.88,
  },

  ctaLabel: {
    color: Palette.duneBeige,
    fontSize: 17,
  },

  ctaSecondary: {
    marginTop: 0.5,
    color: Palette.duneBeige,
    fontSize: 13,
    opacity: 0.85,
    writingDirection: 'ltr',
  },

  storeHint: {
    fontSize: 13,
    lineHeight: 20,
    textAlign: 'center',
  },
});
