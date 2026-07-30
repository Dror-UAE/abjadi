import { Image } from 'expo-image';
import { router } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { ImageBackground, Pressable, StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { ArabicText } from '@/components/arabic-text';
import { Palette, Spacing } from '@/constants/theme';

export function WelcomeScreen() {
  const insets = useSafeAreaInsets();

  return (
    <View style={styles.root}>
      <StatusBar style="dark" />

      <ImageBackground
        source={require('../../assets/images/welcome.jpeg')}
        resizeMode="cover"
        style={styles.background}
      >
        <View style={styles.overlay} />

        <View
          style={[
            styles.content,
            {
              paddingTop: insets.top + Spacing.five,
              paddingBottom:
                Math.max(insets.bottom, Spacing.three) + Spacing.three,
            },
          ]}
        >
          <Image
            source={require('../../assets/images/logo.png')}
            style={styles.logo}
            contentFit="contain"
            accessibilityLabel="Abjadi"
          />

          <View style={styles.textGroup}>
            <ArabicText weight="bold" style={styles.title}>
              𐩣𐩧𐩢𐩨𐩱 𐩨𐩫 𐩰𐩺 𐩱𐩨𐩴𐩵𐩺
            </ArabicText>

            <ArabicText style={styles.translation}>
              مرحباً بك في أبجدي
            </ArabicText>
          </View>

          <View style={styles.textGroup}>
            <ArabicText weight="medium" style={styles.subtitle}>
              𐩱𐩫𐩩𐩦𐩰 𐩩𐩧𐩱𐩻 𐩱𐩡𐩭𐩷𐩥𐩷 𐩥𐩱𐩡𐩬𐩤𐩥𐩦
            </ArabicText>

            <ArabicText style={styles.translation}>
              اكتشف تراث الخطوط والنقوش
            </ArabicText>
          </View>

          <View style={styles.footer}>
            <Pressable
              accessibilityRole="button"
              onPress={() => router.replace('/home')}
              style={({ pressed }) => [
                styles.cta,
                pressed && styles.ctaPressed,
              ]}
            >
              <ArabicText weight="medium" style={styles.ctaLabel}>
                𐩱𐩨𐩵𐩱 𐩧𐩢𐩡𐩩𐩫
              </ArabicText>

              <ArabicText style={styles.buttonTranslation}>
                ابدأ رحلتك
              </ArabicText>
            </Pressable>
          </View>
        </View>
      </ImageBackground>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: Palette.limestone,
  },

  background: {
    flex: 1,
  },

  overlay: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(255,255,255,0.45)',
  },

  content: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: Spacing.four,
  },

  logo: {
    width: 148,
    height: 148,
    marginTop: Spacing.six,
  },

  textGroup: {
    alignItems: 'center',
  },

  title: {
    marginTop: Spacing.three,
    fontSize: 26,
    color: Palette.deepSandBrown,
    textAlign: 'center',
  },

  subtitle: {
    marginTop: Spacing.three,
    fontSize: 17,
    color: Palette.deepSandBrown,
    opacity: 0.85,
    textAlign: 'center',
  },

  translation: {
    marginTop: 5,
    fontSize: 15,
    color: Palette.deepSandBrown,
    opacity: 0.75,
    textAlign: 'center',
  },

  footer: {
    flex: 1,
    width: '100%',
    justifyContent: 'flex-end',
    paddingBottom: Spacing.six + Spacing.one,
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

  ctaPressed: {
    opacity: 0.88,
  },

  ctaLabel: {
    color: Palette.duneBeige,
    fontSize: 17,
    fontWeight: '600',
    letterSpacing: 0.2,
  },

  buttonTranslation: {
    marginTop: 0.5,
    color: Palette.duneBeige,
    fontSize: 13,
    opacity: 0.85,
  },
});