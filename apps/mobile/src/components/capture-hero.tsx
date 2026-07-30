import { LinearGradient } from 'expo-linear-gradient';
import { SymbolView } from 'expo-symbols';
import {
  Platform,
  Pressable,
  StyleSheet,
  View,
  type StyleProp,
  type ViewStyle,
} from 'react-native';

import { ArabicText } from '@/components/arabic-text';
import { Palette, Spacing } from '@/constants/theme';

type CaptureHeroProps = {
  onPress?: () => void;
  style?: StyleProp<ViewStyle>;
};

export function CaptureHero({ onPress, style }: CaptureHeroProps) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel="تصوير نص"
      onPress={onPress}
      style={({ pressed }) => [
        styles.shadowWrap,
        pressed && styles.pressed,
        style,
      ]}
    >
      <View style={styles.card}>
        <LinearGradient
          colors={['#A67C4A', '#8B5A2B', '#6B4220', '#3F2818']}
          locations={[0, 0.32, 0.68, 1]}
          start={{ x: 0.15, y: 0 }}
          end={{ x: 0.85, y: 1 }}
          style={StyleSheet.absoluteFill}
        />

        <LinearGradient
          colors={[
            'rgba(255,248,230,0.18)',
            'transparent',
          ]}
          start={{ x: 0.1, y: 0 }}
          end={{ x: 0.7, y: 0.7 }}
          style={StyleSheet.absoluteFill}
        />

        <LinearGradient
          colors={[
            'rgba(255,250,240,0.16)',
            'transparent',
          ]}
          start={{ x: 0.5, y: 0 }}
          end={{ x: 0.5, y: 0.45 }}
          style={styles.glossBand}
        />

        <View pointerEvents="none" style={styles.innerBevel} />

        <View style={styles.content}>
          <View style={styles.cameraButton}>
            <SymbolView
              name={{
                ios: 'camera.fill',
                android: 'photo_camera',
                web: 'photo_camera',
              }}
              size={26}
              tintColor={Palette.duneBeige}
            />
          </View>

          <View style={styles.textBlock}>
            <ArabicText weight="bold" style={styles.title}>
              تصوير نص
            </ArabicText>

            <ArabicText style={styles.subtitle}>
              التقط صورة لاكتشاف الكتابة والحروف
            </ArabicText>
          </View>
        </View>
      </View>
    </Pressable>
  );
}

const CAMERA_SIZE = 64;

const styles = StyleSheet.create({
  shadowWrap: {
    borderRadius: 24,

    ...Platform.select({
      ios: {
        shadowColor: '#3A2414',
        shadowOffset: {
          width: 0,
          height: 12,
        },
        shadowOpacity: 0.26,
        shadowRadius: 18,
      },

      android: {
        elevation: 8,
      },

      default: {},
    }),
  },

  card: {
    borderRadius: 24,
    overflow: 'hidden',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(205, 189, 157, 0.28)',
  },

  glossBand: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    height: 72,
  },

  innerBevel: {
    ...StyleSheet.absoluteFill,

    borderRadius: 24,
    borderWidth: 1,
    borderColor: 'rgba(255,248,235,0.16)',
  },

  content: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: Spacing.five,
    paddingBottom: Spacing.five,
    paddingHorizontal: Spacing.four,
    gap: Spacing.three,
  },

  cameraButton: {
    width: CAMERA_SIZE,
    height: CAMERA_SIZE,
    borderRadius: CAMERA_SIZE / 2,

    alignItems: 'center',
    justifyContent: 'center',

    backgroundColor: 'rgba(247,244,236,0.18)',

    borderWidth: 1,
    borderColor: 'rgba(255,248,235,0.4)',
  },

  textBlock: {
    alignItems: 'center',
    gap: Spacing.one,
    paddingHorizontal: Spacing.two,
  },

  title: {
    color: Palette.duneBeige,
    fontSize: 22,
    lineHeight: 32,
    textAlign: 'center',

    textShadowColor: 'rgba(58,36,20,0.3)',
    textShadowOffset: {
      width: 0,
      height: 1,
    },
    textShadowRadius: 3,
  },

  subtitle: {
    color: 'rgba(247,244,236,0.84)',
    fontSize: 14,
    lineHeight: 22,
    textAlign: 'center',
  },

  pressed: {
    opacity: 0.94,
    transform: [
      {
        scale: 0.985,
      },
    ],
  },
});