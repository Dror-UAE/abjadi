import { router, useLocalSearchParams } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SymbolView } from 'expo-symbols';
import { Platform, Pressable, StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { ArabicText } from '@/components/arabic-text';
import { Palette, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

export function DocumentationSubmittedScreen() {
  const insets = useSafeAreaInsets();
  const { colors, isDark } = useTheme();
  const { id, title } = useLocalSearchParams<{ id?: string; title?: string }>();

  const inscriptionId = Array.isArray(id) ? id[0] : id ?? 'ABJ-00000000';
  const inscriptionTitle = Array.isArray(title) ? title[0] : title ?? 'نقش موثّق';

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
        <View style={[styles.badge, { backgroundColor: 'rgba(193, 138, 59, 0.14)' }]}>
          <SymbolView
            name={{ ios: 'checkmark.seal.fill', android: 'verified', web: 'verified' }}
            size={36}
            tintColor={Palette.ochreClay}
          />
        </View>

        <ArabicText weight="bold" style={[styles.title, { color: colors.text }]}>
          تم إرسال التوثيق
        </ArabicText>

        <ArabicText style={[styles.subtitle, { color: colors.textSecondary }]}>
          {inscriptionTitle}
        </ArabicText>

        <View
          style={[
            styles.card,
            {
              backgroundColor: colors.backgroundElement,
              borderColor: colors.border,
            },
          ]}
        >
          <View style={styles.row}>
            <View style={[styles.statusPill, { backgroundColor: 'rgba(193, 138, 59, 0.16)' }]}>
              <ArabicText weight="semiBold" style={styles.statusText}>
                قيد المراجعة
              </ArabicText>
            </View>
            <ArabicText style={[styles.label, { color: colors.textSecondary }]}>
              الحالة
            </ArabicText>
          </View>

          <View style={[styles.divider, { backgroundColor: colors.border }]} />

          <View style={styles.row}>
            <ArabicText weight="semiBold" style={[styles.idValue, { color: colors.text }]}>
              {inscriptionId}
            </ArabicText>
            <ArabicText style={[styles.label, { color: colors.textSecondary }]}>
              معرّف النقش
            </ArabicText>
          </View>
        </View>

        <ArabicText style={[styles.hint, { color: colors.textSecondary }]}>
          سيقوم الفريق البحثي بمراجعة التوثيق قبل اعتماده في الأرشيف.
        </ArabicText>
      </View>

      <View style={styles.actions}>
        <Pressable
          onPress={() => router.replace('/home')}
          style={({ pressed }) => [styles.primaryBtn, pressed && styles.pressed]}
        >
          <ArabicText weight="bold" style={styles.primaryLabel}>
            العودة للرئيسية
          </ArabicText>
        </Pressable>

        <Pressable
          onPress={() => router.replace('/history')}
          style={({ pressed }) => [
            styles.secondaryBtn,
            {
              backgroundColor: colors.backgroundElement,
              borderColor: colors.border,
            },
            pressed && styles.pressed,
          ]}
        >
          <ArabicText weight="semiBold" style={[styles.secondaryLabel, { color: colors.text }]}>
            عرض السجل
          </ArabicText>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    paddingHorizontal: Spacing.four,
    justifyContent: 'space-between',
  },

  content: {
    alignItems: 'center',
    gap: Spacing.three,
    paddingTop: Spacing.six,
  },

  badge: {
    width: 84,
    height: 84,
    borderRadius: 42,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.two,
  },

  title: {
    fontSize: 24,
    textAlign: 'center',
  },

  subtitle: {
    fontSize: 15,
    textAlign: 'center',
  },

  card: {
    width: '100%',
    borderRadius: 18,
    padding: Spacing.three,
    gap: Spacing.three,
    borderWidth: StyleSheet.hairlineWidth,
    marginTop: Spacing.two,
    ...Platform.select({
      ios: {
        shadowColor: '#3A2414',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.1,
        shadowRadius: 14,
      },
      android: { elevation: 3 },
      default: {},
    }),
  },

  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },

  label: {
    fontSize: 13,
  },

  statusPill: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
  },

  statusText: {
    color: Palette.ochreClay,
    fontSize: 13,
  },

  idValue: {
    fontSize: 16,
    letterSpacing: 0.5,
  },

  divider: {
    height: StyleSheet.hairlineWidth,
  },

  hint: {
    fontSize: 13,
    lineHeight: 20,
    textAlign: 'center',
    maxWidth: 280,
  },

  actions: {
    gap: Spacing.two,
  },

  primaryBtn: {
    alignItems: 'center',
    paddingVertical: 15,
    borderRadius: 999,
    backgroundColor: Palette.deepSandBrown,
  },

  primaryLabel: {
    color: Palette.duneBeige,
    fontSize: 16,
  },

  secondaryBtn: {
    alignItems: 'center',
    paddingVertical: 14,
    borderRadius: 999,
    borderWidth: StyleSheet.hairlineWidth,
  },

  secondaryLabel: {
    fontSize: 15,
  },

  pressed: {
    opacity: 0.88,
  },
});
