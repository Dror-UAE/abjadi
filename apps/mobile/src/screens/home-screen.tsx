import * as ImagePicker from 'expo-image-picker';
import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Image,
  Platform,
  Pressable,
  StyleSheet,
  View,
} from 'react-native';
import Animated, {
  FadeInDown,
} from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { ArabicText } from '@/components/arabic-text';
import { CaptureHero } from '@/components/capture-hero';
import { BottomTabInset, Palette, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { fetchScanHistory } from '@/lib/api';
import {
  buildHistoryItems,
  openHistoryItem,
  syncStatusLabel,
  type HistoryListItem,
} from '@/lib/history';
import type { ScanSummary } from '@/lib/ocr-types';
import { listScans, loadScans, mergeRemoteDocumentationTitles } from '@/lib/scan-store';

const RECENT_LIMIT = 5;

export function HomeScreen() {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const [records, setRecords] = useState<HistoryListItem[]>([]);
  const [loading, setLoading] = useState(true);

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      await loadScans();

      let remote: ScanSummary[] = [];
      try {
        const response = await fetchScanHistory(RECENT_LIMIT);
        if (response.ok) {
          remote = response.scans;
          await mergeRemoteDocumentationTitles(remote);
        }
      } catch {
        // Show local scans when API is unreachable.
      }

      setRecords(buildHistoryItems(listScans(), remote).slice(0, RECENT_LIMIT));
    } finally {
      setLoading(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void reload();
    }, [reload])
  );

  const openLibrary = useCallback(async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 1,
      allowsEditing: false,
    });

    if (!result.canceled && result.assets[0]?.uri) {
      router.push({
        pathname: '/analyzing',
        params: { uri: encodeURIComponent(result.assets[0].uri) },
      });
    }
  }, []);

  const hasRecords = records.length > 0;
  const archiveCountLabel =
    records.length === 1
      ? 'فحص واحد محفوظ'
      : records.length === 2
        ? 'فحصان محفوظان'
        : `${records.length} فحوصات محفوظة`;

  return (
    <View
      style={[
        styles.screen,
        {
          backgroundColor: colors.background,
          paddingTop: insets.top + Spacing.three,
          paddingBottom:
            insets.bottom +
            Spacing.three +
            (Platform.OS === 'android' ? BottomTabInset : 0),
        },
      ]}
    >
      <View style={styles.content}>
        <CaptureHero
          onCapture={() => router.push('/capture')}
          onLibrary={() => void openLibrary()}
        />

        {loading ? (
          <View style={styles.loading}>
            <ActivityIndicator color={Palette.ochreClay} />
          </View>
        ) : hasRecords ? (
          <View style={styles.recordsContainer}>
            <View style={styles.sectionHeader}>
              <View style={styles.sectionTitles}>
                <ArabicText
                  weight="bold"
                  style={[styles.sectionTitle, { color: colors.text }]}
                >
                  آخر الفحوصات
                </ArabicText>
                <ArabicText
                  style={[styles.sectionMeta, { color: colors.textSecondary }]}
                >
                  {archiveCountLabel}
                </ArabicText>
              </View>

              <Pressable hitSlop={8} onPress={() => router.push('/history')}>
                <ArabicText weight="medium" style={styles.viewAll}>
                  عرض الكل
                </ArabicText>
              </Pressable>
            </View>

            <FlatList
              data={records}
              keyExtractor={(item) => item.key}
              showsVerticalScrollIndicator={false}
              contentContainerStyle={styles.list}
              renderItem={({ item, index }) => (
                <RecordCard
                  item={item}
                  index={index}
                  colors={colors}
                  onPress={() => openHistoryItem(item)}
                />
              )}
            />
          </View>
        ) : (
          <EmptyState colors={colors} />
        )}
      </View>
    </View>
  );
}

function EmptyState({
  colors,
}: {
  colors: { text: string; textSecondary: string };
}) {
  return (
    <View style={styles.emptyContainer}>
      <ArabicText weight="bold" style={[styles.emptyTitle, { color: colors.text }]}>
        أرشيفك بانتظار أول نقش
      </ArabicText>
      <ArabicText style={[styles.emptyCopy, { color: colors.textSecondary }]}>
        صوّر كتابة أو اختر صورة من مكتبتك لتظهر نتائج التحليل هنا.
      </ArabicText>
    </View>
  );
}

function RecordCard({
  item,
  index,
  colors,
  onPress,
}: {
  item: HistoryListItem;
  index: number;
  colors: { text: string; textSecondary: string; backgroundElement: string };
  onPress: () => void;
}) {
  const status = syncStatusLabel(item);
  const preview =
    item.preview && item.preview !== '—' ? item.preview.trim() : null;

  return (
    <Animated.View entering={FadeInDown.delay(index * 70).duration(420)}>
      <Pressable
        onPress={onPress}
        style={({ pressed }) => [
          styles.recordCard,
          { backgroundColor: colors.backgroundElement },
          pressed && styles.pressed,
        ]}
      >
        <View style={styles.recordThumb}>
          {item.imageUri ? (
            <Image
              source={{ uri: item.imageUri }}
              style={styles.recordThumbImage}
              resizeMode="cover"
            />
          ) : (
            <View style={[styles.recordThumbImage, styles.thumbPlaceholder]} />
          )}
          <View style={styles.thumbScrim} />
          {preview ? (
            <ArabicText numberOfLines={1} style={styles.thumbGlyph}>
              {preview}
            </ArabicText>
          ) : null}
        </View>

        <View style={styles.recordBody}>
          <ArabicText
            weight="semiBold"
            style={[styles.recordTitle, { color: colors.text }]}
            numberOfLines={1}
          >
            {item.title}
          </ArabicText>

          <ArabicText
            style={[styles.recordMeta, { color: colors.textSecondary }]}
          >
            {item.time}
          </ArabicText>

          <View style={styles.statusChip}>
            <ArabicText weight="medium" style={styles.statusChipText}>
              {status}
            </ArabicText>
          </View>
        </View>
      </Pressable>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
  },

  content: {
    flex: 1,
    paddingHorizontal: Spacing.three,
    gap: Spacing.four,
  },

  loading: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },

  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: Spacing.two,
    gap: Spacing.two,
  },

  emptyTitle: {
    fontSize: 20,
    textAlign: 'right',
  },

  emptyCopy: {
    fontSize: 15,
    lineHeight: 24,
    textAlign: 'right',
  },

  recordsContainer: {
    flex: 1,
  },

  sectionHeader: {
    flexDirection: 'row-reverse',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: Spacing.three,
  },

  sectionTitles: {
    alignItems: 'flex-end',
    gap: 2,
  },

  sectionTitle: {
    fontSize: 18,
  },

  sectionMeta: {
    fontSize: 12,
  },

  viewAll: {
    color: Palette.desertSage,
    marginTop: 2,
  },

  list: {
    gap: Spacing.two,
    paddingBottom: Spacing.two,
  },

  recordCard: {
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: Palette.sandstone,
    flexDirection: 'row-reverse',
    alignItems: 'stretch',
    padding: Spacing.two,
    borderRadius: 20,
    gap: Spacing.three,
    minHeight: 104,
  },

  recordThumb: {
    width: 92,
    height: 92,
    borderRadius: 16,
    backgroundColor: Palette.sandstone,
    overflow: 'hidden',
    justifyContent: 'flex-end',
  },

  recordThumbImage: {
    ...StyleSheet.absoluteFill,
    width: '100%',
    height: '100%',
  },

  thumbPlaceholder: {
    backgroundColor: Palette.sandstone,
  },

  thumbScrim: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(40, 24, 12, 0.22)',
  },

  thumbGlyph: {
    position: 'relative',
    zIndex: 1,
    color: Palette.duneBeige,
    fontSize: 11,
    lineHeight: 16,
    textAlign: 'center',
    paddingHorizontal: 6,
    paddingBottom: 8,
    textShadowColor: 'rgba(0,0,0,0.45)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 3,
  },

  recordBody: {
    flex: 1,
    alignItems: 'flex-end',
    justifyContent: 'center',
    gap: 4,
    paddingVertical: 2,
  },

  recordTitle: {
    fontSize: 16,
    textAlign: 'right',
  },

  recordMeta: {
    fontSize: 12,
    textAlign: 'right',
  },

  statusChip: {
    marginTop: 4,
    alignSelf: 'flex-end',
    backgroundColor: 'rgba(193, 138, 59, 0.14)',
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },

  statusChipText: {
    color: Palette.ochreClay,
    fontSize: 11,
  },

  pressed: {
    opacity: 0.9,
    transform: [{ scale: 0.99 }],
  },
});
