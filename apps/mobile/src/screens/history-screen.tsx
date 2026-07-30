import {
  ActivityIndicator,
  FlatList,
  Image,
  Platform,
  Pressable,
  RefreshControl,
  StyleSheet,
  View,
} from 'react-native';
import { useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { ArabicText } from '@/components/arabic-text';
import { BottomTabInset, Palette, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { fetchScanHistory } from '@/lib/api';
import { buildHistoryItems, openHistoryItem, syncStatusLabel, type HistoryListItem } from '@/lib/history';
import type { ScanSummary } from '@/lib/ocr-types';
import { listScans, loadScans, mergeRemoteDocumentationTitles } from '@/lib/scan-store';

export function HistoryScreen() {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const [items, setItems] = useState<HistoryListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);

    try {
      await loadScans();

      let remote: ScanSummary[] = [];
      try {
        const response = await fetchScanHistory(50);
        if (response.ok) {
          remote = response.scans;
          await mergeRemoteDocumentationTitles(remote);
        }
      } catch {
        // Offline or API down — still show local scans.
      }

      setItems(buildHistoryItems(listScans(), remote));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'تعذر تحميل السجل');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void reload();
    }, [reload])
  );

  const openItem = (item: HistoryListItem) => {
    openHistoryItem(item);
  };

  return (
    <View
      style={[
        styles.screen,
        {
          backgroundColor: colors.background,
          paddingTop: insets.top + Spacing.three,
          paddingBottom:
            insets.bottom +
            Spacing.two +
            (Platform.OS === 'android' ? BottomTabInset : 0),
        },
      ]}
    >
      <View style={styles.header}>
        <ArabicText weight="bold" style={[styles.title, { color: colors.text }]}>
          سجل المعاينات
        </ArabicText>
        <ArabicText style={[styles.subtitle, { color: colors.textSecondary }]}>
          المعاينات والتحليلات السابقة
        </ArabicText>
      </View>

      {loading ? (
        <View style={styles.center}>
          <ActivityIndicator color={Palette.ochreClay} />
        </View>
      ) : (
        <FlatList
          data={items}
          keyExtractor={(item) => item.key}
          showsVerticalScrollIndicator={false}
          contentContainerStyle={styles.list}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={() => void reload(true)}
              tintColor={Palette.ochreClay}
            />
          }
          ListEmptyComponent={
            <View style={styles.empty}>
              <ArabicText weight="semiBold" style={[styles.emptyTitle, { color: colors.text }]}>
                لا توجد معاينات بعد
              </ArabicText>
              <ArabicText style={[styles.emptyCopy, { color: colors.textSecondary }]}>
                {error || 'صوّر نصاً ليظهر هنا في السجل'}
              </ArabicText>
            </View>
          }
          renderItem={({ item }) => (
            <Pressable
              onPress={() => openItem(item)}
              style={({ pressed }) => [
                styles.card,
                { backgroundColor: colors.backgroundElement },
                pressed && styles.pressed,
              ]}
            >
              <View style={styles.thumb}>
                {item.imageUri ? (
                  <Image source={{ uri: item.imageUri }} style={styles.thumbImage} resizeMode="cover" />
                ) : (
                  <View style={[styles.thumbImage, styles.thumbPlaceholder]} />
                )}
              </View>

              <View style={styles.body}>
                <ArabicText weight="semiBold" style={[styles.cardTitle, { color: colors.text }]}>
                  {item.title}
                </ArabicText>
                <ArabicText style={[styles.meta, { color: colors.textSecondary }]}>
                  {item.time}
                </ArabicText>
                <ArabicText style={[styles.meaning, { color: Palette.ochreClay }]} numberOfLines={2}>
                  {syncStatusLabel(item)}
                </ArabicText>
              </View>
            </Pressable>
          )}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    paddingHorizontal: Spacing.three,
  },

  header: {
    marginBottom: Spacing.three,
    gap: 4,
  },

  title: {
    fontSize: 26,
    textAlign: 'right',
  },

  subtitle: {
    fontSize: 13,
    textAlign: 'right',
  },

  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },

  list: {
    gap: Spacing.two,
    paddingBottom: Spacing.three,
    flexGrow: 1,
  },

  card: {
    flexDirection: 'row-reverse',
    alignItems: 'center',
    padding: Spacing.two,
    borderRadius: 16,
    gap: Spacing.two,
  },

  thumb: {
    width: 64,
    height: 64,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: Palette.sandstone,
  },

  thumbImage: {
    width: '100%',
    height: '100%',
  },

  thumbPlaceholder: {
    backgroundColor: Palette.sandstone,
  },

  body: {
    flex: 1,
    alignItems: 'flex-end',
    gap: 2,
  },

  cardTitle: {
    fontSize: 15,
  },

  meta: {
    fontSize: 12,
  },

  meaning: {
    fontSize: 12,
    marginTop: 2,
    textAlign: 'right',
  },

  empty: {
    paddingTop: Spacing.six,
    alignItems: 'center',
    gap: Spacing.two,
  },

  emptyTitle: {
    fontSize: 17,
    textAlign: 'center',
  },

  emptyCopy: {
    fontSize: 13,
    textAlign: 'center',
  },

  pressed: {
    opacity: 0.88,
  },
});
