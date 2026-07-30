import {
  ActivityIndicator,
  FlatList,
  Image,
  Platform,
  Pressable,
  StyleSheet,
  View,
} from 'react-native';
import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { ArabicText } from '@/components/arabic-text';
import { CaptureHero } from '@/components/capture-hero';
import { BottomTabInset, Palette, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { fetchScanHistory } from '@/lib/api';
import { buildHistoryItems, openHistoryItem, syncStatusLabel, type HistoryListItem } from '@/lib/history';
import type { ScanSummary } from '@/lib/ocr-types';
import { listScans, loadScans, mergeRemoteDocumentationTitles } from '@/lib/scan-store';

const RECENT_LIMIT = 5;

type RecordItem = HistoryListItem & {
  location: string;
  status: 'synced' | 'pending';
};

function toRecordItem(item: HistoryListItem): RecordItem {
  return {
    ...item,
    location: syncStatusLabel(item),
    status: item.serverScanId ? 'synced' : 'pending',
  };
}

export function HomeScreen() {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const [records, setRecords] = useState<RecordItem[]>([]);
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

      setRecords(buildHistoryItems(listScans(), remote).slice(0, RECENT_LIMIT).map(toRecordItem));
    } finally {
      setLoading(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void reload();
    }, [reload])
  );

  const hasRecords = records.length > 0;

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
        <CaptureHero onPress={() => router.push('/capture')} />

        {loading ? (
          <View style={styles.loading}>
            <ActivityIndicator color={Palette.ochreClay} />
          </View>
        ) : hasRecords ? (
          <View style={styles.recordsContainer}>
            <View style={styles.sectionHeader}>
              <ArabicText
                weight="bold"
                style={[styles.sectionTitle, { color: colors.text }]}
              >
                آخر الاكتشافات
              </ArabicText>

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
              renderItem={({ item }) => (
                <RecordCard
                  item={item}
                  colors={colors}
                  onPress={() => openHistoryItem(item)}
                />
              )}
            />
          </View>
        ) : (
          <EmptyState />
        )}
      </View>
    </View>
  );
}

function EmptyState() {
  return (
    <View style={styles.emptyContainer}>
      <ArabicText weight="bold" style={styles.emptyTitle}>
        كيف تستخدم أبجدي؟
      </ArabicText>

      <Step text="وجّه الكاميرا نحو النص أو الكتابة التي تريد اكتشافها." />
      <Step text="تأكد من وضوح الصورة ووجود إضاءة مناسبة." />
      <Step text="اجعل الكتابة كاملة داخل إطار الصورة." />
      <Step text="سيقوم التطبيق بتحليل الحروف والتعرف عليها." />

      <View style={styles.divider} />

      <ArabicText weight="bold" style={styles.emptyTitle}>
        ماذا يقدم لك أبجدي؟
      </ArabicText>

      <ArabicText style={styles.infoText}>
        • التعرف على الحروف والكتابات القديمة.
      </ArabicText>

      <ArabicText style={styles.infoText}>
        • تحويل الصور إلى نصوص قابلة للقراءة.
      </ArabicText>

      <ArabicText style={styles.infoText}>
        • تحليل الخطوط والرموز المكتوبة.
      </ArabicText>

      <ArabicText style={styles.footerText}>
        ابدأ الآن باكتشاف أي كتابة ✨
      </ArabicText>
    </View>
  );
}

function Step({ text }: { text: string }) {
  return (
    <View style={styles.step}>
      <View style={styles.number}>
        <ArabicText>✓</ArabicText>
      </View>

      <ArabicText style={styles.infoText}>
        {text}
      </ArabicText>
    </View>
  );
}

function RecordCard({
  item,
  colors,
  onPress,
}: {
  item: RecordItem;
  colors: { text: string; textSecondary: string; backgroundElement: string };
  onPress: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.recordCard,
        {
          backgroundColor: colors.backgroundElement,
        },
        pressed && styles.pressed,
      ]}
    >
      <View style={styles.recordThumb}>
        {item.imageUri ? (
          <Image source={{ uri: item.imageUri }} style={styles.recordThumbImage} resizeMode="cover" />
        ) : (
          <View style={[styles.recordThumbImage, styles.thumbPlaceholder]} />
        )}
      </View>

      <View style={styles.recordBody}>
        <ArabicText
          weight="semiBold"
          style={[styles.recordTitle, { color: colors.text }]}
        >
          {item.title}
        </ArabicText>

        <ArabicText
          style={[styles.recordMeta, { color: colors.textSecondary }]}
        >
          {item.time}
        </ArabicText>

        <ArabicText
          style={[styles.recordMeta, { color: colors.textSecondary }]}
          numberOfLines={1}
        >
          {item.location}
        </ArabicText>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
  },

  content: {
    flex: 1,
    paddingHorizontal: Spacing.three,
    gap: Spacing.three,
  },

  loading: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },

  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    paddingVertical: Spacing.two,
  },

  emptyTitle: {
    color: Palette.deepSandBrown,
    fontSize: 19,
    marginBottom: 14,
    textAlign: 'right',
  },

  step: {
    flexDirection: 'row-reverse',
    alignItems: 'center',
    marginBottom: 12,
    gap: 10,
  },

  number: {
    width: 28,
    height: 28,
    borderRadius: 14,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(214, 182, 147, 0.45)',
  },

  infoText: {
    flex: 1,
    textAlign: 'right',
    color: Palette.deepSandBrown,
    fontSize: 15,
    lineHeight: 25,
  },

  divider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: Palette.sandstone,
    opacity: 0.55,
    marginVertical: 18,
  },

  footerText: {
    marginTop: 18,
    textAlign: 'center',
    color: Palette.deepSandBrown,
    opacity: 0.7,
  },

  recordsContainer: {
    flex: 1,
  },

  sectionHeader: {
    flexDirection: 'row-reverse',
    justifyContent: 'space-between',
    marginBottom: 12,
  },

  sectionTitle: {
    fontSize: 18,
  },

  viewAll: {
    color: Palette.desertSage,
  },

  list: {
    gap: Spacing.two,
  },

  recordCard: {
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: Palette.sandstone,
    flexDirection: 'row-reverse',
    alignItems: 'center',
    padding: Spacing.two,
    borderRadius: 18,
    gap: Spacing.two,
  },

  recordThumb: {
    width: 64,
    height: 64,
    borderRadius: 14,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: Palette.sandstone,
    overflow: 'hidden',
  },

  recordThumbImage: {
    width: '100%',
    height: '100%',
  },

  thumbPlaceholder: {
    backgroundColor: Palette.sandstone,
  },

  recordBody: {
    flex: 1,
    alignItems: 'flex-end',
  },

  recordTitle: {
    fontSize: 15,
  },

  recordMeta: {
    fontSize: 12,
    marginTop: 3,
    textAlign: 'right',
  },

  pressed: {
    opacity: 0.88,
  },
});
