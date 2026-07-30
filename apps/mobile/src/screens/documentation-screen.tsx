import * as ImagePicker from 'expo-image-picker';
import { router, useLocalSearchParams } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SymbolView } from 'expo-symbols';
import { useMemo, useState, type ReactNode } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
  type ImageSourcePropType,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { ArabicText } from '@/components/arabic-text';
import { Fonts, Palette, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import { ApiError, submitDocumentation, uriToBase64Image } from '@/lib/api';
import { getScan, setScanDocumentationTitle } from '@/lib/scan-store';
import { translateMusnadLines } from '@/lib/musnad-translate';

const FALLBACK_IMAGE = require('../../assets/images/scanned-image.jpg');
const DEFAULT_OCR = '𐩱𐩡 𐩭𐩸𐩣 𐩫𐩡';

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

function SectionCard({
  title,
  icon,
  children,
}: {
  title: string;
  icon: { ios: string; android: string };
  children: ReactNode;
}) {
  const { colors } = useTheme();

  return (
    <View
      style={[
        styles.card,
        {
          backgroundColor: colors.backgroundElement,
          borderColor: colors.border,
        },
      ]}
    >
      <View style={styles.cardHeader}>
        <ArabicText weight="bold" style={[styles.cardTitle, { color: colors.text }]}>
          {title}
        </ArabicText>
        <View style={styles.cardIcon}>
          <SymbolView
            name={{ ios: icon.ios as never, android: icon.android as never, web: icon.android as never }}
            size={16}
            tintColor={Palette.ochreClay}
          />
        </View>
      </View>
      {children}
    </View>
  );
}

function FieldLabel({ children }: { children: string }) {
  const { colors } = useTheme();
  return (
    <ArabicText weight="medium" style={[styles.fieldLabel, { color: colors.textSecondary }]}>
      {children}
    </ArabicText>
  );
}

function TextField({
  value,
  onChangeText,
  placeholder,
  multiline,
  style,
}: {
  value: string;
  onChangeText: (text: string) => void;
  placeholder: string;
  multiline?: boolean;
  style?: object;
}) {
  const { colors } = useTheme();

  return (
    <TextInput
      value={value}
      onChangeText={onChangeText}
      placeholder={placeholder}
      placeholderTextColor={colors.textSecondary}
      multiline={multiline}
      textAlign="right"
      textAlignVertical={multiline ? 'top' : 'center'}
      style={[
        styles.input,
        multiline && styles.inputMultiline,
        {
          color: colors.text,
          backgroundColor: colors.background,
          borderColor: colors.border,
          fontFamily: Fonts.arabic.regular,
        },
        style,
      ]}
    />
  );
}

export function DocumentationScreen() {
  const insets = useSafeAreaInsets();
  const { colors, isDark } = useTheme();
  const { uri, confidence: confidenceParam, scanId: scanIdParam } = useLocalSearchParams<{
    uri?: string;
    confidence?: string;
    scanId?: string;
  }>();
  const primaryImage = useMemo(() => resolveImageSource(uri), [uri]);
  const localScanId = Array.isArray(scanIdParam) ? scanIdParam[0] : scanIdParam;
  const scan = useMemo(() => getScan(localScanId), [localScanId]);
  const arabicFromScan = useMemo(() => {
    if (!scan) return '';
    if (scan.result.arabicText?.trim()) return scan.result.arabicText.trim();
    if (scan.result.lines?.length) return translateMusnadLines(scan.result.lines);
    return '';
  }, [scan]);

  const confidence = useMemo(() => {
    const value = Array.isArray(confidenceParam) ? confidenceParam[0] : confidenceParam;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 96;
  }, [confidenceParam]);

  const [title, setTitle] = useState('');
  const [scriptType, setScriptType] = useState('');
  const [language, setLanguage] = useState('');
  const [region, setRegion] = useState('');
  const [country, setCountry] = useState('');
  const [description, setDescription] = useState('');
  const [condition, setCondition] = useState('');
  const [source, setSource] = useState('');
  const [ocrText, setOcrText] = useState(
    () => scan?.result.text?.trim() || DEFAULT_OCR
  );
  const [notes, setNotes] = useState('');
  const [extraImages, setExtraImages] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);

  const close = () => {
    if (router.canGoBack()) router.back();
    else router.replace('/home');
  };

  const addImages = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsMultipleSelection: true,
      quality: 0.9,
    });
    if (!result.canceled) {
      setExtraImages((prev) => [...prev, ...result.assets.map((a) => a.uri)]);
    }
  };

  const submit = async () => {
    const serverScanId = scan?.serverScanId || scan?.result.scanId;
    if (!serverScanId) {
      Alert.alert(
        'تعذر الإرسال',
        'لا يوجد سجل تحليل محفوظ على الخادم. أعد التحليل بعد تفعيل Supabase على الـ API.'
      );
      return;
    }

    setSubmitting(true);
    try {
      const extras = await Promise.all(
        extraImages.map(async (imgUri) => {
          const part = await uriToBase64Image(imgUri);
          return {
            base64: part.base64,
            filename: part.filename,
            mimeType: part.mimeType,
          };
        })
      );

      const saved = await submitDocumentation({
        scanId: serverScanId,
        title: title.trim() || 'نقش بدون عنوان',
        scriptType,
        language,
        region,
        country,
        description,
        condition,
        imageSource: source,
        ocrTextEdited: ocrText,
        notes,
        confidence,
        extraImages: extras,
      });

      if (!saved.ok) {
        throw new ApiError(saved.detail || saved.error);
      }

      const docTitle = title.trim() || 'نقش بدون عنوان';
      await setScanDocumentationTitle(localScanId, docTitle);

      router.replace({
        pathname: '/documentation-submitted',
        params: {
          id: saved.publicId,
          title: docTitle,
        },
      });
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : 'فشل حفظ التوثيق';
      Alert.alert('تعذر الإرسال', message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <View style={[styles.root, { backgroundColor: colors.background }]}>
      <StatusBar style={isDark ? 'light' : 'dark'} />

      <View style={[styles.topBar, { paddingTop: insets.top + Spacing.two }]}>
        <Pressable
          onPress={close}
          style={[styles.iconBtn, { backgroundColor: colors.backgroundElement }]}
        >
          <SymbolView
            name={{ ios: 'chevron.right', android: 'arrow_forward', web: 'arrow_forward' }}
            size={16}
            tintColor={colors.text}
          />
        </Pressable>
        <ArabicText weight="bold" style={[styles.topTitle, { color: colors.text }]}>
          توثيق النقش
        </ArabicText>
        <View style={styles.iconBtnPlaceholder} />
      </View>

      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <ScrollView
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
          contentContainerStyle={[
            styles.scroll,
            { paddingBottom: insets.bottom + 100 },
          ]}
        >
          <SectionCard
            title="المعلومات الأساسية"
            icon={{ ios: 'doc.text', android: 'description' }}
          >
            <FieldLabel>عنوان النقش</FieldLabel>
            <TextField
              value={title}
              onChangeText={setTitle}
              placeholder="مثال: نقش من مليحة"
            />

            <FieldLabel>نوع الخط</FieldLabel>
            <TextField
              value={scriptType}
              onChangeText={setScriptType}
              placeholder="مسند، آرامي، ثمودية، عربية مبكرة، غير معروف"
            />

            <FieldLabel>اللغة</FieldLabel>
            <TextField
              value={language}
              onChangeText={setLanguage}
              placeholder="عربية قديمة، آرامي، سبئية، أخرى"
            />
          </SectionCard>

          <SectionCard
            title="الموقع"
            icon={{ ios: 'mappin.and.ellipse', android: 'location_on' }}
          >
            <FieldLabel>المنطقة</FieldLabel>
            <TextField
              value={region}
              onChangeText={setRegion}
              placeholder="مثال: الشارقة — مليحة"
            />

            <FieldLabel>الدولة</FieldLabel>
            <TextField
              value={country}
              onChangeText={setCountry}
              placeholder="مثال: الإمارات العربية المتحدة"
            />
          </SectionCard>

          <SectionCard
            title="وصف النقش"
            icon={{ ios: 'text.alignright', android: 'notes' }}
          >
            <FieldLabel>وصف الحجر وحالة النقش</FieldLabel>
            <TextField
              value={description}
              onChangeText={setDescription}
              placeholder="صف المادة، الحجم، واتجاه الكتابة..."
              multiline
            />

            <FieldLabel>الحالة</FieldLabel>
            <TextField
              value={condition}
              onChangeText={setCondition}
              placeholder="واضح، متآكل، تالف، مُجزّأ"
            />
          </SectionCard>

          <SectionCard
            title="توثيق الصور"
            icon={{ ios: 'photo.on.rectangle', android: 'photo_library' }}
          >
            <View style={styles.imagesRow}>
              <Image source={primaryImage} style={styles.previewThumb} resizeMode="cover" />
              {extraImages.map((img) => (
                <Image key={img} source={{ uri: img }} style={styles.previewThumb} resizeMode="cover" />
              ))}
              <Pressable
                onPress={() => void addImages()}
                style={[styles.addImageBtn, { borderColor: colors.border }]}
              >
                <SymbolView
                  name={{ ios: 'plus', android: 'add', web: 'add' }}
                  size={20}
                  tintColor={Palette.ochreClay}
                />
                <ArabicText style={{ color: colors.textSecondary, fontSize: 11 }}>
                  إضافة
                </ArabicText>
              </Pressable>
            </View>

            <FieldLabel>مصدر الصورة</FieldLabel>
            <TextField
              value={source}
              onChangeText={setSource}
              placeholder="صورة شخصية، متحف، كتاب، أرشيف"
            />
          </SectionCard>

          <SectionCard
            title="قراءة التعرف"
            icon={{ ios: 'text.magnifyingglass', android: 'document_scanner' }}
          >
            <View style={styles.confidenceRow}>
              <ArabicText weight="semiBold" style={{ color: Palette.ochreClay, fontSize: 13 }}>
                {confidence}%
              </ArabicText>
              <ArabicText style={{ color: colors.textSecondary, fontSize: 12 }}>
                درجة الثقة
              </ArabicText>
            </View>

            <Text
              style={[
                styles.ocrPreview,
                { color: Palette.sandstone, backgroundColor: colors.background, borderColor: colors.border },
              ]}
            >
              {ocrText}
            </Text>

            {arabicFromScan ? (
              <>
                <FieldLabel>الترجمة العربية</FieldLabel>
                <ArabicText style={[styles.arabicPreview, { color: colors.text }]}>
                  {arabicFromScan}
                </ArabicText>
              </>
            ) : null}

            <FieldLabel>تعديل القراءة</FieldLabel>
            <TextField
              value={ocrText}
              onChangeText={setOcrText}
              placeholder="حرّر النص المكتشف"
              multiline
              style={{ fontFamily: Fonts.script, fontSize: 22, letterSpacing: 3 }}
            />
          </SectionCard>

          <SectionCard
            title="ملاحظات بحثية"
            icon={{ ios: 'pencil.and.outline', android: 'edit_note' }}
          >
            <TextField
              value={notes}
              onChangeText={setNotes}
              placeholder="أضف ملاحظاتك العلمية أو السياقية..."
              multiline
            />
          </SectionCard>
        </ScrollView>
      </KeyboardAvoidingView>

      <View
        style={[
          styles.footer,
          {
            paddingBottom: Math.max(insets.bottom, Spacing.three),
            backgroundColor: colors.background,
            borderTopColor: colors.border,
          },
        ]}
      >
        <Pressable
          onPress={() => void submit()}
          disabled={submitting}
          style={({ pressed }) => [
            styles.submitBtn,
            submitting && styles.submitBtnDisabled,
            pressed && !submitting && styles.pressed,
          ]}
        >
          {submitting ? (
            <ActivityIndicator color={Palette.duneBeige} />
          ) : (
            <ArabicText weight="bold" style={styles.submitLabel}>
              إرسال التوثيق
            </ArabicText>
          )}
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  flex: { flex: 1 },

  topBar: {
    paddingHorizontal: Spacing.three,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: Spacing.two,
  },

  topTitle: { fontSize: 17 },

  iconBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },

  iconBtnPlaceholder: { width: 36, height: 36 },

  scroll: {
    paddingHorizontal: Spacing.three,
    gap: Spacing.three,
  },

  card: {
    borderRadius: 18,
    padding: Spacing.three,
    gap: Spacing.two,
    borderWidth: StyleSheet.hairlineWidth,
    ...Platform.select({
      ios: {
        shadowColor: '#3A2414',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.08,
        shadowRadius: 12,
      },
      android: { elevation: 2 },
      default: {},
    }),
  },

  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: Spacing.one,
  },

  cardTitle: { fontSize: 16 },

  cardIcon: {
    width: 30,
    height: 30,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(193, 138, 59, 0.12)',
  },

  fieldLabel: {
    fontSize: 12,
    textAlign: 'right',
    marginTop: Spacing.one,
  },

  input: {
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: Platform.OS === 'ios' ? 12 : 10,
    fontSize: 15,
  },

  inputMultiline: {
    minHeight: 96,
    paddingTop: 12,
  },

  imagesRow: {
    flexDirection: 'row-reverse',
    flexWrap: 'wrap',
    gap: 10,
  },

  previewThumb: {
    width: 72,
    height: 72,
    borderRadius: 12,
  },

  addImageBtn: {
    width: 72,
    height: 72,
    borderRadius: 12,
    borderWidth: 1,
    borderStyle: 'dashed',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 2,
  },

  confidenceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: 8,
  },

  ocrPreview: {
    fontFamily: Fonts.script,
    fontSize: 26,
    lineHeight: 40,
    letterSpacing: 4,
    textAlign: 'center',
    paddingVertical: 14,
    paddingHorizontal: 12,
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    writingDirection: 'rtl',
  },

  arabicPreview: {
    fontSize: 18,
    lineHeight: 28,
    textAlign: 'right',
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 12,
    backgroundColor: 'rgba(214, 182, 147, 0.18)',
  },

  footer: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    paddingHorizontal: Spacing.three,
    paddingTop: Spacing.two,
    borderTopWidth: StyleSheet.hairlineWidth,
  },

  submitBtn: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 15,
    borderRadius: 999,
    backgroundColor: Palette.deepSandBrown,
  },

  submitBtnDisabled: {
    opacity: 0.7,
  },

  submitLabel: {
    color: Palette.duneBeige,
    fontSize: 16,
  },

  pressed: { opacity: 0.88, transform: [{ scale: 0.985 }] },
});
