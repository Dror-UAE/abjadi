import {
  NotoNaskhArabic_400Regular,
  NotoNaskhArabic_500Medium,
  NotoNaskhArabic_600SemiBold,
  NotoNaskhArabic_700Bold,
  useFonts,
} from '@expo-google-fonts/noto-naskh-arabic';
import { NotoSansOldSouthArabian_400Regular } from '@expo-google-fonts/noto-sans-old-south-arabian';

export const arabicFontMap = {
  NotoNaskhArabic_400Regular,
  NotoNaskhArabic_500Medium,
  NotoNaskhArabic_600SemiBold,
  NotoNaskhArabic_700Bold,
  NotoSansOldSouthArabian_400Regular,
};

export function useAppFonts() {
  return useFonts(arabicFontMap);
}
