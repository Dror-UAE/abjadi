/**
 * Learn more about light and dark modes:
 * https://docs.expo.dev/develop/user-interface/color-themes/
 */

import { Colors, FlowChrome, Fonts, Spacing, type ColorScheme } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';

export function useTheme() {
  const colorScheme: ColorScheme = useColorScheme();
  const colors = Colors[colorScheme];
  const flow = FlowChrome[colorScheme];

  return {
    colors,
    flow,
    colorScheme,
    isDark: colorScheme === 'dark',
    fonts: Fonts,
    spacing: Spacing,
  };
}
