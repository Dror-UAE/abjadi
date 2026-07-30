import { Text, type TextProps } from 'react-native';

import { Fonts } from '@/constants/theme';

type ArabicWeight = keyof typeof Fonts.arabic;

type ArabicTextProps = TextProps & {
  weight?: ArabicWeight;
};

/**
 * Text component for Arabic copy — always uses Noto Naskh Arabic.
 * Prefer this over `Text` for any Arabic string.
 */
export function ArabicText({
  weight = 'regular',
  style,
  ...props
}: ArabicTextProps) {
  return (
    <Text
      {...props}
      style={[{ fontFamily: Fonts.arabic[weight], writingDirection: 'rtl' }, style]}
    />
  );
}
