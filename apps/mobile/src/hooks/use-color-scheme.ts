import { useColorScheme as useRNColorScheme } from 'react-native';

import type { ColorScheme } from '@/constants/theme';

/**
 * Returns the active color scheme, falling back to light when unset.
 */
export function useColorScheme(): ColorScheme {
  const scheme = useRNColorScheme();
  return scheme === 'dark' ? 'dark' : 'light';
}
