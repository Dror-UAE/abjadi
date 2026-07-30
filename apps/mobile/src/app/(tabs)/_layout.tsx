import { NativeTabs } from 'expo-router/unstable-native-tabs';
import { DynamicColorIOS } from 'react-native';

import { Colors } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

/**
 * iOS: system native tab bar (liquid glass where available).
 */
export default function TabsLayout() {
  const { colors } = useTheme();

  return (
    <NativeTabs
      tintColor={DynamicColorIOS({
        light: Colors.light.tint,
        dark: Colors.dark.tint,
      })}
      labelStyle={{
        color: DynamicColorIOS({
          light: Colors.light.textSecondary,
          dark: Colors.dark.textSecondary,
        }),
      }}
      minimizeBehavior="onScrollDown"
    >
      <NativeTabs.Trigger name="home" contentStyle={{ backgroundColor: colors.background }}>
        <NativeTabs.Trigger.Label>Home</NativeTabs.Trigger.Label>
        <NativeTabs.Trigger.Icon sf={{ default: 'house', selected: 'house.fill' }} md="home" />
      </NativeTabs.Trigger>
      <NativeTabs.Trigger name="history" contentStyle={{ backgroundColor: colors.background }}>
        <NativeTabs.Trigger.Icon
          sf={{ default: 'clock.arrow.circlepath', selected: 'clock.arrow.circlepath' }}
          md="history"
        />
        <NativeTabs.Trigger.Label>History</NativeTabs.Trigger.Label>
      </NativeTabs.Trigger>
    </NativeTabs>
  );
}
