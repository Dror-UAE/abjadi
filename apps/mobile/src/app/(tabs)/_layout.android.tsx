import { Tabs } from 'expo-router';
import { SymbolView } from 'expo-symbols';
import {
  Pressable,
  StyleSheet,
  Text,
  View,
  type GestureResponderEvent,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Fonts, Palette, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

type RouteMeta = {
  label: string;
  ios: string;
  iosSelected: string;
  android: string;
};

const TAB_META: Record<string, RouteMeta> = {
  home: {
    label: 'الرئيسية',
    ios: 'house',
    iosSelected: 'house.fill',
    android: 'home',
  },
  history: {
    label: 'السجل',
    ios: 'clock.arrow.circlepath',
    iosSelected: 'clock.arrow.circlepath',
    android: 'history',
  },
};

function AndroidTabBar({
  state,
  descriptors,
  navigation,
}: {
  state: { index: number; routes: { key: string; name: string }[] };
  descriptors: Record<
    string,
    { options: { title?: string; tabBarAccessibilityLabel?: string } }
  >;
  navigation: {
    emit: (event: {
      type: 'tabPress' | 'tabLongPress';
      target: string;
      canPreventDefault?: boolean;
    }) => { defaultPrevented: boolean };
    navigate: (name: string) => void;
  };
}) {
  const insets = useSafeAreaInsets();

  return (
    <View
      pointerEvents="box-none"
      style={[styles.wrap, { paddingBottom: Math.max(insets.bottom, Spacing.two) }]}
    >
      <View
        style={[
          styles.bar,
          {
            backgroundColor: 'rgba(255, 252, 247, 0.96)',
            borderColor: 'rgba(89, 58, 35, 0.1)',
            shadowColor: Palette.deepSandBrown,
          },
        ]}
      >
        {state.routes.map((route, index) => {
          const focused = state.index === index;
          const meta: RouteMeta = TAB_META[route.name] ?? {
            label: descriptors[route.key]?.options.title ?? route.name,
            ios: 'circle',
            iosSelected: 'circle.fill',
            android: 'circle',
          };
          const color = focused ? Palette.deepSandBrown : 'rgba(89, 58, 35, 0.42)';

          const onPress = () => {
            const event = navigation.emit({
              type: 'tabPress',
              target: route.key,
              canPreventDefault: true,
            });

            if (!focused && !event.defaultPrevented) {
              navigation.navigate(route.name);
            }
          };

          const onLongPress = (_event: GestureResponderEvent) => {
            navigation.emit({
              type: 'tabLongPress',
              target: route.key,
            });
          };

          return (
            <Pressable
              key={route.key}
              accessibilityRole="button"
              accessibilityState={focused ? { selected: true } : {}}
              accessibilityLabel={
                descriptors[route.key]?.options.tabBarAccessibilityLabel ?? meta.label
              }
              onPress={onPress}
              onLongPress={onLongPress}
              android_ripple={{ color: 'rgba(89, 58, 35, 0.08)', borderless: false }}
              style={({ pressed }) => [
                styles.item,
                focused && styles.itemActive,
                pressed && styles.itemPressed,
              ]}
            >
              <SymbolView
                name={{
                  ios: focused ? (meta.iosSelected as never) : (meta.ios as never),
                  android: meta.android as never,
                  web: meta.android as never,
                }}
                size={22}
                tintColor={color}
              />
              <Text
                style={[
                  styles.label,
                  {
                    color,
                    fontFamily: focused
                      ? Fonts.arabic.semiBold
                      : Fonts.arabic.medium,
                  },
                ]}
              >
                {meta.label}
              </Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

/**
 * Android: floating capsule tab bar (avoids Material 3 native tabs).
 */
export default function AndroidTabsLayout() {
  const { colors } = useTheme();

  return (
    <Tabs
      tabBar={(props) => (
        <AndroidTabBar
          state={props.state}
          descriptors={props.descriptors as never}
          navigation={props.navigation as never}
        />
      )}
      screenOptions={{
        headerShown: false,
        sceneStyle: { backgroundColor: colors.background },
      }}
    >
      <Tabs.Screen name="home" options={{ title: 'الرئيسية' }} />
      <Tabs.Screen name="history" options={{ title: 'السجل' }} />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  wrap: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    paddingHorizontal: Spacing.four,
    backgroundColor: 'transparent',
  },
  bar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.one,
    paddingVertical: 6,
    paddingHorizontal: 6,
    borderRadius: 999,
    borderWidth: StyleSheet.hairlineWidth,
    elevation: 10,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.14,
    shadowRadius: 20,
  },
  item: {
    minWidth: 108,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    paddingHorizontal: Spacing.three,
    borderRadius: 999,
    gap: 3,
  },
  itemActive: {
    backgroundColor: 'rgba(214, 182, 147, 0.38)',
  },
  itemPressed: {
    opacity: 0.85,
  },
  label: {
    fontSize: 12,
    textAlign: 'center',
    writingDirection: 'rtl',
  },
});
