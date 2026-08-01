import { useCallback, useEffect, useRef, useState } from 'react';
import { AppState, type AppStateStatus } from 'react-native';
import { DarkTheme, DefaultTheme, Stack, ThemeProvider } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import * as SplashScreen from 'expo-splash-screen';
import * as SystemUI from 'expo-system-ui';

import { useAppFonts } from '@/hooks/use-app-fonts';
import { useTheme } from '@/hooks/use-theme';
import { loadApiConfig } from '@/lib/api-config';
import { loadScans } from '@/lib/scan-store';
import {
  checkAppVersion,
  type VersionCheckResult,
} from '@/lib/version-check';
import { ForceUpdateScreen } from '@/screens/force-update-screen';

// Keep native launch screen up until fonts + first frame are ready.
SplashScreen.preventAutoHideAsync().catch(() => {});

export default function RootLayout() {
  const { colors, flow, isDark } = useTheme();
  const [fontsLoaded, fontError] = useAppFonts();
  const [bootReady, setBootReady] = useState(false);
  const [versionResult, setVersionResult] = useState<VersionCheckResult | null>(
    null
  );
  const checkingRef = useRef(false);

  useEffect(() => {
    void SystemUI.setBackgroundColorAsync(colors.background);
  }, [colors.background]);

  const runVersionCheck = useCallback(async () => {
    if (checkingRef.current) return;
    checkingRef.current = true;
    try {
      const result = await checkAppVersion();
      setVersionResult(result);
    } finally {
      checkingRef.current = false;
    }
  }, []);

  // Boot: fonts → API config → version policy → reveal UI.
  useEffect(() => {
    if (!fontsLoaded && !fontError) return;

    let cancelled = false;

    async function boot() {
      await loadApiConfig();
      void loadScans();
      const result = await checkAppVersion();
      if (cancelled) return;
      setVersionResult(result);
      setBootReady(true);
    }

    void boot();
    return () => {
      cancelled = true;
    };
  }, [fontsLoaded, fontError]);

  useEffect(() => {
    if (bootReady) {
      void SplashScreen.hideAsync();
    }
  }, [bootReady]);

  // Re-check whenever the app returns to the foreground (e.g. after Store).
  useEffect(() => {
    if (!bootReady) return;

    const onChange = (next: AppStateStatus) => {
      if (next === 'active') {
        void runVersionCheck();
      }
    };

    const sub = AppState.addEventListener('change', onChange);
    return () => sub.remove();
  }, [bootReady, runVersionCheck]);

  if (!bootReady) {
    return null;
  }

  if (versionResult?.status === 'force-update') {
    return (
      <ThemeProvider value={isDark ? DarkTheme : DefaultTheme}>
        <ForceUpdateScreen result={versionResult} />
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider value={isDark ? DarkTheme : DefaultTheme}>
      <StatusBar style={isDark ? 'light' : 'dark'} />
      <Stack
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: colors.background },
        }}
      >
        <Stack.Screen name="index" />
        <Stack.Screen name="(tabs)" />
        <Stack.Screen
          name="capture"
          options={{
            presentation: 'fullScreenModal',
            animation: 'slide_from_bottom',
            contentStyle: { backgroundColor: flow.background },
          }}
        />
        <Stack.Screen
          name="analyzing"
          options={{
            presentation: 'fullScreenModal',
            animation: 'fade',
            gestureEnabled: false,
            contentStyle: { backgroundColor: flow.background },
          }}
        />
        <Stack.Screen
          name="result"
          options={{
            presentation: 'fullScreenModal',
            animation: 'fade',
            contentStyle: { backgroundColor: flow.background },
          }}
        />
        <Stack.Screen
          name="documentation"
          options={{
            presentation: 'fullScreenModal',
            animation: 'slide_from_bottom',
            contentStyle: { backgroundColor: colors.background },
          }}
        />
        <Stack.Screen
          name="documentation-submitted"
          options={{
            presentation: 'fullScreenModal',
            animation: 'fade',
            gestureEnabled: false,
            contentStyle: { backgroundColor: colors.background },
          }}
        />
      </Stack>
    </ThemeProvider>
  );
}
