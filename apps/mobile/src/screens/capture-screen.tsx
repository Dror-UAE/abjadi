import { Asset } from 'expo-asset';
import { BlurView } from 'expo-blur';
import {
  CameraView,
  useCameraPermissions,
  type CameraType,
  type FlashMode,
} from 'expo-camera';
import * as Device from 'expo-device';
import * as ImageManipulator from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import { router, useFocusEffect } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SymbolView } from 'expo-symbols';
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  ActivityIndicator,
  Image,
  Linking,
  Platform,
  Pressable,
  StyleSheet,
  useWindowDimensions,
  View,
} from 'react-native';
import { Gesture, GestureDetector, GestureHandlerRootView } from 'react-native-gesture-handler';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  type SharedValue,
} from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { ArabicText } from '@/components/arabic-text';
import { Palette, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';

/** iOS Simulator has no camera hardware — never mount CameraView there. */
const HAS_CAMERA = Device.isDevice;

const FRAME_RADIUS = 16;
const DIM = 'rgba(8, 5, 3, 0.62)';
const MIN_FRAME_W = 120;
const MIN_FRAME_H = 48;
const HANDLE = 30;
const CORNER = 34;
const CORNER_STROKE = 2.5;
const SIMULATOR_PREVIEW = require('../../assets/images/scanned-image.jpg');
const SOURCE_META = Image.resolveAssetSource(SIMULATOR_PREVIEW);

type ScreenRect = { x: number; y: number; width: number; height: number };

function getDisplayedImageRect(
  containerW: number,
  containerH: number,
  imageW: number,
  imageH: number,
  mode: 'contain' | 'cover'
) {
  const scale =
    mode === 'contain'
      ? Math.min(containerW / imageW, containerH / imageH)
      : Math.max(containerW / imageW, containerH / imageH);
  const width = imageW * scale;
  const height = imageH * scale;
  return {
    left: (containerW - width) / 2,
    top: (containerH - height) / 2,
    width,
    height,
    scale,
  };
}

function selectionToCrop(
  selection: ScreenRect,
  imageW: number,
  imageH: number,
  containerW: number,
  containerH: number,
  mode: 'contain' | 'cover'
) {
  const displayed = getDisplayedImageRect(containerW, containerH, imageW, imageH, mode);
  let originX = (selection.x - displayed.left) / displayed.scale;
  let originY = (selection.y - displayed.top) / displayed.scale;
  let width = selection.width / displayed.scale;
  let height = selection.height / displayed.scale;

  originX = Math.max(0, Math.min(originX, imageW - 1));
  originY = Math.max(0, Math.min(originY, imageH - 1));
  width = Math.max(1, Math.min(width, imageW - originX));
  height = Math.max(1, Math.min(height, imageH - originY));

  return {
    originX: Math.round(originX),
    originY: Math.round(originY),
    width: Math.round(width),
    height: Math.round(height),
  };
}

async function cropImageToSelection(
  sourceUri: string,
  imageW: number,
  imageH: number,
  selection: ScreenRect,
  containerW: number,
  containerH: number,
  mode: 'contain' | 'cover'
) {
  const crop = selectionToCrop(selection, imageW, imageH, containerW, containerH, mode);
  const result = await ImageManipulator.manipulateAsync(
    sourceUri,
    [{ crop }],
    { compress: 1, format: ImageManipulator.SaveFormat.JPEG }
  );
  return result.uri;
}

type GlassButtonProps = {
  onPress: () => void;
  accessibilityLabel: string;
  children: ReactNode;
  size?: number;
  active?: boolean;
};

function GlassButton({
  onPress,
  accessibilityLabel,
  children,
  size = 44,
  active = false,
}: GlassButtonProps) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel}
      onPress={onPress}
      style={({ pressed }) => [
        styles.glassOuter,
        { width: size, height: size, borderRadius: size / 2 },
        active && styles.glassActive,
        pressed && styles.pressed,
      ]}
    >
      <BlurView
        intensity={Platform.OS === 'ios' ? 48 : 32}
        tint="dark"
        style={[styles.glassInner, { borderRadius: size / 2 }]}
      >
        {children}
      </BlurView>
    </Pressable>
  );
}

type RegionBounds = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

type SelectableRegionProps = {
  bounds: RegionBounds;
  frameX: SharedValue<number>;
  frameY: SharedValue<number>;
  frameW: SharedValue<number>;
  frameH: SharedValue<number>;
  showGrid: boolean;
};

function SelectableRegion({
  bounds,
  frameX,
  frameY,
  frameW,
  frameH,
  showGrid,
}: SelectableRegionProps) {
  const startX = useSharedValue(0);
  const startY = useSharedValue(0);
  const startW = useSharedValue(0);
  const startH = useSharedValue(0);

  const moveGesture = Gesture.Pan()
    .onStart(() => {
      startX.value = frameX.value;
      startY.value = frameY.value;
    })
    .onUpdate((event) => {
      const w = frameW.value;
      const h = frameH.value;
      const maxX = Math.max(bounds.minX, bounds.maxX - w);
      const maxY = Math.max(bounds.minY, bounds.maxY - h);
      frameX.value = Math.min(Math.max(startX.value + event.translationX, bounds.minX), maxX);
      frameY.value = Math.min(Math.max(startY.value + event.translationY, bounds.minY), maxY);
    });

  const makeResizeGesture = (corner: 'tl' | 'tr' | 'bl' | 'br') =>
    Gesture.Pan()
      .onStart(() => {
        startX.value = frameX.value;
        startY.value = frameY.value;
        startW.value = frameW.value;
        startH.value = frameH.value;
      })
      .onUpdate((event) => {
        let x = startX.value;
        let y = startY.value;
        let w = startW.value;
        let h = startH.value;
        const dx = event.translationX;
        const dy = event.translationY;

        if (corner === 'tl') {
          x = startX.value + dx;
          y = startY.value + dy;
          w = startW.value - dx;
          h = startH.value - dy;
        } else if (corner === 'tr') {
          y = startY.value + dy;
          w = startW.value + dx;
          h = startH.value - dy;
        } else if (corner === 'bl') {
          x = startX.value + dx;
          w = startW.value - dx;
          h = startH.value + dy;
        } else {
          w = startW.value + dx;
          h = startH.value + dy;
        }

        if (w < MIN_FRAME_W) {
          if (corner === 'tl' || corner === 'bl') {
            x = startX.value + startW.value - MIN_FRAME_W;
          }
          w = MIN_FRAME_W;
        }
        if (h < MIN_FRAME_H) {
          if (corner === 'tl' || corner === 'tr') {
            y = startY.value + startH.value - MIN_FRAME_H;
          }
          h = MIN_FRAME_H;
        }

        x = Math.min(Math.max(x, bounds.minX), bounds.maxX - MIN_FRAME_W);
        y = Math.min(Math.max(y, bounds.minY), bounds.maxY - MIN_FRAME_H);
        w = Math.min(Math.max(w, MIN_FRAME_W), bounds.maxX - x);
        h = Math.min(Math.max(h, MIN_FRAME_H), bounds.maxY - y);

        frameX.value = x;
        frameY.value = y;
        frameW.value = w;
        frameH.value = h;
      });

  const topBandStyle = useAnimatedStyle(() => ({ height: frameY.value }));
  const bottomBandStyle = useAnimatedStyle(() => ({ top: frameY.value + frameH.value }));
  const leftBandStyle = useAnimatedStyle(() => ({
    top: frameY.value,
    width: frameX.value,
    height: frameH.value,
  }));
  const rightBandStyle = useAnimatedStyle(() => ({
    top: frameY.value,
    left: frameX.value + frameW.value,
    height: frameH.value,
  }));
  const frameStyle = useAnimatedStyle(() => ({
    top: frameY.value,
    left: frameX.value,
    width: frameW.value,
    height: frameH.value,
  }));
  const handleTLStyle = useAnimatedStyle(() => ({
    top: frameY.value - HANDLE / 2,
    left: frameX.value - HANDLE / 2,
  }));
  const handleTRStyle = useAnimatedStyle(() => ({
    top: frameY.value - HANDLE / 2,
    left: frameX.value + frameW.value - HANDLE / 2,
  }));
  const handleBLStyle = useAnimatedStyle(() => ({
    top: frameY.value + frameH.value - HANDLE / 2,
    left: frameX.value - HANDLE / 2,
  }));
  const handleBRStyle = useAnimatedStyle(() => ({
    top: frameY.value + frameH.value - HANDLE / 2,
    left: frameX.value + frameW.value - HANDLE / 2,
  }));

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="box-none">
      <Animated.View pointerEvents="none" style={[styles.maskBand, styles.maskTop, topBandStyle]} />
      <Animated.View
        pointerEvents="none"
        style={[styles.maskBand, styles.maskBottom, bottomBandStyle]}
      />
      <Animated.View
        pointerEvents="none"
        style={[styles.maskBand, styles.maskLeft, leftBandStyle]}
      />
      <Animated.View
        pointerEvents="none"
        style={[styles.maskBand, styles.maskRight, rightBandStyle]}
      />

      <GestureDetector gesture={moveGesture}>
        <Animated.View style={[styles.frameSlot, frameStyle]}>
          {showGrid ? (
            <View pointerEvents="none" style={styles.grid}>
              <View style={[styles.gridLineV, { left: '33.333%' }]} />
              <View style={[styles.gridLineV, { left: '66.666%' }]} />
              <View style={[styles.gridLineH, { top: '33.333%' }]} />
              <View style={[styles.gridLineH, { top: '66.666%' }]} />
            </View>
          ) : null}

          <View pointerEvents="none" style={[styles.corner, styles.cornerTL]} />
          <View pointerEvents="none" style={[styles.corner, styles.cornerTR]} />
          <View pointerEvents="none" style={[styles.corner, styles.cornerBL]} />
          <View pointerEvents="none" style={[styles.corner, styles.cornerBR]} />
        </Animated.View>
      </GestureDetector>

      <GestureDetector gesture={makeResizeGesture('tl')}>
        <Animated.View style={[styles.handle, handleTLStyle]} />
      </GestureDetector>
      <GestureDetector gesture={makeResizeGesture('tr')}>
        <Animated.View style={[styles.handle, handleTRStyle]} />
      </GestureDetector>
      <GestureDetector gesture={makeResizeGesture('bl')}>
        <Animated.View style={[styles.handle, handleBLStyle]} />
      </GestureDetector>
      <GestureDetector gesture={makeResizeGesture('br')}>
        <Animated.View style={[styles.handle, handleBRStyle]} />
      </GestureDetector>
    </View>
  );
}

function PermissionGate({
  onRequest,
  onClose,
  canAskAgain,
}: {
  onRequest: () => void;
  onClose: () => void;
  canAskAgain: boolean;
}) {
  const insets = useSafeAreaInsets();
  const { flow } = useTheme();

  const handlePress = () => {
    if (canAskAgain) {
      onRequest();
      return;
    }
    void Linking.openSettings();
  };

  return (
    <View
      style={[
        styles.permissionScreen,
        {
          paddingTop: insets.top + Spacing.four,
          backgroundColor: flow.background,
        },
      ]}
    >
      <StatusBar style={flow.statusBar} />
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Close"
        onPress={onClose}
        style={[
          styles.closeAlone,
          {
            top: insets.top + Spacing.two,
            backgroundColor: flow.iconBtn,
          },
        ]}
      >
        <SymbolView
          name={{ ios: 'xmark', android: 'close', web: 'close' }}
          size={18}
          tintColor={flow.icon}
        />
      </Pressable>

      <View style={styles.permissionBody}>
        <View style={styles.permissionIcon}>
          <SymbolView
            name={{
              ios: 'camera.fill',
              android: 'photo_camera',
              web: 'photo_camera',
            }}
            size={28}
            tintColor={Palette.sandstone}
          />
        </View>
        <ArabicText weight="bold" style={[styles.permissionTitle, { color: flow.text }]}>
          إذن الكاميرا مطلوب
        </ArabicText>
        <ArabicText style={[styles.permissionCopy, { color: flow.textSecondary }]}>
          {canAskAgain
            ? 'يحتاج أبجدي إلى الكاميرا لتصوير النصوص بوضوح.'
            : 'تم رفض إذن الكاميرا. افتح الإعدادات وفعّل الكاميرا لأبجدي.'}
        </ArabicText>
        <Pressable
          accessibilityRole="button"
          onPress={handlePress}
          style={({ pressed }) => [styles.permissionButton, pressed && styles.pressed]}
        >
          <ArabicText weight="semiBold" style={styles.permissionButtonLabel}>
            {canAskAgain ? 'السماح بالكاميرا' : 'فتح الإعدادات'}
          </ArabicText>
        </Pressable>
      </View>
    </View>
  );
}

export function CaptureScreen() {
  const insets = useSafeAreaInsets();
  const { width: screenWidth, height: screenHeight } = useWindowDimensions();
  const cameraRef = useRef<CameraView>(null);

  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<CameraType>('back');
  const [flash, setFlash] = useState<FlashMode>('off');
  const [showGrid, setShowGrid] = useState(false);
  const [ready, setReady] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [shutterFlash, setShutterFlash] = useState(false);
  const [isFocused, setIsFocused] = useState(true);
  const [cameraUnavailable, setCameraUnavailable] = useState(!HAS_CAMERA);

  const layout = useMemo(() => {
    const topChrome = insets.top + 64;
    const bottomChrome = Math.max(insets.bottom, 12) + 148;
    const minX = 16;
    const maxX = screenWidth - 16;
    const minY = topChrome;
    const maxY = screenHeight - bottomChrome;
    const availableW = maxX - minX;
    const availableH = Math.max(0, maxY - minY);

    // Default around the tablet inscription (center, wide landscape).
    const width = Math.min(availableW * 0.88, 360);
    const height = Math.min(Math.round(width * 0.55), availableH * 0.7);
    const x = minX + (availableW - width) / 2;
    const y = minY + Math.max(0, (availableH - height) / 2);

    return {
      bounds: { minX, minY, maxX, maxY },
      initial: { x, y, width, height },
    };
  }, [screenWidth, screenHeight, insets.top, insets.bottom]);

  const frameX = useSharedValue(layout.initial.x);
  const frameY = useSharedValue(layout.initial.y);
  const frameW = useSharedValue(layout.initial.width);
  const frameH = useSharedValue(layout.initial.height);

  useEffect(() => {
    frameX.value = layout.initial.x;
    frameY.value = layout.initial.y;
    frameW.value = layout.initial.width;
    frameH.value = layout.initial.height;
  }, [layout.initial, frameX, frameY, frameW, frameH]);

  useFocusEffect(
    useCallback(() => {
      setIsFocused(true);
      return () => {
        setIsFocused(false);
        setReady(false);
      };
    }, [])
  );

  const close = useCallback(() => {
    if (router.canGoBack()) {
      router.back();
    } else {
      router.replace('/home');
    }
  }, []);

  const toggleFlash = useCallback(() => {
    setFlash((current) => (current === 'off' ? 'on' : 'off'));
  }, []);

  const toggleFacing = useCallback(() => {
    setFacing((current) => (current === 'back' ? 'front' : 'back'));
  }, []);

  const openGallery = useCallback(async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 1,
      allowsEditing: false,
    });

    if (!result.canceled && result.assets[0]?.uri) {
      router.replace({
        pathname: '/analyzing',
        params: { uri: encodeURIComponent(result.assets[0].uri) },
      });
    }
  }, []);

  const takePhoto = useCallback(async () => {
    if (capturing) return;

    const selection: ScreenRect = {
      x: frameX.value,
      y: frameY.value,
      width: frameW.value,
      height: frameH.value,
    };

    try {
      setCapturing(true);
      setShutterFlash(true);

      if (cameraUnavailable) {
        const asset = Asset.fromModule(SIMULATOR_PREVIEW);
        await asset.downloadAsync();
        const sourceUri = asset.localUri ?? asset.uri;
        if (!sourceUri || !SOURCE_META?.width || !SOURCE_META?.height) {
          router.replace({
            pathname: '/analyzing',
            params: { uri: 'mock' },
          });
          return;
        }

        const croppedUri = await cropImageToSelection(
          sourceUri,
          SOURCE_META.width,
          SOURCE_META.height,
          selection,
          screenWidth,
          screenHeight,
          'contain'
        );

        router.replace({
          pathname: '/analyzing',
          params: { uri: encodeURIComponent(croppedUri) },
        });
        return;
      }

      if (!cameraRef.current || !ready) {
        setCapturing(false);
        setShutterFlash(false);
        return;
      }

      const photo = await cameraRef.current.takePictureAsync({
        quality: 1,
        shutterSound: true,
      });

      if (photo?.uri) {
        const croppedUri = await cropImageToSelection(
          photo.uri,
          photo.width,
          photo.height,
          selection,
          screenWidth,
          screenHeight,
          'cover'
        );
        router.replace({
          pathname: '/analyzing',
          params: { uri: encodeURIComponent(croppedUri) },
        });
      } else {
        setCapturing(false);
        setShutterFlash(false);
      }
    } catch {
      setCapturing(false);
      setShutterFlash(false);
    }
  }, [
    ready,
    capturing,
    cameraUnavailable,
    frameX,
    frameY,
    frameW,
    frameH,
    screenWidth,
    screenHeight,
  ]);

  if (!cameraUnavailable) {
    if (!permission) {
      return <View style={[styles.root, { backgroundColor: '#050403' }]} />;
    }

    if (!permission.granted) {
      return (
        <PermissionGate
          onRequest={() => void requestPermission()}
          onClose={close}
          canAskAgain={permission.canAskAgain}
        />
      );
    }
  }

  const flashOn = flash === 'on';
  const showLiveCamera = !cameraUnavailable && isFocused;
  const shutterBusy = cameraUnavailable ? capturing : !ready || capturing;

  return (
    <GestureHandlerRootView style={styles.root}>
      <StatusBar style="light" />

      {showLiveCamera ? (
        <CameraView
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          facing={facing}
          flash={flash}
          mode="picture"
          animateShutter
          onCameraReady={() => setReady(true)}
          onMountError={() => setCameraUnavailable(true)}
        />
      ) : (
        <Image
          source={SIMULATOR_PREVIEW}
          style={styles.simulatorPreview}
          resizeMode="contain"
        />
      )}

      <SelectableRegion
        bounds={layout.bounds}
        frameX={frameX}
        frameY={frameY}
        frameW={frameW}
        frameH={frameH}
        showGrid={showGrid}
      />

      {shutterFlash ? <View pointerEvents="none" style={styles.shutterFlash} /> : null}

      <View style={[styles.topBar, { paddingTop: insets.top + Spacing.two }]}>
        <GlassButton onPress={close} accessibilityLabel="Back">
          <SymbolView
            name={{ ios: 'chevron.left', android: 'arrow_back', web: 'arrow_back' }}
            size={18}
            tintColor={Palette.duneBeige}
          />
        </GlassButton>

        {cameraUnavailable ? (
          <View style={styles.simChip}>
            <ArabicText weight="medium" style={styles.simChipText}>
              معاينة
            </ArabicText>
          </View>
        ) : (
          <View style={styles.topSpacer} />
        )}

        <View style={styles.topActions}>
          <GlassButton
            onPress={() => setShowGrid((value) => !value)}
            accessibilityLabel="Toggle grid"
            active={showGrid}
          >
            <SymbolView
              name={{ ios: 'grid', android: 'grid_on', web: 'grid_on' }}
              size={17}
              tintColor={showGrid ? Palette.sandstone : Palette.duneBeige}
            />
          </GlassButton>

          <GlassButton
            onPress={toggleFlash}
            accessibilityLabel="Toggle flash"
            active={flashOn}
          >
            <SymbolView
              name={{
                ios: flashOn ? 'bolt.fill' : 'bolt.slash.fill',
                android: flashOn ? 'flash_on' : 'flash_off',
                web: flashOn ? 'flash_on' : 'flash_off',
              }}
              size={17}
              tintColor={flashOn ? Palette.ochreClay : Palette.duneBeige}
            />
          </GlassButton>
        </View>
      </View>

      <View
        style={[
          styles.bottomChrome,
          { paddingBottom: Math.max(insets.bottom, Spacing.three) },
        ]}
      >
        <ArabicText weight="medium" style={styles.hint}>
          {cameraUnavailable
            ? 'اسحب الإطار أو زواياه لتحديد النص'
            : 'ضع النص داخل الإطار'}
        </ArabicText>

        <View style={styles.controlsRow}>
          <GlassButton
            onPress={() => void openGallery()}
            accessibilityLabel="Open gallery"
            size={50}
          >
            <SymbolView
              name={{
                ios: 'photo.on.rectangle',
                android: 'photo_library',
                web: 'photo_library',
              }}
              size={20}
              tintColor={Palette.duneBeige}
            />
          </GlassButton>

          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Capture photo"
            disabled={shutterBusy}
            onPress={() => void takePhoto()}
            style={({ pressed }) => [
              styles.shutterOuter,
              pressed && styles.shutterPressed,
              shutterBusy && styles.shutterDisabled,
            ]}
          >
            <View style={styles.shutterGap}>
              {capturing ? (
                <ActivityIndicator color={Palette.deepSandBrown} />
              ) : (
                <View style={styles.shutterInner} />
              )}
            </View>
          </Pressable>

          <GlassButton
            onPress={toggleFacing}
            accessibilityLabel="Switch camera"
            size={50}
          >
            <SymbolView
              name={{
                ios: 'camera.rotate',
                android: 'cameraswitch',
                web: 'cameraswitch',
              }}
              size={20}
              tintColor={Palette.duneBeige}
            />
          </GlassButton>
        </View>
      </View>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#050403',
  },

  simulatorPreview: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    width: '100%',
    height: '100%',
    backgroundColor: '#120c08',
  },

  maskBand: {
    position: 'absolute',
    backgroundColor: DIM,
  },

  maskTop: {
    top: 0,
    left: 0,
    right: 0,
  },

  maskBottom: {
    left: 0,
    right: 0,
    bottom: 0,
  },

  maskLeft: {
    left: 0,
  },

  maskRight: {
    right: 0,
  },

  frameSlot: {
    position: 'absolute',
    borderRadius: FRAME_RADIUS,
    overflow: 'visible',
  },

  grid: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
  },

  gridLineV: {
    position: 'absolute',
    top: 14,
    bottom: 14,
    width: StyleSheet.hairlineWidth,
    backgroundColor: 'rgba(247, 244, 236, 0.32)',
  },

  gridLineH: {
    position: 'absolute',
    left: 14,
    right: 14,
    height: StyleSheet.hairlineWidth,
    backgroundColor: 'rgba(247, 244, 236, 0.32)',
  },

  corner: {
    position: 'absolute',
    width: CORNER,
    height: CORNER,
    borderColor: Palette.sandstone,
  },

  cornerTL: {
    top: 0,
    left: 0,
    borderTopWidth: CORNER_STROKE,
    borderLeftWidth: CORNER_STROKE,
    borderTopLeftRadius: 12,
  },

  cornerTR: {
    top: 0,
    right: 0,
    borderTopWidth: CORNER_STROKE,
    borderRightWidth: CORNER_STROKE,
    borderTopRightRadius: 12,
  },

  cornerBL: {
    bottom: 0,
    left: 0,
    borderBottomWidth: CORNER_STROKE,
    borderLeftWidth: CORNER_STROKE,
    borderBottomLeftRadius: 12,
  },

  cornerBR: {
    bottom: 0,
    right: 0,
    borderBottomWidth: CORNER_STROKE,
    borderRightWidth: CORNER_STROKE,
    borderBottomRightRadius: 12,
  },

  handle: {
    position: 'absolute',
    width: HANDLE,
    height: HANDLE,
    borderRadius: HANDLE / 2,
    backgroundColor: Palette.sandstone,
    borderWidth: 2,
    borderColor: 'rgba(247, 244, 236, 0.95)',
    zIndex: 3,
  },

  topBar: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    paddingHorizontal: Spacing.three,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    zIndex: 2,
  },

  topSpacer: {
    flex: 1,
  },

  simChip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: 'rgba(12, 8, 5, 0.55)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(214, 182, 147, 0.28)',
  },

  simChipText: {
    color: Palette.sandstone,
    fontSize: 11,
    letterSpacing: 0.2,
  },

  topActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },

  bottomChrome: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    gap: Spacing.three,
    paddingTop: Spacing.two,
    paddingHorizontal: Spacing.four,
    zIndex: 2,
  },

  hint: {
    color: 'rgba(247, 244, 236, 0.9)',
    fontSize: 15,
    lineHeight: 24,
    textAlign: 'center',
    textShadowColor: 'rgba(0,0,0,0.5)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 6,
  },

  controlsRow: {
    width: '100%',
    maxWidth: 340,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.three,
  },

  glassOuter: {
    overflow: 'hidden',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(247, 244, 236, 0.18)',
    backgroundColor: 'rgba(18, 12, 8, 0.35)',
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 10,
      },
      android: {
        elevation: 5,
      },
      default: {},
    }),
  },

  glassInner: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },

  glassActive: {
    borderColor: 'rgba(214, 182, 147, 0.55)',
    backgroundColor: 'rgba(89, 58, 35, 0.48)',
  },

  shutterOuter: {
    width: 78,
    height: 78,
    borderRadius: 39,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 3.5,
    borderColor: 'rgba(247, 244, 236, 0.95)',
    backgroundColor: 'transparent',
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.35,
        shadowRadius: 12,
      },
      android: {
        elevation: 6,
      },
      default: {},
    }),
  },

  shutterGap: {
    width: 64,
    height: 64,
    borderRadius: 32,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'transparent',
  },

  shutterInner: {
    width: 62,
    height: 62,
    borderRadius: 31,
    backgroundColor: Palette.duneBeige,
  },

  shutterPressed: {
    transform: [{ scale: 0.92 }],
    opacity: 0.9,
  },

  shutterDisabled: {
    opacity: 0.5,
  },

  shutterFlash: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    backgroundColor: 'rgba(247, 244, 236, 0.5)',
  },

  pressed: {
    opacity: 0.8,
    transform: [{ scale: 0.96 }],
  },

  permissionScreen: {
    flex: 1,
    paddingHorizontal: Spacing.four,
  },

  closeAlone: {
    position: 'absolute',
    left: Spacing.three,
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },

  permissionBody: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.three,
    paddingBottom: Spacing.six,
  },

  permissionIcon: {
    width: 72,
    height: 72,
    borderRadius: 36,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(214, 182, 147, 0.18)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(214, 182, 147, 0.35)',
    marginBottom: Spacing.two,
  },

  permissionTitle: {
    fontSize: 22,
    textAlign: 'center',
  },

  permissionCopy: {
    fontSize: 15,
    lineHeight: 24,
    textAlign: 'center',
    maxWidth: 280,
  },

  permissionButton: {
    marginTop: Spacing.two,
    paddingHorizontal: Spacing.four,
    paddingVertical: Spacing.three,
    borderRadius: 999,
    backgroundColor: Palette.sandstone,
  },

  permissionButtonLabel: {
    color: Palette.deepSandBrown,
    fontSize: 16,
  },
});
