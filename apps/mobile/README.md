# @abjadi/mobile

Expo (SDK 57) app for Abjadi.

## Commands

From the monorepo root:

```bash
pnpm install
pnpm dev:mobile
```

Or from this package:

```bash
pnpm --filter @abjadi/mobile start
pnpm --filter @abjadi/mobile ios
pnpm --filter @abjadi/mobile android
```

## Notes

- Dependencies are managed by the root pnpm workspace — do not use `npm install` here.
- Native projects live under `ios/` and `android/` (Expo prebuild output).
