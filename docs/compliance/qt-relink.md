# Qt Relink And Replacement Instructions

The macOS v0.2 GUI uses Qt 6.10.3 dynamic libraries under the LGPLv3 path. The
engine remains a separate UCI executable and does not link Qt.

These instructions are provided so a recipient can replace the bundled Qt
dynamic libraries with compatible modified Qt builds, or rebuild the application
against a modified Qt installation.

## Bundle Layout

Release packaging places the GUI at:

```sh
LocalChessEngine.app
```

Qt runtime libraries live under:

```sh
LocalChessEngine.app/Contents/Frameworks
```

Qt macOS plugins live under:

```sh
LocalChessEngine.app/Contents/PlugIns
```

The embedded UCI engine lives at:

```sh
LocalChessEngine.app/Contents/MacOS/localchessengine
```

The GUI autodetects that engine path through `QCoreApplication::applicationDirPath()`.

## Replace Bundled Qt Libraries

1. Build or obtain compatible modified Qt dynamic libraries for macOS arm64.
2. Replace the matching `libQt6*.dylib` files in
   `LocalChessEngine.app/Contents/Frameworks`.
3. Replace matching Qt plugin dylibs in `Contents/PlugIns` if the modified Qt
   build requires plugin changes.
4. Preserve install names expected by the app, or update them with
   `install_name_tool` so dependencies resolve through
   `@executable_path/../Frameworks` or plugin-relative `@rpath` entries.
5. Re-sign the modified local bundle. For local testing, an ad-hoc signature is
   sufficient:

```sh
find LocalChessEngine.app/Contents/Frameworks -type f -name '*.dylib' -print0 \
  | xargs -0 -n 1 codesign --force --sign -
find LocalChessEngine.app/Contents/PlugIns -type f -name '*.dylib' -print0 \
  | xargs -0 -n 1 codesign --force --sign -
codesign --force --sign - LocalChessEngine.app/Contents/MacOS/localchessengine
codesign --force --sign - LocalChessEngine.app
codesign --verify --deep --strict --verbose=2 LocalChessEngine.app
```

A modified local bundle will not carry the original notarization ticket. If the
modified bundle is redistributed, notarize that new artifact with the
distributor's Apple Developer account.

## Rebuild Against Modified Qt

1. Build the modified Qt through vcpkg or provide a compatible CMake Qt package.
2. Configure LocalChessEngine with the desired vcpkg root, installed directory,
   overlay triplet, and Qt package path.
3. Build the release targets:

```sh
cmake --preset release
cmake --build --preset release --target localchessengine localchessengine_gui
```

4. Run the release packaging script to copy the engine, legal materials, and
   dynamic Qt runtime into a new signed/notarized distribution artifact.

## Verification

Use these checks after replacement or rebuild:

```sh
otool -L LocalChessEngine.app/Contents/MacOS/localchessengine_gui
otool -L LocalChessEngine.app/Contents/MacOS/localchessengine
codesign --verify --deep --strict --verbose=2 LocalChessEngine.app
LocalChessEngine.app/Contents/MacOS/localchessengine bench
LocalChessEngine.app/Contents/MacOS/localchessengine_gui --smoke-exit
```

Expected: the GUI resolves Qt from `Contents/Frameworks`, the engine has no Qt
dependencies, and both executables launch successfully.
