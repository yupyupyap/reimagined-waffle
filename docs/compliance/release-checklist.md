# Release Compliance Checklist

This checklist is the release packaging gate for macOS v0.2 and later.

## Legal Inputs

- `LICENSE` exists and records the project MIT license.
- `COPYING.LGPL-3.0.txt` and `COPYING.GPL-3.0.txt` exist.
- `NOTICE` and `THIRD_PARTY_NOTICES.md` are current.
- `docs/compliance/privacy.md` documents local-first behavior.
- `docs/compliance/qt-relink.md` documents replacement/relink steps.
- The release compliance package includes vcpkg copyright/SPDX files and source
  archives for Qt and redistributed dependency sources.

## Supply Chain

- `vcpkg.json` declares C++ dependencies.
- `trainer/pyproject.toml` declares Python runtime and optional dependencies.
- `python3 tools/build_compliance_package.py --check` passes.
- Every bundled non-system dylib in `Contents/Frameworks` or `Contents/PlugIns`
  is listed in `THIRD_PARTY_NOTICES.md`.
- `otool -L` shows the CLI engine links only macOS system libraries.
- `otool -L` shows the GUI links Qt dynamically from `@executable_path`.

## Signing And Notarization

Use a Developer ID Application identity and a local notarytool keychain profile.
Do not store Apple passwords, app-specific passwords, API keys, or notary logs
with secrets in the repository.

```sh
security find-identity -v -p codesigning
xcrun notarytool store-credentials localchessengine-notary \
  --apple-id "<apple-id>" \
  --team-id "<TEAMID>"
python3 tools/package_macos_release.py \
  --identity "Developer ID Application: <Name> (<TEAMID>)" \
  --notary-profile localchessengine-notary
```

The packaging script signs nested code explicitly before signing the app bundle.
Use `--deep` only for verification, not as a substitute for explicit nested
signing.

## Verification

- `codesign --verify --deep --strict --verbose=2 LocalChessEngine.app` passes.
- `spctl --assess --type execute --verbose LocalChessEngine.app` passes after
  notarization.
- `xcrun stapler validate <dmg>` passes.
- The DMG mounts.
- `LocalChessEngine.app/Contents/MacOS/localchessengine bench` passes from the
  packaged app.
- `LocalChessEngine.app/Contents/MacOS/localchessengine_gui --smoke-exit`
  passes from the packaged app.
- `tools/reports/task-047-validation.md` records commands, outputs, hashes, and
  any notarization submission identifier.

## Rollback

If notarization fails for product code reasons, keep TASK-047 in `In Progress`
with the failing log and fix the package. If notarization fails because the
Apple Developer account, team, agreements, or keychain profile are unavailable,
move TASK-047 to `Blocked` and record the exact unblock condition.

## References

- Qt LGPL obligations: https://www.qt.io/development/open-source-lgpl-obligations
- Qt licensing: https://doc.qt.io/qt-6/licensing.html
- Qt macOS deployment: https://doc.qt.io/qt-6/macos-deployment.html
- GNU LGPLv3: https://www.gnu.org/licenses/lgpl-3.0.en.html
- vcpkg SBOM/SPDX: https://learn.microsoft.com/en-us/vcpkg/reference/software-bill-of-materials
- Apple Developer ID certificates: https://developer.apple.com/help/account/certificates/create-developer-id-certificates/
- Apple notarization: https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution
- Apple code signing: https://support.apple.com/guide/security/app-code-signing-process-sec3ad8e6e53/web
