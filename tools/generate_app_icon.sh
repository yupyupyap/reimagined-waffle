#!/usr/bin/env bash
# Regenerates the LocalChessEngine macOS app icon from the SVG master.
#
# Inputs : gui/resources/app/icon.svg (canonical 1024x1024 master)
# Outputs:
#   gui/resources/macos/AppIcon.icns  (multi-resolution Apple icon bundle)
#   gui/resources/app/icon-512.png    (Qt-friendly PNG used in README banner)
#
# Stock macOS tooling only: qlmanage (QuickLook SVG rasterizer), sips, iconutil.
# Deterministic: same input SVG produces byte-identical PNG/ICNS across runs.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
src_svg="${repo_root}/gui/resources/app/icon.svg"
app_dir="${repo_root}/gui/resources/app"
out_dir="${repo_root}/gui/resources/macos"
iconset_dir="$(mktemp -d -t lce-iconset.XXXXXX)/AppIcon.iconset"
trap 'rm -rf "$(dirname "${iconset_dir}")"' EXIT

if [[ ! -f "${src_svg}" ]]; then
    echo "error: master SVG not found at ${src_svg}" >&2
    exit 1
fi

mkdir -p "${iconset_dir}" "${out_dir}" "${app_dir}"

# Render a single high-resolution PNG (1024 px) from the SVG, then downscale via sips.
master_png_dir="$(mktemp -d -t lce-iconmaster.XXXXXX)"
trap 'rm -rf "$(dirname "${iconset_dir}")" "${master_png_dir}"' EXIT

qlmanage -t -s 1024 -o "${master_png_dir}" "${src_svg}" >/dev/null 2>&1
master_png="${master_png_dir}/icon.svg.png"
if [[ ! -f "${master_png}" ]]; then
    echo "error: qlmanage failed to render ${src_svg}" >&2
    exit 1
fi

# iconset naming follows Apple convention: <size>x<size>[@2x].png with declared sizes.
emit() {
    local pixels="$1"
    local target="$2"
    sips -s format png -z "${pixels}" "${pixels}" "${master_png}" \
        --out "${iconset_dir}/${target}" >/dev/null
}

emit 16   icon_16x16.png
emit 32   icon_16x16@2x.png
emit 32   icon_32x32.png
emit 64   icon_32x32@2x.png
emit 128  icon_128x128.png
emit 256  icon_128x128@2x.png
emit 256  icon_256x256.png
emit 512  icon_256x256@2x.png
emit 512  icon_512x512.png
emit 1024 icon_512x512@2x.png

iconutil -c icns "${iconset_dir}" -o "${out_dir}/AppIcon.icns"

# Convenience raster consumed by the README banner (Qt itself renders from the SVG master).
sips -s format png -z 512 512 "${master_png}" --out "${app_dir}/icon-512.png" >/dev/null

echo "Generated:"
echo "  $(shasum -a 256 "${out_dir}/AppIcon.icns" | awk '{print $1}')  AppIcon.icns"
echo "  $(shasum -a 256 "${app_dir}/icon-512.png" | awk '{print $1}')  icon-512.png"
