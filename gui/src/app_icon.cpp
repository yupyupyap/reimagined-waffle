#include "app_icon.hpp"

#include <QPainter>
#include <QRectF>
#include <QString>
#include <QSvgRenderer>

#include <algorithm>
#include <array>
#include <cmath>

namespace {

constexpr auto kIconResourcePath = ":/app/icon.svg";

} // namespace

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
QPixmap render_app_icon_pixmap(int logical_size, qreal device_pixel_ratio) {
    QSvgRenderer renderer{QString::fromLatin1(kIconResourcePath)};
    if (!renderer.isValid() || logical_size <= 0) {
        return {};
    }
    const qreal dpr = device_pixel_ratio > 0.0 ? device_pixel_ratio : 1.0;
    const int pixel_size =
        std::max(1, static_cast<int>(std::ceil(static_cast<qreal>(logical_size) * dpr)));
    QPixmap pixmap{pixel_size, pixel_size};
    pixmap.setDevicePixelRatio(dpr);
    pixmap.fill(Qt::transparent);
    QPainter painter{&pixmap};
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
    renderer.render(&painter, QRectF{0.0, 0.0, static_cast<qreal>(logical_size),
                                     static_cast<qreal>(logical_size)});
    return pixmap;
}

QIcon make_app_icon() {
    constexpr std::array<int, 6> kLogicalSizes = {16, 24, 32, 64, 128, 256};
    QIcon icon;
    for (const int size : kLogicalSizes) {
        QPixmap pixmap = render_app_icon_pixmap(size, 1.0);
        if (!pixmap.isNull()) {
            icon.addPixmap(pixmap);
        }
    }
    return icon;
}
