#ifndef GUI_APP_ICON_HPP
#define GUI_APP_ICON_HPP

#include <QIcon>
#include <QPixmap>

QPixmap render_app_icon_pixmap(int logical_size, qreal device_pixel_ratio);
QIcon make_app_icon();

#endif // GUI_APP_ICON_HPP
