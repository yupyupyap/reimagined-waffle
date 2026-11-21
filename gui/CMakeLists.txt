find_package(Qt6 REQUIRED COMPONENTS Widgets Svg SvgWidgets)

qt_standard_project_setup()

qt_add_resources(LOCALCHESSENGINE_GUI_RESOURCES
    resources/pieces.qrc
    resources/app.qrc
)

set(LOCALCHESSENGINE_APP_ICON
    "${CMAKE_CURRENT_SOURCE_DIR}/resources/macos/AppIcon.icns"
)
set_source_files_properties("${LOCALCHESSENGINE_APP_ICON}" PROPERTIES
    MACOSX_PACKAGE_LOCATION "Resources"
)

qt_add_executable(localchessengine_gui
    MACOSX_BUNDLE
    ${LOCALCHESSENGINE_GUI_RESOURCES}
    ${LOCALCHESSENGINE_APP_ICON}
    src/app_icon.cpp
    src/app_icon.hpp
    src/app_settings.cpp
    src/app_settings.hpp
    src/board_widget.cpp
    src/board_widget.hpp
    src/engine_controller.cpp
    src/engine_controller.hpp
    src/engine_info_panel.cpp
    src/engine_info_panel.hpp
    src/evaluation_bar_widget.cpp
    src/evaluation_bar_widget.hpp
    src/evaluation_display.cpp
    src/evaluation_display.hpp
    src/game_sidebar.cpp
    src/game_sidebar.hpp
    src/game_state.cpp
    src/game_state.hpp
    src/main.cpp
    src/main_window.cpp
    src/main_window.hpp
    src/pgn.cpp
    src/pgn.hpp
    src/preferences_dialog.cpp
    src/preferences_dialog.hpp
)

localchessengine_apply_common_target_settings(localchessengine_gui)
localchessengine_apply_sanitizer(localchessengine_gui)

target_link_libraries(localchessengine_gui PRIVATE
    localchessengine_core
    Qt6::Widgets
    Qt6::Svg
    Qt6::SvgWidgets
)

set_target_properties(localchessengine_gui PROPERTIES
    MACOSX_BUNDLE_BUNDLE_NAME "LocalChessEngine"
    MACOSX_BUNDLE_COPYRIGHT "Copyright (c) 2026 LocalChessEngine contributors"
    MACOSX_BUNDLE_GUI_IDENTIFIER "dev.localchessengine.gui"
    MACOSX_BUNDLE_INFO_STRING "LocalChessEngine Qt Widgets desktop shell"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
    MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
    MACOSX_BUNDLE_ICON_FILE "AppIcon.icns"
)

if(APPLE AND DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    set(LOCALCHESSENGINE_QT_PLUGIN_ROOT
        "$<IF:$<CONFIG:Debug>,${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/Qt6/plugins,${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/Qt6/plugins>"
    )
    set(LOCALCHESSENGINE_QT_DEBUG_SUFFIX "$<$<CONFIG:Debug>:_debug>")
    add_custom_command(TARGET localchessengine_gui POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "$<TARGET_BUNDLE_CONTENT_DIR:localchessengine_gui>/PlugIns/platforms"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "$<TARGET_BUNDLE_CONTENT_DIR:localchessengine_gui>/PlugIns/styles"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${LOCALCHESSENGINE_QT_PLUGIN_ROOT}/platforms/libqcocoa${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.dylib"
                "$<TARGET_BUNDLE_CONTENT_DIR:localchessengine_gui>/PlugIns/platforms/"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${LOCALCHESSENGINE_QT_PLUGIN_ROOT}/styles/libqmacstyle${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.dylib"
                "$<TARGET_BUNDLE_CONTENT_DIR:localchessengine_gui>/PlugIns/styles/"
        COMMAND install_name_tool -rpath
                "@loader_path/../../../lib"
                "@loader_path/../../Frameworks"
                "$<TARGET_BUNDLE_CONTENT_DIR:localchessengine_gui>/PlugIns/platforms/libqcocoa${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.dylib"
        COMMAND install_name_tool -change
                "@rpath/libQt6Core${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.6.dylib"
                "@rpath/libQt6Core${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.${Qt6_VERSION}.dylib"
                "$<TARGET_BUNDLE_CONTENT_DIR:localchessengine_gui>/PlugIns/platforms/libqcocoa${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.dylib"
        COMMAND install_name_tool -change
                "@rpath/libQt6Gui${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.6.dylib"
                "@rpath/libQt6Gui${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.${Qt6_VERSION}.dylib"
                "$<TARGET_BUNDLE_CONTENT_DIR:localchessengine_gui>/PlugIns/platforms/libqcocoa${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.dylib"
        COMMAND install_name_tool -rpath
                "@loader_path/../../../lib"
                "@loader_path/../../Frameworks"
                "$<TARGET_BUNDLE_CONTENT_DIR:localchessengine_gui>/PlugIns/styles/libqmacstyle${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.dylib"
        COMMAND install_name_tool -change
                "@rpath/libQt6Core${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.6.dylib"
                "@rpath/libQt6Core${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.${Qt6_VERSION}.dylib"
                "$<TARGET_BUNDLE_CONTENT_DIR:localchessengine_gui>/PlugIns/styles/libqmacstyle${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.dylib"
        COMMAND install_name_tool -change
                "@rpath/libQt6Gui${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.6.dylib"
                "@rpath/libQt6Gui${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.${Qt6_VERSION}.dylib"
                "$<TARGET_BUNDLE_CONTENT_DIR:localchessengine_gui>/PlugIns/styles/libqmacstyle${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.dylib"
        COMMAND install_name_tool -change
                "@rpath/libQt6Widgets${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.6.dylib"
                "@rpath/libQt6Widgets${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.${Qt6_VERSION}.dylib"
                "$<TARGET_BUNDLE_CONTENT_DIR:localchessengine_gui>/PlugIns/styles/libqmacstyle${LOCALCHESSENGINE_QT_DEBUG_SUFFIX}.dylib"
        COMMENT "Copying Qt macOS platform plugins into LocalChessEngine.app"
        VERBATIM
    )
endif()
