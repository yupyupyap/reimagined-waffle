find_package(doctest CONFIG REQUIRED)

add_executable(localchessengine_tests
    bench_test.cpp
    bitboard_test.cpp
    eval_test.cpp
    gui_game_state_test.cpp
    make_unmake_test.cpp
    movegen_test.cpp
    movepick_test.cpp
    nnue_test.cpp
    nnue_data_test.cpp
    nnue_file_test.cpp
    perft_test.cpp
    pgn_test.cpp
    position_test.cpp
    rules_test.cpp
    search_test.cpp
    see_test.cpp
    smoke_test.cpp
    texel_test.cpp
    types_move_test.cpp
    tt_test.cpp
    uci_test.cpp
    ../gui/src/game_state.cpp
    ../gui/src/pgn.cpp
)

localchessengine_apply_common_target_settings(localchessengine_tests)
localchessengine_apply_sanitizer(localchessengine_tests)

target_link_libraries(localchessengine_tests PRIVATE
    localchessengine_core
    doctest::doctest
)

target_include_directories(localchessengine_tests PRIVATE
    "${CMAKE_SOURCE_DIR}/gui/src"
)

add_test(NAME localchessengine.smoke COMMAND localchessengine_tests)

if(LOCALCHESSENGINE_BUILD_GUI)
    find_package(Qt6 REQUIRED COMPONENTS Core)

    add_executable(localchessengine_gui_tests
        engine_controller_test.cpp
        ../gui/src/app_settings.cpp
        ../gui/src/app_settings.hpp
        ../gui/src/engine_controller.cpp
        ../gui/src/engine_controller.hpp
        ../gui/src/evaluation_display.cpp
        ../gui/src/evaluation_display.hpp
    )

    set_target_properties(localchessengine_gui_tests PROPERTIES
        AUTOMOC ON
    )

    localchessengine_apply_common_target_settings(localchessengine_gui_tests)
    localchessengine_apply_sanitizer(localchessengine_gui_tests)

    target_link_libraries(localchessengine_gui_tests PRIVATE
        localchessengine_core
        Qt6::Core
    )

    target_include_directories(localchessengine_gui_tests PRIVATE
        "${CMAKE_SOURCE_DIR}/gui/src"
    )

    add_test(NAME localchessengine.gui_engine_controller
             COMMAND localchessengine_gui_tests $<TARGET_FILE:localchessengine>)
endif()
