cmake_minimum_required(VERSION 3.28)

project(LocalChessEngine VERSION 0.2.0 LANGUAGES CXX)

include(CTest)

option(LOCALCHESSENGINE_ENABLE_WARNINGS "Enable baseline compiler warnings." ON)
option(LOCALCHESSENGINE_PROFILE_BUILD "Enable profiling-friendly compile flags." OFF)
option(LOCALCHESSENGINE_BUILD_GUI "Build the Qt Widgets desktop GUI target." ON)
option(LOCALCHESSENGINE_BUILD_FUZZERS "Build bounded libFuzzer parser harnesses." OFF)

set(LOCALCHESSENGINE_FUZZER_RUNS
    "128"
    CACHE STRING "Number of libFuzzer iterations for the fuzz-smoke target.")
set(LOCALCHESSENGINE_FUZZER_MAX_TOTAL_TIME
    "8"
    CACHE STRING "Maximum seconds per libFuzzer harness in the fuzz-smoke target.")
set(LOCALCHESSENGINE_FUZZER_MAX_LEN
    "4096"
    CACHE STRING "Maximum generated input length for parser fuzz harnesses.")

set(LOCALCHESSENGINE_PGO_MODE
    "Off"
    CACHE STRING "PGO mode for engine/performance targets: Off, Generate, or Use.")
set_property(CACHE LOCALCHESSENGINE_PGO_MODE PROPERTY STRINGS Off Generate Use)
set(LOCALCHESSENGINE_PGO_PROFILE
    ""
    CACHE FILEPATH "Merged LLVM .profdata file consumed when LOCALCHESSENGINE_PGO_MODE=Use.")

set(LOCALCHESSENGINE_SANITIZER
    ""
    CACHE STRING "Sanitizer to enable for supported compilers: Address, Undefined, Thread, or empty.")
set_property(CACHE LOCALCHESSENGINE_SANITIZER PROPERTY STRINGS "" Address Undefined Thread)

function(localchessengine_apply_common_target_settings target_name)
    target_compile_features(${target_name} PUBLIC cxx_std_23)

    set_target_properties(${target_name} PROPERTIES
        CXX_EXTENSIONS OFF
    )

    if(LOCALCHESSENGINE_ENABLE_WARNINGS)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>
            $<$<CXX_COMPILER_ID:MSVC>:/W4>
        )
    endif()
endfunction()

function(localchessengine_apply_sanitizer target_name)
    string(TOLOWER "${LOCALCHESSENGINE_SANITIZER}" sanitizer_name)

    if(sanitizer_name STREQUAL "" OR sanitizer_name STREQUAL "none")
        return()
    elseif(sanitizer_name STREQUAL "address")
        set(sanitizer_flags -fsanitize=address -fno-omit-frame-pointer)
    elseif(sanitizer_name STREQUAL "undefined")
        set(sanitizer_flags -fsanitize=undefined -fno-omit-frame-pointer)
    elseif(sanitizer_name STREQUAL "thread")
        set(sanitizer_flags -fsanitize=thread -fno-omit-frame-pointer)
    else()
        message(FATAL_ERROR
            "Unsupported LOCALCHESSENGINE_SANITIZER='${LOCALCHESSENGINE_SANITIZER}'. "
            "Use Address, Undefined, Thread, or leave it empty.")
    endif()

    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:${sanitizer_flags}>
    )
    target_link_options(${target_name} PRIVATE
        $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:${sanitizer_flags}>
    )
endfunction()

function(localchessengine_apply_apple_silicon_engine_flags target_name)
    if(APPLE
            AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$"
            AND CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
        target_compile_options(${target_name} PRIVATE
            $<$<CONFIG:Release>:-O3;-mcpu=apple-m4;-march=armv8.5-a+dotprod;-flto=thin;-fno-exceptions;-fno-rtti>
            $<$<BOOL:${LOCALCHESSENGINE_PROFILE_BUILD}>:-g;-fno-omit-frame-pointer>
        )
        target_link_options(${target_name} PRIVATE
            $<$<CONFIG:Release>:-flto=thin>
        )
    endif()
endfunction()

function(localchessengine_apply_pgo_flags target_name)
    string(TOLOWER "${LOCALCHESSENGINE_PGO_MODE}" pgo_mode)

    if(pgo_mode STREQUAL "off" OR pgo_mode STREQUAL "")
        return()
    elseif(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
        message(FATAL_ERROR
            "LOCALCHESSENGINE_PGO_MODE currently supports AppleClang/Clang LLVM profiles only.")
    elseif(pgo_mode STREQUAL "generate")
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang>:-fprofile-instr-generate>
        )
        target_link_options(${target_name} PRIVATE
            $<$<CXX_COMPILER_ID:AppleClang,Clang>:-fprofile-instr-generate>
        )
    elseif(pgo_mode STREQUAL "use")
        if(LOCALCHESSENGINE_PGO_PROFILE STREQUAL "")
            message(FATAL_ERROR
                "LOCALCHESSENGINE_PGO_PROFILE must point to a merged .profdata file "
                "when LOCALCHESSENGINE_PGO_MODE=Use.")
        endif()
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang>:-fprofile-instr-use=${LOCALCHESSENGINE_PGO_PROFILE}>
        )
        target_link_options(${target_name} PRIVATE
            $<$<CXX_COMPILER_ID:AppleClang,Clang>:-fprofile-instr-use=${LOCALCHESSENGINE_PGO_PROFILE}>
        )
    else()
        message(FATAL_ERROR
            "Unsupported LOCALCHESSENGINE_PGO_MODE='${LOCALCHESSENGINE_PGO_MODE}'. "
            "Use Off, Generate, or Use.")
    endif()
endfunction()

function(localchessengine_apply_fuzzer_instrumentation target_name)
    if(NOT LOCALCHESSENGINE_BUILD_FUZZERS)
        return()
    elseif(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
        message(FATAL_ERROR
            "LOCALCHESSENGINE_BUILD_FUZZERS requires AppleClang/Clang with libFuzzer support.")
    endif()

    set(fuzzer_flags -fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer)
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang>:${fuzzer_flags}>
    )
    target_link_options(${target_name} PRIVATE
        $<$<CXX_COMPILER_ID:AppleClang,Clang>:${fuzzer_flags}>
    )
endfunction()

function(localchessengine_apply_fuzzer_executable target_name)
    if(NOT LOCALCHESSENGINE_BUILD_FUZZERS)
        return()
    elseif(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
        message(FATAL_ERROR
            "LOCALCHESSENGINE_BUILD_FUZZERS requires AppleClang/Clang with libFuzzer support.")
    endif()

    set(fuzzer_flags -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang>:${fuzzer_flags}>
    )
    target_link_options(${target_name} PRIVATE
        $<$<CXX_COMPILER_ID:AppleClang,Clang>:${fuzzer_flags}>
    )
endfunction()

set(LOCALCHESSENGINE_FORMAT_SOURCES
    "${CMAKE_SOURCE_DIR}/engine/include/eng/bench.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/bitboard.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/eval.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/move.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/movegen.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/movepick.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/nnue.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/nnue_data.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/nnue_file.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/perft.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/position.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/rules.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/search.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/see.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/texel.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/types.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/tt.hpp"
    "${CMAKE_SOURCE_DIR}/engine/include/eng/uci.hpp"
    "${CMAKE_SOURCE_DIR}/engine/src/bench.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/bitboard.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/eval.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/main.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/movegen.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/movepick.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/nnue.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/nnue_data.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/nnue_file.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/perft.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/position.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/rules.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/search.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/see.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/texel.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/tt.cpp"
    "${CMAKE_SOURCE_DIR}/engine/src/uci.cpp"
    "${CMAKE_SOURCE_DIR}/gui/src/app_icon.cpp"
    "${CMAKE_SOURCE_DIR}/gui/src/app_icon.hpp"
    "${CMAKE_SOURCE_DIR}/gui/src/app_settings.cpp"
    "${CMAKE_SOURCE_DIR}/gui/src/app_settings.hpp"
    "${CMAKE_SOURCE_DIR}/gui/src/board_widget.cpp"
    "${CMAKE_SOURCE_DIR}/gui/src/board_widget.hpp"
    "${CMAKE_SOURCE_DIR}/gui/src/engine_controller.cpp"
    "${CMAKE_SOURCE_DIR}/gui/src/engine_controller.hpp"
    "${CMAKE_SOURCE_DIR}/gui/src/engine_info_panel.cpp"
    "${CMAKE_SOURCE_DIR}/gui/src/engine_info_panel.hpp"
    "${CMAKE_SOURCE_DIR}/gui/src/evaluation_bar_widget.cpp"
    "${CMAKE_SOURCE_DIR}/gui/src/evaluation_bar_widget.hpp"
    "${CMAKE_SOURCE_DIR}/gui/src/evaluation_display.cpp"
    "${CMAKE_SOURCE_DIR}/gui/src/evaluation_display.hpp"
    "${CMAKE_SOURCE_DIR}/gui/src/game_sidebar.cpp"
    "${CMAKE_SOURCE_DIR}/gui/src/game_sidebar.hpp"
    "${CMAKE_SOURCE_DIR}/gui/src/game_state.cpp"
    "${CMAKE_SOURCE_DIR}/gui/src/game_state.hpp"
    "${CMAKE_SOURCE_DIR}/gui/src/main.cpp"
    "${CMAKE_SOURCE_DIR}/gui/src/main_window.cpp"
    "${CMAKE_SOURCE_DIR}/gui/src/main_window.hpp"
    "${CMAKE_SOURCE_DIR}/gui/src/pgn.cpp"
    "${CMAKE_SOURCE_DIR}/gui/src/pgn.hpp"
    "${CMAKE_SOURCE_DIR}/gui/src/preferences_dialog.cpp"
    "${CMAKE_SOURCE_DIR}/gui/src/preferences_dialog.hpp"
    "${CMAKE_SOURCE_DIR}/tests/bitboard_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/bench_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/eval_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/engine_controller_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/gui_game_state_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/make_unmake_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/movegen_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/movepick_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/nnue_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/nnue_data_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/nnue_file_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/perft_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/pgn_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/position_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/rules_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/search_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/see_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/smoke_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/texel_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/types_move_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/tt_test.cpp"
    "${CMAKE_SOURCE_DIR}/tests/uci_test.cpp"
    "${CMAKE_SOURCE_DIR}/tools/fuzz/fuzz_fen.cpp"
    "${CMAKE_SOURCE_DIR}/tools/fuzz/fuzz_nnue_file.cpp"
    "${CMAKE_SOURCE_DIR}/tools/fuzz/fuzz_pgn.cpp"
    "${CMAKE_SOURCE_DIR}/tools/fuzz/fuzz_uci.cpp"
    "${CMAKE_SOURCE_DIR}/trainer/src/localchessengine_trainer/native/binpack.cpp"
    "${CMAKE_SOURCE_DIR}/tools/bitboard_bench.cpp"
    "${CMAKE_SOURCE_DIR}/tools/nnue_bench.cpp"
    "${CMAKE_SOURCE_DIR}/tools/nnue_data.cpp"
    "${CMAKE_SOURCE_DIR}/tools/perft.cpp"
    "${CMAKE_SOURCE_DIR}/tools/texel_tune.cpp"
)
set(LOCALCHESSENGINE_TIDY_SOURCES
    ${LOCALCHESSENGINE_FORMAT_SOURCES}
)
list(REMOVE_ITEM LOCALCHESSENGINE_TIDY_SOURCES
    "${CMAKE_SOURCE_DIR}/trainer/src/localchessengine_trainer/native/binpack.cpp"
)

find_program(LOCALCHESSENGINE_CLANG_FORMAT
    NAMES clang-format
    HINTS "/opt/homebrew/opt/llvm/bin"
)
find_program(LOCALCHESSENGINE_CLANG_TIDY
    NAMES clang-tidy
    HINTS "/opt/homebrew/opt/llvm/bin"
)

set(LOCALCHESSENGINE_TIDY_EXTRA_ARGS)
list(APPEND LOCALCHESSENGINE_TIDY_EXTRA_ARGS
    "--extra-arg=-std=c++23"
    "--extra-arg=-I${CMAKE_SOURCE_DIR}/engine/include"
    "--extra-arg=-I${CMAKE_SOURCE_DIR}/gui/src"
)
if(APPLE)
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE LOCALCHESSENGINE_APPLE_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(LOCALCHESSENGINE_APPLE_SDK_PATH)
        list(APPEND LOCALCHESSENGINE_TIDY_EXTRA_ARGS
            "--extra-arg=-isysroot"
            "--extra-arg=${LOCALCHESSENGINE_APPLE_SDK_PATH}"
        )
    endif()
endif()

if(LOCALCHESSENGINE_CLANG_FORMAT)
    add_custom_target(format
        COMMAND "${LOCALCHESSENGINE_CLANG_FORMAT}" -i ${LOCALCHESSENGINE_FORMAT_SOURCES}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Formatting LocalChessEngine C++ sources"
        VERBATIM
    )
    add_custom_target(format-check
        COMMAND "${LOCALCHESSENGINE_CLANG_FORMAT}" --dry-run --Werror
                ${LOCALCHESSENGINE_FORMAT_SOURCES}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Checking LocalChessEngine C++ formatting"
        VERBATIM
    )
else()
    add_custom_target(format
        COMMAND "${CMAKE_COMMAND}" -E false
        COMMENT "clang-format was not found; install Homebrew LLVM or add clang-format to PATH"
    )
    add_custom_target(format-check
        COMMAND "${CMAKE_COMMAND}" -E false
        COMMENT "clang-format was not found; install Homebrew LLVM or add clang-format to PATH"
    )
endif()

if(LOCALCHESSENGINE_CLANG_TIDY)
    add_custom_target(tidy-check
        COMMAND "${LOCALCHESSENGINE_CLANG_TIDY}" --warnings-as-errors=* -p
                "${CMAKE_BINARY_DIR}" ${LOCALCHESSENGINE_TIDY_EXTRA_ARGS}
                ${LOCALCHESSENGINE_TIDY_SOURCES}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Running clang-tidy on LocalChessEngine C++ sources"
        VERBATIM
    )
else()
    add_custom_target(tidy-check
        COMMAND "${CMAKE_COMMAND}" -E false
        COMMENT "clang-tidy was not found; install Homebrew LLVM or add clang-tidy to PATH"
    )
endif()

add_custom_target(lint
    DEPENDS format-check tidy-check
)

add_subdirectory(engine)
if(LOCALCHESSENGINE_BUILD_GUI)
    add_subdirectory(gui)
endif()
if(LOCALCHESSENGINE_BUILD_FUZZERS)
    add_subdirectory(tools/fuzz)
endif()

add_executable(localchessengine_bitboard_bench
    tools/bitboard_bench.cpp
)

target_link_libraries(localchessengine_bitboard_bench PRIVATE
    localchessengine_core
)

localchessengine_apply_common_target_settings(localchessengine_bitboard_bench)
localchessengine_apply_sanitizer(localchessengine_bitboard_bench)
localchessengine_apply_apple_silicon_engine_flags(localchessengine_bitboard_bench)
localchessengine_apply_pgo_flags(localchessengine_bitboard_bench)

add_executable(localchessengine_perft
    tools/perft.cpp
)

target_link_libraries(localchessengine_perft PRIVATE
    localchessengine_core
)

localchessengine_apply_common_target_settings(localchessengine_perft)
localchessengine_apply_sanitizer(localchessengine_perft)
localchessengine_apply_apple_silicon_engine_flags(localchessengine_perft)
localchessengine_apply_pgo_flags(localchessengine_perft)

add_executable(localchessengine_texel_tune
    tools/texel_tune.cpp
)

target_link_libraries(localchessengine_texel_tune PRIVATE
    localchessengine_core
)

localchessengine_apply_common_target_settings(localchessengine_texel_tune)
localchessengine_apply_sanitizer(localchessengine_texel_tune)
localchessengine_apply_pgo_flags(localchessengine_texel_tune)

add_executable(localchessengine_nnue_data
    tools/nnue_data.cpp
)

target_link_libraries(localchessengine_nnue_data PRIVATE
    localchessengine_core
)

localchessengine_apply_common_target_settings(localchessengine_nnue_data)
localchessengine_apply_sanitizer(localchessengine_nnue_data)
localchessengine_apply_apple_silicon_engine_flags(localchessengine_nnue_data)
localchessengine_apply_pgo_flags(localchessengine_nnue_data)

add_executable(localchessengine_nnue_bench
    tools/nnue_bench.cpp
)

target_link_libraries(localchessengine_nnue_bench PRIVATE
    localchessengine_core
)

localchessengine_apply_common_target_settings(localchessengine_nnue_bench)
localchessengine_apply_sanitizer(localchessengine_nnue_bench)
localchessengine_apply_apple_silicon_engine_flags(localchessengine_nnue_bench)
localchessengine_apply_pgo_flags(localchessengine_nnue_bench)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
