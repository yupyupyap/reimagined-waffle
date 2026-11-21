set(LOCALCHESSENGINE_CORE_SOURCES
    include/eng/bench.hpp
    include/eng/bitboard.hpp
    include/eng/eval.hpp
    include/eng/move.hpp
    include/eng/movegen.hpp
    include/eng/movepick.hpp
    include/eng/nnue.hpp
    include/eng/nnue_data.hpp
    include/eng/nnue_file.hpp
    include/eng/perft.hpp
    include/eng/position.hpp
    include/eng/rules.hpp
    include/eng/search.hpp
    include/eng/see.hpp
    include/eng/texel.hpp
    include/eng/types.hpp
    include/eng/tt.hpp
    include/eng/uci.hpp
    src/bench.cpp
    src/bitboard.cpp
    src/eval.cpp
    src/movegen.cpp
    src/movepick.cpp
    src/nnue.cpp
    src/nnue_data.cpp
    src/nnue_file.cpp
    src/perft.cpp
    src/position.cpp
    src/rules.cpp
    src/search.cpp
    src/see.cpp
    src/texel.cpp
    src/tt.cpp
    src/uci.cpp
)

add_library(localchessengine_core STATIC
    ${LOCALCHESSENGINE_CORE_SOURCES}
)

target_include_directories(localchessengine_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

localchessengine_apply_common_target_settings(localchessengine_core)
localchessengine_apply_sanitizer(localchessengine_core)
localchessengine_apply_apple_silicon_engine_flags(localchessengine_core)
localchessengine_apply_pgo_flags(localchessengine_core)

if(LOCALCHESSENGINE_BUILD_FUZZERS)
    add_library(localchessengine_fuzz_core STATIC
        ${LOCALCHESSENGINE_CORE_SOURCES}
    )

    target_include_directories(localchessengine_fuzz_core PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )

    localchessengine_apply_common_target_settings(localchessengine_fuzz_core)
    localchessengine_apply_fuzzer_instrumentation(localchessengine_fuzz_core)
endif()

add_executable(localchessengine
    src/main.cpp
)

target_link_libraries(localchessengine PRIVATE
    localchessengine_core
)

localchessengine_apply_common_target_settings(localchessengine)
localchessengine_apply_sanitizer(localchessengine)
localchessengine_apply_apple_silicon_engine_flags(localchessengine)
localchessengine_apply_pgo_flags(localchessengine)
