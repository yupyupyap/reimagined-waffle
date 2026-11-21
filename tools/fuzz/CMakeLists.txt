function(localchessengine_add_parser_fuzzer target_name source_file corpus_name)
    add_executable(${target_name}
        ${source_file}
    )

    target_link_libraries(${target_name} PRIVATE
        localchessengine_fuzz_core
    )

    target_include_directories(${target_name} PRIVATE
        "${CMAKE_SOURCE_DIR}/gui/src"
    )

    localchessengine_apply_common_target_settings(${target_name})
    localchessengine_apply_fuzzer_executable(${target_name})

    set(seed_corpus_dir "${CMAKE_CURRENT_SOURCE_DIR}/corpus/${corpus_name}")
    set(work_corpus_dir "${CMAKE_BINARY_DIR}/fuzz-corpus/${target_name}")
    set(artifact_dir "${CMAKE_BINARY_DIR}/fuzz-artifacts")
    add_custom_target(${target_name}_smoke
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${work_corpus_dir}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${work_corpus_dir}"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${seed_corpus_dir}" "${work_corpus_dir}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${artifact_dir}"
        COMMAND $<TARGET_FILE:${target_name}> "${work_corpus_dir}"
                -runs=${LOCALCHESSENGINE_FUZZER_RUNS}
                -max_total_time=${LOCALCHESSENGINE_FUZZER_MAX_TOTAL_TIME}
                -max_len=${LOCALCHESSENGINE_FUZZER_MAX_LEN}
                -rss_limit_mb=768
                -timeout=5
                -print_final_stats=1
                -artifact_prefix=${artifact_dir}/${target_name}-
        DEPENDS ${target_name}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Running bounded libFuzzer smoke for ${target_name}"
        VERBATIM
        USES_TERMINAL
    )
    set_property(GLOBAL APPEND PROPERTY LOCALCHESSENGINE_FUZZ_SMOKE_TARGETS
        ${target_name}_smoke
    )
endfunction()

localchessengine_add_parser_fuzzer(localchessengine_fuzz_fen fuzz_fen.cpp fen)
localchessengine_add_parser_fuzzer(localchessengine_fuzz_uci fuzz_uci.cpp uci)

add_executable(localchessengine_fuzz_pgn
    fuzz_pgn.cpp
    "${CMAKE_SOURCE_DIR}/gui/src/pgn.cpp"
)

target_link_libraries(localchessengine_fuzz_pgn PRIVATE
    localchessengine_fuzz_core
)

target_include_directories(localchessengine_fuzz_pgn PRIVATE
    "${CMAKE_SOURCE_DIR}/gui/src"
)

localchessengine_apply_common_target_settings(localchessengine_fuzz_pgn)
localchessengine_apply_fuzzer_executable(localchessengine_fuzz_pgn)

set(pgn_seed_corpus_dir "${CMAKE_CURRENT_SOURCE_DIR}/corpus/pgn")
set(pgn_work_corpus_dir "${CMAKE_BINARY_DIR}/fuzz-corpus/localchessengine_fuzz_pgn")
set(pgn_artifact_dir "${CMAKE_BINARY_DIR}/fuzz-artifacts")
add_custom_target(localchessengine_fuzz_pgn_smoke
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${pgn_work_corpus_dir}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${pgn_work_corpus_dir}"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${pgn_seed_corpus_dir}" "${pgn_work_corpus_dir}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${pgn_artifact_dir}"
    COMMAND $<TARGET_FILE:localchessengine_fuzz_pgn> "${pgn_work_corpus_dir}"
            -runs=${LOCALCHESSENGINE_FUZZER_RUNS}
            -max_total_time=${LOCALCHESSENGINE_FUZZER_MAX_TOTAL_TIME}
            -max_len=${LOCALCHESSENGINE_FUZZER_MAX_LEN}
            -rss_limit_mb=768
            -timeout=5
            -print_final_stats=1
            -artifact_prefix=${pgn_artifact_dir}/localchessengine_fuzz_pgn-
    DEPENDS localchessengine_fuzz_pgn
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running bounded libFuzzer smoke for localchessengine_fuzz_pgn"
    VERBATIM
    USES_TERMINAL
)
set_property(GLOBAL APPEND PROPERTY LOCALCHESSENGINE_FUZZ_SMOKE_TARGETS
    localchessengine_fuzz_pgn_smoke
)

localchessengine_add_parser_fuzzer(localchessengine_fuzz_nnue_file fuzz_nnue_file.cpp nnue_file)

get_property(localchessengine_fuzz_smoke_targets GLOBAL PROPERTY
    LOCALCHESSENGINE_FUZZ_SMOKE_TARGETS
)
add_custom_target(fuzz-smoke
    DEPENDS ${localchessengine_fuzz_smoke_targets}
)
