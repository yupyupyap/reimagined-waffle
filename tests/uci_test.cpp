#include <eng/uci.hpp>

#include <doctest/doctest.h>

#include <sstream>
#include <string>
#include <string_view>

namespace {

[[nodiscard]] bool contains(std::string_view haystack, std::string_view needle) noexcept {
    return haystack.contains(needle);
}

[[nodiscard]] int count_occurrences(std::string_view haystack, std::string_view needle) noexcept {
    int count = 0;
    std::size_t offset = 0;
    while (offset < haystack.size()) {
        const auto found = haystack.find(needle, offset);
        if (found == std::string_view::npos) {
            break;
        }
        ++count;
        offset = found + needle.size();
    }
    return count;
}

[[nodiscard]] std::string run_uci_script(std::string_view script) {
    std::istringstream input{std::string{script}};
    std::ostringstream output;
    eng::UciSession session{input, output};
    session.run();
    return output.str();
}

} // namespace

TEST_CASE("uci handshake identifies engine and declares base options") {
    const auto output = run_uci_script("uci\nquit\n");

    CHECK(contains(output, "id name LocalChessEngine\n"));
    CHECK(contains(output, "id author LocalChessEngine contributors\n"));
    CHECK(contains(output, "option name Hash type spin default 16 min 1 max 65536\n"));
    CHECK(contains(output, "option name Threads type spin default 10 min 1 max 512\n"));
    CHECK(contains(output, "option name MultiPV type spin default 1 min 1 max 256\n"));
    CHECK(contains(output, "option name Move Overhead type spin default 10 min 0 max 5000\n"));
    CHECK(contains(output, "option name Use NNUE type check default false\n"));
    CHECK(contains(output, "option name EvalFile type string default \n"));
    CHECK(contains(output,
                   "option name NNUE Backend type combo default Scalar var Scalar var Neon var "
                   "Auto\n"));
    CHECK(contains(output, "option name Null Move Pruning type check default false\n"));
    CHECK(contains(output, "option name Late Move Reductions type check default false\n"));
    CHECK(contains(output, "option name Reverse Futility Pruning type check default false\n"));
    CHECK(contains(output, "option name Late Move Pruning type check default false\n"));
    CHECK(contains(output, "option name Futility Pruning type check default false\n"));
    CHECK(contains(output, "option name SEE Pruning type check default false\n"));
    CHECK(contains(output, "option name Singular Extensions type check default false\n"));
    CHECK(contains(output, "option name Multi Cut type check default false\n"));
    CHECK(contains(output, "option name Internal Iterative Reductions type check default false\n"));
    CHECK(contains(output, "option name Check Extensions type check default false\n"));
    CHECK(contains(output, "uciok\n"));
}

TEST_CASE("setoption updates experimental NNUE options without requiring a valid network") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    session.handle_line("setoption name Use NNUE value true");
    session.handle_line("setoption name EvalFile value /tmp/localchessengine-missing.nnue");
    session.handle_line("setoption name NNUE Backend value Auto");

    CHECK(session.options().use_nnue);
    CHECK(session.options().eval_file == "/tmp/localchessengine-missing.nnue");
    CHECK(session.options().nnue_backend == eng::NnueInferenceBackend::Auto);

    session.handle_line("setoption name Use NNUE value maybe");
    session.handle_line("setoption name NNUE Backend value bad");

    CHECK(session.options().use_nnue);
    CHECK(session.options().nnue_backend == eng::NnueInferenceBackend::Auto);
}

TEST_CASE("isready always responds readyok") {
    const auto output = run_uci_script("isready\nquit\n");

    CHECK(output == "readyok\n");
}

TEST_CASE("setoption updates valid spin values and keeps prior values on invalid input") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    session.handle_line("setoption name Hash value 128");
    session.handle_line("setoption name Threads value 4");
    session.handle_line("setoption name MultiPV value 3");
    session.handle_line("setoption name Move Overhead value 25");

    CHECK(session.options().hash_mb == 128);
    CHECK(session.options().threads == 4);
    CHECK(session.options().multi_pv == 3);
    CHECK(session.options().move_overhead_ms == 25);

    session.handle_line("setoption name Hash value 0");
    session.handle_line("setoption name Threads value nope");
    session.handle_line("setoption name MultiPV value 999");
    session.handle_line("setoption name Move Overhead value -1");
    session.handle_line("setoption name Unknown value 42");

    CHECK(session.options().hash_mb == 128);
    CHECK(session.options().threads == 4);
    CHECK(session.options().multi_pv == 3);
    CHECK(session.options().move_overhead_ms == 25);
}

TEST_CASE("setoption updates pruning check options and rejects invalid boolean input") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    session.handle_line("setoption name Null Move Pruning value true");
    session.handle_line("setoption name Late Move Reductions value true");
    session.handle_line("setoption name Reverse Futility Pruning value true");
    session.handle_line("setoption name Late Move Pruning value true");
    session.handle_line("setoption name Futility Pruning value true");
    session.handle_line("setoption name SEE Pruning value true");

    CHECK(session.options().pruning.null_move_pruning);
    CHECK(session.options().pruning.late_move_reductions);
    CHECK(session.options().pruning.reverse_futility_pruning);
    CHECK(session.options().pruning.late_move_pruning);
    CHECK(session.options().pruning.futility_pruning);
    CHECK(session.options().pruning.see_pruning);

    session.handle_line("setoption name Null Move Pruning value maybe");
    session.handle_line("setoption name SEE Pruning value false");

    CHECK(session.options().pruning.null_move_pruning);
    CHECK(!session.options().pruning.see_pruning);
}

TEST_CASE("setoption updates advanced extension check options and rejects invalid boolean input") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    session.handle_line("setoption name Singular Extensions value true");
    session.handle_line("setoption name Multi Cut value true");
    session.handle_line("setoption name Internal Iterative Reductions value true");
    session.handle_line("setoption name Check Extensions value true");

    CHECK(session.options().extensions.singular_extensions);
    CHECK(session.options().extensions.multi_cut);
    CHECK(session.options().extensions.internal_iterative_reductions);
    CHECK(session.options().extensions.check_extensions);

    session.handle_line("setoption name Singular Extensions value maybe");
    session.handle_line("setoption name Multi Cut value false");

    CHECK(session.options().extensions.singular_extensions);
    CHECK(!session.options().extensions.multi_cut);
}

TEST_CASE("hash option resizes the transposition table") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    const auto default_buckets = session.transposition_table().bucket_count();
    REQUIRE(default_buckets > 0U);

    session.handle_line("setoption name Hash value 1");
    const auto one_mb_buckets = session.transposition_table().bucket_count();
    CHECK(session.options().hash_mb == 1);
    CHECK(one_mb_buckets > 0U);
    CHECK(one_mb_buckets < default_buckets);

    session.handle_line("setoption name Hash value 2");
    CHECK(session.options().hash_mb == 2);
    CHECK(session.transposition_table().bucket_count() > one_mb_buckets);

    session.handle_line("setoption name Hash value 0");
    CHECK(session.options().hash_mb == 2);
    CHECK(session.transposition_table().bucket_count() > one_mb_buckets);
}

TEST_CASE("position startpos applies legal UCI moves atomically") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    session.handle_line("position startpos moves e2e4 e7e5 g1f3");
    CHECK(session.position().to_fen() ==
          "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");

    session.handle_line("position startpos moves e2e5");
    CHECK(session.position().to_fen() ==
          "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
}

TEST_CASE("position fen supports castling and promotion moves") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    session.handle_line("position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1 moves e1g1");
    CHECK(session.position().to_fen() == "r3k2r/8/8/8/8/8/8/R4RK1 b kq - 1 1");

    session.handle_line("position fen 4k3/6P1/8/8/8/8/8/4K3 w - - 0 1 moves g7g8q");
    CHECK(session.position().to_fen() == "4k1Q1/8/8/8/8/8/8/4K3 b - - 0 1");
}

TEST_CASE("go depth emits search info and respects searchmoves restrictions") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    session.handle_line("position startpos");
    session.handle_line("go searchmoves g1f3 depth 1");
    session.wait_for_search();

    CHECK(contains(output.str(), "info depth 1 seldepth"));
    CHECK(contains(output.str(), " nodes "));
    CHECK(contains(output.str(), " nps "));
    CHECK(contains(output.str(), " score cp "));
    CHECK(contains(output.str(), " hashfull "));
    CHECK(contains(output.str(), " pv g1f3"));
    CHECK(contains(output.str(), "bestmove g1f3\n"));
}

TEST_CASE("go depth emits one info line per completed iterative depth") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    session.handle_line("position startpos");
    session.handle_line("go depth 3");
    session.wait_for_search();

    const auto text = output.str();
    CHECK(count_occurrences(text, "info depth ") == 3);
    CHECK(contains(text, "info depth 1 "));
    CHECK(contains(text, "info depth 2 "));
    CHECK(contains(text, "info depth 3 "));
    CHECK(count_occurrences(text, " hashfull ") == 3);
    CHECK(count_occurrences(text, " score cp ") == 3);
    CHECK(count_occurrences(text, " nodes ") == 3);
    CHECK(count_occurrences(text, " nps ") == 3);
    CHECK(contains(text, " pv "));
    CHECK(count_occurrences(text, "bestmove ") == 1);
}

TEST_CASE("go depth keeps iterative pv inside searchmoves restrictions") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    session.handle_line("position startpos");
    session.handle_line("go searchmoves g1f3 b1c3 depth 3");
    session.wait_for_search();

    const auto text = output.str();
    CHECK(count_occurrences(text, "info depth ") == 3);
    CHECK((contains(text, " pv g1f3") || contains(text, " pv b1c3")));
    CHECK((contains(text, "bestmove g1f3\n") || contains(text, "bestmove b1c3\n")));
    CHECK(count_occurrences(text, "bestmove ") == 1);
}

TEST_CASE("go emits null bestmove when no legal moves exist") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    session.handle_line("position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    session.handle_line("go depth 1");
    session.wait_for_search();

    CHECK(contains(output.str(), "info depth 1 seldepth 0"));
    CHECK(contains(output.str(), "nodes 0"));
    CHECK(contains(output.str(), "score cp -30000"));
    CHECK(contains(output.str(), "bestmove 0000\n"));
}

TEST_CASE("go infinite waits for stop while isready still responds") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    session.handle_line("position startpos");
    session.handle_line("go infinite");

    session.handle_line("isready");
    CHECK(contains(output.str(), "readyok\n"));
    CHECK(!contains(output.str(), "bestmove"));

    session.handle_line("stop");
    CHECK(contains(output.str(), "info depth"));
    CHECK(contains(output.str(), "bestmove"));
    CHECK(count_occurrences(output.str(), "bestmove ") == 1);
}

TEST_CASE("go nodes and movetime stop cooperatively with a bestmove") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    session.handle_line("position startpos");
    session.handle_line("setoption name Threads value 1");
    session.handle_line("go nodes 1");
    session.wait_for_search();

    CHECK(contains(output.str(), "nodes 1"));
    CHECK(contains(output.str(), "bestmove "));

    output.str(std::string{});
    output.clear();
    session.handle_line("position startpos");
    session.handle_line("setoption name Move Overhead value 5000");
    session.handle_line("go movetime 1");
    session.wait_for_search();

    CHECK(contains(output.str(), "info depth "));
    CHECK(contains(output.str(), "bestmove "));
    CHECK(count_occurrences(output.str(), "bestmove ") == 1);
}

TEST_CASE("go depth works with Threads set to Apple Silicon default") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    session.handle_line("setoption name Threads value 10");
    session.handle_line("position startpos");
    session.handle_line("go depth 2");
    session.wait_for_search();

    const auto text = output.str();
    CHECK(session.options().threads == 10);
    CHECK(contains(text, "info depth 1 "));
    CHECK(contains(text, "info depth 2 "));
    CHECK(contains(text, "bestmove "));
    CHECK(count_occurrences(text, "bestmove ") == 1);
}

TEST_CASE("go clock limits and searchmoves return legal restricted bestmove") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    session.handle_line("position startpos");
    session.handle_line("go searchmoves g1f3 wtime 100 btime 100 winc 0 binc 0 movestogo 10");
    session.wait_for_search();

    CHECK(contains(output.str(), "info depth "));
    CHECK(contains(output.str(), " pv g1f3"));
    CHECK(contains(output.str(), "bestmove g1f3\n"));
    CHECK(count_occurrences(output.str(), "bestmove ") == 1);
}

TEST_CASE("unknown commands are ignored without protocol output") {
    std::istringstream input;
    std::ostringstream output;
    eng::UciSession session{input, output};

    session.handle_line("not-a-command with tokens");

    CHECK(output.str().empty());
}
