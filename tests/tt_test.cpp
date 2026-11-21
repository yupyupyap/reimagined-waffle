#include <eng/tt.hpp>

#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <thread>
#include <vector>

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-function-cognitive-complexity)

namespace {

constexpr int file_a = 0;
constexpr int file_b = 1;
constexpr int file_c = 2;
constexpr int file_d = 3;
constexpr int rank_1 = 0;
constexpr int rank_2 = 1;
constexpr int worker_count = 4;
constexpr int worker_iterations = 512;

[[nodiscard]] constexpr eng::Square square(int file_index, int rank_index) {
    return eng::Square::from_file_rank(eng::File::from_index(file_index),
                                       eng::Rank::from_index(rank_index));
}

[[nodiscard]] eng::Move sample_move(int file_index) {
    return eng::Move::normal(square(file_index, rank_1), square(file_index, rank_2));
}

[[nodiscard]] eng::ZobristKey key_in_bucket(const eng::TranspositionTable& table,
                                            std::uint64_t bucket_index, std::uint64_t ordinal) {
    return 0x4C43455454000000ULL | bucket_index |
           (ordinal * static_cast<eng::ZobristKey>(table.bucket_count()));
}

[[nodiscard]] eng::TranspositionEntry
require_entry(const std::optional<eng::TranspositionEntry>& entry) {
    REQUIRE(entry.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return entry.value();
}

} // namespace

TEST_CASE("transposition table uses cache-line buckets and aligned allocation") {
    CHECK(eng::tt_entry_size_bytes == 16U);
    CHECK(eng::tt_bucket_size_bytes == 64U);
    CHECK(eng::tt_entries_per_bucket == 4U);

    eng::TranspositionTable table;
    REQUIRE(table.resize_mb(1));

    CHECK(table.bucket_count() > 0U);
    CHECK(table.size_bytes() == table.bucket_count() * eng::tt_bucket_size_bytes);
    CHECK(reinterpret_cast<std::uintptr_t>(table.allocation_base()) %
              eng::tt_allocation_alignment ==
          0U);
    CHECK(table.hashfull_permill() == 0);
}

TEST_CASE("transposition table saves and probes all bound types") {
    eng::TranspositionTable table;
    REQUIRE(table.resize_mb(1));
    table.new_search();

    constexpr eng::ZobristKey exact_key = 1ULL;
    constexpr eng::ZobristKey lower_key = 2ULL;
    constexpr eng::ZobristKey upper_key = 3ULL;

    table.save(exact_key, sample_move(file_a), eng::Score::centipawns(42),
               eng::Score::centipawns(7), eng::Depth::plies(5), eng::TranspositionBound::Exact,
               eng::Ply::from_root(0), true);
    table.save(lower_key, sample_move(file_b), eng::Score::centipawns(120),
               eng::Score::centipawns(9), eng::Depth::plies(4), eng::TranspositionBound::Lower,
               eng::Ply::from_root(0), false);
    table.save(upper_key, sample_move(file_c), eng::Score::centipawns(-30),
               eng::Score::centipawns(-11), eng::Depth::plies(3), eng::TranspositionBound::Upper,
               eng::Ply::from_root(0), false);

    const auto exact_entry = require_entry(table.probe(exact_key, eng::Ply::from_root(0)));
    CHECK(exact_entry.move == sample_move(file_a));
    CHECK(exact_entry.score == eng::Score::centipawns(42));
    CHECK(exact_entry.static_eval == eng::Score::centipawns(7));
    CHECK(exact_entry.depth == eng::Depth::plies(5));
    CHECK(exact_entry.bound == eng::TranspositionBound::Exact);
    CHECK(exact_entry.is_pv);

    const auto lower_entry = require_entry(table.probe(lower_key, eng::Ply::from_root(0)));
    CHECK(lower_entry.bound == eng::TranspositionBound::Lower);
    CHECK(lower_entry.score == eng::Score::centipawns(120));

    const auto upper_entry = require_entry(table.probe(upper_key, eng::Ply::from_root(0)));
    CHECK(upper_entry.bound == eng::TranspositionBound::Upper);
    CHECK(upper_entry.score == eng::Score::centipawns(-30));

    CHECK_FALSE(table.probe(0x9999999999999999ULL, eng::Ply::from_root(0)).has_value());
    CHECK(table.hashfull_permill() > 0);

    table.clear();
    CHECK(table.hashfull_permill() == 0);
    CHECK_FALSE(table.probe(exact_key, eng::Ply::from_root(0)).has_value());
}

TEST_CASE("mate scores are normalized relative to root ply") {
    const auto mate_for_us = eng::Score::centipawns(eng::tt_mate_score - 3);
    const auto stored = eng::score_to_tt(mate_for_us, eng::Ply::from_root(3));
    CHECK(stored == eng::Score::centipawns(eng::tt_mate_score));
    CHECK(eng::score_from_tt(stored, eng::Ply::from_root(3)) == mate_for_us);

    const auto mate_against_us = eng::Score::centipawns(-eng::tt_mate_score + 5);
    const auto stored_loss = eng::score_to_tt(mate_against_us, eng::Ply::from_root(5));
    CHECK(stored_loss == eng::Score::centipawns(-eng::tt_mate_score));
    CHECK(eng::score_from_tt(stored_loss, eng::Ply::from_root(5)) == mate_against_us);
}

TEST_CASE("replacement prefers depth and age") {
    eng::TranspositionTable table;
    REQUIRE(table.resize_mb(1));
    table.new_search();

    constexpr std::uint64_t bucket_index = 17;
    const std::array keys{
        key_in_bucket(table, bucket_index, 1),
        key_in_bucket(table, bucket_index, 2),
        key_in_bucket(table, bucket_index, 3),
        key_in_bucket(table, bucket_index, 4),
    };

    for (std::size_t index = 0; index < keys.size(); ++index) {
        table.save(keys[index], sample_move(static_cast<int>(index)),
                   eng::Score::centipawns(static_cast<int>(index)), eng::Score::centipawns(0),
                   eng::Depth::plies(static_cast<int>(index + 1U)), eng::TranspositionBound::Exact,
                   eng::Ply::from_root(0), false);
    }

    for (int search = 0; search < 3; ++search) {
        table.new_search();
    }

    const auto replacement_key = key_in_bucket(table, bucket_index, 5);
    table.save(replacement_key, sample_move(file_d), eng::Score::centipawns(99),
               eng::Score::centipawns(0), eng::Depth::plies(8), eng::TranspositionBound::Exact,
               eng::Ply::from_root(0), false);

    CHECK_FALSE(table.probe(keys.front(), eng::Ply::from_root(0)).has_value());
    CHECK(table.probe(keys.back(), eng::Ply::from_root(0)).has_value());

    const auto replacement_entry =
        require_entry(table.probe(replacement_key, eng::Ply::from_root(0)));
    CHECK(replacement_entry.move == sample_move(file_d));

    table.save(replacement_key, sample_move(file_a), eng::Score::centipawns(101),
               eng::Score::centipawns(0), eng::Depth::plies(10), eng::TranspositionBound::Lower,
               eng::Ply::from_root(0), true);
    const auto updated_entry = require_entry(table.probe(replacement_key, eng::Ply::from_root(0)));
    CHECK(updated_entry.move == sample_move(file_a));
    CHECK(updated_entry.depth == eng::Depth::plies(10));
    CHECK(updated_entry.bound == eng::TranspositionBound::Lower);
}

TEST_CASE("concurrent transposition table probe and save smoke") {
    eng::TranspositionTable table;
    REQUIRE(table.resize_mb(1));
    table.new_search();

    std::vector<std::jthread> workers;
    workers.reserve(worker_count);
    for (int worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&table, worker] {
            for (int iteration = 0; iteration < worker_iterations; ++iteration) {
                const auto key = 0xABCDEF0000000000ULL |
                                 (static_cast<eng::ZobristKey>(worker) << 20U) |
                                 static_cast<eng::ZobristKey>(iteration);
                table.save(key, sample_move(iteration % 4), eng::Score::centipawns(iteration),
                           eng::Score::centipawns(worker), eng::Depth::plies(iteration % 32),
                           eng::TranspositionBound::Exact, eng::Ply::from_root(iteration % 8),
                           false);
                (void)table.probe(key, eng::Ply::from_root(iteration % 8));
            }
        });
    }
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-function-cognitive-complexity)
