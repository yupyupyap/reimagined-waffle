#include <eng/perft.hpp>

#include <eng/movegen.hpp>

namespace eng {

PerftNodeCount perft(Position& position, int depth) noexcept {
    if (depth < 0) {
        return 0;
    }
    if (depth == 0) {
        return 1;
    }

    const auto moves = generate_moves(position);
    if (depth == 1) {
        PerftNodeCount nodes = 0;
        for (const auto move : moves) {
            StateInfo state;
            if (position.make_move(move, state)) {
                position.unmake_move(move, state);
                ++nodes;
            }
        }
        return nodes;
    }

    PerftNodeCount nodes = 0;
    for (const auto move : moves) {
        StateInfo state;
        if (!position.make_move(move, state)) {
            continue;
        }

        nodes += perft(position, depth - 1);
        position.unmake_move(move, state);
    }

    return nodes;
}

std::vector<PerftDivideEntry> divide(Position& position, int depth) {
    std::vector<PerftDivideEntry> result;
    if (depth <= 0) {
        return result;
    }

    const auto moves = generate_moves(position);
    result.reserve(moves.size());
    for (const auto move : moves) {
        StateInfo state;
        if (!position.make_move(move, state)) {
            continue;
        }

        result.push_back(PerftDivideEntry{.move = move, .nodes = perft(position, depth - 1)});
        position.unmake_move(move, state);
    }

    return result;
}

} // namespace eng
