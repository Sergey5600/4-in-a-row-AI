"""Classic alpha-beta opponent with a threat-counting heuristic, used to measure the AI's strength."""
import random

from .game import State, WIDTH, HEIGHT, H1, CENTER_ORDER

BOARD_MASK = sum(((1 << HEIGHT) - 1) << (c * H1) for c in range(WIDTH))
CENTER_COL = ((1 << HEIGHT) - 1) << (3 * H1)
WIN = 10000


def winning_cells(p, mask):
    """Empty cells that would complete four for the player with stones p."""
    r = (p << 1) & (p << 2) & (p << 3)
    for s in (H1, H1 - 1, H1 + 1):
        t = (p << s) & (p << 2 * s)
        r |= t & (p << 3 * s)
        r |= t & (p >> s)
        t = (p >> s) & (p >> 2 * s)
        r |= t & (p << s)
        r |= t & (p >> 3 * s)
    return r & (BOARD_MASK ^ mask)


def heuristic(st):
    opp = st.mask ^ st.cur
    own_t = bin(winning_cells(st.cur, st.mask)).count("1")
    opp_t = bin(winning_cells(opp, st.mask)).count("1")
    center = bin(st.cur & CENTER_COL).count("1") - bin(opp & CENTER_COL).count("1")
    return 8 * (own_t - opp_t) + 2 * center


def negamax(st, depth, alpha, beta):
    moves = [c for c in CENTER_ORDER if st.can_play(c)]
    children = []
    for c in moves:
        nxt, won = st.play(c)
        if won:
            return WIN - st.moves
        children.append(nxt)
    if not children or st.moves >= 41:
        return 0
    if depth == 0:
        return heuristic(st)
    best = -WIN * 2
    for nxt in children:
        v = -negamax(nxt, depth - 1, -beta, -alpha)
        if v > best:
            best = v
        if v > alpha:
            alpha = v
        if alpha >= beta:
            break
    return best


def minimax_move(st, depth, rng=random):
    scores = {}
    for c in CENTER_ORDER:
        if not st.can_play(c):
            continue
        nxt, won = st.play(c)
        if won:
            return c
        scores[c] = 0 if nxt.is_full() else -negamax(nxt, depth - 1, -WIN * 2, WIN * 2)
    best = max(scores.values())
    return rng.choice([c for c, v in scores.items() if v == best])
