"""Measure the trained AI against a random player and alpha-beta opponents.

Usage: python evaluate.py --model models/latest.npz --sims 200 --games 20
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
import random

import numpy as np

from connect4.game import State
from connect4.mcts import search, best_move
from connect4.minimax import minimax_move
from connect4.net import Net


def play_one(args):
    model, sims, opponent, depth, ai_first, seed = args
    net = Net.load(model)
    rnd = random.Random(seed)
    st = State()
    ai_turn = ai_first
    # two random opening plies so the games differ
    for _ in range(2):
        st, _ = st.play(rnd.choice(st.legal_moves()))
        ai_turn = not ai_turn
    while True:
        if ai_turn:
            a = best_move(search(net, st, sims)) if sims > 0 else int(np.argmax(
                np.where([st.can_play(c) for c in range(7)], net.predict(st.encode()[None])[0][0], -1)))
        elif opponent == "random":
            a = rnd.choice(st.legal_moves())
        else:
            a = minimax_move(st, depth, rnd)
        st, won = st.play(a)
        if won:
            return 1 if ai_turn else -1
        if st.is_full():
            return 0
        ai_turn = not ai_turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/latest.npz")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--opponents", default="random,minimax2,minimax4,minimax6")
    args = ap.parse_args()
    with mp.Pool() as pool:
        for opp in args.opponents.split(","):
            depth = int(opp[7:]) if opp.startswith("minimax") else 0
            kind = "minimax" if depth else "random"
            jobs = [(args.model, args.sims, kind, depth, i % 2 == 0, i) for i in range(args.games)]
            res = pool.map(play_one, jobs)
            w, d, l = res.count(1), res.count(0), res.count(-1)
            print(f"vs {opp:9s}: wins {w:3d}  draws {d:3d}  losses {l:3d}", flush=True)


if __name__ == "__main__":
    main()
