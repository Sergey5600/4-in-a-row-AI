"""AlphaZero-style training: self-play with MCTS -> train network -> repeat.

Usage: python train.py --minutes 60
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
import multiprocessing as mp

import numpy as np

from connect4.net import Net
from connect4.selfplay import play_games

MODEL_DIR = "models"


def worker(args):
    weights, net_shape, n_games, sims, seed = args
    net = Net(hidden=net_shape[0], value_hidden=net_shape[1])
    net.set_weights(weights)
    rng = np.random.default_rng(seed)
    return play_games(net, n_games, sims, rng)


def lr_for(it):
    if it < 40:
        return 1e-3
    if it < 80:
        return 3e-4
    return 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--games", type=int, default=600, help="self-play games per iteration")
    ap.add_argument("--sims", type=int, default=120, help="MCTS simulations per move")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--reuse", type=float, default=4.0, help="how many times each new sample is trained on")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt = os.path.join(MODEL_DIR, "latest.npz")
    log_path = os.path.join(MODEL_DIR, "train_log.jsonl")
    if args.resume and os.path.exists(ckpt):
        net = Net.load(ckpt)
        state = json.load(open(os.path.join(MODEL_DIR, "state.json")))
        start_it, total_games = state["iteration"] + 1, state["games"]
        print(f"resumed from iteration {start_it - 1}")
    else:
        net = Net()
        start_it, total_games = 0, 0
    shape = (net.hidden, net.value_hidden)

    buf_x, buf_pi, buf_z = [], [], []
    rng = np.random.default_rng(int(time.time()))
    t_end = time.time() + args.minutes * 60
    pool = mp.Pool(args.workers)
    it = start_it
    while time.time() < t_end:
        t0 = time.time()
        per = [args.games // args.workers + (1 if i < args.games % args.workers else 0)
               for i in range(args.workers)]
        w = net.get_weights()
        jobs = [(w, shape, n, args.sims, int(rng.integers(1 << 31))) for n in per]
        results = pool.map(worker, jobs)
        n_new, outcomes = 0, []
        for X, PI, Z, res in results:
            buf_x.append(X); buf_pi.append(PI); buf_z.append(Z)
            n_new += len(X); outcomes += res
        total_games += args.games
        # keep only the newest samples
        while sum(len(x) for x in buf_x) - len(buf_x[0]) > args.buffer:
            buf_x.pop(0); buf_pi.pop(0); buf_z.pop(0)
        t_play = time.time() - t0

        X = np.concatenate(buf_x); PI = np.concatenate(buf_pi); Z = np.concatenate(buf_z)
        steps = max(1, int(n_new * args.reuse / args.batch))
        lr = lr_for(it)
        pl, vl = [], []
        for _ in range(steps):
            idx = rng.integers(0, len(X), args.batch)
            a, b = net.train_step(X[idx], PI[idx], Z[idx], lr=lr)
            pl.append(a); vl.append(b)
        t_train = time.time() - t0 - t_play

        net.save(ckpt)
        if it % 10 == 0:
            net.save(os.path.join(MODEL_DIR, f"iter_{it:03d}.npz"))
        net.export_js(os.path.join("web", "weights.js"),
                      meta={"iteration": it, "games": total_games})
        json.dump({"iteration": it, "games": total_games},
                  open(os.path.join(MODEL_DIR, "state.json"), "w"))
        outcomes = np.array(outcomes)
        rec = {"iter": it, "games": total_games, "buffer": int(len(X)), "steps": steps, "lr": lr,
               "policy_loss": round(float(np.mean(pl)), 4), "value_loss": round(float(np.mean(vl)), 4),
               "first_wins": int((outcomes == 1).sum()), "second_wins": int((outcomes == -1).sum()),
               "draws": int((outcomes == 0).sum()),
               "avg_len": round(n_new / 2 / args.games, 1),
               "t_play": round(t_play, 1), "t_train": round(t_train, 1)}
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        it += 1
    pool.close()


if __name__ == "__main__":
    main()
