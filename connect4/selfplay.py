"""Batched self-play: many games advance together so the network sees large batches."""
import numpy as np

from .game import State, WIDTH, mirror_encoding
from .mcts import Node, run_searches


def play_games(net, n_games, sims, rng, temp_moves=12, q_mix=0.5):
    """Returns (X, PI, Z) training arrays, mirrored copies included."""
    roots = [Node(State()) for _ in range(n_games)]
    histories = [[] for _ in range(n_games)]  # (encoding, pi, root_q, player)
    results = [None] * n_games                # +1 first player won, -1 second, 0 draw
    active = list(range(n_games))

    while active:
        act_roots = [roots[i] for i in active]
        run_searches(net, [r for r in act_roots if not r.expanded], 0, rng)
        for r in act_roots:
            r.add_noise(rng)
        run_searches(net, act_roots, sims, rng)

        still = []
        for i in active:
            root = roots[i]
            counts = np.array(root.N, dtype=np.float64)
            pi = counts / counts.sum()
            player = root.state.moves % 2
            histories[i].append((root.state.encode(), pi.astype(np.float32), root.q(), player))
            if root.state.moves < temp_moves:
                a = int(rng.choice(WIDTH, p=pi))
            else:
                best = np.flatnonzero(counts == counts.max())
                a = int(rng.choice(best))
            child = root.child(a)
            if child.terminal:
                if child.tvalue < 0:
                    results[i] = 1 if player == 0 else -1
                else:
                    results[i] = 0
            else:
                roots[i] = child  # reuse the subtree
                still.append(i)
        active = still

    X, PI, Z = [], [], []
    for hist, res in zip(histories, results):
        for enc, pi, q, player in hist:
            z = res if player == 0 else -res
            target = (1 - q_mix) * z + q_mix * q
            X.append(enc); PI.append(pi); Z.append(target)
            X.append(mirror_encoding(enc)); PI.append(pi[::-1].copy()); Z.append(target)
    return (np.array(X, np.float32), np.array(PI, np.float32), np.array(Z, np.float32),
            results)
