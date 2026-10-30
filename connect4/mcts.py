"""PUCT Monte-Carlo tree search guided by the network (AlphaZero style).

Values are always from the side of the player to move at the node.
Edge statistics N/W are stored in the parent, from the parent's side.
"""
import math
import numpy as np

from .game import State, WIDTH, mirror_encoding

C_PUCT = 1.5


class Node:
    __slots__ = ("state", "terminal", "tvalue", "expanded", "P", "N", "W", "n_total",
                 "children", "legal")

    def __init__(self, state, terminal=False, tvalue=0.0):
        self.state = state
        self.terminal = terminal
        self.tvalue = tvalue
        self.expanded = False
        self.children = {}
        self.n_total = 0

    def expand(self, priors):
        self.legal = self.state.legal_moves()
        total = sum(priors[a] for a in self.legal)
        if total <= 0:
            self.P = [1.0 / len(self.legal) if a in self.legal else 0.0 for a in range(WIDTH)]
        else:
            self.P = [priors[a] / total if a in self.legal else 0.0 for a in range(WIDTH)]
        self.N = [0] * WIDTH
        self.W = [0.0] * WIDTH
        self.expanded = True

    def add_noise(self, rng, alpha=1.0, eps=0.25):
        noise = rng.dirichlet([alpha] * len(self.legal))
        for a, n in zip(self.legal, noise):
            self.P[a] = (1 - eps) * self.P[a] + eps * n

    def select(self):
        sq = math.sqrt(self.n_total + 1)
        best, best_score = -1, -1e9
        for a in self.legal:
            n = self.N[a]
            q = self.W[a] / n if n else 0.0
            score = q + C_PUCT * self.P[a] * sq / (1 + n)
            if score > best_score:
                best, best_score = a, score
        return best

    def child(self, a):
        ch = self.children.get(a)
        if ch is None:
            st, won = self.state.play(a)
            if won:
                ch = Node(st, True, -1.0)   # player to move there has lost
            elif st.is_full():
                ch = Node(st, True, 0.0)
            else:
                ch = Node(st)
            self.children[a] = ch
        return ch

    def q(self):
        if self.n_total == 0:
            return 0.0
        return sum(self.W) / self.n_total


def descend(root):
    """Walk to a leaf. Returns (leaf, path) where path is a list of (node, action)."""
    node, path = root, []
    while node.expanded and not node.terminal:
        a = node.select()
        path.append((node, a))
        node = node.child(a)
    return node, path


def backup(path, v):
    for node, a in reversed(path):
        v = -v
        node.N[a] += 1
        node.W[a] += v
        node.n_total += 1


def evaluate(net, states, rng=None):
    """Batch network evaluation; randomly mirrors positions for variety."""
    x = np.stack([s.encode() for s in states])
    flip = None
    if rng is not None:
        flip = rng.random(len(states)) < 0.5
        for i in np.nonzero(flip)[0]:
            x[i] = mirror_encoding(x[i])
    probs, values = net.predict(x)
    if flip is not None:
        probs[flip] = probs[flip][:, ::-1]
    return probs, values


def run_searches(net, roots, sims, rng=None):
    """Run `sims` simulations on each root, evaluating leaves in one batch per round."""
    pending = [r for r in roots if not r.expanded and not r.terminal]
    if pending:
        probs, _ = evaluate(net, [r.state for r in pending], rng)
        for r, p in zip(pending, probs):
            r.expand(p)
    for _ in range(sims):
        leaves = []
        for r in roots:
            if r.terminal:
                continue
            leaf, path = descend(r)
            if leaf.terminal:
                backup(path, leaf.tvalue)
            else:
                leaves.append((leaf, path))
        if not leaves:
            continue
        probs, values = evaluate(net, [l.state for l, _ in leaves], rng)
        for (leaf, path), p, v in zip(leaves, probs, values):
            leaf.expand(p)
            backup(path, float(v))


def search(net, state, sims, rng=None):
    """Single-position search. Returns the root node."""
    root = Node(state)
    run_searches(net, [root], sims, rng)
    return root


def best_move(root):
    return max(root.legal, key=lambda a: (root.N[a], root.P[a]))
