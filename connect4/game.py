"""Connect Four rules on bitboards.

Each column takes 7 bits (6 cells + 1 sentinel bit), so 49 bits in total.
A position is stored from the side of the player to move:
  cur  - stones of the player to move
  mask - all stones on the board
"""
import numpy as np

WIDTH, HEIGHT = 7, 6
H1 = HEIGHT + 1
N_CELLS = WIDTH * HEIGHT

BOTTOM = [1 << (c * H1) for c in range(WIDTH)]
TOP = [1 << (c * H1 + HEIGHT - 1) for c in range(WIDTH)]
# Cell bits in row-major order: row 0 is the bottom row.
CELL_BITS = np.array([c * H1 + r for r in range(HEIGHT) for c in range(WIDTH)])
CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]


def has_four(p):
    for s in (1, H1, H1 - 1, H1 + 1):  # vertical, horizontal, two diagonals
        m = p & (p >> s)
        if m & (m >> (2 * s)):
            return True
    return False


def int_to_cells(x):
    raw = np.frombuffer(x.to_bytes(8, "little"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")[CELL_BITS]


class State:
    __slots__ = ("cur", "mask", "moves")

    def __init__(self, cur=0, mask=0, moves=0):
        self.cur = cur
        self.mask = mask
        self.moves = moves

    def can_play(self, c):
        return not (self.mask & TOP[c])

    def legal_moves(self):
        return [c for c in range(WIDTH) if not (self.mask & TOP[c])]

    def play(self, c):
        """Return (next_state, mover_won)."""
        new_mask = self.mask | (self.mask + BOTTOM[c])
        mover = self.cur | (new_mask ^ self.mask)
        return State(mover ^ new_mask, new_mask, self.moves + 1), has_four(mover)

    def is_full(self):
        return self.moves >= N_CELLS

    def encode(self):
        """Network input: 84 floats (own stones, opponent stones), row-major 6x7."""
        own = int_to_cells(self.cur)
        opp = int_to_cells(self.mask ^ self.cur)
        return np.concatenate([own, opp]).astype(np.float32)

    def to_grid(self):
        """6x7 grid (row 0 = bottom): 1 = player to move, 2 = opponent, 0 = empty."""
        own = int_to_cells(self.cur).reshape(HEIGHT, WIDTH)
        opp = int_to_cells(self.mask ^ self.cur).reshape(HEIGHT, WIDTH)
        return own + 2 * opp


def mirror_encoding(x):
    """Flip an 84-vector encoding left-right."""
    return x.reshape(2, HEIGHT, WIDTH)[:, :, ::-1].reshape(-1).copy()
