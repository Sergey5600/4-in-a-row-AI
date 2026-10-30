"""Play Connect Four against the trained AI in the terminal.

Usage: python play.py [--sims 800] [--second]
"""
import argparse

from connect4.game import State, HEIGHT, WIDTH
from connect4.mcts import search, best_move
from connect4.net import Net


def show(st, human_first):
    grid = st.to_grid()  # 1 = side to move, 2 = other side
    human_to_move = (st.moves % 2 == 0) == human_first
    sym = {0: ".", 1: "X" if human_to_move else "O", 2: "O" if human_to_move else "X"}
    print()
    for r in reversed(range(HEIGHT)):
        print(" " + " ".join(sym[int(v)] for v in grid[r]))
    print(" " + " ".join(str(c + 1) for c in range(WIDTH)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/latest.npz")
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--second", action="store_true", help="let the AI move first")
    args = ap.parse_args()
    net = Net.load(args.model)
    human_first = not args.second
    st = State()
    print("Вы — X, ИИ — O. Введите номер столбца 1-7.")
    while True:
        show(st, human_first)
        human_turn = (st.moves % 2 == 0) == human_first
        if human_turn:
            try:
                c = int(input("Ваш ход: ")) - 1
            except ValueError:
                continue
            if not 0 <= c < WIDTH or not st.can_play(c):
                print("Этот столбец недоступен.")
                continue
        else:
            root = search(net, st, args.sims)
            c = best_move(root)
            print(f"ИИ ходит в {c + 1} (оценка ИИ: {(1 + root.W[c] / root.N[c]) / 2:.0%} в его пользу)")
        st, won = st.play(c)
        if won:
            show(st, human_first)
            print("Вы победили!" if human_turn else "ИИ победил.")
            return
        if st.is_full():
            show(st, human_first)
            print("Ничья.")
            return


if __name__ == "__main__":
    main()
