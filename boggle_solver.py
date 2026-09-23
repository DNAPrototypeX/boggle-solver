from types import SimpleNamespace
import numpy as np
from treelib import Tree
from itertools import zip_longest
import tqdm
import string
import curses
import os

USE_DOUBLE_LETTERS = True
rng = np.random.default_rng(seed=2)


# Returns a list of index tuples corresponding to the
# surrounding tiles on the board, accounts for edges.
def get_neighbors(idx):
    r, c = idx
    # All 8 relative (row, col) offsets
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    neighbors = []
    for dr, dc in offsets:
        nr, nc = r + dr, c + dc
        # Check boundary limits
        if 0 <= nr < 5 and 0 <= nc < 5:
            neighbors.append((nr, nc))

    return np.array(neighbors)


# Finds valid moves (nodes on the board that are not part of the path to the root):
def find_moves(tree, current_node):
    id = current_node.identifier
    idx = current_node.tag
    neighbors = get_neighbors(idx)
    path = list(tree.rsearch(id))
    path_names = [tree[nid].tag for nid in path]
    moves = []
    for a, b in neighbors:
        if (a, b) not in path_names:
            moves.append((a, b))
    return moves


# Returns the reverse of the current "word" i.e. the path
# from the root to node.
def get_node_word(tree, node):
    path = list(tree.rsearch(node.identifier))
    path_names = [game[tree[nid].tag] for nid in path]
    suffix = "".join(path_names)
    return suffix


def count_possible_words(tree, node):
    # Convert the indexes of the path to root into a string
    suffix = get_node_word(tree, node)
    # Return the number of words in the reversed dictionary that end
    # with the current "word". This is slow because the dictionary is large.
    # Easy solution: Preprocessing
    # We know all words not starting with the root node are not possible.
    # Split the dictionary by first letter of each word and only search the relevant one
    # There is certainly some more sophisticated pruning method, but this is easiest and works
    # 2:57 solve -> 0:07
    return sum(
        word.endswith(suffix) for word in dictionaries[tree[tree.root].data.letter[0]]
    )


# Builds one tree using DFS of the board. (Recursive)
def build_tree(tree, current_node):
    # recurse until no more possibe moves
    moves = find_moves(tree, current_node)
    if not count_possible_words(tree, current_node) or not moves:
        return
    for move in moves:
        next_node = tree.create_node(
            move,
            parent=current_node.identifier,
            data=SimpleNamespace(letter=game[move]),
        )
        build_tree(tree, next_node)
    return


# Builds 25 trees (one for each node), and returns a set of all possible unique words.
def solve_game(game):
    # Build one tree for each letter and initialize the root:
    trees = np.empty((5, 5), dtype=object)
    for i in range(5):
        for j in range(5):
            trees[i, j] = Tree()
            trees[i, j].create_node((i, j), data=SimpleNamespace(letter=game[i, j]))
    words = set()
    for tree in tqdm.tqdm(trees.flat, desc="Building Trees..."):
        build_tree(tree, tree[tree.root])
        for leaf in tree.leaves():
            word = get_node_word(tree, leaf)
            for i in range(len(word)):
                suffix = word[len(word) - i :]
                if suffix in dictionaries[tree[tree.root].data.letter[0]]:
                    words.add(suffix[::-1])
    return words


DIGRAPHS = {"A": "n", "E": "r", "I": "n", "Q": "u", "T": "h", "H": "e"}


# Put a fresh letter at pos. Advance unless it could start a digraph.
def place(grid, pos, key):
    size = len(grid)
    r, c = divmod(pos, size)
    letter = key.upper()
    grid[r][c] = letter
    if letter in DIGRAPHS:
        return pos  # wait: next letter may complete the pair
    return min(pos + 1, size * size - 1)


# Open a curses window for entering a size x size matrix of letters.
# Returns a size x size np.ndarray of strings, or None if cancelled.
def enter_matrix(size=5):
    os.environ.setdefault("ESCDELAY", "25")  # make Esc respond quickly
    n = size * size
    cell_w = 5  # "| Th " -> 5 columns per cell
    top, left = 2, 2

    def _run(stdscr):
        curses.curs_set(1)
        stdscr.keypad(True)
        grid = [["" for _ in range(size)] for _ in range(size)]
        pos = 0
        msg = ""

        while True:
            stdscr.erase()
            r, c = divmod(pos, size)
            try:
                stdscr.addstr(
                    0,
                    0,
                    "Tab/Shift-Tab/arrows: move   Backspace: delete   "
                    "Enter: submit   Esc: cancel",
                )
                sep = "+" + ("-" * (cell_w - 1) + "+") * size
                for i in range(size):
                    y = top + 2 * i
                    stdscr.addstr(y, left, sep)
                    for j in range(size):
                        x = left + cell_w * j
                        stdscr.addstr(y + 1, x, "|")
                        attr = curses.A_REVERSE if (i, j) == (r, c) else curses.A_NORMAL
                        stdscr.addstr(y + 1, x + 1, f" {grid[i][j]:<2} ", attr)
                    stdscr.addstr(y + 1, left + cell_w * size, "|")
                stdscr.addstr(top + 2 * size, left, sep)
                stdscr.addstr(top + 2 * size + 2, left, msg)
                stdscr.move(top + 2 * r + 1, left + cell_w * c + 2 + len(grid[r][c]))
            except curses.error:
                stdscr.erase()
                stdscr.addstr(0, 0, "Terminal too small - please enlarge it.")
            stdscr.refresh()

            try:
                key = stdscr.get_wch()
            except curses.error:
                continue
            msg = ""

            if key == "\x1b":  # Esc
                return None
            elif key in ("\n", "\r") or key == curses.KEY_ENTER:
                empty = [k for k in range(n) if not grid[k // size][k % size]]
                if not empty:
                    return np.char.upper(np.array(grid, dtype="<U2"))
                msg = f"Matrix incomplete: {len(empty)} empty cell(s)."
                pos = empty[0]
            elif key == "\t":
                pos = (pos + 1) % n
            elif key == curses.KEY_BTAB:
                pos = (pos - 1) % n
            elif key == curses.KEY_RIGHT:
                pos = (pos + 1) % n
            elif key == curses.KEY_LEFT:
                pos = (pos - 1) % n
            elif key == curses.KEY_DOWN:
                pos = (pos + size) % n
            elif key == curses.KEY_UP:
                pos = (pos - size) % n
            elif key in (curses.KEY_BACKSPACE, "\x7f", "\b", curses.KEY_DC):
                grid[r][c] = grid[r][c][:-1]
            elif isinstance(key, str) and key.isalpha() and key.isascii():
                cell = grid[r][c]
                if len(cell) == 1 and key.lower() in DIGRAPHS.get(cell, ""):
                    # Complete a digraph like Th or Qu, then move on
                    grid[r][c] = cell + key.lower()
                    pos = min(pos + 1, n - 1)
                elif len(cell) == 1 and cell in DIGRAPHS:
                    # Pending first letter not completed: this letter goes in the next cell
                    if pos == n - 1:
                        curses.beep()
                        msg = f"'{cell}{key.lower()}' is not a valid pair, and this is the last cell."
                    else:
                        pos += 1
                        pos = place(grid, pos, key)
                else:
                    # Empty cell, or overwriting a finished cell
                    pos = place(grid, pos, key)

    return curses.wrapper(_run)


if __name__ == "__main__":
    # Get board from user:
    game = enter_matrix()
    if game is None:
        exit(0)
    # Read boggle dictionary
    dictionaries = {}
    for letter in string.ascii_uppercase:
        dictionaries[letter] = np.loadtxt(
            f"dictionary/{letter}_reversed.csv", dtype=str
        )

    words = solve_game(game)
    print(f"Found {len(words)} words:")

    # Sort words into groups based on their points
    words = sorted(words, key=len)
    groups = {"4": [], "5": [], "6": [], "7": [], "8+": []}
    for w in words:
        key = "8+" if len(w) >= 8 else str(len(w))
        groups.get(key, []).append(w)

    # Sort alphabetically
    for group in groups:
        groups[group] = sorted(groups[group])

    # Display all possible words
    print("".join(f"{h + ' letters':<14}" for h in groups))
    for row in zip_longest(*groups.values(), fillvalue=""):
        print("".join(f"{w:<14}" for w in row))
