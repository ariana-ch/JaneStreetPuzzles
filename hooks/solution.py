"""
hooks.py

Solver for the nested-hook puzzle (9x9) used in the accompanying notebook.

The puzzle places numbers 1..9 in hooks (concentric L-shaped regions) so that
- the outermost hook contains nine 9's, the next contains eight 8's, ..., down to one 1
- each row and column has a required sum (given by ROW_SUMS and COL_SUMS)

This module exposes helper functions to construct valid candidate combinations
for each hook, propagate forced placements into a values grid, complete the
grid by considering diagonal cells, and validate/iterate to find the unique
solution.

Public functions:
- column(arr, idx): return the idx-th column from a 2D list.
- combination_sum(options, target): return combinations (as tuples) of elements
  from `options` that sum to `target` subject to non-decreasing ordering.
- valid_options(): generate all valid (column, row) option pairs for each hook
  consistent with the outer-to-inner construction and immediate constraint
  propagation; returns (options, values) where `values` contains forced cells
  (0 for empty, n for assigned) and -1 for undecided.
- complete(options, values): given a selection of (column, row) option pairs
  for each hook, fill in non-diagonal cells and enumerate possibilities for
  diagonal entries when multiple placements are possible.
- validate(values): verify a fully-specified grid meets the row/column sums.
- iterate(options, values): brute-force over combinations of options (one
  choice per hook) and return the first valid completed grid.
"""

from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product
from typing import Sequence

COL_SUMS = [31, 19, 45, 16, 5, 47, 28, 49, 45]
ROW_SUMS = [26, 42, 11, 22, 42, 36, 29, 32, 45]


def column(arr: list, idx: int) -> list:
    """Return the idx-th column from a 2D list `arr`.

    Args:
        arr (list): 2D list (list of rows) representing the square grid.
        idx (int): column index in range 0..8.

    Returns:
        list: column values (one element per row).

    Raises:
        AssertionError: if idx is out of range.
    """
    assert 0 <= idx <= 8, 'There are only 9 columns'
    return [x[idx] for x in arr]


def combination_sum(options: Sequence[int], target: int) -> set:
    """Enumerate unique combinations of elements from `options` whose sum is `target`.

    This routine treats the elements in `options` as distinct choices but
    returns combinations in non-decreasing order (so repeated values from
    `options` are considered distinct if they come from different positions).

    The algorithm is a constrained search that grows candidate "root" lists and
    prunes branches where the partial sum exceeds `target` or even taking all
    remaining elements cannot reach `target`.

    Args:
        options (Sequence[int]): list of integers (may contain duplicates).
        target (int): target sum for combinations.

    Returns:
        set: a set of tuples; each tuple is a non-decreasing sequence of
             integers from `options` that sums to `target`.

    Notes:
        - If `options` is empty and `target` is 0, this returns a set containing
          the empty tuple; if `options` is empty and `target` != 0 it returns
          an empty set-like structure.
    """
    # sort the options in ascending order
    if not options:
        return set((), )
    if sum(options) == target:
        return {options}
    options = sorted(options)
    found = set()
    combos = [[options[i:i + 1], options[i + 1:]] for i in range(len(options))]
    while combos:
        x = combos[0]
        combos.remove(x)
        root, tail = x
        if not root:
            raise ValueError
        if sum(root) == target:
            found.add(tuple(root))
            continue
        root_sum = sum(root)
        tail_sum = sum(tail)

        if root_sum > target or tail_sum + root_sum < target:
            continue
        for y in tail:
            if root_sum + y <= target and y >= root[-1]:
                root_ = root.copy()
                root_.append(y)
                tail_ = tail.copy()
                tail_.remove(y)
                combos.append([root_, tail_])
    return found


def valid_options():
    """Generate possible (column, row) option pairs for each hook index.

    The hooks are considered from outermost (9) down to innermost (1). For
    each hook i (0-based loop index but representing value i+1), the function
    computes all possible multiset choices for the intersection of the hook's
    row and column that satisfy the required row and column sums, while
    propagating immediate forced placements into the `values` matrix.

    Returns:
        tuple:
            - options (dict): mapping hook_value -> list of (col_list, row_list)
              pairs (each list contains the values assigned in that hook's
              column/row section). Hook keys are integers 1..9.
            - values (list): 9x9 matrix with forced entries:
                -1 = undecided, 0 = known empty (not this value), n = cell fixed to n

    Implementation notes:
        - iterates hooks in reverse (from outer to inner) because outer hooks
          constrain inner hooks via row/column sums and forced placements.
        - uses `combination_sum` to enumerate multisets that reach the required
          partial sums after accounting for the mandatory single occurrence of
          the hook value itself.
    """
    options = dict()
    # instantiate the NxN square with -1
    values = [[-1 for _ in range(len(COL_SUMS))] for _ in range(len(COL_SUMS))]

    for i in range(len(COL_SUMS) - 1, -1, -1):
        pairs = []
        # all rows/columns will contain **at least** the row number (i+1) once to satisfy the conditions
        column_options = defaultdict(list)
        row_required = Counter([x for x in values[i] if x > 0])
        col_required = Counter([x for x in column(values, i) if x > 0])

        combinations = combination_sum(options=[i + 1 for _ in range(i + 1)] + list(range(i + 2, 10)),
                                       target=COL_SUMS[i] - (i + 1))
        if not combinations:
            column_options[1].append([i + 1])

        for entry in combinations:
            entry = list(entry)
            entry.insert(0, i + 1)
            assert sum(entry) == COL_SUMS[i]
            m = Counter(entry)
            if not all(col_required[j] <= m[j] for j in col_required):
                continue
            assert m.get(i + 1) >= 1
            column_options[m.get(i + 1)].append(entry)

        for entry in combination_sum(options=[i + 1] * i + list(range(i + 2, 10)), target=ROW_SUMS[i] - (i + 1)):
            entry = list(entry)
            entry.insert(0, i + 1)
            assert sum(entry) == ROW_SUMS[i]
            m = Counter(entry)
            if not all(row_required[j] <= m[j] for j in col_required):
                continue
            m = m.get(i + 1)
            assert m >= 1
            assert i + 1 - m >= 0

            pairs.extend(product(column_options[i + 1 - m] + column_options[i + 1 - m + 1], [entry]))

        for c_, r_ in pairs:
            assert sum(c_) == COL_SUMS[i], f'{c_} != {COL_SUMS[i]} for {i}'
            assert sum(r_) == ROW_SUMS[i], f'{r_} != {ROW_SUMS[i]} for {i}'

        # update diagonal if applicable
        # update values row number + 1/ col number + 1
        col_counter = Counter([x for y in pairs for x in y[0]])
        row_counter = Counter([x for y in pairs for x in y[1]])
        for j, c in col_counter.items():
            if c == len(pairs) and (j != i + 1):
                values[j - 1][i] = j
            if c == len(pairs) * (i + 1):
                assert i == j - 1 or i == 0
                for k in range(i + 1):
                    values[k][i] = j
        for j, c in row_counter.items():
            if c == len(pairs) and (j != i + 1):
                values[i][j - 1] = j
            if c == len(pairs) * (i + 1):
                assert i == j - 1 or i == 0
                for k in range(i + 1):
                    values[i][k] = j
        for j in set(range(i + 1, 10, 1)).difference(row_counter):
            values[i][j - 1] = 0
        for j in set(range(i + 1, 10, 1)).difference(col_counter):
            values[j - 1][i] = 0

        options[i + 1] = pairs
    options = dict(sorted(options.items(), key=lambda x: x[0]))
    return options, values


def complete(options: dict, values: list):
    """Given fixed option pairs for each hook, produce completed grids.

    For the non-diagonal part of each hook the presence/absence of a value is
    deterministic once a particular (column, row) pair is chosen for that hook.
    The diagonal entries (the overlapping L-shape intersection cells) can be
    ambiguous; this function enumerates all consistent ways to fill those
    diagonal cells (using combinations) and returns a list of completed grids
    (or partial grids) for further validation.

    Args:
        options (dict): mapping hook_value -> (col_list, row_list) pair(s). This
                        function expects a dictionary where each key maps to a
                        single (col,row) pair (e.g., after selecting one
                        candidate per hook).
        values (list): base values matrix with forced cells (0/-1/n) that will
                       be copied and completed.

    Returns:
        list: a list of 9x9 grids (lists of lists) consistent with the supplied
              options. If impossible, returns an empty list.
    """
    from itertools import combinations
    all_options = []
    values = deepcopy(values)
    n = list(range(1, 10))
    for i, (col, row) in options.items():
        for j in n[i:]:
            if j in row:
                values[i - 1][j - 1] = j
            else:
                values[i - 1][j - 1] = 0
            if j in col:
                values[j - 1][i - 1] = j
            else:
                values[j - 1][i - 1] = 0

    # handle diagonal entries last
    for i in range(1, 10, 1):
        col = column(values, i - 1)
        n_row = len([x for x in values[i - 1][:i] if x == i])
        n_col = len([x for x in col[:i - 1] if x == i])
        missing = i - n_col - n_row
        empty = len(
            [x for x in values[i - 1][:i] + col[:i - 1] if x == -1])  # the i-1 is so we don't double count the diagonal
        if missing > empty or missing < 0:
            return []
        if missing == empty:
            for j in range(i):
                if values[i - 1][j] == -1:
                    values[i - 1][j] = i
                if col[j] == -1:
                    values[j][i - 1] = i
        elif empty > 0 and missing == 0:
            for j in range(i):
                if values[i - 1][j] == -1:
                    values[i - 1][j] = 0
                if col[j] == -1:
                    values[j][i - 1] = 0
        elif empty > missing != 0:
            empty_cells = [(i - 1, j) for j in range(i) if values[i - 1][j] == -1] + [(i, j) for j in range(i - 1) if
                                                                                      col[j] == -1]
            combos = combinations(empty_cells, missing)
            for combo in combos:
                values_ = deepcopy(values)
                for i_, j_ in combo:
                    values_[i_][j_] = i
                zero_cells = [x for x in empty_cells if x not in combo]
                for i_, j_ in zero_cells:
                    values_[i_][j_] = 0
                all_options.append(values_)
    if not all_options:
        all_options.append(values)
    return all_options


def validate(values: list):
    """Validate that a fully-filled grid satisfies the required row/column sums.

    Args:
        values (list): 9x9 matrix with no -1 entries.

    Returns:
        bool: True if valid, False otherwise.
    """
    if any([x == -1 for y in values for x in y]):
        return False
    col = [31, 19, 45, 16, 5, 47, 28, 49, 45]
    row = [26, 42, 11, 22, 42, 36, 29, 32, 45]
    for i, x in enumerate(values):
        if sum(x) != row[i]:
            return False
    for i, x in enumerate(col):
        if sum(column(values, i)) != x:
            return False
    return True


def iterate(options: dict, values: list):
    """Try combinations of options (one choice per hook) and return a valid grid.

    This function forms the Cartesian product over the lists of candidate
    pairs in `options` and for each tuple of choices uses `complete` to
    generate concrete grids, returning the first grid that passes `validate`.

    Args:
        options (dict): mapping hook_value -> list of (col_list, row_list) pairs
        values (list): base values matrix (with -1, 0, or fixed numbers)

    Returns:
        list or None: a valid completed 9x9 grid if found, otherwise None.
    """
    combinations = product(*list(options.copy().values()))
    for combination in combinations:
        d = dict(zip(options.keys(), combination))
        v = complete(options=d, values=values)
        for i in v:
            if validate(i):
                return i


if __name__ == '__main__':
    from pprint import pp

    options, values = valid_options()
    pp(iterate(options, values))
    # print(options)
