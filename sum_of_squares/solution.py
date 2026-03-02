import heapq
from typing import List, Tuple, Union, Sequence, Generator, Set

import math

Number = Union[int, float]

def is_valid(x: List[Number]) -> bool:
    """
    Check if a combination of numbers is a valid solution
    to the problem

    Args:
        x (Tuple[Number]): a tuple of numbers representing a
        flatten solution
    Returns:
        bool: True if x is valid solution, False otherwise
    """
    x = list(x) # creates a copy on the heap
    # Row 1 divisible by 2
    if x[9] % 2 != 0:
        return False
    # Row 2 divisible by 3
    if sum(x[5: 10]) % 3 != 0:
        return False
    # Row 3 divisible by 4
    if (x[18] * 10 + x[19]) % 4 != 0:
        return False
    # Row 4 divisible by 5/Column 4 divisible by 10
    if x[24] != 0:
        return False
    # Column 0 divisible by 6
    if sum(x[::5]) % 3 != 0 or (x[20] % 2 != 0):
        return False
    # Column 1 divisible by 7 (use digit 1 * (10^4 % 7) + ...) % 7
    if (x[1] * 4 + x[6] * 6 + x[11] * 2 + x[16] * 3 + x[21]) % 7 != 0:
        return False
    # Column 2 divisible by 8 - use same approach as for 7
    if (x[12] * 2 + x[17] * 3 + x[22]) % 8 != 0:
        return False
    # Column 3 divisible by 9:
    if sum(x[3::5]) % 9 != 0:
        return False
    return True


def get_neighbors(x: List[Number], multiplicities: List[int]) -> List[List[Number]]:
    """
    Args:
        x (Tuple[Number]): a tuple of numbers representing a
        flatten solution input
        multiplicities (List[int]): multiplicities of each
        element of x
    Returns:
        List[Tuple[Number]]: list of all possible solution
        combinations whose sum is equal to the sum of x - 1
    """
    neighbors = []
    for i, m in enumerate(multiplicities):
        if x[i] < m - 1:
            x_ = list(x)
            x_[i] += 1
            neighbors.append(tuple(x_))
    return neighbors


def encode(x: List[Number], multiplicities: List[int]) -> int:
    """
    Encode a tuple of numbers representing a flatten solution
    into a radix encoding

    Args:
        x (List[Number]): a list of numbers representing a
        flatten solution
        multiplicities (List[int]): The multiplicities of
        each element of the tuple
    Returns:
        int: The radix encoding
    """
    id_ = 0
    multiplicities = multiplicities.copy()
    multiplicities.insert(0, 1)
    multiplicities.pop(-1)
    for i in range(len(x) - 1, -1, -1):
        id_ += x[i]
        id_ *= multiplicities[i]
    return id_

def decode(id_: int, multiplicities: List[int]) -> List[Number]:
    """
    Decode a radix encoding of a flatten solution
    into a tuple of numbers representing a flatten solution
    Args:
        id_ (int): the id of the solution
        multiplicities (List[int]): The multiplicities of
        each element of the tuple
    Returns:
        List[Number]: the list of numbers representing
        a flatten solution
    """
    x = []
    multiplicities = multiplicities.copy()
    multiplicities.insert(0, 1)
    multiplicities.pop(-1)
    for i in range(len(multiplicities)):
        m = math.prod(multiplicities[:len(multiplicities) - i])
        x.insert(0, id_ // m)
        id_ %= m
    return x


def ordered_combinations(options: Sequence[Sequence[int]]) -> Generator[List[int]]:
    options:        List[List[int]] = [sorted(x, reverse=True) for x in options]
    multiplicities: List[int] = [len(x) for x in options]
    start_sum:      int = sum(x[0] for x in options)
    visited:        Set[int] = {0}
    queue: List[Tuple[Number, ...]] = [(-start_sum, 0)]

    while queue:
        neg_sum, id_ = heapq.heappop(queue) # id_ is the encoded indices
        indices = decode(id_, multiplicities)
        combination = [options[i][j] for i, j in enumerate(indices)]
        visited.remove(id_)
        yield combination

        for i, m in enumerate(multiplicities):
            if indices[i] < m - 1:
                new_id = id_ + math.prod(multiplicities[:i])
                if new_id not in visited:
                    visited.add(new_id)
                    new_neg_sum = neg_sum + options[i][indices[i]] - options[i][indices[i] + 1]
                    heapq.heappush(queue, (new_neg_sum, new_id))


def solve() -> List[List[int]]:
    options_dict = {(i, j): list(range(10)) for i in range(5) for j in range(5)}
    fixed = {(4, 4): [0],
             (1, 4): [0, 2, 4, 6, 8],
             (3, 4): [0, 2, 4, 6, 8],
             (4, 0): [0, 2, 4, 6, 8],
             (4, 2): [0, 2, 4, 6, 8]}
    options_dict.update(fixed)
    options = list(options_dict.values())
    combinations = ordered_combinations(options)
    for combination in combinations:
        if is_valid(combination):
            return [combination[i:i+5] for i in range(0, 25, 5)]
    return [[]]

if __name__ == "__main__":
    print(solve())