import numpy as np
from typing import Tuple

"""
A Numpy implementation of the Asymmetric Greedy Search (AGS) algorithm 
described in Peter Brown et al. "A heuristic for the time constrained asymmetric linear sum 
assignment problem" - doi:10.1007/s10878-015-9979-2
"""


def _initial(benefit_matrix: np.ndarray, shuffle: bool = False) -> np.ndarray:
    """
    Initialize the assignment solution array by assigning each row to an unassigned
    column with the maximum benefit.

    Args:
    - benefit_matrix (np.ndarray): A 2D array of benefit values.
    - shuffle (bool, optional): If True, randomly shuffle the order of the rows
                                  prior to the initial assignment. Defaults to False.

    Returns:
    - np.ndarray: A 1D array of row assignments.
    """
    bm = benefit_matrix.copy()
    assignment = np.empty((bm.shape[0]), dtype=np.int64)
    rows = np.arange(bm.shape[0])
    if shuffle:
        np.random.shuffle(rows)
    for n in rows:
        max_idx = np.argmax(bm[n, :])
        assignment[n] = max_idx
        bm[:, max_idx] = np.NINF

    return assignment


def _row_swap_cost(benefit_matrix: np.ndarray, assignment: np.ndarray,
                   row_idx: int) -> Tuple[int, float]:
    """
    Calculate the costs of swapping column assignments for a given row with all
    other rows and return the swap with the greatest benefit.

    Args:
    - benefit_matrix (np.ndarray): A 2D array of benefit values.
    - assignment (np.ndarray): A 1D array of column assignments.
    - row_idx (int): The row index on which to calculate swap costs.

    Returns:
    - Tuple[int, float]: A tuple of the best swap row and the associated benefit.
    """
    swap_cost = benefit_matrix[row_idx, assignment] + \
                benefit_matrix[:, assignment[row_idx]]
    curr_cost = benefit_matrix[row_idx, assignment[row_idx]] + \
                benefit_matrix[np.arange(benefit_matrix.shape[0]), assignment]
    cost = swap_cost - curr_cost
    cost[row_idx] = np.NINF
    best_row = np.argmax(cost)
    best_row_benefit = cost[best_row]
    return best_row, best_row_benefit


def _best_row_swap(benefit_matrix: np.ndarray,
                   assignment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine the best row swap for all rows.

    Args:
    - benefit_matrix (np.ndarray): A 2D array of benefit values.
    - assignment (np.ndarray): A 1D array of column assignments.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple of arrays for best swap row and
                                      the associated benefits.
    """
    best_row, best_row_benefit = np.stack(
        [_row_swap_cost(benefit_matrix, assignment, r) for r in
         np.arange(assignment.shape[0])]).T
    best_row = best_row.astype(np.int64)
    return best_row, best_row_benefit


def _col_swap_cost(benefit_matrix: np.ndarray, assignment: np.ndarray,
                   row_idx: int) -> Tuple[int, float]:
    """
    Calculate the cost of swapping column assignments for a given row to unassigned
    columns and return the best column and the associated benefit.

    Args:
    - benefit_matrix (np.ndarray): A 2D array of benefit values.
    - assignment (np.ndarray): A 1D array of column assignments.
    - row_idx (int): The row index on which to calculate swap costs.

    Returns:
    - Tuple[int, float]: A tuple of the best swap column and the associated benefit.
    """
    valid_idx = np.delete(np.arange(benefit_matrix.shape[1]), assignment)
    best_col = valid_idx[benefit_matrix[row_idx, valid_idx].argmax()]
    best_col_benefit = benefit_matrix[row_idx, best_col]
    return best_col, best_col_benefit


def _best_col_swap(benefit_matrix: np.ndarray,
                   assignment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine the best unassigned column swap for all rows.

    Args:
    - benefit_matrix (np.ndarray): A 2D array of benefit values.
    - assignment (np.ndarray): A 1D array of column assignments.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple of arrays for best unassigned columns
                                      and the associated benefits.
    """
    row_idx = np.arange(assignment.shape[0])
    bm_unused = benefit_matrix.copy()
    bm_unused[:, assignment] = np.NINF
    best_col = np.argmax(bm_unused, axis=1)
    best_col_benefit = bm_unused[row_idx, best_col]
    return best_col, best_col_benefit


def _row_swap(benefit_matrix: np.ndarray, assignment: np.ndarray, best_row: np.ndarray,
              br_benefit: np.ndarray, best_col: np.ndarray, bc_benefit: np.ndarray,
              r_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Swap columns assignments of given row with the best option and then update
    swap benefit matrices.

    Args:
    - benefit_matrix (np.ndarray): a 2d array of benefit values.
    - assignment (np.ndarray): a 1d array of column assignments.
    - best_row (np.ndarray): a 1d array of the best row swap for each row.
    - br_benefit (np.ndarray): a 1d array of the benefit associated with the best row swap.
    - best_col (np.ndarray): a 1d array of the best unassigned column for each row.
    - bc_benefit (np.ndarray): a 1d array of the benefit associated with the best column.
    - r_idx (int): Index of the row to swap.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple of updated assignment
                                                                          and benefit/swap matrices.
    """
    rs_idx = best_row[r_idx]
    # switch assignments
    assignment[[r_idx, rs_idx]] = assignment[[rs_idx, r_idx]]
    # update row swap matrices
    for idx in (r_idx, rs_idx):
        new_row, new_benefit = _row_swap_cost(benefit_matrix, assignment, idx)
        best_row[idx] = new_row
        br_benefit[idx] = new_benefit
    # update the column assignment matrices
    for idx in (r_idx, rs_idx):
        new_row, new_benefit = _col_swap_cost(benefit_matrix, assignment, idx)
        best_col[idx] = new_row
        bc_benefit[idx] = new_benefit
    return assignment, best_row, br_benefit, best_col, bc_benefit


def _col_swap(benefit_matrix: np.ndarray, assignment: np.ndarray, best_row: np.ndarray,
              br_benefit: np.ndarray, best_col: np.ndarray, bc_benefit: np.ndarray,
              r_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Swap columns assignment of given row with the best option unassigned column
    and then update swap benefit matrices.

    Args:
    - benefit_matrix (np.ndarray): a 2d array of benefit values.
    - assignment (np.ndarray): a 1d array of column assignments.
    - best_row (np.ndarray): a 1d array of the best row swap for each row.
    - br_benefit (np.ndarray): a 1d array of the benefit associated with the best row swap.
    - best_col (np.ndarray): a 1d array of the best unassigned column for each row.
    - bc_benefit (np.ndarray): a 1d array of the benefit associated with the best column.
    - r_idx (int): Index of the row to swap.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple of updated assignment
                                                                          and benefit/swap matrices.
    """
    assignment[r_idx] = best_col[r_idx]
    # update best row (benefit)
    new_row, new_benefit = _row_swap_cost(benefit_matrix, assignment, r_idx)
    best_row[r_idx] = new_row
    br_benefit[r_idx] = new_benefit
    # update best column (benefit)
    new_row, new_benefit = _col_swap_cost(benefit_matrix, assignment, r_idx)
    best_col[r_idx] = new_row
    bc_benefit[r_idx] = new_benefit
    return assignment, best_row, br_benefit, best_col, bc_benefit


def asymmetric_greedy_search(benefit_matrix: np.ndarray, shuffle: bool = False,
                             minimize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    A Python implementation of the Asymmetric Greedy Search (AGS) algorithm. This
    algorithm finds the optimal assignment based on the given benefit matrix.

    Args:
    - benefit_matrix (np.ndarray): A 2D array of benefit or cost values.
    - shuffle (bool, optional): Set to True to randomize order of row initialization.
                                  Defaults to False.
    - minimize (bool, optional): Set to True if a cost matrix rather than a benefit
                                   matrix is provided. Defaults to False.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple of row indices and assigned column indices.
    """

    bm = benefit_matrix
    if minimize:
        bm = -benefit_matrix.copy()

    assignment = _initial(bm, shuffle=shuffle)
    brs, brb = _best_row_swap(bm, assignment)
    bcs, bcb = _best_col_swap(bm, assignment)

    brb_max = np.amax(brb)
    bcb_max = np.amax(bcb)

    while brb_max > 0 or bcb_max > 0:
        while brb_max > 0 or bcb_max > 0:
            if brb_max > bcb_max:
                r = np.argmax(brb)
                assignment, brs, brb, bcs, bcb = \
                    _row_swap(bm, assignment, brs, brb, bcs, bcb, r)
            else:
                r = np.argmax(bcb)
                assignment, brs, brb, bcs, bcb = \
                    _col_swap(bm, assignment, brs, brb, bcs, bcb, r)
            brb_max = np.amax(brb)
            bcb_max = np.amax(bcb)
        brs, brb = _best_row_swap(bm, assignment)
        bcs, bcb = _best_col_swap(bm, assignment)
        brb_max = np.amax(brb)
        bcb_max = np.amax(bcb)

    return np.arange(bm.shape[0]), assignment
