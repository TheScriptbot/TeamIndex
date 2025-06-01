from TeamIndex.evaluation import *

import numpy as np
import pandas as pd
import math
import json
import os
from itertools import product
from collections import defaultdict

from scipy.optimize import linprog, lsq_linear
import cvxpy as cp

from numba import njit
from concurrent.futures import ThreadPoolExecutor, as_completed


def aggregate_dicts(dict_list):
    aggregated = defaultdict(int)
    for d in dict_list:
        for key, value in d.items():
            aggregated[key] += value
    return dict(aggregated)




def build_laplacian(b, d, adjacency='manhattan', weight=1.0):
    """
    Currently not used for anything...?
    Was initially intended to be used for solving distributions, but KL projection seems more simple.


    Build a discrete Laplacian L (size k x k) for a d-dimensional grid with
    b bins in each dimension, using either Manhattan or Chebyshev adjacency (radius=1).

    Parameters
    ----------
    b : int
        Number of bins per dimension.
    d : int
        Number of dimensions.
    adjacency : str
        Either 'manhattan' or 'chebyshev'.
        - 'manhattan': neighbors differ by 1 in exactly one coordinate (L1 distance = 1).
        - 'chebyshev': neighbors differ by <= 1 in each coordinate (L∞ distance <= 1),
                       i.e. 3^d - 1 possible neighbors in d dimensions if inside the grid.
    weight : float
        Uniform weight w_{i,j} = weight for each adjacency.

    Returns
    -------
    L : np.ndarray of shape (k, k)
        The Laplacian matrix. k = b^d.
        L_{i,i} = sum of weights to neighbors, L_{i,j} = -weight if j is a neighbor, else 0.
    """
    # Total number of bins
    k = b ** d

    # Prepare the Laplacian as a dense array
    L = np.zeros((k, k), dtype=np.float64)

    # Helper function: flatten (x_1, ..., x_d) -> single index
    # Row-major (lexicographic) ordering
    def coord_to_index(coord):
        idx = 0
        mul = 1
        for c in range(d):
            idx += coord[c] * mul
            mul *= b
        return idx

    # Helper function: decode single index -> (x_1, ..., x_d)
    def index_to_coord(idx):
        out = []
        for _ in range(d):
            out.append(idx % b)
            idx //= b
        return tuple(out)

    # Precompute possible neighbor offsets
    # For radius=1:
    # - 'manhattan': sum(abs(offset)) == 1
    # - 'chebyshev': max(abs(offset)) == 1 (and not all zero)
    offsets = []
    if adjacency == 'manhattan':
        # e.g. in 2D: (±1,0) or (0,±1). In dD: exactly one coordinate = ±1, others=0
        # We'll generate all offsets in [-1, 0, 1]^d and filter where sum of abs(...)=1
        for diff in np.ndindex(*(3,)*d):  # (3,3,...,3) = 3^d
            offset = tuple(x-1 for x in diff)  # shift range from [0..2] to [-1..1]
            if sum(abs(o) for o in offset) == 1:
                offsets.append(offset)
    elif adjacency == 'chebyshev':
        # e.g. in 2D: up to 8 neighbors. Condition: max(abs(offset)) <= 1, not all zero
        for diff in np.ndindex(*(3,)*d):
            offset = tuple(x-1 for x in diff)  # in [-1..1]
            if any(offset):  # exclude the (0,0,..,0) offset
                if max(abs(o) for o in offset) <= 1:
                    offsets.append(offset)
    else:
        raise ValueError("adjacency must be 'manhattan' or 'chebyshev'")

    # Build L by enumerating each bin
    for i in range(k):
        coord = index_to_coord(i)
        neighbor_count = 0

        # Try each possible offset
        for off in offsets:
            # new coordinate
            neigh_coord = tuple(coord[dim] + off[dim] for dim in range(d))

            # check if it's inside the grid
            if all(0 <= neigh_coord[dim] < b for dim in range(d)):
                j = coord_to_index(neigh_coord)
                # Set L[i, j] = -weight
                L[i, j] = L[i, j] - weight
                neighbor_count += 1

        # After we've processed neighbors, set L[i, i]
        # L[i,i] = sum of neighbor weights = neighbor_count * weight
        L[i, i] = neighbor_count * weight

    return L

# if __name__ == "__main__":
#     # Example usage
#     b = 3   # bins per dimension
#     d = 2   # 2D grid => total bins = 3^2 = 9
#     adjacency = 'chebyshev'  # 8-neighbor in 2D
#     L = build_laplacian(b, d, adjacency)
#     print("Laplacian matrix shape:", L.shape)
#     print(L)


def count_bins_by_shell(d, b):
    """
    Count how many bins in a d-dimensional grid fall at each 'shell index'
    (integer-rounded Euclidean distance from the grid center).
    
    Parameters
    ----------
    d : int
        Number of dimensions.
    b : int
        Number of bins along each dimension.
    
    Returns
    -------
    shell_counts : dict
        A dictionary where keys are shell indices (0, 1, 2, ...) and
        values are how many bins lie at that distance (rounded) from the center.
    """

    # Center coordinate in each dimension
    center = (b - 1) / 2.0
    
    shell_counts = defaultdict(int)
    
    # Iterate through all possible bin coordinates in the d-dimensional grid
    for coords in product(range(b), repeat=d):
        # Euclidean distance from center
        dist_sq = 0.0
        for c in coords:
            diff = c - center
            dist_sq += diff * diff
        dist = math.sqrt(dist_sq)
        
        # Round to nearest integer shell index
        shell_idx = int(round(dist))
        
        shell_counts[shell_idx] += 1
    
    # Convert to a regular dict sorted by shell index (optional)
    shell_counts_sorted = dict(sorted(shell_counts.items()))
    return shell_counts_sorted

# # Example usage:
# if __name__ == "__main__":
#     d = 3   # 3-dimensional
#     b = 7   # 7 bins per dimension
#     result = count_bins_by_shell(d, b)
#     print(f"Number of bins by shell index (d={d}, b={b}):")
#     for shell_idx, count in result.items():
#         print(f"  Shell {shell_idx}: {count} bins")


def radial_shell_uniform_marginals(d, b, max_shell=None):
    """
    Build a radially symmetric probability distribution in a d-dim grid of size b^d,
    ensuring uniform marginals in each dimension.

    Step-by-step:
      1) For each bin (x_1,...,x_d), compute distance from center in Euclidean norm.
      2) Round that distance to nearest integer => shell index.
      3) Possibly restrict shells to <= max_shell (if provided), so bins in shells above that are removed (prob=0).
      4) Build one variable per shell => x_r.
      5) For dimension j=1..d, each slice alpha=0..(b-1) => sum of x_r * (#bins in that shell∩slice) = 1/b
      6) Summation of all x_r * (#bins in shell r) = 1 (normalization).
      7) Solve these constraints as a linear program with x_r >= 0.
      8) Return final distribution p_i for each bin i.

    Parameters
    ----------
    d : int
        Number of dimensions.
    b : int
        Bins per dimension (grid size = b^d).
    max_shell : int or None
        If int, only shells with index <= max_shell are included in the support.
        If None, use all shells.

    Returns
    -------
    p : dict
        A dictionary {bin_coords: probability}, summing to 1, with uniform 1D marginals.
        Each bin_coords is a d-tuple (c_1,...,c_d).

    Raises
    ------
    ValueError
        If no feasible solution is found.
    """

    # ---- 1) Identify each bin's "shell index" ----
    center = (b - 1) / 2.0
    bin_to_shell = {}
    shell_to_bins = defaultdict(list)

    # function to compute Euclidean distance from center
    def dist_euclidean(coords):
        return math.sqrt(sum((c - center)**2 for c in coords))

    for coords in product(range(b), repeat=d):
        dist = dist_euclidean(coords)
        shell_idx = int(round(dist))  # integer-rounded radius
        bin_to_shell[coords] = shell_idx
        shell_to_bins[shell_idx].append(coords)

    # Possibly restrict shells if max_shell is specified
    if max_shell is not None:
        # discard bins whose shell > max_shell
        for shell_idx in list(shell_to_bins.keys()):
            if shell_idx > max_shell:
                # remove these bins from the mapping
                for c in shell_to_bins[shell_idx]:
                    bin_to_shell[c] = None  # mark as excluded
                del shell_to_bins[shell_idx]

    # Collect all "used" shell indices
    shells = sorted(shell_to_bins.keys())

    # ---- 2) Build linear constraints for uniform marginals ----
    # We want for each dimension j=0..d-1, each slice alpha=0..b-1:
    # sum_{shell r} x_r * (#bins in that slice & shell) = 1/b
    #
    # We'll have "d*b" constraints + 1 "normalization" constraint:
    # sum_{shell r} x_r * (number_of_bins_in_shell[r]) = 1
    #
    # We'll define an LP: minimize 0 subject to A_eq * x = b_eq, x >= 0

    # Let's build a mapping from shell r -> index in variable vector
    r_to_varidx = {r: i for i, r in enumerate(shells)}
    num_vars = len(shells)

    # We'll create a list of constraints:
    # total eq constraints = d*b + 1
    n_constraints = d * b + 1

    A_eq = np.zeros((n_constraints, num_vars), dtype=float)
    b_eq = np.zeros(n_constraints, dtype=float)

    # a helper structure: slice_bin_count[(j, alpha, r)] = how many bins in slice alpha of dim j that are in shell r
    slice_bin_count = defaultdict(int)

    # count bins in each dimension-slice-shell
    for r in shells:
        for coords in shell_to_bins[r]:
            for j in range(d):
                alpha = coords[j]
                slice_bin_count[(j, alpha, r)] += 1

    # fill A_eq rows for dimension-slice constraints
    row_idx = 0
    for j in range(d):
        for alpha in range(b):
            # sum_{r} x_r * slice_bin_count[(j, alpha, r)] = 1/b
            for r in shells:
                cnt = slice_bin_count.get((j, alpha, r), 0)
                A_eq[row_idx, r_to_varidx[r]] = cnt
            b_eq[row_idx] = 1.0 / b
            row_idx += 1

    # Add final row: sum_{r} x_r * (#bins_in_shell[r]) = 1
    # This is the overall normalization
    for r in shells:
        shell_size = len(shell_to_bins[r])
        A_eq[row_idx, r_to_varidx[r]] = shell_size
    b_eq[row_idx] = 1.0

    # ---- 3) Solve via linear programming "feasibility" approach ----
    # Objective: c=0, so we just want any feasible x >= 0
    c = np.zeros(num_vars)

    # We'll use bounds=(0, None) to enforce x_r >= 0
    bounds = [(0, None)] * num_vars

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success is False:
        raise ValueError("No feasible solution found by the LP solver. Possibly too restrictive constraints.")

    x_shell = res.x  # shell probabilities

    # ---- 4) Construct final distribution p_i for each bin i ----
    p = {}
    for coords in product(range(b), repeat=d):
        # if bin was excluded or shell>max_shell => p=0
        shell_idx = bin_to_shell[coords]
        if shell_idx is None:
            p[coords] = 0.0
        else:
            # each bin in shell r shares the same probability x_r
            p[coords] = x_shell[r_to_varidx[shell_idx]] if shell_idx in r_to_varidx else 0.0

    return p

# # Example usage:
# if __name__ == "__main__":
#     d = 3   # 3-dimensional
#     b = 10  # 10 bins per dimension => 1000 bins total
#     max_shell = 5  # keep only shells with radius <= 5

#     distribution = radial_shell_uniform_marginals(d, b, max_shell=max_shell)
#     # distribution is a dict: {(x1,x2,x3): probability, ...}

#     # Summarize
#     nonzero_bins = sum(1 for v in distribution.values() if v > 1e-14)
#     total_prob = sum(distribution.values())
#     print(f"Found distribution with {nonzero_bins} non-zero bins, total probability = {total_prob:.6f}")
#     # You could check 1D marginals or slice sums if you want to verify correctness.



def solve_kl_projection_with_uniform_marginals(prior_weights, grid_shape):
    """
    Solves for a probability distribution over a grid that satisfies uniform marginals
    along each axis, while staying close (in KL divergence) to a given prior distribution.

    Parameters
    ----------
    prior_weights : dict
        Each key represents a coordinate in the grid (e.g., (x, y, z)) that is part of the SUPPORT.
        Dictionary mapping bin coordinates to prior weights (w_i > 0). All coordinates must be in bin_coords.
    grid_shape : tuple of ints
        The shape of the full grid (e.g., (10, 10, 10)) indicating number of bins in each dimension.

    Returns
    -------
    result : dict
        A dictionary mapping bin coordinates to optimized probability values that satisfy the uniform marginal constraints.
    """

    # Validate input
    # assert all(coord in prior_weights for coord in bin_coords), "All bin_coords must have a prior weight"

    bin_coords = prior_weights.keys()

    k = len(bin_coords)  # size of the support set, i.e., the number of non-zero bins
    d = len(grid_shape)  # the dimensions of the grid
    b = grid_shape[0]  # Assuming uniform grid shape for all dimensions
    assert all(np.array(grid_shape) == b), "Requiring uniform grid shape for all dimensions!"

    # Variable: probability for each bin in the support (same order as bin_coords)
    p = cp.Variable(k, nonneg=True)  # holds our result!

    # Construct the prior weight vector (same order as bin_coords)
    assert np.all(np.array(list(prior_weights.values())) > 0)
    w_vec = np.array([weight for coord, weight in prior_weights.items()])
    w_vec = np.maximum(w_vec, 1e-12)  # Prevent log(0) and zero divisions

    # Build constraints for uniform marginals in each dimension
    constraints = []

    # Enforce uniform marginal: for each dimension and each slice
    for dim in range(d):
        for alpha in range(b):
            slice_indices = [i for i, coord in enumerate(bin_coords) if coord[dim] == alpha]
            if not slice_indices:
                raise ValueError(f"No support found for slice {alpha} in dimension {dim} — problem is infeasible.")
            constraints.append(cp.sum(p[slice_indices]) == 1.0 / b)

    # Normalize to total probability = 1
    constraints.append(cp.sum(p) == 1.0)

    # KL divergence objective: min sum(p_i * log(p_i / w_i))
    objective = cp.Minimize(cp.sum(cp.kl_div(p, w_vec)))

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"Optimization failed: {problem.status}")

    result_array = np.zeros(grid_shape, dtype=float)

    # Return the resulting distribution
    for i, coord in enumerate(bin_coords):
        result_array[coord] = float(p.value[i])

    return result_array

import numpy as np

def generate_double_cluster_prior(
    grid_shape=(10, 10, 10, 10),
    center1=(2, 2, 2, 2),
    center2=(7, 7, 7, 7),
    weight1=1.0,
    weight2=0.5,
    sigma1=1.0,
    sigma2=1.0,
    crop_threshold=1e-6,
):
    """
    Generate a mD prior weight array with two Gaussian-like clusters.

    Parameters:
        grid_shape (tuple): Shape of the mD grid (default 10x10x10x10)
        center1 (tuple): Center of the first (stronger) cluster
        center2 (tuple): Center of the second (weaker) cluster
        weight1 (float): Amplitude of the first cluster
        weight2 (float): Amplitude of the second cluster
        sigma1 (float): Spread of the first cluster
        sigma2 (float): Spread of the second cluster
        crop_threshold (float): Values below this are set to zero

    Returns:
        prior_weights (np.ndarray): mD array of prior weights
    """
    dim = len(grid_shape)
    
    # Create coordinate grid
    coords = np.indices(grid_shape).reshape(dim, -1).T

    # Compute squared distances to each center
    def sqdist(center, sigma):
        diff = coords - np.array(center)
        return np.sum(diff**2, axis=1) / (2 * sigma**2)

    dist1 = sqdist(center1, sigma1)
    dist2 = sqdist(center2, sigma2)

    # Evaluate weighted sum of Gaussians
    flat_weights = (
        weight1 * np.exp(-dist1) +
        weight2 * np.exp(-dist2)
    )

    # Crop small values
    flat_weights[flat_weights < crop_threshold] = 0.0

    # Reshape back into a matrix
    prior_weights = flat_weights.reshape(grid_shape)

    return prior_weights

def prior_array_to_dict(prior_array, crop_threshold=0.0):
    """
    Convert an N-dimensional NumPy array into a dictionary of non-zero cells.

    Parameters:
        prior_array (np.ndarray): N-dimensional array of prior weights
        crop_threshold (float): Minimum value to include in output (default: 0.0)

    Returns:
        dict: Keys are coordinate tuples, values are prior weights
    """
    nonzero_indices = np.argwhere(prior_array > crop_threshold)
    return {
        tuple(idx): prior_array[tuple(idx)]
        for idx in nonzero_indices
    }

#### Benchmark Index Generation

def preprocess_grid_distribution(p_matrix, query_slices):
    """
    Preprocess a grid distribution by conditioning it on the query region.
    
    Parameters
    ----------
    p_matrix : np.ndarray
        A NumPy array representing the full distribution [p^g] for one grid.
        The element in the i-th bin is denoted p_i^g.
        p_matrix may contain np.nan values to indicate bins that should be ignored.
    query_slices : tuple of slice objects
        A tuple of slices that selects the query region Q_g from the grid.
        Q_g must be a proper subset of the full support of p_matrix.
    
    Returns
    -------
    p_conditioned : np.ndarray
        The conditioned distribution [p^g | Q_g], i.e. p_matrix restricted to Q_g and normalized to sum to 1.
    query_mass : float
        The total probability mass in the query region (i.e., the sum of p_matrix over Q_g).
    """
    # Extract the query region from the full grid distribution.
    p_query = p_matrix[query_slices]  # drops all cells outside of the query region
    # Replace any np.nan values with 0.
    p_query = np.nan_to_num(p_query, nan=0.0)
    # Compute the total mass in the query region.
    query_mass = np.sum(p_query)
    if query_mass <= 0:
        raise ValueError("The query region has zero total mass; check p_matrix and query_slices.")
    # Normalize the query region to obtain the conditioned distribution.
    p_conditioned = p_query / query_mass
    return p_conditioned, query_mass

def preprocess_all_grids(queries, distributions):
    """
    Preprocess multiple grids by conditioning each grid's distribution on its query region.
    
    Parameters
    ----------
    queries : dict
        A dictionary mapping a grid identifier to query_slices
        where query_slices is a tuple of slice objects defining the query region Q_g.
    distributions : dict
        A dictionary mapping a grid identifier to a p_matrix,
        where p_matrix is a NumPy array representing the full distribution [p^g]
    
    Returns
    -------
    conditioned_data : dict
        A dictionary mapping each grid identifier to a tuple:
          (p_conditioned, query_mass)
        where p_conditioned is the conditioned distribution [p^g | Q_g] and query_mass is the total mass in Q_g.
    """
    conditioned_data = {}
    for grid_id, query_slices in queries.items():
        p_matrix = distributions[grid_id]  # a multi-dimensional array, representing the grid's distribution
        unconditioned_shape = p_matrix.shape
        p_conditioned, query_mass = preprocess_grid_distribution(p_matrix, query_slices)
        conditioned_data[grid_id] = {"p_conditioned":p_conditioned,
                                     "w": query_mass,
                                     "unconditioned_shape": unconditioned_shape,
                                     "query_slices": query_slices}
    return conditioned_data

## Utility function to split work across several tasks. Used to define work packages for parallel code (data generator below)
def distribute_tasks(n_threads, intersection_count, drag_count):
    tasks = []
    total_work = intersection_count + drag_count
    if total_work == 0:
        return tasks

    # Total number of tasks is at most the lesser of n_threads or total work units.
    total_chunks = min(n_threads, total_work)

    # Allocate tasks per type.
    if intersection_count > 0 and drag_count > 0:
        # Allocate proportionally.
        tasks_intersection = max(1, round(total_chunks * intersection_count / total_work))
        tasks_drag = total_chunks - tasks_intersection
        # Guarantee at least one task per type.
        if tasks_drag == 0:
            tasks_drag = 1
            tasks_intersection = total_chunks - 1
    elif intersection_count > 0:
        tasks_intersection = total_chunks
        tasks_drag = 0
    else:
        tasks_drag = total_chunks
        tasks_intersection = 0

    def split_work(count, chunks):
        base = count // chunks
        remainder = count % chunks
        # First 'remainder' chunks get an extra unit.
        return [base + 1 if i < remainder else base for i in range(chunks)]

    # Create tasks for intersections.
    if tasks_intersection:
        for chunk in split_work(intersection_count, tasks_intersection):
            tasks.append(('intersection', chunk))
    # Create tasks for drag.
    if tasks_drag:
        for chunk in split_work(drag_count, tasks_drag):
            tasks.append(('drag', chunk))

    return tasks



############################################################
# 1) Utility: Weighted k-subset selection w/o replacement  #
############################################################
def gumbel_top_k(weights, k, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    weights = np.asarray(weights)
    log_weights = np.log(weights)
    gumbels = -np.log(-np.log(rng.uniform(size=len(weights))))
    scores = log_weights + gumbels
    top_k_indices = np.argpartition(-scores, k)[:k]
    return top_k_indices
    
def efraimidis_spirakis(weights, k, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    weights = np.asarray(weights)
    u = rng.uniform(size=len(weights))
    keys = u ** (1.0 / weights)
    return np.argpartition(-keys, k)[:k]

def numpy_weighted_subset(weights, k, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice(range(weights.size), size=k, replace=False, p=weights)
    
def choose_k_grids_weighted(k, grid_ids, weights, method=numpy_weighted_subset):
    """
    Select exactly k distinct grids from a list [1,...,weights.size], using weights in 'weights',
    all normalized so that sum(weights) = 1.

    Returns:
        chosen_list : list of chosen grid identifiers
    """
    assert(len(grid_ids) == len(weights)), "Weights need to correspond to grid_ids!"
    assert(all(weights)), "All weights need to be > 0!"
    rng = np.random.default_rng()
    ids = method(weights, k, rng)
    return [grid_ids[i] for i in ids]

############################################################
# 2) Intersection & Drag chunk generation                  #
############################################################

@njit
def sample_cell_from_conditioned(cdf, query_shape, query_offset):
    """
    Sample a coordinate from a flattened multidimensional probability distribution
    defined over a restricted subregion of a full grid.

    Parameters
    ----------
    cdf : 1D np.ndarray
        Cumulative distribution function (CDF) over the query region.
    query_shape : tuple of ints
        The shape of the query region.
    query_offset : tuple of ints
        The starting indices of the query region in the full grid.

    Returns
    -------
    full_coords : np.ndarray of ints
        Coordinates in the full grid corresponding to the sampled cell.
    """
    # Step 1: Sample a flat index from the CDF using binary search
    r = np.random.random()
    flat_idx = np.searchsorted(cdf, r, side='right')

    # Step 2: Convert flat index to local coordinates in the query region
    ndim = len(query_shape)
    local_coords = np.empty(ndim, dtype=np.int64)
    for d in range(ndim - 1, -1, -1):
        local_coords[d] = flat_idx % query_shape[d]
        flat_idx //= query_shape[d]

    # Step 3: Offset local coordinates to get full-grid coordinates
    full_coords = np.empty(ndim, dtype=np.int64)
    for d in range(ndim):
        full_coords[d] = local_coords[d] + query_offset[d]

    return full_coords


def _generate_intersection_chunk(count, conditioned_data):
    local_results = []
    stats = dict()
    for g in conditioned_data:
        stats[g] = 0
    for _ in range(count):
        sample_assignment = []
        for g in conditioned_data:
            query_slices = conditioned_data[g]["query_slices"]
            query_shape = conditioned_data[g]["p_conditioned"].shape
            # allows us to emit coordinates in the original grid, not just relative to the query region
            query_offset = [s.start for s in query_slices]
            cell_idx = sample_cell_from_conditioned(conditioned_data[g]["cdf"],
                                                    query_shape,
                                                    query_offset)
            sample_assignment.extend(cell_idx)
            stats[g] += 1
        local_results.append(tuple(sample_assignment))
    return local_results, stats


def _generate_drag_chunk(count, grid_ids, conditioned_data, P_k, relative_grid_weights, method):
    local_results = []
    stats = dict()
    assert len(grid_ids) == len(relative_grid_weights)
    for g in grid_ids:
        stats[g] = 0
    for _ in range(count):
        ## first, determine the number of grids we can choose to insert values into
        ## Note that we have to ignore grids that are not pruned in any way (and therefore have 0 weight)
        k = np.random.choice(range(1, len(relative_grid_weights)), p=P_k)
        
        ## second, we choose the actual subset of grids we insert the drag value into
        chosen_grids = choose_k_grids_weighted(k, grid_ids, relative_grid_weights, method=method)

        ## form the tuple that represents the insertion
        sample_assignment = []
        for g in grid_ids:
            if g in chosen_grids:
                query_slices = conditioned_data[g]["query_slices"]
                query_shape = conditioned_data[g]["p_conditioned"].shape
                # allows us to emit coordinates in the original grid, not just relative to the query region
                query_offset = [s.start for s in query_slices]
                cell_idx = sample_cell_from_conditioned(conditioned_data[g]["cdf"],
                                                        query_shape,
                                                        query_offset)
                sample_assignment.extend(cell_idx)
                stats[g] += 1
            else:
                dim_g = len(conditioned_data[g]["unconditioned_shape"])
                sample_assignment.extend([None] * dim_g)
        local_results.append(tuple(sample_assignment))
    return local_results, stats

############################################################
# 3) The main generation function                          #
############################################################
def generate_benchmark_data(
        conditioned_data, T_rel, N,
        n_threads=5,
        start_idx=0,
        shuffle_afterward=True,
        random_subset_method=gumbel_top_k):
    """
    This function creates a test dataset with N samples, using the meta data in conditioned_data as baseline.
    Note that we keep the grid-distributions and the relative size of each grid's relative data volume in tact,
    but the inter-grid dependencies are entirely random. However, we precisely control how large the result of
    their intersection will be. I.e., when intersecting all grids, the final result will allways yield: 
     - T*N items in the intersection
     - (1-T)*N "drag" items, that get pruned.

    Instead of selecting T directly, we choose it as T = T_rel * min_g(w_g)
    This ensures we can not enter an invalid T value, which depends on the distributions and query

    Note that we have to take care on how to create "drag" items, in order to not distort the relative grid sizes.
    Further, after assigning values to a grid (trivial in the intersection case), we need to insert them according
    to the respective distribution (which is assumed to be conditioned to the a query region).

    Also note that each grid's query needs to discard some cells/blocks/inverted lists. Otherwise the intersection
    would not make sense and we would not reasonable be able to declare "drag" values - an unselective Team would
    never contain values that are pruned!  

    Parameters
    ----------
    conditioned_data : dict
        grid_id -> { "p_conditioned": np.array, "w": float, "dim": int }
    T_rel : float
        fraction of samples that are intersection, relative to the maximum intersection size, as implied by the most selective grid-query 
    N : int
        total sample count
    n_threads : int
        number of threads to use
    start_idx : int
        First tuple id in the result. Useful for appending the results of multiple calls of this function. 
    shuffle_afterward : bool
        if True, shuffle final samples in-place
    random_subset_method : function
        Method to choose subsets.

    Returns
    -------
    df : pd.DataFrame
        containing the final bin assignments.
    """
    N = int(N)
    grid_ids = list(conditioned_data.keys())

    # "w" is the respective grid's selectivity.
    # We normalize [w] to use it to weigh our grid-subset selection, such that the final result will yield
    # indices that are of the expected relative size.
    grid_weights = np.array([conditioned_data[g]["w"] for g in grid_ids])
    min_w = min(grid_weights)
    T = T_rel * min_w
    if T > min_w:
        raise ValueError(f"T={T} > min w_g={min_w}, not feasible.")
    
    print("Generating benchmark data with N =", N)
    print("Maximum selectivity:", min_w)
    print("Configured Seclectivity: T =", T)

    ## calculate the number of values we generate for each type
    intersection_count = int(math.ceil(T * N))
    drag_count = N - intersection_count
    
    # Extract raw weights for the drag case. We normalize to form relative_grid_weights distribution
    weights_for_grids_with_drag = grid_weights-T
    grid_positions_with_drag = np.nonzero(weights_for_grids_with_drag)
    weights_for_grids_with_drag = weights_for_grids_with_drag[grid_positions_with_drag]
    weights_for_grids_with_drag = weights_for_grids_with_drag / weights_for_grids_with_drag.sum()

    grid_ids_with_drag = [grid_ids[int(grid_pos)] for grid_pos in grid_positions_with_drag[0]]
    

    ## Do some preprocessing to avoid repeated work
    for g, meta_data in conditioned_data.items():
        ## compute the CDF for the grid's distribution for later sampling:
        dist = meta_data["p_conditioned"]
        dist = dist/dist.sum()  # normalize to sum to 1, just to be sure

        cdf = np.cumsum(meta_data["p_conditioned"])
        conditioned_data[g]["cdf"] = cdf

    #######################################
    # Determine the probability distribution for the number of grids we insert a drag value into
    # This has to be done with care, otherwise we will not be able to keep the index volume for each grid
    # in the same proportions as implied by the relative sizes in [w_g]

    # Subset sizes: 1 to n-1 (proper subsets only! Empty is useless and full would be an intersection value)
    # subset_sizes = np.arange(1, n_grids)

    # Function to compute expected inclusion probabilities I_{g,k}
    def compute_I_matrix(w, n_grids):
        I = np.zeros((n_grids, n_grids - 1))
        for k in range(1, n_grids):
            probs = []
            for _ in range(10000):
                subset = np.random.choice(n_grids, size=k, replace=False, p=w/w.sum())
                inclusion = np.zeros(n_grids)
                inclusion[subset] = 1
                probs.append(inclusion)
            I[:, k - 1] = np.mean(probs, axis=0)
        return I
    print("Computing linear constraint matrix...")
    # Compute I matrix, which defines the linear constraints for the optimization problem
    I_matrix = compute_I_matrix(weights_for_grids_with_drag, weights_for_grids_with_drag.size)

    # Solve for optimal P_k (subset size distribution)
    print("Solving the linear equation for the correct subset size distribution...")
    res = lsq_linear(I_matrix, weights_for_grids_with_drag, bounds=(0, 1))
    P_k = res.x / res.x.sum()  # normalize to sum to 1
    #########################################
    # Print some meta data before we process anything
    # print("Linear constraint matrix (for obtaining P_k) I_k:", I_matrix)
    print("Subset size distribution P_k:", P_k)
    expected_k = np.sum((np.arange(1, len(P_k) + 1)) * P_k)
    drag_volume = (1 - T) * expected_k * N

    print("Intersection count:", intersection_count)
    print("Drag count:", drag_count)
    print("Drag volume:", drag_volume)

    #########################################

    tasks = distribute_tasks(n_threads, intersection_count, drag_count)
    
    def run_task(task):
        ttype, c = task
        print("Running task:", ttype, "with count", c)
        if ttype == "intersection":
            return _generate_intersection_chunk(c, conditioned_data)
        else:
            return _generate_drag_chunk(c, grid_ids_with_drag, conditioned_data, P_k, weights_for_grids_with_drag, method=random_subset_method)

    results = []
    stats_vec = []

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for fut in futures:
            data, stats = fut.result()
            print("Thread finished generating",len(data),"tuples.")
            results.extend(data)
            stats_vec.append(stats)

    if shuffle_afterward and len(results) > 1:
        np.random.shuffle(results)  # along the first axis only

    df = pd.DataFrame(results, columns=np.concat(list(conditioned_data.keys())), dtype=pd.Int8Dtype())
    df.index = range(start_idx, start_idx + len(df))

    print("Initial weights for grids with drag:", weights_for_grids_with_drag)
    rel_cards = np.array([stats[g] for g in grid_ids_with_drag])
    print("Relative sizes for grids with drag:", rel_cards/rel_cards.sum())

    return df, aggregate_dicts(stats_vec)



def compute_inverted_postings(df, grid_specs, start_id=0, n_jobs=1):
    """
    Turn a "VA-file" style table into inverted postings for each grid
    with one list per non-emtpy grid cell.
    """
    def _compute_inverted_postings_for_grid(df, grid_cols, shape):
        # 1) Slice out only columns for this grid
        sub_df = df[list(grid_cols)]
        print("Creating postings for dataframe of shape:", sub_df.shape)

        # 2) Filter out rows with NA only for these columns
        valid_mask = sub_df.notna().all(axis=1)

        # 3) Grab the row IDs that passed
        valid_ids = np.flatnonzero(valid_mask).astype(np.uint32)

        # 4) Convert the valid subset to int, flatten coords
        coords = sub_df[valid_mask].astype(np.int32).to_numpy()
        flat_idxs = np.ravel_multi_index(coords.T, dims=shape).astype(np.uint32)

        # 5) Pair them up and sort
        pairs = np.column_stack((flat_idxs, valid_ids))
        # Sort primarily by bin index, then row ID
        pairs.sort(axis=0)
        return pairs, shape
    df = df.reset_index(drop=True)
    
    inverted_postings = {}
    grid_cols_list = list(grid_specs.items())

    if n_jobs == 1:
        # Serial loop
        for (cols, shape) in grid_cols_list:
            pairs, shape_ = _compute_inverted_postings_for_grid(df, cols, shape)
            # Adjust row_id by start_id
            pairs[:, 1] += start_id
            # if type(cols) is tuple:
                # cols = list(cols)
            inverted_postings[cols] = (pairs, shape_)
            print(inverted_postings[cols], ":", len(pairs),"postings created!")
        return inverted_postings

    # Otherwise, multi-threaded
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {}
        for (cols, shape) in grid_cols_list:
            fut = executor.submit(_compute_inverted_postings_for_grid, df, cols, shape)
            futures[fut] = (cols, shape)

        for fut in as_completed(futures):
            (cols, shape) = futures[fut]
            pairs, shape_ = fut.result()
            pairs[:, 1] += start_id
            inverted_postings[cols] = (pairs, shape_)

    return inverted_postings

def dump_inverted_postings(inverted_postings, output_dir, pagesize=4096, codec_id=1):
    """
    Simple function to dump the inverted lists to storage.
    """
    for grid_cols, (pairs, shape) in inverted_postings.items():
        base_name = "-".join(grid_cols)
        prefix = os.path.join(output_dir, base_name + ".copy")

        total_bins = np.prod(shape)
        cardinalities = np.zeros(total_bins,      dtype=np.uint32)
        offsets = np.zeros(total_bins + 1, dtype=np.uint32)
        sizes   = np.zeros(total_bins,      dtype=np.uint64)
        codecs_ = np.full(total_bins, codec_id, dtype=np.uint8)

        # Build .lists
        page_offset = 0
        with open(prefix + ".lists", "wb") as list_file:
            idx_pos = 0
            n_pairs = pairs.shape[0]

            for bin_idx in range(total_bins):
                offsets[bin_idx] = page_offset

                if idx_pos >= n_pairs or pairs[idx_pos, 0] != bin_idx:
                    # This bin has no postings
                    continue

                # Otherwise, gather consecutive rows for this bin
                start_pos = idx_pos
                while idx_pos < n_pairs and pairs[idx_pos, 0] == bin_idx:
                    idx_pos += 1
                postings = pairs[start_pos:idx_pos, 1]

                cardinalities[bin_idx] = len(postings)
                
                # Convert to bytes
                postings_bytes = postings.tobytes()
                size_in_bytes = len(postings_bytes)

                # Page padding
                pad_len = (pagesize - (size_in_bytes % pagesize)) % pagesize
                padded_bytes = postings_bytes + b"\x00" * pad_len

                # Write
                list_file.write(padded_bytes)

                # Update metadata
                sizes[bin_idx] = size_in_bytes
                page_offset += len(padded_bytes) // pagesize

            # Final offset
            offsets[-1] = page_offset

        # Write metadata
        cardinalities.tofile(prefix + ".cardinalities")
        offsets.tofile(prefix + ".offsets")
        sizes.tofile(prefix + ".sizes")
        codecs_.tofile(prefix + ".codecs")


def load_inverted_metadata(prefix, shape):
    """
    Load the metadata for an inverted index from disk, given a prefix and known shape.

    Parameters
    ----------
    prefix : str
        Full path prefix (e.g. "/path/to/A1-A2.copy"), without the file extension.
    shape : tuple
        The cardinalities (e.g. (3, 3)) for this grid. We assume the user
        already knows or has read it from .cardinalities.

    Returns
    -------
    offsets : np.ndarray
        1D np.uint32 array of length total_bins + 1
    sizes : np.ndarray
        1D np.uint32 array of length total_bins
    codecs : np.ndarray
        1D np.uint32 array of length total_bins
    """
    total_bins = np.prod(shape)

    cardinalities_file = prefix + ".cardinalities"
    offsets_file = prefix + ".offsets"
    sizes_file   = prefix + ".sizes"
    codecs_file  = prefix + ".codecs"

    # Load them from file
    cardinalities = np.fromfile(cardinalities_file, dtype=np.uint32)
    offsets = np.fromfile(offsets_file, dtype=np.uint32)
    sizes   = np.fromfile(sizes_file,   dtype=np.uint64)
    codecs_ = np.fromfile(codecs_file,  dtype=np.uint8)

    # Optional: sanity checks
    if len(cardinalities) != (total_bins):
        raise ValueError(f"Cardinalities file length mismatch. Expected {total_bins}, got {len(cardinalities)}")
    if len(offsets) != (total_bins + 1):
        raise ValueError(f"Offsets file length mismatch. Expected {total_bins + 1}, got {len(offsets)}")
    if len(sizes) != total_bins:
        raise ValueError(f"Sizes file length mismatch. Expected {total_bins}, got {len(sizes)}")
    if len(codecs_) != total_bins:
        raise ValueError(f"Codecs file length mismatch. Expected {total_bins}, got {len(codecs_)}")

    return cardinalities.reshape(shape), offsets, sizes.reshape(shape), codecs_.reshape(shape)


def create_dummy_json_config(grid_specs, input_folder, output_path, quantiles, query):
    """
    Create a JSON config file using dummy data for quantiles, empty special_values,
    empty queries, etc., and write it to 'output_path'.

    Parameters
    ----------
    grid_specs : dict
        Dictionary mapping tuples of column names (a "team") 
        to a tuple of cardinalities for each dimension.
        Example: {
          ("Kplus_TRACK_CHI2_PER_NDOF","piminus_TRACK_CHI2_PER_NDOF","B0_LOKI_DTF_CTAU","Kplus_PIDK"): (10,10,10,10),
          ...
        }
    input_folder : str
        Folder path to be stored as "index_folder".
    output_path : str
        Where the resulting JSON file will be written. Parent folder must exist.
    quantiles : dict
        Dictionary mapping column names to dummy quantiles.
    query : str
        A query string to be stored in the "queries" field.
    """

    output_path = Path(output_path)
    input_folder = Path(input_folder)
    assert input_folder.exists(), f"Input folder {input_folder} does not exist."
    assert input_folder.is_dir(), f"Input folder {input_folder} is not a directory."

    assert output_path.parent.exists(), f"Output parent folder {output_path.parent} does not exist!"

    # Gather all unique column names from grid_specs
    all_columns = set()
    for cols in grid_specs.keys():
        all_columns.update(cols)
    all_columns = sorted(all_columns)  # for stable ordering
        
    # Build "teams" from grid_specs keys
    #    A "team" is simply the list of columns in one grid
    teams = [list(cols) for cols in grid_specs.keys()]

    # move quantiles into a standard list and remove -Inf/Int values at the start and end, if they are there:
    quantiles_dict = dict()
    for col, quantiles in quantiles.items():
        quantiles_dict[col] = []
        for q in quantiles:
            # convert to float and remove -Inf and Inf
            if q == -np.inf or q == np.inf:
                continue
            quantiles_dict[col].append(float(q))
    
    # Build the final config data structure
    config_data = {
        "compressions": ["copy"],        # single compression: 'copy'
        "index_folder": str(input_folder.absolute()),    # from function argument
        "quantiles": quantiles_dict,          # actual values for correct query mapping later
        "queries": [query],                   # query used to generate the data
        "source_table": None,            # explicitly None
        "special_values": {},            # empty dict
        "teams": teams
    }
    # Write JSON with indentation
    with open(output_path, "w") as f:
        json.dump(config_data, f, indent=2)


def generate_indices(N, T_rel_list, team_dists, team_queries, destination_folder, quantiles, query, n_jobs = 4):
    """
    Generate indices for the given parameters and save them to a folder.

    N is the number of tuples involved in the intersection, T_rel gives the selectivity, relative to the smallest Team.
    Team sizes are implied by the query, i.e., the probability mass of the respective distribution within the query region.

    We only generate data for the query region!
    """
    destination_folder = Path(destination_folder)  # we create folders within this one
    assert destination_folder.exists(), f"Destination folder {destination_folder} does not exist!"

    N = int(N)

    grid_specs = {team: dist.shape for team, dist in team_dists.items()}

    # convenience function to re-structure input. Conditions each Team' distribution to the respective query region
    config = preprocess_all_grids(team_queries, team_dists)

    for T_rel in T_rel_list:
        Trel_str = str(T_rel).replace(".", "")
        subfolder = destination_folder / f"selectivity_Trel{Trel_str}_N{int(N)}"
        if subfolder.exists():
            print(f"Folder {subfolder} already exists. Skipping.")
            continue

        # Generate the benchmark data
        print(f"Generating indices for T_rel = {T_rel}")
        benchmark_data, stats = generate_benchmark_data(config, T_rel, N, n_threads=n_jobs)
        postings = compute_inverted_postings(benchmark_data, grid_specs, start_id=0, n_jobs=n_jobs)

        # Save the benchmark data
        print("Creating folder to dump data:", subfolder.absolute())
        subfolder.mkdir(parents=True, exist_ok=False)
        dump_inverted_postings(postings, subfolder)
        
        # Save the config
        cfg_file_path = subfolder / f"index.json"
        create_dummy_json_config(grid_specs, subfolder, cfg_file_path, quantiles, query)
        del benchmark_data, postings


def generate_lhcb_benchmark_data(destination_folder="./indices/", N=1e8, T_rel_list=[0.0, 0.1, 0.5, 0.9, 1.0], n_jobs=13):
    """
    Generate the benchmark data for the LHCb dataset.
    """
    query = """muplus_PIDmu > 0 and muplus_PT > 500 and muminus_PIDmu > 0 and muminus_PT > 500 and J_psi_1S_M < 3176.9 and J_psi_1S_M > 3016.9 and J_psi_1S_ENDVERTEX_CHI2 < 16 and Kst_892_0_M > 826 and Kst_892_0_M < 966 and Kst_892_0_PT > 1300 and Kst_892_0_ENDVERTEX_CHI2 < 25 and piminus_TRACK_CHI2_PER_NDOF < 5 and piminus_PIDK < 0 and Kplus_TRACK_CHI2_PER_NDOF < 5 and Kplus_PIDK > 0 and B0_M > 5150 and B0_M < 5450 and B0_ENDVERTEX_CHI2_PER_NDOF < 20 and B0_LOKI_DTF_CTAU > 0.0598"""
    # open up existing index and use it's distribution for generating the data set!
    ti = TeamIndex("lhcb_index.json", compression= "roaring")
    team_dists = {tuple(ti.teams[team]): cards/ti.stats["number_of_tuples"] for team, cards in ti.cardinalities.items()}
    team_queries = {tuple(ti.teams[team]): ti._make_histogram_slicer(query, ti.teams[team]) for team in ti.teams}

    # grid_specs = {team: dist.shape for team, dist in team_dists.items()}

    generate_indices(N,
                     T_rel_list=T_rel_list,
                     team_dists=team_dists,
                     team_queries=team_queries,
                     destination_folder=destination_folder,
                     quantiles=ti.quantiles,
                     query=query,
                     n_jobs=n_jobs)