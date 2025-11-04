from TeamIndex import evaluation

import math
import numpy as np
import random
from typing import Dict, Iterable, Sequence, Tuple, List, Optional

# Type aliases for clarity
SliceTuple = Tuple[slice, ...]
QueryDict  = Dict[str, SliceTuple]
ShiftDict  = Dict[str, int]
GridSizes  = Dict[str, int]



def compute_translation_vector(
    query_slices: QueryDict,
    grid_sizes: GridSizes,
    rng: random.Random | None = None,
) -> ShiftDict:
    """Compute a *uniformly random* translation vector that keeps the hyper-rectangle
    inside the overall grid.

    Parameters
    ----------
    query_slices
        Original dictionary mapping dashed-concatenated axis names to tuples of
        :pyclass:`slice` objects.
    grid_sizes
        Mapping *axis → grid-length*.  Each key must appear in *query_slices*.
    rng
        Optional instance of :pyclass:`random.Random` (or compatible) to draw
        the shifts.  If *None*, :pyclass:`random.SystemRandom` is used.

    Returns
    -------
    dict
        ``{axis_name: integer_shift}``
    """
    if rng is None:
        rng = random.SystemRandom()

    def _collect_axis_slices(query_slices: QueryDict) -> Dict[str, slice]:
        """Return *one* canonical :pyclass:`slice` per axis found in *query_slices*."""
        axis_slice: Dict[str, slice] = {}
        for key, tup in query_slices.items():
            for axis, slc in zip(key.split("-"), tup):
                axis_slice[axis] = slc  # later assignments are identical by design
        return axis_slice

    axis_slice = _collect_axis_slices(query_slices)

    # --- build admissible intervals per axis ---------------------------------
    shift_range: Dict[str, Tuple[int, int]] = {}
    for axis, slc in axis_slice.items():
        start = 0 if slc.start is None else int(slc.start)
        stop  = grid_sizes[axis] if slc.stop  is None else int(slc.stop)
        v_min = -start
        v_max = grid_sizes[axis] - stop
        if v_min > v_max:
            raise ValueError(
                f"Slice for axis '{axis}' is larger than the grid ({start}:{stop} vs {grid_sizes[axis]})."
            )
        shift_range[axis] = (v_min, v_max)

    # --- sample one vector uniformly -----------------------------------------
    translation_vector: ShiftDict = {
        axis: rng.randint(v_min, v_max) for axis, (v_min, v_max) in shift_range.items()
    }
    return translation_vector



def apply_translation_vector(
    query_slices: QueryDict,
    translation_vector: ShiftDict,
) -> QueryDict:
    """Return a *new* query-slice dictionary with all slices shifted.

    Parameters
    ----------
    query_slices
        Original slice dictionary.
    translation_vector
        Output of :pyfunc:`compute_translation_vector`.  Every axis present in
        *query_slices* must be in this dict.

    Returns
    -------
    dict
        Same structure as *query_slices* but with shifted slice objects.
    """
    transformed: QueryDict = {}
    for key, tup in query_slices.items():
        shifted_slices: list[slice] = []
        for axis, slc in zip(key.split("-"), tup):
            if axis not in translation_vector:
                raise KeyError(f"Axis '{axis}' missing from translation vector.")
            shift = translation_vector[axis]
            start_new = (0 if slc.start is None else int(slc.start)) + shift
            stop_new  = (slc.stop if slc.stop is not None else None)
            if stop_new is not None:
                stop_new += shift
            shifted_slices.append(slice(start_new, stop_new, slc.step))
        transformed[key] = tuple(shifted_slices)
    return transformed



def random_partition(
    elements: Iterable,
    sizes: Sequence[int],
    *,
    canonical: bool = True,
    rng: Optional[random.Random] = None
) -> Tuple[Tuple]:
    """

    A universal implementation of a simple algorithm: Simply partition a shuffled list of elements!

    Return a random partition of `elements` into blocks whose lengths equal `sizes`.

    Parameters
    ----------
    elements : Iterable
        The items to partition (they are copied, so any iterable is fine).
    sizes : Sequence[int]
        Block lengths; must sum to len(elements).
    canonical : bool, default True
        If True, each block is returned sorted and the list of blocks is sorted
        lexicographically, so the result is a deterministic canonical form that
        ignores block order.  Set False if you want the natural left-to-right
        order produced by the shuffle.
    rng : random.Random, optional
        Source of randomness (useful for reproducibility in tests).

    Returns
    -------
    Tuple[Tuple]
        A tuple of blocks, each block itself being a tuple.
    """
    elems: List = list(elements)
    total = len(elems)
    if sum(sizes) != total:
        raise ValueError("Block sizes must sum to the number of elements")

    rng = rng or random
    rng.shuffle(elems)

    blocks: List[Tuple] = []
    idx = 0
    for sz in sizes:
        block = tuple(sorted(elems[idx:idx + sz])) if canonical else tuple(elems[idx:idx + sz])
        blocks.append(block)
        idx += sz

    if canonical:
        blocks = sorted(blocks)            # remove label ordering
    return tuple(blocks)



def sample_unique_partitions(elements: List, sizes: Tuple[int], k: int, seed=None, max_attempts=1_000_000):
    """
    Rejection sampling for random Team compositions.
    Generate k unique random partitions of n elements into subsets of given sizes.

    Parameters
    ----------
    elements : List
        List of elements to partition.
    sizes : list of int
        The sizes of the subsets (must sum to n).
    k : int
        Number of unique partitions to return.
    seed : int, optional
        Seed for reproducible randomness.
    max_attempts : int, default 1_000_000
        Fail if unable to collect k unique partitions in this many attempts.

    Returns
    -------
    set of tuple of tuples
        A set of `k` unique canonicalized partitions.
    """
    assert(sum(s for s in sizes) == len(elements)), "sizes must sum to len(elements)!"
    rng = random.Random(seed)
    seen = set()
    attempts = 0
    while len(seen) < k and attempts < max_attempts:
        part = random_partition(elements, sizes, rng=rng)
        seen.add(part)
        attempts += 1
    if len(seen) < k:
        raise RuntimeError(f"Only found {len(seen)} unique partitions after {max_attempts} attempts.")
    return seen



def create_random_queries(query_count: int, pred_sel: float, bin_borders: dict, rng: random.Random | None = None):
    if rng is None:
        rng = random.SystemRandom()
        
    query_slice_list = list()
    grid_sizes = dict()
    hr_size = 1
    columns = list(bin_borders.keys())
    
    for column in columns:
        assert bin_borders[column][0] == -np.inf and bin_borders[column][-1] == np.inf
        b = len(bin_borders[column])-1
        grid_sizes[column] = b

        pred_bin_sel = math.floor(pred_sel*b)
        assert pred_bin_sel > 0
        # pred_bin_sel = max(1, min(b-1, pred_bin_sel))
        hr_size *= pred_bin_sel
        query_slice_list.append(slice(0, pred_bin_sel))
    
    total_size = math.prod(s for s in grid_sizes.values())
    print(f"Hyperrectangle size in {len(columns)}-dim space:", hr_size,"bins vs.", total_size)
    print(f"Volume: ~{100*hr_size/total_size:.12%}")
    initial_query_slice_dict = {"-".join(columns): tuple(query_slice_list)}
    random_queries = list()
    for _ in range(query_count):
        
        random_query_slices = apply_translation_vector(initial_query_slice_dict,
                                                       compute_translation_vector(initial_query_slice_dict,
                                                                                  grid_sizes, rng))
        query_string = evaluation.slicer_to_conjunctive_query(list(random_query_slices.values())[0], columns, bin_borders)

        random_queries.append(query_string)
    return random_queries


def create_random_queries_non_uniform(query_count: int, pred_sel: dict, bin_borders: dict):
    """
    Create random queries using non-uniform predicate selectivitities, as provided in the pred_sel dict.

    bin_borders can be taken directly from a TeamIndex instance, e.g. \"ti.quantiles\".
    """
    query_slice_list = list()
    grid_sizes = dict()
    hr_size = 1
    columns = list(bin_borders.keys())
    
    for column in columns:
        assert bin_borders[column][0] == -np.inf and bin_borders[column][-1] == np.inf
        b = len(bin_borders[column])-1
        grid_sizes[column] = b

        pred_bin_sel = math.floor(pred_sel[column]*b)
        assert pred_bin_sel > 0
        # pred_bin_sel = max(1, min(b-1, pred_bin_sel))
        hr_size *= pred_bin_sel
        query_slice_list.append(slice(0, pred_bin_sel))
        print(f"Column {column}: {pred_bin_sel} bins selected out of {b} total bins. (Initial selectivity: {pred_sel[column]:.2%})")
    

    total_size = math.prod(s for s in grid_sizes.values())
    print(f"Hyperrectangle size in {len(columns)}-dim space:", hr_size, "bins vs.", total_size)
    print(f"Volume: {hr_size/total_size:.8%}")
    initial_query_slice_dict = {"-".join(columns): tuple(query_slice_list)}
    random_queries = list()
    for _ in range(query_count):
        
        random_query_slices = apply_translation_vector(initial_query_slice_dict,
                                                       compute_translation_vector(initial_query_slice_dict,
                                                                                  grid_sizes))
        query_string = evaluation.slicer_to_conjunctive_query(list(random_query_slices.values())[0], columns, bin_borders)

        random_queries.append(query_string)
    return random_queries