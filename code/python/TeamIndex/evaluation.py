# This Python file uses the following encoding: utf-8
import os

from TeamIndex.lib._TeamIndex import run_read_all, run, TeamMetaInfo, RequestInfo, StorageConfig, PlanConfig
from TeamIndex.lib._TeamIndex import CodecID, liburingBackendConfig, StorageBackendID, ExecutorConfig
from TeamIndex.lib._TeamIndex import string_to_backend_id, backend_id_to_string, codec_id_to_string

from TeamIndex import creation

from itertools import chain, count as itcount

from pathlib import Path, PosixPath
from typing import List, Optional
import re
import json

import math

import numpy as np

import copy

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy

def get_new_default_runtime_config(worker_count=32, backend="liburing"):
    runtime_config = {
                'worker_count': worker_count,
                'implementation': "standard",  # options:  only_io, standard
                'backend': backend,  # options: liburing, dram
                'experiment_name': None,
                'verbose_runtime': False,
                'task_graph_path' : None,  # a file "./test_task_profile.json"
                'print_execution_plan': None, # a file "./test_execution_plan.json"
                'print_task_stats': None, # a file "./task_stats.json". Will be modified with the execution date
                'print_result_stats': None, # a file "./result_stats.json"
                'return_result': True,
                'StorageConfig': {
                    'submission_batch_size': 4,
                    'await_batch_size': 4,
                    'queue_pair_count': 16,
                    'liburingBackendConfig': {
                        'queue_depth': 256,
                        'o_direct': True,
                        'sq_poll': True,
                        'io_poll': False,
                        'sq_thread_idle': 4000
                    }
                },
                'OptimizerConfig': {
                    'request_merge_dist': 0,  # 0 means no merging, 1 means merge adjacent requests. Larger values allow for gaps (should be in pages)
                    'ise_cpl_limit': 100, ## unused?
                    'ise_limit': 20, ## unused?
                    'ignore_empty_teams': True,  # if True, we ignore Teams that select nothing or everything
                    'allow_exclusion': True,  # if True, we allow SUBTRACTING COMPLEMENTS of teams -> their ids are "excluded", not "intersected"
                    'allow_exclusion_expansion': True,
                    'exclusion_threshold': 0.55, ## unused?
                    'assign_groups_per_request': True,
                    'order_requests': False,
                    'leaf_union_max_size': 128, ## unused?
                    'leaf_union_group_max_size': 8, ## unused?
                    'leaf_union_grouping_threshold': 8, ## unused?
                    'leaf_union_list_parallel_threshold': 8,
                    'distributed_intersection_parallel_threshold': 10000
                }
            }
    return runtime_config

def _refine_min_max_by_predicate(edges, new_min, new_max, op, value):
    """
    For half-open bins [edges[i], edges[i+1]), we keep bin i if it intersects
    the set implied by (op, value). For '>' or '>=' => edges[i+1] > value,
    for '<' or '<=' => edges[i] < value.

    edges must be a numeric array, possibly containing -np.inf and np.inf
    at the extremes. new_min, new_max is our existing slice range.
    """
    # Ensure edges is a float array
    edges = np.asarray(edges, dtype=float)
    nbins = len(edges) - 1

    # Distinguish partial overlap rules:
    if op in (">", ">="):
        # bins whose upper boundary is > value
        valid_bins = np.where(edges[1:] > value)[0]
    elif op in ("<", "<="):
        # bins whose lower boundary is < value
        valid_bins = np.where(edges[:-1] < value)[0]
    else:
        raise ValueError(f"Unsupported op: {op}")

    if valid_bins.size == 0:
        return (0, 0)

    candidate_min = valid_bins[0]
    candidate_max = valid_bins[-1] + 1

    # Intersect with the existing [new_min, new_max)
    final_min = max(new_min, candidate_min)
    final_max = min(new_max, candidate_max)
    if final_min >= final_max:
        return (0, 0)

    return (final_min, final_max)

def complement_slice(slicers, max_shape):
    """
    Given a tuple of slicers (each either a slice(...) or a 1D array of indices)
    and a list of integers 'max_shape' (one for each dimension),
    return a boolean mask array of shape 'max_shape' that is True exactly
    where the slicer does NOT include that cell (the complement).

    Example:
       slicers = (slice(1,3), np.array([0,2]))
       max_shape = [5, 4]
       -> returns a 5x4 mask with True for rows outside 1..2 or columns not in {0,2}.
    """
    # First build an inclusion mask (False by default)
    included_mask = np.ones(max_shape, dtype=bool)

    ndims = len(max_shape)
    for dim in range(ndims):
        # dimension_mask will mark which indices are included along this dimension
        dim_size = max_shape[dim]
        dim_inclusion = np.zeros(dim_size, dtype=bool)

        if isinstance(slicers[dim], slice):
            # Mark indices in that slice range as True
            start, stop, step = slicers[dim].start, slicers[dim].stop, slicers[dim].step
            # handle None as 0 or dim_size
            start = 0 if (start is None) else start
            stop  = dim_size if (stop is None) else stop
            step  = 1 if (step is None) else step
            dim_inclusion[start:stop:step] = True
        else:
            # slicers[dim] is a 1D array of indices
            dim_inclusion[slicers[dim]] = True

        # Reshape dim_inclusion for broadcasting
        # e.g. if dim=1 and shape=(5,4,6), we want (1,4,1) shape
        shape_ones = [1]*ndims
        shape_ones[dim] = dim_size
        dim_inclusion_reshaped = dim_inclusion.reshape(shape_ones)

        # Combine with global included_mask using logical AND
        included_mask = included_mask & dim_inclusion_reshaped

    # The complement is everything not included
    return ~included_mask



# def _select_complements_if_smaller(inv_lists_sizes, slices, threshold=0.52):
#     """
#     Takes the complement of the selection, if it results in smaller I/O volume.
#     """

#     total_size = inv_lists_sizes.sum()  # total size

#     # size of the current selection:
#     sel_size = inv_lists_sizes[slices].sum()
#     if sel_size / total_size <= threshold:  # Note: no worth taking the complement for minor gains
#         return False, slices  # a normal selection
#     else:
#         # complement of the selection is smaller, we should load that instead!
#         complement = complement_slice(slices, inv_lists_sizes.shape)
#         if sel_size + inv_lists_sizes[complement].sum() != total_size:
#             assert False
#         return True, complement


def _merge_requests(ranges, dist=1, keep_ranges=False):
    """
    Merge overlapping and adjacent ranges and yield the merged ranges
    in order. The argument must be an iterable of pairs (start, stop).
    We assume that "stop" in each pair is inclusive.

    "dist" governs merging behaviour.
    dist=0 indicates only merges pairs that fully overlap, i.e., where start > end.
    dist=1 also merges pairs that are consecutive.
    dist=2 also merges pairs, where end and next start 1 or less appart, etc.

    "keep_starts=True" also returns a list of starts that go into each returned,
    combined range.

    Empty ranges get dropped.

    >>> list(_merge_requests([(5,7), (3,5), (-1,3)]))
    [(-1, 7)]
    >>> list(_merge_requests([(5,6), (3,4), (1,2)]))
    [(1, 2), (3, 4), (5, 6)]
    >>> list(_merge_requests([]))
    []

    Code adopted from:
    https://codereview.stackexchange.com/a/21333
    """
    if dist is None:
        dist = 0

    ranges = iter(sorted(ranges))
    try:
        current_start, current_stop = next(ranges)
    except StopIteration:
        return
    local_ranges = list()
    if keep_ranges:
        local_ranges.append((current_start, current_stop))
    for start, stop in ranges:

        if start == stop:
            if keep_ranges:
                local_ranges.append((start, stop))
            continue
        
        if start > int(current_stop) + int(dist)-1:
            # Gap between segments: output current segment and start a new one.
            if keep_ranges:
                yield current_start, current_stop, local_ranges
                local_ranges = [(start, stop)]
            else:
                yield current_start, current_stop
            current_start, current_stop = start, stop
        else:
            # Segments adjacent or overlapping: merge.
            if keep_ranges:
                local_ranges.append((start, stop))
            current_stop = max(current_stop, stop)

    if keep_ranges:
        yield current_start, current_stop, local_ranges
    else:
        yield current_start, current_stop

# def multiplicative_knapsack(lst, max_value):
#     """
#     Given a list of non-zero integers and a threshold max_value,
#     return the positions (indices) of a subset of lst such that the
#     product of the selected numbers is less than max_value.
#     We choose the subset that uses as many integers as possible.

#     Note: should only be efficient for small lists, as it is a brute-force approach.

#     Query that generated this code snippet:
#         In python, I have a list of integers and an threshold value max_value.

#         Write me a python function, potentially using numpy and/or the math lib, which returns the position (in the initial list) of values whose product is smaller than max_value.
#         I would like to use as many integers from the list as possible.

#         Assume that neither integer is 0 and that the list is not very long, i.e., between about 1 and 20 integer values.

#         This seems to be a "multiplicative knapsack" problem.
#     """
#     best_solution = []  # will store the best (largest) subset of indices

#     def backtrack(start, current_product, current_indices):
#         nonlocal best_solution
#         # If the current subset is larger than the best found so far, update best_solution.
#         if len(current_indices) > len(best_solution):
#             best_solution = current_indices.copy()
#         # Optional pruning: if even using all remaining items we cannot beat best_solution, stop.
#         if len(current_indices) + (len(lst) - start) <= len(best_solution):
#             return
#         # Try to include each of the remaining items.
#         for i in range(start, len(lst)):
#             new_product = current_product * lst[i]
#             if new_product < max_value:
#                 # Choose item i and continue searching
#                 current_indices.append(i)
#                 backtrack(i + 1, new_product, current_indices)
#                 current_indices.pop()  # backtrack

#     backtrack(0, 1, [])
#     return best_solution

def po2_near_sqrt(x):
    """
    Given a positive number x, return a power of 2 that is roughly equal to sqrt(x).
    """
    if x < 0:
        raise ValueError("x must be non-negative")
    if x == 0:
        return 0
    sqrt_x = math.sqrt(x)
    # Find the exponent by taking log base 2 of the square root and rounding to the nearest, lower integer
    exponent = math.floor(math.log(sqrt_x, 2))
    return 2 ** exponent

def _determine_groups(merged_list_sizes, group_count, list_count, assign_per_request=False, verbose=False):
    assert(group_count > 0)
    if group_count == 1:
        # i.e., all lists are unioned in the same group
        return [np.repeat(0, len(list_sizes)).astype(np.uint32) for list_sizes in merged_list_sizes], group_count, group_count
    elif group_count == list_count:
        # i.e., every list is in its own group, i.e., there will be no  per-group-union!
        counter = itcount()
        return [np.array([next(counter) for _ in list_sizes], dtype=np.uint32) for list_sizes in merged_list_sizes], 1, 1

    ## In all other cases, we employ a greedy algorithm to get roughly evenly sized groups (in terms of union size)
    cummulative_union_size = [0 for _ in range(group_count)]
    cummultive_union_cardinality = [0 for _ in range(group_count)]
    result = list()

    for list_sizes in merged_list_sizes:
        if assign_per_request:
            # i.e., all lists in this request share the same GroupID
            size = sum(list_sizes)
            group_id = int(np.argmin(cummulative_union_size))
            cummulative_union_size[group_id] += size
            cummultive_union_cardinality[group_id] += len(list_sizes)  # we assign multiple groups at once
            result.append(np.repeat(group_id, len(list_sizes)))
        else:
            # i.e., each list in this request can have a different GroupID
            group_ids = list()
            for size in list_sizes:
                group_id = int(np.argmin(cummulative_union_size))
                cummulative_union_size[group_id] += size
                cummultive_union_cardinality[group_id] += 1
                group_ids.append(group_id)
            result.append(np.array(group_ids, dtype=np.uint32))
    if verbose:
        print("Group-Union cardinalities:", dict(zip(range(group_count),cummulative_union_size)))
    return result, min(cummultive_union_cardinality), max(cummultive_union_cardinality)

def _dict_to_plan_config(global_info: dict):
    pcfg = PlanConfig()
    pcfg.table_cardinality = global_info["table_cardinality"]
    pcfg.ise_count = global_info["ise_count"]
    pcfg.outer_union_term_count = global_info["outer_union_term_count"]
    pcfg.outer_union_group_count = global_info["outer_union_group_count"]
    pcfg.outer_intersection_term_count = global_info["outer_intersection_term_count"]
    pcfg.outer_intersection_group_count = global_info["outer_intersection_group_count"]
    pcfg.leaf_union_list_parallel_threshold = global_info["leaf_union_list_parallel_threshold"]
    pcfg.distributed_intersection_parallel_threshold = global_info["distributed_intersection_parallel_threshold"]
    return pcfg

def _build_partition_matrix(offsets, counts, data, shape):
    """
    Build a multi-dimensional object array where each cell corresponds
    to one bin in the partition, containing the variable-length slice
    of `data` that belongs to that bin.

    Parameters
    ----------
    offsets : np.ndarray of shape (N,) or (N+1,)
        The start offset (in 32-bit elements) for each bin. If you have
        N bins, you might have N offsets plus one extra for the upper boundary.
    counts : np.ndarray of shape (N,)
        Number of valid elements for each bin.  If not given, one usually
        computes length from consecutive offsets, e.g. offsets[i+1]-offsets[i].
    data : np.ndarray
        The raw 1D array read from the file (dtype=uint32 or whichever).
    shape : tuple
        The shape of the partition (e.g. (nx, ny, nz, …)), so that
        N = nx * ny * nz * … matches the number of bins.

    Returns
    -------
    partition : np.ndarray of shape `shape`, dtype=object
        An N-dimensional object array where each cell is a 1D subarray
        (NumPy slice) of the underlying data.
    """
    # N is total number of bins
    N = np.prod(shape)

    # If offsets only has size=N instead of N+1, you’ll need a boundary:
    # e.g. offsets = np.append(offsets, [len(data)]) or something similar.

    partition = np.empty(shape, dtype=object)

    # Fill each cell with the proper slice
    for i in range(N):
        start = offsets[i]
        length = counts[i]  # or offsets[i+1] - start if no separate `counts`.
        cell_data = data[start : start + length]

        # Convert linear index `i` into ND coordinates
        coords = np.unravel_index(i, shape)
        partition[coords] = cell_data

    return partition


class TeamIndex(object):
    """
    Class that handles access to a TeamIndex.
    """

    def __init__(self, cfg_or_file_path, compression: Optional[str] = None):
        """
        Constructor for TeamIndex class from specified config/json.

        cfg_or_file_path: str or Path
            Path to the config file or a dictionary with the configuration.
        compression: str
            Compression type to use. If None, the first compression in the config file will be used.
        """

        if type(cfg_or_file_path) in {str, Path, PosixPath}:
            cfg_or_file_path = Path(cfg_or_file_path)
            if not (cfg_or_file_path.exists()
                    and cfg_or_file_path.is_file()):
                raise ValueError("Path to config \'{}\' does not exist or is no file!".format(cfg_or_file_path))
            cfg = creation.open_json(cfg_or_file_path)
        else:
            cfg = cfg_or_file_path

        if type(cfg) is not dict:
            raise TypeError("Argument \'cfg_or_file_path\' has to have either str, pathlib.PosixPath, pathlib.Path or dict as type!")

        if compression is None:
            if "compressions" not in cfg.keys():
                raise Exception("Config file faulty, no compressions defined!")
            compression = cfg["compressions"][0]  # first compression in list is default
        else:
            if compression not in cfg["compressions"]:
                raise Exception("Compression \'{}\' not available in config file!".format(compression))

        self.path = Path(cfg["index_folder"])
        if not (self.path.is_dir() and self.path.exists()):
            raise Exception("Index folder \'{}\' does not exist, can not load index!".format(self.path))
        # print("Opening Index at \'{}\' ({} compression)".format(self.path, compression))
        self.compression = compression

        ## check if sample queries were stored in the config
        if "queries" in cfg.keys():
            self.queries = cfg["queries"]
        else:
            self.queries = None

        # "-" char not allowed in attribute names!:
        if np.any(list("-" in attr for team in cfg["teams"] for attr in team)):
            raise Exception("\'-\' characters not allowed in attribute names, check configuration!")

        # if cfg["id_type"] not in ["np.uint32", "np.uint16", "np.uint64"]:
        #     raise Exception("ID type \'{}\' not supported, check configuration!\n"
        #                     "\tSupported types: {}".format(cfg["id_type"], ["np.uint32", "np.uint16", "np.uint64"]))
        #
        # self.id_type = eval(cfg["id_type"])  # we expect something like "np.uint32"
        self.id_type = np.uint32

        self.teams = dict()
        for team_attributes in cfg["teams"]:
            team_file_name = "-".join(team_attributes)
            self.teams[team_file_name] = team_attributes


        ## TODO rename quantiles to "bin_edges" or similar
        self.quantiles = {attr: np.array([-np.inf]+q+[np.inf]) for attr, q in cfg["quantiles"].items()}
        if "special_values" in cfg.keys():
            self.special_values = cfg["special_values"]
        else:
            self.special_values = {}

        if set.union(*list(set(t) for t in cfg["teams"])) != self.quantiles.keys():
            raise Exception("Not all Team attributes have quantiles defined, check configuration!")

        # inner prefix for .lists files, e.g., => ".copy.lists":
        inv_lists_pfx = "." + compression + ".lists"
        bc_sfx = "." + compression + ".cardinalities"
        offset_sfx = "." + compression + ".offsets"
        size_sfx = "." + compression + ".sizes"
        codecs_sfx = "." + compression + ".codecs"

        # data structures used on the python side to derive I/O requests for later execution
        self.cardinalities = dict()
        self.offsets = dict()
        self.compressed_sizes = dict()
        self.codecs = dict()

        self.file_paths = dict()

        self.stats = dict()
        self.stats["block_size"] = 4096  # TODO: Import value from C++ lib
        self.stats["team_shapes"] = dict()
        self.stats["number_of_tuples"] = 0
        self.stats["index_in_memory_structure_sizes"] = dict()
        self.stats["index_size_on_disk"] = dict()
        self.stats["compressed_size"] = dict()
        self.stats["compression_savings"] = dict()
        self.stats["total_padding"] = dict()
        self.stats["codec_usage"] = dict()

        # C++ class, that will handle the query execution/index evaluation. Requires I/O requests as input
        # self.executor = TeamIndexExecutor()
        self.team_list_files = dict()

        for team_name, team_attributes in self.teams.items():

            # derive shape of multidimensional arrays, that hold cardinalities and offsets for this team:
            team_shape = tuple([len(self.quantiles[attr]) - 1 for attr in team_attributes])
            # check all relevant files, names are implied by the .yaml-config
            self.team_list_files[team_name] = self.path.joinpath(team_name + inv_lists_pfx)
            assert self.team_list_files[team_name].exists(), "Path {} doest not exist!".format(self.team_list_files[team_name])

            self._off_paths = self.path.joinpath(team_name + offset_sfx)
            assert self._off_paths.exists(), "Path {} doest not exist!".format(self._off_paths)

            self._size_paths = self.path.joinpath(team_name + size_sfx)
            assert self._size_paths.exists(), "Path {} doest not exist!".format(self._size_paths)

            self._codecs_paths = self.path.joinpath(team_name + codecs_sfx)
            assert self._codecs_paths.exists(), "Path {} doest not exist!".format(self._codecs_paths)
           
            self._cards_path = self.path.joinpath(team_name + bc_sfx)
            assert self._cards_path.exists(), "Path {} doest not exist!".format(self._cards_path)

            # print("Opening data for Team ", team_name)
            # print("\tShape:", team_shape)
            # print("\tinv_list_path:", inv_list_path)
            # print("\toff_paths:", off_paths)
            # print("\tcards_path:", cards_path)

            # load offsets and cardinalities for later formulation of I/O requests.
            # we assume, these files were written via numpy's binary dump:
            cardinalities: np.ndarray = np.fromfile(self._cards_path, dtype=self.id_type)
            offsets: np.ndarray = np.fromfile(self._off_paths, dtype=self.id_type)
            sizes: np.ndarray = np.fromfile(self._size_paths, dtype=np.uint64)
            codec_ids: np.ndarray = np.fromfile(self._codecs_paths, dtype=np.uint8)

            # did we read the appropriate number of values?
            assert cardinalities.size == math.prod(team_shape), "Cardinalities file length mismatch. Expected {}, got {}".format(math.prod(team_shape), cardinalities.size)
            assert offsets.size == math.prod(team_shape) + 1

            # calculate the compressed inverted list sizes from the offsets (Note: empty bins repeat the last offset):
            self.cardinalities[team_name] = cardinalities.reshape(team_shape)
            self.offsets[team_name] = offsets  # contains one value more (the total size in blocks) than the other arrays
            self.compressed_sizes[team_name] = sizes.reshape(team_shape)
            self.codecs[team_name] = codec_ids.reshape(team_shape)

            # calculate some stats:
            self.stats["team_shapes"][team_name] = team_shape
            self.stats["index_size_on_disk"][team_name] = self.team_list_files[team_name].stat().st_size
            self.stats["compressed_size"][team_name] = int(self.compressed_sizes[team_name].sum().sum())
            savings = int((self.cardinalities[team_name].astype(np.uint64)*self.id_type().nbytes).sum().sum()) - int(self.stats["compressed_size"][team_name])
            self.stats["compression_savings"][team_name] = savings
            total_padding = int(self.stats["index_size_on_disk"][team_name] - self.stats["compressed_size"][team_name])
            self.stats["total_padding"][team_name] = total_padding

            self.stats["codec_usage"][team_name] = {
                                                    CodecID(code): int(count)
                                                    for code, count in zip(*np.unique(self.codecs[team_name], return_counts=True))
                                                    }            
            if self.stats["index_size_on_disk"][team_name] != int(self.offsets[team_name][-1])*self.stats["block_size"]:
                print("Warning: Mismatch of filesize and statistics!",
                      self.stats["index_size_on_disk"][team_name],
                      int(self.offsets[team_name][-1])*self.stats["block_size"])

            self.stats["index_in_memory_structure_sizes"][team_name] = self.cardinalities[team_name].nbytes \
                                                                       + self.compressed_sizes[team_name].nbytes \
                                                                       + self.offsets[team_name].nbytes \
                                                                       + self.codecs[team_name].nbytes

            
            self.stats["number_of_tuples"] = max(int(self.cardinalities[team_name].sum()), self.stats["number_of_tuples"])
        # print("Index opened.")

        self.default_runtime_config = get_new_default_runtime_config()

    def _make_histogram_slicer(self, query: str, columns: list):
        """
        Produces a python slice tuple from a query, using quantiles.
        The query is a string with conditions on columns, e.g. "A > 0.05 and B < 0.3".
        The columns list is the list of column names in the order they appear in the quantiles.

        - Each dimension is initially [0, nbins).
        - For each condition col op val (with op in {>, >=, <, <=}), we refine that dimension's
        [min_bin, max_bin) by intersecting with the bins that overlap the condition.
        - The result is a tuple of slices for each dimension, or we might produce fancy indexing
        if the user extends it to handle 'OR' or multiple disjoint intervals.

        For simplicity, this version only does a single pass over conditions joined by 'and'.

        Example usage:
            query_str = "A > 0.05 and B < 0.3"
            slicers = index._make_histogram_slicer(query_str, ["A", "B"])
            subhist = hist[slicers]
        """
        import re

        # Regex to find (column op value)
        pattern = re.compile(
            r'([a-zA-Z0-9_]+)\s*(>=|<=|>|<)\s*([\-\+]?\d*\.?\d+(?:[eE][\-\+]?\d+)?)'
        )

        # Initialize [min_bin, max_bin) = entire range for each column
        dim_ranges = {}
        for col in columns:
            nbins = len(self.quantiles[col]) - 1
            dim_ranges[col] = [0, nbins]  # inclusive lower, exclusive upper

        # Parse the query by splitting on 'and' for a purely conjunctive example
        conditions = [c.strip() for c in query.split("and")]

        for cond in conditions:
            match = pattern.search(cond)
            if not match:
                continue
            col_name, op, val_str = match.groups()
            value = float(val_str)

            if col_name not in dim_ranges:
                continue  # or raise an error if col_name not recognized

            # Refine the existing slice [new_min, new_max) by the bins that overlap
            # the condition (op, value)
            new_min, new_max = dim_ranges[col_name]
            final_min, final_max = _refine_min_max_by_predicate(
                self.quantiles[col_name], new_min, new_max, op, value
            )

            dim_ranges[col_name] = [final_min, final_max]

        # Build final slice objects for each column
        slice_tuple = []
        for col in columns:
            min_bin, max_bin = dim_ranges[col]
            slice_tuple.append(slice(min_bin, max_bin))

        return tuple(slice_tuple)
    
    def prepare_optimization(self, query, optimizer_config=None, verbose=False, **optimization_overrides):
        """
        This function determines relevant Teams and takes complement of selections, if it results in smaller I/O volume.
        We only consider simple, conjunctive queries!

        We emit a list of Teams, ordered in ascending order of ID count (BEFORE considering complements).
        We also attach slices and information about inclusion/exclusion and some meta data for convenience.

        """

        optimizer_config = optimizer_config or self.default_runtime_config["OptimizerConfig"]

        for k, v in optimization_overrides.items():
            if k in optimizer_config:
                if verbose:
                    print("Patching optimizer option {} with {}".format(k, v))
                optimizer_config[k] = v

        slices_dict = self.query_to_slices(query)
        optimization_config = dict()

        for team_name, slices in slices_dict.items():            
            cards = self.cardinalities[team_name][slices]
            union_card = cards.sum()
            if cards.sum() in [0, self.cardinalities[team_name].sum()] and optimizer_config["ignore_empty_teams"]:
                # if index selects NOTHING OR EVERYTHING (in terms of lists), we skip it
                if verbose:
                    print("Skipping Team", team_name, "as it offers no selectivity/selects everything.")
                continue
            optimization_config[team_name] = dict()
            optimization_config[team_name]["union_cardinality"] = int(union_card)

            ## add some fields to be potentially altered by optimization procedures
            optimization_config[team_name]["is_included"] = True  # will be set to False later, if complement is taken instead
            # the following will not be altered (only initialized) by this function:
            optimization_config[team_name]["is_expanded"] = False  # by default, nothing will be expanded
            optimization_config[team_name]["group_count"] = np.count_nonzero(cards)  # maximum group count means no grouping at all!
            # to make the config more assessible by a human reader
            optimization_config[team_name]["max_group_count"] = optimization_config[team_name]["group_count"]
            optimization_config[team_name]["netto_data_volume_KiB"] = int(self.compressed_sizes[team_name][slices].sum()) // 1024
            io_vol = int((np.ceil(self.compressed_sizes[team_name][slices] / self.stats["block_size"]) * self.stats["block_size"]).sum()) // 1024
            optimization_config[team_name]["io_volume_KiB"] = io_vol



        included_team_count = 0  # used to make sure we have at least one included Team (the smallest one is default)
        opts = list()
        for team_name in sorted(optimization_config.keys(), key=lambda t: optimization_config[t]["union_cardinality"]):
            card = self.cardinalities[team_name].sum()
            optimization_config[team_name]["selectivity"] = float(optimization_config[team_name]["union_cardinality"] / card)
            if verbose:
                print(team_name,"- Team selectivity:", optimization_config[team_name]["selectivity"])
        
            if included_team_count > 0 and optimizer_config["allow_exclusion"]:
                # Team is considered "excluded", if complement was taken:
                complement_preferred = (optimization_config[team_name]["union_cardinality"] / card) > optimizer_config["exclusion_threshold"]
                if complement_preferred:
                    optimization_config[team_name]["is_included"] = False
                    # need to alter the stats
                    new_slices = complement_slice(slices_dict[team_name], self.cardinalities[team_name].shape)
                    cards = self.cardinalities[team_name][new_slices]
                    if verbose:
                        print("\tTaking complement...")
                        print("\tBefore:", optimization_config[team_name]["union_cardinality"])
                        print("\tAfter:", cards.sum())
                    optimization_config[team_name]["union_cardinality"] = int(cards.sum())
                    optimization_config[team_name]["group_count"] = np.count_nonzero(cards)
                    
                else:
                    included_team_count += 1
            else:
                included_team_count += 1
            opts.append((team_name, optimization_config[team_name]))
            
        return opts

    def _optimize_access(self, query, optimizer_options=None, manual_optimizations=None, verbose=False, **kwargs):
        """
        Given the initial (trivial) optimizations, we now apply "grouping" and "expansion" optimizations.
        
        If manual_optimizations are provided, we will use the "group_count" value and "is_expanded" flag without further modifications.

        kwargs is used to patch the optimizer_options, e.g., for testing purposes. Only used to overwrite default values.
        """

        optimizer_options = optimizer_options or self.default_runtime_config["OptimizerConfig"]

        for k, v in kwargs.items():
            if k in optimizer_options:
                if verbose:
                    print("Patching optimizer option {} with {}".format(k, v))
                optimizer_options[k] = v


        if manual_optimizations is not None:
            assert(len(manual_optimizations) > 0), "No Teams to optimize!"
            assert(manual_optimizations[0][1]["is_included"]), "First Team must be included!"

        slices_dict = self.query_to_slices(query)

        if manual_optimizations is None:
            # get initial optimization options, which will be modified further below
            optimizations = self.prepare_optimization(query, optimizer_options, verbose=verbose)
        else:
            # when optimzations are provided, we will not alter them further
            optimizations = manual_optimizations
        
        ## determine group counts
        minimum_list_count = np.inf
        total_input_cardinality = 0
        total_read_volume = 0
        total_compressed_size = 0

        for team, opt in optimizations:
            slices = slices_dict[team]

            if not opt["is_included"]:
                slices = complement_slice(slices, self.cardinalities[team].shape)
            cards = self.cardinalities[team][slices]
            union_card = cards.sum()
            lst_count = np.count_nonzero(cards)
            minimum_list_count = min(lst_count, minimum_list_count)
            
            assert(opt["union_cardinality"] == union_card), "Union cardinality missmatch!"

            ## determine group counts and expansion
            if manual_optimizations is None:
                # square-root rule:
                # po2_near_sqrt(lst_cnt) if lst_cnt > 8 else lst_cnt

                # only alter group count, if it is large enough
                if opt["group_count"] > optimizer_options["leaf_union_grouping_threshold"]:
                    # we may want to avoid huge groups of too many lists, so we determine how many groups we need at least
                    min_group_count = math.ceil(lst_count / optimizer_options["leaf_union_max_size"])
                    opt["group_count"] = min(max(min_group_count, po2_near_sqrt(opt["group_count"])), lst_count)
                
            else:
                assert(opt["group_count"] > 0), "Group count must be positive!"
                assert(opt["group_count"] <= lst_count), "Impossible to have more groups than lists!"
            
            total_input_cardinality += union_card
            total_read_volume += opt["io_volume_KiB"]
            total_compressed_size += self.compressed_sizes[team][slices].sum() 


        ## determine which Teams to expand, if any
        # Generally, we eagerly expand, but stop, if ISE count becomes too high
        # Our heuristic is to balance ISE count and ise_path_length

        # if we expand, we always expand the smallest Team. Each expanded Team contributes just one term to the ISE length.
        # The following term is the maximum length and it shortens for every expansion (-> minus group_count+1 terms)       

        if manual_optimizations is None:
            ## We start expanding the smallest Team, if it is not already expanded
            current_ise_path_length = 1+sum([opt["group_count"] for _, opt in optimizations[1:]])
            ise_count = optimizations[0][1]["group_count"]
            for team, opt in optimizations:
                ## TODO
                pass

        else:
            ise_count = math.prod([opt["group_count"]
                                   for _, opt in optimizations
                                   if opt["is_expanded"]])
        expanded_team_count = sum([1 for _, opt in optimizations if opt["is_expanded"]])
        included_team_count = sum([1 for _, opt in optimizations if opt["is_included"]])
        outer_union_term_count = math.prod([opt["group_count"]
                                            for _, opt in optimizations
                                            if opt["is_expanded"] and opt["is_included"]]) if expanded_team_count > 0 else 0
        # Note: if no Team is expanded, we will have one as many intersections as there are included Teams!
        # Excluded Teams are subtracted instead of intersected.
        outer_intersection_term_count = math.prod([opt["group_count"]
                                                   for _, opt in optimizations
                                                   if opt["is_expanded"] and not opt["is_included"]]) if expanded_team_count > 0 else included_team_count

        # having no "big" union term implies no team is excluded 
        # This usually also means there are no ISEs (or just a single one)
        outer_union_group_count = po2_near_sqrt(outer_union_term_count) if outer_union_term_count > 8 else outer_union_term_count
        
        # Intersections are few. Also, having only one intersection term means there will not be any intersection.
        outer_intersection_group_count = po2_near_sqrt(outer_intersection_term_count) if outer_intersection_term_count > 8 else outer_intersection_term_count

        global_info = dict()
        global_info["ise_count"] = ise_count if expanded_team_count > 0 else 0
        global_info["table_cardinality"] = self.stats["number_of_tuples"]
        global_info["outer_union_term_count"] = outer_union_term_count 
        global_info["outer_union_group_count"] = outer_union_group_count
        global_info["outer_intersection_term_count"] = outer_intersection_term_count
        global_info["outer_intersection_group_count"] = outer_intersection_group_count
        global_info["total_input_cardinality"] = int(total_input_cardinality)
        global_info["total_read_volume_KiB"] = float(total_read_volume)
        global_info["total_compressed_size_KB"] = float(total_compressed_size // 1000)

        return optimizations, global_info

    # def _optimize_access(self, query, optimizer_config, verbose=False):
    #     """
    #     This function optimizes index access by considering the following optimizations:
    #     - Skip Teams that offer no selectivity/select everything, e.g., if no predicate was defined for them.
    #     - Take the complement of a selection, if it results in smaller I/O volume. This results in set subtraction operations, instead of intersection
    #     - Expand a Team's terms by pushing intersections from non-expanded Teams into their union. This allows better parallelism and early pruning.
    #     - Partition large union operations (over many partial results/lists) by splitting the union into groups, aggregating the result in two stages.
    #       For expanded Teams, this leads to better parallelism and a shorter, final aggregation step
    #       For distributed Teams, this leads to a smaller critical path for each "independent sub expression", which is largely executed sequentially
    #     """

    #     if "expansion_group_count_heuristic" not in optimizer_config:
    #         optimizer_config["expansion_group_count_heuristic"] = lambda lst_cnt: po2_near_sqrt(lst_cnt) if lst_cnt > 8 else lst_cnt
    #     if "distributed_group_count_heuristic" not in optimizer_config:                
    #         optimizer_config["distributed_group_count_heuristic"] = lambda lst_cnt, budget: min(po2_near_sqrt(lst_cnt), po2_near_sqrt(budget)) if lst_cnt > budget else lst_cnt

    #     slices_dict = self.query_to_slices(query)

    #     assert(optimizer_config["ise_cpl_limit"] > len(slices_dict)), "ISE critical path can not be shorter than the number of Teams"
    #     skip_team = dict()
    #     team_is_excluded = dict()

    #     team_volumes = dict()
    #     team_list_counts = dict()
    #     group_counts = dict()
        
    #     # determine smallest Team, which will be treated differently during processing
    #     smallest_team_size = np.inf
    #     smallest_team = None
    #     for team_name, slices in slices_dict.items():
    #         sizes = self.compressed_sizes[team_name] 
    #         # Alternatively, we could use the cardinalities, but this is more accurate..?

    #         ### FIRST optimization: skip teams entirely that offer no selectivity/select everything
    #         if sizes[slices].sum() in [0, sizes.sum()]:
    #             skip_team[team_name] = True
    #             print("Skipping Team", team_name, "as it offers no selectivity/selects everything.")
    #             continue
    #         skip_team[team_name] = False 
        
    #         size = sizes[slices_dict[team_name]].reshape(-1).sum()
    #         if size < smallest_team_size:
    #             smallest_team_size = size
    #             smallest_team = team_name

    #     for team_name, slices in slices_dict.items():
    #         if skip_team[team_name]:
    #             continue

    #         ### SECOND optimization: take the complement of a selection, if it results in smaller I/O volume
    #         # Note that "complement Teams" are subtracted from other Teams, not intersected with
    #         if (team_name != smallest_team) and optimizer_config["allow_exclusion"]:
    #             # will only be done for teams other than the smallest one
    #             team_is_excluded[team_name], slices_dict[team_name] = _select_complements_if_smaller(self.cardinalities[team_name], slices)
    #         else:
    #             team_is_excluded[team_name] = False
            
    #         # track some stats, for internal usage in this function
    #         relevant_cardinalities = self.cardinalities[team_name][slices_dict[team_name]]
    #         team_volumes[team_name] = relevant_cardinalities.sum()
    #         team_list_counts[team_name] = np.count_nonzero(relevant_cardinalities)  # i.e., number of relevant lists
        
    #     ### THIRD optimization: "expand" a Team's terms to obtain more parallelism
    #     # We determine which Teams to expand. May be none!
        
    #     ### FOURTH optimization:
    #     # Consider groups of lists (instead of individual lists):
    #     # For distributed Teams, we group to fit a maximum critical path length requirement for the respective ISE
    #     # For expanded Teams, we group
    #     # - individual lists, to reduce the number of ISEs, dampening the combinatorial explosion of ISEs
    #     # - the ISE terms to improve the parallelism and reduce critical path length of the "outer" union/intersection

    #     expandable_team_names = [
    #                              t
    #                              for t in team_list_counts.keys() 
    #                              if (not team_is_excluded[t]) or optimizer_config["allow_exclusion_expansion"]
    #                             ]
        
    #     # determine which Team to expand, assuming we are grouping some of it's lists via union, using a heuristic
    #     # expandable_team_list_cnts = [team_list_counts[t] for t in expandable_team_names]
    #     expandable_team_group_cnts = [
    #                                   optimizer_config["expansion_group_count_heuristic"](team_list_counts[t])
    #                                   for t in expandable_team_names
    #                                  ]
    #     expansion_team_positions = multiplicative_knapsack(expandable_team_group_cnts, optimizer_config["ise_limit"])
        
    #     expanded_teams = list(np.array(expandable_team_names)[expansion_team_positions])
    #     distributed_teams = np.array([k for k in team_list_counts.keys() if k not in expanded_teams])

    #     inclusive_expanded = list(sorted([str(t) for t in expanded_teams if not team_is_excluded[t]], key=lambda t: team_volumes[t]))
    #     exclusive_expanded = list(sorted([str(t) for t in expanded_teams if team_is_excluded[t]], key=lambda t: team_volumes[t]))
    #     inclusive_distributed = list(sorted([str(t) for t in distributed_teams if not team_is_excluded[t]], key=lambda t: team_volumes[t]))
    #     exclusive_distributed = list(sorted([str(t) for t in distributed_teams if team_is_excluded[t]], key=lambda t: team_volumes[t]))


    #     ## group counts for Distributed Teams:
    #     ise_path_length = len(expanded_teams)-1
    #     remaining_budget = optimizer_config["ise_cpl_limit"]-ise_path_length
        
    #     # current heuristic:
    #     # Give each Team the same number of groups, such that each team has an about equal number of terms in the ISE critical path
        
    #     for team in distributed_teams:
    #         team = str(team)
    #         team_share = remaining_budget // len(distributed_teams)  # every distributed Team gets about the same number of terms
    #         assert(team_share > 0)
    #         group_count = optimizer_config["distributed_group_count_heuristic"](team_list_counts[team], team_share)
    #         ise_path_length += group_count
    #         group_counts[team] = group_count
        
    #     assert(ise_path_length <= optimizer_config["ise_cpl_limit"])

    #     if verbose:
    #         print("ISE path length:", ise_path_length, "(Remaining budget:", optimizer_config["ise_cpl_limit"]-ise_path_length,")")

    #     # Group count will be 1, if there are only few lists/terms.
    #     # If there are more, we balance the number of terms in the inner (group size) and outer union (group count)
    #     # So this is roughly the square root of the the number of lists

    #     ise_count = 1
    #     outer_union_term_count = 1
    #     outer_intersection_term_count = 1

    #     for team in expanded_teams:
    #         team = str(team)
    #         group_counts[team] = optimizer_config["expansion_group_count_heuristic"](team_list_counts[team])
            
    #         ise_count *= group_counts[team]
    #         if team in inclusive_expanded:
    #             outer_union_term_count *= group_counts[team]
    #         else:
    #             outer_intersection_term_count *= group_counts[team]
        
    #     outer_union_group_count = po2_near_sqrt(outer_union_term_count) if outer_union_term_count > 8 else outer_union_term_count
    #     outer_intersection_group_count = po2_near_sqrt(outer_intersection_term_count) if outer_intersection_term_count > 8 else outer_intersection_term_count

    #     if verbose:
    #         print(ise_path_length, "(Unused budget:", optimizer_config["ise_cpl_limit"]-ise_path_length,")")
    #         print("Included expansion teams:", inclusive_expanded)
    #         print("Excluded expansion teams:", exclusive_expanded)
    #         print("Included distributed teams:", inclusive_distributed)
    #         print("Excluded distributed teams:", exclusive_distributed)
    #         print("Team List Counts: ", team_list_counts)
    #         shared_list_cnt = sum([group_counts[t] for t in distributed_teams])
    #         print("Expansion will produce {} ISEs.".format(ise_count))
    #         print("Each ISE shares {} components.".format(shared_list_cnt))
    #         print("Group Counts:", group_counts)

    #     global_info = dict()
    #     global_info["ise_count"] = ise_count
    #     global_info["outer_union_term_count"] = outer_union_term_count
    #     global_info["outer_union_group_count"] = outer_union_group_count
    #     global_info["outer_intersection_term_count"] = outer_intersection_term_count
    #     global_info["outer_intersection_group_count"] = outer_intersection_group_count

    #     return [inclusive_expanded, exclusive_expanded, inclusive_distributed, exclusive_distributed], slices_dict, group_counts, global_info

    def query_to_slices(self, query: str, optimizations=None):
        slices_dict = dict()
        # convert query to "slices" which are used to select relevant meta data with convenient array access.
        if optimizations is not None:
            for team_name, opt in optimizations:
                slices = self._make_histogram_slicer(query, self.teams[team_name])
                if not opt["is_included"]:
                    # This Team's IDs will not be included, but instead excluded via subtraction.
                    # We therefore need to consider the complement, i.e., invert the selection/predicates for this Team
                    slices = complement_slice(slices, self.cardinalities[team_name].shape)
                slices_dict[team_name] = slices
        else:
            for team_name, team_attributes in self.teams.items():
                # convert predicates to slices for each team
                slices_dict[team_name] = self._make_histogram_slicer(query, team_attributes)

        return slices_dict

    def _create_requests(self, query: str, 
                         optimizer_cfg: Optional[dict] = None, 
                         manual_optimizations: Optional[List] = None, 
                         verbose: bool = False, 
                         **optmization_overwrites):
        

        if optimizer_cfg is None:
            optimizer_cfg = self.default_runtime_config["OptimizerConfig"]
        
        # check optimizer config
        assert isinstance(optimizer_cfg, dict), "Optimizer config must be a dictionary!"
        assert (optimizer_cfg["request_merge_dist"] in {0, 1} or
                (not optimizer_cfg["request_merge_dist"] % 2) or optimizer_cfg["request_merge_dist"] is None)

        # for each team, create the requests. But first, optimize!
        # We determine how each team's information is included and how partial results are formed
        optimizations, global_info = self._optimize_access(query,
                                                           optimizer_cfg,
                                                           manual_optimizations=manual_optimizations, 
                                                           verbose=verbose, 
                                                           **optmization_overwrites)
        
        req_id = 0  # assign a number to each request, starting from 0
        requests = list()
        rstats = dict()
        team_info = list()

        def append_requests(team_name, slices, group_count, is_included:bool , is_expanded: bool):
            nonlocal req_id
        
            relevant_bin_cardinalities = self.cardinalities[team_name][slices].reshape(-1)
            relevant_offsets = self.offsets[team_name][:-1].reshape(self.cardinalities[team_name].shape)[slices].reshape(-1)
            relevant_il_sizes = self.compressed_sizes[team_name][slices].reshape(-1)
            relevant_codecs = self.codecs[team_name][slices].reshape(-1)
            
            # skip entries for empty bins:
            non_zero_bins = np.nonzero(relevant_bin_cardinalities)
            relevant_bin_cardinalities = relevant_bin_cardinalities[non_zero_bins]
            relevant_offsets = relevant_offsets[non_zero_bins]
            relevant_il_sizes = relevant_il_sizes[non_zero_bins]
            relevant_codecs = relevant_codecs[non_zero_bins]

            # formulate actual work-items ("RequestInfo" objects), one for each I/O request
            # May merge requests, if they are adjacent/close enough. This reduces system call overhead for I/O.
            team_reqs, min_group_size, max_group_size = self._create_request_infos(req_id,
                                                                                    team_name,
                                                                                    relevant_bin_cardinalities,
                                                                                    relevant_offsets,
                                                                                    relevant_il_sizes,
                                                                                    relevant_codecs,
                                                                                    group_count,
                                                                                    optimizer_cfg["request_merge_dist"],
                                                                                    optimizer_cfg["assign_groups_per_request"])
            requests.extend(team_reqs)
            req_id += len(team_reqs)


            rstats[team_name] = dict()
            rstats[team_name]["request_count"] = len(team_reqs)
            rstats[team_name]["combined_id_count"] = int(relevant_bin_cardinalities.sum())
            # rstats[team_name]["cardinalities"] = relevant_bin_cardinalities
            rstats[team_name]["combined_request_size"] = sum([int(req.total_block_cnt*self.stats["block_size"])
                                                              for req in team_reqs])
            rstats[team_name]["combined_list_count"] = sum([len(req.decomp_info) for req in team_reqs])
            # may differ from request sizes due to both padding and merging:
            rstats[team_name]["combined_list_size"] = int(relevant_il_sizes.sum())
            assert(relevant_il_sizes.sum() == sum([list_info[3] for req in team_reqs for list_info in req.decomp_info]))
            rstats[team_name]["combined_read_amplification"] = float(rstats[team_name]["combined_request_size"]/
                                                                     rstats[team_name]["combined_list_size"])
            rstats[team_name]["average_request_size_byte"] = float(rstats[team_name]["combined_request_size"]/rstats[team_name]["request_count"])
            rstats[team_name]["is_included"] = is_included
            rstats[team_name]["is_expanded"] = is_expanded
            rstats[team_name]["group_count"] = int(group_count)
            rstats[team_name]["min_group_size"] = int(min_group_size)
            rstats[team_name]["max_group_size"] = int(max_group_size)           

            team_info.append(TeamMetaInfo(team_name,
                                        rstats[team_name]["combined_list_size"],
                                        rstats[team_name]["combined_id_count"],
                                        rstats[team_name]["request_count"],
                                        rstats[team_name]["combined_list_count"],
                                        str(self.team_list_files[team_name].absolute()),
                                        is_included,
                                        is_expanded,
                                        group_count,
                                        min_group_size,
                                        max_group_size))
            
        slices_dict = self.query_to_slices(query)

        for team_name, opts in optimizations:
            slices = slices_dict[team_name]
            if not opts["is_included"]:
                # This Team's IDs will not be included, but instead excluded via subtraction.
                # We therefore need to consider the complement, i.e., invert the selection/predicates for this Team
                slices = complement_slice(slices, self.cardinalities[team_name].shape)
            append_requests(team_name, slices, opts["group_count"], opts["is_included"], opts["is_expanded"])
 
        return requests, team_info, rstats, global_info



    def _create_request_infos(self,
                              req_id_start,
                              team_name,
                              rbcs,
                              roffs,
                              rsizes,
                              rcodecs,
                              group_count,
                              request_merge_dist,
                              assign_per_request):
        """
        For a given Team, create a list of RequestInfo objects.

        We also determine offsets within a request, which are used to perform decompression.
        Note that an I/O request may span multiple lists.

        """
        # roffs example values:
        # [219, 223, 227, 231, 235,..]

        # Note: offsets are a number of blocks, not a byte offset! We instead use rsizes, which is the compressed size of each list in byte
        blocksizes = (rsizes+self.stats["block_size"]-1)//self.stats["block_size"]
        # e.g., [4, 4, ...], one size per list

        request_ranges = zip(roffs, roffs + blocksizes)
        # e.g., "[(219, 223), (223, 227), ...]", one entry per list
        
        # Note that we only merged requests that are already adjacent in the initial arguments to this function
        # length corresponds to the number of requests for this Team
        request_ranges = list(_merge_requests(request_ranges, dist=request_merge_dist, keep_ranges=True))
        # e.g.,  "[(np.uint32(219), np.uint64(239), [(np.uint32(219), np.uint64(223)),...,np.uint64(239))]), (np.uint32(319), np.uint64(339),[..."
        # NOW one tuple per REQUEST

        # bring other arrays into the same shape by grouping values that are in the same merged request into one list
        split_points = np.cumsum([len(r) for _, _, r in request_ranges])[:-1]
        # e.g., array([ 5, 10]), i.e., one entry per request - 1

        rbcs = np.split(rbcs, split_points)
        # e.g., [array([3266, 3187, 3141, 3218, 3237], dtype=uint32), array([3197, 3208, 3231, 3264, 3155], dtype=uint32), array([3210, 3168, 3188, 3146, 3251], dtype=uint32)]
        rcodecs = np.split(rcodecs, split_points)
        # e.g., [array([1, 1, 1, 1, 1], dtype=uint8), array([1,1,..
        rsizes = np.split(rsizes, split_points)
        
        team_reqs = list()
        group_id_lists, min_group_size, max_group_size = _determine_groups(rbcs,
                                                                           group_count,
                                                                           len(request_ranges),
                                                                           assign_per_request=assign_per_request)
        # e.g., [array([0,1,2,3], dtype=uint32), array([0,1,2], dtype=uint32), ...

        for rid, rr, bcs, codecs, sizes, group_ids in zip(np.arange(req_id_start,
                                                                    req_id_start + len(request_ranges)),
                                                                    request_ranges,
                                                                    rbcs,
                                                                    rcodecs,
                                                                    rsizes,
                                                                    group_id_lists):
            # one iteration for each request, which may span multiple lists
            req_start, req_end, list_ranges = rr  # rr is a value from request_ranges
            decomp_info = [(ls-req_start, lcard, CodecID(lcodec), lsize, group_id)
                           for (ls, _), lcard, lcodec, lsize, group_id in zip(list_ranges, bcs, codecs, sizes, group_ids)
                           if lsize > 0]
            if len(decomp_info) > 0:  # we do not emit a request if there are no lists with non-zero cardinality!
                # RequestInfo(RequestID, TeamName, StartBlock, BlockCount, [ <Offset, ILCardinality, ILSizeCompressed> ])
                team_reqs.append(RequestInfo(rid, team_name, req_start, req_end-req_start, decomp_info))

        return team_reqs, min_group_size, max_group_size

    def _group_requests(self, requests, queue_count, team_infos, order_requests):
        """
        Greedy algorithm that splits a list of requests in queue_count-many lists.
        Given that requests may be very different in size, we try to equalize the overall volume
        assigned to each worker.

        Should work best with order_requests == True.
        """
        # first, we split requests for the primary team and the rest
        team_request_lists = [[r for r in requests if r.team_name == team_info.team_name] for team_info in team_infos]
        
        if order_requests:
            req_order_lists = [np.array([req.total_block_cnt for req in team_reqs]).argsort()[::-1]
                               for team_reqs in team_request_lists]
            team_request_lists = [list(np.array(team_reqs, dtype=object)[req_order])
                                  for team_reqs, req_order in zip(team_request_lists, req_order_lists)]

        if queue_count > 1:
            # the following structures keep/track the result of this function
            grouped_requests = [list() for w in range(queue_count)]
            cumulative_sums = [0 for w in range(queue_count)]
            
            # first assign primary team requests, which are always needed first
            for team_reqs in team_request_lists:
                for req in team_reqs:
                    # Determine queue to assign:
                    # We greedily assign to the first queue that has the smallest cumulative_sums so far
                    wid = np.argmin(cumulative_sums)
                    ## assign request
                    cumulative_sums[wid] += req.total_block_cnt
                    grouped_requests[wid].append(req)
        else:
            grouped_requests = [list(chain.from_iterable(team_request_lists))]  # concatenate all request lists into the only queue

        return grouped_requests

    def enlarge_query(self, query: str, quantiles = None) -> str:
            """
            Adjusts each boundary in a simple conjunctive query to the next-smaller (for '>','>=')
            or next-larger (for '<','<=') quantile. This code follows half-open interval logic:
            - If a '>' boundary is lowered, it becomes '>=' to include that boundary value.
            - If a '<' boundary is raised, it remains '<' so the new boundary is excluded.
            - For '>=' or '<=', no operator change occurs upon shifting.
            - No shift happens if the threshold is already one of the quantiles or if there's no
                strictly smaller/larger quantile.
            """
            if quantiles is None:
                quantiles = self.quantiles
            # Regex to capture (attribute, operator, float-value)
            pattern = re.compile(r"(\w+)\s*(>=|<=|>|<)\s*([\d\.eE\+\-]+)")

            def adjust_threshold(attribute, operator, raw_value):
                value = float(raw_value)

                # Skip attributes not in quantiles
                if attribute not in quantiles:
                    return f"{attribute} {operator} {repr(value)}"

                q_vals = quantiles[attribute]

                # Handle '>' or '>=' by looking for the largest quantile strictly below 'value'
                if operator in (">", ">="):
                    # If the threshold is exactly a known quantile, make sure it's inclusive
                    if value in q_vals:
                        return f"{attribute} >= {repr(value)}"
                    
                    smaller_candidates = [float(q) for q in q_vals if q < value]
                    if smaller_candidates:
                        new_boundary = max(smaller_candidates)
                        if new_boundary < value:
                            # # Only shift if genuinely smaller
                            # # '>' becomes '>=' to include that boundary
                            # # '>=' stays '>=' if it was that already
                            # if operator == ">":
                            #     return f"{attribute} >= {new_boundary}"
                            return f"{attribute} >= {repr(new_boundary)}"
                            # return f"{attribute} {operator} {new_boundary}"
                    return f"{attribute} >= {repr(value)}"

                # Handle '<' or '<=' by looking for the smallest quantile strictly above 'value'
                elif operator in ("<", "<="):
                    # If the threshold is exactly a known quantile, make sure it's inclusive
                    if value in q_vals:
                        return f"{attribute} {operator} {value}"
                    larger_candidates = [q for q in q_vals if q > value]
                    if larger_candidates:
                        new_boundary = min(larger_candidates)
                        if new_boundary > value:
                            # 'operator' remains the same, bin is half-open and therefore excludes the new boundary
                            return f"{attribute} {operator} {new_boundary}"
                    return f"{attribute} {operator} {repr(value)}"

                # Fallback should never occur if we only parse >, >=, <, <=
                return f"{attribute} {operator} {repr(value)}"

            # Replace each condition with the adjusted version
            return pattern.sub(lambda m: adjust_threshold(m.group(1), m.group(2), m.group(3)), query)


    def split_query_by_partition(self, query: str) -> dict:
        """
        Splits a simple conjunctive query (only 'and' connectives) into sub-queries
        according to a provided set partition of attributes.

        
        :param query: A string like 
            "Kplus_TRACK_CHI2_PER_NDOF > 1.5 and piminus_TRACK_CHI2_PER_NDOF < 0.5 and B0_LOKI_DTF_CTAU > 0"
        :return: A dict with the same keys as partition_dict, where each value
            is the sub-query (joined by " and ") containing conditions that match
            the attributes belonging to that key. If no conditions matched, returns "".
        """

        # Regex to capture: <attribute> <operator> <numeric-value>
        # e.g. "Kplus_TRACK_CHI2_PER_NDOF >= 1.0", capturing groups: attribute, operator, value
        pattern = re.compile(r'([a-zA-Z0-9_]+)\s*([<>]=?|==|!=)\s*([\-\+]?[\d\.eE]+)')

        # Split the query into individual conditions by "and"
        # We'll re-match them with the above pattern to extract attribute/operator/value cleanly
        raw_conditions = [cond.strip() for cond in query.split('and')]

        # Prepare a dict of lists to collect conditions for each partition key
        partition_conditions = {key: [] for key in self.teams}

        # For each raw condition, parse out attribute/operator/value
        for cond in raw_conditions:
            match = pattern.search(cond)
            if not match:
                # If there's any condition that doesn't match the pattern,
                # skip it or handle as needed.
                continue

            attr, op, val = match.groups()

            # Find which partition(s) this attribute belongs to.
            # In most typical scenarios, each attribute belongs to exactly one partition.
            # But if, for some reason, it belongs to multiple, we'll add the condition there too.
            for pkey, pattrs in self.teams.items():
                if attr in pattrs:
                    partition_conditions[pkey].append(cond)
                    # If an attribute can only appear in one partition, you could break here.

        # Build sub-queries by joining conditions with ' and '
        result = {}
        for pkey, conds in partition_conditions.items():
            # Join relevant conditions into one string
            sub_query = " and ".join(conds) if conds else ""
            result[pkey] = sub_query

        return result
    
    def run_query(self, query: str,
                  config: Optional[dict] = None,
                  manual_optimizations: Optional[List] = None ,
                  dry_run: Optional[Path] = None,
                  experiment_name: str = None,
                  **optmization_overwrites):
        
        config = config or copy.deepcopy(self.default_runtime_config)

        # set environment variable for TaskFlow, it will dump the task flow graph to file indicated by the variable:
        if "task_graph_path" in config:
            if config["task_graph_path"] is not None:
                os.environ["TF_ENABLE_PROFILER"] = str(config["task_graph_path"])

        requests, team_infos, request_info, global_info = self._create_requests(query,
                                                                                config["OptimizerConfig"],
                                                                                manual_optimizations,
                                                                                verbose = config["verbose_runtime"],
                                                                                **optmization_overwrites)

        ## add some user side configuration options to be passed on to runtime
        global_info["leaf_union_list_parallel_threshold"] = config["OptimizerConfig"]["leaf_union_list_parallel_threshold"]
        global_info["distributed_intersection_parallel_threshold"]= config["OptimizerConfig"]["distributed_intersection_parallel_threshold"]

        global_info["compression"] = self.compression


        global_info["total_request_count"] = sum(request_info[team_name]["request_count"]
                                                   for team_name in request_info.keys())
        if global_info["total_request_count"] == 0:
            print("Nothing to do, index offers no selectivity and returns all IDs!")
            return range(0, self.stats["number_of_tuples"]), None, request_info, global_info

        # group requests into a number of lists equal to the I/O queue count
        # goal is to balance the load between each queue (in terms of I/O volume)
        grouped_requests = self._group_requests(requests,
                                                config["StorageConfig"]["queue_pair_count"],
                                                team_infos,
                                                config["OptimizerConfig"]["order_requests"])  # one group for each worker
        
        
        ecfg = ExecutorConfig()
        ecfg.worker_count = config["worker_count"]
        if config["print_execution_plan"] is not None:
            ecfg.print_execution_plan = str(config["print_execution_plan"])
        if config["print_result_stats"] is not None:
            ecfg.print_result_stats = str(config["print_result_stats"])
        if config["print_task_stats"] is not None:
            ecfg.print_task_stats = str(config["print_task_stats"])
            if experiment_name is not None:
                ecfg.experiment_name = experiment_name
            elif "experiment_name" in config:
                ecfg.experiment_name = config["experiment_name"]

        ecfg.return_result = config["return_result"]
        ecfg.verbose = config["verbose_runtime"]

        if "backend" in config:
            ecfg.backend = string_to_backend_id(config["backend"])
        else:
            ecfg.backend = StorageBackendID.liburing


        liburingcfg = liburingBackendConfig(config["StorageConfig"]["liburingBackendConfig"]["queue_depth"],
                                            config["StorageConfig"]["liburingBackendConfig"]["o_direct"],
                                            config["StorageConfig"]["liburingBackendConfig"]["sq_poll"],
                                            config["StorageConfig"]["liburingBackendConfig"]["io_poll"],
                                            config["StorageConfig"]["liburingBackendConfig"]["sq_thread_idle"])
        acfg = StorageConfig(config["StorageConfig"]["submission_batch_size"],
                             config["StorageConfig"]["await_batch_size"],
                             config["StorageConfig"]["queue_pair_count"],
                             liburingcfg)
        

        if dry_run is not None:
            if not isinstance(dry_run, Path):
                return grouped_requests, team_infos, request_info, global_info, ecfg, acfg
            assert dry_run.parent.exists(), "Dry run path does not exist!"
            serialize_workload_to_json(dry_run, query, grouped_requests, team_infos, global_info, ecfg, acfg)
            # print("Serialized workload to", dry_run)
            return

        if config["implementation"] == "only_io":
            result, runtime_stats = run_read_all(grouped_requests, team_infos, ecfg, acfg)
        elif config["implementation"] == "standard":
            pcfg = _dict_to_plan_config(global_info)
            result, runtime_stats = run(grouped_requests, team_infos, pcfg, ecfg, acfg)

        else:
            raise RuntimeError("Implementation " + config["implementation"] + " not supported!")
        
        # if runtime_stats.plan_execution_runtime is not None:

        return result, runtime_stats, request_info, global_info

    def get_index_data(self, slices_dict: dict):
        """
        Numpy-based index access. Can be used to perform index intersection manually.

        Loads index files to memory and partitions them into a multi-dimensional object array,
        according to meta data (cardinalities/offsets).
        It then uses python slices to select relevant bins for each team, based on the query.

        """

        data = dict()

        for team_name, slices in slices_dict.items():
            
            path = self.path.joinpath(team_name + ".copy.lists")
            offset_path = self.path.joinpath(team_name + ".copy.offsets")
            offsets = np.fromfile(offset_path, dtype=np.uint32)
            if not path.exists():
                raise FileNotFoundError(f"File {path} not found - python-based retrieval only possible with uncompressed lists! ")

            lists = np.fromfile(path, dtype=np.uint32)
            
            cards = self.cardinalities[team_name]
            offs = 4096//4*offsets.astype(np.uint64)
            
            result = _build_partition_matrix(offs, cards.flatten(), lists, cards.shape)

            data[team_name] = result[slices]
        return data

    def plot_kde_with_entropy(self, file_path, query=None, grid_resolution=2000, kde_method=0.2):
        def _calculate_entropy(values):
            values, counts = np.unique(values, return_counts=True)
            probabilities = counts / counts.sum()
            return entropy(probabilities, base=2)

        plt.figure(figsize=(12, 8))

        for name, matrix in self.cardinalities.items():
            # Apply query if provided
            if query and name in query:
                matrix = matrix[query[name]]

            values = matrix.flatten()
            total_values = len(values)

            # Separate zeros and non-zero values
            zeros = values[values == 0]
            non_zeros = values[values != 0]

            # KDE for non-zero values
            if non_zeros.size > 0:
                kde = gaussian_kde(non_zeros, bw_method=kde_method)
                x_grid = np.linspace(non_zeros.min(), non_zeros.max(), grid_resolution)
                kde_values = kde(x_grid)
                plt.plot(x_grid, kde_values, label=f'{name} (Entropy: {_calculate_entropy(values):.3f})')

            # Plot zeros explicitly as relative frequency
            if zeros.size > 0:
                zero_freq = len(zeros) / total_values
                plt.scatter(0, zero_freq, marker='o', s=100, label=f'{name} zeros freq: {zero_freq:.3f}')

        plt.xlabel('List Cardinality')
        plt.ylabel('Density')
        plt.yscale('log')
        if query:
            plt.title('Cardinality Distributions with Query')
        else:
            plt.title('Cardinality Distributions')
        plt.legend()
        plt.grid(True)
        plt.savefig(file_path, format='pdf')
        plt.close()


def slicer_to_conjunctive_query(slicers, columns, bin_borders):
    """
    Given:
    - slicers: A tuple of slicing objects (one per dimension). Each slicer can be:
        slice(start, stop[, step])   -> contiguous bin indices
        (We ignore step if not None, for simplicity)
        bool array of shape (#bins,)
        1D array of bin indices
    - columns: The list of attribute/dimension names, same order as slicers
    - bin_borders: dict mapping each column name -> sorted numeric array of bin edges
                    e.g. edges_dict["B0_LOKI_DTF_CTAU"] = np.array([-inf, ..., inf])
    Produces:
    - A string that is an AND of dimension-specific predicates,
        each in half-open form [edge[i], edge[i+1]).

    Example:
    If slicers = (
        slice(0,10),          # dimension 0 - no restriction => no clause
        slice(0,10),          # dimension 1 - no restriction => no clause
        slice(8,10),          # dimension 2 - restricted to bins 8..9
        slice(6,10)           # dimension 3 - restricted to bins 6..9
    )
    => we build predicates for dimension 2 & 3 only, e.g.:
        "dim2 >= edges[8] and dim2 < edges[10] and dim3 >= edges[6] and dim3 < edges[10]"
    (if those edges are not ±inf).
    """
    def fmt_edge(val):
        """
        Convert float edge (possibly ±inf) to a nice string.
        If it's integral or close, may want to shorten. 
        """
        if np.isinf(val):
            return "inf" if val > 0 else "-inf"
        return f"{repr(float(val))}"
    
    clauses = []

    for col, slicer in zip(columns, slicers):
        edges = bin_borders[col]
        nbins = len(edges) - 1

        # Convert the slicer to a sorted array of selected bin indices
        if isinstance(slicer, slice):
            start = 0 if slicer.start is None else slicer.start
            stop  = nbins if slicer.stop is None else slicer.stop
            # If the slice covers all bins, skip
            if start <= 0 and stop >= nbins:
                continue
            # If empty => no data => the entire query is effectively False
            if start >= stop:
                return "False"
            selected_bins = np.arange(start, stop, 1, dtype=int)
        else:
            arr = np.asarray(slicer)
            if arr.dtype == bool:
                # bool mask
                if arr.size != nbins:
                    raise ValueError(f"Boolean slicer has wrong size for {col}")
                selected_bins = np.where(arr)[0]
            else:
                # array of indices
                selected_bins = arr

            if len(selected_bins) == 0:
                return "False"
            # If it covers all bins, skip
            if len(selected_bins) == nbins and np.all(selected_bins == np.arange(nbins)):
                continue
            selected_bins = np.unique(selected_bins)

        # Group the selected bin indices into consecutive runs
        runs = []
        rstart = rend = selected_bins[0]
        for b in selected_bins[1:]:
            if b == rend + 1:
                rend = b
            else:
                runs.append((rstart, rend))
                rstart = rend = b
        runs.append((rstart, rend))

        # Build a set of OR'ed intervals for this dimension
        # Each run (i0..i1) => [edges[i0], edges[i1+1])
        # If edges[i0] == -inf, skip the lower bound
        # if edges[i1+1] == inf, skip the upper bound
        subclauses = []
        for (i0, i1) in runs:
            lower = edges[i0]
            upper = edges[i1 + 1]  # half-open => < upper

            if np.isinf(lower) and lower < 0:
                if np.isinf(upper) and upper > 0:
                    # entire real line => no restriction, but we got a run => odd case
                    # Means [ -inf, +inf ), so no real condition
                    continue
                else:
                    # -inf up to upper
                    subclauses.append(f"{col} < {fmt_edge(upper)}")
            elif np.isinf(upper) and upper > 0:
                # [lower, +inf)
                subclauses.append(f"{col} >= {fmt_edge(lower)}")
            else:
                # [lower, upper)
                sc = f"{col} >= {fmt_edge(lower)} and {col} < {fmt_edge(upper)}"
                subclauses.append(sc)

        # If only one run => single clause
        if len(subclauses) == 1:
            clauses.append(subclauses[0])
        elif len(subclauses) > 1:
            # multiple intervals => combine with OR => parenthesize
            joined = " or ".join(subclauses)
            clauses.append(f"({joined})")
        # else if subclauses is empty => skip?

    # If we have no clauses => no dimension was restricted => all bins => empty query
    if not clauses:
        return ""

    # Join all dimension clauses with AND
    return " and ".join(clauses)

def translate_queries(queries, from_index, to_index):
    """
    Translate the queries from the attribute domains index by one index to the domain indexed  by another, e.g., from uniform to real.
    """
    translated_queries = []
    for query in queries:
        ti_from = TeamIndex(from_index)
        ti_to = TeamIndex(to_index)

        # use slice representation of the query to formulate the new query
        slice_dict = ti_from.query_to_slices(query)

        query_parts = list()
        for team, slices in slice_dict.items():
            # translate the slices back to a query string, using the new index's bin borders
            query_part = slicer_to_conjunctive_query(slices, ti_from.teams[team], bin_borders=ti_to.quantiles)
            query_parts.append(query_part)
        
        
        new_query = " and ".join(query_parts)
        translated_queries.append(new_query)
    
    return translated_queries

def serialize_optimizations_to_json(file_path: str, optimizations: List, query: str, experiment_name = "experiment_"):
    output_path = Path(file_path)
    data = {
        "experiment_name": experiment_name,  # to be altered by the user
        "optimizations": optimizations,
        "query": query
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, separators=(",", ": "))
    
    print(f"Query optimization options serialized to {output_path}")

def serialize_runtime_config_to_json(file_path: str, config: dict):
    output_path = Path(file_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, separators=(",", ": "))
    
    print(f"Runtime config serialized to {output_path}")


def serialize_workload_to_json(file_path: str,
                               query: str,
                               request_infos,
                               team_workload_infos,
                               plan_cfg,
                               executor_cfg,
                               storage_cfg,
                               other_info=None):
    output_path = Path(file_path)
    def format_decomp_info(decomp_info):
        return json.dumps(
                [[start_block, list_cardinality, codec_id_to_string(codec_id), list_size_compressed, group_id] 
                for start_block, list_cardinality, codec_id, list_size_compressed, group_id in decomp_info],
                separators=(",", ":")
            )
    
    data = {
        "query": query,
        "request_infos": [
            [
                {
                    "rid": req.rid,
                    "team_name": req.team_name,
                    "start_block": req.start_block,
                    "total_block_cnt": req.total_block_cnt,
                    "decomp_info": json.loads(format_decomp_info(req.decomp_info))
                }
                for req in req_group
            ]
            for req_group in request_infos
        ],
        "team_workload_infos": [
            {
                "team_name": team.team_name,
                "total_size_comp": int(team.total_size_comp),
                "total_cardinality": int(team.total_cardinality),
                "request_cnt": int(team.request_cnt),
                "list_cnt": int(team.list_cnt),
                "team_file_path": team.team_file_path,
                "is_included": bool(team.is_included),
                "expand": bool(team.expand),
                "group_count": int(team.group_count),
                "min_group_size": int(team.min_group_size),
                "max_group_size": int(team.max_group_size)
            }
            for team in team_workload_infos
        ],
        "executor_config": {
            "worker_count": executor_cfg.worker_count,
            "backend": backend_id_to_string(executor_cfg.backend),
            "verbose": executor_cfg.verbose,
            "return_result": executor_cfg.return_result
        },
        "storage_config": {
            "submit_batch_size": storage_cfg.submit_batch_size,
            "await_batch_size": storage_cfg.await_batch_size,
            "queue_pair_count": storage_cfg.queue_pair_count,
            "liburing_cfg": {
                "queue_depth": storage_cfg.liburing_cfg.queue_depth,
                "o_direct": storage_cfg.liburing_cfg.o_direct,
                "sq_poll": storage_cfg.liburing_cfg.sq_poll,
                "io_poll": storage_cfg.liburing_cfg.io_poll,
                "sq_thread_idle": storage_cfg.liburing_cfg.sq_thread_idle
            }
        },
        "plan_config": {
            "ise_count": plan_cfg["ise_count"],
            "table_cardinality": plan_cfg["table_cardinality"],
            "outer_union_term_count": plan_cfg["outer_union_term_count"],
            "outer_union_group_count": plan_cfg["outer_union_group_count"],
            "outer_intersection_term_count": plan_cfg["outer_intersection_term_count"],
            "outer_intersection_group_count": plan_cfg["outer_intersection_group_count"],
            "leaf_union_list_parallel_threshold": plan_cfg["leaf_union_list_parallel_threshold"],
            "distributed_intersection_parallel_threshold": plan_cfg["distributed_intersection_parallel_threshold"],
            
        }
    }

    # patch in potential None values:
    if executor_cfg.print_execution_plan:
        data["executor_config"]["print_execution_plan"] = str(executor_cfg.print_execution_plan)
    if executor_cfg.print_task_stats:
        data["executor_config"]["print_task_stats"] = str(executor_cfg.print_task_stats)
    if executor_cfg.print_result_stats:
        data["executor_config"]["print_result_stats"] = str(executor_cfg.print_result_stats)
    
    if executor_cfg.experiment_name:
        data["executor_config"]["experiment_name"] = str(executor_cfg.experiment_name)
    

    data.update(other_info or {})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, separators=(",", ": "))