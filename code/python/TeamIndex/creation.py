import hashlib
import numpy as np
import pandas as pd
# import yaml
import json

from pathlib import Path, PosixPath
from typing import Optional, List

from glob import glob

import pyarrow as pa
import pyarrow.parquet as pq

import re

from TeamIndex.lib._TeamIndex import BatchConverter, string_to_codec_id, codec_id_to_string, TableQuantizer, TableInverter



def create_configs(input_path, target_path, quantiles, team_compositions, queries, compressions, special_values):
    """
    Creates one config in a subfolder for each team. Use the variables above to define the each index and the associated queries!
    Each call to this function will always create new sub folders, regardless of which ones already exist.

    Experiments can be differentiated by a hash calculated by the config, which is stored in the config file (invariant to the index location).
    """
    hash_length = 10
    assert target_path.is_dir(), f"Target folder {target_path} does not exist!"
    if not input_path.exists():
        print(f"Warning: Input file {input_path} does not exist!")

    assert quantiles is not None, "Quantiles are None! Load them first (e.g., with open_json(\".../quantiles.json\")!"

    # always create new folders for the indices. We continue instances of a predefined naming schema
    next_index = 1 + max(
        (int(m.group(1)) for m in 
        (re.match(r'^team_(\d+)$', p.name) for p in target_path.iterdir() if p.is_dir())
        if m), 
        default=-1
    )
    print("Next index:", next_index)
    for i, team in enumerate(team_compositions, next_index):
        target_sub_folder = target_path / f"team_{i}/"
        config = create_config(
            str(target_sub_folder.absolute()),
            quantiles,
            team,
            compressions=compressions,
            source_table=str(input_path.absolute()),
            special_values=special_values,
            queries=queries,
        )

        # create a unique hash for this config for later use. Invariant to the location of the index, but dependent
        # on queries, quantiles and the composition. Used to differentiate experiments for file naming etc.
        config["tag"] = config_to_hash(config, hash_length)
        
        # patch the config with the target folder
        # config["index_folder"] = str(target_sub_folder.absolute())

        print("Creating folder", target_sub_folder)

        target_sub_folder.mkdir(parents=True, exist_ok=True)
        # dump the config to the target folder
        fpath = target_sub_folder / f"index_{i}.json"
        with open(fpath, "w") as stream:
            json.dump(config, stream, indent=4)
        print(f"Config {i} with tag {config["tag"]} written to {fpath}\n")

def find_matching_files(base_path: Path, pattern: str) -> list[Path]:
    return [p for p in base_path.rglob(pattern) if p.is_file()]

def find_matching_dirs(root_path: Path, pattern: str) -> list[Path]:
    return [p for p in root_path.rglob(pattern) if p.is_dir()]

def create_indices(target_path, file_glob="index_*.json"):
    target_path = Path(target_path)
    
    paths = find_matching_files(target_path, file_glob)
    for path in paths:
        print(f"Loading config from {path}")
        cfg = open_json(path)
        print(f"Creating index in {path.parent}...")
        index_table(cfg, do_not_dump=False, overwrite_existing=False)
        print("Index created!")


def determine_quantiles(quantile_count=10, target_folder=None, ifx="", special_values=None):
    global table
    assert table is not None, "Load table first with load_table()!"
    # special values get their own bins, otherwise quantile calculation will be messy (and it makes sense for later queries)

    quantiles = get_all_quantiles(table, quantile_count, special_values)
    if target_folder is not None:
        return quantiles
    target_folder = Path(target_folder)
    assert target_folder.is_dir(), f"Target folder {target_folder} does not exist!"

    with open(target_folder / f"quantiles{ifx}_{quantile_count}.json", "w") as stream:
        json.dump(quantiles, stream, indent=4)

# HELPER FUNCTIONS:
# def get_new_team_composition(team_count, attributes):
#     """
#     Create a new random team composition for the given number of teams and attributes.
#     The attributes are shuffled and then split into the number of teams.
#     """
#     attributes = attributes.copy()
#     table_dim = len(attributes)

#     # create evenly sized teams:
#     team_sizes = np.array(team_count*[table_dim//team_count])
#     team_sizes[:int(table_dim-team_sizes.sum())] += 1
#     assert team_sizes.sum() == table_dim

#     np.random.shuffle(attributes)

#     return sorted([[str(att) for att in team] for team in np.split(attributes, np.cumsum(team_sizes)[:-1])])


# def get_new_team_composition_unbalanced(attributes, max_dimensionality):
#     shuffled = attributes.copy()
#     np.random.shuffle(shuffled)
    
#     return [shuffled[i:i + max_dimensionality] for i in range(0, len(shuffled), max_dimensionality)]

def get_all_quantiles(df, bin_cnt, special_vals=None):
    qnts = dict()
    for col in df.columns:
        qnts[col] = get_quantiles(col, special_values=special_vals, data=df[col], bin_cnt=bin_cnt)
        print("Quantiles for column {}: {}".format(col, qnts[col]))

    return {str(key): [float(q) for q in array] for key, array in qnts.items()}


def get_quantiles(column, special_values, data, bin_cnt, verbose=False):
    # Access a single column and calculate quantiles. Special values can be defined and will have their own "bin".
    special_values = special_values or dict()

    if type(data) in {str, PosixPath}:
        pds = pa.parquet.ParquetDataset(data)
        assert column in pds.schema.names

        data = pds.read(columns=[column],
                        use_pandas_metadata=True).to_pandas(split_blocks=True, self_destruct=True)[column]

    if column in special_values:  # special values are placed in a separate bin
        def find_right_border(spec_val, data_):
            # for buckets with constant (special) value
            # find a value between the special value and the next actual value for use as border
            # For this to be robust after insertion, the logic for matching bins has to be specialized.
            # This work around is used to make special value bins behave the same way as regular bins (for the already indexed data)
            rhs = data_[spec_val < data_].min()
            if spec_val >= 0:
                return (rhs-spec_val) * 0.5
            return (spec_val-rhs) * 0.5
        
        masks = list()
        is_not_extreme_val = list()
        right_bin_borders = list()
        
        for sval in special_values[column]:
            masks.append(data == sval)
            right_bin_borders.append(find_right_border(sval, data))
            is_not_extreme_val.append((data.min() != sval)
                                      and (data.max() != sval))
        if len(special_values[column]) > 1:
            spec_val_mask = np.logical_or(*masks)
        else:
            spec_val_mask = masks[0]

        # 1 border for extreme values, 2 for special value-bins between regular bins:
        bin_border_cnt = bin_cnt - sum(np.array(is_not_extreme_val)+1)

        if bin_border_cnt <= 1:  # ignore special values if there are insufficiently many bins
            qnts = np.linspace(0, 1, bin_cnt+1)[1:-1]
        else:
            qnts = np.linspace(0, 1, bin_border_cnt+1)[1:-1]

        quantile_values = np.quantile(data[~spec_val_mask], qnts, axis=0).T

        # patch special bins in:
        if not bin_border_cnt <= 1:  # ignore special values if there are insufficiently many bins
            for sval, rbb, inev in zip(special_values[column], right_bin_borders, is_not_extreme_val):

                pos = quantile_values.searchsorted(rbb)
                quantile_values = np.insert(quantile_values, pos, rbb)

                if inev:
                    pos = quantile_values.searchsorted(sval)
                    quantile_values = np.insert(quantile_values, pos, sval)
                if verbose:
                    print("Bin for special value {} inserted!".format(sval))
    else:

        qnts = np.linspace(0, 1, bin_cnt+1)[1:-1]
        quantile_values = np.quantile(data, qnts, axis=0).T

    return quantile_values


def create_config(index_folder: str, quantiles: dict, teams: List[List[str]], compressions: List[str] = ["roaring"],
                  source_table: Optional[str] = None, special_values: Optional[dict] = None, queries: Optional[List[str]] = None):
    cfg = dict()
    cfg["index_folder"] = str(index_folder)
    cfg["compressions"] = compressions
    cfg["teams"] = teams
    index_columns = set([att for team in teams for att in team])
    cfg["quantiles"] = {att: quantiles[att] for att in index_columns}
    if queries is not None:
        cfg["queries"] = queries
    if source_table is not None:
        cfg["source_table"] = str(source_table)
    if special_values is not None:
        cfg["special_values"] = special_values
    return cfg


def write_config(_cfg: dict, cfg_file_path: Path):
    cfg = _cfg.copy()
    cfg_file_path = Path(cfg_file_path)
    assert cfg_file_path.parent.exists()

    assert "teams" in cfg.keys()
    assert "compressions" in cfg.keys()
    assert "quantiles" in cfg.keys()

    if "index_folder" not in cfg.keys():
        cfg["index_folder"] = "./"
    else:
        cfg["index_folder"] = str(cfg["index_folder"])

    if "source_table" in cfg.keys():
        cfg["source_table"] = str(cfg["source_table"])

    # with open(cfg_file_path, "w") as stream:
    #     yaml.safe_dump(cfg, stream, default_flow_style=False)
    with open(cfg_file_path, "w") as stream:
        json.dump(cfg, stream, indent=4)
    write_json(cfg, cfg_file_path)
    
def write_json(config: dict, file_path: Path):
    with open(file_path, "w") as stream:
        json.dump(config, stream, indent=4)


def open_json(file_path: Path):
    file_path = Path(file_path)
    assert file_path.exists()

    with open(file_path, "r") as stream:
        cfg = json.load(stream)
        return cfg


def config_to_hash(config: dict, hash_length = 10):
    cfg_str = json.dumps(config, separators=(',',':'), indent=4)
    return hashlib.sha256(cfg_str.encode()).hexdigest()[:hash_length]

def string_to_hash(string: str, hash_length = 10):
    """
    Creates a hash from a string. Used to create unique identifiers for the indices.
    """
    return hashlib.sha256(string.encode()).hexdigest()[:hash_length]

def get_team_quantiles(team, cfg: dict, fill_value=np.nan):
    """
    Converts the quantiles from the configuration into a pandas DataFrame.
    """
    qvals = dict()
    b_max = 0
    assert "quantiles" in cfg.keys()
    for col in team:
        qvals[col] = cfg["quantiles"][col]
        b_max = max(b_max, len(qvals[col]))

    for col in team:
        if len(qvals[col]) < b_max:
            assert fill_value is not None  # ERROR: missmatching number of quantiles, but no fill value specified!
            qvals[col].extend((b_max-len(qvals[col]))*[fill_value])
    return pd.DataFrame(qvals)

def merge_data_frames(glob_pattern: str, columns: List[str]) -> pd.DataFrame:
    """
    Append all data frames for a specific set of columns into a single data frame.
    """
    
    dfs = [pd.read_parquet(path, columns=columns) for path in glob(glob_pattern)]
    
    if len(dfs) == 0:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.reset_index(drop=True)
    
    print("Merged data frames into a single data frame with shape:", df.shape)
    
    return df

def index_table(cfg_or_file_path, table = None, do_not_dump: bool = False, overwrite_existing=False):
    """
    Create a Team index from a table, i.e., a pandas DataFrame.
    If none is provided, we look for a path specified in "source_table" in the configuration.


    For configuration, provide either a dictionary object or a path to a json file via cfg_or_file_path argument.

    May create multiple indices, if the "compressions" attribute in the configuration contains multiple compressions.

    Will not recreate indices that already exist.
    """

    if type(cfg_or_file_path) in {str, PosixPath}:
        cfg_or_file_path = Path(cfg_or_file_path)
        assert cfg_or_file_path.exists() \
               and cfg_or_file_path.is_file() \
               and cfg_or_file_path.suffix in {".json", ".cfg"}
        cfg = open_json(cfg_or_file_path)
    else:
        cfg = cfg_or_file_path
    assert type(cfg) is dict

    team_composition = cfg["teams"]
    compressions = cfg["compressions"]

    id_type = np.uint32

    if 'index_folder' not in cfg.keys():
        if type(cfg_or_file_path) is not dict:
            index_folder = cfg_or_file_path.absolute().parent
        else:
            index_folder = Path(".").absolute()
    else:
        index_folder = Path(cfg['index_folder']).absolute()

    if do_not_dump:
        index = dict()
    else:
        print("Creating index at \'{}\'".format(str(index_folder)))

    # get number of quantiles for each dimension
    shape = {att: len(quantile_values) for att, quantile_values in sorted(cfg["quantiles"].items())}

    for team in team_composition:
        team = tuple(team)

        if do_not_dump:
            index[team] = dict()
        print("\tTeam:", team)

        qvals = get_team_quantiles(team, cfg, fill_value=None)
        assert qvals.values.flags["F_CONTIGUOUS"]

        assert set(qvals.columns) == set(team)

        remaining_compr = list()

        ## prepare file names and configuration for all compressions:
        for compr_name in compressions:
            fname = "{}.{}".format("-".join(team), compr_name)
            if not do_not_dump:
                il_path = index_folder.joinpath(fname+".lists")
                bcs_path = index_folder.joinpath(fname+".cardinalities")
                offs_path = index_folder.joinpath(fname+".offsets")
                compr_sizes_path = index_folder.joinpath(fname+".sizes")
                codecs_path = index_folder.joinpath(fname+".codecs")

                if il_path.exists() and not overwrite_existing:
                    print("\tFiles for", compr_name, "already exist! Skipping to next compression/Team...")
                    assert offs_path.exists()
                    assert bcs_path.exists()
                    assert compr_sizes_path.exists()
                    continue
                remaining_compr.append((compr_name, il_path, bcs_path, offs_path, compr_sizes_path, codecs_path))
            else:
                remaining_compr.append((compr_name, None, None, None, None, None))

        if len(remaining_compr) == 0 and not do_not_dump:
            print("\tNothing to do, skipping team..")
            continue
        
        # load table from parquet, if none was provided as argument:
        if table is None:
            assert "source_table" in cfg.keys(), "No table provided and no source_table in configuration!"
            source_table = cfg["source_table"]
            assert source_table is not None, "source_table is None!"
            assert type(source_table) is str, "source_table must be a string!"
            
            if '*' in source_table:
                print("\tLoading Team's tables from glob", source_table)
                team_data = merge_data_frames(source_table, list(team)).to_numpy(copy=False)
            else:
                assert Path(source_table).exists(), f"\'{source_table}\' does not exist!"
                print("\tLoading Team's table from", source_table)
                team_data = pq.read_table(source_table, columns=list(team), memory_map=True).to_pandas().to_numpy(copy=False)
        else:
            assert isinstance(table, pd.DataFrame), "table must be a pandas DataFrame!"
            team_data = table[list(team)].to_numpy(copy=False) # we hope the table has homogeneous data types, or this may become a copy...
        arr_fortran = np.asfortranarray(qvals.values)  # actually unnecessary but we'll try to be safe
        converter = BatchConverter(arr_fortran)
        print("Preprocessing batch...")
        converter.process_batch(team_data, 0)  # currently all in one batch 
        
        for compr_name, il_path, bcs_path, offs_path, compr_sizes_path, codecs_path in remaining_compr:
            print("\tCreating index for team", team, "with", compr_name, "compression..")

            compr_id = string_to_codec_id(compr_name)

            # fetching results:
            if do_not_dump:
                inv_lists, bcs, offs, compr_sizes, codecs = converter.get_result(compr_id)

                team_shape = tuple([shape[att] for att in team])

                bcs = np.array(bcs, copy=False, dtype=id_type).reshape(team_shape)
                offs = np.array(offs, copy=False, dtype=id_type)
                compr_sizes = np.array(compr_sizes, copy=False, dtype=np.uint64)  # size_t
                codecs = np.array(codecs, copy=False, dtype=np.uint8)

                index[team][compr_name] = (inv_lists, bcs, offs, compr_sizes, codecs)
            else:
                index_folder.mkdir(exist_ok=True)
                codecs = converter.dump_index(compr_id,
                                              str(il_path),
                                              str(offs_path),
                                              str(compr_sizes_path),
                                              str(codecs_path),
                                              str(bcs_path))
                print("\tCompressed lists for Team", team, "written.")

    if do_not_dump:
        return index

####### New functions for quantization and inversion:


def quantize_table(table, quantiles):
    """
    Quantizes a table using the provided quantiles.
    :param table: The table to be quantized.
    :param quantiles_dict: The quantiles to be used for quantization.
    :return: The quantized table.
    """
    columns = table.columns.tolist()
    # get quantiles for all columns
    qs = [quantiles[col] for col in columns]  # need to take care this is in the same order
    bin_counts = [len(q) for q in qs]

    ## quantize table 
    qtable = TableQuantizer.quantize_table(table, qs, bin_counts)
    print("Quantized table shape:", qtable.shape)
    return qtable




def invert_by_partitions(quantized_table: pd.DataFrame,
                         team_composition: list[list[str]],
                         quantiles: dict,
                         compression="copy"):
    """
    Run the inversion separately for each subset of columns.

    Parameters:
    - quantized_table: quantized DataFrame, may be unsorted. Column names must match team_composition
    - team_composition: list of column name lists (e.g., [[c0..c7], [c8..c15]])
    - quantiles: dictionary of quantiles for each column
    - compression: compression codec name

    Returns:
    - List of tuples (blob, cardinalities, compressed_sizes, used_codecs, page_offsets)
    """
    assert isinstance(quantized_table, pd.DataFrame), "quantized_table must be a pandas DataFrame"
    assert quantized_table.values.flags["F_CONTIGUOUS"], "DataFrame must be in Fortran order (column-major)"
    
    ## check if all types are uint8
    assert all([quantized_table[col].dtype == np.uint8 for col in quantized_table.columns]), "quantized_table must be of type uint8"
    assert set(sum(team_composition, [])).issubset(quantized_table.columns.to_list()), "team_composition must match quantized_table columns"

    ## check that bin counts match the maximum cell id in the quantized table
    assert isinstance(quantiles, dict), "quantiles must be a dictionary"
    assert all([col in quantiles for col in quantized_table.columns]), "quantized_table columns must be in quantiles"
    assert all([quantized_table[col].max() <= len(quantiles[col]) for col in quantized_table.columns]), \
        "bin_counts must match the largest in each column of the quantized table!"

    results = list()
    codec_id = string_to_codec_id(compression)

    for subset_cols in team_composition:
        # Sort table by the current subset. Needs a copy or we will mess up the original table.
        print("Copying projection...")
        projection = quantized_table[subset_cols].copy()
        projection["row_id"] = np.arange(quantized_table.shape[0], dtype=np.uint64)

        team_bin_counts = [ len(quantiles[col])+1 for col in subset_cols]

        # "stable" retains the sort order within each bucket!
        print("Sorting projection...")
        projection.sort_values(by=subset_cols, kind="stable", ignore_index=True, inplace=True)
        
        # Convert to column-major NumPy array
        row_ids = projection["row_id"].values.copy()
        del projection["row_id"]

        table_array = projection.to_numpy(dtype=np.uint8, copy=False)
        assert(table_array.flags["F_CONTIGUOUS"])
        print("Inverting table with shape", table_array.shape)
        # Call inversion
        inverter = TableInverter()
        result = inverter.invert_quantized_table(
            table_array,
            row_ids,
            team_bin_counts,
            codec_id
        )
        # result is a tuple: (output_data, cardinalities, compressed_sizes, used_codecs, page_offsets)

        assert result[1].sum() == table_array.shape[0], \
            f"Cardinalities ({result[1].sum()}) do not match the number of rows in the table ({table_array.shape[0]})!"

        results.append((subset_cols, result))
        del projection

    return results


def dump_inverted_index_results(result, team_name, output_path: Path, compression="copy"):
    """
    Dumps the result of inverted list creation to binary files using numpy.tofile.

    Parameters:
    - result: tuple (blob, cardinalities, compressed_sizes, codecs, page_offsets)
    - team_name: name of the column subset
    - output_path: Path object pointing to the base directory
    - compression: codec name, e.g. 'copy'
    """

    blob, cardinalities, sizes, codec_ids, offsets = result

    # Build suffixes
    inv_lists_pfx = "." + compression + ".lists"
    bc_sfx = "." + compression + ".cardinalities"
    offset_sfx = "." + compression + ".offsets"
    size_sfx = "." + compression + ".sizes"
    codecs_sfx = "." + compression + ".codecs"

    # Prepare output paths
    inv_list_path = output_path / f"{team_name}{inv_lists_pfx}"
    cards_path = output_path / f"{team_name}{bc_sfx}"
    offsets_path = output_path / f"{team_name}{offset_sfx}"
    sizes_path = output_path / f"{team_name}{size_sfx}"
    codecs_path = output_path / f"{team_name}{codecs_sfx}"

    # Write binary files
    with open(inv_list_path, "wb") as f:
        f.write(blob)  # py::bytes object

    np.asarray(cardinalities, dtype=np.uint32).reshape(-1).tofile(cards_path)
    np.asarray(offsets, dtype=np.uint32).reshape(-1).tofile(offsets_path)
    np.asarray(sizes, dtype=np.uint64).reshape(-1).tofile(sizes_path)
    np.asarray(codec_ids, dtype=np.uint8).reshape(-1).tofile(codecs_path)

    print(f"Dumped inverted index for team '{team_name}' to {output_path}")


def dump_index(results, target_folder, compression):
    """
    Call dump for every result. Build Team name from columns by simply joining with "-".
    """

    for team, result in results:
        team_name = "-".join(team)
        dump_inverted_index_results(result, team_name, target_folder, compression = compression)
    print("Dumped all indices to", target_folder)




# def index_table2(config_path, quantized_table=None):
#     """
#     Works like index_table, but uses the new quantization, inversion and dump functions.
#     """

#     # Load configuration
#     cfg = open_json(config_path)
#     team_compositions = cfg["teams"]
#     quantiles = cfg["quantiles"]
#     compressions = cfg["compressions"]

#     # Load table, if not provided
#     if quantized_table is None:
#         source_table = cfg.get("source_table")
#         if source_table is not None:
#             table = pq.read_table(source_table).to_pandas()
#         else:
#             raise ValueError("No source table provided in the configuration.")

#         # Quantize table
#         quantized_table = quantize_table(table, quantiles)

#     # Invert by partitions
#     for team_composition in team_compositions:
#         for compression in compressions:
#             results = invert_by_partitions(quantized_table, team_composition quantiles, compression=compression)

#             # Dump results
#             target_folder = Path(cfg["index_folder"])
#             dump_index(results, target_folder)



# def index_table3(config_path, quantized_table=None):
#     """
#     Works like index_table, but uses the new quantization, inversion and dump functions.
#     """

#     # Load configuration
#     cfg = open_json(config_path)
#     team_compositions = cfg["teams"]
#     quantiles = cfg["quantiles"]
#     compressions = cfg["compressions"]

#     # Load table, if not provided
#     if quantized_table is None:
#         source_table = cfg.get("source_table")
#         if source_table is not None:
#             table = pq.read_table(source_table).to_pandas()
#         else:
#             raise ValueError("No source table provided in the configuration.")

#         # Quantize table
#         quantized_table = quantize_table(table, quantiles)
#     for team_composition in team_compositions:

#         for team in team_composition:
#             print("copying projection")
#             projection = quantize_table[team].copy()

#             # df = pd.DataFrame(qtable)
#             # dup_free = df.drop_duplicates()

#             for compression in compressions:
