#!/usr/bin/env python3
"""
Convert SDSS datasweep FITS tables (calibObj-*.fits.gz) to chunked Parquet.

- Globs files under a subfolder (default: ./raw_files)
- Reads each FITS binary table (HDU 1)
- Fixes endianness for NumPy >= 2.0
- Flattens multi-field vector columns (e.g., 5-band arrays) into scalar columns
  * If a field has length 5, suffixes are _u,_g,_r,_i,_z
  * Otherwise numeric suffixes _0, _1, ...
- Accumulates DataFrames in memory until a byte threshold is exceeded, then writes
  out a Parquet file using pyarrow (snappy compression) and clears the buffer.

Example:
  python sdss_calibobj_sweeps_to_parquet.py \
    --input-dir ./raw_files \
    --pattern 'calibObj-*.fits.gz' \
    --threshold-mb 2048 \
    --out ./parquet_out \
    --prefix calibobj_dr17_301
"""

from __future__ import annotations
import argparse
import os
import sys
import glob
from typing import Dict, List

import numpy as np
import pandas as pd
from astropy.io import fits

import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm

BAND_SUFFIXES = ["u", "g", "r", "i", "z"]

_NUMERIC_KINDS = set("iu f c")  # int, unsigned, float, complex

def ensure_native_array(a: np.ndarray) -> np.ndarray:
    """Return array with native-endian dtype; no copy unless needed."""
    if not isinstance(a, np.ndarray):
        return a
    if a.dtype.kind not in _NUMERIC_KINDS:
        return a  # strings, objects, etc.
    bo = a.dtype.byteorder
    if bo in ("|", "="):
        return a
    # NumPy 2.0 style: byteswap data + update dtype to native
    return a.byteswap().view(a.dtype.newbyteorder("="))

def ensure_native_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all numeric columns to native-endian arrays in-place."""
    for col in df.columns:
        a = df[col].to_numpy(copy=False)
        if isinstance(a, np.ndarray) and a.dtype.kind in _NUMERIC_KINDS:
            bo = a.dtype.byteorder if hasattr(a.dtype, "byteorder") else "="
            if bo not in ("|", "="):
                df[col] = ensure_native_array(a)
    return df


def flatten_structured(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """Flatten a structured array with possible vector fields into 1D columns.

    For fields with shape (N, k):
      - If k == 5, map to ugriz suffixes
      - Else, map to numeric suffixes _0.._k-1
    """
    cols: Dict[str, np.ndarray] = {}
    names = arr.dtype.names or []
    n = arr.shape[0]
    for name in names:
        field = arr[name]
        # Convert byte strings to Python strings if needed
        if field.dtype.kind in ("S", "a"):  # fixed-width bytes
            cols[name] = field.astype("U").reshape(n)
            continue
        if field.ndim == 1:
            # scalar per row
            cols[name] = field
        elif field.ndim == 2:
            k = field.shape[1]
            if k == 5:
                for i, b in enumerate(BAND_SUFFIXES):
                    cols[f"{name}_{b}"] = field[:, i]
            else:
                for i in range(k):
                    cols[f"{name}_{i}"] = field[:, i]
        else:
            # Rare higher-rank fields: flatten last axis if possible
            flat = field.reshape(n, -1)
            for i in range(flat.shape[1]):
                cols[f"{name}_{i}"] = flat[:, i]
    return cols


def fits_to_dataframe(path: str) -> pd.DataFrame:
    """Load a calibObj FITS (gz ok) and return a flattened pandas DataFrame."""
    with fits.open(path, memmap=True) as hdul:
        if len(hdul) < 2 or not hasattr(hdul[1], "data"):
            raise RuntimeError(f"No binary table HDU[1] in {path}")
        data = hdul[1].data  # recarray-like
        arr = np.array(data)  # plain structured ndarray
        # arr = to_native_endian(arr)
        cols = flatten_structured(arr)
        df = pd.DataFrame(cols)
        df = ensure_native_df(df)

        return df


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"

def enforce_schema(df, all_cols):
    out = {}
    for col in all_cols:
        if col in df.columns:
            out[col] = df[col]
        else:
            out[col] = pd.Series([pd.NA] * len(df), dtype="Float64")
    return pd.DataFrame(out)

def dump_parquet(dfs: List[pd.DataFrame], out_dir: str, prefix: str, seq: int, compression: str = "snappy") -> int:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_{seq:05d}.parquet")
    # Concatenate along rows; allow differing columns (outer join) by reindexing
    # Align columns union
    # all_cols = sorted(set().union(*[df.columns for df in dfs]))
    # aligned = [enforce_schema(df, all_cols) for df in dfs]
    # big = pd.concat(aligned, ignore_index=True)
    # big.to_parquet(out_path, engine="pyarrow", compression="snappy")
    # return big.shape[0]

    # Convert each df -> arrow table
    tables = [pa.Table.from_pandas(df, preserve_index=False) for df in dfs]

    # Concatenate at Arrow level (fast, schema-aware)
    big_table = pa.concat_tables(tables, promote_options='default')

    # Write out
    pq.write_table(big_table, out_path, compression=compression)

    return big_table.num_rows


def main():
    p = argparse.ArgumentParser(description="Convert SDSS calibObj FITS to Parquet chunks")
    p.add_argument("--input-dir", default="./raw_files", help="Folder containing FITS/FITS.GZ files")
    p.add_argument("--pattern", default="calibObj-*.fits.gz", help="Glob pattern under input-dir")
    p.add_argument("--threshold-mb", type=float, default=4096.0, help="Max in-memory DF bytes before dumping (MB)")
    p.add_argument("--out", default="./parquet_files", help="Output directory for Parquet files")
    p.add_argument("--prefix", default="calibobj", help="Output file prefix")
    p.add_argument("--limit", type=int, default=None, help="Optional max number of files to process")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"No files matched: {os.path.join(args.input_dir, args.pattern)}", file=sys.stderr)
        sys.exit(2)

    buf: List[pd.DataFrame] = []
    buf_bytes = 0
    seq = 0
    total_rows = 0

    thresh_bytes = int(args.threshold_mb * 1024 * 1024)

    print(f"Found {len(files)} files. Threshold {args.threshold_mb} MB per chunk.\n")

    for idx, path in tqdm(enumerate(files, 1), desc="Processing files", unit="file", total=len(files)):
        try:
            df = fits_to_dataframe(path)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}", file=sys.stderr)
            continue

        df_bytes = int(df.memory_usage(deep=True).sum())
        buf.append(df)
        buf_bytes += df_bytes

        print(f"[{idx}/{len(files)}] {os.path.basename(path)} -> rows={len(df):,}, size≈{human_bytes(df_bytes)}; buffer≈{human_bytes(buf_bytes)}")

        if buf_bytes >= thresh_bytes:
            rows = dump_parquet(buf, args.out, args.prefix, seq)
            total_rows += rows
            print(f"  -> wrote {rows:,} rows to {args.prefix}_{seq:05d}.parquet; total={total_rows:,}")
            seq += 1
            buf.clear()
            buf_bytes = 0

    # Flush remainder
    if buf:
        rows = dump_parquet(buf, args.out, args.prefix, seq)
        total_rows += rows
        print(f"  -> wrote {rows:,} rows to {args.prefix}_{seq:05d}.parquet; total={total_rows:,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
