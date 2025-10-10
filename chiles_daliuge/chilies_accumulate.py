#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from casatasks import concat, split
from casatools import ms as ms_tool
from chiles_daliuge.common import *
import logging
import sqlite3
from pathlib import Path
from typing import List
import json
import tempfile

process_ms_flag = True

LOG = logging.getLogger(f"dlg.{__name__}")


def _normalize_year_list(year_list) -> List[str]:
    """Accept a Python list[str], or a JSON string like '["2013-2014","2015"]'."""
    if year_list is None:
        return []
    if isinstance(year_list, str):
        year_list = year_list.strip()
        # Try JSON first
        try:
            parsed = json.loads(year_list)
            if isinstance(parsed, list):
                return [str(y).strip() for y in parsed if str(y).strip()]
            # If it was a single string like "2013-2014"
            return [str(parsed).strip()] if str(parsed).strip() else []
        except json.JSONDecodeError:
            # Fallback: comma-separated string "2013-2014,2015"
            parts = [p.strip() for p in year_list.split(",")]
            return [p for p in parts if p]
    # Already a list/iterable
    return [str(y).strip() for y in year_list if str(y).strip()]

def fetch_uvsub(
        year_list: List[str],
        frequencies: List[List[int]],
        db_path: str,
        concat_time: bool,
        trigger_in: bool,
) -> List[str]:
    """
    Build a list of strings describing uv_sub_paths grouped by frequency ranges, and optionally by base_name.

    Output format
    -------------
    If concat_time is True:
        "concat_all;start_freq_range;end_freq_range;uv_sub_path1,uv_sub_path2,..."

    If concat_time is False:
        "base_name;start_freq_range;end_freq_range;uv_sub_path1,uv_sub_path2,..."
        (one line per base_name that has at least one path in the range)

    Notes
    -----
    - Year filtering is an exact string match: values in `year_list` must match the DB's `year` strings exactly.
    - Includes only rows where uv_sub_path is non-null and non-blank.
    - start_freq/end_freq columns are TEXT in DB; we CAST the *columns* to INTEGER to compare numerically.
    - All SQL parameters are passed as strings.
    - Paths are deduplicated and sorted for stable output.
    """
    # Normalize to unique (lo, hi) integer pairs
    freq_set = {tuple(sorted(map(int, pair))) for pair in frequencies}
    if not freq_set:
        return []

    # Year list: treat strictly as strings (no substring or parsing)
    year_list_str = _normalize_year_list(year_list)

    # WHERE clause (exact year match when provided)
    where_clauses = [
        "uv_sub_path IS NOT NULL",
        "TRIM(uv_sub_path) <> ''",
        "CAST(start_freq AS INTEGER) >= CAST(? AS INTEGER)",
        "CAST(end_freq   AS INTEGER) <= CAST(? AS INTEGER)",
    ]
    if year_list_str:
        placeholders = ",".join("?" for _ in year_list_str)
        where_clauses.append(f"year IN ({placeholders})")

    where_sql = " AND ".join(where_clauses)
    query_sql = f"""
        SELECT base_name,
               uv_sub_path,
               year,
               CAST(start_freq AS INTEGER) AS sf,
               CAST(end_freq   AS INTEGER) AS ef
        FROM metadata
        WHERE {where_sql}
    """

    matching_dlg_names: List[str] = []

    def _sort_key(row):
        # row = (base_name, uv_sub_path, year, sf, ef)
        base, path, yr, sf, ef = row
        return (str(yr or ""), str(sf), str(ef), str(path or ""))

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        exist_sql = """
            SELECT 1
            FROM concat_freq
            WHERE base_name = ?
              AND start_freq = ?
              AND end_freq = ?
            LIMIT 1
        """

        for (lo, hi) in sorted(freq_set):
            # All parameters passed as strings
            params = [str(lo), str(hi)]
            if year_list_str:
                params.extend(year_list_str)

            cur.execute(query_sql, params)
            rows = cur.fetchall()  # [(base_name, uv_sub_path, year, sf, ef), ...]
            LOG.info(f"[RANGE {lo}-{hi}] fetched {len(rows)} rows")

            if not rows:
                continue

            if concat_time:
                # Skip if concat_freq already has this (concat_all, lo, hi)
                cur.execute(exist_sql, ("concat_all", str(lo), str(hi)))
                if cur.fetchone():
                    continue

                paths = sorted({(str(r[1]) or "").strip() for r in rows if (str(r[1]) or "").strip()})
                if not paths:
                    continue
                line = f"concat_all;{lo};{hi};{','.join(paths)}"
                matching_dlg_names.append(line)
            else:
                from collections import defaultdict
                rows_sorted = sorted(rows, key=_sort_key)

                by_base = defaultdict(set)
                for base_name, uv_path, _year, _sf, _ef in rows_sorted:
                    p = (str(uv_path) or "").strip()
                    if p:
                        by_base[str(base_name)].add(p)

                for base_name, path_set in sorted(by_base.items(), key=lambda kv: kv[0] or ""):
                    if not path_set:
                        continue
                    # Skip if concat_freq already has this (base_name, lo, hi)
                    cur.execute(exist_sql, (base_name, str(lo), str(hi)))
                    if cur.fetchone():
                        continue
                    line = f"{base_name};{lo};{hi};{','.join(sorted(path_set))}"
                    matching_dlg_names.append(line)

    return matching_dlg_names




def extract_tar(tar_path: Path, dest_dir: Path):
    """
    Extract a .tar into dest_dir.
    Returns the list of top-level items extracted.
    """
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(dest_dir)
        names = [dest_dir / m.name.split("/")[0] for m in tf.getmembers() if "/" in m.name or m.name]
    # De-duplicate and keep existing only
    unique = []
    seen = set()
    for p in names:
        if p.exists() and p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def is_valid_ms(ms_path: Path) -> bool:
    """
    Quick check using casatools.ms: open/close to validate MS directory.
    """
    if not ms_path.exists():
        return False
    try:
        ms = ms_tool()
        ms.open(str(ms_path))
        ms.close()
        return True
    except Exception:
        return False


def prep_concat(names_list):

    concat_data_all = []

    for name in names_list:
        # base_name, start_freq, end_freq, paths_blob = name.split(";")
        #
        # if not base_name:
        #     raise ValueError("base_name is empty in concat_data_in")
        # if not start_freq or not end_freq:
        #     raise ValueError("start_freq/end_freq cannot be empty in concat_data_in")

        combined_data = [name]

        concat_data_all.append(stringify_data(combined_data))

    # conn.close()

    concat_data_all = np.array(concat_data_all, dtype=str)

    LOG.info(f"concat_data_all: {concat_data_all}")

    return concat_data_all



def do_concat(
    concat_data_in: str,
    save_dir: str,
    db_path: str,
):
    os.makedirs(save_dir, exist_ok=True)
    LOG.info(f"concat_data_in: {concat_data_in}")
    base_name, start_freq, end_freq, paths_combined = destringify_data_concat(concat_data_in)

    LOG.info(f"base_name: {base_name}")
    LOG.info(f"start_freq: {start_freq}")
    LOG.info(f"end_freq: {end_freq}")
    LOG.info(f"paths_combined: {paths_combined}")
    tar_paths = paths_combined.split(",")

    # if len(concat_data) < 4:
    #     raise ValueError(
    #         "concat_data must have 4 segments separated by semicolons: "
    #         "<base_name>;<start_freq>;<end_freq>;<path1;path2;...>"
    #     )

    bandwidth = int(end_freq) - int(start_freq)

    if save_dir.endswith(".ms"): # should contain .ms by default as specified in dirdrop, if {auto}.ms works
        save_dir = save_dir
    else:
        save_dir = f"{save_dir}.ms"

    save_dir_temp = f"{save_dir}.tmp"

    vis = []
    for tarp in tar_paths:
        try:
            extracted = extract_tar(tarp, save_dir_temp)
            ms_dir_name = tarp.stem  # e.g. "XYZ" from "XYZ.tar"
            candidate_ms = save_dir_temp / ms_dir_name
            if candidate_ms.is_dir() and is_valid_ms(candidate_ms):
                vis.append(str(candidate_ms))
            else:
                # If not found by stem, try any MS dirs extracted
                for item in extracted:
                    if item.is_dir() and is_valid_ms(item):
                        vis.append(str(item))
                        break
                else:
                    LOG.warning(f"[WARN] No valid MS found in {tarp.name}")
        except Exception as e:
            LOG.error(f"[ERROR] Failed to extract or validate {tarp}: {e}")

    if not vis:
        LOG.warning(f"[WARN] No valid MeasurementSets to concat in bucket {start_freq}_{end_freq}. Cleaning up and skipping.")
        shutil.rmtree(save_dir_temp, ignore_errors=True)

    else:
        # Concat
        concat_out = save_dir_temp / f"Accumulated_{start_freq}_{end_freq}.ms"
        if concat_out.exists():
            shutil.rmtree(concat_out)
        LOG.info(f"[INFO] concat -> {concat_out.name} (N={len(vis)})")
        concat(concatvis=str(concat_out), vis=vis)

        # Optionally remove extracted inputs after concat (matches original)
        for v in vis:
            try:
                shutil.rmtree(v, ignore_errors=True)
            except Exception:
                pass

        # Split
        timebin='60s'
        channel_width = 32
        split_out = save_dir
        if split_out.exists():
            shutil.rmtree(split_out)
        try:
            LOG.info(f"[INFO] split -> {split_out.name}, timebin={timebin}, width={channel_width}")
            split(
                vis=str(concat_out),
                outputvis=str(split_out),
                timebin=timebin,
                width=channel_width,
                datacolumn="all",
            )
            size_bytes = os.path.getsize(split_out) if os.path.exists(split_out) else 0
            size = round(float(size_bytes / (1024 * 1024 * 1024)), 3)
            insert_concat_freq_row(db_path, save_dir, base_name, "N/A", start_freq, end_freq, bandwidth, size)
        except Exception as e:
            LOG.error(f"Split failed in bucket {start_freq}_{end_freq}: {e}")

        LOG.info(f"[INFO] removing temporary dir {save_dir_temp}.")
        shutil.rmtree(save_dir_temp, ignore_errors=True)

    LOG.info("\n[INFO] Done.")

# def main(concat_data: list) -> None:
#     LOG.info(f"concat_data (parsed): {concat_data}")
#     do_concat(*concat_data)




if __name__ == "__main__":
    try:

        LOG.info(f"sys.argv: {sys.argv}")

        concat_data_in = str(sys.argv[1:-2])
        LOG.info(f"concat_data_in: {concat_data_in}")

        concat_path_in = sys.argv[-2]
        LOG.info(f"concat_path_in: {concat_path_in}")

        db_path = sys.argv[-1]
        LOG.info(f"db_path: {db_path}")


        do_concat(concat_data_in, concat_path_in, db_path)


    except Exception as e:
        LOG.exception("Failed to run uvsub job")
        print(json.dumps({"status": "error", "message": str(e)}), file=sys.stderr)
        sys.exit(1)


# base_name: str,
#     start_freq: int,
#     end_freq: int,
#     paths_in: str,
