import logging
import shutil
from glob import glob
from os import makedirs, chdir, getcwd, rename, remove
from os.path import join, exists, isdir, basename, isfile
from pathlib import Path

from casatasks import tclean, concat
from casatools import table, image, regionmanager
from typing import Union, List, Any
import sqlite3
from chiles_daliuge.common import *
import ast

LOG = logging.getLogger(__name__)

SEMESTER_VALUES = ["2013-2014", "2015", "2016", "2017-2018", "2019"]

from collections import defaultdict


def fetch_concat_ms_all(
        year_list: List[str],
        frequencies: List[List[int]],
        db_path: str,
        trigger_in: bool,
) -> List[Union[str, List[Any]]]:
    """
    Retrieve `build_concat_all` entries from a metadata SQLite database matching specified years
    and frequency ranges, but grouped by freq‑pair.  For each freq‑pair:

      – If all matching years point to the same file, return a single string
        "build_concat_all;all;start_freq;end_freq"

      – Otherwise return one list: [ [file1, file2, ...], [year1, year2, ...], start_freq, end_freq ]

    Parameters
    ----------
    year_list : list of str
        List of years to filter on (e.g., ["2013", "2014"]).
    frequencies : list of [int, int]
        List of frequency range pairs to match, specified as [start_freq, end_freq].
    db_path : str
        Path to the SQLite metadata database file.
    trigger_in : bool
        Placeholder for future use; currently not used in this function.

    Returns
    -------
    List[Union[str, List[Any]]]
        A list where each element is either:
          - a semicolon‑joined string "file;all;start;end"  (if same file for all years), or
          - a list [ [file1, file2, …], [year1, year2, …], start, end ]  (if multiple files)
    """
    freq_set = {tuple(freq_pair) for freq_pair in frequencies}

    # 1) gather all (year, file) per freq‑tuple
    freq_map: dict[tuple[int,int], list[tuple[str,str]]] = {}
    query = """
        SELECT build_concat_all, year, start_freq, end_freq
        FROM metadata
        WHERE build_concat_all IS NOT NULL AND TRIM(build_concat_all) != ''
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for build_concat_all, year, start_freq, end_freq in cursor.execute(query):
            freq_tuple = (int(start_freq), int(end_freq))
            if year in year_list and freq_tuple in freq_set:
                freq_map.setdefault(freq_tuple, []).append((year, build_concat_all))

    matching_concat_names: List[Union[str, List[Any]]] = []

    # 2) for each freq‑tuple, decide if “all” or a list
    for (start_f, end_f), pairs in freq_map.items():
        years = [y for (y, _) in pairs]
        builds = [b for (_, b) in pairs]
        # unique builds in insertion order
        unique_builds = list(dict.fromkeys(builds))

        if len(unique_builds) == 1:
            # same file for all years
            matching_concat_names.append(
                f"{unique_builds[0]};all;{start_f};{end_f}"
            )
        else:
            build_to_year = {}
            for y, b in pairs:
                if b not in build_to_year:
                    build_to_year[b] = y

            # now align years with unique_builds
            unique_years = [build_to_year[b] for b in unique_builds]
            # multiple distinct files → emit a single list
            matching_concat_names.append(f"{unique_builds};{unique_years};{start_f};{end_f}")

    LOG.info(f"matching_concat_names: {matching_concat_names}")
    return matching_concat_names


def fetch_concat_ms_epoch(
        year_list: List[str],
        frequencies: List[List[int]],
        db_path: str,
        trigger_in: bool,
) -> List[Union[str, List[Any]]]:
    """
    Retrieve `build_concat_epoch` entries from a metadata SQLite database matching specified years
    and frequency ranges, but grouped by freq‑pair.  For each freq‑pair:

      – If all matching years point to the same file, return a single string
        "build_concat_all;all;start_freq;end_freq"

      – Otherwise return one list: [ [file1, file2, ...], [year1, year2, ...], start_freq, end_freq ]

    Parameters
    ----------
    year_list : list of str
        List of years to filter on (e.g., ["2013", "2014"]).
    frequencies : list of [int, int]
        List of frequency range pairs to match, specified as [start_freq, end_freq].
    db_path : str
        Path to the SQLite metadata database file.
    trigger_in : bool
        Placeholder for future use; currently not used in this function.

    Returns
    -------
    List[Union[str, List[Any]]]
        A list where each element is either:
          - a semicolon‑joined string "file;all;start;end"  (if same file for all years), or
          - a list [ [file1, file2, …], [year1, year2, …], start, end ]  (if multiple files)
    """
    freq_set = {tuple(freq_pair) for freq_pair in frequencies}

    # 1) gather all (year, file) per freq‑tuple
    freq_map: dict[tuple[int,int], list[tuple[str,str]]] = {}
    query = """
        SELECT build_concat_epoch, year, start_freq, end_freq
        FROM metadata
        WHERE build_concat_epoch IS NOT NULL AND TRIM(build_concat_epoch) != ''
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for build_concat_epoch, year, start_freq, end_freq in cursor.execute(query):
            freq_tuple = (int(start_freq), int(end_freq))
            if year in year_list and freq_tuple in freq_set:
                freq_map.setdefault(freq_tuple, []).append((year, build_concat_epoch))

    matching_concat_names: List[Union[str, List[Any]]] = []

    # 2) for each freq‑tuple, decide if “all” or a list
    for (start_f, end_f), pairs in freq_map.items():
        years = [y for (y, _) in pairs]
        builds = [b for (_, b) in pairs]
        # unique builds in insertion order
        unique_builds = list(dict.fromkeys(builds))

        if len(unique_builds) == 1:
            # same file for all years
            matching_concat_names.append(
                f"{unique_builds[0]};all;{start_f};{end_f}"
            )
        else:
            build_to_year = {}
            for y, b in pairs:
                if b not in build_to_year:
                    build_to_year[b] = y

            # now align years with unique_builds
            unique_years = [build_to_year[b] for b in unique_builds]
            # multiple distinct files → emit a single list
            matching_concat_names.append(f"{unique_builds};{unique_years};{start_f};{end_f}")

    LOG.info(f"matching_concat_names: {matching_concat_names}")
    return matching_concat_names


def parse_value(value, type_str):
    """Convert value based on CASA-style type string."""
    if type_str.lower() == "integer":
        return int(value)
    elif type_str.lower() == "float":
        return float(value)
    elif type_str.lower() == "boolean":
        return value in [True, "True", "true", "1"]
    elif type_str.lower() == "string":
        return str(value)
    elif type_str.lower() == "list":
        return list(value) if isinstance(value, (list, tuple)) else eval(value)
    else:
        return value  # fallback


def prep_tclean(config: dict, name_list: List[str], METADATA_DB: str) -> np.ndarray:
    """
    Prepare the full tclean CLI argument lists (stringified) for each freq group.

    Returns an array of strings, each of which is a Python‐literal list:
      [out_ms, min_freq, max_freq,
       iterations, arcsec, w_projection_planes,
       clean_weighting_uv, robust, image_size,
       clean_channel_average, region_file, produce_qa,
       temporary_directory, in_ms_list]
    """
    # ensure the tclean_all column exists
    add_column_if_missing(METADATA_DB, "tclean_all")
    add_column_if_missing(METADATA_DB, "tclean_epoch")

    # --- parse all config params we need ---
    try:
        iterations = parse_value(config["iterations"]["value"], config["iterations"]["type"])
        arcsec_low = parse_value(config["arcsec_low"]["value"], config["arcsec_low"]["type"])
        arcsec_high = parse_value(config["arcsec_high"]["value"], config["arcsec_high"]["type"])
        arcsec_cutover = parse_value(config["arcsec_cutover"]["value"], config["arcsec_cutover"]["type"])
        w_projection_planes = parse_value(config["w_projection_planes"]["value"], config["w_projection_planes"]["type"])
        clean_weighting_uv = parse_value(config["clean_weighting_uv"]["value"], config["clean_weighting_uv"]["type"])
        robust = parse_value(config["robust"]["value"], config["robust"]["type"])
        image_size = parse_value(config["image_size"]["value"], config["image_size"]["type"])
        clean_channel_average = parse_value(config["clean_channel_average"]["value"], config["clean_channel_average"]["type"])
        region_file = parse_value(config["region_file"]["value"], config["region_file"]["type"])
        semesters = parse_value(config["semesters"]["value"], config["semesters"]["type"])
        concatenate = parse_value(config["concatenate"]["value"], config["concatenate"]["type"])

    except KeyError as e:
        LOG.error(f"Missing required config key: {e}")
        raise
    except Exception as e:
        LOG.error(f"Failed to parse config: {e}")
        raise

    stringified_results: List[str] = []

    if semesters == 'all' and not concatenate:
        LOG.info(f"Generating for all?: {semesters}")
        # group by (start_freq, end_freq)
        grouped = defaultdict(list)
        for entry in name_list:
            LOG.info(f"entry: {entry}")
            filename, year, start_str, end_str = entry.split(";")
            try:
                parsed = ast.literal_eval(filename)
                if isinstance(parsed, list):
                    grouped[(start_str, end_str)].extend(parsed)
                else:
                    grouped[(start_str, end_str)].append(parsed)
            except (SyntaxError, ValueError):
                grouped[(start_str, end_str)].append(filename)

        # for each freq‐group build full arg list
        for (start_freq, end_freq), files in grouped.items():
            out_ms = generate_hashed_ms_name(
                ms_name="all",
                year="all",
                start_freq=start_freq,
                end_freq=end_freq
            )
            arcsec = arcsec_low if int(end_str) < arcsec_cutover else arcsec_high
            produce_qa = 'False'
            LOG.info(f"files: {files}")
            # note: in_ms is the list of filenames for this group
            entry_args: List[Any] = [
                out_ms,                # <out_ms>
                start_str,             # <min_freq>
                end_str,               # <max_freq>
                str(iterations),       # <iterations>
                str(arcsec),           # <arcsec>
                str(w_projection_planes),     # <w_projection_planes>
                str(clean_weighting_uv),      # <clean_weighting_uv>
                str(robust),                  # <robust>
                str(image_size),              # <image_size>
                str(clean_channel_average),   # <clean_channel_average>
                str(region_file),             # <region_file>
                str(produce_qa),              # <produce_qa>
                #str(temporary_directory),     # <temporary_directory>
                files                         # <in_ms>  (stringified list)
            ]
            LOG.info(f"entry_args: {entry_args}")
            # stringify_data should turn that Python list into a single string
            stringified_results.append(stringify_data(entry_args))

    elif semesters == 'epoch' and not concatenate:
        # group by (start_freq, end_freq)
        LOG.info(f"Generating for epoch?: {semesters}")
        LOG.info(f"name_list: {name_list}")
        grouped = defaultdict(list)
        for entry in name_list:
            LOG.info(f"entry: {entry}")
            filename, year, start_str, end_str = entry.split(";")

            try:
                parsed = ast.literal_eval(filename)
                if isinstance(parsed, list):
                    grouped[(year, start_str, end_str)].extend(parsed)
                else:
                    grouped[(year, start_str, end_str)].append(parsed)
            except (SyntaxError, ValueError):
                grouped[(year, start_str, end_str)].append(filename)

        # for each freq‐group build full arg list
        for (year, start_freq, end_freq), files in grouped.items():
            out_ms = generate_hashed_ms_name(
                ms_name="all",
                year=str(year),
                start_freq=start_freq,
                end_freq=end_freq
            )
            arcsec = arcsec_low if int(end_str) < arcsec_cutover else arcsec_high
            produce_qa = 'False'
            LOG.info(f"files: {files}")
            # note: in_ms is the list of filenames for this group
            entry_args: List[Any] = [
                out_ms,                # <out_ms>
                year,
                start_str,             # <min_freq>
                end_str,               # <max_freq>
                str(iterations),       # <iterations>
                str(arcsec),           # <arcsec>
                str(w_projection_planes),     # <w_projection_planes>
                str(clean_weighting_uv),      # <clean_weighting_uv>
                str(robust),                  # <robust>
                str(image_size),              # <image_size>
                str(clean_channel_average),   # <clean_channel_average>
                str(region_file),             # <region_file>
                str(produce_qa),              # <produce_qa>
                #str(temporary_directory),     # <temporary_directory>
                files                         # <in_ms>  (stringified list)
            ]
            LOG.info(f"entry_args: {entry_args}")
            # stringify_data should turn that Python list into a single string
            stringified_results.append(stringify_data(entry_args))
    else:
        LOG.info("Skipping prep_tclean: concatenate = True")

    # return as array of str
    LOG.info(f"stringified_results: {stringified_results}")
    array_out = np.array(stringified_results, dtype=str)
    LOG.info(f"array_out: {array_out}")
    return array_out
