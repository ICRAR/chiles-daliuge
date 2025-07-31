import logging
import shutil
from glob import glob
from os import makedirs, chdir, getcwd, rename, remove
from os.path import join, exists, isdir, basename, isfile
from pathlib import Path

from casatasks import tclean, concat
from casatools import table, image, regionmanager
from typing import Union, List
import sqlite3
from chiles_daliuge.common import *
import json

LOG = logging.getLogger(__name__)

SEMESTER_VALUES = ["2013-2014", "2015", "2016", "2017-2018", "2019"]

from collections import defaultdict


def fetch_uvsub_ms(
        year_list: List[str],
        frequencies: List[List[int]],
        db_path: str,
        trigger_in: bool,
) -> List[str]:
    """
    Retrieve `uv_sub_name` entries from a metadata SQLite database matching specified years and frequency ranges.

    This function queries the `metadata` table to find measurement sets whose `year` matches any
    value in `year_list` and whose `[start_freq, end_freq]` pair matches any tuple in `frequencies`.
    The result is a list of formatted strings with uv_sub_name and associated metadata.

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
    list of str
        List of matching entries formatted as:
        "uv_sub_name;year;start_freq;end_freq"
    """
    freq_set = {tuple(freq_pair) for freq_pair in frequencies}

    query = """
        SELECT uv_sub_path, year, start_freq, end_freq 
        FROM metadata
        WHERE uv_sub_path IS NOT NULL AND TRIM(uv_sub_path) != ''
    """

    matching_uv_sub_names = []

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for row in cursor.execute(query):
            uv_sub_path, year, start_freq, end_freq = row
            # print(f"\nChecking row: dlg_name={dlg_name}, year={year}, start_freq={start_freq}, end_freq={end_freq}")

            freq_tuple = (int(start_freq), int(end_freq))  # ✅ convert to int

            if year in year_list and freq_tuple in freq_set:
                # print(f"  → Appending: {dlg_name}")
                matching_uv_sub_names.append(f"{uv_sub_path};{year};{freq_tuple[0]};{freq_tuple[1]}")

    LOG.info(f"matching_uv_sub_names: {matching_uv_sub_names}")
    return matching_uv_sub_names

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





def prep_build_concat(config: dict, name_list: List, METADATA_DB) -> List:
    """
    Prepare metadata and group Measurement Set (MS) files for concatenation based on shared frequency ranges.

    This function:
    - Parses configuration parameters to determine whether to process all semesters and concatenate.
    - Adds a metadata column to the provided metadata database (if missing).
    - Groups entries in `name_list` by (start_freq, end_freq).
    - Generates unique output names for each frequency group using `generate_hashed_ms_name`.
    - Returns a list of grouped file info dictionaries.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing parameters including 'semesters' and 'concatenate'.
    name_list : List[str]
        List of MS file descriptions in the format "filename;year;start_freq;end_freq".
    METADATA_DB : Any
        Reference to the metadata database where new columns may be added.

    Returns
    -------
    List[dict]
        A list of dictionaries, each with keys:
        - "start_freq": int, start of the frequency range
        - "end_freq": int, end of the frequency range
        - "files": list of str, filenames sharing this frequency range
        - "output_name": str, hashed name for the concatenated output
    """

    add_column_if_missing(METADATA_DB, "build_concat_all")
    add_column_if_missing(METADATA_DB, "build_concat_epoch")

    try:
        # region_files_tar = parse_value(config["region_files_tar"]["value"], config["region_files_tar"]["type"])
        # iterations = parse_value(config["iterations"]["value"], config["iterations"]["type"])
        # arcsec_low = parse_value(config["arcsec_low"]["value"], config["arcsec_low"]["type"])
        # arcsec_high = parse_value(config["arcsec_high"]["value"], config["arcsec_high"]["type"])
        # arcsec_cutover = parse_value(config["arcsec_cutover"]["value"], config["arcsec_cutover"]["type"])
        # w_projection_planes = parse_value(config["w_projection_planes"]["value"], config["w_projection_planes"]["type"])
        # clean_weighting_uv = parse_value(config["clean_weighting_uv"]["value"], config["clean_weighting_uv"]["type"])
        # robust = parse_value(config["robust"]["value"], config["robust"]["type"])
        # image_size = parse_value(config["image_size"]["value"], config["image_size"]["type"])
        # clean_channel_average = parse_value(config["clean_channel_average"]["value"], config["clean_channel_average"]["type"])
        # region_file = parse_value(config["region_file"]["value"], config["region_file"]["type"])
        # semesters = parse_value(config["semesters"]["value"], config["semesters"]["type"])
        # concatenate = parse_value(config["concatenate"]["value"], config["concatenate"]["type"])

        semesters = parse_value(config["semesters"]["value"], config["semesters"]["type"])
        concatenate = parse_value(config["concatenate"]["value"], config["concatenate"]["type"])

    except KeyError as e:
        LOG.error(f"Missing required config key: {e}")
        raise
    except Exception as e:
        LOG.error(f"Failed to parse config: {e}")
        raise

    if semesters == 'all' and concatenate:
        LOG.info("Concatenating all semesters in one")
        grouped = defaultdict(list)

        # ✅ Parse and group files by (start_freq, end_freq)
        for entry in name_list:
            filename, year, start_freq, end_freq = entry.split(";")
            key = (start_freq, end_freq)
            grouped[key].append(filename)

        # ✅ Now build stringified output
        stringified_results = []
        for (start, end), files in grouped.items():
            output_name = generate_hashed_ms_name(
                ms_name="all",
                year="build_concat_all",
                start_freq=str(start),
                end_freq=str(end)
            )
            entry_data = [str(start), str(end)] + files + [output_name]
            stringified_results.append(stringify_data(entry_data))  # converts to single string

    elif semesters == 'epoch' and concatenate:
        LOG.info("Concatenating each semester in one.")
        grouped = defaultdict(list)

        # group by (year, start_freq, end_freq)
        for entry in name_list:
            filename, year, start_freq, end_freq = entry.split(";")
            key = (year, start_freq, end_freq)
            grouped[key].append(filename)

        stringified_results = []
        for (year, start, end), files in grouped.items():
            output_name = generate_hashed_ms_name(
                ms_name="epoch_concat",
                year=year,
                start_freq=start,
                end_freq=end
            )
            entry_data = [year, start, end] + files + [output_name]
            stringified_results.append(stringify_data(entry_data))

    else:
        LOG.info("Wrong arguments for 'semesters' or 'concatenate'")
        stringified_results = []

    stringified_results = np.array(stringified_results, dtype=str)
    LOG.info(f"stringified_results - {stringified_results}")
    return stringified_results
