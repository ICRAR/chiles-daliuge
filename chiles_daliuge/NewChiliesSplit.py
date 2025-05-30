from pathlib import Path
import ast
from subprocess import run, PIPE

from typing import List

from numpy.core.multiarray import ndarray

from chiles_daliuge.common import *
import logging
import sqlite3
import numpy as np

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

process_ms_flag = True


def fetch_original_ms(
        source_dir: str,
        year_list: list[str],
        copy_directory: str,
        trigger_in: bool,
        METADATA_DB: str,
        process_ms: bool = process_ms_flag,
) -> list[str]:
    """
    Fetch measurement sets from a remote directory structure using rclone, track them with metadata in SQLite.

    This function scans a remote source directory structured by year/date to locate
    Measurement Set (.ms) directories. It optionally creates local placeholder files
    or copies the MS directories to a specified local directory, renaming each using a
    hash of metadata fields. Metadata is logged in a persistent SQLite database.

    Parameters
    ----------
    METADATA_DB: str
        path to the database
    source_dir : str
        The root remote directory containing year/date subfolders with .ms sets.
    year_list : list of str
        A list of year or year-ranges to process (e.g., ["2013-2014", "2015"]).
    copy_directory : str
        Local destination directory where the .ms files or placeholders will be stored.
    process_ms : bool, optional
        If True, copies the .ms data using rclone. If False, creates an empty placeholder
        file with the hashed name. Default is False.

    Returns
    -------
    list of str
        List of local file paths (either real .ms directories or placeholder `.ms` files)
        corresponding to the processed or identified measurement sets.

    Notes
    -----
    - Uses rclone for folder traversal and copying.
    - Logs metadata to "Chilies_metadata.db" SQLite database.
    - Also exports the metadata table to "Chilies_metadata.csv" for inspection.
    """

    # verify_db_integrity()

    make_directory = True
    start_freq = "0944"
    end_freq = "1420"
    bandwidth = int(end_freq) - int(start_freq)
    name_list = []

    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()

    result = run(["rclone", "lsf", source_dir, "--dirs-only"], stdout=PIPE, stderr=PIPE, text=True)
    if result.returncode != 0:
        LOG.error(f"Failed to list {source_dir}: {result.stderr}")
        return []

    year_dirs = [line.strip("/") for line in result.stdout.strip().splitlines()]

    for year in year_dirs:
        if year not in year_list:
            continue

        year_path = f"{source_dir}{year}/"

        result = run(["rclone", "lsf", year_path, "--dirs-only"], stdout=PIPE, stderr=PIPE, text=True)
        if result.returncode != 0:
            LOG.warning(f"Skipping {year_path}: {result.stderr}")
            continue

        date_dirs = [line.strip("/") for line in result.stdout.strip().splitlines()]

        for date in date_dirs:
            date_path = f"{year_path}{date}/"

            result = run(["rclone", "lsf", date_path, "--dirs-only"], stdout=PIPE, stderr=PIPE, text=True)
            if result.returncode != 0:
                LOG.warning(f"Skipping {date_path}: {result.stderr}")
                continue

            ms_dirs = [f"{date_path}{line.strip('/')}" for line in result.stdout.strip().splitlines() if
                       line.endswith(".ms/")]

            for ms_name in ms_dirs:
                base_name = os.path.basename(ms_name.strip("/"))
                dlg_name = generate_hashed_ms_name(ms_name, year, start_freq, end_freq)
                copy_path = os.path.join(copy_directory, dlg_name)

                cursor.execute("SELECT 1 FROM metadata WHERE dlg_name = ?", (dlg_name,))
                if cursor.fetchone():
                    LOG.info(f"Skipping existing MS: {base_name}, already recorded as {dlg_name}")
                    name_list.append(str(dlg_name))
                    continue

                if make_directory:
                    os.makedirs(copy_directory, exist_ok=True)

                if process_ms:
                    LOG.info(f"Copying {ms_name} → {copy_path}")
                    cp_result = run(["rclone", "copy", "-P", ms_name, copy_path], stdout=PIPE, stderr=PIPE, text=True)

                    size = "Unknown"

                    if cp_result.returncode == 0:
                        LOG.info(f"Copied MS to: {copy_path}")

                        size_result = run(["rclone", "size", copy_path], stdout=PIPE, stderr=PIPE, text=True)
                        if size_result.returncode == 0:

                            for line in size_result.stdout.strip().splitlines():
                                if line.startswith("Total size:") and "(" in line:
                                    try:
                                        size_bytes_str = line.split("(")[-1].split()[0]  # robust extraction
                                        size_bytes = int(size_bytes_str)
                                        size = round(size_bytes / (1024 * 1024 * 1024), 3)  # GB
                                        break
                                    except ValueError:
                                        LOG.warning(f"Could not parse size from line: {line}")

                        LOG.info(f"Size of copied MS: {size} GB.")
                    else:
                        LOG.error(f"Failed to copy {ms_name}: {cp_result.stderr}")


                else:
                    Path(copy_path).touch()
                    size = "0"
                    LOG.info(f"Created placeholder for MS: {copy_path}")

                cursor.execute("""
                    INSERT INTO metadata (dir_path, dlg_name, base_name, year, start_freq, end_freq, bandwidth, size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (copy_directory, dlg_name, base_name, year, start_freq, end_freq, str(bandwidth), str(size)))
                conn.commit()

                name_list.append(str(dlg_name))

    # Export database table to CSV
    #export_metadata_to_csv(METADATA_DB, METADATA_CSV)
    LOG.info(f"Complete fetch_original_ms list: {name_list}")
    return name_list

def split_ms_list(ms_list: list[str], parallel_processes: int) -> list[list[str]]:
    """
    Split a list of measurement set paths into size-balanced sublists.

    This version divides the input list into `parallel_processes` sublists
    such that the sizes of the sublists differ by at most 1.

    Parameters
    ----------
    ms_list : list of str
        The list of measurement set paths or identifiers to be split.
    parallel_processes : int
        Number of sublists to create.

    Returns
    -------
    list of list of str
        A list containing `parallel_processes` sublists with nearly equal lengths.

    Raises
    ------
    ValueError
        If `parallel_processes` is not a positive integer.

    Examples
    --------
    >>> split_ms_list(["a", "b", "c", "d", "e"], 2)
    [['a', 'b', 'c'], ['d', 'e']]

    >>> split_ms_list(["a", "b", "c"], 5)
    [['a'], ['b'], ['c'], [], []]
    """
    if parallel_processes <= 0:
        raise ValueError("parallel_processes must be a positive integer.")

    total = len(ms_list)
    avg = total // parallel_processes
    remainder = total % parallel_processes

    LOG.debug(f"Splitting {total} items into {parallel_processes} parts.")
    LOG.debug(f"Base chunk size: {avg}, with {remainder} chunk(s) getting an extra item.")

    ms_list_list = []
    start = 0
    for i in range(parallel_processes):
        end = start + avg + (1 if i < remainder else 0)
        chunk = ms_list[start:end]
        ms_list_list.append(chunk)
        LOG.debug(f"Chunk {i + 1}: {chunk}")
        start = end

    LOG.debug(f"Final list of chunks: {ms_list_list}")
    return ms_list_list


def split_out_frequencies(
        ms_in_list: List[str],
        output_directory: str,
        frequencies: List[List[int]],
        METADATA_DB: str,
        process_ms: bool = process_ms_flag
) -> ndarray:
    """
    Splits measurement sets into sub-MSs based on provided frequency ranges,
    using metadata stored in a SQLite database.

    Parameters
    ----------
    METADATA_DB : str
        Path to the database
    ms_in_list : list of str
        List of hashed input MS names to be processed (matches dlg_name in DB).
    output_directory : str
        Path where output measurement sets will be stored.
    frequencies : list of [int, int]
        List of frequency ranges in MHz as [start_freq, end_freq].
    process_ms : bool
        If True, will run CASA imager to extract frequency ranges.
        If False, will only create placeholder files in the output directory.
    """

    LOG.info("#" * 60)
    LOG.info("#" * 60)
    LOG.info(f"Frequencies: {frequencies}")

    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    transform_data_all = []

    for ms_in in ms_in_list:
        # Fetch year and base_name from DB
        cursor.execute("SELECT year, base_name FROM metadata WHERE dlg_name = ?", (ms_in,))
        row = cursor.fetchone()

        if row:
            year, base_name = row
        else:
            LOG.warning(f"No metadata found for {ms_in}, skipping.")
            continue

        for freq_pair in frequencies:
            freq_start = freq_pair[0]
            freq_end = freq_pair[1]
            outfile_name = generate_hashed_ms_name(
                ms_name=ms_in,
                year=str(year),
                start_freq=str(freq_start),
                end_freq=str(freq_end)
            )

            outfile_name_tar = f"{outfile_name}.tar"
            outfile = os.path.join(output_directory, outfile_name)

            ms_in_path = os.path.join(output_directory, ms_in)
            # Check if already exists in DB

            cursor.execute("SELECT 1 FROM metadata WHERE dlg_name = ?", (outfile_name_tar,))
            if cursor.fetchone():
                LOG.info(f"Skipping {outfile_name_tar}, already in metadata DB.")

            else:
                transform_data = [ms_in_path, outfile, output_directory, outfile_name_tar, base_name,
                                  str(year), str(freq_start), str(freq_end)]

                transform_data_string = stringify_data(transform_data)
                transform_data_all.append(transform_data_string)
                LOG.info(f"Queueing {outfile_name_tar} for processing.")


    conn.close()
    transform_data_all = np.array(transform_data_all, dtype=str)
    LOG.info(f"transform_data_all: {transform_data_all}")
    return transform_data_all


def ensure_list_then_destringify(arg_or_list) -> list[str]:
    """
    If input is a stringified list, evaluate it to a list first,
    then pass it to destringify_data(). If it's already a list, pass it directly.
    """
    if isinstance(arg_or_list, str):
        try:
            parsed = ast.literal_eval(arg_or_list)
            if not isinstance(parsed, list):
                raise ValueError("Parsed string is not a list")
            return destringify_data(parsed)
        except Exception as e:
            raise ValueError(f"Failed to parse transform_data string: {e}")
    elif isinstance(arg_or_list, list):
        return destringify_data(arg_or_list)
    else:
        raise TypeError("Expected str or list[str] for transform_data")


def insert_metadata_from_transform(transform_data: str, METADATA_DB: str) -> None:
    """
    Insert metadata for a transformed measurement set into the database.

    Parameters
    ----------
    transform_data : str
        A string containing a Python-style list:
        "['ms_in_path', 'outfile', 'spw_range', 'output_directory', 'outfile_name_tar', 'base_name', 'year', 'freq_start', 'freq_end']"

    METADATA_DB : str
        Path to the SQLite database file.
    """
    LOG.info(f"transform_data before destringing: {transform_data}")
    try:
        # Safely evaluate the string into a Python list
        data_list = ensure_list_then_destringify(transform_data)
        if not isinstance(data_list, list) or len(data_list) != 8:
            raise ValueError("transform_data must be a list of 8 elements")
    except Exception as e:
        raise ValueError(f"Invalid transform_data format: {e}")

    LOG.info(f"transform_data after destringing: {data_list}")

    (
        ms_in_path, outfile, output_directory,
        outfile_name_tar, base_name, year, freq_start, freq_end
    ) = data_list

    outfile_tar = os.path.join(output_directory, outfile_name_tar)
    size_bytes = os.path.getsize(outfile_tar) if os.path.exists(outfile_tar) else 0
    size = round(float(size_bytes / (1024 * 1024 * 1024)), 3)
    bandwidth = int(freq_end) - int(freq_start)

    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO metadata (
            dir_path, dlg_name, base_name, year,
            start_freq, end_freq, bandwidth, size
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        output_directory, outfile_name_tar, base_name, year,
        freq_start, freq_end, bandwidth, size
    ))
    conn.commit()
    conn.close()

    LOG.info(f"Appended {outfile_name_tar} to metadata DB.")


