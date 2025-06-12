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
    Fetches Measurement Set (.ms) directories from a remote directory structure using rclone,
    and tracks them using metadata stored in an SQLite database.

    The function navigates a remote source directory structured by year/date (e.g., /remote/2013/01/)
    to locate .ms directories. It either creates placeholder files (if `process_ms=False`) or copies
    the MS directories locally (if `process_ms=True`) into a destination folder using rclone.

    Metadata such as hashed names, frequency information, and sizes are recorded in a persistent
    SQLite database. Previously recorded MS entries are skipped.

    Parameters
    ----------
    source_dir : str
        Root remote directory containing year/date subfolders with .ms directories.
    year_list : list of str
        List of years to include in the search (e.g., ["2013", "2014"]).
    copy_directory : str
        Local destination where .ms files or placeholders will be stored.
    trigger_in : bool
        Currently unused; placeholder for future logic or triggering conditions.
    METADATA_DB : str
        Path to the SQLite database used to store and check MS metadata.
    process_ms : bool, optional
        If True, rclone will copy full .ms directories to `copy_directory`.
        If False, only empty placeholder files will be created. Default is taken from `process_ms_flag`.

    Returns
    -------
    list of str
        List of hashed .ms names (real or placeholder) that were processed or already recorded.

    Notes
    -----
    - Uses rclone commands (`lsf`, `copy`, `size`) to interact with the remote filesystem.
    - Assumes MS directories have `.ms/` suffix.
    - Avoids reprocessing MS sets that already exist in the metadata database.
    - Logs all significant actions and errors.
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
    Split a list of measurement set paths into nearly equal-sized sublists.

    The function divides `ms_list` into `parallel_processes` sublists such that the
    lengths of the sublists differ by at most one. This is useful for distributing
    work evenly across parallel processes.

    Parameters
    ----------
    ms_list : list of str
        List of measurement set paths or identifiers to be split.
    parallel_processes : int
        Number of sublists (i.e., parallel processes) to create.

    Returns
    -------
    list of list of str
        A list containing `parallel_processes` sublists, each with approximately equal length.

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
    Converts a stringified list or a list of strings into a destringified list.

    If the input is a string representation of a list (e.g., '["a", "b"]'),
    it is parsed using `ast.literal_eval()` and then passed to `destringify_data()`.
    If the input is already a list, it is passed directly.
    The function ensures the final output is a list of destringified strings.

    Parameters
    ----------
    arg_or_list : str or list of str
        A stringified list (e.g., from a CSV or database) or an actual list of strings.

    Returns
    -------
    list of str
        A destringified list derived from the input.

    Raises
    ------
    ValueError
        If a string input is not a valid list or fails to parse.
    TypeError
        If the input is neither a string nor a list.
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
    Parses transformation metadata and inserts it into a SQLite database.

    This function expects a stringified list containing metadata about a
    transformed measurement set, including input/output paths and frequency details.
    It destringifies the input, calculates bandwidth and file size (if available),
    and appends the entry to the metadata table in the database.

    Parameters
    ----------
    transform_data : str
        A stringified Python-style list with exactly 8 elements:
        [
            ms_in_path (str),
            outfile (str),
            output_directory (str),
            outfile_name_tar (str),
            base_name (str),
            year (str or int),
            freq_start (str or int),
            freq_end (str or int)
        ]

    METADATA_DB : str
        Path to the SQLite metadata database file.

    Raises
    ------
    ValueError
        If the input string cannot be parsed into a valid list of 8 elements.
    OSError
        If reading the file size of `outfile_name_tar` fails for reasons
        other than the file not existing.

    Side Effects
    ------------
    - Inserts a new row into the `metadata` table of the SQLite database.
    - Logs messages to the standard logger.
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


