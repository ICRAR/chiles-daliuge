import sys
import os
import sqlite3
from subprocess import run, PIPE

from pathlib import Path
import ast

from typing import List

from numpy import ndarray

from chiles_daliuge.common import *
import logging
import sqlite3
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

process_ms_flag = True

def fetch_original_ms(
        source_dir: str,
        year_list: list[str],
        copy_directory: str,
        METADATA_DB: str,
        add_to_db: bool = True,
        process_ms: bool = process_ms_flag,
) -> list[str]:

    copy_directory = os.path.expandvars(copy_directory)
    make_directory = True
    start_freq = "0944"
    end_freq = "1420"
    bandwidth = int(end_freq) - int(start_freq)
    name_list = []

    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()

    result = run(["rclone", "lsjson", source_dir, "-R", "--dirs-only"], stdout=PIPE,
                 stderr=PIPE, text=True)
    if result.returncode != 0:
        LOG.error(f"Failed to list {source_dir}: {result.stderr}")
        return []

    # year_dirs = [line.strip("/") for line in result.stdout.strip().splitlines()]

    copy_tasks = []

    pattern = re.compile(r".+\.ms$") # Only accept path results that end in .ms/
    dirs = json.loads(result.stdout)
    all_ms_paths = [e['Path'] for e in dirs if pattern.match(e['Path'])]
    selected_paths = []
    for year in year_list:
        for path in all_ms_paths:
            if year in path:
                selected_paths.append({"path": os.path.normpath(f"{source_dir}/{path}"),
                                      "year": year})

    if make_directory:
        os.makedirs(copy_directory, exist_ok=True)

    for ms_name in selected_paths:
        base_name = os.path.basename(ms_name["path"])
        dlg_name = generate_hashed_ms_name(ms_name["path"],
                                           ms_name["year"],
                                           start_freq,
                                            end_freq)
        ms_path = os.path.join(copy_directory, dlg_name)

        if add_to_db:
            cursor.execute("SELECT 1 FROM metadata WHERE ms_path = ?", (ms_path,))
            if cursor.fetchone():
                LOG.info(f"Skipping fetch of existing MS: {base_name}, already recorded as {ms_path}")
                name_list.append(str(ms_path))
                continue

        command = {
            "base_name": base_name,
            "ms_name": ms_name["path"],
            "ms_path": ms_path,
            "year": ms_name["year"],
            "cmd": [
                "rclone", "copy", ms_name["path"], ms_path,
                "--progress",
                "--s3-disable-checksum",
                "--s3-chunk-size", "1024M",
                "--s3-upload-concurrency", "8",
                "--transfers", "4",
                "--ignore-times",
                "--retries", "1",
                "--local-no-set-modtime",
                "--log-level", "INFO"
            ]
        }

        LOG.info(f"Adding command: {command['cmd']}")
        LOG.info(f"Queued for copy: {ms_name['path']} → {ms_path}")
        copy_tasks.append(command)

    # Execute copy tasks in parallel (max 5 at a time)
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(run, task["cmd"], stdout=PIPE, stderr=PIPE, text=True): task for task in copy_tasks}
        for future in as_completed(futures):
            task = futures[future]
            base_name = task["base_name"]
            ms_path = task["ms_path"]
            year = task["year"]

            try:
                cp_result = future.result()
            except Exception as e:
                LOG.error(f"Copy failed for {base_name}: {e}")
                continue

            size = "Unknown"

            if cp_result.returncode == 0:
                LOG.info(f"Copied MS to: {ms_path}")
                if not os.path.exists(ms_path):
                    LOG.error("%s does not exist!", ms_path)
                size_result = run(["rclone", "size", ms_path], stdout=PIPE, stderr=PIPE, text=True)
                if size_result.returncode == 0:
                    for line in size_result.stdout.strip().splitlines():
                        if line.startswith("Total size:") and "(" in line:
                            try:
                                size_bytes_str = line.split("(")[-1].split()[0]
                                size_bytes = int(size_bytes_str)
                                size = round(size_bytes / (1024 * 1024 * 1024), 3)  # GB
                                break
                            except ValueError:
                                LOG.warning(f"Could not parse size from line: {line}")

                LOG.info(f"Size of copied MS: {size} GB.")
                if size != "Unknown" and float(size) > 0:
                    cursor.execute("""
                        INSERT INTO metadata (ms_path, base_name, year, start_freq, end_freq, bandwidth, size)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (ms_path, base_name, year, start_freq, end_freq, str(bandwidth), str(size)))
                    conn.commit()
                    name_list.append(str(ms_path))
            else:
                LOG.error(f"Failed to copy {task['ms_name']}: {cp_result.stderr}")

    conn.close()
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

    os.makedirs(output_directory, exist_ok=True)
    LOG.info("#" * 60)
    LOG.info("#" * 60)
    LOG.info(f"Frequencies: {frequencies}")


    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    transform_data_all = []

    for ms_in in ms_in_list:
        # Fetch year and base_name from DB
        cursor.execute("SELECT year, base_name FROM metadata WHERE ms_path = ?", (ms_in,))
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


            outfile_path = os.path.join(output_directory, outfile_name)
            outfile_tar_path = f"{outfile_path}.tar"

            ms_in_path = os.path.join(output_directory, ms_in)
            # Check if already exists in DB

            cursor.execute("SELECT 1 FROM metadata WHERE ms_path = ?", (outfile_tar_path,))
            if cursor.fetchone():
                LOG.info(f"Skipping {outfile_path}, already in metadata DB.")

            else:
                transform_data = [ms_in_path, base_name, outfile_path, outfile_tar_path,
                                  str(year), str(freq_start), str(freq_end)]

                transform_data_string = stringify_data(transform_data)
                transform_data_all.append(transform_data_string)
                LOG.info(f"Queueing {outfile_tar_path} for processing.")


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


def update_db_after_transform(transform_data: str, db_path: str) -> None:
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
            base_name (str),
            outfile_path (str),
            outfile_name_tar (str),
            year (str or int),
            freq_start (str or int),
            freq_end (str or int)
        ]

    db_path : str
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
        if not isinstance(data_list, list) or len(data_list) != 7:
            raise ValueError("transform_data must be a list of 7 elements")
    except Exception as e:
        raise ValueError(f"Invalid transform_data format: {e}")

    LOG.info(f"transform_data after destringing: {data_list}")

    (
        ms_in_path, base_name, outfile_path, outfile_tar_path, year, freq_start, freq_end
    ) = data_list

    size_bytes = os.path.getsize(outfile_tar_path) if os.path.exists(outfile_tar_path) else 0
    size = round(float(size_bytes / (1024 * 1024 * 1024)), 3)
    bandwidth = int(freq_end) - int(freq_start)

    if size <= 0:
        size = -1
        LOG.warning(f"{outfile_tar_path} is not a valid MS. ")
        try:
            os.makedirs(os.path.dirname(outfile_tar_path), exist_ok=True)
            with open(outfile_tar_path, "w"):
                pass  # equivalent to touch
        except Exception as e:
            LOG.error(f"Failed to create dummy file at {outfile_tar_path}: {e}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO metadata (
            ms_path, base_name, year,
            start_freq, end_freq, bandwidth, size
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        outfile_tar_path, base_name, year,
        freq_start, freq_end, bandwidth, size
    ))
    conn.commit()
    conn.close()

    LOG.info(f"Appended {outfile_tar_path} to metadata DB.")




