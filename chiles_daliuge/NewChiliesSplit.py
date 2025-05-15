from os import rename
from pathlib import Path
from pprint import pformat
import sys
import os
from subprocess import run, PIPE
from casatasks import mstransform
from casatools import ms, imager
from typing import List
from common import *

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


process_ms_flag = True



db_dir = "/home/00103780/dlg/db"
METADATA_CSV = db_dir+"/Chilies_metadata.csv"

METADATA_DB = os.path.join(db_dir, "Chilies_metadata.db")






def fetch_original_ms(
        source_dir: str,
        year_list: list[str],
        copy_directory: str,
        process_ms: bool = process_ms_flag
) -> list[str]:
    """
    Fetch measurement sets from a remote directory structure using rclone, track them with metadata in SQLite.

    This function scans a remote source directory structured by year/date to locate
    Measurement Set (.ms) directories. It optionally creates local placeholder files
    or copies the MS directories to a specified local directory, renaming each using a
    hash of metadata fields. Metadata is logged in a persistent SQLite database.

    Parameters
    ----------
    source_dir : str
        The root remote directory containing year/date subfolders with .ms sets.
    year_list : list of str
        A list of year or year-ranges to process (e.g., ["2013-2014", "2015"]).
    copy_directory : str
        Local destination directory where the .ms files or placeholders will be stored.
    make_directory : bool, optional
        If True, creates the `copy_directory` if it doesn't exist. Default is True.
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

    verify_db_integrity()


    make_directory = True
    start_freq = "0944"
    end_freq = "1420"
    bandwidth = int(end_freq) - int(start_freq)
    name_list = []

    initialize_metadata_environment(db_dir)

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

            ms_dirs = [f"{date_path}{line.strip('/')}" for line in result.stdout.strip().splitlines() if line.endswith(".ms/")]

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
    export_metadata_to_csv(METADATA_DB, METADATA_CSV)
    LOG.info(f"Complete fetch_original_ms list: {name_list}")
    return name_list


# def do_ms_transform(ms_in: str, outfile_ms: str, spw_range: str, frequencies: List[List[int]]) -> None:
#     """
#     Performs a frequency-based transformation (splitting and regridding) of a Measurement Set (MS)
#     and outputs the result to a new MS. Handles both single and multi-IF (spectral window) cases.
#
#     Steps:
#     - Checks and removes any previously created output files or intermediate temporary files.
#     - Uses CASA's `mstransform` to extract and optionally combine specific spectral windows.
#     - If multiple spectral windows are involved, creates an intermediate MS with combined SPWs,
#       then extracts the desired channels into the final output.
#     - Logs the spectral window structure of the output MS.
#     - Creates a tar archive of the resulting MS and marks it as completed using a `.tar` file, where as incomplete tar files have _tmp as suffix.
#
#     Parameters
#     ----------
#     ms_in : str
#         Path to the input Measurement Set to be processed.
#     outfile_ms : str
#         Output path (without `.tar`) where the processed MS and its tar archive will be stored.
#     spw_range : str
#         Spectral window range string, e.g., "0:10~50" or "0:10~50,1:20~60".
#     frequencies : list of [int, int]
#         Frequency range (in MHz) used in the current processing step. Used only for logging.
#
#     Returns
#     -------
#     None
#     """
#     # if exists(outfile_ms):
#     #     LOG.info(f"Removing: {outfile_ms}") # should not get here as check on DB exists before call of this function
#     #     remove_file_or_directory(outfile_ms)
#
#
#     for suffix in ["", ".tmp", ".tar"]:
#         path = f"{outfile_ms}{suffix}"
#         if exists(path):
#             LOG.info(f"Removing: {path}")
#             remove_file_or_directory(path)
#
#
#     LOG.info(f"ms_in: {ms_in}")
#     LOG.info(f"outfile_ms: {outfile_ms}")
#     LOG.info(f"spw_range: {spw_range}")
#
#     if len(spw_range.split(",")) == 1:
#         # Single spectral window
#         mstransform(
#             vis=ms_in,
#             outputvis=outfile_ms,
#             regridms=True,
#             restfreq="1420.405752MHz",
#             mode="channel",
#             outframe="TOPO",
#             interpolation="linear",
#             veltype="radio",
#             width=1,
#             spw=spw_range,
#             combinespws=False,
#             nspw=0,
#             createmms=False,
#             datacolumn="data",
#             numsubms=1
#         )
#     else:
#         # Multiple spectral windows - process in two steps
#         tmp_spws = [entry.split(":")[0] for entry in spw_range.split(",")]
#         outfile_tmp = outfile_ms.replace(".ms",".ms.tmp")
#         LOG.info(f"outfile_tmp: {outfile_tmp}")
#
#         if exists(outfile_tmp):
#             shutil.rmtree(outfile_tmp)
#
#         # First step: combine SPWs
#         mstransform(
#             vis=ms_in,
#             outputvis=outfile_tmp,
#             regridms=True,
#             restfreq="1420.405752MHz",
#             mode="channel",
#             outframe="TOPO",
#             interpolation="linear",
#             veltype="radio",
#             width=1,
#             spw=",".join(tmp_spws),
#             combinespws=True,
#             nspw=0,
#             createmms=False,
#             datacolumn="data",
#             numsubms=1
#         )
#
#         # Second step: extract desired channels from combined SPW
#         tmp1_start = int(spw_range.split("~")[0].split(":")[1])
#         #nchan = int(spw_range.split("~")[1])
#         nchans_per_spw = int(spw_range.split(",")[0].split("~")[1]) - tmp1_start
#         total_nchans = nchans_per_spw * len(tmp_spws)
#
#         spw_final = f"*:{tmp1_start}~{tmp1_start + total_nchans}"
#         mstransform(
#             vis=outfile_tmp,
#             outputvis=outfile_ms,
#             regridms=True,
#             restfreq="1420.405752MHz",
#             mode="channel",
#             outframe="TOPO",
#             interpolation="linear",
#             veltype="radio",
#             width=1,
#             spw=spw_final,
#             combinespws=False,
#             nspw=0,
#             createmms=False,
#             datacolumn="data",
#         )
#
#         if exists(outfile_tmp):
#             shutil.rmtree(outfile_tmp)
#             remove_file_or_directory(path)
#
#     # Log spectral window information
#     ms_ = ms()
#     ms_.open(thems=outfile_ms)
#     LOG.info(
#         f"Created File: {outfile_ms}\n"
#         f"Frequency: {frequencies}\n"
#         f"Spectral Window Range: {spw_range}\n"
#         f"Spectral Window Info: {pformat(ms_.getspectralwindowinfo(), indent=2)}\n"
#     )
#     ms_.close()
#
#     # Archive the result
#     create_tar_file(outfile_ms, suffix="tmp")
#     rename(f"{outfile_ms}.tar.tmp", f"{outfile_ms}.tar")
#     LOG.info(f"Created final file {outfile_ms}.tar from {outfile_ms}.tar.tmp")



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
        LOG.debug(f"Chunk {i+1}: {chunk}")
        start = end

    LOG.debug(f"Final list of chunks: {ms_list_list}")
    return ms_list_list




def split_out_frequencies(
        ms_in_list: List[str],
        output_directory: str,
        frequencies: List[List[int]],
        process_ms: bool = process_ms_flag
) -> List:
    """
    Splits measurement sets into sub-MSs based on provided frequency ranges,
    using metadata stored in a SQLite database.

    Parameters
    ----------
    ms_in_list : list of str
        List of hashed input MS names to be processed (matches dlg_name in DB).
    output_directory : str
        Path where output measurement sets will be stored.
    frequencies : list of [int, int]
        List of frequency ranges in MHz as [start_freq, end_freq].
    process_ms : bool
        If True, will run CASA imager to extract frequency ranges.
        If False, will only create placeholder files in the output directory.
    metadata_db_path : str
        Path to the SQLite database storing metadata info.
    """
    CHANNEL_WIDTH = 15625.0  # Hz
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
            freq_start=freq_pair[0]
            freq_end=freq_pair[1]
            outfile_name = generate_hashed_ms_name(
                ms_name=ms_in,
                year=str(year),
                start_freq=str(freq_start),
                end_freq=str(freq_end)
            )

            outfile_name_tar = f"{outfile_name}.tar"
            outfile = os.path.join(output_directory, outfile_name)

            outfile_tar = os.path.join(output_directory, outfile_name_tar)

            ms_in_path = os.path.join(output_directory, ms_in)
            # Check if already exists in DB

            cursor.execute("SELECT 1 FROM metadata WHERE dlg_name = ?", (outfile_name_tar,))
            if cursor.fetchone():
                LOG.info(f"Skipping {outfile_name_tar}, already in metadata DB.")
                continue

            if not process_ms:

                LOG.info(f"Working on: {outfile_tar} with freq {freq_start} and {freq_end}.")
                LOG.info(f"ms_in_path: {ms_in_path}")

                os.makedirs(os.path.dirname(outfile_tar), exist_ok=True)
                Path(f"{outfile_tar}").touch()
                LOG.info(f"Created placeholder file: {outfile_tar}")

            else:
                LOG.info(f"Working on: {outfile} with freq {freq_start} and {freq_end}.")

                im = imager()
                LOG.info(f"ms_in_path: {ms_in_path}")
                im.selectvis(vis=ms_in_path)
                selinfo = im.advisechansel(
                    freqstart=freq_start * 1e6,
                    freqend=freq_end * 1e6,
                    freqstep=CHANNEL_WIDTH,
                    freqframe="BARY",
                )
                LOG.info(f"advisechansel result: {selinfo}")
                spw_range = ""
                for n in range(len(selinfo["ms_0"]["spw"])):
                    spw_range += (
                        f"{selinfo['ms_0']['spw'][n]}:"
                        f"{selinfo['ms_0']['start'][n]}~"
                        f"{selinfo['ms_0']['start'][n] + selinfo['ms_0']['nchan'][n]}"
                    )
                    if (n + 1) < len(selinfo["ms_0"]["spw"]):
                        spw_range += ","
                im.close()

                LOG.info(f"spw_range: {spw_range}, width_freq: {CHANNEL_WIDTH}")
                if spw_range.startswith("-1") or spw_range.endswith("-1"):
                    LOG.warning(f"The spw_range is {spw_range} which is outside the spectral window")
                    continue

                try:
                    if len(spw_range):
                        transform_data = [ms_in_path, outfile, spw_range, output_directory, outfile_name_tar, base_name, str(year), str(freq_start), str(freq_end)]
                        transform_data_all.append(transform_data)
                        #do_ms_transform(ms_in_path, outfile, spw_range, [freq_pair])
                    else:
                        LOG.warning("*********\nmstransform spw out of range:\n***********")
                        continue
                except Exception:
                    LOG.exception("*********\nmstransform exception:\n***********")
                    continue


    conn.close()
    LOG.info(f"transform_data_all: {transform_data_all}")
    return transform_data_all

def stringify_transform_data(transform_data: list):
    return str([str(x) for x in transform_data])

def insert_metadata_from_transform(transform_data: list) -> None:
    """
    Insert metadata for a transformed measurement set into the database.

    Parameters
    ----------
    transform_data : list
        A list containing the following elements:
        [ms_in_path, outfile, spw_range, output_directory, outfile_name_tar,
         base_name, year, freq_start, freq_end]
    """
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    (
        ms_in_path, outfile, spw_range, output_directory,
        outfile_name_tar, base_name, year, freq_start, freq_end
    ) = transform_data

    outfile_tar = os.path.join(output_directory, outfile_name_tar)
    size_bytes = os.path.getsize(outfile_tar) if os.path.exists(outfile_tar) else 0
    size = round(float(size_bytes / (1024 * 1024 * 1024)), 3)
    bandwidth = int(freq_end) - int(freq_start)

    cursor.execute("""
        INSERT INTO metadata (dir_path, dlg_name, base_name, year, start_freq, end_freq, bandwidth, size)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        output_directory, outfile_name_tar, base_name, year,
        freq_start, freq_end, bandwidth, size
    ))
    conn.commit()
    LOG.info(f"Appended {outfile_name_tar} to metadata DB.")


#verify_db_integrity()
#export_metadata_to_csv(METADATA_DB, METADATA_CSV)