from os.path import isdir
import tempfile
from datetime import datetime, timedelta
from os import makedirs, rename, listdir
import numpy as np
import pylab as pl
from casaplotms import plotms
from casatasks import uvsub, statwt, split, phaseshift
from casatools import imager, ms, table, quanta, image
from typing import Union, List, Tuple
import ast
from pathlib import Path
from common import *

process_ms_flag = True


LOG = logging.getLogger(__name__)


# db_dir = "/home/00103780/chiles-daliuge/db"
# METADATA_CSV = db_dir+"/Chilies_metadata.csv"
# METADATA_DB = os.path.join(db_dir, "Chilies_metadata.db")










def copy_sky_model(sky_model_source: Union[str, bytes], temporary_directory: str) -> str:
    """
    Extract or copy the sky model to a temporary directory.

    Parameters
    ----------
    sky_model_source : str or bytes
        Path to the `.tar` archive or a directory containing the sky model.
        If a `.tar` file, it will be extracted.
        If a directory, it will be copied directly.
    temporary_directory : str
        Path to the directory where the sky model should be placed.

    Returns
    -------
    str
        Path to the temporary directory containing the sky model.

    Notes
    -----
    - If the input is a .tar file, assumes it is uncompressed.
    - For directories, performs a full recursive copy.
    """
    if isinstance(sky_model_source, str) and sky_model_source.endswith(".tar"):
        LOG.info(f"Untarring {sky_model_source} to {temporary_directory}")
        untar_file(sky_model_source, temporary_directory, gz=False)
    elif os.path.isdir(sky_model_source):
        LOG.info(f"Copying directory {sky_model_source} to {temporary_directory}")
        if os.path.exists(temporary_directory):
            shutil.rmtree(temporary_directory)
        shutil.copytree(sky_model_source, temporary_directory)
    else:
        raise ValueError(f"Invalid sky model input: {sky_model_source} is neither a .tar file nor a directory.")

    return temporary_directory


# def copy_region_files(region_file_tar_file, temporary_directory):
#     untar_file(region_file_tar_file, temporary_directory, gz=False)
#     return join(temporary_directory, "region-files")



def fetch_split_ms(
        year_list: List[str],
        frequencies: List[List[int]],
        db_path: str
) -> List[str]:
    """
    Fetch dlg_name values from the metadata DB where year is in year_list and
    [start_freq, end_freq] matches any pair in the frequencies list.

    Parameters
    ----------
    year_list : list of str
        List of acceptable year values.
    frequencies : list of [int, int]
        List of acceptable [start_freq, end_freq] pairs.
    db_path : str
        Path to the SQLite metadata database.

    Returns
    -------
    list of str
        Matching dlg_name values.
    """
    verify_db_integrity()

    freq_set = {tuple(freq_pair) for freq_pair in frequencies}

    query = """
        SELECT dlg_name, year, start_freq, end_freq FROM metadata
    """

    matching_dlg_names = []

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for row in cursor.execute(query):
            dlg_name, year, start_freq, end_freq = row
            # print(f"\nChecking row: dlg_name={dlg_name}, year={year}, start_freq={start_freq}, end_freq={end_freq}")

            freq_tuple = (int(start_freq), int(end_freq))  # ✅ convert to int

            # if year in year_list:
            #     print(f"  ✓ Year '{year}' is in year_list")
            # else:
            #     print(f"  ✗ Year '{year}' is NOT in year_list")

            # print(f"  → Frequency tuple: {freq_tuple}")
            # if freq_tuple in freq_set:
            #     print(f"  ✓ Frequency {freq_tuple} is in frequency list")
            # else:
            #     print(f"  ✗ Frequency {freq_tuple} is NOT in frequency list")

            if year in year_list and freq_tuple in freq_set:
                #print(f"  → Appending: {dlg_name}")
                matching_dlg_names.append(f"{dlg_name};{year};{freq_tuple[0]};{freq_tuple[1]}")



    return matching_dlg_names





def time_convert(mytime: Union[float, int, str, List[Union[float, int, str]]],
                 myunit: str = "s") -> List[str]:
    """
    Convert one or more time values into human-readable date strings using CASA's quanta module.

    Parameters
    ----------
    mytime : float, int, str, or list of float/int/str
        A single time value or a list of time values to convert. These represent time in the specified unit.
    myunit : str, optional
        The unit of the input time value(s), e.g., "s" for seconds, "d" for days, "Hz", etc.
        Default is "s".

    Returns
    -------
    list of str
        A list of converted date-time strings in "YYYY/MM/DD/HH:MM:SS" format.

    Notes
    -----
    This uses CASA's `quanta` tool to convert time values based on the given unit.
    Even if a single value is passed, the result will be a list of one string.
    """
    if type(mytime).__name__ != "list":
        mytime = [mytime]
    my_timestr = []
    for time in mytime:
        qa = quanta()
        q1 = qa.quantity(time, myunit)
        time1 = qa.time(q1, form="ymd")
        my_timestr.append(time1)
    return my_timestr





def do_uvsub(names_list, source_dir, sky_model_tar_file,
    taylor_terms, outliers, channel_average, produce_qa, w_projection_planes, METADATA_DB
):
    verify_db_integrity()
    sky_model_location = None

    add_column_if_missing("uv_sub_name")

    uvsub_data_all = []

    temporary_sky_model = tempfile.mkdtemp(dir=source_dir, prefix="__SKY_TEMP__")
    if sky_model_location is None:
        sky_model_location = copy_sky_model(sky_model_tar_file, temporary_sky_model)

    conn = sqlite3.connect(METADATA_DB)


    for name in names_list:
        split_name, year, freq_st, freq_en = name.split(";")
        freq_start = int(freq_st)
        freq_end = int(freq_en)

        LOG.info("#" * 60)
        LOG.info("#" * 60)
        LOG.info(f"Processing: {name}")

        tar_file_split = join(
            source_dir, split_name
        )
        LOG.info(f"Checking: {tar_file_split}")

        uv_sub_name = generate_hashed_ms_name(str(tar_file_split), year, str(freq_start), str(freq_end))


        LOG.info("uv_sub_name:",uv_sub_name)
        uv_sub_tar = f"{uv_sub_name}.tar"
        LOG.info("uv_sub_tar:",uv_sub_tar)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM metadata WHERE uv_sub_name = ?", (uv_sub_tar,))
        if cursor.fetchone():
            LOG.info(f"Skipping {uv_sub_tar}, already in metadata DB.")
            continue



        combined_data = [
            taylor_terms, outliers, channel_average, produce_qa, w_projection_planes,
            source_dir,  sky_model_location,
            split_name, year, freq_st, freq_en, "uv_sub_name", METADATA_DB
        ]

        uvsub_data_all.append(stringify_data(combined_data))




    #verify_db_integrity()
    LOG.info(f"uvsub_data_all: {uvsub_data_all}")
    LOG.info("All Done with stringifying uvsub data!!!")
    return np.array(uvsub_data_all, dtype=str)


# def extract_and_update_uvsub_metadata(stringified_input: str) -> None:
#     """
#     Parse a stringified list, clean it, and call update_metadata_column
#     with the last six elements.
#
#     Parameters
#     ----------
#     stringified_input : str
#         A string that represents a list of at least 6 items
#     """
#     try:
#         parsed = ast.literal_eval(stringified_input)
#         if not isinstance(parsed, list) or len(parsed) < 6:
#             raise ValueError("Input must be a stringified list with at least 6 elements")
#
#         cleaned = destringify_data(parsed)
#
#         # Extract the last 6 elements
#         split_name, year, freq_st, freq_en, uv_sub_name, uv_sub_tar = cleaned[-6:]
#
#         LOG.info("Calling update_metadata_column with:")
#         LOG.info(f"  split_name: {split_name}")
#         LOG.info(f"  year: {year}")
#         LOG.info(f"  freq_st: {freq_st}")
#         LOG.info(f"  freq_en: {freq_en}")
#         LOG.info(f"  uv_sub_name: {uv_sub_name}")
#         LOG.info(f"  uv_sub_tar: {uv_sub_tar}")
#
#         update_metadata_column(split_name, year, freq_st, freq_en, uv_sub_name, uv_sub_tar)
#
#     except Exception as e:
#         LOG.error(f"Failed to extract and update metadata: {e}")
#         raise