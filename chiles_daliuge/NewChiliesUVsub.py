import tempfile
from os.path import join

from casatools import quanta
from typing import Union, List
from chiles_daliuge.common import *

process_ms_flag = True

LOG = logging.getLogger(__name__)


def copy_sky_model(sky_model_source: Union[str, bytes], temporary_directory: str) -> str:
    """
    Copy or extract a sky model archive or directory into a temporary directory,
    and convert any absolute symbolic links into relative links.

    Parameters
    ----------
    sky_model_source : str or bytes
        Path to the sky model source. This can be either:
        - A `.tar` file containing the sky model, which will be extracted.
        - A directory containing the sky model, which will be recursively copied.
        Absolute symbolic links within the source will be rewritten as relative
        if they point within the original source tree.

    temporary_directory : str
        Destination directory where the sky model will be placed.

    Returns
    -------
    str
        Path to the extracted or copied sky model inside the temporary directory.

    Raises
    ------
    ValueError
        If `sky_model_source` is neither a `.tar` file nor a valid directory.

    Notes
    -----
    - If the `.tar` file does not extract into a subdirectory, the function returns `temporary_directory`.
    - Relative symbolic links help preserve portability across filesystems and environments.
    """

    def make_symlinks_relative(target_dir: str):
        for root, dirs, files in os.walk(target_dir):
            for name in dirs + files:
                full_path = os.path.join(root, name)
                if os.path.islink(full_path):
                    link_target = os.readlink(full_path)
                    # Only rewrite absolute symlinks that are within the original source tree
                    if os.path.isabs(link_target):
                        abs_target_path = os.path.realpath(link_target)
                        if abs_target_path.startswith(os.path.realpath(sky_model_source)):
                            rel_target_path = os.path.relpath(abs_target_path, start=os.path.dirname(full_path))
                            os.remove(full_path)
                            os.symlink(rel_target_path, full_path)

    if isinstance(sky_model_source, str) and sky_model_source.endswith(".tar"):
        LOG.info(f"Untarring {sky_model_source} to {temporary_directory}")
        untar_file(sky_model_source, temporary_directory, gz=False)

        # Infer the directory created during untar
        basename = os.path.basename(sky_model_source).replace(".tar", "")
        dest_path = os.path.join(temporary_directory, basename)

        if not os.path.exists(dest_path):
            # fallback: return temporary_directory if .tar file doesn't untar into a named subfolder
            return temporary_directory

        make_symlinks_relative(dest_path)
        return dest_path

    elif os.path.isdir(sky_model_source):
        LOG.info(f"Copying directory {sky_model_source} to {temporary_directory}")
        basename = os.path.basename(os.path.abspath(sky_model_source))
        dest_path = os.path.join(temporary_directory, basename)

        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)

        shutil.copytree(sky_model_source, dest_path, symlinks=True)
        make_symlinks_relative(dest_path)
        return dest_path

    else:
        raise ValueError(f"Invalid sky model input: {sky_model_source} is neither a .tar file nor a directory.")




def fetch_split_ms(
        year_list: List[str],
        frequencies: List[List[int]],
        db_path: str,
        trigger_in: bool,
) -> List[str]:
    """
    Retrieve `dlg_name` entries from a metadata SQLite database matching specified years and frequency ranges.

    This function queries the `metadata` table to find measurement sets whose `year` matches any
    value in `year_list` and whose `[start_freq, end_freq]` pair matches any tuple in `frequencies`.
    The result is a list of formatted strings with dlg_name and associated metadata.

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
        "dlg_name;year;start_freq;end_freq"
    """
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

            if year in year_list and freq_tuple in freq_set:
                # print(f"  → Appending: {dlg_name}")
                matching_dlg_names.append(f"{dlg_name};{year};{freq_tuple[0]};{freq_tuple[1]}")

    return matching_dlg_names


def do_uvsub(names_list, source_dir, sky_model_tar_file,
             taylor_terms, outliers, channel_average, produce_qa, w_projection_planes, METADATA_DB):
    """
    Prepares data for UV subtraction by checking for existing processed entries,
    extracting sky models, and assembling configuration metadata.

    Parameters
    ----------
    names_list : list of str
        List of semicolon-separated strings of the form "split_name;year;freq_start;freq_end".
    source_dir : str
        Path to the directory containing the source tar files.
    sky_model_tar_file : str
        Path to the tar file containing the sky model to be used.
    taylor_terms : list of str
        File paths to Taylor term sky models.
    outliers : list of str
        File paths to outlier models.
    channel_average : int
        Number of channels to average before CLEAN or UV subtraction.
    produce_qa : bool
        Whether or not to produce QA plots during processing.
    w_projection_planes : int
        Number of W-projection planes to use in imaging.
    METADATA_DB : str
        Path to the SQLite metadata database used for tracking processed files.

    Returns
    -------
    np.ndarray
        Array of stringified configuration entries for UV subtraction tasks,
        excluding any already present in the metadata database.
    """
    sky_model_location = None
    add_column_if_missing(METADATA_DB, "uv_sub_name")

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

        LOG.info(f"uv_sub_name: {uv_sub_name}")
        uv_sub_tar = f"{uv_sub_name}.tar"
        LOG.info(f"uv_sub_tar: {uv_sub_tar}")
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM metadata WHERE uv_sub_name = ?", (uv_sub_tar,))
        if cursor.fetchone():
            LOG.info(f"Skipping {uv_sub_tar}, already in metadata DB.")
            continue

        combined_data = [
            taylor_terms, outliers, channel_average, produce_qa, w_projection_planes,
            source_dir, sky_model_location,
            split_name, year, freq_st, freq_en, uv_sub_name, METADATA_DB
        ]

        uvsub_data_all.append(stringify_data(combined_data))

    conn.close()
    # sky_model_location = str(sky_model_location)
    uvsub_data_all = np.array(uvsub_data_all, dtype=str)

    LOG.info(f"uvsub_data_all: {uvsub_data_all}")
    # LOG.info(f"sky_model_location: {sky_model_location}")

    return uvsub_data_all
