import tempfile
from os.path import join
import subprocess
from pathlib import Path
from chiles_daliuge.common import *
import logging

process_ms_flag = True

LOG = logging.getLogger(f"dlg.{__name__}")


def ensure_local_sky_model(
    acacia_bucket: str,
    dirname: str
) -> str:
    """
    Ensure a local sky model directory exists and return its absolute path as str.
    Raises on failure (never returns an empty string).
    """
    LOG.info(f"Starting ensure_local_sky_model")
    sky_model_dir = dirname #"/home/00103780/dlg/LSM"
    if not isinstance(sky_model_dir, str):
        raise ValueError("sky_model_local must be a filesystem path string")

    src = Path(sky_model_dir)
    src = src.expanduser().resolve()

    def _rclone_copyto(remote: str, local: str):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        LOG.info(f"Fetching {remote} -> {local} via rclone")
        subprocess.run(["rclone", "copyto", remote, local], check=True)

    # If caller passed a directory (or a path without suffix they intend to be a dir)
    if src.suffix == "" or src.is_dir():
        dir_path = src
        tar_path = (dir_path if dir_path.suffix == ".tar" else dir_path.with_suffix(".tar"))

        if dir_path.is_dir():
            if any(dir_path.iterdir()):  # checks if the directory has at least one entry
                LOG.info(f"[ensure] Found local non-empty directory: {dir_path}")
                return src
            else:
                LOG.warning(f"[ensure] Directory exists but is empty: {dir_path}")
        else:
            LOG.warning(f"[ensure] Directory does not exist: {dir_path}")

        # Ensure we have a tar to untar
        if not tar_path.exists():
            LOG.info(f"[ensure] {dir_path} not found. {tar_path} not found. Attempting rclone fetch.")
            _rclone_copyto(acacia_bucket, str(tar_path))
        else:
            LOG.info(f"[ensure] Using existing tar: {tar_path}")

        LOG.info(f"[ensure] Untarring {tar_path} to parent {tar_path.parent}")
        untar_file(str(tar_path), str(tar_path.parent), gz=False)

        produced_dir = tar_path.parent / tar_path.stem
        if produced_dir.is_dir():
            LOG.info(f"[ensure] Created directory: {produced_dir}")
            return src

        # Fallback: sometimes tars expand flat. Use parent if it now contains files.
        if any(p.is_dir() or p.is_file() for p in tar_path.parent.iterdir()):
            LOG.warning(f"[ensure] Tar did not produce a named subfolder. Returning parent {tar_path.parent}")

        raise FileNotFoundError(f"[ensure] After untar, no directory found for {tar_path}")

    # If caller passed a .tar file
    if src.suffix == ".tar":
        tar_path = src
        dir_path = tar_path.parent / tar_path.stem

        if dir_path.is_dir():
            LOG.info(f"[ensure] Found local directory next to tar: {dir_path}")
            return src

        if not tar_path.exists():
            LOG.info(f"[ensure] {tar_path} not found. Attempting rclone fetch.")
            _rclone_copyto(acacia_bucket, str(tar_path))
        else:
            LOG.info(f"[ensure] Using existing tar: {tar_path}")

        LOG.info(f"[ensure] Untarring {tar_path} to parent {tar_path.parent}")
        untar_file(str(tar_path), str(tar_path.parent), gz=False)

        if dir_path.is_dir():
            LOG.info(f"[ensure] Created directory: {dir_path}")
            return src

        if any(p.is_dir() or p.is_file() for p in tar_path.parent.iterdir()):
            LOG.warning(f"[ensure] Tar did not produce a named subfolder. Returning parent {tar_path.parent}")
            return src

        raise FileNotFoundError(f"[ensure] After untar, no directory found for {tar_path}")

    # Unsupported suffix
    raise ValueError(f"[ensure] Unsupported path: {src} (suffix {src.suffix})")


def copy_sky_model(
    sky_model_dir: list,
    temporary_directory: str,
) -> str:
    """
    Copy the local sky model directory (ensured by ensure_local_sky_model) into a
    temporary directory and convert any absolute symbolic links into relative links.

    This preserves original external behavior while delegating presence/fetch/untar
    to `ensure_local_sky_model`.
    """

    def make_symlinks_relative(target_dir: str, source_root: str):
        for root, dirs, files in os.walk(target_dir):
            for name in dirs + files:
                full_path = os.path.join(root, name)
                if os.path.islink(full_path):
                    link_target = os.readlink(full_path)
                    if os.path.isabs(link_target):
                        abs_target_path = os.path.realpath(link_target)
                        # Only rewrite absolute symlinks that point within the original source tree
                        if abs_target_path.startswith(os.path.realpath(source_root)):
                            rel_target_path = os.path.relpath(abs_target_path, start=os.path.dirname(full_path))
                            os.remove(full_path)
                            os.symlink(rel_target_path, full_path)

    # 2) Copy directory into `temporary_directory`
    #basename = os.path.basename(sky_model_dir.rstrip(os.sep))

    LOG.info(f"copy: sky_model_dir: {sky_model_dir}")

    # Ensure basename is a string path
    basename = os.path.basename(os.path.normpath(sky_model_dir))

    # Ensure temporary_directory is a string (not a function)
    if callable(temporary_directory):
        temporary_directory = temporary_directory()

    dest_path = os.path.join(temporary_directory, basename)

    LOG.info(f"Copying directory {sky_model_dir} to {dest_path}")
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)

    shutil.copytree(sky_model_dir, dest_path, symlinks=True)
    make_symlinks_relative(dest_path, source_root=sky_model_dir)

    return dest_path


def fetch_split_ms(
        year_list: List[str],
        frequencies: List[List[int]],
        db_path: str,
        trigger_in: bool,
) -> List[str]:
    """
    Retrieve `ms_path` entries from a metadata SQLite database matching specified years and frequency ranges.

    This function queries the `metadata` table to find measurement sets whose `year` matches any
    value in `year_list` and whose `[start_freq, end_freq]` pair matches any tuple in `frequencies`.
    The result is a list of formatted strings with ms_path and associated metadata.

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
        "ms_path;year;start_freq;end_freq"
    """
    db_path = expand_path(db_path)
    db_path = str(db_path)
    LOG.info(f"Adding 'uv_sub_path' column to DB.")
    add_column_if_missing(db_path, "uv_sub_path")

    freq_set = {tuple(freq_pair) for freq_pair in frequencies}
    LOG.info(f"freq_set: {freq_set}")
    query = """
        SELECT ms_path, year, start_freq, end_freq, size
        FROM metadata
        WHERE uv_sub_path IS NULL
           OR TRIM(uv_sub_path) = ''
    """

    matching_dlg_names = []

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for row in cursor.execute(query):
            ms_path, year, start_freq, end_freq , size = row
            #print(f"\nChecking row: ms_path={ms_path}, year={year}, start_freq={start_freq}, end_freq={end_freq}")

            if float(size) > 0:
                freq_tuple = (int(start_freq), int(end_freq))  # ✅ convert to int

                if year in year_list and freq_tuple in freq_set:
                    # print(f"  → Appending: {ms_path}")
                    matching_dlg_names.append(f"{ms_path};{year};{freq_tuple[0]};{freq_tuple[1]}")
            else:
                LOG.warning(f"Measurement Set with following details is not valid, ignoring it for uvsub list. {ms_path}, {year}, {start_freq}, {end_freq}")

    return matching_dlg_names


def do_uvsub(names_list, save_dir, sky_model_dir,
             taylor_terms, outliers, channel_average, produce_qa, w_projection_planes, METADATA_DB):
    """
    Prepares data for UV subtraction by checking for existing processed entries,
    extracting sky models, and assembling configuration metadata.

    Parameters
    ----------
    names_list : list of str
        List of semicolon-separated strings of the form "split_name;year;freq_start;freq_end".
    save_dir : str
        Path to the directory to save tar files.
    sky_model_dir : str
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

    LOG.info(f"sky_model_dir: {sky_model_dir}")
    os.makedirs(save_dir, exist_ok=True)

    sky_model_location = None

    uvsub_data_all = []

    temporary_sky_model = tempfile.mkdtemp(dir=save_dir, prefix="__SKY_TEMP__")
    if sky_model_location is None:
        sky_model_location = copy_sky_model(sky_model_dir, temporary_sky_model)

    # conn = sqlite3.connect(METADATA_DB)

    for name in names_list:
        tar_file_split, year, freq_st, freq_en = name.split(";")

        combined_data = [
            taylor_terms, outliers, channel_average, produce_qa, w_projection_planes,
            sky_model_location,
            tar_file_split, year, freq_st, freq_en, METADATA_DB
        ]

        uvsub_data_all.append(stringify_data(combined_data))

    # conn.close()

    uvsub_data_all = np.array(uvsub_data_all, dtype=str)

    LOG.info(f"uvsub_data_all: {uvsub_data_all}")
    # LOG.info(f"sky_model_location: {sky_model_location}")

    return uvsub_data_all
