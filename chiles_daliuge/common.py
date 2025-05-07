import logging
import os
import pandas as pd
from os.path import join, exists, isfile, basename
import tarfile
import shutil
import hashlib
import sqlite3
from os.path import isdir

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)






db_dir = "/home/00103780/dlg/db"
METADATA_CSV = db_dir+"/Chilies_metadata.csv"

METADATA_DB = os.path.join(db_dir, "Chilies_metadata.db")


def get_list_frequency_groups(frequency_width: int, minimum_frequency: int, maximum_frequency: int) -> list[list[int]]:
    """
    Generate frequency ranges based on the given width and bounds.

    This function divides the frequency range from `minimum_frequency` to `maximum_frequency`
    into contiguous intervals of width `frequency_width`.

    Parameters
    ----------
    frequency_width : int
        Width of each frequency bin.
    minimum_frequency : int
        Starting frequency of the range (inclusive).
    maximum_frequency : int
        Ending frequency of the range (exclusive).

    Returns
    -------
    list of list of int
        A list of [start, end] pairs representing the frequency bins.

    Examples
    --------
    >>> get_list_frequency_groups(2, 0, 6)
    [[0, 2], [2, 4], [4, 6]]
    """
    frequency_width = int(frequency_width)
    result = [
        [start, start + frequency_width]
        for start in range(minimum_frequency, maximum_frequency, frequency_width)
    ]
    return result

def remove_file_or_directory(filename: str) -> None:
    """
    Remove a file or directory if it exists.

    This function checks whether the given path exists. If it points to a file,
    the file is removed. If it points to a directory, the entire directory tree
    is removed.

    Parameters
    ----------
    filename : str
        Path to the file or directory to be removed.

    Returns
    -------
    None

    Notes
    -----
    - Uses `os.remove()` for files.
    - Uses `shutil.rmtree()` for directories.
    - If the path does not exist, the function does nothing.
    """
    if exists(filename):
        if isfile(filename):
            os.remove(filename)
        else:
            shutil.rmtree(filename)

def verify_db_integrity() -> None:
    """
    Verify that all file references in the metadata database exist on disk.

    - If a file in `dlg_name` or other dynamic columns (after 'size') is missing, clear that field.
    - If all file references in a row are missing, delete the row.

    Parameters
    ----------
    db_path : str, optional
        Full path to the SQLite metadata database. Uses default METADATA_DB if None.
    """
    db_path = METADATA_DB

    if not os.path.exists(db_path):
        LOG.warning(f"Metadata database not found at {db_path}. Skipping integrity check.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all columns in order
    cursor.execute("PRAGMA table_info(metadata);")
    columns = [col[1] for col in cursor.fetchall()]

    # Identify fixed + dynamic columns
    fixed_columns = ["dir_path", "dlg_name", "base_name", "year", "start_freq", "end_freq", "bandwidth", "size"]
    dynamic_columns = [col for col in columns if col not in fixed_columns]

    # Fetch all rows
    select_cols = ["rowid", "dir_path", "dlg_name"] + dynamic_columns
    query = f"SELECT {', '.join(select_cols)} FROM metadata"
    cursor.execute(query)

    rows = cursor.fetchall()

    for row in rows:
        rowid = row[0]
        dir_path = row[1]
        dlg_name = row[2]
        dynamic_values = dict(zip(dynamic_columns, row[3:]))

        existing_files = []

        # Check dlg_name
        dlg_path = os.path.join(dir_path, dlg_name)
        if os.path.exists(dlg_path):
            existing_files.append(dlg_name)

        # Check dynamic fields
        missing_fields = []
        for col, val in dynamic_values.items():
            if val is None or val.strip() == "":
                continue
            full_path = os.path.join(dir_path, val)
            if os.path.exists(full_path):
                existing_files.append(val)
            else:
                missing_fields.append(col)

        if not existing_files:
            # No valid files → delete entire row
            cursor.execute("DELETE FROM metadata WHERE rowid = ?", (rowid,))
            LOG.info(f"Deleted row {rowid}: no existing files found.")
        else:
            # Clear individual missing fields
            for col in missing_fields:
                cursor.execute(f"UPDATE metadata SET {col} = NULL WHERE rowid = ?", (rowid,))
                LOG.info(f"Cleared column {col} in row {rowid}: file not found.")

    conn.commit()
    conn.close()
    LOG.info("Database integrity check completed.")



def export_metadata_to_csv(db_path: str, csv_path: str) -> None:
    """
    Export the entire 'metadata' table from a SQLite database to a CSV file.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (e.g., METADATA_DB).
    csv_path : str
        Path to the output CSV file (e.g., METADATA_CSV).

    Returns
    -------
    None
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM metadata", conn)
        df.to_csv(csv_path, index=False)
    finally:
        conn.close()

    LOG.info(f"Export complete.")


def add_column_if_missing(column_name: str, column_type: str = "TEXT") -> None:
    """
    Add a new column to the metadata table if it does not already exist.

    Parameters
    ----------
    column_name : str
        The name of the column to add.
    column_type : str, optional
        The SQLite data type of the new column (default is "TEXT").

    Notes
    -----
    - If the column already exists, the function does nothing.
    - This modifies the existing `metadata` table schema.
    """
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()

    # Fetch existing column names
    cursor.execute("PRAGMA table_info(metadata);")
    existing_columns = [row[1] for row in cursor.fetchall()]

    if column_name not in existing_columns:
        alter_query = f"ALTER TABLE metadata ADD COLUMN {column_name} {column_type}"
        LOG.info(alter_query)
        cursor.execute(alter_query)
        conn.commit()

    conn.close()


def update_metadata_column(
        dlg_name: str,
        year: str,
        start_freq: str,
        end_freq: str,
        column_name: str,
        column_value: str
) -> None:
    """
    Update a specific column value in the metadata table for a matched row.

    Parameters
    ----------
    dlg_name : str
        The value of the `dlg_name` field to match.
    year : str
        The value of the `year` field to match.
    start_freq : str
        The value of the `start_freq` field to match.
    end_freq : str
        The value of the `end_freq` field to match.
    column_name : str
        The name of the column to update.
    column_value : str
        The value to insert into the specified column.

    Notes
    -----
    - Only updates the row if an exact match is found.
    - Raises ValueError if the column does not exist.
    """
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()

    # Validate that the column exists
    cursor.execute("PRAGMA table_info(metadata);")
    valid_columns = [row[1] for row in cursor.fetchall()]
    if column_name not in valid_columns:
        conn.close()
        raise ValueError(f"Column '{column_name}' does not exist in metadata table.")

    # Perform the update
    cursor.execute(f"""
        UPDATE metadata
        SET {column_name} = ?
        WHERE dlg_name = ? AND year = ? AND start_freq = ? AND end_freq = ?
    """, (column_value, dlg_name, year, start_freq, end_freq))

    conn.commit()
    conn.close()

def remove_file_directory(path_name):
    if isdir(path_name):
        LOG.info(f"Removing directory {path_name}")
        shutil.rmtree(path_name)
    else:
        LOG.info(f"Removing file {path_name}")
        os.remove(path_name)

def create_tar_file(outfile, suffix=None):
    """
    Create a gzipped tar archive of the specified directory and remove the original directory.

    Parameters
    ----------
    outfile : str
        Path to the directory that needs to be archived (without `.tar` extension).
    suffix : str, optional
        Optional suffix to append to the archive filename after `.tar`, e.g., `outfile.tar.suffix`.
        If None, the output will be named `outfile.tar`.

    Returns
    -------
    None
    """
    file_name = f"{outfile}.tar" if suffix is None else f"{outfile}.tar.{suffix}"
    LOG.info(f"Creating tar file: {file_name}")
    with tarfile.open(file_name, "w:gz") as tar:
        tar.add(outfile, arcname=basename(outfile))

    LOG.info(f"Removing directory: {outfile}")
    shutil.rmtree(outfile)


def untar_file(infile, output_directory, gz=True):
    """
    Extract a tar or tar.gz archive to a specified output directory.

    Parameters
    ----------
    infile : str
        Path to the `.tar` or `.tar.gz` archive file.
    output_directory : str
        Path to the directory where contents will be extracted.
    gz : bool, optional
        Whether the input archive is gzipped. If True, uses gzip mode `"r:gz"`; otherwise, `"r"`.

    Returns
    -------
    None
    """
    LOG.info(f"Untarring file: {infile}")
    with tarfile.open(infile, "r:gz" if gz else "r") as tar:
        tar.extractall(output_directory)


def generate_hashed_ms_name(
        ms_name: str,
        year: str,
        start_freq: str,
        end_freq: str,
        prefix: str = "",
        hash_length: int = 16
) -> str:
    """
    Generate a hashed filename for a measurement set using its metadata.

    This function creates a deterministic, hash-based name by combining the base
    name of the measurement set with the year and frequency range. It returns a
    string in the format "<prefix><hash>.ms", where <hash> is a truncated SHA-256 hash.

    Parameters
    ----------
    ms_name : str
        Full path to the measurement set directory.
    year : str
        Year or year range associated with the dataset.
    start_freq : str
        Start frequency of the observation.
    end_freq : str
        End frequency of the observation.
    prefix : str, optional
        Optional string to prefix the hash with (e.g., "ms_"). Default is "".
    hash_length : int, optional
        Length of the hash string to use. Default is 16 characters.

    Returns
    -------
    str
        A hashed filename ending in ".ms", optionally prefixed.

    Examples
    --------
    >>> generate_hashed_ms_name("path/to/file.ms", "2014", "0944", "1420")
    '3c9a1dfb49e3f1bc.ms'

    >>> generate_hashed_ms_name("file.ms", "2014", "0944", "1420", prefix="ms_")
    'ms_3c9a1dfb49e3f1bc.ms'
    """
    base = os.path.basename(ms_name)
    combined_str = f"{base}_{year}_{start_freq}_{end_freq}"
    hash_str = hashlib.sha256(combined_str.encode()).hexdigest()[:hash_length]
    return f"{prefix}{hash_str}.ms"


def initialize_metadata_environment(copy_directory: str) -> None:
    """
    Ensure the metadata database, table, and output directory exist.

    Parameters
    ----------
    copy_directory : str
        The directory where measurement sets or placeholders will be stored.

    Notes
    -----
    - Creates the SQLite database file and metadata table if not present.
    - Creates the `copy_directory` if it does not exist.
    """
    if not os.path.exists(copy_directory):
        os.makedirs(copy_directory, exist_ok=True)

    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            dir_path TEXT,
            dlg_name TEXT PRIMARY KEY,
            base_name TEXT,
            year TEXT,
            start_freq TEXT,
            end_freq TEXT,
            bandwidth TEXT,
            size TEXT
        )
    """)
    conn.commit()
    conn.close()

