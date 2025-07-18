import logging
import os
import pandas as pd
from os.path import exists, isfile, basename
import tarfile
import shutil
import hashlib
import sqlite3
from os.path import isdir
import numpy as np
import json

LOG = logging.getLogger(f"dlg.{__name__}")
logging.basicConfig(level=logging.INFO)


def stringify_data(data: list):
    """
    Convert a list of items into a stringified list of strings.

    Each element in the input list is first converted to a string, and the entire
    list is then represented as a Python-style stringified list. This is useful for
    safe serialization into text-based formats such as CSV or database fields.

    Parameters
    ----------
    data : list
        A list of arbitrary elements to stringify.

    Returns
    -------
    str
        A string representation of the list with each element converted to a string.

    Example
    -------
    >>> stringify_data([123, 'abc', 4.5])
    "['123', 'abc', '4.5']"
    """

    stringified_data = str([str(x) for x in data])
    LOG.info(f"stringified_data: {stringified_data}")
    return stringified_data


def convert_type(s):
    """
    Convert a string to its most appropriate Python type (None, int, float, or str).

    The function strips leading/trailing whitespace and quotes, then attempts
    to interpret the string as:
    - None (if the string is "None")
    - int (if the string contains only digits)
    - float (if it can be converted to a float)
    - str (fallback if all else fails)

    Parameters
    ----------
    s : str
        The input string to convert.

    Returns
    -------
    None, int, float, or str
        The converted value based on the string content.

    Examples
    --------
    >>> convert_type("42")
    42

    >>> convert_type(" 3.14 ")
    3.14

    >>> convert_type("'None'")
    None

    >>> convert_type("hello")
    'hello'
    """

    s = s.strip().strip('"').strip("'")
    if s == "None":
        return None
    elif s.isdigit():
        return int(s)
    try:
        return float(s)
    except ValueError:
        return s


def destringify_data_uvsub(args: list[str]) -> list:
    """
    Parse and convert a list of stringified command-line arguments into native Python types.

    This function is designed to handle arguments that may include embedded lists
    or mixed types (e.g., strings, ints, floats, None), particularly those passed
    as stringified input from the command line or from serialized text formats.

    It:
    - Strips surrounding brackets.
    - Splits by top-level commas while respecting quoted sublists.
    - Parses quoted list-like strings via `json.loads`.
    - Converts primitive values using `convert_type()`.

    Parameters
    ----------
    args : list of str
        A list of strings (typically from `sys.argv[1:]` or similar) representing
        serialized Python values or lists.

    Returns
    -------
    list
        A list of native Python objects (e.g., lists, strings, ints, floats, None),
        suitable for unpacking into function calls.

    Examples
    --------
    >>> destringify_data_uvsub(["['a', 'b']", '123', '3.14', "'None'"])
    [['a', 'b'], 123, 3.14, None]
    """

    # Join and strip outer brackets
    full_str = " ".join(args).strip()
    if full_str.startswith("[") and full_str.endswith("]"):
        full_str = full_str[1:-1]

    # Now split top-level comma-separated values, being careful with quoted strings
    raw_parts = full_str.split(", ")

    parsed_args = []
    temp_buffer = ""
    inside_list = False

    for part in raw_parts:
        part = part.strip()
        # Start of a list
        if part.startswith('["') and not inside_list:
            inside_list = True
            temp_buffer = part
        elif inside_list:
            temp_buffer += ", " + part
            if part.endswith('"]'):
                # Finish list
                inside_list = False
                try:
                    parsed_args.append(json.loads(temp_buffer.replace("'", '"')))
                except Exception:
                    parsed_args.append(temp_buffer)  # fallback
                temp_buffer = ""
        else:
            # Not a list, parse individual type
            parsed_args.append(convert_type(part))

    return parsed_args

def destringify_concat_input(arg: str) -> tuple:
    """
    Parse a stringified flat list into start_freq, end_freq, file list, and output_name.

    Example:
    "['948', '952', 'file1.ms.tar', 'file2.ms.tar', 'abc.ms']"
    → (948, 952, ['file1.ms.tar', 'file2.ms.tar'], 'abc.ms')
    """
    import ast
    flat = ast.literal_eval(arg)  # returns list of strings
    start_freq = int(flat[0])
    end_freq = int(flat[1])
    output_name = flat[-1]
    files = flat[2:-1]
    return {
        "start_freq": start_freq,
        "end_freq": end_freq,
        "files": files,
        "output_name": output_name
    }

def destringify_data(args: list[str]) -> list[str]:
    """
    Clean transform arguments by:
    - Stripping leading/trailing whitespace
    - Removing unexpected brackets and commas
    - Ensuring all elements are clean strings
    """
    cleaned = []
    for i, arg in enumerate(args):
        if i == 0:
            arg = arg.lstrip("[").strip()
        if i == len(args) - 1:
            arg = arg.rstrip("]").strip()
        cleaned.append(arg.strip().rstrip(","))
    return cleaned


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


def remove_file_or_directory(filename: str, trigger) -> None:
    """
    Remove a file or directory if it exists.

    This function checks whether the given path exists. If it points to a file,
    the file is removed. If it points to a directory, the entire directory tree
    is removed.

    Parameters
    ----------
    filename : str
        Path to the file or directory to be removed.
    trigger : any
        Just to make it wait

    Returns
    -------
    None
    """
    filename = str(filename)

    if not filename or filename == "None":
        LOG.warning(f"Skipping removal: invalid filename input ('{filename}').")
        return

    if exists(filename):
        try:
            if isfile(filename):
                LOG.info(f"[{trigger}] Removing file: {filename}")
                os.remove(filename)
            else:
                LOG.info(f"[{trigger}] Removing directory: {filename}")
                shutil.rmtree(filename)
            LOG.info(f"[{trigger}] Successfully removed: {filename}")
        except Exception as e:
            LOG.error(f"[{trigger}] Failed to remove {filename}: {e}")
    else:
        LOG.info(f"[{trigger}] Nothing to remove, path does not exist: {filename}")

def verify_db_integrity(db_path: str, trigger_in: bool) -> bool: # , trigger_in: bool
    """
    Verify that all file references in the metadata SQLite database exist on disk,
    and clean up invalid or missing references accordingly.

    This verifies the 'ms_path' and all dynamic columns (those not fixed).

    Parameters
    ----------
    db_path : str
        Path to the SQLite metadata database file.
    trigger_in : bool
        If True, perform integrity check. If False, skip.

    Returns
    -------
    bool
        True if check was performed and completed successfully.
    """
    if not os.path.exists(db_path):
        LOG.warning(f"[VERIFY] Metadata DB not found at {db_path}. Skipping integrity check.")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all columns
    cursor.execute("PRAGMA table_info(metadata);")
    all_columns = [col[1] for col in cursor.fetchall()]

    # Columns not to be touched
    fixed_columns = ["ms_path", "base_name", "year", "start_freq", "end_freq", "bandwidth", "size"]
    dynamic_columns = [col for col in all_columns if col not in fixed_columns]

    # Select needed fields
    select_cols = ["rowid", "ms_path"] + dynamic_columns
    cursor.execute(f"SELECT {', '.join(select_cols)} FROM metadata")
    rows = cursor.fetchall()

    for row in rows:
        rowid = row[0]
        ms_path = row[1]
        dynamic_values = dict(zip(dynamic_columns, row[2:]))

        existing_refs = []
        missing_fields = []

        # Check ms_path
        if ms_path and os.path.exists(ms_path.strip()):
            existing_refs.append(ms_path)
        else:
            LOG.debug(f"[VERIFY] ms_path missing in row {rowid}")

        # Check dynamic fields
        for col, val in dynamic_values.items():
            if not val or not val.strip():
                continue
            path = val.strip()
            if os.path.exists(path):
                existing_refs.append(path)
            else:
                missing_fields.append(col)

        if not existing_refs:
            cursor.execute("DELETE FROM metadata WHERE rowid = ?", (rowid,))
            LOG.info(f"[DELETE] Row {rowid} removed: all file references missing.")
        else:
            for col in missing_fields:
                cursor.execute(f"UPDATE metadata SET {col} = NULL WHERE rowid = ?", (rowid,))
                LOG.info(f"[CLEAN] Cleared column '{col}' in row {rowid}: file not found.")

    conn.commit()
    conn.close()
    LOG.info("[VERIFY] Metadata DB integrity check complete.")
    return True


def export_metadata_to_csv(db_path: str, csv_path: str, trigger_in: bool) -> None:
    """
    Export the entire 'metadata' table from a SQLite database to a CSV file.

    Parameters
    ----------
    trigger_in:
        to make function wait
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


def add_column_if_missing(db_path: str, column_name: str, column_type: str = "TEXT") -> None:
    """
    Add a new column to the metadata table if it does not already exist.

    Parameters
    ----------
    db_path:
        path to the database
    column_name : str
        The name of the column to add.
    column_type : str, optional
        The SQLite data type of the new column (default is "TEXT").

    Notes
    -----
    - If the column already exists, the function does nothing.
    - This modifies the existing `metadata` table schema.
    """
    conn = sqlite3.connect(db_path)
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


def log_input(x_in):
    LOG.info(f"Input: {x_in}")
    LOG.info(f"Input type: {type(x_in)}")

    try:
        size = len(x_in)
    except TypeError:
        size = "N/A (not a sized object)"

    LOG.info(f"Input size: {size}")


def trigger_db(x_in):
    """
    Determines whether the input should trigger a database operation.

    Returns True if:
    - The input is a non-empty NumPy array and its first element is NOT a string.
    - The input is an empty NumPy array.
    - The input is not a NumPy array.

    Returns False only if the input is a non-empty NumPy array and the first element is a string.
    """
    LOG.info(f"Received input: {x_in}")
    LOG.info(f"Type: {type(x_in)}")

    if isinstance(x_in, np.ndarray):
        LOG.info(f"Input is a NumPy array with size {x_in.size}")
        if x_in.size > 0:
            first_element = x_in[0]
            LOG.info(f"First element: {first_element}")
            LOG.info(f"First element type: {type(first_element)}")
            result = not isinstance(first_element, str)
            LOG.info(f"Inverted result: {result}")
            return result
        else:
            LOG.info("Array is empty. Returning True.")
            return True
    else:
        LOG.info("Input is not a NumPy array. Returning True.")
        return True



# Setup basic logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

def log_data(input_data):
    """
    Log type and content of input_data. If input_data behaves like a string, log its value.
    """
    try:
        LOG.info(f"[log_data] Input type: {type(input_data).__name__}")
    except Exception:
        LOG.info("[log_data] Input type could not be determined")

    try:
        if isinstance(input_data, str):
            LOG.info(f"[log_data] Input string content: {input_data}")
        else:
            LOG.info(f"[log_data] Input (non-str) value: {str(input_data)}")
    except Exception as e:
        LOG.warning(f"[log_data] Failed to stringify input_data: {e}")

    # if isinstance(input_data, np.str_):
    #     input_data = str(input_data)
    #     LOG.info(f"[log_data] Updated input type: {type(input_data).__name__}")

    return input_data


import sqlite3

def update_metadata_column(
    db_path: str,
    match_column: str,
    match_value: str,
    year: str,
    start_freq: str,
    end_freq: str,
    column_name: str,
    column_value: str
) -> None:
    """
    Update a single column value in the metadata table for all matching rows.

    Supports wildcard '*' in match_value, year, start_freq, or end_freq to match multiple rows.

    Parameters
    ----------
    db_path : str
        Path to the SQLite metadata database.
    match_column : str
        Name of the column to match (e.g., "ms_path", "base_name", etc.).
    match_value : str
        Value to match in match_column, or '*' to match all.
    year : str
        Value to match for `year`, or '*' to match all.
    start_freq : str
        Value to match for `start_freq`, or '*' to match all.
    end_freq : str
        Value to match for `end_freq`, or '*' to match all.
    column_name : str
        The name of the column to update.
    column_value : str
        The value to insert into the specified column.

    Raises
    ------
    ValueError
        If the target or match column does not exist in the metadata table.
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Validate column names
    cursor.execute("PRAGMA table_info(metadata);")
    valid_columns = [row[1] for row in cursor.fetchall()]
    if column_name not in valid_columns:
        conn.close()
        raise ValueError(f"Column '{column_name}' does not exist in metadata table.")
    if match_column not in valid_columns:
        conn.close()
        raise ValueError(f"Match column '{match_column}' does not exist in metadata table.")

    # Build WHERE clause dynamically
    conditions = []
    params = []

    if match_value != "*":
        conditions.append(f"{match_column} = ?")
        params.append(match_value)

    for field, value in [("year", year), ("start_freq", start_freq), ("end_freq", end_freq)]:
        if value != "*":
            conditions.append(f"{field} = ?")
            params.append(value)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Perform the update
    sql = f"""
        UPDATE metadata
        SET {column_name} = ?
        WHERE {where_clause}
    """
    cursor.execute(sql, [column_value] + params)
    updated_count = cursor.rowcount

    conn.commit()
    conn.close()

    print(f"[INFO] Updated {updated_count} row(s) in column '{column_name}' with value '{column_value}'.")



def remove_temp_dir(trigger_in, base_dir: str, prefix="__SKY_TEMP__"):
    """
    Removes all temporary directories in the given base directory that start with a specific prefix.

    Parameters
    ----------
    trigger_in:
        to make function wait
    base_dir : str
        The directory in which to search for temp directories.
    prefix : str
        The prefix of temp directories to be deleted.

    Returns
    -------
    None
    """

    LOG.info(f"Scanning for temp dirs in: {base_dir} with prefix: {prefix}")

    if not os.path.exists(base_dir):
        LOG.warning(f"Base directory does not exist: {base_dir}")
        return

    count = 0
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and entry.startswith(prefix):
            try:
                shutil.rmtree(full_path)
                LOG.info(f"Removed temp directory: {full_path}")
                count += 1
            except Exception as e:
                LOG.error(f"Failed to remove {full_path}: {e}")

    LOG.info(f"Completed removal. Total directories removed: {count}")


def remove_file_directory(path_name):
    """
    Remove a file or directory at the specified path.

    If the path points to a directory, it is removed recursively using `shutil.rmtree`.
    If it is a file, it is removed using `os.remove`.

    Parameters
    ----------
    path_name : str
        Full path to the file or directory to be removed.

    Notes
    -----
    - Logs the removal action using `LOG.info`.
    - No action is taken if the path does not exist; any resulting `FileNotFoundError` is not explicitly handled.
    """

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


def initialize_metadata_environment(db_path: str) -> bool:
    """
    Ensure the metadata database and metadata table exist.

    Parameters
    ----------
    db_path : str
        Full path to the SQLite database file.

    Notes
    -----
    - Creates the SQLite database file and metadata table if not present.
    - Creates the parent directory of `db_path` if it does not exist.
    """
    # Ensure parent directory exists
    parent_dir = os.path.dirname(db_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    # Connect to the DB and create table if needed
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            ms_path TEXT PRIMARY KEY,
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
    initialized = True
    return initialized
