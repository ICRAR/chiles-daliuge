import logging
import os
import pandas as pd
from os.path import exists, isfile, basename
import tarfile
import shutil
import hashlib
import sqlite3, os
from pathlib import Path

from os.path import isdir
import numpy as np
import json
import re
from typing import List, Any

sqlite3.register_adapter(Path, os.fspath)

LOG = logging.getLogger(f"dlg.{__name__}")
logging.basicConfig(level=logging.INFO)


def expand_path(p: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(p))).resolve(strict=False)

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


def destringify_data_tclean(args: List[str]) -> List[Any]:
    """
    Turn something like:
      ["[3c2ef5fab83c5355.ms, 952, 956, ..., False, ['/path/to/in.ms']]"]
    back into:
      ['3c2ef5fab83c5355.ms','952','956',...,'False', ['/path/to/in.ms']]
    """
    # 1) reassemble into one string and strip outer brackets
    raw = " ".join(args).strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]

    # 2) split on commas that are NOT inside square‑brackets
    parts = re.split(r',\s*(?![^\[]*\])', raw)

    out: List[Any] = []
    for part in parts:
        token = part.strip().rstrip(",")  # remove extra spaces/trailing commas

        # 3) nested list at the end?
        if token.startswith("[") and token.endswith("]"):
            inner = token[1:-1].strip()
            if inner:
                # split inner list on commas
                sub = [s.strip().strip("\"'") for s in inner.split(",")]
            else:
                sub = []
            out.append(sub)
            continue

        # 4) strip quotes if any
        if (token.startswith("'") and token.endswith("'")) or \
           (token.startswith('"') and token.endswith('"')):
            token = token[1:-1]

        # leave everything else as string
        out.append(token)

    return out

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


def destringify_data_concat(args: list[str]) -> list[str]:
    """
    Expect input like:
        ['[concat_all;944;992;/path/a.ms.tar,/path/b.ms.tar,...]']
    Return:
        ['concat_all', '944', '992', '/path/a.ms.tar,/path/b.ms.tar,...']
    """
    if not args:
        raise ValueError("No arguments provided")

    input_arg = str(args[3:-3])

    LOG.info(f"input_arg: {input_arg}")

    # # 1) Collapse to a single string
    # blob = " ".join(a for a in input_arg if a is not None).strip()
    #
    # LOG.info(f"blob: {blob}")
    #
    # # 2) Remove outermost [ ... ] if present
    # #    (use first '[' and last ']' so doubled brackets don't confuse us)
    # if "['[" in blob and "]']" in blob:
    #     i = blob.find("['[")
    #     j = blob.rfind("]']")
    #     if 0 <= i < j:
    #         blob = blob[i+1:j].strip()
    #
    # # 3) Strip one pair of matching outer quotes if present
    # if len(blob) >= 2 and blob[0] == blob[-1] and blob[0] in ("'", '"'):
    #     blob = blob[1:-1].strip()
    #
    # blob = str(blob)
    # 4) Split strictly by semicolons into 4 parts
    parts = [p.strip() for p in input_arg.split(";", 3)]
    if len(parts) != 4:
        raise ValueError(f"Expected 4 semicolon-separated parts, got {len(parts)} from: {blob!r}")

    base_name, start_freq, end_freq, paths_combined = parts
    return [base_name, start_freq, end_freq, paths_combined]





def get_list_frequency_groups(
    frequency_width: int,
    frequency_step: int,
    minimum_frequency: int,
    maximum_frequency: int
) -> List[List[int]]:
    """
    Generate sliding frequency ranges within [minimum_frequency, maximum_frequency).

    The window has width `frequency_width` and advances by `frequency_step` each time:
      - Overlap if frequency_step < frequency_width
      - Adjacent bins if frequency_step == frequency_width
      - Gaps if frequency_step > frequency_width

    Parameters
    ----------
    frequency_width : int
        Width of each frequency bin (> 0).
    frequency_step : int
        Step between successive bin starts (> 0).
    minimum_frequency : int
        Starting frequency of the range (inclusive).
    maximum_frequency : int
        Ending frequency of the range (exclusive).

    Returns
    -------
    list[list[int]]
        A list of [start, end] pairs where end = min(start + width, maximum_frequency).

    Examples
    --------
    >>> get_list_frequency_groups(2, 2, 0, 6)   # adjacent bins
    [[0, 2], [2, 4], [4, 6]]

    >>> get_list_frequency_groups(4, 2, 0, 10)  # overlap (step < width)
    [[0, 4], [2, 6], [4, 8], [6, 10], [8, 10]]

    >>> get_list_frequency_groups(3, 5, 0, 12)  # gaps (step > width)
    [[0, 3], [5, 8], [10, 12]]
    """
    frequency_width = int(frequency_width)
    frequency_step = int(frequency_step)
    minimum_frequency = int(minimum_frequency)
    maximum_frequency = int(maximum_frequency)

    if frequency_width <= 0:
        raise ValueError("frequency_width must be > 0")
    if frequency_step <= 0:
        raise ValueError("frequency_step must be > 0")
    if minimum_frequency >= maximum_frequency:
        return []

    result: List[List[int]] = []
    for start in range(minimum_frequency, maximum_frequency, frequency_step):
        if start >= maximum_frequency:
            break
        end = start + frequency_width
        if end > maximum_frequency:
            end = maximum_frequency
        result.append([start, end])

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


def verify_db_integrity(db_path: str) -> str:
    """
    Verify that file/directory paths stored in the metadata DB actually exist.
    - Clears invalid path columns to NULL.
    - Deletes rows where *all* path columns are invalid/missing.

    Checked columns (if present):
      ms_path, uv_sub_path, build_concat_all, tclean_all, build_concat_epoch, tclean_epoch
    """

    db_path_out = expand_path(db_path)
    db_path = str(db_path_out)



    if not os.path.exists(db_path):
        LOG.warning(f"[VERIFY] Metadata DB not found at {db_path}. Skipping integrity check.")
        return db_path_out

    # Columns we consider as path-bearing (limit to these to avoid nuking non-path fields)
    PATH_COLUMNS = [
        "ms_path",
        "uv_sub_path",
        "build_concat_all",
        "tclean_all",
        "build_concat_epoch",
        "tclean_epoch",
    ]

    def _is_real_path(p: str) -> bool:
        """Return True if p (after strip/expanduser) exists as file or dir."""
        if not p:
            return False
        p = p.strip()
        if not p:
            return False
        # expand ~ and normalize; we don't require absolute here
        q = Path(p).expanduser()
        return q.exists()

    def _table_exists(cur, table_name: str) -> bool:
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cur.fetchone() is not None

    def _validate_table(cur, table_name: str, candidate_path_cols: list[str]) -> None:
        """Core validator for a single table."""
        if not _table_exists(cur, table_name):
            LOG.info(f"[VERIFY:{table_name}] Table not found; skipping.")
            return db_path_out

        # Discover which of the candidate path columns actually exist in the table
        cur.execute(f"PRAGMA table_info({table_name});")
        cols_present = {col[1] for col in cur.fetchall()}
        path_cols = [c for c in candidate_path_cols if c in cols_present]

        if not path_cols:
            LOG.info(f"[VERIFY:{table_name}] No known path columns found; nothing to check.")
            return db_path_out

        select_cols = ["rowid"] + path_cols
        cur.execute(f"SELECT {', '.join(select_cols)} FROM {table_name}")
        rows = cur.fetchall()

        for row in rows:
            rowid = row[0]
            values_by_col = dict(zip(path_cols, row[1:]))

            existing_any = False
            missing_cols = []

            for col, val in values_by_col.items():
                # SQLite may return bytes if the column affinity is BLOB or similar
                val_str = val if isinstance(val, str) else (val.decode() if isinstance(val, bytes) else "")
                if _is_real_path(val_str):
                    existing_any = True
                else:
                    # Only mark as missing if the cell is non-empty; empty/NULL stays NULL
                    if val_str:
                        missing_cols.append(col)

            if not existing_any:
                cur.execute(f"DELETE FROM {table_name} WHERE rowid = ?", (rowid,))
                LOG.info(f"[DELETE:{table_name}] Row {rowid} removed: all path columns invalid/missing.")
            else:
                for col in missing_cols:
                    # Column name comes from PRAGMA (trusted), value parameterized
                    cur.execute(f"UPDATE {table_name} SET {col} = NULL WHERE rowid = ?", (rowid,))
                    LOG.info(f"[CLEAN:{table_name}] Cleared '{col}' in row {rowid}: path not found.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Single transaction spanning all checks
        cursor.execute("BEGIN")
        # Validate metadata table (same columns as your original function)
        _validate_table(
            cursor,
            "metadata",
            [
                "ms_path",
                "uv_sub_path",
                "build_concat_all",
                "tclean_all",
                "build_concat_epoch",
                "tclean_epoch",
            ],
        )
        # Validate concat_freq table (new)
        _validate_table(
            cursor,
            "concat_freq",
            ["concat_freq_path"],
        )

        conn.commit()
        LOG.info("[VERIFY] DB integrity check complete for 'metadata' and 'concat_freq'.")

        return db_path_out
    except Exception as e:
        conn.rollback()
        LOG.exception(f"[VERIFY] Integrity check failed: {e}")

        return db_path_out
    finally:
        conn.close()



def export_metadata_to_csv(db_path: str, csv_path: str) -> None:
    """
    Export the entire 'metadata' and 'concat_freq' tables from a SQLite database to CSV files.

    Behavior
    --------
    - If `csv_path` is a file path ending with .csv, append the table name before the extension,
      e.g., '/tmp/export.csv' -> '/tmp/export_metadata.csv' and '/tmp/export_concat_freq.csv'.
    - If `csv_path` is a directory (existing or not), writes '/path/metadata.csv' and '/path/concat_freq.csv'.

    Parameters
    ----------
    trigger_in : bool
        If False, the function returns immediately without exporting.
    db_path : str
        Path to the SQLite database file.
    csv_path : str
        Base path for the output CSV(s).

    Returns
    -------
    None
    """

    db_path = expand_path(db_path)
    csv_path = expand_path(csv_path)
    def _derive_out_path(base: str, table: str) -> Path:
        p = Path(base)
        # If base looks like a .csv file, insert the table name before the suffix.
        if p.suffix.lower() == ".csv":
            stem = p.stem
            return p.with_name(f"{stem}_{table}.csv")
        # Otherwise treat as a directory
        return p.joinpath(f"{table}.csv")

    # Ensure parent directory exists for directory-like base
    base_is_dir = Path(csv_path).suffix.lower() != ".csv"
    if base_is_dir:
        Path(csv_path).mkdir(parents=True, exist_ok=True)

    tables = ["metadata", "concat_freq"]

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        for table in tables:
            # Check table existence
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
            if cur.fetchone() is None:
                LOG.info(f"[EXPORT] Table '{table}' not found; skipping.")
                continue

            out_path = _derive_out_path(csv_path, table)
            # Ensure parent dir exists for file-like base, too
            out_path.parent.mkdir(parents=True, exist_ok=True)

            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            df.to_csv(out_path, index=False)
            LOG.info(f"[EXPORT] Wrote {table} -> {out_path}")
    finally:
        conn.close()

    LOG.info("Export complete for available tables: metadata, concat_freq.")



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

    db_path = expand_path(db_path)

    conn = sqlite3.connect(str(db_path))
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
) -> bool:
    """
    Update a single column value in the metadata table for all matching rows.

    Returns
    -------
    bool
        True if one or more rows matched the criteria (regardless of whether the
        UPDATE changed the value); False if no rows matched.

    Notes
    -----
    - Supports wildcard '*' in match_value, year, start_freq, and end_freq.
    - Validates that `column_name` and `match_column` exist to prevent SQL injection.
    """
    db_path = expand_path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()

        # Validate column names
        cursor.execute("PRAGMA table_info(metadata);")
        valid_columns = [row[1] for row in cursor.fetchall()]
        if column_name not in valid_columns:
            raise ValueError(f"Column '{column_name}' does not exist in metadata table.")
        if match_column not in valid_columns:
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

        # First, check if any rows match
        cursor.execute(f"SELECT COUNT(1) FROM metadata WHERE {where_clause}", params)
        match_count = cursor.fetchone()[0]

        if match_count == 0:
            # No matches -> nothing to update; tell caller to insert
            print("[INFO] No matching rows found; nothing updated.")
            return False

        # Perform the update only if there are matches
        sql = f"UPDATE metadata SET {column_name} = ? WHERE {where_clause}"
        cursor.execute(sql, [column_value] + params)
        updated_count = cursor.rowcount  # may be 0 if values were already identical

        conn.commit()

        if updated_count > 0:
            print(f"[INFO] Updated {updated_count} row(s) in column '{column_name}' with value '{column_value}'.")
        else:
            print(f"[INFO] {match_count} row(s) matched but already had '{column_name}' = '{column_value}'. No changes made.")

        # Return True because matches existed (update not needed or succeeded)
        return True

    finally:
        conn.close()


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
def insert_concat_freq_row(
    db_path: str,
    concat_freq_path: str,  # keep signature as str to discourage passing Path
    base_name: str,
    year: str,
    start_freq: str,
    end_freq: str,
    bandwidth: str,
    size: str,
) -> None:
    db_path = expand_path(db_path)
    sql = """
        INSERT INTO concat_freq (
            concat_freq_path, base_name, year, start_freq, end_freq, bandwidth, size
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(sql, (
            str(concat_freq_path),  # <= ensure str
            base_name,
            year,
            start_freq,
            end_freq,
            bandwidth,
            size,
        ))
        conn.commit()


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
    db_path = expand_path(db_path)
    # Ensure parent directory exists
    parent_dir = os.path.dirname(db_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    # Connect to the DB and create table if needed
    conn = sqlite3.connect(str(db_path))
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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS concat_freq (
            concat_freq_path TEXT PRIMARY KEY,
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
