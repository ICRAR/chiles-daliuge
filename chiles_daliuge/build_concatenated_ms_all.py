import os
import sys
import shutil
import tempfile
import ast
import logging
from os.path import join, basename, exists, isdir, isfile
from chiles_daliuge.common import untar_file
from casatasks import concat
from chiles_daliuge.common import *
from typing import List
import re
# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def sanitize_broken_dict_str(s: str) -> dict:
    """
    Sanitize a malformed dict-like string and convert it into a proper Python dict.

    Steps:
    1. Add quotes around keys.
    2. Add quotes around unquoted list elements.
    3. Add quotes around standalone string values.
    4. Parse using ast.literal_eval and return dict.
    """

    LOG.info("[sanitize] Original input string:")
    LOG.info(s)

    # Step 1: Add quotes around keys
    s1 = re.sub(r'([{,]\s*)(\w+)\s*:', r'\1"\2":', s)
    LOG.info("[sanitize] After quoting keys:")
    LOG.info(s1)

    # Step 2: Quote unquoted list elements
    s2 = re.sub(
        r'\[\s*([^,\[\]]+?)\s*(?=[,\]])',
        lambda m: f'"{m.group(1).strip()}"' if not m.group(1).strip().isdigit() else m.group(1),
        s1
    )
    LOG.info("[sanitize] After quoting list values:")
    LOG.info(s2)

    # Step 3: Quote unquoted string values (outside of lists)
    s3 = re.sub(r':\s*([a-zA-Z_][\w\.\-]*)\s*([,}])', r': "\1"\2', s2)
    LOG.info("[sanitize] After quoting string values:")
    LOG.info(s3)

    # Step 4: Convert to dict
    try:
        result_dict = ast.literal_eval(s3)
        LOG.info("[sanitize] Final parsed dictionary:")
        LOG.info(result_dict)
        return result_dict
    except Exception as e:
        LOG.error(f"[sanitize] Failed to parse sanitized string: {e}")
        raise ValueError("Sanitized string could not be parsed as a valid dictionary.") from e


# --- Main logic ---
def build_concatenated_ms_all(data: dict, output_directory: str, db_path: str):
    # Validate input keys
    required_keys = {"start_freq", "end_freq", "files", "output_name"}
    missing = required_keys - data.keys()


    if missing:
        raise ValueError(f"Missing required keys in input: {missing}")

    start_freq = data["start_freq"]
    end_freq = data["end_freq"]
    files = data["files"]
    output_name = data["output_name"]

    # Create temp directory
    with tempfile.TemporaryDirectory(dir=output_directory, prefix="concat_tmp_") as temporary_directory:
        LOG.info(f"Created temp directory: {temporary_directory}")
        input_measurement_sets = []



        # Build output paths
        combine_file_build = join(output_directory, f"{output_name}.building")
        combine_file_final = join(output_directory, output_name)

        # Skip if already built
        if exists(combine_file_final):
            LOG.info(f"{combine_file_final} already exists, skipping.")
            return

        # Untar all files
        for tar_path in files:
            if not exists(tar_path):
                raise FileNotFoundError(f"Missing file: {tar_path}")

            untar_file(tar_path, temporary_directory, gz=False)
            ms_path = join(temporary_directory, basename(tar_path)[:-4])
            input_measurement_sets.append(ms_path)

        # Remove old .building dir if exists
        if exists(combine_file_build):
            LOG.info(f"Removing existing build dir: {combine_file_build}")
            shutil.rmtree(combine_file_build)

        # Call concat
        concat(vis=input_measurement_sets, concatvis=combine_file_build)

        # Move final output
        if exists(combine_file_build):
            shutil.move(combine_file_build, combine_file_final)
            LOG.info(f"Build complete: {combine_file_final}")
            for tar_path in files:
                update_metadata_column(db_path, "uv_sub_path", tar_path, "*", str(start_freq), str(end_freq), "build_concat_all", combine_file_final)
        else:
            raise RuntimeError("Concat failed — build file not created.")

def main():
    try:
        raw_args = sys.argv[1:]  # multiple items like '948', '952', 'f1.ms.tar', ..., 'output_name.ms'
        LOG.info(f"[RAW ARGS] {raw_args}")

        if len(raw_args) < 3:
            print("Usage: python build_concatenated_ms_all.py <stringified_list> <output_directory> <db_path>")
            sys.exit(1)

        # Extract paths from the end
        output_directory = raw_args[-2]
        db_path = raw_args[-1]
        arg_str = " ".join(raw_args[:-2])  # join list parts

        LOG.info(f"[ARGS] output_directory: {output_directory}")
        LOG.info(f"[ARGS] db_path: {db_path}")
        LOG.info(f"[ARGS] input_str: {arg_str}")

        os.makedirs(output_directory, exist_ok=True)

        # Validate paths
        if not os.path.isdir(output_directory):
            LOG.error(f"Invalid output_directory: {output_directory}")
            sys.exit(1)

        if not os.path.isfile(db_path):
            LOG.error(f"Invalid db_path: {db_path}")
            sys.exit(1)

        # Parse list using your destringify logic
        parsed_list = destringify_data_uvsub([arg_str])
        LOG.info(f"[PARSED] {parsed_list}")

        # Unpack the list: [start, end, file1, file2, ..., output_name]
        if len(parsed_list) < 4:
            raise ValueError("Parsed input must have at least start, end, 1 file, and output_name")

        start_freq = parsed_list[0]
        end_freq = parsed_list[1]
        output_name = parsed_list[-1]
        files = parsed_list[2:-1]

        # Reconstruct input dict
        data = {
            "start_freq": int(start_freq),
            "end_freq": int(end_freq),
            "files": files,
            "output_name": output_name
        }

        build_concatenated_ms_all(data, output_directory, db_path)

    except Exception as e:
        LOG.exception("Failed to run build job")
        sys.exit(1)


if __name__ == "__main__":
    main()
