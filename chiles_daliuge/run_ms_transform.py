#!/usr/bin/env python3
from typing import List
from pprint import pformat
import sys
from chiles_daliuge.common import *
import json
import shlex
import sqlite3
import logging

# CASA imports
from casatasks import mstransform
from casatools import ms

#from chiles_daliuge.NewChiliesSplit import insert_metadata_from_transform

# Set up logging
LOG = logging.getLogger("ms_transform")
logging.basicConfig(level=logging.INFO)


def do_ms_transform(transform_data: List[str]) -> None:
    """
    Transforms and regrids a Measurement Set (MS) based on the input data list.

    Parameters
    ----------
    transform_data : list
        List containing:
        [ms_in_path, outfile_ms, spw_range, output_directory, outfile_name_tar,
         base_name, year, freq_start, freq_end]

    Returns
    -------
    list
        The same list as input (transform_data), unchanged.
    """

    (
        ms_in, outfile_ms, spw_range, output_directory,
        outfile_name_tar, base_name, year, freq_start, freq_end
    ) = transform_data

    for suffix in ["", ".tmp", ".tar"]:
        path = f"{outfile_ms}{suffix}"
        if os.path.exists(path):
            LOG.info(f"Removing: {path}")
            remove_file_or_directory(path)

    LOG.info(f"ms_in: {ms_in}")
    LOG.info(f"outfile_ms: {outfile_ms}")
    LOG.info(f"spw_range: {spw_range}")

    if len(spw_range.split(",")) == 1:
        # Single spectral window
        mstransform(
            vis=ms_in,
            outputvis=outfile_ms,
            regridms=True,
            restfreq="1420.405752MHz",
            mode="channel",
            outframe="TOPO",
            interpolation="linear",
            veltype="radio",
            width=1,
            spw=spw_range,
            combinespws=False,
            nspw=0,
            createmms=False,
            datacolumn="data",
            numsubms=1
        )
    else:
        # Multi-SPW
        tmp_spws = [entry.split(":")[0] for entry in spw_range.split(",")]
        outfile_tmp = outfile_ms.replace(".ms", ".ms.tmp")
        LOG.info(f"outfile_tmp: {outfile_tmp}")

        if os.path.exists(outfile_tmp):
            shutil.rmtree(outfile_tmp)

        # Step 1: combine SPWs
        mstransform(
            vis=ms_in,
            outputvis=outfile_tmp,
            regridms=True,
            restfreq="1420.405752MHz",
            mode="channel",
            outframe="TOPO",
            interpolation="linear",
            veltype="radio",
            width=1,
            spw=",".join(tmp_spws),
            combinespws=True,
            nspw=0,
            createmms=False,
            datacolumn="data",
            numsubms=1
        )

        # Step 2: extract channels from combined SPW
        tmp1_start = int(spw_range.split("~")[0].split(":")[1])
        nchans_per_spw = int(spw_range.split(",")[0].split("~")[1]) - tmp1_start
        total_nchans = nchans_per_spw * len(tmp_spws)
        spw_final = f"*:{tmp1_start}~{tmp1_start + total_nchans}"

        mstransform(
            vis=outfile_tmp,
            outputvis=outfile_ms,
            regridms=True,
            restfreq="1420.405752MHz",
            mode="channel",
            outframe="TOPO",
            interpolation="linear",
            veltype="radio",
            width=1,
            spw=spw_final,
            combinespws=False,
            nspw=0,
            createmms=False,
            datacolumn="data",
        )

        if os.path.exists(outfile_tmp):
            shutil.rmtree(outfile_tmp)

    # Log SPW info
    ms_ = ms()
    ms_.open(thems=outfile_ms)
    LOG.info(
        f"Created File: {outfile_ms}\n"
        f"Spectral Window Range: {spw_range}\n"
        f"Spectral Window Info: {pformat(ms_.getspectralwindowinfo(), indent=2)}\n"
    )
    ms_.close()

    # Archive output
    create_tar_file(outfile_ms, suffix="tmp")
    os.rename(f"{outfile_ms}.tar.tmp", f"{outfile_ms}.tar")
    LOG.info(f"Created final file {outfile_ms}.tar from {outfile_ms}.tar.tmp")

def clean_transform_data(args: list[str]) -> list[str]:
    """
    Clean transform arguments by:
    - Stripping leading/trailing whitespace
    - Removing unexpected brackets and commas
    - Ensuring all elements are clean strings
    """
    cleaned = []

    for i, arg in enumerate(args):
        # Remove leading [ on the first arg
        if i == 0:
            arg = arg.lstrip("[").strip()
        # Remove trailing ] on the last arg
        if i == len(args) - 1:
            arg = arg.rstrip("]").strip()

        # Remove trailing commas and whitespace
        cleaned.append(arg.strip().rstrip(","))

    return cleaned



def main(transform_data: list) -> None:
    """
    Main entry point to perform MS transform and insert metadata into the database.

    Parameters
    ----------
    transform_data : list
        A list of 9 elements containing:
        [ms_in, outfile_ms, spw_range, output_directory, outfile_name_tar,
         base_name, year, freq_start, freq_end]
    """
    LOG.info(f"transform_data: {transform_data}")
    do_ms_transform(transform_data)
    #NewChiliesSplit.insert_metadata_from_transform(transform_data)


if __name__ == "__main__":
    if len(sys.argv) != 10:
        print("Usage: python run_ms_transform.py <ms_in> <ms_out> <spw_range> <out_dir> <tar_name> <base_name> <year> <freq_start> <freq_end>", file=sys.stderr)
        sys.exit(1)

    try:
        raw_args = sys.argv[1:]
        transform_data = clean_transform_data(raw_args)
        main(transform_data)
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}), file=sys.stderr)
        sys.exit(1)






