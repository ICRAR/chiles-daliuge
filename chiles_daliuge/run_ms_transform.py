#!/usr/bin/env python3
from typing import List
from pprint import pformat
import sys
from chiles_daliuge.common import *
import json
import logging
from os.path import join
# CASA imports
from casatasks import mstransform
from casatools import ms, imager

# Set up logging
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def do_ms_transform(transform_data: List[str]) -> None:
    """
    Transform and regrid a Measurement Set (MS) based on frequency bounds,
    producing a frequency-sliced and regridded MS using CASA's `mstransform`.

    The function performs:
    - Channel selection using `im.advisechansel` based on input frequency range.
    - Regridding and subsetting via `mstransform`, supporting both single and multi-SPW handling.
    - Cleanup of existing outputs.
    - Archiving of the final MS as a `.tar` file.
    - Logging of spectral window metadata.

    Parameters
    ----------
    transform_data : list of str
        List containing the following 6 elements:
        [ms_in_path, base_name, outfile_path, outfile_tar_path, year, freq_start, freq_end]

    Returns
    -------
    None

    Side Effects
    ------------
    - Creates a regridded MS file on disk.
    - Removes old output files with `.ms`, `.tmp`, and `.tar` extensions.
    - Writes logs with SPW information.
    - Generates a `.tar` archive of the final MS.
    """

    (
        ms_in_path, base_name, year, freq_start, freq_end, outfile_path, db_path
    ) = transform_data

    if outfile_path.endswith(".ms"): # should contain .ms by default as specified in dirdrop, if {auto}.ms works
        uv_split_dir = outfile_path
    else:
        uv_split_dir = f"{outfile_path}.ms"

    os.makedirs(uv_split_dir, exist_ok=True)

    CHANNEL_WIDTH = 15625.0  # Hz

    LOG.info(f"Working on: {uv_split_dir} with freq {freq_start} and {freq_end}.")

    im = imager()
    # LOG.info(f"ms_in_path: {ms_in}")
    im.selectvis(vis=ms_in_path)
    selinfo = im.advisechansel(
        freqstart=int(freq_start) * 1e6,
        freqend=int(freq_end) * 1e6,
        freqstep=CHANNEL_WIDTH,
        freqframe="BARY",
    )
    # LOG.info(f"advisechansel result: {selinfo}")
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
        spw_range = None

    if len(spw_range):
        for suffix in ["", ".tmp", ".tar"]:
            path = f"{uv_split_dir}{suffix}"
            if os.path.exists(path):
                LOG.info(f"Removing: {path}")
                remove_file_or_directory(path, trigger=True)

        LOG.info(f"ms_in: {ms_in_path}")
        LOG.info(f"outfile_ms: {uv_split_dir}")
        LOG.info(f"spw_range: {spw_range}")

        if len(spw_range.split(",")) == 1:
            # Single spectral window
            mstransform(
                vis=ms_in_path,
                outputvis=uv_split_dir,
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
            outfile_tmp = f"{uv_split_dir}.tmp"
            LOG.info(f"outfile_tmp: {outfile_tmp}")

            if os.path.exists(outfile_tmp):
                shutil.rmtree(outfile_tmp)

            # Step 1: combine SPWs
            mstransform(
                vis=ms_in_path,
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
                outputvis=uv_split_dir,
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

    else:
        LOG.warning("*********\nmstransform spw out of range:\n***********")

    # Final checks on MS split
    try:
        ms_ = ms()
        ms_.open(thems=uv_split_dir)
        LOG.info(
            f"Created File: {uv_split_dir}\n"
            f"Spectral Window Range: {spw_range}\n"
            f"Spectral Window Info: {pformat(ms_.getspectralwindowinfo(), indent=2)}\n"
        )
        ms_.close()

        create_tar_file(uv_split_dir, suffix="tmp")
        outfile_tar_path = f"{uv_split_dir}.tar"
        os.rename(f"{uv_split_dir}.tar.tmp", f"{outfile_tar_path}")
        LOG.info(f"Created final file {outfile_tar_path} from {outfile_tar_path}.tmp")

        size_bytes = os.path.getsize(outfile_tar_path) if os.path.exists(outfile_tar_path) else 0
        size = round(float(size_bytes / (1024 * 1024 * 1024)), 3)

    except Exception as ee:
        LOG.error(f"Invalid MS split {uv_split_dir}: {ee}")
        outfile_tar_path = f"{uv_split_dir}.tar"
        size = -1
        LOG.warning(f"{uv_split_dir} is not a valid MS. ")
        try:
            os.makedirs(os.path.dirname(outfile_tar_path), exist_ok=True)
            with open(outfile_tar_path, "w"):
                LOG.info(f"Created dummy {outfile_tar_path}")
                remove_file_or_directory(uv_split_dir, trigger=True)
                pass  # equivalent to touch
        except Exception as e:
            LOG.error(f"Failed to create dummy file at {outfile_tar_path}: {e}")
            remove_file_or_directory(uv_split_dir, trigger=True)

    bandwidth = int(freq_end) - int(freq_start)

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

    return



def main(transform_data: list) -> None:
    """
    Main entry point to perform MS transform and insert metadata into the database.

    Parameters
    ----------
    transform_data : list
        A list of 7 elements containing:
        [ms_in_path, base_name, year, freq_start, freq_end, outfile_path]
    """
    LOG.info(f"transform_data: {transform_data}")
    do_ms_transform(transform_data)
    # NewChiliesSplit.insert_metadata_from_transform(transform_data)


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(
            "Usage: python run_ms_transform.py <ms_in_path> <base_name> <year> <freq_start> <freq_end> <out_dir> <db_path>",
            file=sys.stderr)
        sys.exit(1)

    try:
        raw_args = sys.argv[1:-2]
        LOG.info(f"raw_args: {raw_args}")
        out_dir = sys.argv[-2]
        LOG.info(f"out_dir: {out_dir}")
        db_path = sys.argv[-1]
        LOG.info(f"db_path: {db_path}")
        transform_data = destringify_data(raw_args)
        transform_data.append(out_dir)
        transform_data.append(db_path)
        main(transform_data)
    except Exception as e:
        LOG.error(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)
