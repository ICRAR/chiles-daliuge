#!/usr/bin/env python3
from typing import List
from pprint import pformat
import sys
from chiles_daliuge.common import *
import json
import logging

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
        ms_in_path, base_name, outfile_path, outfile_tar_path, year, freq_start, freq_end
    ) = transform_data

    CHANNEL_WIDTH = 15625.0  # Hz

    LOG.info(f"Working on: {outfile_path} with freq {freq_start} and {freq_end}.")

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
        return

    if len(spw_range):
        for suffix in ["", ".tmp", ".tar"]:
            path = f"{outfile_path}{suffix}"
            if os.path.exists(path):
                LOG.info(f"Removing: {path}")
                remove_file_or_directory(path, trigger=True)

        LOG.info(f"ms_in: {ms_in_path}")
        LOG.info(f"outfile_ms: {outfile_path}")
        LOG.info(f"spw_range: {spw_range}")

        if len(spw_range.split(",")) == 1:
            # Single spectral window
            mstransform(
                vis=ms_in_path,
                outputvis=outfile_path,
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
            outfile_tmp = outfile_path.replace(".ms", ".ms.tmp")
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
                outputvis=outfile_path,
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
        ms_.open(thems=outfile_path)
        LOG.info(
            f"Created File: {outfile_path}\n"
            f"Spectral Window Range: {spw_range}\n"
            f"Spectral Window Info: {pformat(ms_.getspectralwindowinfo(), indent=2)}\n"
        )
        ms_.close()

        # Archive output
        create_tar_file(outfile_path, suffix="tmp")
        os.rename(f"{outfile_tar_path}.tmp", f"{outfile_tar_path}")
        LOG.info(f"Created final file {outfile_tar_path} from {outfile_tar_path}.tmp")
    else:
        LOG.warning("*********\nmstransform spw out of range:\n***********")

    return



def main(transform_data: list) -> None:
    """
    Main entry point to perform MS transform and insert metadata into the database.

    Parameters
    ----------
    transform_data : list
        A list of 7 elements containing:
        [ms_in_path, base_name, outfile_path, outfile_tar_path, year, freq_start, freq_end]
    """
    LOG.info(f"transform_data: {transform_data}")
    do_ms_transform(transform_data)
    # NewChiliesSplit.insert_metadata_from_transform(transform_data)


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(
            "Usage: python run_ms_transform.py <ms_in_path> <base_name> <outfile_path> <outfile_tar_path> <year> <freq_start> <freq_end>",
            file=sys.stderr)
        sys.exit(1)

    try:
        raw_args = sys.argv[1:]
        transform_data = destringify_data(raw_args)
        main(transform_data)
    except Exception as e:
        LOG.error(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)
