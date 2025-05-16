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

def rejig_paths(taylor_terms: List[str],
                outliers: List[str],
                sky_model_location: str,
                spectral_window: int) -> Tuple[List[str], List[str]]:
    """
    Construct full file paths for Taylor term and outlier models using the given spectral window.

    Parameters
    ----------
    taylor_terms : list of str
        List of template paths for Taylor term models, each containing a `{0}` placeholder
        for the spectral window number.
        Example: ["LSM/epoch1gt4k_si_spw_{0}.model.tt0", ...]
    outliers : list of str
        List of template paths for outlier models, each containing a `{0}` placeholder
        for the spectral window number.
        Example: ["LSM/Outliers/Outlier_1.0,8.spw_{0}.model", ...]
    sky_model_location : str
        Path to the sky model base directory (e.g., a tar-extracted folder path or root directory).
    spectral_window : int
        The spectral window ID used to format the `{0}` placeholders in the file paths.

    Returns
    -------
    tuple of (list of str, list of str)
        Two lists containing the full file paths to the Taylor term and outlier models, respectively.

    Example
    -------
    >>> rejig_paths(
            ["LSM/epoch1gt4k_si_spw_{0}.model.tt0"],
            ["LSM/Outliers/Outlier_1.0,8.spw_{0}.model"],
            "/tmp/sky_model_untar",
            3
        )
    (["/tmp/sky_model_untar/LSM/epoch1gt4k_si_spw_3.model.tt0"],
     ["/tmp/sky_model_untar/LSM/Outliers/Outlier_1.0,8.spw_3.model"])
    """
    taylor_terms_ = [
        join(sky_model_location, taylor_term.format(spectral_window))
        for taylor_term in taylor_terms
    ]
    outliers_ = [
        join(sky_model_location, outlier.format(spectral_window))
        for outlier in outliers
    ]
    return taylor_terms_, outliers_


def do_single_uvsub(
        taylor_terms, outliers, channel_average, produce_qa, w_projection_planes,
        data_dir, in_ms, out_measurement_set, sky_model_location, spectral_window,
        split_name, year, freq_st, freq_en, uv_sub_name, uv_sub_tar
):

    taylor_terms, outliers = rejig_paths(
        taylor_terms, outliers, sky_model_location, spectral_window
    )

    out_ms = basename(out_measurement_set)

    out_rot_data = "no"  # Keep a version of the subtracted data
    sub_uzero = True  # False #or True
    calc_stats = False  # or True
    pre_average = channel_average  # average in final split
    if pre_average<=0 | pre_average>2:
        pre_average=2
        LOG.info(f'Resetting PRE_AVERAGE from {str(channel_average)} to {pre_average}')

    # Temporary names
    tmp_name = join(data_dir, f"{out_ms}.tmp")
    tmp_name1 = f"{tmp_name}.0"
    tmp_name2 = f"{tmp_name}.1"

    png_directory = join(data_dir, "qa_pngs")
    if produce_qa:
        if not exists(png_directory):
            makedirs(png_directory)

    try:
        im = imager()
        # Create/Flush model_data column
        im.open(thems=in_ms, usescratch=True)

        # Select all data in this case
        im.selectvis()

        # These are the parameters for the generation of the model
        # Not sure how many of them are important here -- all except mode?
        # Increased nx/ny to match the new wider models
        im.defineimage(
            nx=8192, ny=8192, cellx="2arcsec", celly="2arcsec", mode="mfs", facets=1
        )
        im.setoptions(ftmachine="wproject", wprojplanes=w_projection_planes)

        # Find the reference frequency and set no. taylor terms
        ms_ = ms()
        ms_.open(in_ms)
        fq = ms_.getspectralwindowinfo()["0"]["RefFreq"]
        ms_.close()

        ntt = len(taylor_terms)
        if ntt > 0:  # In-beam models
            im.settaylorterms(ntaylorterms=ntt, reffreq=fq)

            #
            LOG.info(f"Models in this pass: {taylor_terms}")
            for mn in taylor_terms:
                tb = table()
                tb.open(mn)
                tb.clearlocks()
                tb.close()
            #
            im.ft(model=taylor_terms, incremental=False)
            im.close()

            # Now do the subtraction
            uvsub(vis=in_ms, reverse=False)
            if produce_qa:
                ret_d = plotms(
                    vis=in_ms,
                    xaxis="freq",
                    yaxis="real",
                    avgtime="43200",
                    overwrite=True,
                    avgbaseline=True,
                    showgui=False,
                    ydatacolumn="data",
                    xdatacolumn="data",
                    plotfile=join(
                        png_directory,
                        in_ms.rsplit("/")[-1]
                        + "_infield_subtraction_data.png",
                        ),
                )
                ret_m = plotms(
                    vis=in_ms,
                    xaxis="freq",
                    yaxis="real",
                    avgtime="43200",
                    overwrite=True,
                    avgbaseline=True,
                    showgui=False,
                    ydatacolumn="model",
                    xdatacolumn="model",
                    plotfile=join(
                        png_directory,
                        +in_ms.rsplit("/")[-1]
                        + "_infield_subtraction_model.png",
                        ),
                )
                ret_c = plotms(
                    vis=in_ms,
                    xaxis="freq",
                    yaxis="real",
                    avgtime="43200",
                    overwrite=True,
                    avgbaseline=True,
                    showgui=False,
                    ydatacolumn="corrected",
                    xdatacolumn="corrected",
                    plotfile=join(
                        png_directory,
                        in_ms.rsplit("/")[-1]
                        + "_infield_subtraction_corrected.png",
                        ),
                )
                if not (ret_d & ret_c & ret_m):
                    LOG.info(
                        f"Reporting In-field PlotMS Failure! State for Data, Corrected and Model is: {ret_d}&{ret_c}&{ret_m}"
                    )
        else:  # No in-beam models
            tmp_name = in_ms
            # os.path.join(out_dir, out_ms)
        # End ntt>0

        # Do we have outliers??
        if len(outliers) > 0:
            LOG.info(f"Using {len(outliers)} for outlier subtraction")
            ha_model = []
            for m in range(len(outliers)):
                ha_model.append(outliers[m].replace("Outliers/", "Outliers/HA_0/"))
            ms_ = ms()
            ms_.open(thems=in_ms)
            # Select data by HA in this case
            ret = ms_.getdata(["axis_info", "ha"], ifraxis=True)
            ms_phasecentre = fd2radec(ms_.getfielddirmeas())
            ms_.close()
            ha = ret["axis_info"]["time_axis"]["HA"] / 3600.0
            LOG.info("HA Range: " + str(ha[0]) + " to " + str(ha[-1]))
            ut = (
                    np.mod(ret["axis_info"]["time_axis"]["MJDseconds"] / 3600.0 / 24.0, 1)
                    * 24.0
            )
            not_first = False
            for nmodel in range(len(outliers)):  # (ntt,len(model)):
                if (ntt > 0) & (not_first == False):  # in-beam model done
                    # If the temp MS exists remove it
                    if exists(tmp_name):
                        remove_file_directory(tmp_name)
                    split(
                        vis=in_ms, outputvis=tmp_name, datacolumn="corrected"
                    )

                ia = image()
                LOG.info(f"Opening: {ha_model[nmodel]}")
                # if os.path.isfile(ha_model[nmodel]):
                #     LOG.warning(f"Deleting old tar.")
                #     os.remove("LSM.tar")
                ia.open(ha_model[nmodel])
                model_pc = ia.coordmeasures(ia.coordsys().referencepixel()["numeric"])[
                    "measure"
                ]["direction"]
                model_pc = fd2radec(model_pc)
                ia.close()
                if exists(tmp_name2):
                    remove_file_directory(tmp_name2)
                # if nmodel==(ntt): # First
                #    tmp_name1=tmp_name
                # Rotate to the direction of ha_model[nmodel]
                phaseshift(vis=tmp_name, outputvis=tmp_name2, phasecenter=model_pc)
                # tmp_name1='%s.%d'%(tmp_name,0)
                im = imager()
                im.open(thems=tmp_name2, usescratch=True)
                # delmod(otf=True,vis=tmp_name,scr=True)
                for m in range(-16, 16):
                    ptr1 = np.where(ha > (0.5 * m))[0]
                    ptr2 = np.where(ha > (0.5 * (m + 1)))[0]
                    LOG.info(
                        "No. samples in this HA range: " + str(len(ptr1) - len(ptr2))
                    )
                    if (len(ptr1) != 0) & (len(ptr1) != len(ptr2)):
                        LOG.info(f"Change Model to point to directory: HA_{m}")
                        ha_model[nmodel] = outliers[nmodel].replace(
                            f"Outliers", f"Outliers/HA_{m}"
                        )
                        ptr = ptr1[0]
                        LOG.info(
                            f"This HA ({m * 0.5}) will start at {ptr} and use the following adjusted model: {ha_model[nmodel]}"
                        )
                        if not isdir(ha_model[nmodel]):
                            LOG.info(
                                "Model does not exist! Continuing but residuals will be larger"
                            )
                            if m < -10:
                                ha_model[nmodel] = outliers[nmodel].replace(
                                    "Outliers", "Outliers/HA_-10"
                                )
                            if m > 7:
                                ha_model[nmodel] = outliers[nmodel].replace(
                                    "Outliers", "Outliers/HA_7"
                                )

                        # ut_start=ut[ptr]
                        date_start = time_convert(
                            ret["axis_info"]["time_axis"]["MJDseconds"][ptr]
                        )[0][0]
                        LOG.info(f"to start at {date_start}")
                        if len(ptr2) == 0:
                            ptr = -1
                        else:
                            ptr = ptr2[0]
                        # ut_end=ut[ptr]
                        date_end = time_convert(
                            ret["axis_info"]["time_axis"]["MJDseconds"][ptr]
                        )[0][0]
                        LOG.info(f"and to end at { date_end } sample no. {ptr}")
                        # if (ut_end<ut_start):
                        #    ut_end=ut_end+24
                        # timerange=str(ut_start)+'~'+str(ut_end)
                        timerange = f"{date_start}~{date_end}"
                        im.selectvis(time=timerange)
                        # These are the parameters for the generation of the model
                        # Not sure how many of them are important here -- all except mode?
                        im.defineimage(
                            nx=4096,
                            ny=4096,
                            cellx="2arcsec",
                            celly="2arcsec",
                            mode="mfs",
                            facets=1,
                        )
                        im.setoptions(
                            ftmachine="wproject",
                            wprojplanes=w_projection_planes,
                            freqinterp="linear",
                        )
                        im.settaylorterms(ntaylorterms=1)
                        #
                        LOG.info("Model in this pass: " + str(ha_model[nmodel]))
                        LOG.info("Time range in this pass: " + timerange)
                        tb = table()
                        tb.open(ha_model[nmodel])
                        tb.clearlocks()
                        tb.close()
                        #
                        im.ft(model=ha_model[nmodel], incremental=False)  # not_first)
                        not_first = True
                    # if samples in this HA range
                # next HA m
                im.close()
                uvsub(vis=tmp_name2, reverse=False)
                if exists(tmp_name):
                    remove_file_directory(tmp_name)
                if exists(tmp_name1):
                    remove_file_directory(tmp_name1)
                split(vis=tmp_name2, outputvis=tmp_name1, datacolumn="corrected")
                if out_rot_data == "yes":
                    split(
                        vis=tmp_name1,
                        datacolumn="data",
                        width=64,
                        timebin="30s",
                        outputvis=f"{png_directory}/O{nmodel}_{out_ms}",
                    )
                phaseshift(
                    vis=tmp_name1, outputvis=tmp_name, phasecenter=ms_phasecentre
                )
                # End of run through outlier models
            if produce_qa:
                ret_d = plotms(
                    vis=tmp_name1,
                    xaxis="freq",
                    yaxis="real",
                    avgtime="43200",
                    overwrite=True,
                    avgbaseline=True,
                    showgui=False,
                    ydatacolumn="data",
                    xdatacolumn="data",
                    plotfile=join(
                        png_directory,
                        in_ms.rsplit("/")[-1]
                        + "_outfield_subtraction_data.png",
                        ),
                )
                ret_m = plotms(
                    vis=tmp_name1,
                    xaxis="freq",
                    yaxis="real",
                    avgtime="43200",
                    overwrite=True,
                    avgbaseline=True,
                    showgui=False,
                    ydatacolumn="model",
                    xdatacolumn="model",
                    plotfile=join(
                        png_directory,
                        in_ms.rsplit("/")[-1]
                        + "_outfield_subtraction_model.png",
                        ),
                )
                ret_c = plotms(
                    vis=tmp_name1,
                    xaxis="freq",
                    yaxis="real",
                    avgtime="43200",
                    overwrite=True,
                    avgbaseline=True,
                    showgui=False,
                    ydatacolumn="corrected",
                    xdatacolumn="corrected",
                    plotfile=join(
                        png_directory,
                        in_ms.rsplit("/")[-1]
                        + "_outfield_subtraction_corrected.png",
                        ),
                )
                if not (ret_d & ret_c & ret_m):
                    LOG.info(
                        f"Reporting Outlier PlotMS Failure! State for Data, Corrected and Model is: {ret_d}&{ret_c}&{ret_m}"
                    )
            # Could be a copy
            split(
                vis=tmp_name,
                outputvis=join(data_dir, out_ms),
                datacolumn="data",
                width=pre_average,
            )
        else:
            split(
                vis=in_ms,
                outputvis=join(data_dir, out_ms),
                datacolumn="corrected",
                width=pre_average,
            )
        tmp_name = join(data_dir, out_ms)
        if sub_uzero:
            tb = table()
            tb.open(tmp_name, nomodify=False)
            tq = tb.query("", columns="UVW,FLAG")
            uv = tq.getcol("UVW")
            if tb.isvarcol("FLAG"):
                pass
                # fg_normal = tq.getvarcol("FLAG")
                # TODO: This is a hack to get around the fact that the flag column is a variable shaped array, and resulted in OOM, so needs fixing
            else:
                fg_normal = tq.getcol("FLAG")
                fg = fg_normal.T
                I = np.where(np.abs(uv[0]) < 50)[0]
                LOG.info(
                    f"Flagging {len(I)} baselines on {join(data_dir, out_ms)}, on which u is ~zero"
                )
                fg[I] = True
                tb.putcol("FLAG", fg.T)
            tb.close()

        # Add stat wt calculation
        if calc_stats:
            statwt(vis=tmp_name, chanbin=1, timebin="64s", datacolumn="data")
            if produce_qa:
                ret_d = plotms(
                    vis=tmp_name,
                    xaxis="Frequency",
                    yaxis="WtSp",
                    avgtime="43200",
                    overwrite=True,
                    showgui=False,
                    ydatacolumn="data",
                    xdatacolumn="data",
                    plotfile=png_directory
                             + "/"
                             + in_ms.rsplit("/")[-1]
                             + "_weight.png",
                )
                tb = table()
                tb.open(tmp_name)
                w = tb.getcol("WEIGHT_SPECTRUM")
                tb.close()
                pl.semilogy(np.nanmax(w, axis=(2)).T / 100)
                # pl.semilogy(f,np.min(w,axis=(2)))
                pl.semilogy(np.nanmedian(w, axis=(2)).T, ".")
                pl.title("Weights: " + in_ms.rsplit("/")[-1])
                pl.xlabel("Channel")  # Freq. (MHz)')
                pl.ylabel("Weight Sigma")
                pl.legend(["Max Weight/100", "Median Weight"])
                pl.savefig(
                    join(
                        png_directory,
                        in_ms.rsplit("/")[-1] + "_lg_weight.png",
                        )
                )

    except Exception:
        LOG.exception("*********\nUVSub exception: \n***********")

    # Clean up the temporary files
    tmp_name = join(data_dir, f"{out_ms}.tmp")
    if exists(tmp_name):
        remove_file_directory(tmp_name)

    tmp_name1 = f"{tmp_name}.0"
    if exists(tmp_name1):
        remove_file_directory(tmp_name1)

    tmp_name2 = f"{tmp_name}.1"
    if exists(tmp_name2):
        remove_file_directory(tmp_name2)

    LOG.info(f"Tarring file: {output_measurement_set}")
    create_tar_file(output_measurement_set, suffix="temp")

    # Clean up the measurement sets
    if exists(measurement_set):
        LOG.info(f"Removing {measurement_set}")
        remove_file_directory(measurement_set)

    rename(
        f"{output_measurement_set}.tar.temp",
        f"{output_measurement_set}.tar",
    )

    LOG.info("Finished uvsub")




def main(uvsub_data: list) -> None:
    LOG.info(f"uvsub_data: {uvsub_data}")
    # Now do_single_uvsub() can use this list
    # You may still want to parse taylor_terms/outliers from strings to actual lists here
    do_single_uvsub(uvsub_data)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python do_single_uvsub.py '<stringified_uvsub_list>'", file=sys.stderr)
        sys.exit(1)

    try:
        # Step 1: Convert the single string argument to a list
        parsed_list = ast.literal_eval(sys.argv[1])

        if not isinstance(parsed_list, list) or len(parsed_list) != 16:
            raise ValueError("Expected a stringified list of 16 elements")

        # Step 2: Clean the parsed elements
        uvsub_data = destringify_data(parsed_list)

        main(uvsub_data)

    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}), file=sys.stderr)
        sys.exit(1)







