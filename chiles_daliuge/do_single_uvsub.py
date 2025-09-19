#!/usr/bin/env python3
import sys
from os.path import join
from chiles_daliuge.common import *
import json
import tempfile
import logging
from os import makedirs, rename
import pylab as pl
# CASA imports
from casaplotms import plotms
from casatasks import uvsub, statwt, split, phaseshift
from casatools import imager, ms, table, quanta, image
from casatools import msmetadata as _msmetadata
from typing import List, Tuple, Union

try:
    from casatasks import plotms  # preferred if available
except ImportError:
    # Some CASA builds ship plotms in a separate module
    from casaplotms import plotms

# Instantiate the msmetadata tool
msmd = _msmetadata()

# Set up logging
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



def time_convert(mytime: Union[float, int, str, List[Union[float, int, str]]],
                 myunit: str = "s") -> List[str]:
    """
    Convert one or more time values into human-readable date-time strings using CASA's quanta module.

    Parameters
    ----------
    mytime : float, int, str, or list of float/int/str
        A single time value or a list of time values to convert. Each value represents time
        in the specified unit (e.g., seconds, days).
    myunit : str, optional
        The unit of the input time value(s), such as "s" (seconds), "d" (days), "Hz", etc.
        Default is "s".

    Returns
    -------
    list of str
        A list of converted time strings in the format "YYYY/MM/DD/HH:MM:SS".

    Notes
    -----
    - CASA's `quanta` tool is used for conversion.
    - If a single time value is provided, the result will still be a list of one string.
    - Input values are internally wrapped into a list if not already one.
    """
    if type(mytime).__name__ != "list":
        mytime = [mytime]
    my_timestr = []
    for time in mytime:
        qa = quanta()
        q1 = qa.quantity(time, myunit)
        time1 = qa.time(q1, form="ymd")
        my_timestr.append(time1)
    return my_timestr


def fd2radec(fd):
    """Given a field direction return the string that is good for FIXVIS and friends"""
    qa = quanta()
    ra = qa.time(fd["m0"])[0]
    dec = qa.angle(fd["m1"])[0]
    dec = dec.split(".")
    ra = ra.split(":")
    ra = [int(ra[0]), int(ra[1]), int(ra[2])]
    dec = [int(dec[0]), int(dec[1]), int(dec[2]), np.mod(float(dec[2]), 1)]
    return "J2000 %02dh%02dm%06.4f %02dd%02dm%02d%0.4f" % (
        ra[0],
        ra[1],
        ra[2],
        dec[0],
        dec[1],
        dec[2],
        dec[3],
    )


def rejig_paths(taylor_terms: List[str],
                outliers: List[str],
                sky_model_location: str,
                spectral_window: int) -> Tuple[List[str], List[str]]:
    """
    Construct full file paths for Taylor term and outlier model templates using a given spectral window.

    This function formats file path templates for Taylor terms and outliers with the provided
    spectral window index, and joins them with the base sky model directory. It also handles
    normalization to avoid duplicate "LSM/" prefixes when `sky_model_location` already ends with "LSM".

    Parameters
    ----------
    taylor_terms : list of str
        List of file path templates for Taylor term models, containing a `{}` placeholder
        for the spectral window (e.g., "LSM/si_spw_{}.model.tt0").
    outliers : list of str
        List of file path templates for outlier models, also containing a `{}` placeholder
        (e.g., "LSM/Outliers/Outlier_{}.model").
    sky_model_location : str
        Base directory where the sky models are stored.
    spectral_window : int
        Spectral window index to format into the file path templates.

    Returns
    -------
    tuple of (list of str, list of str)
        Full file paths for the Taylor term models and outlier models, respectively.

    Notes
    -----
    - If `sky_model_location` ends with "LSM", the function avoids prepending "LSM/" again.
    - Path formatting uses `str.format(spectral_window)` on each template string.
    """

    def normalize(path_template: str) -> str:
        # Remove leading "LSM/" if sky_model_location already ends with LSM
        if os.path.basename(os.path.normpath(sky_model_location)) == "LSM" and path_template.startswith("LSM/"):
            return path_template[len("LSM/"):]
        return path_template

    taylor_terms_ = [
        join(sky_model_location, normalize(t.format(spectral_window)))
        for t in taylor_terms
    ]
    outliers_ = [
        join(sky_model_location, normalize(o.format(spectral_window)))
        for o in outliers
    ]
    return taylor_terms_, outliers_


def do_single_uvsub(
        taylor_terms, outliers, channel_average, produce_qa, w_projection_planes,
        sky_model_location,
        tar_file_split, year, freq_st, freq_en, uv_sub_path, METADATA_DB
):

    """
    Performs UV subtraction on a single measurement set using in-beam and outlier models.

    This function:
    - Extracts and preprocesses a measurement set tarball.
    - Applies in-beam and/or outlier sky models to perform UV subtraction.
    - Generates QA plots if requested.
    - Handles phase shifting, HA-based model selection, model FT, subtraction, and final measurement set export.
    - Updates a metadata database to record the UV subtraction result.

    Parameters
    ----------
    taylor_terms : list of str
        File paths to Taylor term sky models for in-beam sources.
    outliers : list of str
        File paths to outlier sky models used for HA-based subtraction.
    channel_average : int
        Number of channels to average in the final output.
    produce_qa : bool
        Whether to generate QA plots using `plotms`.
    w_projection_planes : int
        Number of W-projection planes to use for FT modeling.
    source_dir : str
        Path to the directory containing input and output MS files.
    sky_model_location : str
        Directory containing untarred sky models.
    split_name : str
        Name of the input tarred measurement set file (e.g., `split_1234.ms.tar`).
    year : str
        Observation year used for bookkeeping and naming.
    freq_st : str
        Start frequency of the sub-band.
    freq_en : str
        End frequency of the sub-band.
    uvsub_name : str
        Base name for the output MS and tar file after UV subtraction.
    METADATA_DB : str
        Path to the SQLite metadata database to be updated after processing.

    Returns
    -------
    None
        All outputs are written to disk and the metadata DB is updated in-place.
        Final result is a `.tar` archive of the UV-subtracted measurement set.

    Notes
    -----
    - HA-based model selection is based on estimated Hour Angle ranges.
    - Phase shifting is used to align outlier models with the target MS phase center.
    - Temporary directories are used for intermediate files and cleaned up after execution.
    - The function also flags baselines with nearly zero `u` values to reduce contamination.
    - QA plots are written to `qa_pngs/` within the output MS directory if `produce_qa=True`.
    """
    _, split_name = os.path.split(tar_file_split)
    save_dir, uvsub_name = os.path.split(uv_sub_path)

    with tempfile.TemporaryDirectory(
            dir=save_dir, prefix=f"__{split_name}__TEMP__"
    ) as temporary_directory:
        freq_start = int(freq_st)
        freq_end = int(freq_en)

        uv_sub_tar = f"{uv_sub_path}.tar"

        LOG.info(f"Untarring file: {tar_file_split}")
        untar_file(str(tar_file_split), temporary_directory)
        in_ms = join(temporary_directory, basename(tar_file_split)[:-4])

        spectral_window = int(((freq_start + freq_end) / 2 - 946) / 32)

        taylor_terms, outliers = rejig_paths(
            taylor_terms, outliers, sky_model_location, spectral_window
        )

        out_ms = basename(uv_sub_path)

        out_rot_data = "no"  # Keep a version of the subtracted data
        sub_uzero = True  # False #or True
        calc_stats = False  # or True
        pre_average = channel_average  # average in final split
        if pre_average <= 0 | pre_average > 2:
            pre_average = 2
            LOG.info(f'Resetting PRE_AVERAGE from {str(channel_average)} to {pre_average}')

        # Temporary names
        tmp_name = join(temporary_directory, f"{out_ms}.tmp")
        tmp_name1 = f"{tmp_name}.0"
        tmp_name2 = f"{tmp_name}.1"

        # if produce_qa:


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
                    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
                    png_directory = uv_sub_path + "_qa_pngs"
                    os.makedirs(png_directory, exist_ok=True)

                    # --- Helper: pick a safe SPW + central channel range to allow avgbaseline later ---
                    def central_spw_slice(ms_path: str):
                        """
                        Return a safe central SPW:channel selection string like 'spw:lo~hi',
                        or ('', None) if unavailable.
                        """
                        try:
                            msmd.open(ms_path)
                            # Prefer first field’s SPWs; fall back to first observation’s SPWs
                            fields = msmd.fields()
                            spws = msmd.spwsforfield(fields[0]) if fields else msmd.spwsforobs(0)
                            if not len(spws):
                                return "", None
                            spw = int(spws[0])
                            nchan = int(msmd.nchan(spw))
                            msmd.close()

                            if nchan <= 0:
                                return "", None

                            lo = int(0.25 * nchan)
                            hi = max(lo + 16, int(0.75 * nchan) - 1)
                            return f"{spw}:{lo}~{hi}", (spw, lo, hi)

                        except Exception as e:
                            try:
                                msmd.close()
                            except Exception:
                                pass
                            # Log and fall back to full band
                            LOG.warning(f"[msmd] metadata fallback due to: {e}")
                            return "", None


                    def _safe_plotms(ms_path: str, ycol: str, xcol: str, suffix: str, yaxis: str = "real", spw: str = ""):
                        out_png = join(png_directory, f"{basename(ms_path)}_{suffix}.png")
                        configs = [
                            # 1) Medium detail but safe: longer time avg, milder channel avg (better S/N across band)
                            dict(averagedata=True, avgtime="900s", avgchannel="4", avgbaseline=False, spw=spw),
                            # 2) Heavier channel avg, same time (good for noisier data)
                            dict(averagedata=True, avgtime="900s", avgchannel="8", avgbaseline=False, spw=spw),
                            # 3) Shorter time avg, no channel avg (preserve more spectral detail)
                            dict(averagedata=True, avgtime="120s", avgchannel="0", avgbaseline=False, spw=spw),
                        ]
                        last_ret = None
                        for i, cfg in enumerate(configs, 1):
                            try:
                                LOG.info(f"[plotms] attempt {i} for {suffix} with cfg={cfg}")
                                last_ret = plotms(
                                    vis=ms_path,
                                    xaxis="freq", yaxis=yaxis,
                                    xdatacolumn=xcol, ydatacolumn=ycol,
                                    transform=False,
                                    showgui=False, clearplots=True,
                                    plotfile=out_png, expformat="png",
                                    highres=True, dpi=150, overwrite=True,
                                    verbose=True,
                                    **cfg
                                )
                                if last_ret not in (False, None):
                                    break
                            except Exception as e:
                                LOG.exception(f"[plotms] failed attempt {i} for {suffix}: {e}")
                        return last_ret

                    LOG.info("Starting QA plot generation (enriched, still-safe).")

                    # A) Full band (no avgbaseline), but with richer averaging than before
                    ret_d_real = _safe_plotms(in_ms, "data",      "data",      "qa_data_real")
                    ret_m_real = _safe_plotms(in_ms, "model",     "model",     "qa_model_real")
                    ret_c_real = _safe_plotms(in_ms, "corrected", "corrected", "qa_corrected_real")

                    # Also amplitude & phase (super useful to spot calibration/subtraction issues)
                    ret_d_amp  = _safe_plotms(in_ms, "data",      "data",      "qa_data_amp",  yaxis="amp")
                    ret_c_amp  = _safe_plotms(in_ms, "corrected", "corrected", "qa_corrected_amp", yaxis="amp")
                    ret_d_phs  = _safe_plotms(in_ms, "data",      "data",      "qa_data_phase", yaxis="phase")
                    ret_c_phs  = _safe_plotms(in_ms, "corrected", "corrected", "qa_corrected_phase", yaxis="phase")

                    # B) NOW, try baseline-averaging but only on a central SPW slice (keeps it tractable)
                    spw_slice, meta = central_spw_slice(in_ms)
                    if spw_slice:
                        def _plot_baseline_avg(ms_path: str, ycol: str, xcol: str, suffix: str):
                            out_png = join(png_directory, f"{basename(ms_path)}_{suffix}.png")
                            try:
                                LOG.info(f"[plotms] baseline-avg on slice spw='{spw_slice}' → {suffix}")
                                return plotms(
                                    vis=ms_path,
                                    xaxis="freq", yaxis="amp",
                                    xdatacolumn=xcol, ydatacolumn=ycol,
                                    transform=False, showgui=False, clearplots=True,
                                    plotfile=out_png, expformat="png",
                                    highres=True, dpi=150, overwrite=True, verbose=True,
                                    averagedata=True, avgtime="1800s", avgchannel="16",
                                    avgbaseline=True, spw=spw_slice
                                )
                            except Exception as e:
                                LOG.exception(f"[plotms] avgbaseline slice failed for {suffix}: {e}")
                                return None

                        ret_d_base = _plot_baseline_avg(in_ms, "data",      "data",      "qa_data_amp_baselineAvg_slice")
                        ret_c_base = _plot_baseline_avg(in_ms, "corrected", "corrected", "qa_corrected_amp_baselineAvg_slice")
                    else:
                        ret_d_base = ret_c_base = None
                        LOG.info("[plotms] Skipping baseline-avg slice (no SPW/channel info).")

                    LOG.info(
                        "QA results: "
                        f"d_real={ret_d_real}, m_real={ret_m_real}, c_real={ret_c_real}, "
                        f"d_amp={ret_d_amp}, c_amp={ret_c_amp}, d_phs={ret_d_phs}, c_phs={ret_c_phs}, "
                        f"d_base={ret_d_base}, c_base={ret_c_base}"
                    )


                # if produce_qa:
                #     png_directory = uv_sub_path + "_qa_pngs_A"  # changed from uv_sub_path to tmp_name
                #     if not exists(png_directory):
                #         makedirs(png_directory)
                #     LOG.info(f"Starting QA plot generation A.")
                #     ret_d = plotms(
                #         vis=in_ms,
                #         xaxis="freq",
                #         yaxis="real",
                #         avgtime="43200",
                #         overwrite=True,
                #         avgbaseline=True,
                #         showgui=False,
                #         ydatacolumn="data",
                #         xdatacolumn="data",
                #         plotfile=join(png_directory, in_ms.rsplit("/")[-1] + "_infield_subtraction_data.png")
                #     )
                #     ret_m = plotms(
                #         vis=in_ms,
                #         xaxis="freq",
                #         yaxis="real",
                #         avgtime="43200",
                #         overwrite=True,
                #         avgbaseline=True,
                #         showgui=False,
                #         ydatacolumn="model",
                #         xdatacolumn="model",
                #         plotfile=join(
                #             png_directory,
                #             + in_ms.rsplit("/")[-1]
                #             + "_infield_subtraction_model.png"
                #         )
                #     )
                #     ret_c = plotms(
                #         vis=in_ms,
                #         xaxis="freq",
                #         yaxis="real",
                #         avgtime="43200",
                #         overwrite=True,
                #         avgbaseline=True,
                #         showgui=False,
                #         ydatacolumn="corrected",
                #         xdatacolumn="corrected",
                #         plotfile=join(png_directory,in_ms.rsplit("/")[-1] + "_infield_subtraction_corrected.png")
                #     )
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
                            LOG.info(f"and to end at {date_end} sample no. {ptr}")
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
                    png_directory = uv_sub_path + "_qa_pngs_B"  # changed from uv_sub_path to tmp_name
                    if not exists(png_directory):
                        makedirs(png_directory)
                    LOG.info(f"Starting QA plot generation B.")
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
                        plotfile=join(png_directory,in_ms.rsplit("/")[-1] + "_outfield_subtraction_data.png")
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
                        plotfile=join(png_directory,in_ms.rsplit("/")[-1] + "_outfield_subtraction_model.png")
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
                        plotfile=join(png_directory,in_ms.rsplit("/")[-1] + "_outfield_subtraction_corrected.png")
                    )
                    if not (ret_d & ret_c & ret_m):
                        LOG.info(
                            f"Reporting Outlier PlotMS Failure! State for Data, Corrected and Model is: {ret_d}&{ret_c}&{ret_m}"
                        )
                # Could be a copy
                split(
                    vis=tmp_name,
                    outputvis=uv_sub_path,
                    datacolumn="data",
                    width=pre_average,
                )
            else:
                split(
                    vis=in_ms,
                    outputvis=uv_sub_path,
                    datacolumn="corrected",
                    width=pre_average,
                )
            tmp_name = join(temporary_directory, f"{out_ms}.tmp")
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
                        f"Flagging {len(I)} baselines on {uv_sub_path}, on which u is ~zero"
                    )
                    fg[I] = True
                    tb.putcol("FLAG", fg.T)
                tb.close()

            # Add stat wt calculation
            if calc_stats:
                statwt(vis=tmp_name, chanbin=1, timebin="64s", datacolumn="data")
                if produce_qa:
                    png_directory = uv_sub_path + "_qa_pngs_C"  # changed from uv_sub_path to tmp_name
                    if not exists(png_directory):
                        makedirs(png_directory)
                    LOG.info(f"Starting QA plot generation C.")
                    ret_d = plotms(
                        vis=tmp_name,
                        xaxis="Frequency",
                        yaxis="WtSp",
                        avgtime="43200",
                        overwrite=True,
                        showgui=False,
                        ydatacolumn="data",
                        xdatacolumn="data",
                        plotfile=png_directory + "/" + in_ms.rsplit("/")[-1] + "_weight.png")
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
                    pl.savefig(join(png_directory, in_ms.rsplit("/")[-1] + "_lg_weight.png", ))

        except Exception:
            LOG.exception("*********\nUVSub exception: \n***********")

        # Clean up the temporary files
        # tmp_name = join(temporary_directory, f"{out_ms}.tmp")
        # tmp_name1 = f"{tmp_name}.0"
        # tmp_name2 = f"{tmp_name}.1"

        from pathlib import Path
        p = Path(uv_sub_path)

        # --- Preflight checks ---
        if not p.exists():
            LOG.error(f"Path does not exist, skipping: {p}")
        else:
            if p.is_dir():
                # Readability + non-empty dir check
                if not os.access(p, os.R_OK):
                    LOG.error(f"Directory not readable: {p}")
                else:
                    try:
                        it = p.iterdir()
                        first = next(it, None)
                        if first is None:
                            LOG.warning(f"Directory is empty, skipping: {p}")
                        else:
                            LOG.info(f"Tarring file: {uv_sub_path}")
                            create_tar_file(uv_sub_path, suffix="temp")

                            # Clean up the measurement sets
                            if exists(uv_sub_tar):
                                LOG.info(f"Removing {uv_sub_tar}")
                                remove_file_directory(uv_sub_tar)

                            rename(
                                f"{uv_sub_tar}.temp",
                                uv_sub_tar,
                            )

                            if exists(tmp_name):
                                remove_file_directory(tmp_name)

                            if exists(tmp_name1):
                                remove_file_directory(tmp_name1)

                            if exists(tmp_name2):
                                remove_file_directory(tmp_name2)

                            update_metadata_column(METADATA_DB, "ms_path", tar_file_split, year, freq_st, freq_en, "uv_sub_path", uv_sub_tar)

                    except Exception as e:
                        LOG.exception(f"Could not iterate directory {p}: {e}")

    LOG.info("Finished uvsub")


def main(uvsub_data: list) -> None:
    LOG.info(f"uvsub_data (parsed): {uvsub_data}")
    do_single_uvsub(*uvsub_data)


if __name__ == "__main__":
    try:
        raw_args = sys.argv[1:]
        LOG.info(f"raw_args before destringifying: {raw_args}")

        uvsub_data = destringify_data_uvsub(raw_args)  # This should already return final list
        LOG.info(f"uvsub_data going to function: {uvsub_data}")

        main(uvsub_data)

    except Exception as e:
        LOG.exception("Failed to run uvsub job")
        print(json.dumps({"status": "error", "message": str(e)}), file=sys.stderr)
        sys.exit(1)
