from os.path import isdir
import tempfile
from datetime import datetime, timedelta
from os import makedirs, rename, listdir
import numpy as np
import pylab as pl
from casaplotms import plotms
from casatasks import uvsub, statwt, split, phaseshift
from casatools import imager, ms, table, quanta, image
from typing import Union, List, Tuple
from pathlib import Path
from common import *

process_ms_flag = True


LOG = logging.getLogger(__name__)


db_dir = "/home/00103780/dlg/db"
METADATA_CSV = db_dir+"/Chilies_metadata.csv"
METADATA_DB = os.path.join(db_dir, "Chilies_metadata.db")




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





def copy_sky_model(sky_model_source: Union[str, bytes], temporary_directory: str) -> str:
    """
    Extract or copy the sky model to a temporary directory.

    Parameters
    ----------
    sky_model_source : str or bytes
        Path to the `.tar` archive or a directory containing the sky model.
        If a `.tar` file, it will be extracted.
        If a directory, it will be copied directly.
    temporary_directory : str
        Path to the directory where the sky model should be placed.

    Returns
    -------
    str
        Path to the temporary directory containing the sky model.

    Notes
    -----
    - If the input is a .tar file, assumes it is uncompressed.
    - For directories, performs a full recursive copy.
    """
    if isinstance(sky_model_source, str) and sky_model_source.endswith(".tar"):
        LOG.info(f"Untarring {sky_model_source} to {temporary_directory}")
        untar_file(sky_model_source, temporary_directory, gz=False)
    elif os.path.isdir(sky_model_source):
        LOG.info(f"Copying directory {sky_model_source} to {temporary_directory}")
        if os.path.exists(temporary_directory):
            shutil.rmtree(temporary_directory)
        shutil.copytree(sky_model_source, temporary_directory)
    else:
        raise ValueError(f"Invalid sky model input: {sky_model_source} is neither a .tar file nor a directory.")

    return temporary_directory


# def copy_region_files(region_file_tar_file, temporary_directory):
#     untar_file(region_file_tar_file, temporary_directory, gz=False)
#     return join(temporary_directory, "region-files")



def fetch_split_ms(
        year_list: List[str],
        frequencies: List[List[int]],
        db_path: str
) -> List[str]:
    """
    Fetch dlg_name values from the metadata DB where year is in year_list and
    [start_freq, end_freq] matches any pair in the frequencies list.

    Parameters
    ----------
    year_list : list of str
        List of acceptable year values.
    frequencies : list of [int, int]
        List of acceptable [start_freq, end_freq] pairs.
    db_path : str
        Path to the SQLite metadata database.

    Returns
    -------
    list of str
        Matching dlg_name values.
    """
    verify_db_integrity()

    freq_set = {tuple(freq_pair) for freq_pair in frequencies}

    query = """
        SELECT dlg_name, year, start_freq, end_freq FROM metadata
    """

    matching_dlg_names = []

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for row in cursor.execute(query):
            dlg_name, year, start_freq, end_freq = row
            # print(f"\nChecking row: dlg_name={dlg_name}, year={year}, start_freq={start_freq}, end_freq={end_freq}")

            freq_tuple = (int(start_freq), int(end_freq))  # ✅ convert to int

            # if year in year_list:
            #     print(f"  ✓ Year '{year}' is in year_list")
            # else:
            #     print(f"  ✗ Year '{year}' is NOT in year_list")

            # print(f"  → Frequency tuple: {freq_tuple}")
            # if freq_tuple in freq_set:
            #     print(f"  ✓ Frequency {freq_tuple} is in frequency list")
            # else:
            #     print(f"  ✗ Frequency {freq_tuple} is NOT in frequency list")

            if year in year_list and freq_tuple in freq_set:
                #print(f"  → Appending: {dlg_name}")
                matching_dlg_names.append(f"{dlg_name};{year};{freq_tuple[0]};{freq_tuple[1]}")



    return matching_dlg_names





def time_convert(mytime: Union[float, int, str, List[Union[float, int, str]]],
                 myunit: str = "s") -> List[str]:
    """
    Convert one or more time values into human-readable date strings using CASA's quanta module.

    Parameters
    ----------
    mytime : float, int, str, or list of float/int/str
        A single time value or a list of time values to convert. These represent time in the specified unit.
    myunit : str, optional
        The unit of the input time value(s), e.g., "s" for seconds, "d" for days, "Hz", etc.
        Default is "s".

    Returns
    -------
    list of str
        A list of converted date-time strings in "YYYY/MM/DD/HH:MM:SS" format.

    Notes
    -----
    This uses CASA's `quanta` tool to convert time values based on the given unit.
    Even if a single value is passed, the result will be a list of one string.
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
        data_dir, in_ms, out_measurement_set, sky_model_location, spectral_window
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



    LOG.info("Finished uvsub")

    return True



def remove_old_temporary_directories(output_directory, prefix):
    for name in listdir(output_directory):
        if name.startswith(prefix):
            path = join(output_directory, name)
            if isdir(path):
                LOG.info(f"Removing old temporary directory: {path}")
                shutil.rmtree(path)



def do_uvsub(names_list, source_dir, wall_time, sky_model_tar_file,
    taylor_terms, outliers, channel_average, produce_qa, w_projection_planes, process_files
):
    verify_db_integrity()
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=wall_time)

    sky_model_location = None

    add_column_if_missing("uv_sub_name")

    with tempfile.TemporaryDirectory(
            dir=source_dir, prefix="__SKY_TEMP__"
    ) as temporary_sky_model:
        if sky_model_location is None:
            sky_model_location = copy_sky_model(sky_model_tar_file, temporary_sky_model)

        conn = sqlite3.connect(METADATA_DB)


        for name in names_list:
            split_name, year, freq_st, freq_en = name.split(";")
            freq_start = int(freq_st)
            freq_end = int(freq_en)

            LOG.info("#" * 60)
            LOG.info("#" * 60)
            LOG.info(f"Processing: {name}")

            tar_file_split = join(
                source_dir, split_name
            )
            LOG.info(f"Processing: {tar_file_split}")


            spectral_window = int(((freq_start + freq_end) / 2 - 946) / 32)

            with tempfile.TemporaryDirectory(
                    dir=source_dir, prefix=f"__{split_name}__TEMP__"
            ) as temporary_directory:
                now = datetime.now()
                LOG.info(
                    f"Checking time left to run. now: {now} : end_time: {end_time} : start_time: {start_time}"
                )
                if now + timedelta(minutes=45) > end_time:
                    LOG.info("Stopping as we are close to the wall time")
                    break

                uv_sub_name = generate_hashed_ms_name(str(tar_file_split), year, str(freq_start), str(freq_end))


                output_measurement_set = join(
                    source_dir, uv_sub_name
                )
                #if exists(f"{output_measurement_set}.tar"):
                #    LOG.info(f"Skipping {output_measurement_set}")
                #    continue
                LOG.info("uv_sub_name:",uv_sub_name)
                uv_sub_tar = f"{uv_sub_name}.tar"
                LOG.info("uv_sub_tar:",uv_sub_tar)
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM metadata WHERE uv_sub_name = ?", (uv_sub_tar,))
                if cursor.fetchone():
                    LOG.info(f"Skipping {uv_sub_tar}, already in metadata DB.")
                    continue




                if process_files:
                    LOG.info(f"Untarring file: {tar_file_split}")
                    untar_file(str(tar_file_split), temporary_directory)
                    measurement_set = join(temporary_directory, basename(tar_file_split)[:-4])


                    if do_single_uvsub(
                        taylor_terms, outliers, channel_average, produce_qa, w_projection_planes,
                        source_dir, measurement_set, output_measurement_set, sky_model_location, spectral_window
                    ):

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

                else:
                    os.makedirs(os.path.dirname(output_measurement_set), exist_ok=True)
                    Path(f"{output_measurement_set}.tar").touch()
                    LOG.info(f"Created placeholder file: f{uv_sub_tar}")

                update_metadata_column(split_name, year, freq_st, freq_en, "uv_sub_name", uv_sub_tar)



    verify_db_integrity()
    LOG.info("All Done with uvsub!!!")

