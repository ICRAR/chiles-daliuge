#!/usr/bin/env python3
import sys
import json
import logging

from chiles_daliuge.common import *
import tempfile
from pathlib import Path
import shutil
import os
from os import makedirs
from os.path import join, exists, isfile

from casatasks import tclean, concat
from casatools import table, image, regionmanager

# Set up logging
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def prune_ms_dirs(root_path: str, keep_suffix: str = ".centre") -> None:
    """
    Scan `root_path` for **directories** and delete all except those whose
    names end with `keep_suffix`.

    Parameters
    ----------
    root_path : str
        Path to the directory you want to clean up.
    keep_suffix : str, optional
        Only directories ending with this suffix will be kept.
        Default is ".centre".
    """
    if not os.path.isdir(root_path):
        raise ValueError(f"{root_path!r} is not a valid directory")

    for name in os.listdir(root_path):
        full_path = os.path.join(root_path, name)
        if os.path.isdir(full_path):
            # If it doesn't end with the desired suffix, remove it
            if not name.endswith(keep_suffix):
                print(f"Removing directory: {name}")
                shutil.rmtree(full_path)
            else:
                print(f"Keeping directory:   {name}")

def do_tclean_implementation_all(
    out_ms,
    min_freq,
    max_freq,
    iterations,
    arcsec,
    w_projection_planes,
    clean_weighting_uv,
    robust,
    image_size,
    clean_channel_average,
    region_file,
    produce_qa,
    in_ms,
    out_dir,
    db_path
):
    """
    Perform the CLEAN step

    """
    max_freq = int(max_freq)
    min_freq = int(min_freq)
    clean_channel_average = float(clean_channel_average)
    iterations = int(iterations)
    w_projection_planes = int(w_projection_planes)
    robust = float(robust)
    image_size = int(image_size)

    if not exists(out_dir):
        makedirs(out_dir, exist_ok=True)

    delete_wtsp = True  ## Temp hack till weight spectrum makes sense

    if isinstance(in_ms, list):
        if len(in_ms) > 1:
            tmpdir_obj = tempfile.TemporaryDirectory(dir=out_dir, prefix="tclean_tmp_")
            temporary_directory = tmpdir_obj.name  # this is the actual path string

            if not Path(temporary_directory).exists():
                # normally unnecessary, since TemporaryDirectory already made it
                Path(temporary_directory).mkdir(parents=True, exist_ok=True)

            combine_file = join(out_dir, temporary_directory, out_ms)
        else:
            combine_file = in_ms[0]

    else:
        combine_file = in_ms

    outdir = join(out_dir, out_ms)
    outfile = join(out_dir, out_ms, out_ms)


    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM metadata WHERE tclean_all = ?", (outdir,))
    if cursor.fetchone():
        LOG.info(f"Skipping {outdir}, already in tclean_all of metadata DB.")
        return

    if exists(outdir):
        LOG.info(f"Removing old: {outdir}")
        shutil.rmtree(outdir)

    # If region_file is given check it exists. If not try some paths.
    # If still not set to blank to prevent failure
    if region_file and not isfile(region_file):
        LOG.error(f"Did not find file: {region_file}")
        region_file = ""
        LOG.error(f"Setting file to blank: {region_file}")


        # dump_all()
    if delete_wtsp:  # Temp hack till weight spectrum makes sense
        if isinstance(in_ms, list):
            for vis_file in in_ms:
                print(f"Removing weight spectrum from {vis_file}")
                tb = table()
                tb.open(vis_file, nomodify=False)
                if "WEIGHT_SPECTRUM" in tb.colnames():
                    tb.removecols("WEIGHT_SPECTRUM")
                    tb.removecols("SIGMA_SPECTRUM")
                tb.close()
        else:
            # else single ms file
            print(f"Removing weight spectrum from {in_ms}")
            tb = table()
            tb.open(in_ms, nomodify=False)
            if "WEIGHT_SPECTRUM" in tb.colnames():
                tb.removecols("WEIGHT_SPECTRUM")
                tb.removecols("SIGMA_SPECTRUM")
            tb.close()

    # Combine all the MS into a single MS
    if isinstance(in_ms, list) and len(in_ms) > 1:
        concat(vis=in_ms, concatvis=combine_file)
        LOG.info(f"concat(vis={in_ms}, concatvis={combine_file})")

    LOG.info(f"tclean(vis={combine_file}, imagename={outfile})")

    number_channels = round((max_freq - min_freq) / clean_channel_average)
    if clean_weighting_uv == "briggs":
        tclean(
            vis=combine_file,
            field="deepfield",
            spw="*",
            datacolumn="data",
            imagename=outfile,
            imsize=[image_size],
            cell=[arcsec],
            specmode="cube",
            # chanchunks=-1,
            nchan=number_channels,
            start=f"{min_freq}MHz",
            width=f"{clean_channel_average:.4f}MHz",
            mask=region_file,
            outframe="BARY",
            restfreq="1420.405752MHz",
            interpolation="linear",
            gridder="widefield",
            wprojplanes=w_projection_planes,
            pblimit=-0.2,
            deconvolver="clark",
            weighting="briggs",
            robust=robust,
            niter=iterations,
            gain=0.1,
            threshold="0.0mJy",
            savemodel="virtual",
            restart=False,  # TODO: Check if this is needed
            verbose=True,
        )  # Don't overwrite the model data col
    else:
        tclean(
            vis=combine_file,
            field="deepfield",
            spw="*",
            datacolumn="data",
            imagename=outfile,
            imsize=[image_size],
            cell=[arcsec],
            specmode="cube",
            # chanchunks=-1,
            nchan=number_channels,
            start=f"{min_freq}MHz",
            width=f"{clean_channel_average:.4f}MHz",
            mask=region_file,
            outframe="BARY",
            restfreq="1420.405752MHz",
            interpolation="linear",
            gridder="widefield",
            wprojplanes=w_projection_planes,
            pblimit=-0.2,
            deconvolver="clark",
            weighting=clean_weighting_uv,
            robust=robust,
            niter=iterations,
            gain=0.1,
            threshold="0.0mJy",
            savemodel="virtual",
        )  # Don't overwrite the model data col

    # Make a smaller verision of the image cube
    rg = regionmanager()
    if image_size > 2048:
        ia = image()
        ia.open(outfile + ".image")
        # box=rg.box([image_size/4,image_size*3/4],[image_size/4,image_size*3/4])
        # box=rg.box([1024,1024],[3072,3072])
        box = rg.box(
            [image_size / 2 - 600, image_size / 2 - 600],
            [image_size / 2 + 600, image_size / 2 + 600],
        )
        im2 = ia.subimage(outfile + ".image.centre", box, overwrite=True)
        im2.done()
        ia.close()

    # Make a smaller version of the PDF cube
    ia = image()
    ia.open(outfile + ".psf")
    box = rg.box(
        [image_size / 2 - 600, image_size / 2 - 600],
        [image_size / 2 + 600, image_size / 2 + 600],
    )
    im2 = ia.subimage(outfile + ".psf.centre", box, overwrite=True)
    im2.done()
    ia.close()

    if produce_qa == 'True':
        import numpy as np
        import matplotlib.pyplot as pl

        # IA used to report the statistics to the log file
        ia = image()
        ia.open(outfile + ".image")
        ia.statistics(verbose=True, axes=[0, 1])
        # IA used to make squashed images.
        # ia.moments(moments=[-1], outfile=outfile+'.image.mom.mean_freq')
        # ia.moments(moments=[-1], axis=0, outfile=outfile+'.image.mom.mean_ra')
        c = ia.collapse(
            function="mean",
            axes=3,
            outfile=outfile + ".image.mom.mean_freq",
            overwrite=True,
        )
        c.done()
        c = ia.collapse(
            function="mean",
            axes=0,
            outfile=outfile + ".image.mom.mean_ra",
            overwrite=True,
        )
        c.done()
        # IA used to make slices.
        smry = ia.summary()
        xpos = 2967.0 / 4096 * smry["shape"][0]
        ypos = 4095.0 / 4096 * smry["shape"][1]
        box = rg.box([xpos - 2, 0], [xpos + 2, ypos])
        c = ia.moments(
            moments=[-1],
            axis=0,
            region=box,
            outfile=outfile + "image.mom.slice_ra",
            overwrite=True,
        )
        c.done()
        #
        # We will get rid of this if the slice above works
        slce = []
        for m in range(smry["shape"][3]):
            slce.append(ia.getslice(x=[xpos, xpos], y=[0, ypos], coord=[0, 0, 0, m]))
        #
        # Print out text version
        with open(outfile + ".image.slice.txt", "w") as fo:
            for n in range(len(slce[0]["ypos"])):
                line = [slce[0]["ypos"][n]]
                for m in range(len(slce)):
                    line.append(slce[m]["pixel"][n])
                print(line, file=fo)

        for m in range(len(slce)):
            pl.plot(slce[m]["ypos"], slce[m]["pixel"] * 1e3)
        pl.xlabel("Declination (pixels)")
        pl.ylabel("Amplitude (mJy)")
        pl.title("Slice along sidelobe for " + outfile)
        pl.savefig(outfile + ".image.slice.svg")
        pl.clf()
        #
        # IA used to make profiles.
        # Source near centre profile
        xpos = 1992.0 / 4096 * smry["shape"][0]
        ypos = 2218.0 / 4096 * smry["shape"][1]
        box = rg.box([xpos - 2, ypos - 2], [xpos + 2, ypos + 2])
        slce = ia.getprofile(region=box, unit="MHz", function="mean", axis=3)
        with open(outfile + ".image.onsource_centre.txt", "w") as fo:
            for n in range(len(slce["coords"])):
                print(slce["coords"][n], slce["values"][n], file=fo)

        pl.plot(slce["coords"], slce["values"] * 1e3)
        pl.xlabel("Frequency (MHz)")
        pl.ylabel("Amplitude (mJy)")
        pl.title("Slice central source " + outfile)
        pl.savefig(outfile + ".image.onsource_centre.svg")
        pl.clf()
        # Source near edge profile
        xpos = 2972.0 / 4096 * smry["shape"][0]
        ypos = 155.0 / 4096 * smry["shape"][1]
        box = rg.box([xpos - 2, ypos - 2], [xpos + 2, ypos + 2])
        slce = ia.getprofile(region=box, unit="MHz", function="mean", axis=3)
        with open(outfile + ".image.onsource_south.txt", "w") as fo:
            for n in range(len(slce["coords"])):
                print(slce["coords"][n], slce["values"][n], file=fo)

        pl.plot(slce["coords"], slce["values"] * 1e3)
        pl.xlabel("Frequency (MHz)")
        pl.ylabel("Amplitude (mJy)")
        pl.title("Slice central source " + outfile)
        pl.savefig(outfile + ".image.onsource_south.svg")
        pl.clf()
        # Boresight profile
        box = rg.box(
            [image_size / 2 - 2, image_size / 2 - 2],
            [image_size / 2 + 2, image_size / 2 + 2],
        )
        slce = ia.getprofile(region=box, unit="MHz", function="mean", axis=3)
        with open(outfile + ".image.boresight.txt", "w") as fo:
            for n in range(len(slce["coords"])):
                print(slce["coords"][n], slce["values"][n], file=fo)

        pl.plot(slce["coords"], slce["values"] * 1e3)
        pl.xlabel("Frequency (MHz)")
        pl.ylabel("Amplitude (mJy)")
        pl.title("Slice central source " + outfile)
        pl.savefig(outfile + ".image.boresight.svg")
        pl.clf()
        # RMS Stats
        sts = ia.statistics(axes=[0, 1], verbose=False)
        with open(outfile + ".image.rms.txt", "w") as fo:
            for n in range(len(sts["rms"])):
                print(slce["coords"][n], sts["rms"][n], file=fo)

        pl.plot(slce["coords"], sts["rms"] * 1e3)
        pl.xlabel("Frequency (MHz)")
        pl.ylabel("RMS (mJy)")
        pl.title("RMS for " + outfile)
        pl.savefig(outfile + ".image.rms.svg")
        pl.clf()
        # Histograms
        sts = ia.histograms()
        with open(outfile + ".image.histo.txt", "w") as fo:
            for n in range(len(sts["values"])):
                print(sts["values"][n], sts["counts"][n], file=fo)

        pl.plot(sts["values"] * 1e3, np.log10(sts["counts"]))
        pl.xlabel("Bin (mJy)")
        pl.ylabel("log10 of No. of Values")
        pl.title("Histogram for " + outfile)
        pl.savefig(outfile + ".image.histo.svg")
        pl.clf()
        # Beams --- sometimes this information is missing, so only generate if what is needed is there
        sts = ia.summary()
        if "perplanebeams" in sts.keys():
            sts = sts["perplanebeams"]
            bmj = []
            bmn = []
            bmp = []
            with open(outfile + ".image.beam.txt", "w") as fo:
                for n in range(sts["nChannels"]):
                    print(slce["coords"][n], sts["beams"]["*" + str(n)]["*0"], file=fo)

            for n in range(sts["nChannels"]):
                bmj.append(sts["beams"]["*" + str(n)]["*0"]["major"]["value"])
                bmn.append(sts["beams"]["*" + str(n)]["*0"]["minor"]["value"])
                bmp.append(
                    sts["beams"]["*" + str(n)]["*0"]["positionangle"]["value"] / 57.3
                )
            pl.plot(slce["coords"], bmj)
            pl.plot(slce["coords"], bmn)
            pl.plot(slce["coords"], bmp)
            pl.xlabel("Frequency (MHz)")
            pl.ylabel("Beam Axes (major, minor & PA (rad)")
            pl.title("Beam Parameters for " + outfile)
            pl.savefig(outfile + ".image.beam.svg")
            pl.clf()
        ia.close()
    prune_ms_dirs(outdir)

    for ms_in in in_ms:
        update_metadata_column(db_path, "build_concat_all", ms_in, "*", str(min_freq), str(max_freq), "tclean_all", outdir)
    return True


def main():
    raw_args = sys.argv[1:]
    LOG.info(f"[RAW ARGS] {raw_args}")

    # we need at least: stringified‑13‑args, out_dir, db_path
    if len(raw_args) < 3:
        print(
            "Usage: python do_tclean_implementation_all.py "
            "<stringified_tclean_args> <out_dir> <db_path>",
            file=sys.stderr
        )
        sys.exit(1)

    # peel off the last two as paths
    out_dir = raw_args[-2]
    db_path = raw_args[-1]
    # everything else (one or more pieces) is our stringified block
    arg_str = " ".join(raw_args[:-2])

    LOG.info(f"[ARGS] out_dir: {out_dir}")
    LOG.info(f"[ARGS] db_path: {db_path}")
    LOG.info(f"[ARGS] arg_str: {arg_str}")


    # ensure database path
    if not os.path.isfile(db_path):
        LOG.error(f"Invalid db_path: {db_path}")
        sys.exit(1)

    # destringify the 13 tclean arguments
    parsed = destringify_data_tclean([arg_str])
    LOG.info(f"[PARSED] {parsed!r}")

    if len(parsed) != 13:
        LOG.error(f"Expected 13 tclean args after destringify, got {len(parsed)}")
        sys.exit(1)

    # build the final 15‑element list and dispatch
    tclean_args = parsed + [out_dir, db_path]
    LOG.info(f"[FINAL TCLEAN_ARGS] {tclean_args!r}")

    try:
        do_tclean_implementation_all(*tclean_args)
    except Exception:
        LOG.exception("tclean implementation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
