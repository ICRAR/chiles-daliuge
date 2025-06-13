#!/usr/bin/env python3
import sys
import json
import logging
from pprint import pformat
from chiles_daliuge.common import destringify_data

import shutil
from os import makedirs
from os.path import join, exists, isfile

from casatasks import tclean, concat
from casatools import table, image, regionmanager

# Set up logging
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def do_tclean_implementation(
    cube_dir,
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
    temporary_directory,
    in_dirs,
):
    """
    Perform the CLEAN step

    """
    if not exists(cube_dir):
        makedirs(cube_dir, exist_ok=True)

    delete_wtsp = True  ## Temp hack till weight spectrum makes sense

    if isinstance(in_dirs, list):
        outfile = join(cube_dir, f"clean_{min_freq:04d}-{max_freq:04d}")
        combine_file = join(
            temporary_directory, f"com_{min_freq:04d}-{max_freq:04d}.ms"
        )

        if exists(outfile):
            LOG.info(f"Removing old: {outfile}")
            shutil.rmtree(outfile)
    else:
        outfile = join(cube_dir, "clean_all")
        combine_file = in_dirs

    # If region_file is given check it exists. If not try some paths.
    # If still not set to blank to prevent failure
    if region_file and not isfile(region_file):
        LOG.error(f"Did not find file: {region_file}")
        region_file = ""
        LOG.error(f"Setting file to blank: {region_file}")

    LOG.info(f"tclean(vis={in_dirs}, imagename={outfile})")
    try:
        # dump_all()
        if delete_wtsp:  # Temp hack till weight spectrum makes sense
            if isinstance(in_dirs, list):
                for vis_file in in_dirs:
                    print(f"Removing weight spectrum from {vis_file}")
                    tb = table()
                    tb.open(vis_file, nomodify=False)
                    if "WEIGHT_SPECTRUM" in tb.colnames():
                        tb.removecols("WEIGHT_SPECTRUM")
                        tb.removecols("SIGMA_SPECTRUM")
                    tb.close()
            else:
                # else single ms file
                print(f"Removing weight spectrum from {in_dirs}")
                tb = table()
                tb.open(in_dirs, nomodify=False)
                if "WEIGHT_SPECTRUM" in tb.colnames():
                    tb.removecols("WEIGHT_SPECTRUM")
                    tb.removecols("SIGMA_SPECTRUM")
                tb.close()

        # Combine all the MS into a single MS
        if isinstance(in_dirs, list):
            concat(vis=in_dirs, concatvis=combine_file)

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
                vis=in_dirs,
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
    except Exception:
        LOG.exception("*********\nClean exception: \n***********")
        return False

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

    if produce_qa == "yes":
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

    return True


def main(tclean_args):
    """
    Main entry point to run the tclean implementation.

    Parameters
    ----------
    tclean_args : list
        A list of 13 elements:
        [cube_dir, min_freq, max_freq, iterations, arcsec, w_projection_planes,
         clean_weighting_uv, robust, image_size, clean_channel_average,
         region_file, produce_qa, temporary_directory, in_dirs]
    """
    LOG.info("Starting CLEAN step with parameters:")
    LOG.info(pformat(tclean_args))

    do_tclean_implementation(*tclean_args)


if __name__ == "__main__":
    if len(sys.argv) != 15:
        print(
            "Usage:\n"
            "python run_tclean_implementation.py "
            "<cube_dir> <min_freq> <max_freq> <iterations> <arcsec> "
            "<w_projection_planes> <clean_weighting_uv> <robust> <image_size> "
            "<clean_channel_average> <region_file> <produce_qa> <temporary_directory> <in_dirs>\n\n"
            "Note: Arguments should be passed as strings; lists must be stringified properly.",
            file=sys.stderr
        )
        sys.exit(1)

    try:
        raw_args = sys.argv[1:]
        tclean_args = destringify_data(raw_args)
        main(tclean_args)
    except Exception as e:
        LOG.error(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)
