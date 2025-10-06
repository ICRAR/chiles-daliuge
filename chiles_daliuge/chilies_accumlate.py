#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import tarfile
import shutil
import sys
from pathlib import Path

import numpy as np
from casatasks import concat, split
from casatools import ms as ms_tool


def load_metadata(csv_file: Path):

    # Build a flexible dtype: keep text columns as 'U' (unicode), numbers as appropriate
    # We only actually need: year, start_freq, size, uv_sub_path (indices 2,3,6,7)
    dtype = np.dtype([
        ("ms_path", "U260"),
        ("base_name", "U260"),
        ("year", "U64"),
        ("start_freq", "i4"),
        ("end_freq", "i4"),
        ("bandwidth", "f8"),
        ("size", "f8"),
        ("uv_sub_path", "U260"),
        ("build_concat_all", "U260"),
        ("tclean_all", "U260"),
        ("build_concat_epoch", "U260"),
        ("tclean_epoch", "U260"),
    ])

    data = np.genfromtxt(
        csv_file,
        delimiter=",",
        names=True,
        dtype=dtype,
        encoding="utf-8",
    )

    # Ensure array is at least 1D even if one row
    if data.shape == ():
        data = np.array([data], dtype=dtype)

    return data


def find_uvsub_tars(uvsub_dir: Path):
    """
    Return list of .tar files (Path objects) sorted by name.
    """
    tars = sorted(uvsub_dir.glob("*.tar"))
    return tars


def extract_tar(tar_path: Path, dest_dir: Path):
    """
    Extract a .tar into dest_dir.
    Returns the list of top-level items extracted.
    """
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(dest_dir)
        names = [dest_dir / m.name.split("/")[0] for m in tf.getmembers() if "/" in m.name or m.name]
    # De-duplicate and keep existing only
    unique = []
    seen = set()
    for p in names:
        if p.exists() and p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def is_valid_ms(ms_path: Path) -> bool:
    """
    Quick check using casatools.ms: open/close to validate MS directory.
    """
    if not ms_path.exists():
        return False
    try:
        ms = ms_tool()
        ms.open(str(ms_path))
        ms.close()
        return True
    except Exception:
        return False


def main(
    csv_file: Path,
    uvsub_dir: Path,
    accum_root: Path,
    start_freq_first: int,
    start_freq_last_exclusive: int,
    step_MhZ: int,
    subband_span_MhZ: int,
    channel_width: int,
    timebin: str,
):
    """
    Recreates the original logic:

    For each 'f' in [start_freq_first, start_freq_last_exclusive) stepping by step_MhZ:
      - make a folder accum_root/f/
      - gather uvsub tarballs whose 'start_freq' falls within subbands f2 in [f, f+subband_span_MhZ) stepping by 4
      - extract each tar to accum_root/f/
      - validate each extracted MS; build vis list
      - casatasks.concat -> Accumulated_f.ms
      - casatasks.split  -> Accumulated_Split_f.ms with given channel width and timebin
    """

    print(f"[INFO] Loading metadata from {csv_file}")
    meta = load_metadata(csv_file)

    print(f"[INFO] Scanning for .tar files in {uvsub_dir}")
    tar_files = find_uvsub_tars(uvsub_dir)
    tar_names = {t.stem: t for t in tar_files}  # without .tar

    accum_root.mkdir(parents=True, exist_ok=True)

    # Make a quick index of rows that are usable (size>0, and have uv_sub_path)
    usable_rows = [row for row in meta if (row["size"] > 0.0 and len(row["uv_sub_path"]) > 0)]
    print(f"[INFO] Usable CSV rows with uv_sub_path and size>0: {len(usable_rows)}")

    for f in range(start_freq_first, start_freq_last_exclusive, step_MhZ):
        band_dir = accum_root / f"{f}"
        if band_dir.exists():
            shutil.rmtree(band_dir)
        band_dir.mkdir(parents=True)

        print(f"\n[INFO] === Freq bucket {f} MHz ===")
        # Collect MS tarballs to include for this bucket
        acc = []  # list of tuples: (start_freq, year, tar_path)
        for f2 in range(f, f + subband_span_MhZ, 4):  # original: step 4 MhZ
            for row in usable_rows:
                sf = int(row["start_freq"])
                if (sf >= f2) and (sf < f2 + 4):
                    # The CSV has uv_sub_path like ".../something.tar"
                    uvsp = Path(row["uv_sub_path"])
                    tar_key = uvsp.stem  # name without .tar
                    tar_path = None

                    # Prefer absolute uv_sub_path if it exists; else try matching by name in uvsub_dir
                    if uvsp.is_file():
                        tar_path = uvsp
                    else:
                        maybe = tar_names.get(tar_key)
                        if maybe is not None:
                            tar_path = maybe
                        else:
                            # Fall back: search by exact filename in uvsub_dir
                            candidate = uvsub_dir / uvsp.name
                            if candidate.is_file():
                                tar_path = candidate

                    if tar_path and tar_path.is_file():
                        acc.append((sf, row["year"], tar_path))

        if not acc:
            print(f"[WARN] No UV-sub tarballs found for bucket starting at {f} MHz. Skipping.")
            continue

        # Extract and validate
        vis = []
        for (_, _, tarp) in acc:
            try:
                extracted = extract_tar(tarp, band_dir)
                # Original code expected a single MS dir name equal to tar basename without .tar
                ms_dir_name = tarp.stem  # e.g. "XYZ" from "XYZ.tar"
                candidate_ms = band_dir / ms_dir_name
                if candidate_ms.is_dir() and is_valid_ms(candidate_ms):
                    vis.append(str(candidate_ms))
                else:
                    # If not found by stem, try any MS dirs extracted
                    for item in extracted:
                        if item.is_dir() and is_valid_ms(item):
                            vis.append(str(item))
                            break
                    else:
                        print(f"[WARN] No valid MS found in {tarp.name}")
            except Exception as e:
                print(f"[ERROR] Failed to extract or validate {tarp}: {e}")

        if not vis:
            print(f"[WARN] No valid MeasurementSets to concat in bucket {f}. Cleaning up and skipping.")
            shutil.rmtree(band_dir, ignore_errors=True)
            continue

        # Concat
        concat_out = band_dir / f"Accumulated_{f}.ms"
        if concat_out.exists():
            shutil.rmtree(concat_out)
        print(f"[INFO] concat -> {concat_out.name} (N={len(vis)})")
        concat(concatvis=str(concat_out), vis=vis)

        # Optionally remove extracted inputs after concat (matches original)
        for v in vis:
            try:
                shutil.rmtree(v, ignore_errors=True)
            except Exception:
                pass

        # Split
        split_out = band_dir / f"Accumulated_Split_{f}.ms"
        if split_out.exists():
            shutil.rmtree(split_out)
        try:
            print(f"[INFO] split -> {split_out.name} (timebin={timebin}, width={channel_width})")
            split(
                vis=str(concat_out),
                outputvis=str(split_out),
                timebin=timebin,
                width=channel_width,
                datacolumn="all",
            )
        except Exception as e:
            print(f"[ERROR] Split failed in bucket {f}: {e}")

    print("\n[INFO] Done.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Accumulate UV-sub MS tarballs into frequency buckets, concat, and split (CASA6)."
    )
    p.add_argument(
        "--csv_file",
        default="/home/00103780/chiles-daliuge/db/Chilies_metadata.csv",
        help="Path to Chilies_metadata.csv",
    )
    p.add_argument(
        "--uvsub_dir",
        default="/home/00103780/dlg/MeasurementSets/uvsub",
        help="Directory containing *.tar UV-sub archives",
    )
    p.add_argument(
        "--accum_root",
        default="/home/00103780/dlg/MeasurementSets/accumulated",
        help="Root directory to create per-bucket outputs",
    )
    p.add_argument("--start_freq_first", type=int, default=944, help="First start_freq bucket (MHz)")
    p.add_argument("--start_freq_last_exclusive", type=int, default=1009, help="Exclusive upper bound (MhZ)")
    p.add_argument("--step_MhZ", type=int, default=32, help="Bucket step (MhZ)")
    p.add_argument("--subband_span_MhZ", type=int, default=40, help="Span considered per bucket (MhZ)")
    p.add_argument("--channel_width", type=int, default=32, help="split(width=...)")
    p.add_argument("--timebin", default="60s", help="split(timebin=...)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        main(
            csv_file=Path(args.csv_file).expanduser(),
            uvsub_dir=Path(args.uvsub_dir).expanduser(),
            accum_root=Path(args.accum_root).expanduser(),
            start_freq_first=args.start_freq_first,
            start_freq_last_exclusive=args.start_freq_last_exclusive,
            step_MhZ=args.step_MhZ,
            subband_span_MhZ=args.subband_span_MhZ,
            channel_width=args.channel_width,
            timebin=args.timebin,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(130)
