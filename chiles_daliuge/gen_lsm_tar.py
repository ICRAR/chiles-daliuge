import os
import tarfile
import subprocess
from pathlib import Path

def make_rel_link_filter(root_dir: Path):
    root_dir = root_dir.resolve()

    def _filter(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        if ti.issym():
            link = ti.linkname
            if os.path.isabs(link):
                target_abs = Path(link).resolve()
                try:
                    target_abs.relative_to(root_dir)
                    symlink_fs_path = (root_dir / ti.name).resolve().parent
                    rel = os.path.relpath(str(target_abs), start=str(symlink_fs_path))
                    ti.linkname = rel
                except ValueError:
                    # link points outside root_dir, leave unchanged
                    pass
        return ti
    return _filter

def create_tar_preserving_symlinks(src_dir: str, tar_path: str) -> None:
    src_path = Path(src_dir).resolve()
    arc_root = src_path.name

    with tarfile.open(tar_path, mode="w:", format=tarfile.PAX_FORMAT, dereference=False) as tf:
        tf.add(
            str(src_path),
            arcname=arc_root,
            recursive=True,
            filter=make_rel_link_filter(src_path)
        )
    print(f"[INFO] Created tar archive at {tar_path}")

def upload_with_rclone(local_path: str, remote_path: str) -> None:
    try:
        subprocess.run(
            ["rclone", "copyto", local_path, remote_path],
            check=True
        )
        print(f"[INFO] Uploaded {local_path} -> {remote_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] rclone upload failed: {e}")

if __name__ == "__main__":
    src_dir = "/home/00103780/dlg/LSM"
    tar_path = "/home/00103780/dlg/LSM.tar"
    bucket_path = "acacia-chiles:2025-04-chiles01/LSM.tar"

    if Path(tar_path).exists():
        print(f"[INFO] {tar_path} already exists, skipping creation.")
    else:
        create_tar_preserving_symlinks(src_dir, tar_path)

    # Upload with rclone
    upload_with_rclone(tar_path, bucket_path)
