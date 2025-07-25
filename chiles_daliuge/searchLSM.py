import subprocess
import json
import sys
from typing import List

def list_lsm_and_region(bucket_path: str) -> List[str]:
    """
    For a given bucket_path, list only
      • directories whose name ends with “.ms” (your MS datasets)
      • files whose name contains “region-files”

    This avoids printing all the nested folders inside each .ms.
    """
    try:
        result = subprocess.run(
            ["rclone", "lsjson", "--recursive", bucket_path],
            capture_output=True, text=True, check=True
        )
        entries = json.loads(result.stdout)
        matches: List[str] = []
        for e in entries:
            name = e.get("Name", "")
            # only files (not dirs) whose name ends with .tar
            if not e.get("IsDir", False) and name.lower().endswith(".tar"):
                full = f"{bucket_path}/{e['Path']}"
                print(f"[TAR]  {full}")
                matches.append(full)

        if not matches:
            print(f"[INFO] no .tar files found in {bucket_path}")

        return matches

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] rclone failed on {bucket_path}: {e.stderr}", file=sys.stderr)
        return []
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse error on {bucket_path}: {e}", file=sys.stderr)
        return []

if __name__ == "__main__":
    target_buckets = [
        "acacia-chiles:2025-04-chiles01",
        "acacia-chiles:2025-04-chiles01-lsm",
        "acacia-chiles:chiles"
    ]

    for bucket in target_buckets:
        print(f"\n=== Scanning bucket: {bucket} ===")
        list_lsm_and_region(bucket)
