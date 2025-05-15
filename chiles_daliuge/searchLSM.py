import subprocess
import json
from typing import List

def list_lsm_related_paths(bucket_path: str) -> List[str]:
    """
    List all files and directories containing 'lsm' or 'LSM' in their names
    in an rclone-accessible remote bucket and its subdirectories.

    Parameters
    ----------
    bucket_path : str
        The rclone remote path (e.g., "acacia-chiles:2025-04-chiles01").

    Returns
    -------
    list of str
        Full rclone paths to all matching files and directories found.
    """
    try:
        result = subprocess.run(
            ["rclone", "lsjson", "--recursive", bucket_path],
            capture_output=True, text=True, check=True
        )
        entries = json.loads(result.stdout)
        matching_paths = []

        for entry in entries:
            name = entry.get("Name", "")
            if "lsm" in name.lower():
                full_path = f"{bucket_path}/{entry['Path']}"
                entry_type = "[DIR]" if entry.get("IsDir", False) else "[FILE]"
                print(f"{entry_type} {full_path}")
                matching_paths.append(full_path)

        if not matching_paths:
            print("[INFO] No files or directories with 'lsm' or 'LSM' in the name found.")
        return matching_paths

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] rclone failed: {e.stderr}")
        return []
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse rclone output: {e}")
        return []

# Example usage
if __name__ == "__main__":
    bucket = "acacia-chiles:2025-04-chiles01-lsm"
    lsm_paths = list_lsm_related_paths(bucket)
    print("\nSummary of matching files and directories:")
    for path in lsm_paths:
        print(path)


#rclone copy acacia-chiles:2025-04-chiles01-lsm/LSM /home/00103780/dlg/LSM --create-empty-src-dirs
