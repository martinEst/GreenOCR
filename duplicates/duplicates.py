import os
import hashlib
import shutil
import platform
import re
from datetime import datetime

def file_hash(path, block_size=65536):
    """Compute a hash for a file's content efficiently."""
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            hasher.update(block)
    return hasher.hexdigest()

import os
import platform
import subprocess

def get_file_age(path):
    system = platform.system()
    stat = os.stat(path)

    if system == "Windows":
        return stat.st_ctime
    elif system == "Darwin":  # macOS
        return getattr(stat, "st_birthtime", stat.st_mtime)
    elif system == "Linux":
        # Try stat command to get real creation time
        try:
            output = subprocess.check_output(['stat', '-c', '%W', path], text=True).strip()
            ts = int(output)
            if ts > 0:
                return ts  # true creation time supported
        except Exception:
            pass
        return stat.st_mtime  # fallback
    else:
        return stat.st_mtime


def get_file_ageO(path):
    """
    Return the best 'creation' or 'oldest change' time depending on platform.
    On Windows: uses creation time (ctime).
    On Unix/macOS: uses modification time (mtime), since ctime is inode metadata.
    """
    stat = os.stat(path)
    if platform.system() == "Windows":
        return stat.st_ctime  # real creation time
    else:
        return stat.st_mtime  # modification time


def find_duplicate_groups(root):
    """
    Find duplicate files grouped by identical content.
    Returns list of dicts: {"hash": str, "kept": str, "duplicates": [str, ...]}.
    """
    size_groups = {}
    try:
        test = myvars["extensions"]
        test2 = [p.strip() for p in test.split(",") if p.strip()]
        pattern = re.compile(r".*\.(%s)$" % "|".join(test2), re.IGNORECASE)
    except:
        pattern = re.compile(r".*\.(%s)$" % ".*", re.IGNORECASE)

    

    #pattern = re.compile(r".*\.txt$")

    # Step 1: Group by file size (cheap pre-filter)
    for path, _, files in os.walk(root):
        for name in files:
            if pattern.match(name):
                fullpath = os.path.join(path, name)
                try:
                    size = os.stat(fullpath).st_size
                except FileNotFoundError:
                    continue
                size_groups.setdefault(size, []).append(fullpath)

    groups = []

    # Step 2: Within each size group, compare by hash
    for size, file_list in size_groups.items():
        if len(file_list) <= 1:
            continue

        seen_hashes = {}
        for file in file_list:
            try:
                h = file_hash(file)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

            if h in seen_hashes:
                seen_hashes[h].append(file)
            else:
                seen_hashes[h] = [file]

        # Step 3: Build groups
        for h, files_with_same_hash in seen_hashes.items():
            if len(files_with_same_hash) > 1:
                # Sort by creation/modification time → oldest first
                files_with_same_hash.sort(key=get_file_age)
                kept = files_with_same_hash[0]
                duplicates = files_with_same_hash[1:]
                groups.append({
                    "hash": h,
                    "kept": kept,
                    "duplicates": duplicates
                })

    return groups


def print_duplicate_summary(groups):
    """Nicely print all duplicate groups."""
    print("\n=== Duplicate File Groups ===\n")
    for idx, g in enumerate(groups, 1):
        kept_time = datetime.fromtimestamp(get_file_age(g['kept']))
        print(f"Group {idx}:")
        print(f"  Kept (oldest): {g['kept']}")
        print(f"    Timestamp : {kept_time}")
        for d in g['duplicates']:
            mod_time = datetime.fromtimestamp(get_file_age(d))
            print(f"  Duplicate (newer): {d}")
            print(f"    Timestamp : {mod_time}")
        print()


def move_duplicates(groups, trash_dir):
    """Move duplicates to a safe location."""
    os.makedirs(trash_dir, exist_ok=True)
    total_bytes = 0

    for g in groups:
        for d in g['duplicates']:
            try:
                size = os.stat(d).st_size
                total_bytes += size
                
                
                # Compute relative path
                rel_path = os.path.relpath(d,myvars["rootFolder"])

                # Build target path under trash_dir
                trash_path = os.path.join(trash_dir, rel_path)

                # Create subfolders if needed
                os.makedirs(os.path.dirname(trash_path), exist_ok=True)

                #new_path = os.path.join(trash_dir, os.path.basename(d))

                print(f"Moving {d} -> {trash_path}")
                if int(myvars["toRemove"]) == 1:
                    shutil.move(d, trash_path)
                elif int(myvars["toRemove"]) == -1:
                    shutil.copy(d, trash_path)
            except Exception as e:
                print(f"  ⚠️ Error moving {d}: {e}")

    print(f"\nTotal space freed: {total_bytes / (1024 * 1024):.2f} MB")


myvars = {}
with open("duplicates/reff.cnf") as myfile:
    for line in myfile:
        name, var = line[:-1].strip().partition("=")[::2]
        myvars[name.strip()] = var.strip()



#print(myvars)


# Example usage
if __name__ == "__main__":
    root = myvars["rootFolder"]
    trash_dir = myvars["binFolder"]

    duplicate_groups = find_duplicate_groups(root)
    print_duplicate_summary(duplicate_groups)

    # Uncomment below line to move duplicates to speicfic folder defined in conf file
    move_duplicates(duplicate_groups, trash_dir)
