#!/usr/bin/env python
"""Build Occluded-DukeMTMC from an existing DukeMTMC-reID archive.

The Occluded-DukeMTMC authors only release split lists, not the images.
This script copies images from DukeMTMC-reID.zip or an extracted DukeMTMC-reID
folder into the Occluded_Duke structure expected by this codebase.
"""

import argparse
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


SPLIT_ARCHIVE_URL = "https://github.com/lightas/Occluded-DukeMTMC-Dataset/archive/refs/heads/master.zip"
DUKE_OFFICIAL_URL = "http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip"
DUKE_ACADEMIC_TORRENT_URL = (
    "https://academictorrents.com/download/00099d85f6d8e8134b47b301b64349f469303990.torrent"
)
DUKE_ACADEMIC_MAGNET = (
    "magnet:?xt=urn:btih:00099d85f6d8e8134b47b301b64349f469303990&dn=DukeMTMC-reID.zip"
)
SUBSETS = {
    "bounding_box_train": ("train.list", "bounding_box_train"),
    "query": ("query.list", "query"),
    "bounding_box_test": ("gallery.list", "bounding_box_test"),
}
EXPECTED_COUNTS = {
    "bounding_box_train": 15618,
    "query": 2210,
    "bounding_box_test": 17661,
}
DUKE_REQUIRED_DIRS = ("bounding_box_train", "query", "bounding_box_test")


def safe_extract_zip(zip_path, output_dir):
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            member_path = (output_dir / member.filename).resolve()
            if output_dir not in member_path.parents and member_path != output_dir:
                raise RuntimeError("Unsafe path in zip archive: {}".format(member.filename))
        zip_ref.extractall(output_dir)


def find_dir(root, dirname):
    root = Path(root)
    candidates = [
        root / dirname,
        root / "DukeMTMC-reID" / dirname,
        root / "DukeMTMC-reID.zip" / dirname,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    matches = [path for path in root.rglob(dirname) if path.is_dir()]
    if not matches:
        raise FileNotFoundError("Could not find '{}' under '{}'.".format(dirname, root))
    return matches[0]


def has_duke_reid_dirs(root):
    root = Path(root)
    try:
        for dirname in DUKE_REQUIRED_DIRS:
            find_dir(root, dirname)
    except FileNotFoundError:
        return False
    return True


def validate_duke_zip_contents(zip_path):
    zip_path = Path(zip_path)
    if not zip_path.is_file() or not zipfile.is_zipfile(zip_path):
        return False

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        names = [name.replace("\\", "/").strip("/") for name in zip_ref.namelist()]
    return all(any("/{}".format(dirname) in name or name.startswith(dirname + "/") for name in names)
               for dirname in DUKE_REQUIRED_DIRS)


def download_url(url, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    print("Downloading DukeMTMC-reID from {}".format(url))
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        with tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    tmp_path.replace(output_path)
    return output_path


def download_with_aria2(source, output_path, work_dir):
    output_path = Path(output_path)
    work_dir = Path(work_dir)
    aria2c = shutil.which("aria2c")
    if aria2c is None:
        raise RuntimeError(
            "aria2c is required for Academic Torrents fallback. Install it with: apt-get update && apt-get install -y aria2"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        aria2c,
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--seed-time=0",
        "--summary-interval=30",
        "--dir",
        str(output_path.parent),
        "--out",
        output_path.name,
        source,
    ]
    print("Downloading DukeMTMC-reID via aria2c from {}".format(source))
    subprocess.run(cmd, check=True, cwd=str(work_dir))
    return output_path


def discover_local_duke_zip(args, work_dir):
    candidates = []
    if args.duke_zip:
        candidates.append(Path(args.duke_zip))

    env_path = os.environ.get("DUKEMTMC_REID_ZIP")
    if env_path:
        candidates.append(Path(env_path))

    output_parent = Path(args.output_root).parent
    candidates.extend([
        output_parent / "DukeMTMC-reID.zip",
        output_parent / "dukemtmc-reid" / "DukeMTMC-reID.zip",
        Path("/content/data/DukeMTMC-reID.zip"),
        Path("/content/data/dukemtmc-reid/DukeMTMC-reID.zip"),
        Path(work_dir) / "DukeMTMC-reID.zip",
    ])

    for candidate in candidates:
        if candidate.is_file():
            print("Found local DukeMTMC-reID zip: {}".format(candidate))
            return candidate
    return None


def download_duke_zip(args, work_dir):
    output_path = Path(work_dir) / "DukeMTMC-reID.zip"
    if output_path.is_file() and validate_duke_zip_contents(output_path):
        print("Using cached DukeMTMC-reID zip: {}".format(output_path))
        return output_path

    sources = []
    if args.duke_source in ("auto", "official"):
        sources.append(("official", "url", DUKE_OFFICIAL_URL))
    if args.duke_source in ("auto", "academic-torrents"):
        sources.extend([
            ("academic-torrents", "aria2", DUKE_ACADEMIC_TORRENT_URL),
            ("academic-torrents-magnet", "aria2", DUKE_ACADEMIC_MAGNET),
        ])

    failures = []
    for name, source_type, source in sources:
        try:
            if output_path.exists():
                output_path.unlink()
            if source_type == "url":
                download_url(source, output_path)
            else:
                download_with_aria2(source, output_path, work_dir)

            if validate_duke_zip_contents(output_path):
                print("Downloaded valid DukeMTMC-reID zip from {}".format(name))
                return output_path
            failures.append("{} downloaded file is not a valid DukeMTMC-reID zip".format(name))
        except Exception as exc:
            failures.append("{} failed: {}".format(name, exc))

    raise RuntimeError(
        "Could not automatically download DukeMTMC-reID.zip.\n"
        + "\n".join(" - " + failure for failure in failures)
    )


def find_split_root(root):
    root = Path(root)
    if all((root / list_name).is_file() for list_name, _ in SUBSETS.values()):
        return root

    matches = [
        path for path in root.rglob("Occluded_Duke")
        if path.is_dir() and all((path / list_name).is_file() for list_name, _ in SUBSETS.values())
    ]
    if not matches:
        raise FileNotFoundError(
            "Could not find Occluded_Duke split lists under '{}'. Expected train.list/query.list/gallery.list.".format(root)
        )
    return matches[0]


def read_split_names(split_root, list_name):
    list_path = split_root / list_name
    names = []
    with list_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(Path(line.replace("\\", "/")).name)
    if not names:
        raise RuntimeError("{} is empty.".format(list_path))
    return names


def get_duke_root(args, work_dir):
    if args.duke_root:
        duke_root = Path(args.duke_root)
        if not has_duke_reid_dirs(duke_root):
            raise FileNotFoundError("Could not find DukeMTMC-reID image folders under {}".format(duke_root))
        return duke_root

    duke_zip = discover_local_duke_zip(args, work_dir)
    if duke_zip is None:
        duke_zip = download_duke_zip(args, work_dir)
    if not validate_duke_zip_contents(duke_zip):
        raise RuntimeError("Invalid DukeMTMC-reID zip: {}".format(duke_zip))

    extract_root = work_dir / "DukeMTMC-reID"
    marker = extract_root / ".extracted"
    if args.force or not marker.is_file():
        if extract_root.exists():
            shutil.rmtree(extract_root)
        print("Extracting {} to {}".format(duke_zip, extract_root))
        safe_extract_zip(duke_zip, extract_root)
        marker.touch()
    return extract_root


def get_split_root(args, work_dir):
    if args.split_root:
        return find_split_root(Path(args.split_root))

    archive_path = work_dir / "Occluded-DukeMTMC-Dataset-master.zip"
    extract_root = work_dir / "Occluded-DukeMTMC-Dataset"
    marker = extract_root / ".extracted"

    if args.force or not archive_path.is_file():
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading split lists from {}".format(args.split_archive_url))
        urllib.request.urlretrieve(args.split_archive_url, archive_path)

    if args.force or not marker.is_file():
        if extract_root.exists():
            shutil.rmtree(extract_root)
        print("Extracting split lists to {}".format(extract_root))
        safe_extract_zip(archive_path, extract_root)
        marker.touch()

    return find_split_root(extract_root)


def copy_or_link(src, dst, mode, force):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            return False
        dst.unlink()

    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)
    return True


def build_image_index(source_dirs):
    image_index = {}
    duplicate_names = set()
    for source_subset in DUKE_REQUIRED_DIRS:
        source_dir = source_dirs[source_subset]
        for path in source_dir.iterdir():
            if not path.is_file():
                continue
            if path.name in image_index:
                duplicate_names.add(path.name)
                continue
            image_index[path.name] = path
    return image_index, duplicate_names


def resolve_source_image(name, preferred_dir, image_index):
    preferred_path = preferred_dir / name
    if preferred_path.is_file():
        return preferred_path, False
    fallback_path = image_index.get(name)
    if fallback_path is not None and fallback_path.is_file():
        return fallback_path, True
    return None, False


def build_dataset(args):
    output_root = Path(args.output_root)
    work_dir = Path(args.work_dir) if args.work_dir else output_root.parent / ".occluded_duke_prepare"
    work_dir.mkdir(parents=True, exist_ok=True)

    duke_root = get_duke_root(args, work_dir)
    split_root = get_split_root(args, work_dir)

    print("Using DukeMTMC-reID root: {}".format(duke_root))
    print("Using Occluded-Duke split lists: {}".format(split_root))
    print("Writing Occluded-DukeMTMC to: {}".format(output_root))

    source_dirs = {
        source_subset: find_dir(duke_root, source_subset)
        for _, source_subset in SUBSETS.values()
    }
    image_index, duplicate_names = build_image_index(source_dirs)
    if duplicate_names:
        print("Warning: {} duplicate image names found across DukeMTMC-reID folders; preferred subset paths are used first.".format(
            len(duplicate_names)
        ))

    total_copied = 0
    total_existing = 0
    for target_subset, (list_name, source_subset) in SUBSETS.items():
        names = read_split_names(split_root, list_name)
        source_dir = source_dirs[source_subset]
        target_dir = output_root / target_subset
        missing = []
        copied = 0
        existing = 0
        fallback_used = 0

        for name in names:
            src, used_fallback = resolve_source_image(name, source_dir, image_index)
            dst = target_dir / name
            if src is None:
                missing.append(name)
                continue
            fallback_used += int(used_fallback)
            changed = copy_or_link(src, dst, args.mode, args.force)
            copied += int(changed)
            existing += int(not changed)

        if missing:
            preview = ", ".join(missing[:10])
            raise FileNotFoundError(
                "{} missing {} source images. First missing: {}".format(target_subset, len(missing), preview)
            )

        expected = EXPECTED_COUNTS[target_subset]
        status = "OK" if len(names) == expected else "expected {}".format(expected)
        print("{}: {} images ({}) copied={}, existing={}, fallback={}".format(
            target_subset, len(names), status, copied, existing, fallback_used
        ))
        total_copied += copied
        total_existing += existing

    print("Done. copied={}, existing={}".format(total_copied, total_existing))


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Occluded-DukeMTMC for TransReID.")
    parser.add_argument("--duke-zip", default="", help="Path to DukeMTMC-reID.zip.")
    parser.add_argument("--duke-root", default="", help="Path to extracted DukeMTMC-reID folder.")
    parser.add_argument(
        "--duke-source",
        choices=("auto", "official", "academic-torrents"),
        default="auto",
        help="Built-in source used when DukeMTMC-reID.zip is not found locally.",
    )
    parser.add_argument("--output-root", default="data/Occluded_Duke", help="Output Occluded_Duke dataset folder.")
    parser.add_argument("--split-root", default="", help="Local Occluded_Duke split-list folder.")
    parser.add_argument("--split-archive-url", default=SPLIT_ARCHIVE_URL, help="URL for official split-list archive.")
    parser.add_argument("--work-dir", default="", help="Cache/extraction directory.")
    parser.add_argument("--mode", choices=("copy", "symlink"), default="copy", help="How to materialize images.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing extracted files and images.")
    return parser.parse_args()


def main():
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
