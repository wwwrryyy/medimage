"""
fetch_and_unpack_kermany_hf.py
==============================
Downloads Kermany Retinal OCT parquet shards from the curated Hugging Face mirror
(zacharielegault/Kermany2017-OCT) and unpacks images into:
    data/raw/kermany/{CLASS}/{CLASS}-{PATIENT}-{INDEX}.jpeg

Preserves original patient IDs for zero-leakage patient-level splits.
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq
from tqdm import tqdm

HF_BASE_URL = (
    "https://huggingface.co/datasets/zacharielegault/Kermany2017-OCT/resolve/main/data"
)
TOTAL_SHARDS = 16


def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.info().get("Content-Length", 0))
        chunk_size = 1024 * 1024  # 1 MB
        with open(temp_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest_path.name,
        ) as bar:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))

    temp_path.rename(dest_path)


def unpack_shard(parquet_path: Path, output_dir: Path) -> dict:
    table = pq.read_table(str(parquet_path))
    num_rows = len(table)
    img_col = table["image"].combine_chunks()
    class_counts = Counter()
    patient_set = set()

    for i in range(num_rows):
        item = img_col[i].as_py()
        img_bytes = item["bytes"]
        path_str = item["path"]

        filename = Path(path_str).name
        parts = filename.split("-")
        class_name = parts[0].upper()
        if len(parts) >= 2:
            patient_set.add(parts[1])

        class_counts[class_name] += 1
        target_dir = output_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / filename
        if not target_file.exists():
            with open(target_file, "wb") as f:
                f.write(img_bytes)

    print(
        f"[+] Unpacked {num_rows} records from {parquet_path.name} "
        f"across {len(patient_set)} patients: {dict(class_counts)}"
    )
    return dict(class_counts)


def main():
    parser = argparse.ArgumentParser(description="Fetch and unpack Kermany OCT from Hugging Face")
    parser.add_argument(
        "--shards",
        type=str,
        default="1-15",
        help="Shard indices to download: e.g. '1', '1,2,3', '1-5', or 'all' (0-15)",
    )
    parser.add_argument(
        "--keep-parquet",
        action="store_true",
        help="Keep downloaded parquet shards after unpacking",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/kermany",
        help="Directory to extract images into",
    )
    parser.add_argument(
        "--shard-dir",
        type=str,
        default="data/raw/shards",
        help="Directory to store parquet files",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    shard_dir = Path(args.shard_dir).resolve()

    if args.shards.lower() == "all":
        shard_indices = list(range(TOTAL_SHARDS))
    elif "-" in args.shards:
        start, end = map(int, args.shards.split("-"))
        shard_indices = list(range(start, end + 1))
    else:
        shard_indices = [int(s.strip()) for s in args.shards.split(",") if s.strip()]

    print(f"[*] Processing Kermany shards: {shard_indices}")
    for idx in shard_indices:
        shard_name = f"train-{idx:05d}-of-{TOTAL_SHARDS:05d}.parquet"
        shard_path = shard_dir / shard_name
        shard_url = f"{HF_BASE_URL}/{shard_name}"

        if not shard_path.exists():
            print(f"\n[*] Fetching {shard_name}...")
            download_file(shard_url, shard_path)

        unpack_shard(shard_path, out_dir)

        if not args.keep_parquet and shard_path.exists():
            print(f"[*] Cleaning up {shard_name} to conserve disk space...")
            shard_path.unlink()

    print("\n[+] Kermany unpacking complete!")


if __name__ == "__main__":
    main()

