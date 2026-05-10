"""
Step 1: Download and cache a BEIR dataset.

Run once before train_rf.py and benchmark.py.
Saves the local data path to beir_data_path.json under the dataset name as key.
Multiple datasets accumulate in the same file.
"""

import argparse
import json
import os

from beir import util
from beir.datasets.data_loader import GenericDataLoader

BEIR_URL_TEMPLATE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
DATA_PATH_FILE = "beir_data_path.json"


def main():
    parser = argparse.ArgumentParser(description="Download and cache a BEIR dataset")
    parser.add_argument(
        "--dataset",
        default="scidocs",
        help="BEIR dataset name (e.g. scidocs, fiqa, nq, hotpotqa). Default: scidocs",
    )
    parser.add_argument(
        "--data-dir",
        default="./datasets",
        help="Directory to download/unzip the dataset into (default: ./datasets)",
    )
    args = parser.parse_args()

    url = BEIR_URL_TEMPLATE.format(dataset=args.dataset)
    print(f"Downloading BEIR '{args.dataset}' (skipped if already cached)...")
    data_path = util.download_and_unzip(url, args.data_dir)

    _corpus, queries, qrels_dict = GenericDataLoader(data_folder=data_path).load(split="test")
    print(f"Loaded {len(queries)} queries, {len(qrels_dict)} queries with qrels")

    registry = {}
    if os.path.exists(DATA_PATH_FILE):
        with open(DATA_PATH_FILE) as f:
            registry = json.load(f)

    registry[args.dataset] = data_path

    with open(DATA_PATH_FILE, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Data path saved to {DATA_PATH_FILE} under key '{args.dataset}'")


if __name__ == "__main__":
    main()
