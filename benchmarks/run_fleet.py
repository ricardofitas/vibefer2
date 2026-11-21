#!/usr/bin/env python3
"""Replay a fixed proof cohort on concurrent GPUs, then verify every proof."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_proof import digest, stage, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="device mapping, workers and binaries (see fleet.example.json)")
    parser.add_argument("--genefer22", type=Path, required=True, help="independent stock Genefer22 GPU executable")
    parser.add_argument("--output", type=Path, required=True, help="new directory for all proofs and measurements")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("proof_manifest.csv"))
    args = parser.parse_args()
    with args.manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    bases = [r["base"] for r in rows]
    if not bases or any(not b.isascii() or not b.isdecimal() for b in bases) or len(set(bases)) != len(bases):
        parser.error("The manifest must contain distinct numeric bases.")
    devices = json.loads(args.config.read_text())
    names = [d["name"] for d in devices]
    ordinals = [d["device"] for d in devices]
    if len(set(names)) != len(names) or set(names) != {r["device"] for r in rows}:
        parser.error("Configure exactly one GPU per device name in the manifest.")
    if len(set(ordinals)) != len(ordinals) or any(type(i) is not int or i < 0 for i in ordinals):
        parser.error("Device ordinals must be distinct nonnegative integers.")
    checker = args.genefer22.resolve()
    checker_hash = digest(checker)
    pools, environments, settings = {}, {}, {}
    for d in devices:
        name = d["name"]
        if type(d["workers"]) is not int or d["workers"] < 1:
            parser.error(f"{name}: workers must be a positive integer.")
        binary = (args.config.parent / d["binary"]).resolve()
        binary_hash = digest(binary)
        if binary_hash == checker_hash:
            parser.error(f"{name}: Vibefer2 and the stock checker are the same binary.")
        d["binary"] = binary
        environments[name] = {**os.environ, **d.get("environment", {})}
        settings[name] = {"device": d["device"], "workers": d["workers"], "application_sha256": binary_hash}
    by_name = {d["name"]: d for d in devices}
    args.output.mkdir(parents=True, exist_ok=False)
    for base in bases:
        (args.output / base).mkdir()
    generation = {}
    with ExitStack() as cleanup:
        for d in devices:
            pool = ThreadPoolExecutor(max_workers=d["workers"])
            cleanup.callback(pool.shutdown, wait=True, cancel_futures=True)
            pools[d["name"]] = pool
        futures = {}
        start = time.perf_counter()
        for row in rows:
            name, base = row["device"], row["base"]
            dev = by_name[name]
            task = pools[name].submit(stage, dev["binary"], "-p", "generate", base, dev["device"],
                                      args.output / base, environments[name])
            futures[task] = base
        for task in as_completed(futures):
            generation[futures[task]] = task.result()
        wall = time.perf_counter() - start

    check_start = time.perf_counter()
    records = []
    checker_env = {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": "/dev/null"}
    for row in rows:
        base, name = row["base"], row["device"]
        record = verify(row, checker, by_name[name]["device"], args.output / base, generation[base], checker_env)
        records.append({**record, "device_name": name})
    report = {"devices": settings, "genefer22_sha256": checker_hash, "manifest_sha256": digest(args.manifest),
              "proofs": len(records), "generation_wall_seconds": wall, "verified_proofs_per_generation_hour": 3600 * len(records) / wall,
              "verification_wall_seconds": time.perf_counter() - check_start, "runs": records}
    (args.output / "fleet-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(records)} verified proofs, {wall:.2f} s generation wall time, {3600 * len(records) / wall:.2f} proofs/h")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
