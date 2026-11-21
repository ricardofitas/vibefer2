#!/usr/bin/env python3
"""Reproduce the host-only table-driven CRC ablation. Requires Git, g++ and zlib development headers."""
import argparse
import csv
import ctypes
import ctypes.util
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
from statistics import median
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
BASE_DIGEST = "096ab1e59e428a079245ae081cd2febf34dbbac294289140aaa3180c2ccc4348"
NEW_DIGEST = "efa3f9a83f5f96d84e8492c67323fe8b1b16aa1d397abe33db3908e8016d4dd3"
FLAGS = ["-O3", "-std=c++17", "-DNDEBUG", "-fno-lto"]
MIB = 1024 ** 2


def body(header, digest):
    data = header.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Unexpected source digest for {header}")
    text = data.decode()
    start = text.index("static uint32_t rc_crc32(")
    depth = 0
    for end in range(text.index("{", start), len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[start:end + 1]
    raise ValueError("Unterminated method")


def summarize(rows):
    result = []
    for chunk in sorted({int(row["chunk_bytes"]) for row in rows}, reverse=True):
        groups = {}
        all_speeds = {}
        for row in rows:
            if int(row["chunk_bytes"]) != chunk:
                continue
            elapsed = int(row["wall_ns"]) / 1e9
            throughput = int(row["bytes_per_sweep"]) * int(row["sweeps"]) / MIB / elapsed
            groups.setdefault(int(row["pair"]), {})[row["method"]] = elapsed
            all_speeds.setdefault(row["method"], []).append(throughput)
        ratios = sorted(pair["vibefer_bytewise"] / pair["vibefer2_slice8"] for pair in groups.values())
        n = len(ratios)
        # Exact binomial equal-tail interval for a population median under independent, continuous ratios.
        k = max(j for j in range(1, n // 2 + 1)
                if 1 - 2 * sum(math.comb(n, i) for i in range(j)) / 2 ** n >= .95)
        result.append({"chunk_bytes": chunk, "pairs": n, "wins": sum(r > 1 for r in ratios),
                       "median_mib_s": {m: median(v) for m, v in all_speeds.items()},
                       "median_paired_speedup": median(ratios), "paired_speedup_range": [min(ratios), max(ratios)],
                       "median_order_statistic_limits_95": [ratios[k - 1], ratios[-k]]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Vibefer2 repository root")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--cpu", type=int, help="logical CPU, defaults to the first allowed CPU")
    args = parser.parse_args()
    repo = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cpu = args.cpu if args.cpu is not None else min(os.sched_getaffinity(0))
    if cpu not in os.sched_getaffinity(0):
        raise ValueError("CPU is outside the process affinity mask")
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        (work / "src").mkdir()
        hist = work / "src/file.h"
        released = repo / "src/file.h"
        shutil.copyfile(released, hist)
        subprocess.run(["git", "apply", "--whitespace=nowarn", "--no-index", "--reverse",
                        "--include=src/file.h", str(repo / "history/vibefer2.patch")], cwd=work, check=True)
        methods = [("Historical", body(hist, BASE_DIGEST)), ("Released", body(released, NEW_DIGEST))]
        (work / "crc_methods.h").write_text("\n".join(f"struct {name} {{\n{method}\n}};" for name, method in methods))
        subprocess.run(["g++", *FLAGS, "-I", str(work), str(HERE / "benchmark.cpp"),
                        "-lz", "-o", str(work / "bench")], check=True)
        with (out / "trials.csv").open("w") as raw:
            subprocess.run(["taskset", "-c", str(cpu), str(work / "bench")], stdout=raw, check=True)
    with (out / "trials.csv").open() as raw:
        rows = list(csv.DictReader(raw))
    if len(rows) != 80:
        raise RuntimeError("Incomplete run")
    for condition in {r["chunk_bytes"] for r in rows}:
        cohort = [r for r in rows if r["chunk_bytes"] == condition]
        for m in {"vibefer_bytewise", "vibefer2_slice8"}:
            assert sum(r["method"] == m and r["position"] == "1" for r in cohort) == 10
    model = next(x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines()
                 if x.startswith("model name"))
    zlib_library = ctypes.CDLL(ctypes.util.find_library("z"))
    zlib_library.zlibVersion.restype = ctypes.c_char_p
    record = {"system": platform.system(), "release": platform.release(), "arch": platform.machine(),
              "os": platform.freedesktop_os_release()["PRETTY_NAME"], "cpu_model": model,
              "affinity_logical_cpu": cpu, "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
              "flags": FLAGS, "zlib_runtime": zlib_library.zlibVersion().decode(), "warmup_batches_per_arm_and_chunk": 2,
              "sha256": {"upstream_src_file_h": BASE_DIGEST, "vibefer2_src_file_h": NEW_DIGEST},
              "summary": summarize(rows)}
    (out / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["summary"], indent=2))


if __name__ == "__main__":
    main()
