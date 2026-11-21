#!/usr/bin/env python3
"""Replay archived GFN-17 proofs on one GPU with an independent Genefer22 checker."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def stage(binary, mode, name, base, device, work, env=None):
    command = [str(binary), "-n", "17", "-b", base, mode, "-d", str(device), "-f", "candidate"]
    start = time.perf_counter()
    with (work / (name + ".log")).open("wb") as log:
        subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
    elapsed = time.perf_counter() - start
    # The standalone application appends its official status to results.txt.
    # Give each stage its own fresh result, even if a binary exits without one.
    status = work / (name + ".txt")
    (work / "results.txt").rename(status)
    entries = re.findall(r"\b(res64|old64|pkey|ckey)\s*=\s*([a-fA-F0-9]{16})\b", status.read_text())
    keys = {field: value.upper() for field, value in entries}
    if len(keys) != len(entries):
        raise ValueError(f"{base}: ambiguous duplicate key in {name} output")
    return keys, elapsed


def verify(reference, checker, device, work, generation, env=None):
    base = reference["base"]
    generated, proof_time = generation
    proof = work / "candidate.proof"
    if proof.stat().st_size != int(reference["bytes"]) or digest(proof) != reference["proof_sha256"]:
        raise ValueError(f"{base}: proof size or SHA-256 disagrees with the archived proof")
    expected = {"res64": reference["res64"], "old64": reference["old64"], "pkey": reference["proof_key"]}
    if any(generated.get(key) != value.upper() for key, value in expected.items()):
        raise ValueError(f"{base}: generated residue or proof key disagrees with the archive")

    server, server_time = stage(checker, "-s", "server", base, device, work, env)
    if any(server.get(key) != generated[key] for key in expected):
        raise ValueError(f"{base}: stock Genefer22 disagrees on residues or proof key")
    if not (work / "candidate.cert").stat().st_size:
        raise ValueError(f"{base}: stock Genefer22 produced an empty certificate")
    verified, check_time = stage(checker, "-c", "check", base, device, work, env)
    challenge = server.get("ckey")
    if not challenge or int(challenge, 16) == 0 or verified.get("ckey") != challenge:
        raise ValueError(f"{base}: missing, zero or inconsistent certificate key")
    return {
        "base": base, "proof_sha256": reference["proof_sha256"], "proof_bytes": proof.stat().st_size,
        "res64": generated["res64"], "old64": generated["old64"], "proof_key": generated["pkey"],
        "certificate_key": challenge, "proof_seconds": proof_time,
        "server_seconds": server_time, "check_seconds": check_time,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vibefer", required=True, type=Path, help="Vibefer2 GPU executable")
    parser.add_argument("--genefer22", required=True, type=Path, help="independently obtained stock Genefer22 GPU executable")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--device", type=int, default=0, help="device ordinal in both applications (default 0)")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("proof_manifest.csv"))
    parser.add_argument("bases", nargs="+", help="one or more distinct base values in the proof manifest")
    args = parser.parse_args()
    if args.device < 0 or len(set(args.bases)) != len(args.bases):
        parser.error("Supply a nonnegative device ordinal and distinct bases.")
    with args.manifest.open(newline="") as manifest:
        rows = {row["base"]: row for row in csv.DictReader(manifest)}
    unknown = [base for base in args.bases if base not in rows or not base.isascii() or not base.isdecimal()]
    if unknown:
        parser.error("Unknown base(s): " + ", ".join(unknown))
    app, checker = args.vibefer.resolve(), args.genefer22.resolve()
    app_hash, checker_hash = digest(app), digest(checker)
    if app_hash == checker_hash:
        parser.error("The two supplied executables have identical contents.")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"application_sha256": app_hash, "genefer22_sha256": checker_hash, "device": args.device, "runs": []}
    for base in args.bases:
        work = args.output / base
        work.mkdir()
        generation = stage(app, "-p", "generate", base, args.device, work)
        record = verify(rows[base], checker, args.device, work, generation)
        report["runs"].append(record)
        print(f"{base}: proof SHA-256, residues, proof key and fresh certificate key verified", flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
