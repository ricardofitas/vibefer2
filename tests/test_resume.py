"""Resume a Vibefer checkpoint using Vibefer2 and check the official proof digest."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("binary", "checkpoint", "metadata", "references"):
        p.add_argument(arg, type=Path)
    p.add_argument("--timeout", type=int, default=270, help="maximum GPU runtime in seconds")
    args = p.parse_args()
    if list(Path(".").glob("candidate.*")) or list(Path(".").glob("candidate_*.ckpt")):
        p.error("Run in a fresh directory without existing candidate files.")
    meta = json.loads(args.metadata.read_text())
    with args.checkpoint.open("rb") as checkpoint_file:
        digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    if digest != meta["sha256"]:
        raise ValueError("Unexpected checkpoint")
    shutil.copyfile(args.checkpoint, "candidate.ctx")
    command = [
        str(args.binary.resolve()),
        "-n",
        "17",
        "-b",
        str(meta["base"]),
        "-p",
        "-f",
        "candidate",
        "-d",
        "0",
    ]
    run = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        timeout=args.timeout,
    )
    Path("application.log").write_text(run.stdout)
    if (
        "Resuming from a checkpoint" not in run.stdout
        or "Vibefer2 0.2 /" not in run.stdout
    ):
        raise ValueError("New binary did not resume")
    proof = Path("candidate.proof").read_bytes()
    ref = json.loads(args.references.read_text())["bases"][str(meta["base"])]
    digest = hashlib.sha256(proof).hexdigest()
    if digest != ref["proof_sha256"]:
        raise ValueError("Different proof after resumption")
    print(
        "RESULT "
        + json.dumps(
            {
                "base": meta["base"],
                "resumed": True,
                "proof_sha256": digest,
                "bytes": len(proof),
                "checkpoint_sha256": meta["sha256"],
            }
        )
    )
