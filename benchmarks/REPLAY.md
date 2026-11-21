# Replay a proof on one GPU

`run_proof.py` reruns one or more composite bases from the archived GFN-17 cohort. It generates each proof with Vibefer2, then uses a separately supplied stock Genefer22 GPU executable for the server and certificate checks. No project account or network connection is needed during replay.

Build Vibefer2 using [the build guide](../UPSTREAM.md). Supply the independently obtained [Genefer22](https://github.com/galloty/genefer22) GPU executable as the verifier. Set both applications to the same visible GPU and use a new output directory with enough space to retain proofs, certificates and logs. The historical proofs in the manifest are 4,194,380 bytes each. For example, from the repository root:

```sh
python3 benchmarks/run_proof.py \
    --vibefer ./build/vibefer2 --genefer22 /path/to/stock/geneferg \
    --device 0 --output replay-example 517504600 518530058
```

For each base the script checks the proof length, SHA-256, two residues and proof key against [the archived cohort](proof_manifest.csv). It then runs Genefer22 in server mode (`-s`), requires the same residues and proof key, and checks the resulting certificate using Genefer22 in checker mode (`-c`). The two stock invocations must return the same nonzero 64-bit certificate key. Stock Genefer22 selects a fresh challenge during certificate generation, so the new key can differ from the archived certificate key.

A successful run produces `report.json` containing both executable hashes, the GPU ordinal and the elapsed time for each of the three stages. Each base has its own directory with the proof, certificate and per-stage output. A mismatch, missing artifact, child-process error or pre-existing output directory stops the replay with a nonzero exit. A failed run retains the work already written. Run again in a fresh directory.

The script runs the bases sequentially. Its elapsed times include process startup and file I/O. Replaying an individual proof checks correctness and gives single-worker latency. The four-GPU throughput measurements in the paper used concurrent workers and the observation windows in the benchmark data.

## Replay all 32 archived bases on four GPUs

`run_fleet.py` uses the manifest's historical device assignments and launches a separate process for each proof. Six slots on the RTX 4060 Ti generate 12 proofs, three on the RTX 4060 generate six, one on the RTX 3070 generates seven and two on the RTX 2080 Ti generate seven. The archived cohort used baseline 0.2 for the two Ada cards, the experimental warp-leaf build on the RTX 3070, and the `small32` build on the RTX 2080 Ti. These profiles belong to the isolated 32-proof experiment.

For the RTX 3070, create a separate checkout and apply `variants/ampere-warp.patch` on top of baseline 0.2 before building it. Use [the variant guide](../variants/README.md) to build the RTX 2080 Ti executable. The original 3070 executable had SHA-256 `99c0abbe6f031859eb2858af7db47b213cc8e91c053d8d654521c7d0ffb01cce`. The corresponding original Git blob for the patch was `497b62a0df9cfe4dfb2bbc151e30ffed454126c2`.

Copy `fleet.example.json` and set the four device ordinals to those exposed by **both** binaries on your host. Set the application paths relative to the configuration file, or use absolute paths. For the historical MPS configurations, use an already configured MPS server and set `VIBEFER_MPS_COMPILER` to the absolute helper path, globally or in the individual device's `environment` object. If the servers have separate pipes, specify `CUDA_MPS_PIPE_DIRECTORY` per device. The sample requests a 33% active-thread limit for each Ti client, a 100% limit for the other MPS clients, and directs the 3070 worker outside MPS. Device selection, thermal controls and server availability are properties of the host.

```sh
python3 benchmarks/run_fleet.py --config benchmarks/fleet.example.json \
    --genefer22 /path/to/stock/geneferg --output replay-four-gpus
```

The common generation clock begins before the first job is submitted and ends after the last child has exited and its result has been collected. The script then reads each proof, compares its archive digest and keys, and runs both stock certificate stages. Only an entirely successful cohort produces `fleet-report.json` and a throughput. The denominator excludes the subsequent serial verification time, which the report records separately. The launcher uses fresh processes, manifest order within each device queue and the machine's existing settings. The historical 633.12 s interval included the start and exit of isolated BOINC clients, host-specific scheduling and thermal pauses. The local rate therefore measures the supplied machine and configuration.
