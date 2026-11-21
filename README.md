# Vibefer2

Vibefer2 0.2 searches for generalized Fermat probable primes and writes proofs of modular exponentiation compatible with Genefer22. It extends Johannes David's Vibefer with GPU-specific execution graphs, changes to carry propagation, a faster streaming CRC and a Linux compilation helper for CUDA Multi-Process Service (MPS) workers. The original arithmetic and proof protocol are credited in [UPSTREAM.md](UPSTREAM.md).

## Build and run

On Linux, with a C++17 compiler, static GMP, CUDA headers and an OpenCL loader installed:

```sh
python3 build.py --cuda /usr/local/cuda --without-boinc \
    --output build/vibefer2 --mps-compiler build/vibefer-compiler
./build/vibefer2 -v
```

The [build and verification guide](UPSTREAM.md) covers the BOINC-enabled build, prerequisites and CPU-side regression checks. The default source tree is baseline 0.2. The [variant guide](variants/README.md) describes the separate RTX 4060 eight-prime and RTX 2080 Ti small32 builds, their admission rules and the observed production executable hashes.

## Benchmark

Run the published analysis on any machine with Python 3:

```sh
python3 benchmarks/reproduce.py
```

| Endpoint | Observation |
|---|---|
| Matched, proof-producing 3070 run on PCIe 3.0 ×1 | 358.61 s with the earlier graph, 186.39 s with a reusable 32-step graph, 185.40 s with carry fusion added |
| RTX 4060, dependent squarings from three states | Eight-prime exact carry improved median paired throughput by 24.91% over six order-balanced blocks |
| Four-GPU isolated replay | 32 main searches in 633.12 s; all 32 proofs checked by Genefer22 outside the timing interval |
| Nine contiguous production hours | 1,838 locally completed main searches (204.22/h) and 1,831 certificate jobs |

The [benchmark guide](benchmarks/README.md) defines the time boundaries, server validation snapshots and source of every table. The [replay guide](benchmarks/REPLAY.md) covers fresh proof generation on one GPU or all four, checks every proof against archived evidence and uses an independent Genefer22 verifier. The [historical patches](history/) reconstruct the earlier 0.1 implementation.

## Megaprime discovery

The Vibefer2 deployment on an RTX 4060 Ti found `518246744^131072 + 1`, a prime with **1,142,232 decimal digits**. PrimeGrid recorded the work unit on 20 November 2026. PrimePages entry 142659 credits Ricardo Fitas within the project proof code. The attribution to release 0.2 comes from the reporting host and adjacent worker snapshots. [Discovery evidence](benchmarks/DISCOVERY.md) · [PrimePages](https://t5k.org/primes/page.php?id=142659)

The root software license is [MIT](LICENSE). The vendored OpenCL headers carry [their own license](Khronos/CL/LICENSE).
