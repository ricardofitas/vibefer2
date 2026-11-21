# Host CRC-32 ablation

This experiment times the function that updates the streaming checksum during binary reads and writes. Upstream Vibefer consumes one byte per table step. The first Vibefer2 patch introduced slicing by eight and the 0.2 release retained it. The benchmark runs the original method bodies verbatim in the same executable. Input bytes, initial CRC, compiler, optimization flags and loop nesting are matched within each pair. The only intervention within a pair is the selected method. The released method keeps an 8 KiB family of tables in place of the historical 1 KiB table, combining eight independent table lookups per eight input bytes.

## Reproduction

From this benchmark directory, on Linux with Python 3.10+, Git, a C++17 compiler called g++, the system zlib development package and util-linux taskset installed, run:

```sh
python3 run.py --root /path/to/vibefer2 --out /tmp/vibefer2-crc-results
```

Pass `--cpu N` to select a logical processor inside the available affinity mask. The recorded run used logical CPU 7. The runner regenerates the upstream header by reversing the first archived patch on the released header. Before extracting the C++ methods it verifies both entire headers against SHA-256 digests pinned in the script. No network or GPU is required. A file called `trials.csv` lists every timed arm and `summary.json` records the environment and paired summaries.

| Source | Identity |
|---|---|
| [Vibefer header](https://github.com/JohannesDav/Vibefer/blob/94019ef77eb8a9e0680889ff63bda59ee5e42e8e/src/file.h) | SHA-256 `096ab1e59e428a079245ae081cd2febf34dbbac294289140aaa3180c2ccc4348` |
| [Vibefer2 header](https://github.com/ricardofitas/vibefer2/blob/00d900682068d9d1500513da37c3601bdc60f335/src/file.h) | SHA-256 `efa3f9a83f5f96d84e8492c67323fe8b1b16aa1d397abe33db3908e8016d4dd3` |
| [Proof manifest](https://github.com/ricardofitas/vibefer2/blob/00d900682068d9d1500513da37c3601bdc60f335/benchmarks/proof_manifest.csv) | GFN-17 proof size of 4,194,380 bytes |

## Measurement

A deterministic xorshift32 stream fills a 4,194,380-byte buffer, matching the archived proof file size. Both arms process identical in-memory bytes. A timed batch concatenates 12 passes, chaining the CRC across calls and passes. One condition uses one call per buffer. The second uses 4 KiB calls with a short tail. The driver runs two untimed batches per method and chunk size to warm the data and tables, then alternates execution order AB/BA across 20 pairs per condition. Each method runs first in ten pairs. Condition order also alternates. Each pair has a different initial CRC. All 80 measured checksums must match the independent zlib reference for that stream. The checks and CSV output occur outside the timed interval.

The driver records elapsed time with the C++ steady clock and records thread CPU time separately. Rates use elapsed time and MiB means 1,048,576 bytes. Paired speedup is the upstream elapsed time divided by the Vibefer2 elapsed time within each pair. All observations are retained. The optional intervals in the JSON use the sixth and fifteenth ordered ratio of 20, an equal-tail sign interval for a median assuming independent replicates. A single machine and process limit generalization across hardware.

| Bytes per update | Vibefer median (MiB/s) | Vibefer2 median (MiB/s) | Median paired speedup | Full paired range |
|---:|---:|---:|---:|---:|
| 4,194,380 | 406.91 | 2,027.07 | 5.029× | 4.769–7.691× |
| 4,096 | 403.91 | 2,016.84 | 4.972× | 4.501–5.658× |

All 20 pairs favored the new method in each condition on one pinned AMD EPYC 9V74 virtual CPU, using GCC 14.2.0, `-O3 -std=c++17 -DNDEBUG -fno-lto` and Debian 13. The evaluation took place on a separate host from the discovery workstation. Timing covers a warm in-memory checksum using synthetic payloads. File I/O, GPU arithmetic, certificate handling and scheduling fall outside this experiment.
