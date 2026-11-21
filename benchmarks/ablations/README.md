# Component ablations for Vibefer2

Run the following from the directory containing this file. Python 3.8 or later and the standard library suffice.

```sh
python3 reanalyse.py > regenerated.csv
diff -u results.csv regenerated.csv
```

The four shared input CSV files in the parent directory are the [published benchmark at revision 00d9006](https://github.com/ricardofitas/vibefer2/tree/00d900682068d9d1500513da37c3601bdc60f335/benchmarks). To use a checkout of that revision, pass its benchmark directory with `python3 reanalyse.py --benchmark-dir /path/to/vibefer2/benchmarks`.

The additional input, `data/archived_controls.csv`, extracts the fixed-width plain and MPS trials on the 3070 and 4060 and the single-worker graph, fusion and leaf trials on the Ti. Each row points to the pinned machine archive. The source keys are `gpu_3070_mps.loop`, `cuda_mps_4060`, `cuda_mps_4060_widths`, `kernel_ti.fused`, `kernel_ti.dispatch_and_fusion`, and `kernel_ti.integrated`. The single-worker comparisons retain measured 131,072-step runs, including their common terminal hashes. Rows are in archive order within each study. Worker counts are active processes. Times on multi-worker arithmetic trials use the shared interval divided by the number of candidate-squarings. [Machine archive](https://github.com/ricardofitas/prime-grid-comm/blob/8716ffe56f8888494c9379e64735c6929561dcef/reports/data/vibefer2-meta-120/medicoes.json)

Each paired change is 100 times the control-to-candidate time ratio minus 100. The six-block and Ti two-pair studies report the median change across matched blocks; each displayed arm time is its marginal median. The MPS four-worker comparison uses the ratio of its two arithmetic mean times. A range denotes minimum and maximum measured changes across the archived blocks. Blank range and win cells indicate that a blockwise spread was not estimated. The original experiments ran on one machine. This analysis requires no GPU and performs no new accelerator measurements.


The plot requires Matplotlib. Run `python3 plot_ablation.py --output /tmp/vibefer2-paired-contrasts.pdf` to visualize all six blocks in each RTX 4060 comparison.

The host CRC-32 experiment compares pinned upstream and Vibefer2 method bodies on deterministic data with independent zlib checks. Its runner, C++ driver, raw 80-arm measurements, environment and summary are in [host-crc](host-crc/README.md). Running that benchmark requires a Linux host, Git, Python 3.10+, g++, zlib development headers and taskset.
