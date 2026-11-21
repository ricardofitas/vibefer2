# GPU profiles used in the production benchmark

The repository root contains Vibefer2 0.2. The four-GPU experiment used that source throughout hour B. Hour C changed the RTX 2080 Ti executable. Hour D changed the RTX 4060 executable as well. The RTX 4060 Ti and RTX 3070 used the baseline in all three hours. These patches reconstruct the two additional executables on separate copies of the release.

| GPU and build | Source changes from 0.2 | Runtime selection | Observed production binary SHA-256 |
|---|---|---|---|
| Baseline | None | Device profile selected by the application | `abb09a6a3541d3c9174f8fffc328ae9a44c92d0025099bc1bc3c2f8b4d6a9be8` |
| RTX 2080 Ti, `small32` | `turing-small32.patch` | `VIBEFER_2080TI_PROFILE=small32` | `21cdaac1cd256ed350ef80c5488e1f0278e158f7b17089bfefe2494a05fd21f2` |
| RTX 4060, eight-prime exact carry | `ada-centered.patch`, then `ada-eight-default.patch` | Default on eligible even bases from 500,000,000 to 525,000,000 | `6eaa672b12feac9b29a1e96fa6669f747dc134d0907fc8200946d396b2a4ba20` |

`small32` selects three primes, 32-bit root tables, a warp-level leaf and a 128-square graph. It checks the RTX 2080 Ti device name, memory range, exponent and base. The RTX 4060 variant packs four base‑*b* digits into each of 32,768 transform elements. It checks the arithmetic bounds for eight centered fields. The production measurement used the exact integer carry path. Leave `VIBEFER_4060_CENTER_FIELDS` and `VIBEFER_4060_CENTER_METHOD` unset to reproduce the selection.

## Build from a clean clone

Run these commands at the repository root to create two independent worktrees. Use the same CUDA and BOINC SDK paths as in the main build instructions.

```sh
git worktree add --detach ../vibefer2-small32 HEAD
git worktree add --detach ../vibefer2-ada8 HEAD

cd ../vibefer2-small32
git apply variants/turing-small32.patch
python3 build.py --cuda "$CUDA_SDK" --boinc "$BOINC_SDK" \
  --output build/vibefer2 --mps-compiler build/vibefer-compiler

cd ../vibefer2-ada8
git apply variants/ada-centered.patch
git apply variants/ada-eight-default.patch
python3 build.py --cuda "$CUDA_SDK" --boinc "$BOINC_SDK" \
  --output build/vibefer2 --mps-compiler build/vibefer-compiler
```

For command-line builds, replace `--boinc "$BOINC_SDK"` with `--without-boinc`. In a Linux MPS run, set `VIBEFER_MPS_COMPILER` to the absolute path of the helper for that build. Set `VIBEFER_2080TI_PROFILE=small32` in the environment of the Turing worker. Each executable retains the 0.2 version banner. Record both the binary digest and the selected profile when comparing runs. Compiler, linker and SDK versions can change the binary digest.

## Verify the eight-prime arithmetic on a CPU

The independent [CPU oracle](check_eight_centered.py) uses Python, NumPy and gmpy2. It checks 163,840 convolution coefficients against GMP, five resulting integer residues and the carry across the cyclic boundary in both directions. It covers dense inputs and a uniform extremal case. From the repository root, run the following command without Python optimization.

```sh
python3 variants/check_eight_centered.py > eight-prime-run.json
```

Compare the output with the [recorded reference](eight_prime_reference.json). `cpu_wall_s` measures the local machine. The input cases, field primes, outcomes and residue hashes are deterministic. This test exercises the arithmetic specification independently of the CUDA implementation.

## Source provenance

The source patches come from revision `8716ffe56f8888494c9379e64735c6929561dcef` of the experiment archive. The original Git blob IDs allow byte-for-byte comparison using `git hash-object`.

| Distributed patch | Original patch path under `benchmarks/vibefer2/producao/` | Git blob ID |
|---|---|---|
| `turing-small32.patch` | `turing-x4/selector.patch` | `fd91c61251b51b70ab5ba41fe20493a2f3b289bf` |
| `ada-centered.patch` | `ada-packed/centered.patch` | `5ff85ff6e941a550b5802e960e185c6c00711f5e` |
| `ada-eight-default.patch` | `ada-packed/default-eight.patch` | `f83621f78246dc1a6c16b57ad8f0d63e0193d200` |
