# Source provenance and build

Vibefer2 0.2 extends the CUDA edition of [Johannes David's Vibefer](https://github.com/JohannesDav/Vibefer/tree/94019ef77eb8a9e0680889ff63bda59ee5e42e8e), itself derived from [Yves Gallot's Genefer](https://github.com/galloty/genefer22). The primality test, proof protocol and checkpoint format come from Genefer. Vibefer contributes CUDA execution, packed transforms, multiple exact arithmetic paths and device calibration. Vibefer2 changes selected kernels, execution and I/O on top of these foundations.

The root build is the **baseline 0.2 application**. Its build script, source, embedded kernel inputs and profile database reproduce the source obtained by applying the following archived patches in order to the pinned upstream tree. The three regression checks `check_crc.cpp`, `check_carry.py` and `test_resume.py` in `tests/` accompany the original patches. The upstream copyright notices are in `LICENSE`. Vendored OpenCL headers retain their separate notice in `Khronos/CL/LICENSE`.

| Input | Git object ID |
|---|---|
| Upstream Vibefer commit | `94019ef77eb8a9e0680889ff63bda59ee5e42e8e` |
| Experiment archive commit containing the two patches | `8716ffe56f8888494c9379e64735c6929561dcef` |
| First patch, `benchmarks/vibefer2/vibefer2.patch` | `27831b128fca06621bdad2b2c57fb27db0dd8281` |
| Second patch, `benchmarks/vibefer2/vibefer2-0.2.patch` | `dd3e55d18511c96155f86846250c98de38965cd7` |

Historical runs also used model-specific builds for the RTX 4060, RTX 2080 Ti and RTX 3070. See [the variant build guide](variants/README.md) for the deployment profiles and [the replay guide](benchmarks/REPLAY.md) for the isolated cohort's RTX 3070 warp-leaf build.

## Build on Linux

The application uses Python 3 for code generation, a C++17 compiler, static GMP, the CUDA SDK headers and an OpenCL loader. The program loads the CUDA driver at runtime. On the machine used for GPU execution, the selected NVIDIA device must also be visible to NVIDIA OpenCL. On a system with these dependencies installed under their usual paths:

```sh
python3 build.py --cuda /usr/local/cuda --without-boinc \
    --output build/vibefer2 --mps-compiler build/vibefer-compiler
./build/vibefer2 -v
```

For the BOINC-enabled build, replace `--without-boinc` with `--boinc /usr` or the prefix containing the BOINC headers and static archives `libboinc_api.a`, `libboinc_opencl.a` and `libboinc.a`. The linked SDK must export the application API. For example, the Debian build of the 8.2.15 development package supplies these symbols. A missing `boinc_init_options` at link time can indicate an empty archive in the installed development package. The [upstream build guide](https://github.com/JohannesDav/Vibefer/blob/94019ef77eb8a9e0680889ff63bda59ee5e42e8e/doc/cuda.md) also describes how to build the SDK.

For a Linux worker already configured to use an NVIDIA MPS server, set `VIBEFER_MPS_COMPILER` to the absolute path of its helper, for example `export VIBEFER_MPS_COMPILER="$PWD/build/vibefer-compiler"`. The helper compiles on the worker's assigned device outside the MPS server and returns PTX to the worker. Each build also writes `vibefer2.build.json` next to the executable. On Windows, omit the Linux-only `--mps-compiler` option.

## Verification

The following checks run on the CPU from the repository root. They cover parameter admission, profile selection, data handoff, exact CRT arithmetic, sequences of dependent squarings and compatibility of the streaming CRC with its bitwise reference. The packed reference test uses this revision's parameter generator in `tools/packed_parameters.py`.

```sh
mkdir -p build/checks/generated
python3 tests/packed_reference.py
python3 tests/crt_reference.py
python3 tools/embed_cuda_kernels.py build/checks/generated/cuda/embedded_kernels.h
g++ -std=c++17 -O2 -Isrc tests/cuda_parameters_probe.cpp -lgmp -o build/checks/parameters
python3 tests/test_cuda_parameters.py build/checks/parameters
g++ -std=c++17 -O2 -Isrc -Ibuild/checks/generated tests/cuda_profiles.cpp \
    src/cuda/profiles.cpp -lgmp -o build/checks/profiles
./build/checks/profiles
g++ -std=c++17 -O2 -Isrc -Ibuild/checks/generated tests/cuda_handoff.cpp \
    src/cuda/kernel.cpp -lgmp -o build/checks/handoff
./build/checks/handoff
g++ -std=c++17 -O2 -Isrc tests/check_crc.cpp -lgmp -o build/checks/crc
./build/checks/crc
```

With PyOpenCL, NumPy and one visible supported GPU, `python3 tests/check_carry.py kernels/src/three-prime.cl` compares device carries against exact truncating division and the earlier implementation. A full proof-generating run uses `-n 17 -b BASE -p -f PREFIX`. Run in an isolated directory with a fresh prefix and verify the resulting proof against the reference application on the same base.

`tests/test_resume.py` checks a user-supplied checkpoint with the 0.2 executable. It requires Python 3.11 or later, one compatible GPU, a Vibefer checkpoint file, its metadata JSON (`base` and `sha256`), and a reference JSON of the form `{"bases":{"BASE":{"proof_sha256":"SHA256"}}}`. From a fresh directory, run `python3 /path/to/tests/test_resume.py /path/to/vibefer2 /path/to/checkpoint /path/to/metadata.json /path/to/references.json`. The fixture checks the checkpoint digest, requires an explicit resumption message and the 0.2 banner, then compares the final proof digest. The GPU lifecycle test `tests/boinc_lifecycle.py` additionally requires a BOINC-enabled executable and the SDK fixture initializer built from `tests/boinc_init.cpp`.
