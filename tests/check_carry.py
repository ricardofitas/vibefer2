"""Compare the Vibefer2 OpenCL carry with exact truncating division and old code."""

import json
import sys
from pathlib import Path

import numpy as np
import pyopencl as cl


def run(source_path):
    source = source_path.read_text()
    prefix = "#define INLINE static inline\ntypedef long int_64; typedef ulong uint_64; typedef int int_32; typedef uint uint_32;\n"
    reduce = source[
        source.index("INLINE uint_32 barrett(") : source.index(
            "INLINE int_32 reduce96("
        )
    ]
    carry = source[
        source.index("INLINE int3 carry_three(") : source.index(
            "INLINE void normalize_2("
        )
    ]
    kernel = """
    __kernel void compare(__global const long4 *inputs, __global long4 *out, uint b, uint inv, int shift) {
        const size_t i = get_global_id(0);
        const long4 x = inputs[i];
        long carry = x.s0;
        const int3 d = convert_int3(x.yzw);
        const int3 v = carry_three(&carry, d, b, inv, shift);
        out[2*i] = (long4)(v.s0,v.s1,v.s2,carry);
        carry = x.s0 + d.s0;
        int a = reduce64(&carry,b,inv,shift);
        carry += d.s1;
        int c = reduce64(&carry,b,inv,shift);
        carry += d.s2;
        int e = reduce64(&carry,b,inv,shift);
        out[2*i+1] = (long4)(a,c,e,carry);
    }
    """
    devices = [
        d
        for p in cl.get_platforms()
        for d in p.get_devices(device_type=cl.device_type.GPU)
    ]
    if len(devices) != 1:
        raise ValueError("Expose exactly one graphics device")
    ctx = cl.Context(devices)
    queue = cl.CommandQueue(ctx)
    rng = np.random.default_rng(24017)
    count = 0
    for n, balanced in [(17, False), (17, True), (16, False)]:
        flags = f"#define LN_SZ {n}\n" + (
            "#define BALANCED_REMAINDER\n" if balanced else ""
        )
        program = cl.Program(ctx, flags + prefix + reduce + carry + kernel).build()
        compare = cl.Kernel(program, "compare")
        for b in [1026, 2**28 - 2, 2**28, 2**28 + 2, 518539494, 1000000000]:
            shift = b.bit_length() - 2
            inv = (1 << (32 + shift)) // b
            # Small bases in the original code require |carry+digit| < 2^23 b^2.
            bound = min(2**63 - 1, 2**23 * b * b - 1)
            triples = rng.integers(-b + 1, b, size=(4096, 3), dtype=np.int64)
            start = rng.integers(
                -bound + b + 2, bound - b - 1, size=len(triples), dtype=np.int64
            )
            samples = [
                [int(c), *[int(v) for v in z]]
                for c, z in zip(start, triples, strict=True)
            ]
            for c in [-(bound - b - 2), -b - 1, -b, -1, 0, 1, b, b + 1, bound - b - 2]:
                for z in [-b - 2, -b, -1, 0, 1, b, b + 2]:
                    samples.append([c, z, z, z])
            inputs = np.array(samples, np.int64)
            output = np.empty((len(inputs), 2, 4), np.int64)
            a = cl.Buffer(
                ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=inputs
            )
            o = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
            compare(
                queue,
                (len(inputs),),
                None,
                a,
                o,
                np.uint32(b),
                np.uint32(inv),
                np.int32(shift),
            )
            cl.enqueue_copy(queue, output, o).wait()
            ref = []
            for c, *digits in samples:
                row = []
                for d in digits:
                    v = c + d
                    c = abs(v) // b if v >= 0 else -(abs(v) // b)
                    rem = v - c * b
                    if balanced and abs(rem) > b // 2:
                        adj = 1 if rem > 0 else -1
                        c += adj
                        rem -= adj * b
                    row.append(rem)
                ref.append([*row, c])
            if not np.array_equal(output[:, 0], output[:, 1]) or not np.array_equal(
                output[:, 0], np.asarray(ref)
            ):
                raise ValueError(
                    f"Different carry for b={b}, n={n}, balanced={balanced}"
                )
            count += len(inputs)
    return {
        "gpu": devices[0].name,
        "cases": count,
        "exponents": [16, 17],
        "seed": 24017,
    }


if __name__ == "__main__":
    print("RESULT " + json.dumps(run(Path(sys.argv[1]))))
