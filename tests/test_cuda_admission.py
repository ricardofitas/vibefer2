"""Exhaustive ratio-three GFN-19 selection parity and minimum-count check.

Compile after embedding the kernels:
  g++ -std=c++17 -O2 -Isrc -Ibuild/tests/generated \
      tests/cuda_admission_probe.cpp -lgmp -o build/tests/admission
Then run:
  python3 tests/test_cuda_admission.py build/tests/admission [--workers 8]
The interval includes both endpoints. No GPU is required.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import importlib.util
import json
import multiprocessing
import subprocess
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("model", ROOT / "tools/packed_parameters.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
FIRST, LAST = 31296292, 32923372


def check_chunk(args):
    probe, start, stop, length, pack, first = args
    lines = [f"{b} {length} {pack} 3 {first} {first + 1}" for b in range(start, stop, 2)]
    process = subprocess.run([probe], input="\n".join(lines) + "\n",
                             capture_output=True, text=True, check=True)
    output = process.stdout.splitlines()
    assert len(output) == len(lines)
    counts = {}
    examples = {}
    for base, line in zip(range(start, stop, 2), output):
        expected, expected_mask, checked_mask = None, 0, 0
        raw_parameters = model.parameters
        cache = {}
        def cached_parameters(base, fields, **kwargs):
            # Oracle and selector use identical pure arithmetic inputs. Reuse
            # their constants, but exercise the real selection code separately.
            options = {"bits": 32, **kwargs}
            key = (base, fields, tuple(
                (name, tuple(value) if isinstance(value, list) else value)
                for name, value in sorted(options.items())
            ))
            if key not in cache:
                try:
                    cache[key] = raw_parameters(base, fields, **kwargs)
                except model.UnsupportedPackedParameters as error:
                    cache[key] = error
            if isinstance(cache[key], Exception):
                raise cache[key]
            return cache[key]
        for fields in (first, first + 1):
            kwargs = dict(length=length, pack=pack,
                          primes=model.field_pool(fields, length)[0], accurate_error=True)
            for index, ratio in enumerate((3, 2)):
                bit = 1 << (2 * (fields - first) + index)
                checked_mask |= bit
                try:
                    a = cached_parameters(base, fields, digit_ratio=ratio, **kwargs)
                    assert base + 2 < min(kwargs["primes"])
                    model.require(2 * a["e"] * a["B"] + a["e"] + a["Tmax"] < 3 * a["B"], "bridge")
                    expected_mask |= bit
                    expected = (fields, ratio)
                    break
                except model.UnsupportedPackedParameters:
                    pass
            with patch.object(model, "parameters", cached_parameters):
                try:
                    selected = model.admit_profile(base, fields, digit_ratio=3, **kwargs)
                    assert expected == (fields, selected["limit"] // selected["B"])
                except model.UnsupportedPackedParameters:
                    assert expected is None
            if expected is not None:
                break
        selected_fields, selected_ratio, mask = map(int, line.split())
        assert expected is not None and (selected_fields, selected_ratio) == expected, (base, length, line, expected)
        assert mask & checked_mask == expected_mask, (base, length, mask, expected_mask)
        key = f"{expected[0]}/{expected[1]}"
        counts[key] = counts.get(key, 0) + 1
        examples.setdefault(key, base)
    return dict(length=length, count=len(lines), selections=counts, examples=examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # Omitted ratios use the same preference as an explicit ratio-three
    # profile, and the generator must pass the admitted ratio to its codec.
    for base, fields, ratio in [(30000000, 14, 3), (31296294, 14, 2), (31296292, 15, 3)]:
        config = dict(length=65536, pack=8, accurate_error=True)
        explicit, metadata = model.program(base, fields, digit_ratio=3, **config)
        implicit, _ = model.program(base, fields, **config)
        assert explicit == implicit and metadata["digit_ratio"] == ratio
        assert f"#define PG_DIGIT_RATIO {ratio}\n" in explicit
    profiles = json.loads((ROOT / "profiles/nvidia.json").read_text())
    layouts = set()
    def visit(value):
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key == "19" and isinstance(child, dict) and child.get("implementation") == "packed" and child.get("digit_ratio", 3) == 3:
                    length = child["length"]
                    layouts.add((length, (1 << 19) // length))
                else:
                    visit(child)
    visit(profiles)
    tasks, totals = [], {}
    for length, pack in sorted(layouts):
        # Exact rank inequalities prove that every smaller count rejects both
        # ratios throughout this interval. They increase monotonically with b.
        first = None
        for fields in range(2, 33):
            primes = model.field_pool(fields, length)[0]
            P = model.prod(primes)
            delta = sum((Fraction((1 << 30) - p, 2 << 30) for p in primes), Fraction())
            if Fraction(2 * length * (2 * FIRST ** pack) ** 2, P) + delta < Fraction(1, 2):
                first = fields
                break
        assert first is not None
        totals[str(length)] = dict(pack=pack, first_possible_count=first, count=0, selections={}, examples={})
        for start in range(FIRST, LAST + 1, 2000):
            tasks.append((str(args.probe.resolve()), start, min(start + 2000, LAST + 2), length, pack, first))
    started = time.monotonic()
    with multiprocessing.Pool(args.workers) as pool:
        for index, row in enumerate(pool.imap_unordered(check_chunk, tasks), 1):
            total = totals[str(row["length"])]
            total["count"] += row["count"]
            for key, count in row["selections"].items():
                total["selections"][key] = total["selections"].get(key, 0) + count
                total["examples"][key] = min(total["examples"].get(key, LAST + 2), row["examples"][key])
            if index % 25 == 0:
                print(f"Checked {sum(t['count'] for t in totals.values()):,} base/layout pairs in {time.monotonic() - started:.1f}s", flush=True)
    count = (LAST - FIRST) // 2 + 1
    assert all(t["count"] == count for t in totals.values())
    result = dict(status="PASS", first=FIRST, last=LAST, bases_per_layout=count,
                  seconds=time.monotonic() - started, layouts=totals)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
