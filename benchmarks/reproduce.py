#!/usr/bin/env python3
"""Recompute the production, paired arithmetic and proof results using the Python standard library."""
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from itertools import permutations
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
GPUS = ('RTX 4060 Ti', 'RTX 4060', 'RTX 3070', 'RTX 2080 Ti')


def read(stem):
    with (ROOT / (stem + '.csv')).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def production():
    intervals = defaultdict(dict)
    profiles = defaultdict(set)
    for row in read('production_intervals'):
        start, end = (datetime.fromisoformat(row[k]) for k in ('window_start_utc', 'window_end_utc'))
        assert row['attributed'] == row['active'] == 'true'
        device = row['device']
        assert device in GPUS and device not in intervals[start, end]
        intervals[start, end][device] = row
        profiles[device].add(row['profile'])
    assert len(intervals) == 54 and all(len(v) == 1 for v in profiles.values())
    ordered = sorted(intervals)
    assert all(left[1] == right[0] for left, right in zip(ordered, ordered[1:]))
    assert (ordered[-1][1] - ordered[0][0]).total_seconds() == 9 * 3600
    assert all(set(devices) == set(GPUS) for devices in intervals.values())

    hourly = read('production_windows')
    assert len(hourly) == 45
    fleet_rows = [r for r in hourly if r['device'] == 'fleet']
    assert len(fleet_rows) == 9
    fields = ('main_searches', 'certificates', 'unknown', 'nonzero_exits', 'known_errors')
    totals = defaultdict(Counter)
    for row in hourly:
        start, end = (datetime.fromisoformat(row[k]) for k in ('start_utc', 'end_utc'))
        window = [k for k in ordered if start <= k[0] and k[1] <= end]
        assert len(window) == 6 and (end - start).total_seconds() == 3600
        devices = GPUS if row['device'] == 'fleet' else (row['device'],)
        observed = {k: sum(int(intervals[t][d][k]) for t in window for d in devices) for k in fields}
        assert all(observed[k] == int(row[k]) for k in fields)
        totals[row['device']].update(observed)
    for device in ('fleet', *GPUS):
        n = totals[device]
        print(f"{device:12s} {n['main_searches']:4d} main, {n['certificates']:4d} certificates in 9 h; "
              f"{n['main_searches']/9:.2f} main/h")
    print('Hourly main counts:', ', '.join(r['main_searches'] for r in fleet_rows))
    print('Unknown classes, nonzero exits and known errors:',
          *(totals['fleet'][k] for k in fields[2:]))


def arithmetic():
    rows = read('paired_squares')
    for method in ('exact', 'q44'):
        blocks = defaultdict(dict)
        for r in rows:
            if r['carry'] == method:
                block = blocks[int(r['block'])]
                assert int(r['order']) == len(block) + 1
                assert r['arm'] not in block
                block[r['arm']] = float(r['aggregate_us'])
        assert set(tuple(b) for b in blocks.values()) == set(permutations(('stock', 'center8', 'center9')))
        for arm in ('center8', 'center9'):
            ratios = [b['stock']/b[arm] for b in blocks.values()]
            print(f"{method} {arm}: {100*(median(ratios)-1):+.2f}% paired median throughput, "
                  f"{sum(x>1 for x in ratios)}/{len(ratios)} blocks won")
    pairs = defaultdict(list)
    for r in read('ti_alternatives'):
        pairs[r['alternative'], r['setting']].append(float(r['alternative_us'])/float(r['baseline_us']))
    for (arm, setting), ratios in pairs.items():
        print(f"RTX 4060 Ti {arm} {setting}: {100*(median(ratios)-1):+.1f}% paired median time, "
              f"{sum(v<1 for v in ratios)}/{len(ratios)} pairs won")


def proofs():
    manifest = read('proof_manifest')
    assert len({r['base'] for r in manifest}) == len(manifest)
    for row in manifest:
        assert len(bytes.fromhex(row['proof_sha256'])) == 32 and row['genefer22_verified'] == 'yes'
        assert row['official_bytes_reference'] in ('matched', 'unavailable')
    print('Isolated batch:', len(manifest), 'verified proofs;',
          sum(r['official_bytes_reference'] == 'matched' for r in manifest),
          'matched archived official proof files.')
    for row in read('matched_runs'):
        rate = 3600*int(row['main_searches'])/float(row['wall_s'])
        assert abs(rate - float(row['main_per_hour'])) < 1e-6


def selected_profiles():
    blocks = defaultdict(dict)
    for run in read('ti_third_channel'):
        block = blocks[int(run['block'])]
        assert int(run['order']) == len(block) + 1 and run['arm'] not in block
        block[run['arm']] = float(run['aggregate_us'])
    assert set(tuple(b) for b in blocks.values()) == set(permutations(('stock', 'integer', 'fp32')))
    for arm in ('stock', 'integer', 'fp32'):
        times = [b[arm] for b in blocks.values()]
        ratio = median(b['stock']/b[arm] for b in blocks.values())
        print(f"Ti third channel {arm}: {median(times):.3f} us, {100*(ratio-1):+.2f}% paired throughput")
    with (ROOT / 'ti_layout_check.json').open(encoding='utf-8') as stream:
        layout = json.load(stream)
    states, primes, classes, width = layout['capture_shape']
    real = len(layout['real_checkpoint_bases'])
    assert states == real + layout['edge_case_states']
    assert real * primes * classes * width == layout['complete_native_path_residues_matching_GMP']
    for arm in layout['isolated_leaf_matches_per_variant'].values():
        assert states * primes * classes * width == arm['checked_residues']
    cohorts = defaultdict(set)
    reference = {}
    for p in read('profile_proofs'):
        assert p['official_byte_match'] == 'true'
        assert p['proof_sha256'] == reference.setdefault(p['base'], p['proof_sha256'])
        cohorts[p['profile']].add(p['base'])
    assert len(cohorts['eight-prime exact']) == 3
    assert len(cohorts['small32']) == 2 and cohorts['small32'] == cohorts['stock 0.2']
    print('Ti captured layout and selected RTX 4060 / RTX 2080 Ti proof records verified.')


if __name__ == '__main__':
    production()
    arithmetic()
    proofs()
    selected_profiles()
