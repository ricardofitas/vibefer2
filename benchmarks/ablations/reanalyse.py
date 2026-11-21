#!/usr/bin/env python3
"""Reanalyse archived proof runs and dependent-square comparisons; Python 3.8+."""
import argparse
import csv
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
OUT_FIELDS = [
    'comparison', 'device', 'baseline', 'candidate', 'workers', 'endpoint',
    'n_observations_per_arm', 'baseline_time', 'candidate_time',
    'throughput_change_pct', 'lower_observed_pct', 'upper_observed_pct',
    'candidate_wins', 'method'
]


def read_csv(directory, filename):
    with (directory / filename).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def gain(old, new):
    return 100.0 * (old / new - 1.0)


def block_summary(name, device, left, right, workers, endpoint, pairs):
    changes = [gain(a, b) for a, b in pairs]
    return [name, device, left, right, workers, endpoint, len(pairs),
            median(a for a, _ in pairs), median(b for _, b in pairs),
            median(changes), min(changes), max(changes), sum(g > 0 for g in changes),
            'median of within-block throughput changes']


def arithmetic(rows, device, group, endpoint):
    result = []
    stages = defaultdict(dict)
    for r in rows:
        b = stages[r['block']]
        assert int(r['order']) == len(b) + 1 and r['arm'] not in b
        b[r['arm']] = float(r['aggregate_us'])
    treatments = (('stock', 'integer', 'fp32') if device.endswith('Ti')
                  else ('stock', 'center9', 'center8'))
    assert {tuple(b) for b in stages.values()} == set(permutations(treatments))
    comparisons = [(treatments[0], treatments[1]), (treatments[0], treatments[2]),
                   (treatments[1], treatments[2])]
    for left, right in comparisons:
        pairs = [(v[left], v[right]) for v in stages.values()]
        result.append(block_summary(group, device, left, right, rows[0]['workers'], endpoint, pairs))
    return result


def analyse(data, raw):
    result = []
    matched = read_csv(data, 'matched_runs.csv')
    for exp in ('single_base_graph', 'single_base_leaf'):
        rows = [r for r in matched if r['experiment'] == exp]
        assert len(rows) == 3 and {r['workload'] for r in rows} == {'518539494'}
        for a,b in zip(rows, rows[1:]):
            ta,tb = float(a['wall_s']),float(b['wall_s'])
            result.append([exp, a['device'], a['arm'], b['arm'], 1, 'seconds per proof', 1,
                           ta, tb, gain(ta,tb), '', '', '', 'one matched input'])

    for key,left,right in [('ti_isolated_fusion','base','fusion'),
                           ('ti_graph','mix32','mix128'),('ti_leaf','mix128','warp128')]:
        runs = [r for r in raw if r['study'] == key and r['arm'] in (left,right)]
        assert len(runs) == 4 and len({r['final_hash'] for r in runs}) == 1
        assert [r['arm'] for r in runs] in ([left,right,right,left],[right,left,left,right])
        pairs = []
        for pair in (runs[:2],runs[2:]):
            v = {r['arm']: float(r['aggregate_us']) for r in pair}
            pairs.append((v[left],v[right]))
        result.append(block_summary(key,'RTX 4060 Ti',left,right,1,'microseconds per square',pairs))

    groups = defaultdict(lambda: defaultdict(list))
    for r in raw:
        if r['study'] == 'fixed workers':
            groups[r['device'],int(r['workers'])][r['arm']].append(float(r['aggregate_us']))
    for (device,workers),arms in sorted(groups.items()):
        a,b = arms['plain'],arms['mps']
        assert len(a) == len(b)
        result.append(['MPS','RTX '+device,'plain','MPS',workers,'microseconds per aggregate square',
                       len(a),mean(a),mean(b),gain(mean(a),mean(b)),'','','',
                       'throughput ratio of mean times at fixed worker count'])

    packed = read_csv(data,'paired_squares.csv')
    for carry in ('exact','q44'):
        result.extend(arithmetic([r for r in packed if r['carry'] == carry],
                                 'RTX 4060','packed '+carry,'microseconds per aggregate square'))
    result.extend(arithmetic(read_csv(data,'ti_third_channel.csv'),'RTX 4060 Ti',
                             'third channel','microseconds per aggregate square'))
    alternatives = defaultdict(list)
    for row in read_csv(data,'ti_alternatives.csv'):
        alternatives[row['alternative'], row['setting']].append(
            (float(row['baseline_us']),float(row['alternative_us'])))
    for (arm,setting), pairs in alternatives.items():
        result.append(block_summary('alternative kernel','RTX 4060 Ti','stock',arm+' '+setting,
                                    6,'microseconds per aggregate square',pairs))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark-dir', type=Path, default=HERE.parent,
                    help='Directory containing the four public benchmark CSVs.')
    opts = ap.parse_args()
    raw = read_csv(HERE/'data','archived_controls.csv')
    writer = csv.writer(sys.stdout,lineterminator='\n')
    writer.writerow(OUT_FIELDS)
    for row in analyse(opts.benchmark_dir,raw):
        writer.writerow([f'{v:.6f}' if isinstance(v,float) else v for v in row])
