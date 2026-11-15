"""Create paper-compatible tables from actual three-task evaluation outputs.

No metrics are imputed. All means are *unweighted* over five courts, matching
Table 7 of the Indian judiciary paper. Per-seed results are then mean +/- sample std.
"""
import argparse
import json
import csv
from pathlib import Path
import numpy as np
import pandas as pd

TASKS = ('duration', 'case_type', 'disposal')
COURT_COUNT = 5


def find_runs(root):
    runs = {}
    for fp in sorted(Path(root).glob('*/seed_*/per_court_test.csv')):
        task = fp.parent.parent.name
        seed = int(fp.parent.name.rsplit('_', 1)[-1])
        if task not in TASKS:
            continue
        df = pd.read_csv(fp)
        if len(df) != COURT_COUNT or df.court.nunique() != COURT_COUNT:
            raise ValueError(f'{fp} must contain exactly five distinct courts')
        runs[task, seed] = df
    return runs


def create_tables(root, destination):
    runs = find_runs(root)
    if not runs:
        raise ValueError(f'No runs found under {root}')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    per_seed = []
    details = []
    for (task, seed), df in sorted(runs.items()):
        for _, r in df.iterrows():
            details.append(r.to_dict())
        vals = {'task': task, 'seed': seed, 'algorithm': str(df.algorithm.iloc[0]),
                'number_of_courts': COURT_COUNT}
        for col in ('maelog', 'mae_days', 'spearman', 'f1_4', 'macro_f1', 'accuracy'):
            if col in df:
                vals[col] = float(df[col].mean()) if df[col].notna().any() else float('nan')
        if task == 'duration':
            vals['worst_court_maelog'] = float(df.maelog.max())
        elif task == 'disposal':
            vals['worst_court_disposal_f1'] = float(df.macro_f1.min())
        elif task == 'case_type':
            vals['worst_court_type_f1'] = float(df.macro_f1.min())
        per_seed.append(vals)
    pd.DataFrame(details).to_csv(destination/'court_results.csv', index=False)
    ps = pd.DataFrame(per_seed)
    ps.to_csv(destination/'paper_metrics_per_seed.csv', index=False)
    algorithms = sorted(set(ps.algorithm))
    rows = []
    for algorithm in algorithms:
        task_data = {t:ps[(ps.task == t)&(ps.algorithm == algorithm)] for t in TASKS}
        seeds = sorted(set.intersection(*(set(z.seed) for z in task_data.values())))
        if not seeds:
            continue
        for seed in seeds:
            dur = task_data['duration'].set_index('seed').loc[seed]
            typ = task_data['case_type'].set_index('seed').loc[seed]
            dis = task_data['disposal'].set_index('seed').loc[seed]
            rows.append({'method': f'Independent-task {algorithm.upper()}', 'algorithm':algorithm,
                'seed':seed, 'duration_maelog':dur.maelog, 'duration_spearman':dur.get('spearman',float('nan')),
                'duration_f1_4':dur.f1_4, 'duration_mae_days':dur.mae_days,
                'case_type_macro_f1':typ.macro_f1,
                'disposal_macro_f1':dis.macro_f1,
                'worst_court_maelog':dur.worst_court_maelog,
                'worst_court_disposal_f1':dis.worst_court_disposal_f1,
                'worst_court_case_type_f1':typ.worst_court_type_f1})
    paper = pd.DataFrame(rows)
    if paper.empty:
        raise ValueError('No complete triplet of tasks with matching algorithm and seed')
    paper.to_csv(destination/'paper_table_per_seed.csv', index=False)
    cols = [c for c in paper if c not in ('method','algorithm','seed')]
    final = []
    for (method,algorithm), group in paper.groupby(['method','algorithm']):
        out = {'method':method,'algorithm':algorithm,'n_seeds':len(group)}
        for col in cols:
            xs=group[col].to_numpy(float)
            out[col+'_mean']=float(np.mean(xs))
            out[col+'_std']=float(np.std(xs,ddof=1)) if len(xs)>1 else np.nan
        final.append(out)
    pd.DataFrame(final).to_csv(destination/'paper_table_mean_std.csv',index=False)
    with (destination/'paper_table.md').open('w') as f:
        f.write('| Method | MAElog ↓ | Spearman ↑ | F1-4 ↑ | Disposal F1 ↑ | Case Type F1 ↑ | Worst MAElog ↓ | Worst Disposal F1 ↑ |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in final:
            def fmt(key):
                m = r[key+'_mean']; st = r[key+'_std']
                if not np.isfinite(m): return 'N/A'
                return f'{m:.3f} ± {st:.3f}' if np.isfinite(st) else f'{m:.3f} (1 seed)'
            f.write(f"| {r['method']} | {fmt('duration_maelog')} | {fmt('duration_spearman')} | {fmt('duration_f1_4')} | {fmt('disposal_macro_f1')} | {fmt('case_type_macro_f1')} | {fmt('worst_court_maelog')} | {fmt('worst_court_disposal_f1')} |\n")
        f.write('\nValues are newly evaluated baseline runs, not reported published FEDLEGAL scores.\n')
    print((destination/'paper_table.md').read_text())
    return paper

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runs',default='outputs')
    p.add_argument('--out',default='outputs/paper_report')
    a=p.parse_args()
    create_tables(a.runs,a.out)
