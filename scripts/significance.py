"""
One headline number per method + significance tests.

1) Delta_MTL (Maninis et al. 2019; Vandenhende et al. 2021): the average relative improvement over
   a reference method across the three tasks, with one primary metric per task:
       duration  -> MAE_log   (lower is better)
       case type -> macro-F1
       disposal  -> macro-F1
       Delta_MTL = 100/T * sum_t s_t (M_t - M_ref,t) / M_ref,t,   s_t = -1 if lower is better else +1
   Reported for the mean over courts and for the WORST court (fairness across clients).

2) Paired tests over (court, seed) pairs from runs/all_results.csv: Wilcoxon signed-rank and paired
   t-test on each primary metric, and the number of courts where A beats B.

3) Case-level paired bootstrap from two runs' preds.csv (fl_judicial.py --save_preds): 95% CI of the
   metric difference, resampling test cases within each court.

Usage
  python scripts/significance.py --summary results/main_results.csv --ref "Local only"
  python scripts/significance.py --all runs/all_results.csv --a fedavg+ft --b local_only
  python scripts/significance.py --preds_a runs/X/preds.csv --preds_b runs/Y/preds.csv [--method_a .. --method_b ..]
"""
import argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

PRIMARY = [("MAE_log", -1), ("type_F1", +1), ("disp_F1", +1)]

def delta_mtl(row, ref, cols=PRIMARY):
    return 100 * np.nanmean([s * (row[m] - ref[m]) / ref[m] for m, s in cols])

def from_summary(path, ref_name):
    d = pd.read_csv(path)
    ref = d[d.method == ref_name].iloc[0]
    d["Delta_MTL_%"] = d.apply(lambda r: delta_mtl(r, ref), axis=1)
    for m, s in PRIMARY: d[f"rel_{m}_%"] = 100 * s * (d[m] - ref[m]) / ref[m]
    return d[["group", "method", "seeds", "MAE_log", "type_F1", "disp_F1", "Delta_MTL_%",
              "rel_MAE_log_%", "rel_type_F1_%", "rel_disp_F1_%"]]

def pick(d, method, run_filter):
    x = d[d.method == method]
    if run_filter: x = x[x.run.str.contains(run_filter, regex=False)]
    if x.duplicated(["court", "seed"]).any():
        raise SystemExit(f"method '{method}' matches several runs {sorted(x.run.unique())}; narrow it with --run_a/--run_b")
    return x.set_index(["court", "seed"])

def from_all(path, a, b, ra=None, rb=None):
    d = pd.read_csv(path)
    A = pick(d, a, ra); B = pick(d, b, rb)
    idx = A.index.intersection(B.index); A, B = A.loc[idx], B.loc[idx]
    out = []
    for m, s in PRIMARY:
        diff = s * (A[m] - B[m])                                     # > 0 means A is better
        w = stats.wilcoxon(diff) if len(diff) >= 6 and diff.abs().sum() > 0 else None
        t = stats.ttest_rel(A[m], B[m])
        court_mean = diff.groupby(level=0).mean()
        out.append({"metric": m, "pairs": len(diff), "mean_gain_A_over_B": diff.mean(),
                    "wilcoxon_p": w.pvalue if w else np.nan, "paired_t_p": t.pvalue,
                    "courts_A_better": f"{int((court_mean > 0).sum())}/{len(court_mean)}"})
    # Delta_MTL per (court, seed), A vs B as reference, then its own test
    dm = np.array([delta_mtl(A.loc[i], B.loc[i]) for i in idx]); dm = dm[~np.isnan(dm)]
    out.append({"metric": "Delta_MTL_%", "pairs": len(dm), "mean_gain_A_over_B": dm.mean(),
                "wilcoxon_p": stats.wilcoxon(dm).pvalue if len(dm) >= 6 else np.nan,
                "paired_t_p": stats.ttest_1samp(dm, 0).pvalue, "courts_A_better": ""})
    return pd.DataFrame(out)

def case_metrics(p):
    lab = p.y_disp >= 0
    return {"MAE_log": np.abs(p.y_dur - p.p_dur).mean(),
            "type_F1": f1_score(p.y_type, p.p_type, average="macro"),
            "disp_F1": f1_score(p.y_disp[lab], p.p_disp[lab], average="macro", labels=[0, 1, 2, 3]) if lab.any() else np.nan}

def bootstrap(pa, pb, B=1000, seed=0):
    """Paired: both runs predict the same test cases (same cnr), resampled jointly within court.
    Metrics are averaged over courts, as in the report."""
    rng = np.random.default_rng(seed); key = ["court", "cnr"]
    m = pa.merge(pb, on=key, suffixes=("_a", "_b"))
    get = lambda x, sfx: x[[c for c in x.columns if c.endswith(sfx)]].rename(columns=lambda c: c[:-2])
    def score(x):
        A = np.nanmean([list(case_metrics(get(g, "_a")).values()) for _, g in x.groupby("court")], 0)
        Bm = np.nanmean([list(case_metrics(get(g, "_b")).values()) for _, g in x.groupby("court")], 0)
        return A, Bm
    A0, B0 = score(m); groups = [g for _, g in m.groupby("court")]; diffs = []
    for _ in range(B):
        x = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])
        A, Bm = score(x); diffs.append(A - Bm)
    diffs = np.array(diffs); rows = []
    for i, (name, s) in enumerate(PRIMARY):
        lo, hi = np.percentile(diffs[:, i], [2.5, 97.5])
        rows.append({"metric": name, "A": A0[i], "B": B0[i], "A_minus_B": A0[i] - B0[i], "CI95_low": lo, "CI95_high": hi,
                     "P(A not better)": float(np.mean(s * diffs[:, i] <= 0))})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary"); ap.add_argument("--ref", default="Local only")
    ap.add_argument("--all"); ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--run_a", help="substring of the run name, if the method appears in several runs")
    ap.add_argument("--run_b")
    ap.add_argument("--preds_a"); ap.add_argument("--preds_b"); ap.add_argument("--method_a"); ap.add_argument("--method_b")
    ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--out")
    a = ap.parse_args(); pd.set_option("display.width", 200)
    if a.summary: r = from_summary(a.summary, a.ref)
    elif a.all: r = from_all(a.all, a.a, a.b, a.run_a, a.run_b)
    else:
        pa, pb = pd.read_csv(a.preds_a), pd.read_csv(a.preds_b)
        if a.method_a: pa = pa[pa.method == a.method_a]
        if a.method_b: pb = pb[pb.method == a.method_b]
        r = bootstrap(pa, pb, a.boot)
    print(r.round(4).to_string(index=False))
    if a.out: r.to_csv(a.out, index=False)

if __name__ == "__main__":
    main()
