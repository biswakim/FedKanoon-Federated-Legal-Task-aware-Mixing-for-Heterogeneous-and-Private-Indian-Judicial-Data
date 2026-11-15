"""
Builds the revision tables from runs/all_results.csv (+ cross_eval.csv) after
`run_queue.py --plan revision`. Writes CSVs to --out and prints them.

  baselines.csv  every method/config: primary metrics, worst court, Delta_MTL vs local-only TSR
  lowres.csv     10/25/50% data per court: local vs federated vs centralized
  unseen.csv     leave-one-court-out: other courts' local models vs FedAvg/centralized without the court
  control512.csv single-view 512 tokens vs TSR

  python scripts/revision_tables.py --runs runs --out results/revision
"""
import argparse, os, re
import numpy as np, pandas as pd

PRIMARY = [("MAE_log", -1), ("type_F1", +1), ("disp_F1", +1)]

def delta(m, ref):
    return 100 * np.nanmean([s * (m[k] - ref[k]) / ref[k] for k, s in PRIMARY])

def config(run):
    return re.sub(r"_s\d+$", "", run)

def summarise(r):
    """mean over seeds of the court-mean, worst court (on seed-mean), and seed std of MAE_log."""
    rows = []
    for (cfg, method), g in r.groupby(["cfg", "method"]):
        per_seed = g.groupby("seed")[[k for k, _ in PRIMARY]].mean()
        per_court = g.groupby("court")[[k for k, _ in PRIMARY]].mean()
        rows.append({"cfg": cfg, "method": method, "seeds": g.seed.nunique(),
                     **per_seed.mean().to_dict(), "MAE_log_std": per_seed.MAE_log.std(),
                     "worst_MAE_log": per_court.MAE_log.max(), "worst_disp_F1": per_court.disp_F1.min(),
                     "_per_court": per_court})
    return pd.DataFrame(rows)

def worst_delta(pc, ref_pc):
    """Delta_MTL per court against the reference's same court; return the worst court's value."""
    common = pc.index.intersection(ref_pc.index)
    return min(delta(pc.loc[c], ref_pc.loc[c]) for c in common) if len(common) else np.nan

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="results/revision")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); pd.set_option("display.width", 220)
    r = pd.read_csv(os.path.join(a.runs, "all_results.csv")); r["cfg"] = r.run.map(config)
    S = summarise(r)
    full = ~S.cfg.str.contains("_cf-|_ho-")
    tsr = S.cfg.str.contains("_seg2")

    # ---- baselines (full data, TSR model), reference = local-only TSR
    ref = S[full & tsr & (S.method == "local_only")]
    out = {}
    if len(ref):
        ref = ref.sort_values("seeds").iloc[-1]
        B = S[full].copy()
        B["Delta_MTL_%"] = B.apply(lambda x: delta(x, ref), axis=1)
        B["worst_court_Delta_MTL_%"] = B._per_court.map(lambda pc: worst_delta(pc, ref._per_court))
        out["baselines"] = B.drop(columns="_per_court").sort_values("Delta_MTL_%", ascending=False)

    # ---- low-resource: same fraction, local vs federated vs centralized
    rows = []
    for frac in sorted(set(re.findall(r"_cf-[a-z]+([\d.]+)", " ".join(S.cfg)))):
        sub = S[S.cfg.str.contains(f"0{frac[1:]}" if frac.startswith("0") else frac, regex=False) & S.cfg.str.contains("_cf-")]
        loc = sub[sub.method == "local_only"]
        if loc.empty: continue
        loc = loc.iloc[0]
        for _, x in sub.iterrows():
            rows.append({"frac": float(frac), "method": x.method, "cfg": x.cfg, "MAE_log": x.MAE_log, "type_F1": x.type_F1,
                         "disp_F1": x.disp_F1, "Delta_MTL_vs_local_%": delta(x, loc)})
    if rows: out["lowres"] = pd.DataFrame(rows).sort_values(["frac", "Delta_MTL_vs_local_%"], ascending=[True, False])

    # ---- unseen court: held-out court's score under each option, relative to its own local model
    ce = [os.path.join(a.runs, d, "cross_eval.csv") for d in os.listdir(a.runs) if os.path.exists(os.path.join(a.runs, d, "cross_eval.csv"))]
    if ce:
        X = pd.read_csv(ce[0]); rows = []
        for c in sorted(X.test_court.unique()):
            own = X[(X.train_court == c) & (X.test_court == c)].iloc[0]
            oth = X[(X.train_court != c) & (X.test_court == c)]
            d_oth = oth.apply(lambda x: delta(x, own), axis=1)
            row = {"held_out": c, "own_local_MAE_log": own.MAE_log,
                   "best_other_local_Delta_%": d_oth.max(), "mean_other_local_Delta_%": d_oth.mean()}
            for kind, pat in [("fedavg", "fl_fedavg"), ("central", "central")]:
                h = r[r.run.str.startswith(pat) & r.run.str.contains(f"_ho-{c}_") & (r.court == c) & (r.method != "fedavg+ft")]
                if len(h):
                    m = h[[k for k, _ in PRIMARY]].mean()
                    row[f"{kind}_without_court_Delta_%"] = delta(m, own); row[f"{kind}_MAE_log"] = m.MAE_log
            rows.append(row)
        out["unseen"] = pd.DataFrame(rows)

    # ---- 512-token single-view control
    c512 = S[full & S.cfg.str.contains("_L512_")]
    if len(c512): out["control512"] = c512.drop(columns="_per_court")

    for k, v in out.items():
        v.to_csv(os.path.join(a.out, f"{k}.csv"), index=False)
        print(f"\n===== {k} =====\n" + v.round(4).to_string(index=False))
    if not out: print("nothing to summarise yet")

if __name__ == "__main__":
    main()
