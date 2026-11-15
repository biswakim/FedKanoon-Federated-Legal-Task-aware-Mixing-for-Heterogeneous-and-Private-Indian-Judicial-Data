"""
Statistical evidence that the five High Court clients are non-IID.

Answers "how do you know the data are heterogeneous?" with hypothesis tests and effect sizes
for each of the three standard kinds of FL skew (Kairouz et al. 2021, Sec. 3.1):

  quantity skew    n_k differs                     -> counts, max/min ratio
  label skew       P_k(y) differs                  -> chi-square + Cramer's V, pairwise Jensen-Shannon
                                                      divergence, Kruskal-Wallis + epsilon^2 for duration,
                                                      pairwise Wasserstein / KS on log-duration,
                                                      and the EQUIVALENT DIRICHLET alpha of the natural split
  concept shift    P_k(y | x) differs              -> two-way ANOVA duration ~ family * court (interaction),
                                                      stratified chi-square disposal ~ court | family,
                                                      cross-court transfer matrix (train on court i, test on j)
  covariate shift  P_k(x) differs                  -> classifier two-sample test: can a linear model tell
                                                      which court a MASKED judgment opening comes from?

Two input modes
  --meta_dir DIR   the raw eCourts metadata parquet files (AWS Open Data, year=2024) for the five courts.
                   Uses the ~300-character `description` (start of the judgment) as text.
  --data CSV       the project's judgments_labelled.csv (same columns as fl_judicial.py). Uses
                   case_type_group / outcome_from_text / input_text directly. Prefer this for the report,
                   because it is exactly the sample the models were trained on.

Usage
  python scripts/heterogeneity_stats.py --meta_dir /path/to/parquets --out results/heterogeneity
  python scripts/heterogeneity_stats.py --data data/judgments_labelled.csv --out results/heterogeneity_text
"""
import argparse, glob, os, re, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

COURT_CODES = {"27~1": "bombay", "7~26": "delhi", "29~3": "karnataka", "32~4": "kerala", "23~23": "madhya_pradesh"}
SHORT = {"bombay": "BOM", "delhi": "DEL", "karnataka": "KAR", "kerala": "KER", "madhya_pradesh": "MP"}
DISP = ["ALLOWED", "DISMISSED", "DISPOSED", "WITHDRAWN"]

# ----------------------------------------------------------------------------- labels from metadata
FAMILY_RULES = [   # first match wins; codes are upper-cased with spaces, dots, brackets and dashes removed
    ("contempt",   r"^(CONC|CONCR|CONT|CONTCAS|CONCASE|CCC)"),
    ("bail",       r"^(BA|ABA|BAILAPPL|BAILAPPLN|MCRC)$"),
    ("motor_acc",  r"^(MACA|MACAPP|MAC|MACMA)"),
    ("arbitration", r"^(ARB|OMP|AA|AC|ARP|COMAP|COMMAP|FAOOSCOMM|CSCOMM)"),
    ("tax",        r"^(ITA|STA|TA|CEA|GSTA|CUSAA|WTA|STR)"),
    ("review",     r"^(RP|REVIEW|RVW|REVPET)$"),
    ("writ",       r"^(WP|WA|LPA|WPS|WPC|WPCRL|WPHC|WPPIL|PIL)"),
    ("criminal",   r"^(CR|CRA|CRR|CRL|APEAL|APL|ALP|REVN|APPA|APPP|APPR|OPCRL|CRLMC|CRLP)"),
    ("civil",      r"^(CA|FA|SA|RSA|RFA|MFA|MA|MSA|CRP|AO|CP|CS|CMP|CMM|RPFC|RCREV|OP|MATAPP|LAAPP|TRPC|EXA|MCA|CAF|CAW|CAS|CAO|IA|APPLN|MP|MCC)"),
]
def family(code):
    for fam, rx in FAMILY_RULES:
        if re.match(rx, code or ""): return fam
    return "other"

def disposal(s):
    s = (s or "").upper()
    if not s.strip(): return None
    if re.search(r"WITHDRAW|NOT PRESSED", s): return "WITHDRAWN"
    if re.search(r"DISMISS|REJECT", s): return "DISMISSED"
    if re.search(r"ALLOW|GRANT|ADMITTED|ABSOLUTE", s): return "ALLOWED"
    if re.search(r"DISPOS|CLOSED|INFRUCTUOUS", s): return "DISPOSED"
    return None

def parse_reg(s):
    s = pd.Series(s, dtype="string")
    a = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    b = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
    r = a.fillna(b)
    return r.where((r.dt.year >= 1900) & (r.dt.year <= 2025)).astype("datetime64[ns]")

def load_meta(meta_dir):
    fs = sorted(glob.glob(os.path.join(meta_dir, "*.parquet")))
    cols = ["court_code", "title", "description", "cnr", "date_of_registration", "decision_date", "disposal_nature"]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)
    d["court"] = d.court_code.map(COURT_CODES); d = d[d.court.notna()]
    d["decision_date"] = pd.to_datetime(d.decision_date, errors="coerce").astype("datetime64[ns]")
    d = d.sort_values("decision_date").drop_duplicates("cnr", keep="last")          # final order per case
    reg = parse_reg(d.date_of_registration.values); reg.index = d.index
    d["duration_days"] = (d.decision_date - reg).dt.days
    d = d[(d.duration_days >= 0) & (reg.dt.year >= 1980)]
    code = d.title.str.extract(r"^\s*([A-Za-z.()\-\s]+?)\s*/", expand=False).fillna("")
    d["case_type_raw"] = code.str.upper().str.replace(r"[\s.()\-]", "", regex=True)
    d["case_type_group"] = d.case_type_raw.map(family)
    d["disp"] = d.disposal_nature.map(disposal)
    d["text"] = d.description.fillna("")
    return d.reset_index(drop=True)

def load_csv(path):
    d = pd.read_csv(path, low_memory=False)
    if "garbled" in d: d = d[~d.garbled.astype(bool)]
    d["disp"] = d.outcome_from_text.where(d.outcome_from_text.isin(DISP))
    d["text"] = d.input_text.fillna("").astype(str)
    d["case_type_group"] = d.case_type_group.fillna("other")
    return d.reset_index(drop=True)

# ----------------------------------------------------------------------------- masking for the text tests
PLACES = (r"bombay|mumbai|nagpur|aurangabad|goa|panaji|kolhapur|maharashtra|delhi|new delhi|karnataka|bengaluru|"
          r"bangalore|dharwad|kalaburagi|gulbarga|kerala|ernakulam|kochi|thiruvananthapuram|madhya pradesh|jabalpur|"
          r"indore|gwalior|bhopal|m\.p\.|mp")
def mask(t):
    t = t.lower()
    t = re.sub(r"\b(" + PLACES + r")\b", " PLACE ", t)
    t = re.sub(r"\d+", " NUM ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# ----------------------------------------------------------------------------- statistics
def cramers_v(tab):
    tab = tab.loc[tab.sum(1) > 0, tab.sum(0) > 0]
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    n = tab.values.sum(); r, k = tab.shape
    return chi2, p, dof, float(np.sqrt(chi2 / (n * (min(r, k) - 1))))

def pairwise_jsd(tab):
    P = tab.div(tab.sum(1), axis=0)
    M = pd.DataFrame(index=P.index, columns=P.index, dtype=float)
    for a in P.index:
        for b in P.index: M.loc[a, b] = jensenshannon(P.loc[a], P.loc[b], base=2) ** 2
    return M

def equivalent_dirichlet_alpha(tab, sims=400, seed=0):
    """Hsu et al. (2019) build non-IID FL benchmarks by drawing each client's label mix from
    Dir(alpha * p_global). We find the alpha whose simulated clients are, on average, as far from
    the global label distribution (mean JSD) as the real courts are. Small alpha = strong skew;
    alpha <= 0.5 is the usual 'highly non-IID' setting in FL papers, alpha = 100 is close to IID."""
    rng = np.random.default_rng(seed)
    p = tab.sum(0).values.astype(float); p /= p.sum()
    P = tab.div(tab.sum(1), axis=0).values
    obs = np.mean([jensenshannon(r, p, base=2) ** 2 for r in P])
    K = len(P)
    def sim(a):
        out = []
        for _ in range(sims):
            Q = rng.dirichlet(a * p + 1e-12, size=K)
            out.append(np.mean([jensenshannon(q, p, base=2) ** 2 for q in Q]))
        return np.mean(out)
    grid = np.logspace(-2, 3, 101); vals = np.array([sim(a) for a in grid])   # decreasing in alpha
    i = int(np.argmin(np.abs(vals - obs)))
    return float(obs), float(grid[i]), pd.DataFrame({"alpha": grid, "mean_JSD": vals})

def label_skew(d, col, name, out):
    tab = pd.crosstab(d.court, d[col])
    chi2, p, dof, v = cramers_v(tab)
    J = pairwise_jsd(tab); J.to_csv(os.path.join(out, f"jsd_{name}.csv"))
    obs, alpha, curve = equivalent_dirichlet_alpha(tab)
    tab.div(tab.sum(1), axis=0).round(4).to_csv(os.path.join(out, f"dist_{name}.csv"))
    return {"variable": name, "test": "chi-square (court x label)", "stat": chi2, "dof": dof, "p": p,
            "effect": "Cramer's V", "effect_size": v,
            "mean_pairwise_JSD": float(J.values[np.triu_indices(len(J), 1)].mean()),
            "max_pairwise_JSD": float(J.values.max()), "equiv_dirichlet_alpha": alpha,
            "equiv_alpha_per_class": alpha / tab.shape[1]}   # comparable to the common Dir(alpha * 1) splits

def duration_tests(d, out):
    y = np.log1p(d.duration_days.astype(float)); g = d.court
    groups = [y[g == c] for c in sorted(g.unique())]
    H, p = stats.kruskal(*groups); n = len(y); k = len(groups)
    eps2 = (H - k + 1) / (n - k)                                      # Kruskal-Wallis epsilon^2 (Tomczak 2014)
    rows = []
    cs = sorted(g.unique())
    for i, a in enumerate(cs):
        for b in cs[i + 1:]:
            ks = stats.ks_2samp(y[g == a], y[g == b])
            rows.append({"a": a, "b": b, "KS_D": ks.statistic, "KS_p": ks.pvalue,
                         "Wasserstein_logdays": stats.wasserstein_distance(y[g == a], y[g == b])})
    pd.DataFrame(rows).to_csv(os.path.join(out, "pairwise_duration.csv"), index=False)
    desc = d.assign(y=y).groupby("court").duration_days.describe(percentiles=[.25, .5, .75])
    desc.to_csv(os.path.join(out, "duration_by_court.csv"))
    return {"variable": "log_duration", "test": "Kruskal-Wallis", "stat": H, "dof": k - 1, "p": p,
            "effect": "epsilon^2", "effect_size": eps2,
            "mean_pairwise_KS_D": float(np.mean([r["KS_D"] for r in rows])),
            "mean_pairwise_W1": float(np.mean([r["Wasserstein_logdays"] for r in rows]))}

def concept_shift_duration(d, out):
    """Same case family, different court -> different duration? Two-way ANOVA (type II)."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    x = d.assign(y=np.log1p(d.duration_days.astype(float)), fam=d.case_type_group)
    cell = x.groupby(["fam", "court"]).size().unstack(fill_value=0)       # keep families with >= 30 cases
    fams = cell.index[(cell >= 30).all(1)]                                  # in EVERY court (full-rank design)
    x = x[x.fam.isin(fams)]
    m = smf.ols("y ~ C(court) * C(fam)", data=x).fit()
    a = anova_lm(m, typ=2); ss_res = a.loc["Residual", "sum_sq"]
    a["partial_eta2"] = a.sum_sq / (a.sum_sq + ss_res)
    a.to_csv(os.path.join(out, "anova_duration_court_x_family.csv"))
    r = a.loc["C(court):C(fam)"]
    m_add = smf.ols("y ~ C(fam)", data=x).fit(); m_full = m
    return {"variable": "log_duration | family", "test": "two-way ANOVA, court x family interaction",
            "stat": float(r.F), "dof": int(r.df), "p": float(r["PR(>F)"]),
            "effect": "partial eta^2 (interaction)", "effect_size": float(r.partial_eta2),
            "R2_family_only": float(m_add.rsquared), "R2_family_x_court": float(m_full.rsquared)}

def concept_shift_disposal(d, out):
    """Disposal ~ court within each case family (stratified chi-square; summed statistic)."""
    x = d[d.disp.notna()]
    rows, chi_tot, dof_tot = [], 0.0, 0
    for fam, g in x.groupby("case_type_group"):
        tab = pd.crosstab(g.court, g.disp)
        tab = tab.loc[tab.sum(1) >= 30]
        if tab.shape[0] < 2 or tab.shape[1] < 2: continue
        chi2, p, dof, v = cramers_v(tab); chi_tot += chi2; dof_tot += dof
        rows.append({"family": fam, "n": int(tab.values.sum()), "courts": tab.shape[0], "chi2": chi2, "p": p, "cramers_v": v})
    pd.DataFrame(rows).to_csv(os.path.join(out, "disposal_given_family.csv"), index=False)
    return {"variable": "disposal | family", "test": "stratified chi-square (sum over families)",
            "stat": chi_tot, "dof": dof_tot, "p": float(stats.chi2.sf(chi_tot, dof_tot)),
            "effect": "median Cramer's V within family", "effect_size": float(np.median([r["cramers_v"] for r in rows])),
            "families_tested": len(rows)}

def covariate_shift(d, out, per_court=2000, perms=20, seed=0):
    """Classifier two-sample test (Lopez-Paz & Oquab 2017): court identification from masked text."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    s = pd.concat([g.sample(min(per_court, len(g)), random_state=seed) for _, g in d.groupby("court")])
    X = TfidfVectorizer(min_df=3, ngram_range=(1, 2), max_features=50000, sublinear_tf=True).fit_transform(s.text.map(mask))
    y = s.court.to_numpy(dtype=object).astype(str); cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    pred = cross_val_predict(clf, X, y, cv=cv); acc = balanced_accuracy_score(y, pred)
    rng = np.random.default_rng(seed); null = []
    for _ in range(perms):
        yp = rng.permutation(y); null.append(balanced_accuracy_score(yp, cross_val_predict(clf, X, yp, cv=cv)))
    p_perm = (1 + sum(n >= acc for n in null)) / (1 + perms)
    k = int((pred == y).sum()); p = stats.binomtest(k, len(y), 1 / len(np.unique(y)), alternative="greater").pvalue
    pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")).to_csv(os.path.join(out, "court_id_confusion.csv"))
    # what gives a court away (top features), to show these are procedural/legal terms, not names
    clf.fit(X, y); vocab = np.array(TfidfVectorizer(min_df=3, ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
                                    .fit(s.text.map(mask)).get_feature_names_out())
    top = {c: list(vocab[np.argsort(-clf.coef_[i])[:15]]) for i, c in enumerate(clf.classes_)}
    json.dump(top, open(os.path.join(out, "court_id_top_features.json"), "w"), indent=1)
    return {"variable": "masked judgment text", "test": "classifier two-sample test (court ID, 5-fold)",
            "stat": acc, "dof": None, "p": p, "effect": "balanced accuracy (chance = 1/K)",
            "effect_size": acc, "chance": 1 / len(np.unique(y)), "permutations": perms,
            "perm_p": p_perm, "perm_null_max": float(max(null))}

def transfer_matrix(d, out, per_court=3000, seed=0):
    """P(y|x) shift as it matters for learning: a model trained on court i, tested on court j."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import f1_score
    s = pd.concat([g.sample(min(per_court, len(g)), random_state=seed) for _, g in d.groupby("court")]).copy()
    s["court"] = s.court.astype(str); s["disp"] = s.disp.astype(object); s["t"] = s.text.map(mask); s["y"] = np.log1p(s.duration_days.astype(float))
    rng = np.random.default_rng(seed); s["tr"] = rng.random(len(s)) < 0.7
    vec = TfidfVectorizer(min_df=3, ngram_range=(1, 2), max_features=50000, sublinear_tf=True).fit(s.t[s.tr])
    cs = sorted(s.court.unique()); mae = pd.DataFrame(index=cs, columns=cs, dtype=float); f1 = mae.copy()
    for a in cs:
        tr = s[(s.court == a) & s.tr]
        r = Ridge(alpha=3.0).fit(vec.transform(tr.t), tr.y)
        trd = tr[tr.disp.notna()]
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(vec.transform(trd.t), trd.disp) \
            if trd.disp.nunique() > 1 and len(trd) > 100 else None
        for b in cs:
            te = s[(s.court == b) & ~s.tr]
            mae.loc[a, b] = np.abs(r.predict(vec.transform(te.t)) - te.y).mean()
            ted = te[te.disp.notna()]
            if lr is not None and len(ted) > 50:
                f1.loc[a, b] = f1_score(ted.disp, lr.predict(vec.transform(ted.t)), average="macro", labels=DISP)
    mae.to_csv(os.path.join(out, "transfer_MAElog.csv")); f1.to_csv(os.path.join(out, "transfer_dispF1.csv"))
    off = lambda M: np.nanmean(M.values[~np.eye(len(M), dtype=bool)])
    return {"variable": "transfer (TF-IDF linear)", "test": "train court i -> test court j",
            "diag_MAElog": float(np.nanmean(np.diag(mae.values))), "offdiag_MAElog": off(mae),
            "diag_dispF1": float(np.nanmean(np.diag(f1.values))), "offdiag_dispF1": off(f1)}

def plots(d, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for c, g in d.groupby("court"):
        ax[0].hist(np.log1p(g.duration_days), bins=60, density=True, histtype="step", lw=1.5, label=SHORT.get(c, c))
    ax[0].set_xlabel("log(1 + days)"); ax[0].set_title("Duration distribution per court"); ax[0].legend()
    for i, (col, title) in enumerate([("case_type_group", "Case family"), ("disp", "Disposal nature")]):
        t = pd.crosstab(d.court, d[col]); t = t.div(t.sum(1), axis=0); t.index = [SHORT.get(c, c) for c in t.index]
        left = np.zeros(len(t))
        for k in t.columns:
            ax[i + 1].barh(t.index, t[k], left=left, label=k); left += t[k].values
        ax[i + 1].set_title(f"{title} mix per court"); ax[i + 1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_heterogeneity.png"), dpi=160); fig.savefig(os.path.join(out, "fig_heterogeneity.pdf"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir"); ap.add_argument("--data"); ap.add_argument("--out", default="results/heterogeneity")
    ap.add_argument("--skip_text", action="store_true"); ap.add_argument("--perms", type=int, default=20)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    d = load_meta(a.meta_dir) if a.meta_dir else load_csv(a.data)
    d["dur_bucket"] = pd.cut(d.duration_days, [-1, 30, 180, 730, np.inf], labels=["<=30d", "31-180d", "181-730d", ">730d"])
    counts = d.court.value_counts().sort_index()
    print("cases per court:", counts.to_dict(), "| max/min =", round(counts.max() / counts.min(), 2))
    rows = [{"variable": "n_k", "test": "quantity skew", "stat": float(counts.max() / counts.min()),
             "effect": "max/min client size", "effect_size": float(counts.std() / counts.mean())}]
    rows.append(duration_tests(d, a.out))
    rows.append(label_skew(d, "dur_bucket", "duration_bucket", a.out))
    rows.append(label_skew(d, "case_type_group", "case_family", a.out))
    dd = d[d.disp.notna()]
    if dd.court.nunique() >= 2: rows.append(label_skew(dd, "disp", "disposal", a.out))
    rows.append(concept_shift_duration(d, a.out))
    if dd.court.nunique() >= 2: rows.append(concept_shift_disposal(d, a.out))
    if not a.skip_text:
        rows.append(covariate_shift(d, a.out, perms=a.perms))
        rows.append(transfer_matrix(d, a.out))
    plots(d, a.out)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(a.out, "heterogeneity_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(R.to_string(index=False))
    print("\ndisposal labels available per court:", dd.court.value_counts().to_dict())

if __name__ == "__main__":
    main()
