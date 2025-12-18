# ============================================================
# ✅ MAIN (recommended for paper):
#   - Repeated Stratified CV (uncertainty) for LR(isotonic), RF(isotonic)
#   - Condition-blocked CV (deployment shift) for LR(isotonic), RF(isotonic)
#   - Ablation (LR): Geometry-only / Visibility-only / Combined
#   - Calibration quality: Brier + ECE (+ optional reliability plots)
#   - Subgroup + worst-case reporting (contrast/medium/dist_bin/ang_bin)
#   - Mondrian conformal by subgroup (coverage per group) — deployment-style split by condition_group
#   - Robustness stress tests (bias/clipping/quantization)
#   - Cross-fitted residual sufficiency test
#   - CMI-lite
#
# ✅ PAPER OUTPUTS:
#   - CSVs under ./outputs
#   - LaTeX tables under ./outputs/tables_IEEE.tex (ready to paste)
# ============================================================

import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef
)
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

# -------------------------
# MODE: "final" for paper, "quick" for sanity-check
# -------------------------
MODE = "final"  # "quick" or "final"

RANDOM_STATE = 42
DATA_PATH = "reading_visibility_dataset_300rows.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

if MODE == "quick":
    N_REPEATS = 50
    N_SPLITS = 5
    BOOTSTRAP_ROUNDS = 800
    RESID_REPEATS = 30
    RESID_PERM = 300
else:
    N_REPEATS = 200   # paper-ready; (400 is ok but often unnecessary)
    N_SPLITS = 5
    BOOTSTRAP_ROUNDS = 2000
    RESID_REPEATS = 80
    RESID_PERM = 500

DIST_BINS = 4
ANG_BINS = 4

ALPHA = 0.10
MONDRIAN_GROUP_COL = "contrast"  # or "medium"

TARGET = "can_read"

GEOMETRY = ["distance_m", "font_size_pt", "text_height_mm", "angular_size_deg"]
HEADPOSE = ["head_yaw_deg", "head_pitch_deg", "head_roll_deg"]
VIS = ["visibility_score"]
CAT = ["medium", "contrast"]

FULL_X = GEOMETRY + HEADPOSE + VIS + CAT
VIS_ONLY_X = VIS
GEOM_ONLY_X = GEOMETRY + HEADPOSE + CAT
VIS_PLUS_GEOM_X = GEOMETRY + HEADPOSE + VIS + CAT

# ============================================================
# Stage timer (paper-friendly logs)
# ============================================================
_T0 = time.time()

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

class StageTimer:
    def __init__(self):
        self.records = []

    def start(self, name: str):
        print(f"\n[{_ts()}] {name} ...")
        self.records.append({"name": name, "t0": time.time(), "dur": None})

    def end(self):
        self.records[-1]["dur"] = time.time() - self.records[-1]["t0"]
        print(f"[{_ts()}] done. (wall: {time.time()-_T0:,.1f}s)")

    def summary(self):
        total = sum(r["dur"] for r in self.records if r["dur"] is not None)
        print("\n" + "="*88)
        print("STAGE TIME SUMMARY")
        print("="*88)
        for r in self.records:
            dur = r["dur"]
            pct = (dur / total * 100.0) if total > 0 else 0.0
            print(f"- {r['name']:<66} {dur:>10.1f}s  ({pct:>6.1f}%)")
        print("-"*88)
        print(f"TOTAL (stages):{'':<57} {total:>10.1f}s")
        print(f"TOTAL (wall):{'':<59} {time.time()-_T0:>10.1f}s")
        print("="*88)

ST = StageTimer()

# ============================================================
# Metrics / helpers
# ============================================================
def safe_auc(y_true, y_prob) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)

def ece_score(y_true, y_prob, n_bins=15) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & ((y_prob < hi) if i < n_bins - 1 else (y_prob <= hi))
        if mask.sum() == 0:
            continue
        acc_bin = y_true[mask].mean()
        conf_bin = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc_bin - conf_bin)
    return float(ece)

def compute_metrics(y_true, y_prob, threshold=0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": safe_auc(y_true, y_prob),
        "ap": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "brier": brier_score_loss(y_true, y_prob),
        "ece": ece_score(y_true, y_prob, n_bins=15),
        "acc": accuracy_score(y_true, y_pred),
        "bacc": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
    }

def make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if num_cols:
        transformers.append(("num", num_tf, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_tf, cat_cols))
    return ColumnTransformer(transformers)

def make_lr_pipeline(X_cols: List[str], calibrate: Optional[str] = "isotonic"):
    num_cols = [c for c in X_cols if c not in CAT]
    cat_cols = [c for c in X_cols if c in CAT]
    pre = make_preprocessor(num_cols, cat_cols)

    base = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=8000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    if calibrate is None:
        return base
    method = "isotonic" if calibrate == "isotonic" else "sigmoid"
    return CalibratedClassifierCV(base, method=method, cv=3)

def make_rf_pipeline(X_cols: List[str], calibrate: Optional[str] = "isotonic"):
    num_cols = [c for c in X_cols if c not in CAT]
    cat_cols = [c for c in X_cols if c in CAT]
    pre = make_preprocessor(num_cols, cat_cols)

    base = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=800,           # stable, not crazy
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
        )),
    ])
    if calibrate is None:
        return base
    method = "isotonic" if calibrate == "isotonic" else "sigmoid"
    return CalibratedClassifierCV(base, method=method, cv=3)

def qcut_bins(series: pd.Series, q: int, fallback_bins: int = 4) -> pd.Series:
    try:
        return pd.qcut(series, q=q, labels=False, duplicates="drop")
    except Exception:
        return pd.cut(series, bins=fallback_bins, labels=False)

def build_condition_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dist_bin"] = qcut_bins(out["distance_m"], q=DIST_BINS).astype(int)
    out["ang_bin"] = qcut_bins(out["angular_size_deg"], q=ANG_BINS).astype(int)
    out["condition_group"] = (
        out["medium"].astype(str) + "|" +
        out["contrast"].astype(str) + "|" +
        out["dist_bin"].astype(str) + "|" +
        out["ang_bin"].astype(str)
    )
    return out

# ============================================================
# Paper tables (bootstrap CI over CV splits)
# ============================================================
def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"mean": np.nan, "p2.5": np.nan, "p97.5": np.nan, "n": 0}
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(v, size=len(v), replace=True)
        boots.append(np.mean(samp))
    boots = np.array(boots)
    return {
        "mean": float(np.mean(v)),
        "p2.5": float(np.quantile(boots, 0.025)),
        "p97.5": float(np.quantile(boots, 0.975)),
        "n": int(len(v)),
    }

def summarize_cv(dfm: pd.DataFrame, group_cols: List[str], metrics: List[str], title_tag: str) -> pd.DataFrame:
    rows = []
    for keys, sub in dfm.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        keyd = {group_cols[i]: keys[i] for i in range(len(group_cols))}
        for met in metrics:
            ci = bootstrap_ci(sub[met].to_numpy(), n_boot=BOOTSTRAP_ROUNDS, seed=RANDOM_STATE)
            rows.append({**keyd, "metric": met, **ci, "tag": title_tag})
    return pd.DataFrame(rows)

def fmt_ci(mean, lo, hi, digits=3):
    if np.isnan(mean) or np.isnan(lo) or np.isnan(hi):
        return "--"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"

def to_latex_table_main(summary: pd.DataFrame, caption: str, label: str, model_col: str = "model", digits=3) -> str:
    # Expect columns: model, metric, mean, p2.5, p97.5
    metrics_order = ["auc", "ap", "brier", "ece", "acc", "bacc", "f1", "prec", "rec", "mcc"]
    models = list(summary[model_col].unique())
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{" + caption + "}")
    lines.append("\\label{" + label + "}")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    header = "Model"
    for met in metrics_order:
        header += f" & {met.upper()}"
    header += " \\\\"
    lines.append("\\begin{tabular}{l" + "c"*len(metrics_order) + "}")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")
    for m in models:
        row = [str(m)]
        for met in metrics_order:
            sub = summary[(summary[model_col] == m) & (summary["metric"] == met)]
            if len(sub) == 0:
                row.append("--")
            else:
                row.append(fmt_ci(sub["mean"].iloc[0], sub["p2.5"].iloc[0], sub["p97.5"].iloc[0], digits=digits))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table}")
    return "\n".join(lines)

# ============================================================
# CV runners (MAIN)
# ============================================================
def repeated_stratified_cv(df: pd.DataFrame, X_cols: List[str], model_name: str, make_model_fn, calibrate="isotonic") -> pd.DataFrame:
    y = df[TARGET].to_numpy()
    X = df[X_cols]
    rows = []
    total = N_REPEATS * N_SPLITS
    pbar = tqdm(total=total, desc=f"Repeated CV: {model_name}", leave=True)
    for r in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + r)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            pbar.set_postfix(stage="repCV", model=model_name, repeat=r, fold=fold)
            model = make_model_fn(X_cols, calibrate=calibrate)
            model.fit(X.iloc[tr], y[tr])
            prob = model.predict_proba(X.iloc[te])[:, 1]
            m = compute_metrics(y[te], prob, threshold=0.5)
            m.update({"repeat": r, "fold": fold, "model": model_name})
            rows.append(m)
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)

def blocked_condition_cv(df: pd.DataFrame, X_cols: List[str], model_name: str, make_model_fn, calibrate="isotonic", n_splits=5) -> pd.DataFrame:
    y = df[TARGET].to_numpy()
    X = df[X_cols]
    groups = df["condition_group"].to_numpy()

    uniq = np.unique(groups)
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(uniq)

    fold_bins = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    group_counts = {g: int((groups == g).sum()) for g in uniq}
    for g in sorted(uniq, key=lambda x: group_counts[x], reverse=True):
        k = int(np.argmin(fold_sizes))
        fold_bins[k].append(g)
        fold_sizes[k] += group_counts[g]

    rows = []
    pbar = tqdm(total=n_splits, desc=f"Blocked CV: {model_name}", leave=True)
    for fold in range(n_splits):
        test_groups = set(fold_bins[fold])
        te_mask = np.array([g in test_groups for g in groups])
        tr_mask = ~te_mask
        pbar.set_postfix(stage="blkCV", model=model_name, fold=fold, n_test=int(te_mask.sum()))

        model = make_model_fn(X_cols, calibrate=calibrate)
        model.fit(X.iloc[tr_mask], y[tr_mask])
        prob = model.predict_proba(X.iloc[te_mask])[:, 1]
        m = compute_metrics(y[te_mask], prob, threshold=0.5)
        m.update({"fold": fold, "model": model_name, "n_test": int(te_mask.sum())})
        rows.append(m)
        pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)

# ============================================================
# Ablation (LR only; paper-friendly)
# ============================================================
def ablation_repeated_lr(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, cols in [
        ("Geometry-only", GEOM_ONLY_X),
        ("Visibility-only", VIS_ONLY_X),
        ("Visibility+Geometry", VIS_PLUS_GEOM_X),
    ]:
        tmp = repeated_stratified_cv(df, cols, f"LR(isotonic)::" + name, make_lr_pipeline, calibrate="isotonic")
        tmp["ablation"] = name
        out.append(tmp)
    return pd.concat(out, ignore_index=True)

# ============================================================
# Residual sufficiency, CMI-lite, subgroup, conformal, robustness
# ============================================================
def crossfit_residual_test(df: pd.DataFrame, n_splits=5, n_repeats=RESID_REPEATS, n_perm=RESID_PERM) -> pd.DataFrame:
    y = df[TARGET].to_numpy()
    Xv = df[VIS]
    Xg = df[GEOMETRY + HEADPOSE]
    rows = []
    total = n_repeats * n_splits
    pbar = tqdm(total=total, desc="Residual test (cross-fit)", leave=True)
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + 1000 + r)
        for fold, (tr, te) in enumerate(skf.split(Xv, y)):
            pbar.set_postfix(stage="resid", repeat=r, fold=fold)
            pre = make_preprocessor(num_cols=VIS, cat_cols=[])
            model_v = Pipeline([
                ("pre", pre),
                ("clf", LogisticRegression(
                    penalty="l2", solver="liblinear", max_iter=8000,
                    class_weight="balanced", random_state=RANDOM_STATE
                ))
            ])
            model_v.fit(Xv.iloc[tr], y[tr])
            p_te = model_v.predict_proba(Xv.iloc[te])[:, 1]
            resid = y[te] - p_te

            pre_g = make_preprocessor(num_cols=list(Xg.columns), cat_cols=[])
            Xg_te = pre_g.fit_transform(Xg.iloc[te])
            lin = LinearRegression()
            lin.fit(Xg_te, resid)
            r2 = float(lin.score(Xg_te, resid))

            rng = np.random.default_rng(RANDOM_STATE + 2000 + r*10 + fold)
            r2_perm = []
            for _ in range(n_perm):
                rp = rng.permutation(resid)
                lin.fit(Xg_te, rp)
                r2_perm.append(lin.score(Xg_te, rp))
            r2_perm = np.array(r2_perm)
            pval = float((np.sum(r2_perm >= r2) + 1) / (len(r2_perm) + 1))
            rows.append({"repeat": r, "fold": fold, "r2": r2, "p_perm": pval})
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)

def cmi_lite(df: pd.DataFrame, n_v_bins=5, random_state=42) -> pd.DataFrame:
    d = df.copy()
    d["vbin"] = qcut_bins(d["visibility_score"], q=n_v_bins).astype(int)

    cat_map = {c: {k: i for i, k in enumerate(sorted(d[c].unique()))} for c in CAT}
    for c in CAT:
        d[c + "_enc"] = d[c].map(cat_map[c]).astype(int)

    feats = GEOMETRY + HEADPOSE + [c + "_enc" for c in CAT]
    y = d[TARGET].to_numpy()
    weights = d["vbin"].value_counts(normalize=True).to_dict()

    out_rows = []
    for feat in tqdm(feats, desc="CMI-lite", leave=True):
        mi_cond = 0.0
        for vb, w in weights.items():
            sub = d[d["vbin"] == vb]
            if sub[TARGET].nunique() < 2 or len(sub) < 20:
                continue
            mi = float(mutual_info_classif(sub[[feat]], sub[TARGET].to_numpy(), discrete_features=False, random_state=random_state)[0])
            mi_cond += w * mi
        out_rows.append({"feature": feat, "cmi_lite": mi_cond})
    return pd.DataFrame(out_rows).sort_values("cmi_lite", ascending=False)

def fit_full_lr(df: pd.DataFrame, X_cols: List[str]) -> Tuple[np.ndarray, object]:
    y = df[TARGET].to_numpy()
    model = make_lr_pipeline(X_cols, calibrate="isotonic")
    model.fit(df[X_cols], y)
    prob = model.predict_proba(df[X_cols])[:, 1]
    return prob, model

def subgroup_eval_probs(df: pd.DataFrame, probs: np.ndarray, group_col: str, threshold=0.5) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group_col):
        idx = sub.index.to_numpy()
        m = compute_metrics(df.loc[idx, TARGET].to_numpy(), probs[idx], threshold=threshold)
        m.update({"group_col": group_col, "group": str(g), "n": int(len(idx))})
        rows.append(m)
    return pd.DataFrame(rows).sort_values("auc")

def mondrian_conformal_lr(df: pd.DataFrame, X_cols: List[str], group_col: str, alpha=0.10, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    uniq = df["condition_group"].unique()
    rng.shuffle(uniq)
    n_test = max(1, int(0.2 * len(uniq)))
    n_cal = max(1, int(0.2 * (len(uniq) - n_test)))

    test_groups = set(uniq[:n_test])
    cal_groups = set(uniq[n_test:n_test + n_cal])

    df_test = df[df["condition_group"].isin(test_groups)].copy()
    df_cal = df[df["condition_group"].isin(cal_groups)].copy()
    df_train = df[~df["condition_group"].isin(test_groups | cal_groups)].copy()

    model = make_lr_pipeline(X_cols, calibrate="isotonic")
    model.fit(df_train[X_cols], df_train[TARGET].to_numpy())

    p_cal = model.predict_proba(df_cal[X_cols])
    ycal = df_cal[TARGET].to_numpy()
    s_cal = 1.0 - p_cal[np.arange(len(ycal)), ycal]

    qhat_by_g = {}
    for g, sub in df_cal.groupby(group_col):
        if len(sub) < 15:
            continue
        p = model.predict_proba(sub[X_cols])
        ys = sub[TARGET].to_numpy()
        s = 1.0 - p[np.arange(len(ys)), ys]
        qhat_by_g[str(g)] = float(np.quantile(s, 1 - alpha, method="higher"))

    p_test = model.predict_proba(df_test[X_cols])
    ytest = df_test[TARGET].to_numpy()
    gtest = df_test[group_col].astype(str).to_numpy()

    fallback_q = float(np.quantile(s_cal, 1 - alpha, method="higher"))
    sets = []
    for i in range(len(ytest)):
        g = gtest[i]
        qhat = qhat_by_g.get(g, fallback_q)
        S = []
        for k in [0, 1]:
            if p_test[i, k] >= 1 - qhat:
                S.append(k)
        if len(S) == 0:
            S = [0, 1]
        sets.append(tuple(S))

    rows = []
    for g in sorted(np.unique(gtest)):
        idx = np.where(gtest == g)[0]
        covered = float(np.mean([ytest[i] in sets[i] for i in idx]))
        avg_size = float(np.mean([len(sets[i]) for i in idx]))
        singleton = float(np.mean([len(sets[i]) == 1 for i in idx]))
        rows.append({"group_col": group_col, "group": g, "n": int(len(idx)),
                     "coverage": covered, "avg_set_size": avg_size, "singleton_rate": singleton})
    return pd.DataFrame(rows).sort_values("coverage")

def robustness_structured_lr(df: pd.DataFrame, X_cols: List[str]) -> pd.DataFrame:
    base = df.copy()
    y = base[TARGET].to_numpy()
    model = make_lr_pipeline(X_cols, calibrate="isotonic")
    model.fit(base[X_cols], y)

    rows = []
    # A) bias
    for b in [-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20]:
        d = base.copy()
        d["visibility_score"] = np.clip(d["visibility_score"] + b, 0.0, 1.0)
        p = model.predict_proba(d[X_cols])[:, 1]
        rows.append({"type": "vis_bias", "level": b, **compute_metrics(y, p)})

    # B) clipping
    for lo, hi in [(0.0, 1.0), (0.05, 0.95), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]:
        d = base.copy()
        d["visibility_score"] = np.clip(d["visibility_score"], lo, hi)
        d["visibility_score"] = (d["visibility_score"] - lo) / max(1e-9, (hi - lo))
        p = model.predict_proba(d[X_cols])[:, 1]
        rows.append({"type": "vis_clip", "level": f"{lo}-{hi}", **compute_metrics(y, p)})

    # C) quantization
    for step in [1, 2, 5, 10, 15]:
        d = base.copy()
        for c in ["head_yaw_deg", "head_pitch_deg", "head_roll_deg"]:
            d[c] = (np.round(d[c] / step) * step).astype(float)
        p = model.predict_proba(d[X_cols])[:, 1]
        rows.append({"type": "pose_quant", "level": step, **compute_metrics(y, p)})

    return pd.DataFrame(rows)

def plot_reliability(df: pd.DataFrame, probs: np.ndarray, name: str):
    y = df[TARGET].to_numpy()
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
    plt.figure(figsize=(4.5,4))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Reliability: {name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"reliability_{name}.png"), dpi=200)
    plt.close()

# ============================================================
# RUN
# ============================================================
ST.start("[1] Load + condition groups")
df = pd.read_csv(DATA_PATH)
df[TARGET] = df[TARGET].astype(int)
df["medium"] = df["medium"].astype(str)
df["contrast"] = df["contrast"].astype(str)
if "participant_id" in df.columns:
    print("[WARN] participant_id exists but is ignored (no true participant_id).")
df = build_condition_group(df)
print("Shape:", df.shape, " | Class balance:", df[TARGET].mean())
ST.end()

# MAIN: Repeated CV (LR/RF)
ST.start("[2] MAIN: Repeated Stratified CV (LR/RF)")
rep_lr = repeated_stratified_cv(df, VIS_PLUS_GEOM_X, "LR(isotonic)", make_lr_pipeline, calibrate="isotonic")
rep_rf = repeated_stratified_cv(df, VIS_PLUS_GEOM_X, "RF(isotonic)", make_rf_pipeline, calibrate="isotonic")
rep = pd.concat([rep_lr, rep_rf], ignore_index=True)
rep.to_csv(os.path.join(OUT_DIR, "main_repeated_cv_splits.csv"), index=False)
ST.end()

# MAIN: Blocked CV (LR/RF)
ST.start("[3] MAIN: Condition-blocked CV (LR/RF)")
blk_lr = blocked_condition_cv(df, VIS_PLUS_GEOM_X, "LR(isotonic)", make_lr_pipeline, calibrate="isotonic", n_splits=5)
blk_rf = blocked_condition_cv(df, VIS_PLUS_GEOM_X, "RF(isotonic)", make_rf_pipeline, calibrate="isotonic", n_splits=5)
blk = pd.concat([blk_lr, blk_rf], ignore_index=True)
blk.to_csv(os.path.join(OUT_DIR, "main_blocked_cv_splits.csv"), index=False)
ST.end()

# MAIN: Ablation (LR)
ST.start("[4] MAIN: Ablation (LR repeated CV)")
abl = ablation_repeated_lr(df)
abl.to_csv(os.path.join(OUT_DIR, "main_ablation_lr_repeated_cv.csv"), index=False)
ST.end()

# Summaries + LaTeX tables
ST.start("[5] Paper tables (bootstrap CI) + AUC plots")
metrics_main = ["auc","ap","brier","ece","acc","bacc","f1","prec","rec","mcc"]

rep_sum = summarize_cv(rep, group_cols=["model"], metrics=metrics_main, title_tag="repeated_cv")
blk_sum = summarize_cv(blk, group_cols=["model"], metrics=metrics_main, title_tag="blocked_cv")
abl_sum = summarize_cv(abl, group_cols=["ablation"], metrics=["auc","ap","brier","ece","bacc","f1"], title_tag="ablation_lr")

rep_sum.to_csv(os.path.join(OUT_DIR, "table_repeated_cv_summary.csv"), index=False)
blk_sum.to_csv(os.path.join(OUT_DIR, "table_blocked_cv_summary.csv"), index=False)
abl_sum.to_csv(os.path.join(OUT_DIR, "table_ablation_summary.csv"), index=False)

# AUC distribution plots (paper figure)
plt.figure(figsize=(8,4))
for model in sorted(rep["model"].unique()):
    v = rep.loc[rep["model"] == model, "auc"].dropna().to_numpy()
    plt.hist(v, bins=25, alpha=0.6, label=model)
plt.xlabel("AUC per split (repeated stratified CV)")
plt.ylabel("Count")
plt.title("Distributional Uncertainty (AUC)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_auc_distribution_repeated_cv.png"), dpi=220)
plt.close()

# Assemble LaTeX tables (IEEE-friendly: use booktabs in your preamble)
latex_parts = []
latex_parts.append(to_latex_table_main(
    rep_sum.rename(columns={"model":"model"}),  # no-op
    caption="Repeated stratified CV performance (mean [95\\% CI]) across splits.",
    label="tab:repeated_cv",
    model_col="model",
    digits=3
))
latex_parts.append("")
latex_parts.append(to_latex_table_main(
    blk_sum.rename(columns={"model":"model"}),
    caption="Condition-blocked CV performance (mean [95\\% CI]). Entire viewing-condition groups are held out.",
    label="tab:blocked_cv",
    model_col="model",
    digits=3
))
# Ablation table (custom small)
def ablation_to_latex(abl_sum: pd.DataFrame) -> str:
    order = ["Geometry-only", "Visibility-only", "Visibility+Geometry"]
    metrics = ["auc","ap","brier","ece","bacc","f1"]
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation study (LR, repeated CV): mean [95\\% CI].}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\begin{tabular}{l" + "c"*len(metrics) + "}")
    lines.append("\\toprule")
    lines.append("Features & " + " & ".join([m.upper() for m in metrics]) + " \\\\")
    lines.append("\\midrule")
    for a in order:
        row = [a]
        for m in metrics:
            sub = abl_sum[(abl_sum["ablation"] == a) & (abl_sum["metric"] == m)]
            row.append(fmt_ci(sub["mean"].iloc[0], sub["p2.5"].iloc[0], sub["p97.5"].iloc[0], digits=3))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

latex_parts.append("")
latex_parts.append(ablation_to_latex(abl_sum))

with open(os.path.join(OUT_DIR, "tables_IEEE.tex"), "w", encoding="utf-8") as f:
    f.write("\n\n".join(latex_parts))

ST.end()

# Calibration plots (descriptive, full fit)
ST.start("[6] Calibration / reliability plots (full fit, descriptive)")
prob_lr_full, lr_full = fit_full_lr(df, VIS_PLUS_GEOM_X)
plot_reliability(df, prob_lr_full, name="LR_isotonic_full")
ST.end()

# Subgroup + worst-case (use LR full probs, descriptive)
ST.start("[7] Subgroup + worst-case (LR full, descriptive)")
sub_medium = subgroup_eval_probs(df, prob_lr_full, "medium", threshold=0.5)
sub_contrast = subgroup_eval_probs(df, prob_lr_full, "contrast", threshold=0.5)
sub_distbin = subgroup_eval_probs(df, prob_lr_full, "dist_bin", threshold=0.5)
sub_angbin = subgroup_eval_probs(df, prob_lr_full, "ang_bin", threshold=0.5)
sub_medium.to_csv(os.path.join(OUT_DIR, "subgroup_medium.csv"), index=False)
sub_contrast.to_csv(os.path.join(OUT_DIR, "subgroup_contrast.csv"), index=False)
sub_distbin.to_csv(os.path.join(OUT_DIR, "subgroup_distbin.csv"), index=False)
sub_angbin.to_csv(os.path.join(OUT_DIR, "subgroup_angbin.csv"), index=False)

# worst-case rows
worst = pd.concat([
    sub_medium.assign(grouping="medium").head(1),
    sub_contrast.assign(grouping="contrast").head(1),
    sub_distbin.assign(grouping="dist_bin").head(1),
    sub_angbin.assign(grouping="ang_bin").head(1),
], ignore_index=True)
worst.to_csv(os.path.join(OUT_DIR, "subgroup_worst_case.csv"), index=False)
ST.end()

# Mondrian conformal (LR)
ST.start("[8] Mondrian conformal (LR, by subgroup)")
mond = mondrian_conformal_lr(df, VIS_PLUS_GEOM_X, group_col=MONDRIAN_GROUP_COL, alpha=ALPHA, seed=RANDOM_STATE)
mond.to_csv(os.path.join(OUT_DIR, f"mondrian_conformal_{MONDRIAN_GROUP_COL}.csv"), index=False)
ST.end()

# Robustness stress tests (LR)
ST.start("[9] Robustness stress tests (LR)")
rob = robustness_structured_lr(df, VIS_PLUS_GEOM_X)
rob.to_csv(os.path.join(OUT_DIR, "robustness_structured.csv"), index=False)

plt.figure(figsize=(9,4))
for t in rob["type"].unique():
    sub = rob[rob["type"] == t]
    x = np.arange(len(sub))
    plt.plot(x, sub["auc"].to_numpy(), marker="o", label=t)
plt.xlabel("Setting index (see CSV for exact levels)")
plt.ylabel("AUC")
plt.title("Structured Robustness Stress Tests (LR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_robustness_auc.png"), dpi=220)
plt.close()
ST.end()

# Residual test
ST.start("[10] Cross-fitted residual sufficiency test")
resid = crossfit_residual_test(df, n_splits=5, n_repeats=RESID_REPEATS, n_perm=RESID_PERM)
resid.to_csv(os.path.join(OUT_DIR, "crossfit_residual_test.csv"), index=False)
ST.end()

# CMI-lite
ST.start("[11] CMI-lite")
cmi = cmi_lite(df, n_v_bins=5, random_state=RANDOM_STATE)
cmi.to_csv(os.path.join(OUT_DIR, "cmi_lite_results.csv"), index=False)
ST.end()

# Key files
print("\n[DONE] Outputs written to:", OUT_DIR)
for fn in [
    "main_repeated_cv_splits.csv",
    "main_blocked_cv_splits.csv",
    "main_ablation_lr_repeated_cv.csv",
    "table_repeated_cv_summary.csv",
    "table_blocked_cv_summary.csv",
    "table_ablation_summary.csv",
    "tables_IEEE.tex",
    "fig_auc_distribution_repeated_cv.png",
    "reliability_LR_isotonic_full.png",
    "subgroup_worst_case.csv",
    f"mondrian_conformal_{MONDRIAN_GROUP_COL}.csv",
    "robustness_structured.csv",
    "fig_robustness_auc.png",
    "crossfit_residual_test.csv",
    "cmi_lite_results.csv",
]:
    print(" -", os.path.join(OUT_DIR, fn))

ST.summary()
