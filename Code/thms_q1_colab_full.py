# ============================================================
# THMS Q1-Grade Reproducible Pipeline (Google Colab)
# - NO access to original/raw dataset required (uses provided CSV)
# - Avoids pseudo-participant claims; uses condition-blocked evaluation
# - Provides: repeated CV + OOF pooling, DeLong paired AUC test,
#             calibration (none/platt/isotonic), ablation,
#             cross-fitted residual sufficiency (properly out-of-sample),
#             subgroup stress tests, conformal prediction,
#             robustness (structured perturbations)
# ============================================================

# If running in Colab, uncomment:
# !pip -q install numpy pandas scikit-learn scipy matplotlib tqdm

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from scipy import stats

# -----------------------------
# CONFIG
# -----------------------------
RANDOM_STATE = 42
DATA_PATH = "reading_visibility_dataset_300rows.csv"

OUT_DIR = "thms_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

N_SPLITS = 5
N_REPEATS = 200  # 200*5=1000 folds like your zip (adjust if needed)

# Residual test
RESID_REPEATS = 80
RESID_PERM = 250

# Conformal
ALPHA = 0.10
MONDRIAN_GROUP_COL = "contrast"  # or "medium"

# -----------------------------
# SCHEMA (must match dataset)
# -----------------------------
GEOMETRY = ["distance_m", "font_size_pt", "text_height_mm", "angular_size_deg"]
HEADPOSE = ["head_yaw_deg", "head_pitch_deg", "head_roll_deg"]
VIS = ["visibility_score"]
CAT = ["medium", "contrast"]
TARGET = "can_read"

VIS_ONLY_X = VIS
GEOM_ONLY_X = GEOMETRY + HEADPOSE + CAT
VIS_PLUS_GEOM_X = GEOMETRY + HEADPOSE + VIS + CAT

# -----------------------------
# UTIL: preprocessing + models
# -----------------------------
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
    if len(num_cols) > 0:
        transformers.append(("num", num_tf, num_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", cat_tf, cat_cols))
    return ColumnTransformer(transformers)

def make_lr_pipeline(X_cols: List[str], calibrate: Optional[str] = "isotonic"):
    num_cols = [c for c in X_cols if c not in CAT]
    cat_cols = [c for c in X_cols if c in CAT]
    pre = make_preprocessor(num_cols, cat_cols)
    base = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            penalty="l2", solver="liblinear", max_iter=8000,
            class_weight="balanced", random_state=RANDOM_STATE
        ))
    ])
    if calibrate is None:
        return base
    method = "sigmoid" if calibrate == "platt" else "isotonic"
    # IMPORTANT: calibration is trained on TRAINING fold only via inner CV
    return CalibratedClassifierCV(base, method=method, cv=3)

def make_rf_pipeline(X_cols: List[str], calibrate: Optional[str] = "isotonic"):
    num_cols = [c for c in X_cols if c not in CAT]
    cat_cols = [c for c in X_cols if c in CAT]
    pre = make_preprocessor(num_cols, cat_cols)
    base = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=800, n_jobs=-1,
            class_weight="balanced_subsample", random_state=RANDOM_STATE
        ))
    ])
    if calibrate is None:
        return base
    method = "sigmoid" if calibrate == "platt" else "isotonic"
    return CalibratedClassifierCV(base, method=method, cv=3)

# -----------------------------
# METRICS
# -----------------------------
def ece_score(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = y_true[mask].mean()
        conf_bin = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc_bin - conf_bin)
    return float(ece)

def compute_metrics(y_true, y_prob, threshold=0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(ece_score(y_true, y_prob)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "prec": float(precision_score(y_true, y_pred, zero_division=0)),
        "rec": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

def bootstrap_ci_metric(y_true, y_prob, metric_fn, n_boot=4000, alpha=0.05, seed=42):
    r = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = r.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        try:
            vals.append(metric_fn(yt, yp))
        except Exception:
            continue
    vals = np.array(vals, dtype=float)
    lo = np.quantile(vals, alpha/2)
    hi = np.quantile(vals, 1 - alpha/2)
    return float(np.mean(vals)), float(lo), float(hi)

# -----------------------------
# DeLong paired AUC test (fast)
# Reference: DeLong et al. 1988; implementation adapted from common public-domain derivations.
# -----------------------------
def _compute_midrank(x):
    x = np.asarray(x)
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1
        T[i:j] = mid
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out

def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = (tz[:, :m].sum(axis=1) - m*(m+1)/2.0) / (m*n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx/m + sy/n
    return aucs, s

def delong_pvalue(y_true, y_prob_a, y_prob_b):
    y_true = np.asarray(y_true).astype(int)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)

    order = np.argsort(-y_true)  # positives first
    y_true_sorted = y_true[order]
    preds = np.vstack([y_prob_a[order], y_prob_b[order]])
    m = int(y_true_sorted.sum())
    aucs, s = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var = s[0,0] + s[1,1] - 2*s[0,1]
    if var <= 1e-12:
        return float(diff), 1.0
    z = diff / math.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(diff), float(p)

# -----------------------------
# Condition groups (to avoid pseudo-participants)
# -----------------------------
def build_condition_group(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Discretize continuous conditions into coarse bins for "blocked" evaluation
    # Adjust bin edges only if you have strong domain justification.
    d["dist_bin"] = pd.cut(d["distance_m"], bins=[-np.inf, 0.6, 1.0, 1.5, np.inf], labels=False).astype(int)
    d["ang_bin"]  = pd.cut(d["angular_size_deg"], bins=[-np.inf, 1.2, 1.8, 2.6, np.inf], labels=False).astype(int)
    d["condition_group"] = (
        d["dist_bin"].astype(str) + "_" +
        d["ang_bin"].astype(str) + "_" +
        d["medium"].astype(str) + "_" +
        d["contrast"].astype(str)
    )
    return d

# -----------------------------
# MAIN evaluation: repeated CV with pooled OOF for professor-grade stats
# -----------------------------
@dataclass
class OOF:
    y_true: np.ndarray
    y_prob: np.ndarray

def repeated_cv_oof(df: pd.DataFrame, X_cols: List[str], make_model_fn, calibrate="isotonic") -> Tuple[pd.DataFrame, OOF]:
    y = df[TARGET].to_numpy().astype(int)
    X = df[X_cols]
    rows = []
    # store pooled OOF across ALL folds (repeat-fold pooled)
    oof_y, oof_p = [], []
    pbar = tqdm(total=N_REPEATS * N_SPLITS, desc="Repeated CV (OOF pooled)", leave=True)
    for r in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + r)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            model = make_model_fn(X_cols, calibrate=calibrate)
            model.fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[te])[:, 1]
            m = compute_metrics(y[te], p)
            m.update({"repeat": r, "fold": fold})
            rows.append(m)
            oof_y.append(y[te])
            oof_p.append(p)
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows), OOF(y_true=np.concatenate(oof_y), y_prob=np.concatenate(oof_p))

def blocked_condition_cv(df: pd.DataFrame, X_cols: List[str], make_model_fn, calibrate="isotonic", n_splits=5) -> pd.DataFrame:
    y = df[TARGET].to_numpy().astype(int)
    X = df[X_cols]
    groups = df["condition_group"].to_numpy()

    uniq = np.unique(groups)
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(uniq)

    # greedy bin-packing to balance test sizes per fold
    fold_bins = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    group_counts = {g: int((groups == g).sum()) for g in uniq}
    for g in sorted(uniq, key=lambda x: group_counts[x], reverse=True):
        k = int(np.argmin(fold_sizes))
        fold_bins[k].append(g)
        fold_sizes[k] += group_counts[g]

    rows = []
    pbar = tqdm(total=n_splits, desc="Condition-blocked CV", leave=True)
    for fold in range(n_splits):
        test_groups = set(fold_bins[fold])
        te_mask = np.array([g in test_groups for g in groups])
        tr_mask = ~te_mask
        model = make_model_fn(X_cols, calibrate=calibrate)
        model.fit(X.iloc[tr_mask], y[tr_mask])
        p = model.predict_proba(X.iloc[te_mask])[:, 1]
        m = compute_metrics(y[te_mask], p)
        m.update({"fold": fold, "n_test": int(te_mask.sum())})
        rows.append(m)
        pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)

# -----------------------------
# Ablation (visibility-only / geometry-only / full) using pooled OOF + DeLong
# -----------------------------
def ablation_suite(df: pd.DataFrame):
    # Use LR as primary (interpretability & strong baseline)
    rep_full, oof_full = repeated_cv_oof(df, VIS_PLUS_GEOM_X, make_lr_pipeline, calibrate="isotonic")
    rep_vis,  oof_vis  = repeated_cv_oof(df, VIS_ONLY_X,      make_lr_pipeline, calibrate="isotonic")
    rep_geom, oof_geom = repeated_cv_oof(df, GEOM_ONLY_X,     make_lr_pipeline, calibrate="isotonic")

    # DeLong paired AUC: Visibility-only vs Full (pooled OOF)
    diff_v_full, p_v_full = delong_pvalue(oof_full.y_true, oof_vis.y_prob, oof_full.y_prob)

    out = {
        "rep_full": rep_full, "oof_full": oof_full,
        "rep_vis": rep_vis, "oof_vis": oof_vis,
        "rep_geom": rep_geom, "oof_geom": oof_geom,
        "delong_diff_auc_vis_minus_full": diff_v_full,
        "delong_pvalue_vis_vs_full": p_v_full,
    }
    return out

# -----------------------------
# Proper cross-fitted residual sufficiency test
# (Does geometry explain OUT-OF-SAMPLE residuals after visibility?)
# -----------------------------
def residual_sufficiency_test(df: pd.DataFrame, n_splits=5, n_repeats=RESID_REPEATS, n_perm=RESID_PERM):
    y = df[TARGET].to_numpy().astype(int)
    Xv = df[VIS]
    Xg = df[GEOMETRY + HEADPOSE]  # numeric only, no CAT for this test

    rows = []
    pbar = tqdm(total=n_repeats*n_splits, desc="Residual sufficiency (proper)", leave=True)
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + 777 + r)
        for fold, (tr, te) in enumerate(skf.split(Xv, y)):
            # 1) fit visibility model on TRAIN, predict on TRAIN and TEST
            pre_v = make_preprocessor(num_cols=VIS, cat_cols=[])
            mv = Pipeline([
                ("pre", pre_v),
                ("clf", LogisticRegression(
                    penalty="l2", solver="liblinear", max_iter=8000,
                    class_weight="balanced", random_state=RANDOM_STATE
                ))
            ])
            mv.fit(Xv.iloc[tr], y[tr])
            p_tr = mv.predict_proba(Xv.iloc[tr])[:, 1]
            p_te = mv.predict_proba(Xv.iloc[te])[:, 1]

            resid_tr = y[tr] - p_tr
            resid_te = y[te] - p_te

            # 2) fit geometry regressor on TRAIN residuals, score on TEST residuals
            pre_g = make_preprocessor(num_cols=list(Xg.columns), cat_cols=[])
            Xg_tr = pre_g.fit_transform(Xg.iloc[tr])
            Xg_te = pre_g.transform(Xg.iloc[te])

            lin = LinearRegression()
            lin.fit(Xg_tr, resid_tr)
            r2 = float(lin.score(Xg_te, resid_te))

            # 3) permutation test on TEST residuals (keep trained model fixed)
            rng = np.random.default_rng(RANDOM_STATE + 2000 + r*10 + fold)
            r2_perm = []
            for _ in range(n_perm):
                rp = rng.permutation(resid_te)
                r2_perm.append(lin.score(Xg_te, rp))
            r2_perm = np.array(r2_perm, dtype=float)
            pval = float((np.sum(r2_perm >= r2) + 1) / (len(r2_perm) + 1))

            rows.append({"repeat": r, "fold": fold, "r2": r2, "p_perm": pval})
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)

# -----------------------------
# Conformal prediction (split conformal) + Mondrian by subgroup
# -----------------------------
def split_conformal_lr(df: pd.DataFrame, X_cols: List[str], alpha=0.10, group_col: Optional[str]=None, seed=42):
    rng = np.random.default_rng(seed)
    d = df.copy()
    y = d[TARGET].to_numpy().astype(int)

    # simple random split (stratified)
    idx = np.arange(len(d))
    # stratified split by y
    pos = idx[y==1]
    neg = idx[y==0]
    rng.shuffle(pos); rng.shuffle(neg)
    n_test_pos = max(1, int(0.2*len(pos)))
    n_test_neg = max(1, int(0.2*len(neg)))
    test_idx = np.concatenate([pos[:n_test_pos], neg[:n_test_neg]])
    train_idx = np.setdiff1d(idx, test_idx)

    # further split train into proper + calib (stratified)
    ytr = y[train_idx]
    pos2 = train_idx[ytr==1]
    neg2 = train_idx[ytr==0]
    rng.shuffle(pos2); rng.shuffle(neg2)
    n_cal_pos = max(1, int(0.2*len(pos2)))
    n_cal_neg = max(1, int(0.2*len(neg2)))
    cal_idx = np.concatenate([pos2[:n_cal_pos], neg2[:n_cal_neg]])
    prop_idx = np.setdiff1d(train_idx, cal_idx)

    model = make_lr_pipeline(X_cols, calibrate="isotonic")
    model.fit(d.iloc[prop_idx][X_cols], y[prop_idx])

    # nonconformity score for true label y: s = 1 - p(y|x)
    p_cal = model.predict_proba(d.iloc[cal_idx][X_cols])
    y_cal = y[cal_idx]
    s_cal = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]
    qhat = np.quantile(s_cal, 1 - alpha, method="higher")

    # prediction sets on test
    p_test = model.predict_proba(d.iloc[test_idx][X_cols])
    y_test = y[test_idx]
    sets = []
    for i in range(len(y_test)):
        S = []
        for k in [0,1]:
            if p_test[i, k] >= 1 - qhat:
                S.append(k)
        if len(S) == 0:
            S = [0,1]
        sets.append(tuple(S))

    covered = np.mean([y_test[i] in sets[i] for i in range(len(y_test))])
    avg_size = float(np.mean([len(s) for s in sets]))
    singleton = float(np.mean([len(s) == 1 for s in sets]))

    out = pd.DataFrame({
        "idx": test_idx,
        "y_true": y_test,
        "p0": p_test[:,0],
        "p1": p_test[:,1],
        "set": [str(s) for s in sets],
    })

    summary = {"coverage": float(covered), "avg_set_size": avg_size, "singleton_rate": singleton, "n_test": int(len(test_idx))}

    if group_col is not None:
        # Mondrian summary by group (descriptive; note small n risk)
        out[group_col] = d.iloc[test_idx][group_col].to_numpy()
        rows = []
        for g, sub in out.groupby(group_col):
            yt = sub["y_true"].to_numpy().astype(int)
            Ss = sub["set"].tolist()
            cov = np.mean([yt[i] in eval(Ss[i]) for i in range(len(yt))]) if len(yt)>0 else np.nan
            rows.append({
                "group_col": group_col, "group": str(g), "n": int(len(yt)),
                "coverage": float(cov) if not np.isnan(cov) else np.nan,
                "avg_set_size": float(np.mean([len(eval(s)) for s in Ss])) if len(yt)>0 else np.nan,
                "singleton_rate": float(np.mean([len(eval(s))==1 for s in Ss])) if len(yt)>0 else np.nan
            })
        mond = pd.DataFrame(rows)
        return summary, out, mond

    return summary, out, None

# -----------------------------
# Robustness stress tests (structured, interpretable)
# -----------------------------
def robustness_structured_lr(df: pd.DataFrame, X_cols: List[str]) -> pd.DataFrame:
    d0 = df.copy()
    y = d0[TARGET].to_numpy().astype(int)

    # Fit once on full data (explicitly: robustness is about sensitivity, not gen-estimation)
    model = make_lr_pipeline(X_cols, calibrate="isotonic")
    model.fit(d0[X_cols], y)

    rows = []

    # A) visibility clipping (simulate saturation / sensor truncation)
    for clip in [0.90, 0.80, 0.70, 0.60, 0.50]:
        d = d0.copy()
        d["visibility_score"] = np.clip(d["visibility_score"], 0.0, clip)
        p = model.predict_proba(d[X_cols])[:, 1]
        rows.append({"type": "vis_clip", "level": float(clip), **compute_metrics(y, p)})

    # B) visibility bias (simulate systematic bias in visibility measurement)
    for bias in [-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20]:
        d = d0.copy()
        d["visibility_score"] = np.clip(d["visibility_score"] + bias, 0.0, 1.0)
        p = model.predict_proba(d[X_cols])[:, 1]
        rows.append({"type": "vis_bias", "level": float(bias), **compute_metrics(y, p)})

    # C) headpose quantization (simulate coarser head-pose sensors)
    for step in [1, 2, 5, 10, 15]:
        d = d0.copy()
        for c in ["head_yaw_deg", "head_pitch_deg", "head_roll_deg"]:
            d[c] = (np.round(d[c] / step) * step).astype(float)
        p = model.predict_proba(d[X_cols])[:, 1]
        rows.append({"type": "pose_quant", "level": float(step), **compute_metrics(y, p)})

    return pd.DataFrame(rows)

# -----------------------------
# PLOTTING helpers (matplotlib only)
# -----------------------------
def plot_reliability(y_true, probs, name: str):
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
    plt.figure(figsize=(4.6,4.2))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Reliability: {name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"reliability_{name}.png"), dpi=220)
    plt.close()

# ============================================================
# RUN
# ============================================================
print("[1] Load dataset")
df = pd.read_csv(DATA_PATH)
df[TARGET] = df[TARGET].astype(int)
df["medium"] = df["medium"].astype(str)
df["contrast"] = df["contrast"].astype(str)
df = build_condition_group(df)

print("Shape:", df.shape, "| Class balance:", df[TARGET].mean())

# 2) MAIN: repeated CV summary (LR + RF)
print("\n[2] MAIN repeated CV (LR/RF) with isotonic calibration")
rep_lr, oof_lr = repeated_cv_oof(df, VIS_PLUS_GEOM_X, make_lr_pipeline, calibrate="isotonic")
rep_rf, oof_rf = repeated_cv_oof(df, VIS_PLUS_GEOM_X, make_rf_pipeline, calibrate="isotonic")

rep_lr.to_csv(os.path.join(OUT_DIR, "main_repeated_cv_lr_splits.csv"), index=False)
rep_rf.to_csv(os.path.join(OUT_DIR, "main_repeated_cv_rf_splits.csv"), index=False)

# pooled OOF reliability figures
plot_reliability(oof_lr.y_true, oof_lr.y_prob, "LR_isotonic_pooledOOF")
plot_reliability(oof_rf.y_true, oof_rf.y_prob, "RF_isotonic_pooledOOF")

# 3) MAIN: condition-blocked CV (deployment shift proxy)
print("\n[3] MAIN condition-blocked CV (LR/RF)")
blk_lr = blocked_condition_cv(df, VIS_PLUS_GEOM_X, make_lr_pipeline, calibrate="isotonic", n_splits=5)
blk_rf = blocked_condition_cv(df, VIS_PLUS_GEOM_X, make_rf_pipeline, calibrate="isotonic", n_splits=5)
blk_lr.to_csv(os.path.join(OUT_DIR, "main_blocked_cv_lr.csv"), index=False)
blk_rf.to_csv(os.path.join(OUT_DIR, "main_blocked_cv_rf.csv"), index=False)

# 4) Ablation + DeLong
print("\n[4] Ablation suite + DeLong (visibility-only vs full)")
abl = ablation_suite(df)
abl["rep_full"].to_csv(os.path.join(OUT_DIR, "ablation_rep_full_lr.csv"), index=False)
abl["rep_vis"].to_csv(os.path.join(OUT_DIR, "ablation_rep_vis_lr.csv"), index=False)
abl["rep_geom"].to_csv(os.path.join(OUT_DIR, "ablation_rep_geom_lr.csv"), index=False)

pd.DataFrame([{
    "delong_diff_auc_vis_minus_full": abl["delong_diff_auc_vis_minus_full"],
    "delong_pvalue_vis_vs_full": abl["delong_pvalue_vis_vs_full"]
}]).to_csv(os.path.join(OUT_DIR, "delong_vis_vs_full.csv"), index=False)

# 5) Residual sufficiency (proper)
print("\n[5] Residual sufficiency (proper cross-fit)")
resid = residual_sufficiency_test(df)
resid.to_csv(os.path.join(OUT_DIR, "residual_sufficiency_proper.csv"), index=False)

# 6) Conformal prediction
print("\n[6] Split conformal + Mondrian by subgroup (descriptive)")
conf_summary, conf_rows, mond = split_conformal_lr(df, VIS_PLUS_GEOM_X, alpha=ALPHA, group_col=MONDRIAN_GROUP_COL, seed=RANDOM_STATE)
pd.DataFrame([conf_summary]).to_csv(os.path.join(OUT_DIR, "conformal_summary.csv"), index=False)
conf_rows.to_csv(os.path.join(OUT_DIR, "conformal_test_rows.csv"), index=False)
if mond is not None:
    mond.to_csv(os.path.join(OUT_DIR, f"mondrian_conformal_{MONDRIAN_GROUP_COL}.csv"), index=False)

# 7) Robustness structured
print("\n[7] Robustness structured stress tests")
rob = robustness_structured_lr(df, VIS_PLUS_GEOM_X)
rob.to_csv(os.path.join(OUT_DIR, "robustness_structured_lr.csv"), index=False)

# quick plot
plt.figure(figsize=(9,4))
for t in rob["type"].unique():
    sub = rob[rob["type"] == t].reset_index(drop=True)
    plt.plot(np.arange(len(sub)), sub["auc"].to_numpy(), marker="o", label=t)
plt.xlabel("Setting index (see CSV for exact levels)")
plt.ylabel("AUC")
plt.title("Structured Robustness Stress Tests (LR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_robustness_auc.png"), dpi=220)
plt.close()

print("\n[DONE] Outputs in:", OUT_DIR)
