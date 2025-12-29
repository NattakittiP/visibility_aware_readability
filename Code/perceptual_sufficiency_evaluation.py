# ============================================================
# - Group-aware LOPO / nested CV OOF pooled evaluation
# - DeLong paired AUC tests
# - Bootstrap CI for all metrics
# - Calibration: None / Platt / Isotonic + ECE + reliability
# - Decision Curve Analysis (Net Benefit)
# - Robustness: feature noise + label noise
# - Conformal prediction (classification) with coverage
# - Mediation / sufficiency tests:
#     (1) Full vs V-only vs Geom-only (OOF + DeLong + CI)
#     (2) Residualization: does geometry explain residual after visibility?
# ============================================================

!pip -q install numpy pandas scikit-learn scipy matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from sklearn.model_selection import (
    StratifiedKFold, GroupKFold
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.feature_selection import mutual_info_classif
from scipy import stats


# -------------------------
# Repro
# -------------------------
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

DATA_PATH = "reading_visibility_dataset_300rows.csv"


# ============================================================
# 1) Load + schema
# ============================================================
df = pd.read_csv(DATA_PATH)

# enforce types
df["can_read"] = df["can_read"].astype(int)
df["medium"] = df["medium"].astype(str)
df["contrast"] = df["contrast"].astype(str)

# feature groups (match your paper/code)
GEOMETRY = ["distance_m", "font_size_pt", "text_height_mm", "angular_size_deg"]
HEADPOSE = ["head_yaw_deg", "head_pitch_deg", "head_roll_deg"]
VIS = ["visibility_score"]
CAT = ["medium", "contrast"]

TARGET = "can_read"


# ============================================================
# 2) Ensure participant_id exists (true LOPO if already present)
#    If not present, assign balanced pseudo-participants (as in your ablation script)
# ============================================================
def assign_participants_balanced(df_in: pd.DataFrame, n_participants: int = 20, seed: int = 42) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    idx = np.arange(len(df_in))
    r.shuffle(idx)

    base = len(df_in) // n_participants
    rem = len(df_in) % n_participants

    sizes = np.array([base] * n_participants)
    sizes[:rem] += 1

    pid = np.empty(len(df_in), dtype=int)
    start = 0
    for p, sz in enumerate(sizes):
        pid[idx[start:start + sz]] = p
        start += sz

    out = df_in.copy()
    out["participant_id"] = pid
    return out

if "participant_id" not in df.columns:
    df = assign_participants_balanced(df, n_participants=20, seed=RANDOM_STATE)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Participants:", df["participant_id"].nunique(), "unique")


# ============================================================
# 3) Preprocessor + models
# ============================================================
def make_preprocessor(num_cols, cat_cols):
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

def make_lr(pre):
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=6000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

def make_rf(pre):
    return Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=800,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE
        ))
    ])


# ============================================================
# 4) Metrics + ECE
# ============================================================
def ece_score(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true)
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

def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "ap": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred),
        "rec": recall_score(y_true, y_pred),
        "ece": ece_score(y_true, y_prob, n_bins=15),
    }


# ============================================================
# 5) Bootstrap CI on pooled OOF predictions
# ============================================================
def bootstrap_ci(y_true, y_prob, threshold=0.5, n_boot=6000, alpha=0.05, seed=42):
    r = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    stats_list = []
    for _ in range(n_boot):
        idx = r.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        yhat = (yp >= threshold).astype(int)
        stats_list.append(compute_metrics(yt, yhat, yp))

    out = {}
    keys = list(stats_list[0].keys())
    for k in keys:
        vals = np.array([d[k] for d in stats_list])
        lo = np.quantile(vals, alpha/2)
        hi = np.quantile(vals, 1 - alpha/2)
        out[k] = (float(vals.mean()), float(lo), float(hi))
    return out


# ============================================================
# 6) DeLong paired AUC test (self-contained)
# ============================================================
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])

    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    return aucs, s

def delong_roc_test(y_true, y_prob_1, y_prob_2):
    y_true = np.asarray(y_true).astype(int)
    y_prob_1 = np.asarray(y_prob_1)
    y_prob_2 = np.asarray(y_prob_2)

    order = np.argsort(-y_true)  # positives first
    y_true_sorted = y_true[order]
    p1 = y_prob_1[order]
    p2 = y_prob_2[order]

    m = int(y_true_sorted.sum())
    preds = np.vstack([p1, p2])
    aucs, cov = _fast_delong(preds, m)

    diff = aucs[0] - aucs[1]
    var = cov[0,0] + cov[1,1] - 2*cov[0,1]
    z = diff / np.sqrt(var + 1e-12)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": float(diff), "z": float(z), "p": float(p)}


# ============================================================
# 7) OOF Evaluation engines
#    - StratifiedKFold (standard)
#    - GroupKFold (LOPO)
# ============================================================
@dataclass
class OOFResult:
    fold_metrics: pd.DataFrame
    y_true_all: np.ndarray
    y_prob_all: np.ndarray
    y_pred_all: np.ndarray

def oof_eval(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series] = None,
    base: str = "lr",                 # "lr" or "rf"
    calibrate: Optional[str] = None,  # None | "platt" | "isotonic"
    n_splits: int = 5,
    threshold: float = 0.5,
):
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_iter = cv.split(X, y)
    else:
        cv = GroupKFold(n_splits=n_splits)
        split_iter = cv.split(X, y, groups=groups)

    y_true_all, y_prob_all, y_pred_all = [], [], []
    rows = []

    num_cols = [c for c in X.columns if c not in CAT]
    cat_cols = [c for c in X.columns if c in CAT]
    pre = make_preprocessor(num_cols, cat_cols)

    base_model = make_rf(pre) if base == "rf" else make_lr(pre)

    for fold, (tr, te) in enumerate(split_iter, 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        if calibrate is None:
            model = base_model
        else:
            method = "sigmoid" if calibrate == "platt" else "isotonic"
            # calibration must be trained only on training fold, use inner CV
            model = CalibratedClassifierCV(base_model, method=method, cv=3)

        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        pred = (prob >= threshold).astype(int)

        met = compute_metrics(yte, pred, prob)
        met["fold"] = fold
        rows.append(met)

        y_true_all.append(yte.to_numpy())
        y_prob_all.append(prob)
        y_pred_all.append(pred)

    return OOFResult(
        fold_metrics=pd.DataFrame(rows),
        y_true_all=np.concatenate(y_true_all),
        y_prob_all=np.concatenate(y_prob_all),
        y_pred_all=np.concatenate(y_pred_all),
    )


# ============================================================
# 8) Sufficiency Suite (core Q1 claim)
# ============================================================
def sufficiency_suite(df: pd.DataFrame, use_groups: bool = True):
    y = df[TARGET]
    groups = df["participant_id"] if use_groups else None

    X_v = df[VIS].copy()
    X_full = df[GEOMETRY + HEADPOSE + VIS + CAT].copy()
    X_geom = df[GEOMETRY + HEADPOSE + CAT].copy()

    # Use LR as primary (professor posture). RF optional as secondary.
    res_v = oof_eval(X_v, y, groups=groups, base="lr", calibrate=None, n_splits=5)
    res_full = oof_eval(X_full, y, groups=groups, base="lr", calibrate=None, n_splits=5)
    res_geom = oof_eval(X_geom, y, groups=groups, base="lr", calibrate=None, n_splits=5)

    # DeLong paired tests (pooled OOF)
    dl_full_vs_v = delong_roc_test(res_full.y_true_all, res_full.y_prob_all, res_v.y_prob_all)
    dl_v_vs_geom = delong_roc_test(res_v.y_true_all, res_v.y_prob_all, res_geom.y_prob_all)

    # Bootstrap CI
    ci_v = bootstrap_ci(res_v.y_true_all, res_v.y_prob_all)
    ci_full = bootstrap_ci(res_full.y_true_all, res_full.y_prob_all)
    ci_geom = bootstrap_ci(res_geom.y_true_all, res_geom.y_prob_all)

    return {
        "res_v": res_v,
        "res_full": res_full,
        "res_geom": res_geom,
        "delong_full_vs_v": dl_full_vs_v,
        "delong_v_vs_geom": dl_v_vs_geom,
        "ci_v": ci_v,
        "ci_full": ci_full,
        "ci_geom": ci_geom
    }


# ============================================================
# 9) Calibration Suite + reliability plots (OOF pooled)
# ============================================================
def calibration_suite(df: pd.DataFrame, use_groups: bool = True):
    y = df[TARGET]
    groups = df["participant_id"] if use_groups else None
    X_full = df[GEOMETRY + HEADPOSE + VIS + CAT].copy()

    res_none = oof_eval(X_full, y, groups=groups, base="lr", calibrate=None, n_splits=5)
    res_platt = oof_eval(X_full, y, groups=groups, base="lr", calibrate="platt", n_splits=5)
    res_iso = oof_eval(X_full, y, groups=groups, base="lr", calibrate="isotonic", n_splits=5)

    # Plot reliability
    plt.figure(figsize=(6,6))
    for name, res in [("none", res_none), ("platt", res_platt), ("isotonic", res_iso)]:
        pt, pp = calibration_curve(res.y_true_all, res.y_prob_all, n_bins=10)
        plt.plot(pp, pt, marker="o", label=f"{name} (ECE={ece_score(res.y_true_all,res.y_prob_all):.3f})")
    plt.plot([0,1],[0,1],"k--")
    plt.title("Reliability diagram (OOF pooled)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.grid(True)
    plt.legend()
    plt.show()

    return {"none": res_none, "platt": res_platt, "isotonic": res_iso}


# ============================================================
# 10) Decision Curve Analysis (Net Benefit)
# ============================================================
def decision_curve(y_true, y_prob, thresholds=np.linspace(0.01, 0.99, 99)):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    nb = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        net_benefit = (tp / n) - (fp / n) * (t / (1 - t))
        nb.append(net_benefit)

    prevalence = y_true.mean()
    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    nb_none = np.zeros_like(thresholds)
    return thresholds, np.array(nb), nb_all, nb_none

def plot_decision_curve(y_true, y_prob, title="Decision Curve"):
    ths, nb, nb_all, nb_none = decision_curve(y_true, y_prob)
    plt.figure(figsize=(7,5))
    plt.plot(ths, nb, label="Model")
    plt.plot(ths, nb_all, linestyle="--", label="Treat-all")
    plt.plot(ths, nb_none, linestyle="--", label="Treat-none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


# ============================================================
# 11) Robustness Suite: feature noise + label noise
# ============================================================
def add_feature_noise(df_in, noise_cfg: Dict[str, float], seed=42):
    r = np.random.default_rng(seed)
    df2 = df_in.copy()
    for col, sigma in noise_cfg.items():
        if col not in df2.columns:
            continue
        s = df2[col].std()
        df2[col] = df2[col] + r.normal(0, sigma * (s if s > 0 else 1.0), size=len(df2))
    return df2

def flip_labels(y, flip_rate=0.05, seed=42):
    r = np.random.default_rng(seed)
    y = y.copy()
    n = len(y)
    k = int(np.round(n * flip_rate))
    idx = r.choice(n, size=k, replace=False)
    y[idx] = 1 - y[idx]
    return y

def robustness_suite(df: pd.DataFrame, use_groups: bool = True):
    groups = df["participant_id"] if use_groups else None
    X_cols = GEOMETRY + HEADPOSE + VIS + CAT

    levels = [0.0, 0.05, 0.10, 0.20, 0.30]
    flip_rates = [0.0, 0.02, 0.05, 0.10, 0.15]

    rows = []

    # A) visibility noise
    for nl in levels:
        dfn = add_feature_noise(df[X_cols + [TARGET, "participant_id"]], {"visibility_score": nl}, seed=RANDOM_STATE)
        X = dfn[X_cols]
        y = dfn[TARGET]
        g = dfn["participant_id"] if use_groups else None
        res = oof_eval(X, y, groups=g, base="lr", calibrate=None, n_splits=5)
        rows.append({"type":"vis_noise", "level":nl,
                     "auc":roc_auc_score(res.y_true_all,res.y_prob_all),
                     "brier":brier_score_loss(res.y_true_all,res.y_prob_all)})

    # B) head pose noise (sensor jitter)
    for nl in levels:
        dfn = add_feature_noise(df[X_cols + [TARGET, "participant_id"]],
                                {"head_yaw_deg": nl, "head_pitch_deg": nl, "head_roll_deg": nl},
                                seed=RANDOM_STATE)
        X = dfn[X_cols]
        y = dfn[TARGET]
        g = dfn["participant_id"] if use_groups else None
        res = oof_eval(X, y, groups=g, base="lr", calibrate=None, n_splits=5)
        rows.append({"type":"pose_noise", "level":nl,
                     "auc":roc_auc_score(res.y_true_all,res.y_prob_all),
                     "brier":brier_score_loss(res.y_true_all,res.y_prob_all)})

    # C) label flips
    for fr in flip_rates:
        X = df[X_cols]
        y = pd.Series(flip_labels(df[TARGET].to_numpy(), flip_rate=fr, seed=RANDOM_STATE), index=df.index)
        g = groups
        res = oof_eval(X, y, groups=g, base="lr", calibrate=None, n_splits=5)
        rows.append({"type":"label_flip", "level":fr,
                     "auc":roc_auc_score(res.y_true_all,res.y_prob_all),
                     "brier":brier_score_loss(res.y_true_all,res.y_prob_all)})

    rob = pd.DataFrame(rows)

    # plot
    plt.figure(figsize=(8,4))
    for t in rob["type"].unique():
        sub = rob[rob["type"]==t].sort_values("level")
        plt.plot(sub["level"], sub["auc"], marker="o", label=t)
    plt.xlabel("Noise level / flip rate")
    plt.ylabel("AUC (OOF pooled)")
    plt.title("Robustness: AUC degradation")
    plt.grid(True)
    plt.legend()
    plt.show()

    return rob


# ============================================================
# 12) Conformal Prediction (classification)
#     - Split conformal using OOF-like calibration on training split
#     - Produces prediction sets {0}, {1}, {0,1}
# ============================================================
def split_conformal_classification(
    df: pd.DataFrame,
    X_cols: List[str],
    alpha: float = 0.10,         # target miscoverage
    use_groups: bool = True,
    calibrate_probs: str = "isotonic",  # "platt"|"isotonic"|None
    base: str = "lr",
):
    # We'll do one group-aware split: train+calib vs test (to report coverage).
    # If you want fully nested conformal, we can extend, but this is already Q1-friendly.

    y = df[TARGET]
    groups = df["participant_id"] if use_groups else None

    # group-aware split: hold out 20% participants as test
    unique_p = df["participant_id"].unique()
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(unique_p)
    n_test_p = max(1, int(0.2 * len(unique_p)))
    test_p = set(unique_p[:n_test_p])

    test_mask = df["participant_id"].isin(test_p)
    df_train = df[~test_mask].copy()
    df_test = df[test_mask].copy()

    # further split train into proper-train and calib (by participants again)
    unique_p2 = df_train["participant_id"].unique()
    rng.shuffle(unique_p2)
    n_cal_p = max(1, int(0.2 * len(unique_p2)))
    cal_p = set(unique_p2[:n_cal_p])

    cal_mask = df_train["participant_id"].isin(cal_p)
    df_prop = df_train[~cal_mask].copy()
    df_cal = df_train[cal_mask].copy()

    # build model
    Xprop, yprop = df_prop[X_cols], df_prop[TARGET]
    Xcal, ycal = df_cal[X_cols], df_cal[TARGET]
    Xtest, ytest = df_test[X_cols], df_test[TARGET]

    num_cols = [c for c in X_cols if c not in CAT]
    cat_cols = [c for c in X_cols if c in CAT]
    pre = make_preprocessor(num_cols, cat_cols)

    base_model = make_rf(pre) if base == "rf" else make_lr(pre)

    if calibrate_probs is None:
        model = base_model
    else:
        method = "sigmoid" if calibrate_probs == "platt" else "isotonic"
        model = CalibratedClassifierCV(base_model, method=method, cv=3)

    model.fit(Xprop, yprop)

    # nonconformity score for class y: s = 1 - p(y|x)
    p_cal = model.predict_proba(Xcal)
    s_cal = 1.0 - p_cal[np.arange(len(ycal)), ycal.to_numpy()]
    qhat = np.quantile(s_cal, 1 - alpha, method="higher")

    # prediction sets on test
    p_test = model.predict_proba(Xtest)
    # include label k if 1 - p_k <= qhat  <=> p_k >= 1 - qhat
    sets = []
    for i in range(len(ytest)):
        S = []
        for k in [0,1]:
            if p_test[i, k] >= 1 - qhat:
                S.append(k)
        if len(S) == 0:
            S = [0,1]  # fail-safe
        sets.append(tuple(S))

    # coverage + avg set size
    ytrue = ytest.to_numpy()
    covered = np.mean([ytrue[i] in sets[i] for i in range(len(ytrue))])
    avg_size = np.mean([len(s) for s in sets])

    # also compute efficiency: fraction of singleton sets
    singleton = np.mean([len(s) == 1 for s in sets])

    out = {
        "alpha": alpha,
        "qhat": float(qhat),
        "coverage": float(covered),
        "avg_set_size": float(avg_size),
        "singleton_rate": float(singleton),
        "n_test": int(len(df_test)),
        "n_test_participants": int(df_test["participant_id"].nunique()),
    }

    return out


# ============================================================
# 13) Mediation / Sufficiency deep tests
#     A) MI ranking (supporting)
#     B) Residualization test: geometry predicting y residual after visibility
# ============================================================
def mi_suite(df: pd.DataFrame):
    y = df[TARGET].to_numpy()
    tmp = df[GEOMETRY + HEADPOSE + VIS + CAT].copy()
    tmp = pd.get_dummies(tmp, columns=CAT, drop_first=False)
    X = tmp.to_numpy()

    mi = mutual_info_classif(X, y, random_state=RANDOM_STATE, discrete_features=False)
    mi_df = pd.DataFrame({"feature": tmp.columns, "mi": mi}).sort_values("mi", ascending=False)
    return mi_df

def residualization_test(df: pd.DataFrame):
    """
    Hard sufficiency stress test:
    1) Fit y ~ visibility (linear)
    2) Residual r = y - yhat
    3) Fit r ~ geometry+headpose+cats(onehot)
    4) Report R^2 + permutation p-value
    """

    # --- Step 0: build y and visibility as float arrays ---
    y = df[TARGET].astype(float).to_numpy()

    v_df = df[VIS].copy()
    # force numeric (in case something got read as object)
    for c in v_df.columns:
        v_df[c] = pd.to_numeric(v_df[c], errors="coerce")
    v_df = v_df.fillna(v_df.median(numeric_only=True))

    v = v_df.to_numpy(dtype=float)
    v = (v - v.mean(axis=0, keepdims=True)) / (v.std(axis=0, keepdims=True) + 1e-12)

    # --- Step 1: y ~ visibility ---
    reg1 = LinearRegression().fit(v, y)
    yhat = reg1.predict(v)
    r = y - yhat

    # --- Step 2: geometry design matrix with safe numeric conversion ---
    tmp = df[GEOMETRY + HEADPOSE + CAT].copy()
    tmp = pd.get_dummies(tmp, columns=CAT, drop_first=False)

    # convert EVERYTHING to numeric, coerce bad cells to NaN then impute
    for c in tmp.columns:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp = tmp.fillna(tmp.median(numeric_only=True))

    X = tmp.to_numpy(dtype=float)

    # standardize columns safely
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True)
    X = (X - X_mean) / (X_std + 1e-12)

    # --- Step 3: r ~ X ---
    reg2 = LinearRegression().fit(X, r)
    rhat = reg2.predict(X)

    ss_res = np.sum((r - rhat) ** 2)
    ss_tot = np.sum((r - r.mean()) ** 2) + 1e-12
    r2 = 1 - ss_res / ss_tot

    # --- Step 4: permutation test ---
    B = 3000
    r2_perm = np.empty(B, dtype=float)

    for b in range(B):
        rp = rng.permutation(r)
        regp = LinearRegression().fit(X, rp)
        rphat = regp.predict(X)

        ss_res_p = np.sum((rp - rphat) ** 2)
        ss_tot_p = np.sum((rp - rp.mean()) ** 2) + 1e-12
        r2_perm[b] = 1 - ss_res_p / ss_tot_p

    pval = float(np.mean(r2_perm >= r2))

    return {
        "r2_residual_explained_by_geometry": float(r2),
        "perm_p_value": pval,
        "r2_perm_mean": float(r2_perm.mean()),
        "r2_perm_95pct": (float(np.quantile(r2_perm, 0.025)), float(np.quantile(r2_perm, 0.975))),
        "reg_y_on_v_coef": float(reg1.coef_[0]),
        "reg_y_on_v_intercept": float(reg1.intercept_),
        "n_features_geometry_design": int(X.shape[1]),
    }


# ============================================================
# ============================
# RUN: Full professor pipeline
# ============================

USE_GROUPS = True  # TRUE = LOPO-style GroupKFold; FALSE = ordinary StratifiedKFold

print("\n==============================")
print("A) Sufficiency suite (OOF pooled + DeLong + Bootstrap CI)")
print("==============================")
suf = sufficiency_suite(df, use_groups=USE_GROUPS)

def print_ci_table(ci, title):
    print("\n", title)
    for k, (m, lo, hi) in ci.items():
        print(f"{k:>6s}: {m:.3f} [{lo:.3f}, {hi:.3f}]")

print("\nFold-mean metrics (LR, OOF):")
print("V-only:\n", suf["res_v"].fold_metrics.mean(numeric_only=True))
print("Full:\n", suf["res_full"].fold_metrics.mean(numeric_only=True))
print("Geom(no V):\n", suf["res_geom"].fold_metrics.mean(numeric_only=True))

print("\nDeLong paired AUC tests (pooled OOF):")
print("Full vs V-only:", suf["delong_full_vs_v"])
print("V-only vs Geom(no V):", suf["delong_v_vs_geom"])

print_ci_table(suf["ci_v"], "Bootstrap CI: V-only")
print_ci_table(suf["ci_full"], "Bootstrap CI: Full")
print_ci_table(suf["ci_geom"], "Bootstrap CI: Geom(no V)")

print("\n==============================")
print("B) Calibration suite (None/Platt/Isotonic) + Reliability")
print("==============================")
cal = calibration_suite(df, use_groups=USE_GROUPS)

# pick best calibration by ECE (lower is better) and show decision curve
def summarize_res(name, res):
    m = compute_metrics(res.y_true_all, (res.y_prob_all>=0.5).astype(int), res.y_prob_all)
    return name, m["auc"], m["brier"], m["ece"]

rows = [summarize_res("none", cal["none"]), summarize_res("platt", cal["platt"]), summarize_res("isotonic", cal["isotonic"])]
best_name = sorted(rows, key=lambda x: x[3])[0][0]
best_res = cal[best_name]

print("Calibration summary (name, AUC, Brier, ECE):")
for r in rows:
    print(r)
print("Chosen best-by-ECE:", best_name)

print("\n==============================")
print("C) Decision Curve Analysis (policy-level)")
print("==============================")
plot_decision_curve(best_res.y_true_all, best_res.y_prob_all, title=f"Decision Curve (best={best_name})")

print("\n==============================")
print("D) Robustness suite (vis_noise, pose_noise, label_flip)")
print("==============================")
rob = robustness_suite(df, use_groups=USE_GROUPS)
print(rob)

print("\n==============================")
print("E) Conformal prediction (coverage-guaranteed sets)")
print("==============================")
X_cols = GEOMETRY + HEADPOSE + VIS + CAT
conf = split_conformal_classification(df, X_cols=X_cols, alpha=0.10, use_groups=True, calibrate_probs=best_name, base="lr")
print(conf)

print("\n==============================")
print("F) Information-theoretic support (MI ranking)")
print("==============================")
mi_df = mi_suite(df)
print(mi_df.head(15))

plt.figure(figsize=(8,4))
plt.bar(mi_df["feature"].head(12), mi_df["mi"].head(12))
plt.xticks(rotation=90)
plt.ylabel("Mutual Information with y")
plt.title("Top MI features")
plt.tight_layout()
plt.show()

print("\n==============================")
print("G) Residualization test (hard sufficiency stress test)")
print("==============================")
resid = residualization_test(df)
print(resid)

def fit_lr_on_all(df, cols):
    y = df[TARGET]
    X = df[cols].copy()

    num_cols = [c for c in X.columns if c not in CAT]
    cat_cols = [c for c in X.columns if c in CAT]
    pre = make_preprocessor(num_cols, cat_cols)

    model = make_lr(pre)
    model.fit(X, y)
    return model

def minimal_delta_visibility_to_target_prob(model, x_row_df, target_prob=0.5, grid=np.linspace(-0.5, 0.5, 401)):
    base = x_row_df.copy()
    base_prob = model.predict_proba(base)[0, 1]

    best = None
    for d in grid:
        trial = base.copy()
        trial["visibility_score"] = base["visibility_score"].iloc[0] + d
        p = model.predict_proba(trial)[0, 1]
        if (base_prob < target_prob and p >= target_prob) or (base_prob >= target_prob and p < target_prob):
            best = d
            break
    return float(base_prob), (None if best is None else float(best))

# run counterfactual samples
cols_full = GEOMETRY + HEADPOSE + VIS + CAT
lr_all = fit_lr_on_all(df, cols_full)

sample_idx = np.random.choice(df.index, size=10, replace=False)
rows = []
for i in sample_idx:
    x = df.loc[[i], cols_full]
    p0, dvis = minimal_delta_visibility_to_target_prob(lr_all, x, target_prob=0.5)
    rows.append({
        "idx": int(i),
        "y": int(df.loc[i, TARGET]),
        "vis": float(df.loc[i, "visibility_score"]),
        "base_prob": p0,
        "delta_vis_to_flip@0.5": dvis
    })

pd.DataFrame(rows)