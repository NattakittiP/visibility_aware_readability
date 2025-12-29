import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATA_PATH = "reading_visibility_dataset_300rows.csv"

# =========================================================
# Load data
# =========================================================
def load_data():
    return pd.read_csv(DATA_PATH)

# =========================================================
# Preprocessor
# =========================================================
def make_preprocessor(numeric, categorical):
    num_tf = Pipeline([("scaler", StandardScaler())])
    cat_tf = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    transformers = [("num", num_tf, numeric)]
    if categorical:
        transformers.append(("cat", cat_tf, categorical))
    return ColumnTransformer(transformers=transformers)

# =========================================================
# Metric evaluator
# =========================================================
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "acc": accuracy_score(y_test, y_pred),
        "prec": precision_score(y_test, y_pred),
        "rec": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "ap": average_precision_score(y_test, y_proba),
        "brier": brier_score_loss(y_test, y_proba),
    }

    print(f"===== {name} =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")
    print()
    return metrics

# =========================================================
# Ablation Study
# =========================================================
def run_ablation(df):
    target = "can_read"

    geometry = ["distance_m","font_size_pt","text_height_mm","angular_size_deg",
                "head_yaw_deg","head_pitch_deg","head_roll_deg"]
    visibility = ["visibility_score"]
    categorical = ["medium","contrast"]

    configs = {
        "geometry_only": (geometry, categorical),
        "visibility_only": (visibility, []),
        "visibility_plus_geometry": (geometry + visibility, categorical)
    }

    y = df[target]
    results = {}

    for name, (numf, catf) in configs.items():

        print(f"[Ablation] {name}")
        X = df[numf + catf]

        pre = make_preprocessor(numf, catf)
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                     random_state=RANDOM_STATE)
        pipe = Pipeline([("pre", pre), ("clf", clf)])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
        )

        metrics = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)
        results[name] = metrics

    return results

# =========================================================
# Plot Ablation Results
# =========================================================
def plot_ablation(ab_results):
    df = pd.DataFrame(ab_results).T

    plt.figure(figsize=(10,4))
    sns.barplot(x=df.index, y=df["auc"])
    plt.title("Ablation Study – AUC Comparison")
    plt.ylabel("AUC")
    plt.show()

    plt.figure(figsize=(10,4))
    sns.barplot(x=df.index, y=df["brier"])
    plt.title("Ablation Study – Brier Score (Lower = Better)")
    plt.ylabel("Brier Score")
    plt.show()

    plt.figure(figsize=(10,4))
    sns.barplot(x=df.index, y=df["f1"])
    plt.title("Ablation Study – F1 Score Comparison")
    plt.ylabel("F1")
    plt.show()

# =========================================================
# Assign participant IDs (semi-random balance)
# =========================================================
def assign_participants(df, n_participants=20):
    idx = np.arange(len(df))
    np.random.shuffle(idx)

    base = len(df) // n_participants
    rem = len(df) % n_participants

    sizes = np.array([base]*n_participants)
    sizes[:rem] += 1

    pid = np.empty(len(df), dtype=int)
    start = 0
    for p, sz in enumerate(sizes):
        pid[idx[start:start+sz]] = p
        start += sz

    df = df.copy()
    df["participant_id"] = pid
    return df

# =========================================================
# Leave-One-Participant-Out
# =========================================================
def run_group_by_participant(df):
    geometry = ["distance_m","font_size_pt","text_height_mm","angular_size_deg",
                "head_yaw_deg","head_pitch_deg","head_roll_deg"]
    visibility = ["visibility_score"]
    categorical = ["medium","contrast"]

    numf = geometry + visibility
    catf = categorical

    pre = make_preprocessor(numf, catf)
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                 random_state=RANDOM_STATE)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    metrics_per_p = []

    for pid in sorted(df["participant_id"].unique()):
        train = df[df["participant_id"] != pid]
        test  = df[df["participant_id"] == pid]

        X_train = train[numf + catf]
        y_train = train["can_read"]
        X_test = test[numf + catf]
        y_test = test["can_read"]

        print(f"[LOPO] participant {pid}")
        m = evaluate_model(f"participant {pid}", pipe,
                           X_train, X_test, y_train, y_test)

        m["pid"] = pid
        metrics_per_p.append(m)

    return pd.DataFrame(metrics_per_p)

# =========================================================
# Plot participant results
# =========================================================
def plot_groupby(df):
    plt.figure(figsize=(12,4))
    sns.lineplot(x="pid", y="auc", data=df, marker="o")
    plt.title("Leave-One-Participant-Out – AUC per Participant")
    plt.xlabel("Participant ID")
    plt.ylabel("AUC")
    plt.grid()
    plt.show()

    plt.figure(figsize=(10,6))
    sns.boxplot(data=df[["auc","f1","acc"]])
    plt.title("Distribution of Metrics Across Participants")
    plt.show()

# =========================================================
# MAIN
# =========================================================
df = load_data()

print("\n### Running Ablation ###")
ab_results = run_ablation(df)
plot_ablation(ab_results)

print("\n### Running Leave-One-Participant-Out ###")
df_p = assign_participants(df)
lop_results = run_group_by_participant(df_p)
plot_groupby(lop_results)