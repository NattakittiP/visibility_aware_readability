import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, average_precision_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CSV_PATH = "reading_visibility_dataset_300rows.csv"

df = pd.read_csv(CSV_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# ensure dtypes
df["can_read"] = df["can_read"].astype(int)
df["medium"] = df["medium"].astype(str)
df["contrast"] = df["contrast"].astype(str)

num_cols = [
    "distance_m", "font_size_pt", "text_height_mm",
    "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
    "angular_size_deg", "visibility_score", "can_read"
]

df_corr = df.copy()
df_corr = pd.get_dummies(df_corr, columns=["medium", "contrast"], drop_first=False)

corr = df_corr.corr()

plt.figure(figsize=(10, 10))
im = plt.imshow(corr, interpolation="nearest")
plt.title("Correlation heatmap (numeric & dummies)")
plt.colorbar(im, fraction=0.046, pad=0.04)

ticks = np.arange(len(corr.columns))
plt.xticks(ticks, corr.columns, rotation=90)
plt.yticks(ticks, corr.columns)

plt.tight_layout()
plt.savefig("fig_corr_heatmap.png", dpi=300)
plt.show()

# ============================================================
# 3. Scatter plots 2D & 3D (for figures)
# ============================================================

scatter_cols = [
    "distance_m", "font_size_pt", "angular_size_deg",
    "head_yaw_deg", "head_pitch_deg", "head_roll_deg"
]

# 3.1 vs visibility_score
n = len(scatter_cols)
ncols = 3
nrows = int(np.ceil(n / ncols))

plt.figure(figsize=(5 * ncols, 4 * nrows))
for i, col in enumerate(scatter_cols, 1):
    plt.subplot(nrows, ncols, i)
    plt.scatter(df[col], df["visibility_score"], alpha=0.7)
    plt.xlabel(col)
    plt.ylabel("visibility_score")
    plt.title(f"{col} vs visibility_score")
    plt.grid(True)

plt.tight_layout()
plt.savefig("fig_scatter_vs_visibility.png", dpi=300)
plt.show()

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    df["distance_m"],
    df["angular_size_deg"],
    df["font_size_pt"],
    c=df["visibility_score"],
    cmap="viridis",
    alpha=0.9
)
ax.set_xlabel("distance_m")
ax.set_ylabel("angular_size_deg")
ax.set_zlabel("font_size_pt")
ax.set_title("3D: distance vs angular vs font_size (color = visibility_score)")

cbar = fig.colorbar(sc, pad=0.1)
cbar.set_label("visibility_score")

plt.tight_layout()
plt.savefig("fig_3d_visibility.png", dpi=300)
plt.show()

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

sizes = 20 + 120 * (
    (df["angular_size_deg"] - df["angular_size_deg"].min()) /
    (df["angular_size_deg"].max() - df["angular_size_deg"].min())
)

sc = ax.scatter(
    df["distance_m"],
    df["font_size_pt"],
    df["visibility_score"],
    c=df["can_read"],
    s=sizes,
    cmap="viridis",
    alpha=0.9
)
ax.set_xlabel("distance_m")
ax.set_ylabel("font_size_pt")
ax.set_zlabel("visibility_score")
ax.set_title("3D+size: distance, font_size, visibility (size=angular_size, color=can_read)")

cbar = fig.colorbar(sc, pad=0.1)
cbar.set_label("can_read")

plt.tight_layout()
plt.savefig("fig_3d_cheating.png", dpi=300)
plt.show()

# ============================================================
# 4. Train-test split + preprocessing
# ============================================================

X = df.drop(columns=["can_read"])
y = df["can_read"]

numeric_features = [
    "distance_m", "font_size_pt", "text_height_mm",
    "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
    "angular_size_deg", "visibility_score"
]
categorical_features = ["medium", "contrast"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Train size:", X_train.shape, "Test size:", X_test.shape)

# ============================================================
# 5. Define models (Logistic Regression & Random Forest)
# ============================================================

log_reg = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]
)

rf_clf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE
        ))
    ]
)

models = {
    "Logistic Regression": log_reg,
    "Random Forest": rf_clf
}

# ============================================================
# 6. Train & evaluate models
# ============================================================

metrics_rows = []

plt.figure(figsize=(8, 6))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    metrics_rows.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "ap": ap,
        "brier": brier
    })

    # ROC curve
    RocCurveDisplay.from_predictions(
        y_test, y_proba, name=f"{name} (AUC={auc:.3f})", ax=plt.gca()
    )

plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC curves: can_read classifier")
plt.grid(True)
plt.tight_layout()
plt.savefig("fig_roc.png", dpi=300)
plt.show()

metrics_df = pd.DataFrame(metrics_rows)
print("\n=== Performance metrics on test set ===")
print(metrics_df)
metrics_df.to_csv("metrics_summary.csv", index=False)

# Confusion matrices (separate images)
for name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"])
    plt.yticks(tick_marks, ["0", "1"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    fname = f"fig_confusion_{name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=300)
    plt.show()

# ============================================================
# 7. Calibration ploะ
# ============================================================

rf_clf.fit(X_train, y_train)
y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]

prob_true, prob_pred = calibration_curve(y_test, y_proba_rf, n_bins=10)

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, "o-", label="Random Forest")
plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration curve (Random Forest)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig_calibration_rf.png", dpi=300)
plt.show()

# ============================================================
# 8. Feature importance & permutation importance (Random Forest)
# ============================================================


rf_clf.fit(X_train, y_train)
preproc = rf_clf.named_steps["preprocess"]
rf_model = rf_clf.named_steps["model"]

# helper: get feature names
def get_feature_names(preprocessor, numeric_features, categorical_features):
    output = []
    # numeric
    output.extend(numeric_features)
    # categorical
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(categorical_features)
    output.extend(list(cat_names))
    return output

feat_names = get_feature_names(preproc, numeric_features, categorical_features)

# 8.1 tree-based feature_importances_
importances = rf_model.feature_importances_
fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False)

print("\nTree-based feature importances (top 20):")
print(fi_df.head(20))

plt.figure(figsize=(10, 5))
plt.bar(fi_df["feature"], fi_df["importance"])
plt.xticks(rotation=90)
plt.ylabel("importance")
plt.title("Random Forest feature importances")
plt.tight_layout()
plt.savefig("fig_feature_importance.png", dpi=300)
plt.show()

# 8.2 permutation importance 
X_test_trans = preproc.transform(X_test)

perm = permutation_importance(
    rf_model, X_test_trans, y_test,
    n_repeats=30,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "feature": feat_names,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values("importance_mean", ascending=False)

print("\nPermutation importance (top 20):")
print(perm_df.head(20))

plt.figure(figsize=(10, 5))
plt.bar(perm_df["feature"], perm_df["importance_mean"],
        yerr=perm_df["importance_std"])
plt.xticks(rotation=90)
plt.ylabel("mean decrease in score")
plt.title("Permutation importance (Random Forest, test set)")
plt.tight_layout()
plt.savefig("fig_permutation_importance.png", dpi=300)
plt.show()

# ============================================================
# 9. Partial Dependence Plots (PDP) – key features
# ============================================================

key_features = [
    "visibility_score",
    "angular_size_deg",
    "distance_m",
    "font_size_pt"
]

# 9.1 1D PDP
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
PartialDependenceDisplay.from_estimator(
    rf_clf,
    X_train,
    features=key_features,
    kind="average",
    ax=ax
)
plt.suptitle("Partial dependence of key features (Random Forest)")
plt.tight_layout()
plt.savefig("fig_pdp_1d.png", dpi=300)
plt.show()

# 9.2 2D PDP pairs
pairs = [
    ("distance_m", "font_size_pt"),
    ("distance_m", "visibility_score"),
    ("angular_size_deg", "visibility_score"),
]

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
PartialDependenceDisplay.from_estimator(
    rf_clf,
    X_train,
    features=pairs,
    kind="average",
    ax=ax
)
plt.suptitle("2D partial dependence surfaces (Random Forest)")
plt.tight_layout()
plt.savefig("fig_pdp_2d.png", dpi=300)
plt.show()

# ============================================================
# END
# ============================================================

