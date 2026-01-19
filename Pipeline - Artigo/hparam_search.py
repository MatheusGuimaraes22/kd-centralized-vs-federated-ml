import sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.ckd_paper_kfold import load_data, build_preprocessor  # type: ignore
from scripts.feature_selection import load_drop_cols  # type: ignore
RANDOM_STATE = 42
DROP_COLS = load_drop_cols(["sg", "pcc", "pcv", "hemo", "rbcc", "al"], k=6)
SCENARIOS = [
    ("all_nosmote", False, []),
    ("all_smote", True, []),
    ("drop_nosmote", False, DROP_COLS),
    ("drop_smote", True, DROP_COLS),
]

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

rf_param = {
    "model__n_estimators": [100, 200, 400],
    "model__max_depth": [None, 4, 6, 8],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"]
}
xgb_param = {
    "model__n_estimators": [150, 300, 500],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__subsample": [0.7, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.9, 1.0]
}
lr_param = {
    "model__C": [0.1, 1.0, 10.0],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"],
    "model__class_weight": [None, "balanced"]
}
mlp_param = {
    "model__hidden_layer_sizes": [(64, 32), (64, 64), (128, 64)],
    "model__alpha": [1e-4, 1e-3, 1e-2],
    "model__learning_rate_init": [1e-3, 1e-2],
}
dt_param = {
    "model__max_depth": [None, 3, 5, 7],
    "model__min_samples_leaf": [1, 2, 4],
    "model__min_samples_split": [2, 4, 8],
    "model__criterion": ["gini", "entropy"],
}
gbm_param = {
    "model__n_estimators": [100, 200, 400],
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__max_depth": [2, 3, 4],
    "model__subsample": [0.7, 0.9, 1.0],
}
svm_param = {
    "model__C": [0.1, 1.0, 10.0],
    "model__gamma": ["scale", "auto"],
    "model__kernel": ["rbf"],
}
knn_param = {
    "model__n_neighbors": [3, 5, 7, 9],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],
}

results = []
for label, use_smote, drops in SCENARIOS:
    X, y = load_data()
    if drops:
        X = X.drop(columns=drops)
    # remove colunas 100% NaN
    X = X.loc[:, X.notna().any()]

    # preprocessor para dados sem SMOTE
    numeric_cols = [c for c in ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc"] if c in X.columns]
    cat_cols = [c for c in X.columns if c not in numeric_cols]
    pre = build_preprocessor(numeric_cols, cat_cols)

    # pipelines
    rf = ImbPipeline([("prep", pre), ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))])
    lr = ImbPipeline([("prep", pre), ("model", LogisticRegression(max_iter=200, n_jobs=-1))])
    mlp = ImbPipeline([("prep", pre), ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=RANDOM_STATE))])
    dt = ImbPipeline([("prep", pre), ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))])
    gbm = ImbPipeline([("prep", pre), ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))])
    svm = ImbPipeline([("prep", pre), ("model", SVC(probability=True, random_state=RANDOM_STATE))])
    knn = ImbPipeline([("prep", pre), ("model", KNeighborsClassifier(n_neighbors=5))])
    if XGB_OK:
        xgb = ImbPipeline([("prep", pre), ("model", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1))])

    # se SMOTE, insere antes do modelo
    if use_smote:
        rf = ImbPipeline([("prep", pre), ("smote", SMOTE(random_state=RANDOM_STATE)), ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))])
        lr = ImbPipeline([("prep", pre), ("smote", SMOTE(random_state=RANDOM_STATE)), ("model", LogisticRegression(max_iter=200, n_jobs=-1))])
        mlp = ImbPipeline([("prep", pre), ("smote", SMOTE(random_state=RANDOM_STATE)), ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=RANDOM_STATE))])
        dt = ImbPipeline([("prep", pre), ("smote", SMOTE(random_state=RANDOM_STATE)), ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))])
        gbm = ImbPipeline([("prep", pre), ("smote", SMOTE(random_state=RANDOM_STATE)), ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))])
        svm = ImbPipeline([("prep", pre), ("smote", SMOTE(random_state=RANDOM_STATE)), ("model", SVC(probability=True, random_state=RANDOM_STATE))])
        knn = ImbPipeline([("prep", pre), ("smote", SMOTE(random_state=RANDOM_STATE)), ("model", KNeighborsClassifier(n_neighbors=5))])
        if XGB_OK:
            xgb = ImbPipeline([("prep", pre), ("smote", SMOTE(random_state=RANDOM_STATE)), ("model", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1))])

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "recall": "recall",
        "accuracy": "accuracy",
    }

    searches = [
        ("mlp", mlp, mlp_param, 20),
        ("dt", dt, dt_param, 20),
        ("rf", rf, rf_param, 20),
        ("gbm", gbm, gbm_param, 20),
        ("lr", lr, lr_param, 15),
        ("svm", svm, svm_param, 15),
        ("knn", knn, knn_param, 15),
    ]
    if XGB_OK:
        searches.append(("xgb", xgb, xgb_param, 20))

    for name, pipe, grid, n_iter in searches:
        search = RandomizedSearchCV(
            pipe,
            grid,
            n_iter=n_iter,
            scoring=scoring,
            refit="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X, y)
        best_idx = search.best_index_
        best_score = search.best_score_
        cvr = search.cv_results_
        results.append({
            "scenario": label,
            "model": name,
            "best_params": search.best_params_,
            "roc_auc_cv": best_score,
            "f1_cv": float(cvr["mean_test_f1"][best_idx]),
            "recall_cv": float(cvr["mean_test_recall"][best_idx]),
            "accuracy_cv": float(cvr["mean_test_accuracy"][best_idx]),
        })
        print(f"[{label}] {name} best ROC-AUC: {best_score:.4f}")
        print(f"Params: {search.best_params_}")

Path("results").mkdir(exist_ok=True)
pd.DataFrame(results).to_json("results/hparam_results.json", orient="records", indent=2)
pd.DataFrame(results).to_csv("results/hparam_results.csv", index=False)
print("Salvo: results/hparam_results.json e .csv")

