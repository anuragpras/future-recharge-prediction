"""
Train a single-day recharge propensity model.

Use case:
- Input training data contains users who recharged on one cohort day.
- Target column marks whether each user recharged again in the future observation window.
- Model outputs probability scores for another single-day recharge cohort.
"""

from pathlib import Path
import argparse
import joblib
import yaml
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import (
    ensure_dir,
    get_feature_columns,
    add_score_deciles,
    make_decile_distribution,
    make_score_bucket_summary,
    make_feature_bucket_analysis,
)

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pipeline(numeric_features, categorical_features, random_state=42):
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=150,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def feature_importance_table(pipeline, numeric_features, categorical_features):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    encoded_cat_features = []
    if categorical_features:
        encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        encoded_cat_features = encoder.get_feature_names_out(categorical_features).tolist()

    feature_names = numeric_features + encoded_cat_features
    importance = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    df["importance_pct"] = df["importance"] / max(df["importance"].sum(), 1) * 100
    return df.sort_values("importance_pct", ascending=False).round(4)


def run(config_path: str):
    config = load_config(Path(config_path))

    train_file = ROOT / config["paths"]["train_file"]
    predict_file = ROOT / config["paths"]["predict_file"]
    output_dir = ROOT / config["paths"]["output_dir"]
    model_dir = ROOT / config["paths"]["model_dir"]
    ensure_dir(output_dir)
    ensure_dir(model_dir)

    user_id_col = config["columns"]["user_id"]
    target_col = config["columns"]["target"]
    cohort_date_col = config["columns"]["cohort_date"]
    random_state = config["model"]["random_state"]
    test_size = config["model"]["test_size"]
    top_pct = config["model"]["top_percent_export"]

    train_df = pd.read_csv(train_file)
    predict_df = pd.read_csv(predict_file)

    feature_cols = get_feature_columns(train_df, user_id_col, target_col, cohort_date_col)
    X = train_df[feature_cols]
    y = train_df[target_col]
    X_predict = predict_df[feature_cols]

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline(numeric_features, categorical_features, random_state)
    pipeline.fit(X_train, y_train)

    train_scores = pipeline.predict_proba(X_train)[:, 1]
    test_scores = pipeline.predict_proba(X_test)[:, 1]
    predict_scores = pipeline.predict_proba(X_predict)[:, 1]

    test_preds_50 = (test_scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds_50).ravel()
    metrics = pd.DataFrame([{
        "model": "LightGBM",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "prediction_rows": len(X_predict),
        "auc_roc": round(roc_auc_score(y_test, test_scores), 4),
        "accuracy_at_0_50": round(accuracy_score(y_test, test_preds_50), 4),
        "precision_at_0_50": round(precision_score(y_test, test_preds_50, zero_division=0), 4),
        "recall_at_0_50": round(recall_score(y_test, test_preds_50, zero_division=0), 4),
        "f1_at_0_50": round(f1_score(y_test, test_preds_50, zero_division=0), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }])

    train_scored = X_train.copy()
    train_scored[user_id_col] = train_df.loc[X_train.index, user_id_col].values
    train_scored[target_col] = y_train.values
    train_scored["propensity_score"] = train_scores

    test_scored = X_test.copy()
    test_scored[user_id_col] = train_df.loc[X_test.index, user_id_col].values
    test_scored[target_col] = y_test.values
    test_scored["propensity_score"] = test_scores

    pred_scored = predict_df[[user_id_col, cohort_date_col]].copy()
    pred_scored["propensity_score"] = predict_scores
    pred_scored = add_score_deciles(pred_scored, "propensity_score")
    pred_scored = pred_scored.sort_values("propensity_score", ascending=False)

    top_n = max(1, int(len(pred_scored) * top_pct / 100))
    top_users = pred_scored.head(top_n)

    feature_importance = feature_importance_table(pipeline, numeric_features, categorical_features)

    train_deciles = make_decile_distribution(train_scored, "propensity_score", target_col, "train")
    test_deciles = make_decile_distribution(test_scored, "propensity_score", target_col, "test")
    pred_deciles = make_decile_distribution(pred_scored, "propensity_score", None, "predict")
    percentile_distribution = pd.concat([train_deciles, test_deciles, pred_deciles], ignore_index=True)

    test_bucket_summary = make_score_bucket_summary(test_scored, "propensity_score", target_col)
    predict_bucket_summary = make_score_bucket_summary(pred_scored, "propensity_score", None)

    feature_level_analysis = make_feature_bucket_analysis(
        pd.concat([
            test_scored[feature_cols + [target_col, "propensity_score"]],
        ], ignore_index=True),
        feature_cols=feature_cols,
        score_col="propensity_score",
    )

    pred_scored.to_csv(output_dir / "all_predicted_users.csv", index=False)
    top_users.to_csv(output_dir / f"top_{top_pct}pct_users.csv", index=False)
    joblib.dump(pipeline, model_dir / "single_day_recharge_propensity_model.pkl")

    report_path = output_dir / "model_analysis_report.xlsx"
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Model_Metrics", index=False)
        feature_importance.to_excel(writer, sheet_name="Feature_Importance", index=False)
        percentile_distribution.to_excel(writer, sheet_name="Percentile_Distribution", index=False)
        test_bucket_summary.to_excel(writer, sheet_name="Test_Score_Buckets", index=False)
        predict_bucket_summary.to_excel(writer, sheet_name="Predict_Score_Buckets", index=False)
        feature_level_analysis.to_excel(writer, sheet_name="Feature_Level_Buckets", index=False)

    print("Training complete")
    print(f"Model saved: {model_dir / 'single_day_recharge_propensity_model.pkl'}")
    print(f"Predictions saved: {output_dir / 'all_predicted_users.csv'}")
    print(f"Top users saved: {output_dir / f'top_{top_pct}pct_users.csv'}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()
    run(args.config)
