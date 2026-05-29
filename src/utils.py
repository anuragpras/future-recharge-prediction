from pathlib import Path
import pandas as pd
import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_feature_columns(df: pd.DataFrame, user_id_col: str, target_col: str, cohort_date_col: str):
    exclude = {user_id_col, target_col, cohort_date_col}
    return [c for c in df.columns if c not in exclude]


def add_score_deciles(df: pd.DataFrame, score_col: str = "propensity_score") -> pd.DataFrame:
    out = df.copy()
    out["score_rank"] = out[score_col].rank(method="first", ascending=False)
    out["score_decile"] = pd.qcut(out["score_rank"], 10, labels=list(range(1, 11)))
    out["score_decile"] = out["score_decile"].astype(int)
    out = out.drop(columns=["score_rank"])
    return out


def make_decile_distribution(df, score_col="propensity_score", target_col=None, dataset_name="dataset"):
    temp = add_score_deciles(df, score_col)
    agg = temp.groupby("score_decile").agg(
        users=(score_col, "count"),
        min_score=(score_col, "min"),
        max_score=(score_col, "max"),
        avg_score=(score_col, "mean"),
    ).reset_index()
    agg.insert(0, "dataset", dataset_name)
    if target_col and target_col in temp.columns:
        positives = temp.groupby("score_decile")[target_col].sum().values
        agg["future_rechargers"] = positives
        agg["recharge_rate"] = agg["future_rechargers"] / agg["users"]
        total_pos = max(agg["future_rechargers"].sum(), 1)
        agg["capture_rate"] = agg["future_rechargers"] / total_pos
        agg["cumulative_capture_rate"] = agg["capture_rate"].cumsum()
    return agg.round(4)


def make_score_bucket_summary(df, score_col="propensity_score", target_col=None):
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.00001]
    labels = ["0.00-0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", "0.80-1.00"]
    temp = df.copy()
    temp["score_bucket"] = pd.cut(temp[score_col], bins=bins, labels=labels, include_lowest=True)
    agg = temp.groupby("score_bucket", observed=False).agg(
        users=(score_col, "count"),
        min_score=(score_col, "min"),
        max_score=(score_col, "max"),
        avg_score=(score_col, "mean"),
    ).reset_index()
    if target_col and target_col in temp.columns:
        agg["future_rechargers"] = temp.groupby("score_bucket", observed=False)[target_col].sum().values
        agg["recharge_rate"] = agg["future_rechargers"] / agg["users"].replace(0, np.nan)
    return agg.sort_values("score_bucket", ascending=False).round(4)


def make_feature_bucket_analysis(df, feature_cols, score_col="propensity_score", max_categories=8):
    temp = df.copy()
    temp["score_bucket"] = pd.cut(
        temp[score_col],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.00001],
        labels=["0.00-0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", "0.80-1.00"],
        include_lowest=True,
    )
    rows = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(temp[col]):
            summary = temp.groupby("score_bucket", observed=False)[col].mean().reset_index()
            for _, row in summary.iterrows():
                rows.append({
                    "feature": col,
                    "feature_type": "numeric",
                    "score_bucket": row["score_bucket"],
                    "value": "mean",
                    "metric": round(row[col], 4) if pd.notnull(row[col]) else np.nan,
                })
        else:
            top_values = temp[col].fillna("unknown").astype(str).value_counts().head(max_categories).index
            reduced = temp[col].fillna("unknown").astype(str).where(temp[col].fillna("unknown").astype(str).isin(top_values), "Other")
            cross = pd.crosstab(temp["score_bucket"], reduced, normalize="index")
            for bucket in cross.index:
                for value in cross.columns:
                    rows.append({
                        "feature": col,
                        "feature_type": "categorical",
                        "score_bucket": bucket,
                        "value": value,
                        "metric": round(cross.loc[bucket, value], 4),
                    })
    return pd.DataFrame(rows)
