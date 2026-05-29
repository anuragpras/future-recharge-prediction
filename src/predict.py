"""Score a new single-day recharge cohort using a saved model."""

from pathlib import Path
import argparse
import joblib
import pandas as pd
from utils import add_score_deciles, ensure_dir

ROOT = Path(__file__).resolve().parents[1]


def run(model_path, input_file, output_file, user_id_col="user_id", cohort_date_col="cohort_recharge_date"):
    model = joblib.load(model_path)
    df = pd.read_csv(input_file)
    meta_cols = [c for c in [user_id_col, cohort_date_col] if c in df.columns]
    feature_df = df.drop(columns=meta_cols, errors="ignore")

    scores = model.predict_proba(feature_df)[:, 1]
    out = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)
    out["propensity_score"] = scores
    out = add_score_deciles(out, "propensity_score").sort_values("propensity_score", ascending=False)

    output_path = Path(output_file)
    ensure_dir(output_path.parent)
    out.to_csv(output_path, index=False)
    print(f"Predictions saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "single_day_recharge_propensity_model.pkl"))
    parser.add_argument("--input", default=str(ROOT / "data" / "sample_predict_single_day_rechargers.csv"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "scored_new_cohort.csv"))
    args = parser.parse_args()
    run(args.model, args.input, args.output)
