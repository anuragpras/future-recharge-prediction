"""
Generate synthetic sample data for the Single-Day Recharge Propensity Model.

The sample data represents users who recharged on a selected cohort day.
The target indicates whether the user recharged again in a future observation window.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


def make_dataset(n_rows: int, include_target: bool = True) -> pd.DataFrame:
    user_id = np.arange(100000, 100000 + n_rows)

    recharge_amount_on_day = rng.gamma(shape=2.2, scale=250, size=n_rows).round(0)
    recharge_count_on_day = rng.choice([1, 2, 3, 4], size=n_rows, p=[0.72, 0.18, 0.07, 0.03])
    wallet_balance_after_recharge = rng.gamma(shape=1.8, scale=120, size=n_rows).round(0)
    days_since_signup = rng.integers(0, 540, size=n_rows)
    lifetime_recharge_count_before_day = rng.poisson(2.2, size=n_rows)
    lifetime_recharge_amount_before_day = (
        lifetime_recharge_count_before_day * rng.gamma(shape=2.0, scale=300, size=n_rows)
    ).round(0)
    sessions_last_7d = rng.poisson(4.5, size=n_rows)
    app_launches_last_7d = rng.poisson(6.0, size=n_rows)
    notifications_clicked_last_7d = rng.poisson(1.1, size=n_rows)
    consultations_last_7d = rng.poisson(1.5, size=n_rows)
    paid_consult_minutes_last_7d = (consultations_last_7d * rng.gamma(1.7, 5.0, size=n_rows)).round(1)
    avg_consult_rating = np.clip(rng.normal(4.0, 0.9, size=n_rows), 1, 5).round(1)
    has_used_coupon_on_day = rng.choice([0, 1], size=n_rows, p=[0.68, 0.32])
    has_done_promo_recharge = rng.choice([0, 1], size=n_rows, p=[0.76, 0.24])
    is_free_consultation_taken = rng.choice([0, 1], size=n_rows, p=[0.35, 0.65])
    has_push_enabled = rng.choice([0, 1], size=n_rows, p=[0.18, 0.82])

    countries = rng.choice(["India", "USA", "UAE", "Canada", "UK"], size=n_rows, p=[0.76, 0.08, 0.07, 0.05, 0.04])
    device_type = rng.choice(["Android", "iOS", "Web"], size=n_rows, p=[0.78, 0.17, 0.05])
    acquisition_source = rng.choice(["Organic", "Google Ads", "Meta Ads", "Referral", "CRM"], size=n_rows, p=[0.38, 0.22, 0.18, 0.09, 0.13])
    campaign_group = rng.choice(["Brand", "Performance", "Remarketing", "Influencer", "Unmapped"], size=n_rows, p=[0.22, 0.34, 0.18, 0.10, 0.16])
    city_tier = rng.choice(["Tier 1", "Tier 2", "Tier 3"], size=n_rows, p=[0.36, 0.42, 0.22])

    df = pd.DataFrame({
        "user_id": user_id,
        "cohort_recharge_date": "2026-01-27",
        "recharge_amount_on_day": recharge_amount_on_day,
        "recharge_count_on_day": recharge_count_on_day,
        "wallet_balance_after_recharge": wallet_balance_after_recharge,
        "days_since_signup": days_since_signup,
        "lifetime_recharge_count_before_day": lifetime_recharge_count_before_day,
        "lifetime_recharge_amount_before_day": lifetime_recharge_amount_before_day,
        "sessions_last_7d": sessions_last_7d,
        "app_launches_last_7d": app_launches_last_7d,
        "notifications_clicked_last_7d": notifications_clicked_last_7d,
        "consultations_last_7d": consultations_last_7d,
        "paid_consult_minutes_last_7d": paid_consult_minutes_last_7d,
        "avg_consult_rating": avg_consult_rating,
        "has_used_coupon_on_day": has_used_coupon_on_day,
        "has_done_promo_recharge": has_done_promo_recharge,
        "is_free_consultation_taken": is_free_consultation_taken,
        "has_push_enabled": has_push_enabled,
        "country": countries,
        "device_type": device_type,
        "acquisition_source": acquisition_source,
        "campaign_group": campaign_group,
        "city_tier": city_tier,
    })

    if include_target:
        # Synthetic target logic: future recharge is more likely for engaged, higher-value users.
        logit = (
            -2.4
            + 0.004 * df["recharge_amount_on_day"]
            + 0.28 * df["recharge_count_on_day"]
            + 0.08 * df["sessions_last_7d"]
            + 0.10 * df["notifications_clicked_last_7d"]
            + 0.16 * df["consultations_last_7d"]
            + 0.10 * df["has_push_enabled"]
            - 0.18 * df["has_done_promo_recharge"]
            - 0.12 * df["has_used_coupon_on_day"]
            + np.where(df["acquisition_source"].isin(["CRM", "Referral"]), 0.25, 0)
            + np.where(df["device_type"].eq("iOS"), 0.15, 0)
        )
        probability = 1 / (1 + np.exp(-logit))
        df["future_recharged"] = rng.binomial(1, probability)

    return df


if __name__ == "__main__":
    train = make_dataset(5000, include_target=True)
    predict = make_dataset(1500, include_target=False)

    train.to_csv(DATA_DIR / "sample_train_single_day_rechargers.csv", index=False)
    predict.to_csv(DATA_DIR / "sample_predict_single_day_rechargers.csv", index=False)

    print("Sample files created:")
    print(DATA_DIR / "sample_train_single_day_rechargers.csv")
    print(DATA_DIR / "sample_predict_single_day_rechargers.csv")
