# Single-Day Recharge Propensity Model

## Overview

This repository contains an open-source machine learning pipeline for predicting **future recharge propensity** for users who recharged on a specific single cohort day.

The model is useful when a business wants to take all users who recharged on one selected date and predict which of them are most likely to recharge again in a future observation window.

Example use case:

```text
Cohort: Users who recharged on 2026-01-27
Target: Whether the same users recharged again in the next 7 / 14 / 30 days
Output: Propensity score for each user
```

The final output can be used for CRM targeting, remarketing, retention campaigns, user prioritization, and recharge growth experiments.

---

## Problem Statement

Not all users who recharge on a given day have the same likelihood of recharging again.

This model helps answer:

```text
Among users who recharged on a specific single day, who is most likely to recharge again?
```

Instead of targeting the full cohort equally, the model ranks users by predicted recharge propensity so teams can focus campaigns on the highest-intent users.

---

## What This Project Does

The pipeline performs:

- Sample data generation
- Data preprocessing
- Train/test split
- Missing value handling
- Categorical encoding
- LightGBM model training
- Model evaluation
- User-level propensity scoring
- Feature importance reporting
- Train/test/predict percentile distribution
- Score bucket analysis
- Feature-level analysis by probability score buckets
- Top user export for campaigns

---

## Repository Structure

```text
single-day-recharge-propensity-model/
│
├── data/
│   ├── sample_train_single_day_rechargers.csv
│   └── sample_predict_single_day_rechargers.csv
│
├── models/
│   └── single_day_recharge_propensity_model.pkl
│
├── notebooks/
│   └── README.md
│
├── outputs/
│   ├── all_predicted_users.csv
│   ├── top_10pct_users.csv
│   └── model_analysis_report.xlsx
│
├── src/
│   ├── generate_sample_data.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Input Data Logic

### Training Data

The training file should contain users who recharged on a selected cohort day, along with a target column showing whether they recharged again later.

Required columns:

```text
user_id
cohort_recharge_date
future_recharged
```

Target column:

```text
future_recharged

0 = User did not recharge again in the future window
1 = User recharged again in the future window
```

Example:

| user_id | cohort_recharge_date | recharge_amount_on_day | sessions_last_7d | future_recharged |
|---:|---|---:|---:|---:|
| 100001 | 2026-01-27 | 499 | 7 | 1 |
| 100002 | 2026-01-27 | 199 | 2 | 0 |

---

### Prediction Data

The prediction file should contain users from another single-day recharge cohort that needs to be scored.

Required columns:

```text
user_id
cohort_recharge_date
```

It should not contain the target column because the model is predicting that outcome.

Example:

| user_id | cohort_recharge_date | recharge_amount_on_day | sessions_last_7d |
|---:|---|---:|---:|
| 200001 | 2026-02-15 | 399 | 8 |
| 200002 | 2026-02-15 | 149 | 1 |

---

## Sample Features

The sample dataset includes generic recharge, engagement, device, and acquisition features.

### Recharge Features

- recharge_amount_on_day
- recharge_count_on_day
- wallet_balance_after_recharge
- lifetime_recharge_count_before_day
- lifetime_recharge_amount_before_day

### Engagement Features

- sessions_last_7d
- app_launches_last_7d
- notifications_clicked_last_7d
- consultations_last_7d
- paid_consult_minutes_last_7d
- avg_consult_rating

### Offer / Promo Features

- has_used_coupon_on_day
- has_done_promo_recharge
- is_free_consultation_taken

### Device and Acquisition Features

- country
- device_type
- acquisition_source
- campaign_group
- city_tier

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/single-day-recharge-propensity-model.git
cd single-day-recharge-propensity-model
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How To Run

### Step 1: Generate Sample Data

```bash
python src/generate_sample_data.py
```

This creates:

```text
data/sample_train_single_day_rechargers.csv
data/sample_predict_single_day_rechargers.csv
```

---

### Step 2: Train Model and Generate Outputs

```bash
python src/train.py
```

This trains the model and creates:

```text
models/single_day_recharge_propensity_model.pkl
outputs/all_predicted_users.csv
outputs/top_10pct_users.csv
outputs/model_analysis_report.xlsx
```

---

### Step 3: Score a New Cohort

```bash
python src/predict.py \
  --model models/single_day_recharge_propensity_model.pkl \
  --input data/sample_predict_single_day_rechargers.csv \
  --output outputs/scored_new_cohort.csv
```

---

## Output Files

### 1. all_predicted_users.csv

Contains all users from the prediction cohort ranked by propensity score.

Columns:

```text
user_id
cohort_recharge_date
propensity_score
score_decile
```

---

### 2. top_10pct_users.csv

Contains the highest-scoring 10% users from the prediction cohort.

This can be used directly for CRM, push notification, WhatsApp, email, or remarketing campaigns.

---

### 3. model_analysis_report.xlsx

The Excel report contains the following sheets:

#### Model_Metrics

Model performance summary including:

- AUC-ROC
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix counts

#### Feature_Importance

Feature importance from the LightGBM model.

Useful for understanding which variables are driving recharge propensity.

#### Percentile_Distribution

Decile-level distribution for:

- Train set
- Test set
- Prediction set

This helps validate whether the model is ranking users properly across datasets.

#### Test_Score_Buckets

Performance of the test set across probability score buckets:

```text
0.80-1.00
0.60-0.80
0.40-0.60
0.20-0.40
0.00-0.20
```

Includes user count, average score, and actual recharge rate.

#### Predict_Score_Buckets

Distribution of prediction users across probability buckets.

Useful for campaign volume planning.

#### Feature_Level_Buckets

Feature-level analysis by score bucket.

This shows how numeric and categorical features behave across high-score and low-score users.

Examples:

- Average recharge amount by score bucket
- Average session count by score bucket
- Acquisition source mix by score bucket
- Device type mix by score bucket

---

## Model Workflow

```text
Single-Day Recharge Cohort
        ↓
Feature Preparation
        ↓
Train/Test Split
        ↓
Preprocessing
        ↓
LightGBM Training
        ↓
Model Evaluation
        ↓
Future Recharge Propensity Scoring
        ↓
Decile and Score Bucket Analysis
        ↓
Campaign-Ready User Export
```

---

## Example Business Usage

### CRM Targeting

Target users in the top propensity deciles with recharge reminders or personalized offers.

### Retention Campaigns

Identify users from a recharge cohort who are likely to repeat and nudge them before they become inactive.

### Offer Optimization

Compare high-score and low-score users to decide who should receive discounts, coupons, or non-discount nudges.

### Campaign Prioritization

Use the top 10% or top 20% ranked users as the first campaign audience.

---

## Configuration

Default settings are stored in:

```text
config.yaml
```

You can change:

- Input training file path
- Input prediction file path
- Output directory
- Model directory
- Target column name
- Test split percentage
- Top user export percentage

---

## Important Notes

This is a generic open-source implementation.

The included sample data is synthetic and does not contain real user information.

For a real business use case, define the target window clearly. For example:

```text
future_recharged = 1 if user recharged again within 7 days after the cohort recharge date
future_recharged = 0 otherwise
```

Keep the same target definition across training, testing, and production scoring.

---

## Future Improvements

Potential improvements:

- SHAP explainability
- Hyperparameter tuning
- XGBoost and CatBoost benchmarking
- Model monitoring
- Cohort drift analysis
- Automated campaign audience export
- MLflow experiment tracking
- API-based scoring

---

## License

MIT License
