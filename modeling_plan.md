Modeling Plan \- Bad Mood Prediction
Which model families were tried and why.
How they align with project goals.
Justification for the final chosen model, tied explicitly to KPIs and evaluation strategy.
Notes on what didn’t work and why those approaches were discarded.

# Modeling Plan — Predicting Bad Mood

## Goal and evaluation setup

**Task.** Binary classification of each mood report: bad mood (SAD, TENSE/ANXIOUS, or TIRED = 1) vs. neutral/good mood (ALERT, HAPPY, NEUTRAL, RESTED/RELAXED = 0). The research question is whether Fitbit sleep and exercise data predict bad mood **in people the model has never seen**.

**Data.** 3,937 labelled mood reports from 49 participants in the training file (bad-mood rate 37%). The test file (15 different participants) is held out and used once, for the final model only.

**Features.** Last night's sleep (duration, efficiency, stages, HRV, breathing rate, skin temperature); the **previous day's** activity (`_lag1`), so all predictors come before the mood report; age group, gender, and BMI; and the hour of the report. Days with 1,440 sedentary minutes are treated as non-wear (activity set to missing). Leakage columns (`time_until_*`, `time_since_*`, `GOOD_MOOD`, individual mood items) are excluded.

**Evaluation strategy.** `StratifiedGroupKFold`, 5 folds grouped by participant, with identical folds for every model. Hyperparameters are tuned with nested CV: an inner 3-fold grouped random search runs on the outer-training participants only. All preprocessing (median imputation + missing-value indicators, scaling) sits inside sklearn `Pipeline`s, so it is refit within each fold.

**KPIs (from `kpis.md`).**
- **Primary: accuracy**, i.e. how often bad vs. non-bad mood is predicted correctly. Because only 37% of training reports are bad mood, always predicting "not bad" already reaches about 63% accuracy (≈58% in the test set). Accuracy is therefore always reported next to this **no-information rate**.
- **Sensitivity trade-off:** **hit rate** (share of bad moods caught) and **false-alarm rate** (share of non-bad reports flagged as bad). Their gap shows whether the model discriminates at all.
- **Threshold-free summary: ROC-AUC**, the area under the hit-rate vs. false-alarm curve across all thresholds (chance = 0.50). Models are ranked on it, and hyperparameters are tuned on it. Tuning on accuracy would favor predicting "not bad" for everyone.
- Also reported: PR-AUC (chance ≈ 0.37), balanced accuracy, precision/F1 for bad mood, and Brier score.
- All 0/1 metrics use a 0.5 probability threshold for now. The team has not yet decided whether false alarms or misses are more costly. When it does, the threshold will be chosen from the training cross-validation predictions, never from the test set.

We also report fold-to-fold SD and compare models fold by fold, since every model is scored on the same folds.

## Model families and why

| Tier | Model | Why it's included |
|---|---|---|
| Baseline | Dummy (most frequent, stratified) | Chance reference point; every other score is read relative to it. |
| Baseline | Logistic regression (L2) | Simplest linear classifier; interpretable coefficients for sleep/activity effects. |
| Baseline | Decision tree (depth 3) | Tests whether simple thresholds (e.g., "slept < 6 h") carry signal. |
| Experiment | Elastic-net logistic regression | Sleep features are highly correlated; L1/L2 shrinkage stabilizes and selects features while staying interpretable. |
| Experiment | Random forest | Captures non-linear effects and interactions (e.g., poor sleep mattering more after an inactive day) with little tuning. |
| Experiment | LightGBM | Strongest expected learner; handles the 20–55% missing sleep/activity values natively; pairs with SHAP for interpretation. |

**Alignment with project goals.** The linear models answer the *explanatory* side of the question (which sleep and exercise features relate to bad mood, and in which direction). The tree ensembles test whether a more flexible model finds signal the linear models miss. Grouped CV makes every score an estimate of performance on **new people**, which is the stated goal.

**Not pursued.**
- *SVC*: no native handling of missing data, no direct interpretability, and nothing it would add over the models above.
- *MLP*: about 4,000 reports from 49 people don't justify the added complexity.
- *XGBoost/CatBoost*: largely redundant with LightGBM.
- *Regression models (LinearRegression, Ridge, SVR, MLPRegressor)*: the outcome is binary.

## Results (training file, participant-grouped CV)

Means across the 5 folds; 0/1 metrics at a 0.5 threshold. No-information accuracy (always "not bad") = **0.635**.

| Model | Accuracy | Hit rate | False-alarm rate | ROC-AUC | PR-AUC | Balanced acc. |
|---|---|---|---|---|---|---|
| Dummy (always "not bad") | 0.635 | 0.000 | 0.000 | 0.500 | 0.365 | 0.500 |
| Dummy (stratified) | 0.542 | 0.362 | 0.354 | 0.504 ± 0.024 | 0.367 | 0.504 |
| Logistic (L2) | 0.474 | 0.552 | 0.568 | 0.487 ± 0.057 | 0.367 | 0.492 |
| Decision tree (d=3) | 0.459 | 0.445 | 0.526 | 0.473 ± 0.045 | 0.361 | 0.460 |
| Elastic-net LR (tuned) | 0.426 | 0.639 | 0.689 | 0.469 ± 0.071 | 0.351 | 0.475 |
| Random forest (tuned) | 0.469 | 0.563 | 0.582 | 0.493 ± 0.063 | 0.378 | 0.490 |
| **LightGBM (tuned)** | **0.507** | 0.564 | **0.526** | **0.529 ± 0.054** | **0.397** | **0.519** |

**Reading the accuracy column.** No model beats the 63.5% from always predicting "not bad". That isn't because they discriminate worse. With balanced class weights, the models flag about half of all reports as bad, which costs accuracy when only 37% are bad. The hit and false-alarm columns show the real limitation: for every model, the hit rate is close to the false-alarm rate (LightGBM: 56% vs. 53%). The models catch bad moods only about as often as they wrongly flag non-bad ones.

Full fold-level results: `results/model_comparison.csv`, `results/*_fold_scores.csv`.

## Final model choice

**Chosen: LightGBM (tuned via nested grouped CV).**
- Among the real models, it has the highest accuracy (0.507), the largest gap between hit rate and false-alarm rate (0.564 vs. 0.526), and the best ROC-AUC (0.529), PR-AUC (0.397), and balanced accuracy (0.519).
- No model, including LightGBM, beats the 0.635 no-information accuracy at a 0.5 threshold. Accuracy will therefore be reported against that benchmark, and test accuracy will be compared with the test set's own no-information rate (≈0.58).
- On identical folds it beats the L2 logistic baseline in 4 of 5 folds (mean ΔAUC +0.041).
- It handles missing wearable data without imputation and supports SHAP and partial-dependence diagnostics (`results/interpretability/`).

**Important caveat.** The guide says complex models are justified only by *clear* cross-validated gains, and this gain is not clear. LightGBM's fold AUCs (≈0.47–0.60) include chance, and its advantage over the baseline is within fold-to-fold noise. We therefore treat LightGBM as the best available candidate, not as evidence that bad mood is predictable in new people. The L2 logistic model is kept as the interpretable reference. The test set will be scored once with LightGBM, and the result will be reported alongside the logistic baseline.

## What didn't work, and why

1. **Elastic-net regression.** With nested tuning, the search picked an intercept-only model (all 49 coefficients shrunk to zero). That means regularization found no linear sleep/activity signal reliable enough to keep. The approach was discarded as a predictive model; a fixed-penalty fit is reported only descriptively.
2. **Sleep and activity features on their own.** Ablation shows sleep + activity alone score at chance (LightGBM 0.499, logistic 0.454). Time of day + demographics alone score higher (LightGBM 0.556). The small signal that exists is mostly *when* the report happened and *who* the person is, not sleep or exercise.
3. **Models learn individuals, not mood states.** The same baselines score ≈0.58 with a leaky row-level split, compared with ≈0.48 with grouped CV. The logistic model relies mostly on missing-data indicators (device-wear habits), which are person-specific and don't transfer to new people.
4. **Person-centered features (partial success).** Expressing each night relative to the person's own Fitbit average gave LightGBM 0.543 ± 0.028, the most stable result across folds. It isn't adopted yet because the person mean uses the whole study period (mild look-ahead). A prospective (expanding-mean) version is the next iteration.
5. **Error patterns.** Out-of-fold errors cluster by time of day (AUC 0.57 for morning reports, 0.49 for evening) and gender (male 0.54, female 0.50). Median within-person AUC is 0.53, so the model doesn't rank a given person's good and bad days well either.

## Next iteration (back to feature engineering)

Following the guide's "all models perform equally badly → revisit EDA/features" branch:
- Prospective person-centering (expanding mean of each person's prior days).
- Multi-day features (e.g., 3-day sleep debt, rolling activity).
- Restrict to or stratify by morning reports, where last night's sleep is closest in time.
- As a contrast analysis, a within-person evaluation (train on each person's earlier weeks, test on later weeks) to separate "not predictable at all" from "not predictable across people."