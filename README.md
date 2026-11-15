# Loan Default Prediction: from raw data to a business decision

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen) ![AUC](https://img.shields.io/badge/ROC--AUC-0.771-success) ![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)

An end-to-end machine learning project that predicts which loan applicants will have payment difficulties and turns the predictions into an **approve / review / decline** decision, with calibrated probabilities, a cost-based cutoff, SHAP explanations and a fairness audit.

**[Presentation (PPTX)](Loan%20Default%20Prediction.pptx)** · **[Notebook](Walaa_AbuAlghanam_Loan_Default_Model.ipynb)**

---

## The problem

- **Data:** Home Credit application table, 307,511 loans and 122 columns.
- **Target:** `TARGET = 1` means early payment difficulty, which is **8.1%** of loans (imbalanced).
- **Why it matters:** a lender wants to reduce credit losses without rejecting good clients.
- **Metrics:** accuracy is misleading (a do-nothing model scores 91.9%), so the project uses ROC-AUC, PR-AUC, KS, recall and precision.

## Key results

| Metric | Result |
|---|---|
| Baseline (logistic regression) | ROC-AUC 0.749, PR-AUC 0.231 |
| **Best model (tuned LightGBM + financial features)** | **ROC-AUC 0.771** (95% CI 0.764-0.778), PR-AUC 0.262 |
| 5-fold cross-validated AUC | 0.766 +/- 0.003 |
| Train / validation / test AUC | 0.818 / 0.768 / 0.771 |
| Calibration error (ECE) | 0.31 down to 0.006 |
| Business cutoff | Reject above 18.2% predicted risk |
| At that cutoff | Rejects 9.2% of applicants, catches 33.8% of defaulters |
| Net value vs baseline policy | 14.7% vs 11.3% |
| Risk concentration | Riskiest 20% of clients hold 54% of defaulters (top-decile lift 3.5x) |
| Fairness | Gender excluded; approval-rate ratio 0.94 (rule: 0.80+), AUC cost 0.0009 |

> The loss (45%) and margin (10%) behind the business numbers are **assumptions**. Compare policies with each other rather than reading absolute money values.

## Approach

| Step | What was done |
|---|---|
| 1. Audit | Missing values (67 columns, 17 over 60% missing), duplicates, placeholders |
| 2. Clean | Fixed the `365243` days placeholder (55,374 rows) and flagged it, dropped 28 redundant columns, kept missingness as a signal |
| 3. Explore | Default rate by occupation, education, contract type, age and external scores |
| 4. Features | Financial ratios and risk-signal groups, each tested separately |
| 5. Models | Logistic regression vs LightGBM vs XGBoost, plus a tuned LightGBM |
| 6. Fit checks | Train/validation/test gaps, complexity sweep, learning curve |
| 7. Explain | SHAP, global and per client |
| 8. Business value | Probability calibration and a cost-based cutoff |
| 9. Fairness | Gender excluded from the model and audited afterwards |

**Honest evaluation:** stratified split of 64% train / 16% validation / 20% test. Every choice is made on validation; the test set is used only for the final numbers. Preprocessing is fitted on the training set only, so nothing leaks.

## Main findings

**Who defaults more?**
- Low-skill laborers 17.2% vs accountants 4.8% (3.6x)
- Lower secondary education 10.9% vs higher education 5.4%
- Lowest external-score decile 18.4% vs 3.0% for the highest (6.2x)
- Clients under 30 default 11.5%; over 60 only 4.9%
- Credit-to-income ratio on its own barely separates risk (7.1% vs 6.9%)

**What drives the model?** `EXT_SOURCE_3` and `EXT_SOURCE_2` drive risk the most (higher score, lower risk), then `CREDIT_TERM`, `EXT_SOURCE_1` and the goods-to-credit ratio.

**Over- or under-fitting?** Validation AUC peaks at 15 leaves; simpler models under-fit and more complex ones over-fit. The learning curve is still rising, so more data would help.

**Risk bands:** the "very high" band defaults 21.6% (decline or require collateral); the "low" band only 1.9% (fast-track).

## Repository structure

```
loan-default-prediction/
├── README.md
├── Walaa_AbuAlghanam_Loan_Default_Model.ipynb   # full notebook
└── Loan Default Prediction.pptx                 # 16-slide presentation
```

## How to run

```bash
git clone https://github.com/WalaaAbualghanam/loan-default-prediction.git
cd loan-default-prediction
pip install numpy pandas matplotlib seaborn scikit-learn scipy lightgbm xgboost shap joblib jupyter
```

1. Download the **Home Credit Default Risk** application table from Kaggle and save it as `Dataset2.csv` next to the notebook (not included because of its size and license).
2. Open the notebook with Jupyter and run all cells.
3. For a quick dry run, set `SAMPLE_FRAC = 0.1` and lower `N_TUNE_TRIALS` in the settings cell.

## Limitations and next steps

- `TARGET` means early payment difficulty, not necessarily a final write-off.
- Only the application table was used. Bureau, previous-application and instalment tables would likely help.
- Replace the assumed loss and margin with the lender's own numbers and validate out-of-time.
- In production: monitor stability (PSI), re-calibrate regularly, review proxy variables for fairness, and explain declines with SHAP.

## Author

**Walaa Abu Al-Ghanam**: AI graduate, University of Jordan (August 2026). Interests: machine learning, deep learning, AI agents.

[LinkedIn](https://www.linkedin.com/in/walaa-abualghanam/) · [GitHub](https://github.com/WalaaAbualghanam)
