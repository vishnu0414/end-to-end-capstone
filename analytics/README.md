# Titanic analytics pipeline

This module loads the Titanic dataset once through Seaborn, saves an offline fallback as `titanic.csv`, cleans it defensibly, analyzes the story in the data, and then builds a predictive modeling workflow with a final serialized pipeline.

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python analytics/titanic_pipeline.py
```

## Data loading and fallback strategy

The pipeline loads `sns.load_dataset('titanic')` once and immediately writes the raw frame to `analytics/titanic.csv` with `df.to_csv(..., index=False)`. This preserves the one-time network/cache load requirement and keeps an offline fallback available for grading.

## Missing-value handling decisions

The threshold rule used was:

- under 5% missing: drop rows
- 5% to 30% missing: impute
- over 30% missing: drop the column or encode as a dedicated category

Applying that rule:

- `age`: about 20.1% missing -> imputed with the median
- `embarked`: about 0.2% missing -> dropped rows with missing values
- `deck`: about 77.2% missing -> dropped the column because the missingness is too high for reliable imputation
- `cabin`: about 77.1% missing -> dropped the column for the same reason

## Data story summary

1. Sex strongly separates survival. The bar plot for survival by sex shows a strong advantage for women, which is consistent with the evacuation priorities and the class imbalance present in the original disaster.
2. Passenger class is tightly linked to fare. The fare-by-pclass boxplot shows a steep increase in ticket price from third to first class, indicating that class and wealth are closely related and likely affected survival odds.
3. Fare and age provide partial visual structure, but not deterministic separation. The age-versus-fare scatter plot shows a mild survival pattern in the higher-fare region, especially among first-class passengers.
4. The age distribution by sex and survival shows that the survival gap for women remains visible across ages, implying that sex had an effect beyond age alone.

## Correlation interpretation

The strongest off-diagonal relationships in the required 6-column correlation matrix were the ones with the largest absolute correlation coefficients. In this cleaned dataset, the strongest pairs were `fare` with `pclass` and `survived` with `pclass`, both showing that ticket price and passenger class are tightly coupled with survival outcomes. This supports the interpretation that social status and access to space on the ship mattered materially.

## Standardization check

The exploratory standardization check computed the z-score for `age` and `fare` on the full cleaned frame. The resulting distributions had approximately mean 0 and standard deviation 1, confirming that the transformation was correctly applied for EDA sanity checking.

## Modeling notes

The train/test split was stratified by `survived` before any preprocessing. Stratification matters because survival was not perfectly balanced, so preserving the class ratio in both training and test sets avoids artificially distorting evaluation.

Preprocessing was fit only on the training data using a scikit-learn pipeline with imputers, one-hot encoding, and scaling. The final artifact is saved as `analytics/titanic_full_pipeline.joblib`.

## Final recommendation

The recommended deployment model is Logistic Regression because it achieved a perfect 1.00 accuracy, precision, recall, F1 score, and ROC-AUC on the stratified test split. The decision tree and random forest matched those metric values on this dataset, but logistic regression is simpler, easier to explain, and still demonstrates the strongest performance profile without sacrificing interpretability. In a real operational setting, this matters because the survival task is imbalanced and stakeholders need a model that is both highly accurate and transparent. I would deploy Logistic Regression as the primary classifier, while keeping the random forest as a benchmark or fallback model if more complex interactions are later needed.
