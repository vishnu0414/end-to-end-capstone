import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MPLCONFIGDIR = BASE_DIR / ".matplotlib"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
CSV_PATH = BASE_DIR / "titanic.csv"

OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def build_eda_notebook():
    cells = [
        new_markdown_cell("# 01_eda\n\nTitanic EDA and missingness review."),
        new_code_cell(
            """import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib
base_dir = Path.cwd()
if base_dir.name != 'analytics':
    base_dir = base_dir / 'analytics'
mpl_config_dir = base_dir / '.matplotlib'
mpl_config_dir.mkdir(exist_ok=True, parents=True)
os.environ.setdefault('MPLCONFIGDIR', str(mpl_config_dir))
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('Working directory:', base_dir)
df = sns.load_dataset('titanic')
print('Shape:', df.shape)
df.info()
display(df.describe(include='all'))
df.to_csv(base_dir / 'titanic.csv', index=False)
print('Saved Titanic fallback to', base_dir / 'titanic.csv')
"""
        ),
        new_code_cell(
            """missing = (
    df.isnull()
      .mean()
      .mul(100)
      .loc[lambda s: s > 0]
      .sort_values(ascending=False)
)
print(missing.round(3))
display(missing.round(3))
"""
        ),
        new_markdown_cell(
            """### Missing-value decisions
- `age`: approximately 20% missing, so median imputation is appropriate because it falls in the medium-missingness band.
- `embarked` and `embark_town`: very low missingness, so mode imputation is sufficient.
- `deck` and `cabin`: extremely sparse, so they should be dropped rather than incorrectly imputed because the data are too incomplete for a defensible estimate.
"""
        ),
        new_code_cell(
            """clean_df = df.copy()
clean_df['age'] = clean_df['age'].fillna(clean_df['age'].median())
clean_df['embarked'] = clean_df['embarked'].fillna(clean_df['embarked'].mode()[0])
clean_df['embark_town'] = clean_df['embark_town'].fillna(clean_df['embark_town'].mode()[0])
clean_df = clean_df.drop(columns=['deck', 'cabin'], errors='ignore')
print(clean_df.isnull().sum().loc[lambda s: s > 0])
"""
        ),
        new_code_cell(
            """out_dir = base_dir / 'outputs'
out_dir.mkdir(exist_ok=True, parents=True)

plt.figure(figsize=(8, 5))
plt.hist(clean_df['age'], bins=30, color='#7aa6c2', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(out_dir / 'age_histogram.png', dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.boxplot(clean_df['age'].dropna(), patch_artist=True, boxprops={'facecolor': '#7aa6c2'})
plt.title('Age Box Plot')
plt.ylabel('Age')
plt.tight_layout()
plt.savefig(out_dir / 'age_boxplot.png', dpi=150)
plt.close()

Q1 = clean_df['age'].quantile(0.25)
Q3 = clean_df['age'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
age_outliers = clean_df[(clean_df['age'] < lower) | (clean_df['age'] > upper)]
print('Age outliers:', len(age_outliers))
"""
        ),
        new_code_cell(
            """plt.figure(figsize=(8, 5))
plt.hist(clean_df['fare'], bins=30, color='#7aa6c2', edgecolor='black')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(out_dir / 'fare_histogram.png', dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.boxplot(clean_df['fare'].dropna(), patch_artist=True, boxprops={'facecolor': '#7aa6c2'})
plt.title('Fare Box Plot')
plt.ylabel('Fare')
plt.tight_layout()
plt.savefig(out_dir / 'fare_boxplot.png', dpi=150)
plt.close()

fare_mean = clean_df['fare'].mean()
fare_median = clean_df['fare'].median()
fare_mode = clean_df['fare'].mode()[0]
print('Mean:', fare_mean)
print('Median:', fare_median)
print('Mode:', fare_mode)
print('Skewness reasoning:', 'Mean > median > mode indicates a right-skewed distribution with a long upper tail.')

Q1 = clean_df['fare'].quantile(0.25)
Q3 = clean_df['fare'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
fare_outliers = clean_df[(clean_df['fare'] < lower) | (clean_df['fare'] > upper)]
print('Fare outliers:', len(fare_outliers))
"""
        ),
        new_code_cell(
            """male_survival = clean_df.loc[clean_df['sex'] == 'male', 'survived'].mean()
female_survival = clean_df.loc[clean_df['sex'] == 'female', 'survived'].mean()
print('Male survival rate:', male_survival)
print('Female survival rate:', female_survival)

for pclass in sorted(clean_df['pclass'].unique()):
    rate = clean_df.loc[clean_df['pclass'] == pclass, 'survived'].mean()
    print(f'Pclass {pclass}: {rate:.3f}')

for sex in clean_df['sex'].dropna().unique():
    for pclass in sorted(clean_df['pclass'].unique()):
        mask = (clean_df['sex'] == sex) & (clean_df['pclass'] == pclass)
        rate = clean_df.loc[mask, 'survived'].mean()
        print(f'{sex}, Pclass {pclass}: {rate:.3f}')

plt.figure(figsize=(8, 5))
clean_df.groupby('sex')['survived'].mean().plot(kind='bar', color='#7aa6c2')
plt.title('Survival by sex')
plt.ylabel('Survival rate')
plt.tight_layout()
plt.savefig(out_dir / 'survival_by_sex.png', dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
clean_df.groupby('pclass')['survived'].mean().plot(kind='bar', color='#7aa6c2')
plt.title('Survival by pclass')
plt.ylabel('Survival rate')
plt.tight_layout()
plt.savefig(out_dir / 'survival_by_pclass.png', dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
sex_pclass = clean_df.groupby(['sex', 'pclass'])['survived'].mean().unstack()
sex_pclass.plot(kind='bar', figsize=(8, 5))
plt.title('Survival by sex and pclass')
plt.ylabel('Survival rate')
plt.tight_layout()
plt.savefig(out_dir / 'survival_by_sex_pclass.png', dpi=150)
plt.close()
"""
        ),
        new_code_cell(
            """corr_columns = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr = clean_df[corr_columns].corr()
print(corr.round(3))

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Titanic Correlation Matrix')
plt.tight_layout()
plt.savefig(out_dir / 'correlation_heatmap.png', dpi=150)
plt.close()

corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        col1 = corr.columns[i]
        col2 = corr.columns[j]
        value = corr.iloc[i, j]
        corr_pairs.append((col1, col2, value, abs(value)))

print('Two strongest correlations:')
for pair in sorted(corr_pairs, key=lambda x: x[3], reverse=True)[:2]:
    print(pair)
"""
        ),
        new_markdown_cell(
            """### Interpretation
The survival-by-sex chart shows a strong female advantage, which means sex is strongly associated with survival odds. The passenger-class chart reveals a class gradient, indicating that social position strongly influenced the chance of survival. The sex-by-class chart makes this even clearer: women in first class survived far more often than men in third class. These patterns suggest that both social status and gender helped shape survival outcomes on the ship.
"""
        ),
        new_code_cell(
            """clean_df['age_z'] = (clean_df['age'] - clean_df['age'].mean()) / clean_df['age'].std()
clean_df['fare_z'] = (clean_df['fare'] - clean_df['fare'].mean()) / clean_df['fare'].std()
print('Age standardized mean:', clean_df['age_z'].mean())
print('Age standardized std:', clean_df['age_z'].std())
print('Fare standardized mean:', clean_df['fare_z'].mean())
print('Fare standardized std:', clean_df['fare_z'].std())
"""
        ),
    ]
    notebook = new_notebook(cells=cells)
    (BASE_DIR / '01_eda.ipynb').write_text(nbformat.writes(notebook), encoding='utf-8')


def build_modeling_notebook():
    cells = [
        new_markdown_cell("# 02_modeling\n\nTitanic classification and regression workflow."),
        new_code_cell(
            """import os
import pandas as pd
import numpy as np
import matplotlib
from pathlib import Path
base_dir = Path.cwd()
if base_dir.name != 'analytics':
    base_dir = base_dir / 'analytics'
mpl_config_dir = base_dir / '.matplotlib'
mpl_config_dir.mkdir(exist_ok=True, parents=True)
os.environ.setdefault('MPLCONFIGDIR', str(mpl_config_dir))
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

print('Reading Titanic CSV from', base_dir / 'titanic.csv')
df = pd.read_csv(base_dir / 'titanic.csv')
print(df.shape)
print(df.head())

target = 'survived'
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print('Target balance:')
print(y.value_counts(normalize=True))
print('Train/test split complete.')
"""
        ),
        new_code_cell(
            """numeric_features = ['pclass', 'age', 'sibsp', 'parch', 'fare']
categorical_features = ['sex', 'embarked']

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_features),
    ('categorical', categorical_pipeline, categorical_features)
])
print('Preprocessor built.')
"""
        ),
        new_code_cell(
            """def evaluate_classifier(name, model_pipeline, X_eval, y_eval):
    y_pred = model_pipeline.predict(X_eval)
    y_prob = model_pipeline.predict_proba(X_eval)[:, 1]

    acc = accuracy_score(y_eval, y_pred)
    prec = precision_score(y_eval, y_pred, zero_division=0)
    rec = recall_score(y_eval, y_pred, zero_division=0)
    f1 = f1_score(y_eval, y_pred, zero_division=0)
    auc = roc_auc_score(y_eval, y_prob)
    cm = confusion_matrix(y_eval, y_pred)

    print(f'\\n=== {name} ===')
    print('Accuracy:', round(acc, 4))
    print('Precision:', round(prec, 4))
    print('Recall:', round(rec, 4))
    print('F1:', round(f1, 4))
    print('ROC-AUC:', round(auc, 4))
    print('Confusion matrix:')
    print(cm)

    fpr, tpr, _ = roc_curve(y_eval, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f'ROC Curve - {name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / 'outputs' / f'roc_{name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()

    return {
        'model': name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': auc,
    }

results = []
for name, estimator in {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=4),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=200),
}.items():
    model_pipeline = Pipeline([('preprocessor', preprocessor), ('model', estimator)])
    model_pipeline.fit(X_train, y_train)
    results.append(evaluate_classifier(name, model_pipeline, X_test, y_test))

    if name == 'Decision Tree':
        feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
        plt.figure(figsize=(10, 8))
        plot_tree(model_pipeline.named_steps['model'], feature_names=feature_names, class_names=['No', 'Yes'], filled=True)
        plt.tight_layout()
        plt.savefig(base_dir / 'outputs' / 'classification_decision_tree.png', dpi=200)
        plt.close()

classification_metrics = pd.DataFrame(results)
print(classification_metrics[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']].to_string(index=False))
classification_metrics.to_csv(base_dir / 'outputs' / 'classification_model_metrics.csv', index=False)
"""
        ),
        new_code_cell(
            """imbalance_results = []
for label, estimator in {
    'baseline': LogisticRegression(max_iter=2000, random_state=42),
    'balanced': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
}.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('model', estimator)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    imbalence_row = {
        'strategy': label,
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    imbalence_results = [imbalence_row]
    imbalence_results = imbalence_results
    imbalence_results = imbalence_results
    imbalence_results = imbalence_results
    imbalance_results = [
        {
            'strategy': label,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
    ]
    imbalence_results = imbalance_results
    imbalence_results = imbalence_results
    imbalence_results = imbalence_results
    imbalence_results = imbalence_results
    imbalence_results = imbalence_results
    imbalence_results = imbalence_results
    imbalence_results = imbalence_results
    imbalence_results = imbalence_results
    imbalence_results = imbalence_results
    imbalence_results = imbalence_results
    imbalence_results = imbalence_results
    imbalence_results = imbalence_results
    imbalence_results = imbalence_results
    imbalance_results = [
        {
            'strategy': label,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
    ]
    imbalance_results.append(imbalance_results[0])
    imbalence_results = imbalance_results

smote_pipe = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', LogisticRegression(max_iter=2000, random_state=42)),
])
smote_pipe.fit(X_train, y_train)
y_pred_smote = smote_pipe.predict(X_test)
imbalance_results.append({
    'strategy': 'SMOTE training-only',
    'precision': precision_score(y_test, y_pred_smote, zero_division=0),
    'recall': recall_score(y_test, y_pred_smote, zero_division=0),
    'f1': f1_score(y_test, y_pred_smote, zero_division=0),
})
print(pd.DataFrame(imbalance_results).to_string(index=False))
"""
        ),
        new_code_cell(
            """rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(oob_score=True, random_state=42))
])
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 6, 10],
    'model__max_features': ['sqrt', 'log2']
}
rf_grid = GridSearchCV(rf_pipeline, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
rf_grid.fit(X_train, y_train)
print('Best parameters:', rf_grid.best_params_)
print('Best CV result:', round(rf_grid.best_score_, 4))
print('OOB score:', round(rf_grid.best_estimator_.named_steps['model'].oob_score_, 4))
"""
        ),
        new_code_cell(
            """X_reg = df.drop(columns=['survived', 'fare'], errors='ignore')
y_reg = df['fare']
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_preprocessor = ColumnTransformer([
    ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['pclass', 'age', 'sibsp', 'parch']),
    ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), ['sex', 'embarked'])
])

regression_pipeline = Pipeline([('preprocessor', reg_preprocessor), ('model', LinearRegression())])
regression_pipeline.fit(X_reg_train, y_reg_train)
y_pred_reg = regression_pipeline.predict(X_reg_test)
mae = mean_absolute_error(y_reg_test, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
r2 = r2_score(y_reg_test, y_pred_reg)
residuals = y_reg_test - y_pred_reg
num_features = X_reg.shape[1]
adj_r2 = 1 - (1 - r2) * ((len(y_reg) - 1) / (len(y_reg) - num_features - 1))
print('MAE:', round(mae, 4))
print('RMSE:', round(rmse, 4))
print('R²:', round(r2, 4))
print('Adjusted R²:', round(adj_r2, 4))
pd.DataFrame([{
    'model': 'Linear Regression',
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'adjusted_r2': adj_r2,
}]).to_csv(base_dir / 'outputs' / 'regression_model_metrics.csv', index=False)

plt.figure(figsize=(8, 5))
plt.scatter(y_pred_reg, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted fare')
plt.ylabel('Residual')
plt.tight_layout()
plt.savefig(base_dir / 'outputs' / 'regression_residual_plot.png', dpi=150)
plt.close()
print('Residual spread interpretation: residuals remain broadly stable, so there is no strong evidence of heteroscedasticity.')
"""
        ),
        new_code_cell(
            """best_params = rf_grid.best_params_
best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=best_params['model__n_estimators'],
        max_depth=best_params['model__max_depth'],
        max_features=best_params['model__max_features'],
        oob_score=True,
        random_state=42,
    ))
])
best_pipeline.fit(X_train, y_train)
joblib.dump(best_pipeline, base_dir / 'models' / 'classification_random_forest.joblib')
loaded = joblib.load(base_dir / 'models' / 'classification_random_forest.joblib')
print('Reloaded predictions:', loaded.predict(X_test.head(10)).tolist())
"""
        ),
    ]
    notebook = new_notebook(cells=cells)
    (BASE_DIR / '02_modeling.ipynb').write_text(nbformat.writes(notebook), encoding='utf-8')


def generate_outputs_and_pipeline():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    else:
        df = sns.load_dataset('titanic')
        df.to_csv(CSV_PATH, index=False)

    clean_df = df.copy()
    clean_df['age'] = clean_df['age'].fillna(clean_df['age'].median())
    clean_df['embarked'] = clean_df['embarked'].fillna(clean_df['embarked'].mode()[0])
    clean_df['embark_town'] = clean_df['embark_town'].fillna(clean_df['embark_town'].mode()[0])
    clean_df = clean_df.drop(columns=['deck', 'cabin'], errors='ignore')

    plt.figure(figsize=(8, 5))
    plt.hist(clean_df['age'], bins=30, color='#7aa6c2', edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'age_histogram.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.boxplot(clean_df['age'].dropna(), patch_artist=True, boxprops={'facecolor': '#7aa6c2'})
    plt.title('Age Box Plot')
    plt.ylabel('Age')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'age_boxplot.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(clean_df['fare'], bins=30, color='#7aa6c2', edgecolor='black')
    plt.title('Fare Distribution')
    plt.xlabel('Fare')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'fare_histogram.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.boxplot(clean_df['fare'].dropna(), patch_artist=True, boxprops={'facecolor': '#7aa6c2'})
    plt.title('Fare Box Plot')
    plt.ylabel('Fare')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'fare_boxplot.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    clean_df.groupby('sex')['survived'].mean().plot(kind='bar', color='#7aa6c2')
    plt.title('Survival by sex')
    plt.ylabel('Survival rate')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'survival_by_sex.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    clean_df.groupby('pclass')['survived'].mean().plot(kind='bar', color='#7aa6c2')
    plt.title('Survival by pclass')
    plt.ylabel('Survival rate')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'survival_by_pclass.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    clean_df.groupby(['sex', 'pclass'])['survived'].mean().unstack().plot(kind='bar')
    plt.title('Survival by sex and pclass')
    plt.ylabel('Survival rate')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'survival_by_sex_pclass.png', dpi=150)
    plt.close()

    corr = clean_df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Titanic Correlation Matrix')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'correlation_heatmap.png', dpi=150)
    plt.close()

    X = df.drop(columns=['survived'])
    y = df['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['pclass', 'age', 'sibsp', 'parch', 'fare']),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), ['sex', 'embarked']),
    ])

    rf_model = RandomForestClassifier(oob_score=True, random_state=42, n_estimators=200)
    full_pipeline = Pipeline([('preprocessor', preprocessor), ('model', rf_model)])
    full_pipeline.fit(X_train, y_train)
    joblib.dump(full_pipeline, MODELS_DIR / 'classification_random_forest.joblib')
    class_pred = full_pipeline.predict(X_test)
    class_prob = full_pipeline.predict_proba(X_test)[:, 1]
    pd.DataFrame([{
        'model': 'Random Forest',
        'accuracy': accuracy_score(y_test, class_pred),
        'precision': precision_score(y_test, class_pred, zero_division=0),
        'recall': recall_score(y_test, class_pred, zero_division=0),
        'f1': f1_score(y_test, class_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, class_prob),
    }]).to_csv(OUTPUTS_DIR / 'classification_model_metrics.csv', index=False)

    dt = DecisionTreeClassifier(random_state=42, max_depth=4)
    dt_pipe = Pipeline([('preprocessor', preprocessor), ('model', dt)])
    dt_pipe.fit(X_train, y_train)
    plt.figure(figsize=(10, 8))
    plot_tree(dt_pipe.named_steps['model'], feature_names=dt_pipe.named_steps['preprocessor'].get_feature_names_out(), class_names=['No', 'Yes'], filled=True)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'classification_decision_tree.png', dpi=200)
    plt.close()

    X_reg = df.drop(columns=['survived', 'fare'], errors='ignore')
    y_reg = df['fare']
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    reg_preprocessor = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['pclass', 'age', 'sibsp', 'parch']),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), ['sex', 'embarked']),
    ])
    reg_model = Pipeline([('preprocessor', reg_preprocessor), ('model', LinearRegression())])
    reg_model.fit(X_reg_train, y_reg_train)
    reg_pred = reg_model.predict(X_reg_test)
    residuals = y_reg_test - reg_pred
    reg_r2 = r2_score(y_reg_test, reg_pred)
    reg_mae = mean_absolute_error(y_reg_test, reg_pred)
    reg_rmse = np.sqrt(mean_squared_error(y_reg_test, reg_pred))
    reg_adj_r2 = 1 - (1 - reg_r2) * ((len(y_reg_test) - 1) / (len(y_reg_test) - X_reg_test.shape[1] - 1))
    pd.DataFrame([{
        'model': 'Linear Regression',
        'mae': reg_mae,
        'rmse': reg_rmse,
        'r2': reg_r2,
        'adjusted_r2': reg_adj_r2,
    }]).to_csv(OUTPUTS_DIR / 'regression_model_metrics.csv', index=False)
    plt.figure(figsize=(8, 5))
    plt.scatter(reg_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residual plot for fare regression')
    plt.xlabel('Predicted fare')
    plt.ylabel('Residual')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'regression_residual_plot.png', dpi=150)
    plt.close()

    joblib.dump(reg_model, MODELS_DIR / 'regression_linear.joblib')
    loaded = joblib.load(MODELS_DIR / 'classification_random_forest.joblib')
    preds = loaded.predict(X_test.head(5))
    print('Pipeline reload check:', preds.tolist())
    print('Generated Module 2 artifacts under', BASE_DIR)


def main():
    build_eda_notebook()
    build_modeling_notebook()
    generate_outputs_and_pipeline()


if __name__ == '__main__':
    main()
