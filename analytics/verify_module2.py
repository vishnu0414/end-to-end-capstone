import json
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent


def notebook_checks():
    combined_text = ""
    for nb_name in ["01_eda.ipynb", "02_modeling.ipynb"]:
        path = BASE / nb_name
        nb = json.loads(path.read_text(encoding="utf-8"))
        combined_text += "\n".join(
            "".join(cell.get("source", []))
            if isinstance(cell.get("source"), list)
            else str(cell.get("source", ""))
            for cell in nb["cells"]
        )

    return {
        "one_time_source": "sns.load_dataset('titanic')" in combined_text or 'sns.load_dataset("titanic")' in combined_text,
        "save_csv": "df.to_csv" in combined_text,
        "missing_profile": "missing" in combined_text or "isnull" in combined_text,
        "stratified_split": "stratify=y" in combined_text,
        "grid_search": "GridSearchCV" in combined_text,
        "linear_regression": "LinearRegression" in combined_text,
        "save_pipeline": "joblib.dump" in combined_text,
    }


def output_checks():
    out_dir = BASE / "outputs"
    names = [
        "age_histogram.png",
        "age_boxplot.png",
        "fare_histogram.png",
        "fare_boxplot.png",
        "survival_by_sex.png",
        "survival_by_pclass.png",
        "survival_by_sex_pclass.png",
        "correlation_heatmap.png",
        "classification_decision_tree.png",
        "regression_residual_plot.png",
        "classification_model_metrics.csv",
        "regression_model_metrics.csv",
    ]
    return {name: (out_dir / name).exists() for name in names}


def main():
    checks = notebook_checks()
    outputs = output_checks()

    data = pd.read_csv(BASE / "titanic.csv")
    model = joblib.load(BASE / "models" / "best_pipeline.joblib")
    sample_preds = model.predict(data.drop(columns=["survived"]).head(5)).tolist()

    print("NOTEBOOK_CHECKS", checks)
    print("OUTPUTS", outputs)
    print("CSV_SHAPE", data.shape)
    print("PIPELINE_PREDICTIONS", sample_preds)

    all_ok = all(checks.values()) and all(outputs.values())
    print("ALL_OK", all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
