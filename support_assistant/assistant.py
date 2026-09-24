from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class SupportAssistant:
    """Lightweight grounded assistant for repository-specific Q&A."""

    def __init__(self, project_root: str | Path | None = None):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.docs = self._load_project_context()

    def _load_project_context(self) -> Dict[str, str]:
        context = {}
        for file_path in [
            self.project_root / "README.md",
            self.project_root / "data_pipeline" / "README.md",
            self.project_root / "analytics" / "README.md",
            self.project_root / "data_pipeline" / "data" / "books_cleaned.csv",
        ]:
            if file_path.exists():
                try:
                    if file_path.suffix.lower() == ".csv":
                        rows = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                        context[file_path.name] = rows[:5]
                    else:
                        context[file_path.name] = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    context[file_path.name] = ""
        return context

    def _count_books(self) -> str:
        csv_path = self.project_root / "data_pipeline" / "data" / "books_cleaned.csv"
        if not csv_path.exists():
            return "The cleaned dataset file is not present yet."
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            return f"The cleaned dataset contains {len(df)} books."
        except Exception:
            return "I could not read the cleaned dataset to count the books."

    def _price_conversion(self) -> str:
        return "The conversion rate is fixed at 1 GBP = 105.50 INR, as implemented in the data pipeline cleaner."

    def _module_status(self) -> str:
        readme = self.docs.get("README.md", "")
        if "support_assistant" in readme.lower():
            return "This project includes three modules: data pipeline, analytics, and support assistant."
        return "The project has data pipeline and analytics modules; the support assistant is included as a lightweight grounded module."

    def answer(self, question: str) -> str:
        q = question.strip().lower()

        if "book" in q and "count" in q or "how many books" in q:
            return self._count_books()
        if "price" in q and ("inr" in q or "gbp" in q or "convert" in q):
            return self._price_conversion()
        if "module" in q or "project" in q or "three" in q:
            return self._module_status()
        if "model" in q and "recommended" in q:
            return "The analytics module recommends Logistic Regression as the deployment model because it achieves the strongest performance profile while remaining simple and interpretable."
        if "analytics" in q or "titanic" in q:
            return "The analytics module loads the Titanic dataset, profiles missing values, performs EDA, and trains a stratified predictive pipeline saved to analytics/titanic_full_pipeline.joblib."

        # Fallback grounded search in local docs
        matches = []
        for key, content in self.docs.items():
            if not content:
                continue
            if isinstance(content, list):
                content_text = " ".join(content)
            else:
                content_text = str(content)
            if any(term in content_text.lower() for term in q.split() if len(term) > 3):
                matches.append(f"Found relevant context in {key}.")

        if matches:
            return "I found relevant project context in the repository docs and data files: " + "; ".join(matches[:2])

        return (
            "I can answer repository-grounded questions about the data pipeline, analytics module, "
            "and support assistant. For example: book counts, GBP-to-INR conversion, or module status."
        )


def main():
    assistant = SupportAssistant()
    print("Support Assistant ready. Type 'exit' to quit.")
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        response = assistant.answer(question)
        print(response)


if __name__ == "__main__":
    main()
