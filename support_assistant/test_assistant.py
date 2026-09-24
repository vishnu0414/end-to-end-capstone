from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from support_assistant.assistant import SupportAssistant


def main():
    assistant = SupportAssistant(project_root=ROOT)
    answer = assistant.answer("How many books are in the cleaned dataset?")
    assert isinstance(answer, str) and len(answer) > 20
    assert "books" in answer.lower()
    print("MODULE3_TEST_OK")


if __name__ == "__main__":
    main()
