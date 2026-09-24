from __future__ import annotations

from fastapi import FastAPI

try:
    from support_assistant.graph import run_graph
    from support_assistant.models import AskRequest, AskResponse
except ImportError:  # pragma: no cover
    from graph import run_graph
    from models import AskRequest, AskResponse

app = FastAPI(title="Zepto Support Assistant")


@app.post("/ask", response_model=AskResponse)
def ask_support(request: AskRequest) -> AskResponse:
    return run_graph(request.query)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
