# Module 3: Zepto Support Assistant

## Overview

This module implements a chat-based support assistant for Zepto policy questions. It is built as a local, grounded retrieval system that answers policy queries from a small document corpus and returns a deterministic response in the default offline mode.

The app uses:

- OpenAI-compatible response handling in the optional non-mock path
- ChromaDB as the persistent vector store
- SentenceTransformers for local embedding generation
- LangGraph for the request-routing flow
- FastAPI for the browser UI and `/ask` contract

The required baseline is fully offline and uses `MOCK_LLM=1` by default.

## Folder structure

```text
support_assistant/
├── README.md
├── main.py
├── graph.py
├── retrieval.py
├── prompts.py
├── models.py
├── config.py
├── Dockerfile
├── requirements.txt
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── chroma_db/
├── __init__.py
└── test_assistant.py
```

## Quick start

### 1) Create a virtual environment

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

If you are starting from the repository root, you can also install the consolidated dependency list there:

```bash
pip install -r requirements.txt
```

### 3) Start the app

From the repository root:

```bash
python -m uvicorn support_assistant.main:app --host 127.0.0.1 --port 8001
```

Or from the module folder:

```bash
cd support_assistant
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Open:

```text
http://127.0.0.1:8001/
```

The browser page is the visible support chat UI. The API contract remains the internal `POST /ask` endpoint used by the frontend and by grading scripts.

## Document corpus

The assistant loads eight local Zepto policy documents from `support_assistant/docs/`.

- `doc_01.txt` — Delivery Policy
- `doc_02.txt` — Returns & Refunds
- `doc_03.txt` — Membership Tiers
- `doc_04.txt` — Order Tracking
- `doc_05.txt` — Order Cancellation
- `doc_06.txt` — Damaged/Missing Items
- `doc_07.txt` — Gift Cards
- `doc_08.txt` — Customer Support Hours

These files are ingested and indexed into a persistent ChromaDB collection named `zepto_policies`.

## Embedding and retrieval flow

The retrieval layer uses:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

and stores vectors in the local Chroma collection configured in `config.py`:

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "zepto_policies"
```

The retrieval logic reads each document, chunks it, embeds the text, and stores the vectors for cosine-similarity lookup. The app converts incoming policy questions into embeddings and retrieves the most relevant results before generating an answer.

## LangGraph architecture

The graph contains three main nodes:

- `classify_intent`
- `retrieve_and_answer`
- `direct_answer`

Routing is:

```text
classify_intent
    ├── policy question → retrieve_and_answer
    └── general question → direct_answer
```

This ensures grounded policy questions are answered from retrieved sources, while unrelated questions receive the fixed fallback response.

## Mock vs real LLM behavior

`MOCK_LLM` is defined in `config.py` and defaults to `1`:

```python
MOCK_LLM = os.getenv("MOCK_LLM", "1")
```

Behavior:

- `MOCK_LLM` unset → offline mock mode
- `MOCK_LLM=1` → mock mode
- `MOCK_LLM=0` → optional Groq-compatible real LLM path

The grading baseline must remain the default mock mode. The mock path is deterministic, schema-safe, and does not require any API key.

For a local real-LLM run, you can set the environment variables and start the app as follows:

```powershell
$env:MOCK_LLM = "0"
$env:GROQ_MODEL = "openai/gpt-oss-120b"
$env:GROQ_API_KEY = "PASTE_A_NEW_ROTATED_KEY_HERE"
python -m uvicorn support_assistant.main:app --host 0.0.0.0 --port 8001
```

## API contract

The request schema is defined in `models.py`:

```python
class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
```

The FastAPI route is:

```python
@app.post("/ask", response_model=AskResponse)
def ask_support(request: AskRequest) -> AskResponse:
    return run_graph(request.query)
```

## Example requests

### Policy question

```json
{
  "query": "What is the delivery time?"
}
```

Expected shape:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes.",
  "sources": ["doc_01_chunk_01"],
  "confidence": 1.0
}
```

### General question

```json
{
  "query": "What is the capital of France?"
}
```

Expected shape:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Docker

A Dockerfile is available in `support_assistant/Dockerfile`.

Build:

```bash
cd support_assistant
docker build -t zepto-support .
```

Run:

```bash
docker run -p 7860:7860 zepto-support
```

Test:

```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the delivery time?"}'
```

## Resulting behavior

The app behaves as a grounded policy assistant:

- policy questions are answered using retrieved local policy passages
- unrelated questions fall back to a short fixed response
- the visible front-end is a browser chat UI at `/`
- the backend retains a JSON `/ask` API for grading and automation
- the default path remains local and deterministic, with no API-key requirement

## Summary

Module 3 delivers a local, deterministic support assistant that demonstrates end-to-end retrieval and response generation in a grounded RAG workflow while keeping the required grading path fully offline by default.
- Generation: `graph.py` builds the `StateGraph` with `classify_intent`, `retrieve_and_answer`, and `direct_answer`. The conditional edge routes policy questions to retrieval and general questions directly to `direct_answer`. The optional real path uses the role-context-task-format-length prompt in `prompts.py`; the default mock path returns a deterministic top-chunk template or fixed general response.

Data flows as `docs/*.txt -> local embeddings -> ChromaDB -> LangGraph routing -> grounded response -> browser chat`. Retrieval is never replaced by an LLM. With `MOCK_LLM` unset or set to `1`, intent classification uses the required keyword heuristic and both answer nodes use deterministic local responses. Only when `MOCK_LLM=0` is explicitly set do intent classification and final answer generation call the optional Groq-compatible model; invalid structured output is retried twice before a marked error response.

The committed ChromaDB directory is a regenerable local artifact; importing `retrieval.py` also re-ingests any missing policy chunks from `docs/`.
