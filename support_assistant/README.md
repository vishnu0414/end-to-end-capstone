# Support Assistant

## 1. Project Overview

This module implements a lightweight offline RAG-style support assistant for Zepto policy questions. The project follows a grounded retrieval pipeline:

ingestion
   ↓
embedding
   ↓
retrieval
   ↓
generation

It keeps the baseline fully offline and deterministic by default using `MOCK_LLM=1`.

## 2. Folder Structure

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
└── __init__.py
```

## 3. Installation

```bash
cd support_assistant
pip install -r requirements.txt
```

## 4. Document Ingestion

The source policy corpus is stored in `support_assistant/docs/` as eight policy documents. Each file is read and added to Chroma as a document chunk.

- `retrieval.py` handles the ingestion flow.
- `doc_01.txt` → Delivery Policy
- `doc_02.txt` → Returns & Refunds
- `doc_03.txt` → Membership Tiers
- `doc_04.txt` → Order Tracking
- `doc_05.txt` → Order Cancellation
- `doc_06.txt` → Damaged/Missing Items
- `doc_07.txt` → Gift Cards
- `doc_08.txt` → Customer Support Hours

## 5. Embedding

The embedding layer uses the required model:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

This is configured in `config.py` as `EMBEDDING_MODEL = "all-MiniLM-L6-v2"`.

## 6. ChromaDB

The vector store uses persistent ChromaDB:

```python
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)
```

This is implemented in `retrieval.py`.

## 7. LangGraph Architecture

The graph contains the required nodes and conditional routing:

- `classify_intent` in `graph.py`
- `retrieve_and_answer` in `graph.py`
- `direct_answer` in `graph.py`

Routing logic:

```text
classify_intent
    ├── policy_question → retrieve_and_answer
    └── general_question → direct_answer
```

## 8. MOCK_LLM

The application is designed for an offline mock mode baseline. `MOCK_LLM` is read from the environment in `config.py`:

```python
MOCK_LLM = os.getenv("MOCK_LLM", "1")
```

Behavior:

- `MOCK_LLM` unset → mock mode
- `MOCK_LLM=1` → mock mode
- `MOCK_LLM=0` → optional real LLM path

When `MOCK_LLM=0`, the optional Groq-compatible path reads `GROQ_API_KEY` and `GROQ_MODEL` from the environment. Responses are parsed with `AskResponse` and retried twice with a corrective JSON instruction if validation fails. The default mock path does not require either variable.

The configured real-LLM model is `openai/gpt-oss-120b`. Set the key only in your local environment; never place it in source control:

PowerShell:

```powershell
$env:MOCK_LLM = "0"
$env:GROQ_MODEL = "openai/gpt-oss-120b"
$env:GROQ_API_KEY = "PASTE_A_NEW_ROTATED_KEY_HERE"
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

The required grading path remains the default `MOCK_LLM=1` mode and does not call Groq.

The grading baseline is mock mode.

## 9. Pydantic Schema

`models.py` defines:

```python
class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
```

## 10. FastAPI

The browser-first chat interface is exposed at `/`, and the web API is kept as the internal `POST /ask` contract used by the interface and automated grading. Swagger and ReDoc are disabled so the app opens directly as a chatbot rather than an endpoint catalog. Both are implemented in `main.py`:

```python
@app.post("/ask", response_model=AskResponse)
def ask_support(request: AskRequest) -> AskResponse:
    return run_graph(request.query)
```

Run:

```bash
cd support_assistant
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

From the repository root, use `python -m uvicorn support_assistant.main:app --host 0.0.0.0 --port 8000`.

## 11. Example Requests

### Policy Question

Request:

```json
{
  "query": "What is the delivery time?"
}
```

Response:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes.",
  "sources": ["doc_01_chunk_01"],
  "confidence": 1.0
}
```

### General Question

Request:

```json
{
  "query": "What is the capital of France?"
}
```

Response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## 12. Docker

A local Dockerfile is provided at `support_assistant/Dockerfile`.

Build:

```bash
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

## 13. RAG Architecture

The architecture is:

```text
ingestion -> embedding -> retrieval -> generation
```

- Ingestion: `retrieval.py` reads `docs/*.txt` and adds chunks to ChromaDB.
- Embedding: `SentenceTransformer("all-MiniLM-L6-v2")` from `retrieval.py`.
- Retrieval: `retrieve_top_k()` in `retrieval.py` queries the Chroma collection.
- Generation: `retrieve_and_answer()` and `direct_answer()` in `graph.py` produce the final response in mock mode.

The committed ChromaDB directory is a regenerable local artifact; importing `retrieval.py` also re-ingests any missing policy chunks from `docs/`.
