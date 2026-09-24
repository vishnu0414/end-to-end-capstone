from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

try:
    from support_assistant.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL
except ImportError:  # pragma: no cover
    from config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = Path(CHROMA_PATH)
if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = BASE_DIR / CHROMA_DIR
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer(EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def ingest_documents() -> None:
    docs_dir = BASE_DIR / "docs"
    if not docs_dir.exists():
        return

    for doc_path in sorted(docs_dir.glob("*.txt")):
        text = doc_path.read_text(encoding="utf-8")
        doc_id = f"{doc_path.stem}_chunk_01"
        embedding = model.encode(text).tolist()
        collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
        )


def retrieve_top_k(query: str, k: int = 3):
    embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=k)
    items = []
    for doc_id, doc_text in zip(results["ids"][0], results["documents"][0]):
        items.append({"id": doc_id, "content": doc_text})
    return items


ingest_documents()


if __name__ == "__main__":
    for question in [
        "What is the delivery time?",
        "How long do I have to report damaged items?",
        "How much is Zepto Pass?",
    ]:
        print("QUESTION:", question)
        for item in retrieve_top_k(question, k=3):
            print(item["id"], "->", item["content"][:120])
        print("-" * 80)
