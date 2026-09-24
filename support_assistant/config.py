import os

MOCK_LLM = os.getenv("MOCK_LLM", "1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "zepto_policies"
