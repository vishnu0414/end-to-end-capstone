from __future__ import annotations

import json
import os
from typing import TypedDict

import requests
from langgraph.graph import END, START, StateGraph

try:
    from support_assistant.config import GROQ_MODEL, MOCK_LLM
    from support_assistant.models import AskResponse
    from support_assistant.prompts import build_prompt
except ImportError:  # pragma: no cover
    from config import GROQ_MODEL, MOCK_LLM
    from models import AskResponse
    from prompts import build_prompt

POLICY_KEYWORDS = {
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
    "damaged",
    "missing",
}


class GraphState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: list
    answer: str
    sources: list[str]
    confidence: float


def classify_intent(query: str) -> str:
    q = query.lower()
    if str(MOCK_LLM) == "0":
        try:
            response = _call_real_llm(
                f"Classify this query as exactly policy_question or general_question: {query}"
            )
            if response in {"policy_question", "general_question"}:
                return response
        except Exception:
            pass
    if any(keyword in q for keyword in POLICY_KEYWORDS):
        return "policy_question"
    return "general_question"


def _call_real_llm(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("MOCK_LLM=0 requires GROQ_API_KEY")

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _real_answer(query: str, chunks: list[dict]) -> AskResponse:
    context = "\n\n".join(
        f"[{chunk['id']}] {chunk['content']}" for chunk in chunks
    )
    prompt = build_prompt(query, context)
    last_error = "unknown validation error"

    for attempt in range(3):
        try:
            raw = _call_real_llm(prompt)
            payload = json.loads(raw)
            return AskResponse.model_validate(payload)
        except Exception as error:
            last_error = str(error)
            prompt += (
                "\nCORRECTION: Return only valid JSON with answer, sources, "
                "and confidence fields. Do not include markdown fences."
            )

    return AskResponse(
        answer=f"LLM response validation failed after 3 attempts: {last_error}",
        sources=[],
        confidence=0.0,
    )


def retrieve_and_answer(state: GraphState) -> GraphState:
    try:
        from support_assistant.retrieval import retrieve_top_k
    except ImportError:  # pragma: no cover
        from retrieval import retrieve_top_k

    chunks = retrieve_top_k(state["query"], k=3)
    state["retrieved_chunks"] = chunks
    state["sources"] = [chunk["id"] for chunk in chunks]

    if not chunks:
        state["answer"] = "I could not find a matching policy in the local documentation."
        state["confidence"] = 0.0
        return state

    top_chunk = chunks[0]["content"].strip()
    snippet = top_chunk[:250].strip()
    if str(MOCK_LLM) == "0":
        response = _real_answer(state["query"], chunks)
        state["answer"] = response.answer
        state["sources"] = response.sources
        state["confidence"] = response.confidence
    else:
        state["answer"] = f"Based on the retrieved context: {snippet}"
        state["confidence"] = 1.0
    return state


def direct_answer(state: GraphState) -> GraphState:
    state["retrieved_chunks"] = []
    state["sources"] = []
    if str(MOCK_LLM) == "0":
        response = _real_answer(state["query"], [])
        state["answer"] = response.answer
        state["confidence"] = response.confidence
    else:
        state["answer"] = "I can only answer questions about Zepto policies right now."
        state["confidence"] = 1.0
    return state


def route_intent(state: GraphState) -> str:
    return state["intent"]


def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("classify_intent", lambda state: {**state, "intent": classify_intent(state["query"])})
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    workflow.add_node("direct_answer", direct_answer)
    workflow.add_edge(START, "classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "policy_question": "retrieve_and_answer",
            "general_question": "direct_answer",
        },
    )
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)
    return workflow.compile()


GRAPH = build_graph()


def run_graph(query: str) -> AskResponse:
    state = {"query": query, "intent": "", "retrieved_chunks": [], "answer": "", "sources": [], "confidence": 0.0}
    final_state = GRAPH.invoke(state)
    return AskResponse(
        answer=final_state["answer"],
        sources=final_state.get("sources", []),
        confidence=float(final_state.get("confidence", 1.0)),
    )


if __name__ == "__main__":
    print(classify_intent("What is the delivery time?"))
    print(classify_intent("How do I cook rice?"))
    print(run_graph("What is the delivery time?"))
    print(run_graph("What is the capital of France?"))
