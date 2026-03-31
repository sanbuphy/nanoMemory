"""
nanoMemory Level 2: Vector Semantic Retrieval
~100 lines. Replace keyword matching with embedding-based semantic search.

Uses OpenAI embeddings + numpy cosine similarity. No external vector DB needed.
Memories are stored as JSONL with embeddings appended.

References:
  - Mem0 (2025) "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"
    https://arxiv.org/abs/2504.19413
  - LoCoMo Benchmark: "Evaluating Very Long-Term Conversational Memory of LLM Agents"
    https://arxiv.org/abs/2402.17753
"""
import os
import json
import numpy as np
from datetime import datetime
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
MEMORY_FILE = "memory_vector.jsonl"


# ── Embedding ───────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text[:8000])
    return resp.data[0].embedding


def cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── Memory Operations ──────────────────────────────────────────────────

def load_memories() -> list[dict]:
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_memory(text: str, metadata: dict = None):
    embedding = get_embedding(text)
    entry = {
        "text": text,
        "embedding": embedding,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
    }
    with open(MEMORY_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def search_memories(query: str, top_k: int = 5) -> list[dict]:
    """Semantic search: embed query, find nearest neighbors."""
    query_emb = get_embedding(query)
    memories = load_memories()
    if not memories:
        return []
    scored = []
    for m in memories:
        if "embedding" not in m:
            continue
        score = cosine_sim(query_emb, m["embedding"])
        scored.append((m, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{**m, "score": round(s, 4)} for m, s in scored[:top_k]]


def extract_facts(user_input: str, ai_response: str) -> list[str]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"""
Extract key facts worth remembering from this conversation.
Ignore greetings and small talk. Only facts, preferences, and decisions.

User: {user_input}
AI: {ai_response}

Return JSON array of strings. If nothing worth remembering, return []."""}],
        temperature=0,
    )
    try:
        facts = json.loads(resp.choices[0].message.content.strip())
        return facts if isinstance(facts, list) else []
    except json.JSONDecodeError:
        return []


# ── Agent with Vector Memory ───────────────────────────────────────────

def run_agent_with_memory(user_message: str, max_iterations: int = 5) -> str:
    relevant = search_memories(user_message, top_k=5)
    memory_block = ""
    if relevant:
        lines = "\n".join(f"- {m['text']} (score: {m['score']})" for m in relevant)
        memory_block = f"\n\nRelevant memories:\n{lines}"

    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Be concise.{memory_block}"},
        {"role": "user", "content": user_message},
    ]

    ai_response = ""
    for _ in range(max_iterations):
        response = client.chat.completions.create(model=model, messages=messages)
        message = response.choices[0].message
        messages.append(message)
        if message.content:
            ai_response = message.content
        if not message.tool_calls:
            break
        for tc in message.tool_calls:
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": "tool not available"})

    if ai_response:
        facts = extract_facts(user_message, ai_response)
        for fact in facts:
            save_memory(fact)
            print(f"[Memory] Saved: {fact}")

    return ai_response


if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello"
    print(run_agent_with_memory(task))
