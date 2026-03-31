"""
nanoMemory Level 3: Hybrid Scoring + Reflection (Park Framework)
~120 lines. Three-factor scoring: relevance x recency x importance + reflection.

On top of Level 2's vector search, adds:
  1. Importance scoring (LLM rates 1-10 on write)
  2. Recency decay (exponential half-life, Ebbinghaus-inspired)
  3. Reflection: periodically distill memories into higher-level insights

References:
  - Park et al. (2023) "Generative Agents: Interactive Simulacra of Human Behavior"
    https://arxiv.org/abs/2304.03442 (UIST 2023, Stanford/Google)
  - MemoryBank (2024) "Forgetting mechanism with Ebbinghaus curve"
    https://arxiv.org/abs/2401.10917 (AAAI 2024)
  - Survey: "Memory in the Age of AI Agents"
    https://arxiv.org/abs/2512.13564
"""
import os
import json
import math
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
MEMORY_FILE = "memory_scored.jsonl"
REFLECTION_FILE = "memory_reflections.jsonl"

# Scoring weights (Park et al. default values)
ALPHA = 0.5   # similarity weight
BETA = 0.3    # recency weight
GAMMA = 0.2   # importance weight
HALF_LIFE_DAYS = 30.0  # Ebbinghaus forgetting curve


# ── Embedding & Scoring ─────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text[:8000])
    return resp.data[0].embedding


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def recency_score(timestamp: str) -> float:
    """Ebbinghaus-inspired decay: 0.5^(days/half_life)"""
    ts = datetime.fromisoformat(timestamp)
    days = (datetime.now() - ts).total_seconds() / 86400
    return 0.5 ** (days / HALF_LIFE_DAYS)


def score_importance(text: str) -> int:
    """Ask LLM to rate 1-10."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"""
Rate the importance of this memory on a scale of 1-10.
10 = critical fact/preference, 1 = trivial detail.
Only output a single number.

Memory: {text}"""}],
        temperature=0,
    )
    try:
        return max(1, min(10, int(resp.choices[0].message.content.strip())))
    except ValueError:
        return 5


# ── Memory Operations ───────────────────────────────────────────────────

def load_memories():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_memory(text: str):
    embedding = get_embedding(text)
    importance = score_importance(text)
    entry = {
        "text": text,
        "embedding": embedding,
        "importance": importance,
        "timestamp": datetime.now().isoformat(),
        "access_count": 0,
    }
    with open(MEMORY_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[Memory] Saved (importance={importance}): {text[:60]}")


def search_memories(query: str, top_k: int = 5) -> list[dict]:
    """Three-factor hybrid scoring."""
    query_emb = get_embedding(query)
    memories = load_memories()
    if not memories:
        return []
    scored = []
    for m in memories:
        if "embedding" not in m:
            continue
        sim = cosine_sim(query_emb, m["embedding"])
        rec = recency_score(m["timestamp"])
        imp = m.get("importance", 5) / 10.0
        score = ALPHA * sim + BETA * rec + GAMMA * imp
        scored.append((m, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    results = []
    for m, s in scored[:top_k]:
        m["access_count"] = m.get("access_count", 0) + 1
        results.append({k: v for k, v in m.items() if k != "embedding"})
    return results


# ── Reflection ──────────────────────────────────────────────────────────

def reflect(memories: list[dict] = None) -> list[str]:
    """Periodically synthesize memories into higher-level insights.
    Inspired by Park et al. (2023) reflection mechanism."""
    if memories is None:
        memories = load_memories()
    if len(memories) < 3:
        return []
    texts = [m["text"] for m in memories[-20:]]
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"""
Review these recent memories and generate 2-3 high-level insights.
These should be abstract patterns or lessons learned, not raw facts.

Memories:
{chr(10).join(f'- {t}' for t in texts)}

Return JSON array of insight strings."""}],
        temperature=0.3,
    )
    try:
        insights = json.loads(resp.choices[0].message.content.strip())
        with open(REFLECTION_FILE, "a") as f:
            for insight in insights:
                entry = {"text": insight, "timestamp": datetime.now().isoformat()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print(f"[Reflect] {insight}")
        return insights
    except json.JSONDecodeError:
        return []


# ── Agent with Scored Memory + Reflection ────────────────────────────────

def extract_facts(user_input, ai_response):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"""
Extract key facts worth remembering. Ignore small talk.
Return JSON array of strings or [].

User: {user_input}
AI: {ai_response}"""}],
        temperature=0,
    )
    try:
        return json.loads(resp.choices[0].message.content.strip())
    except json.JSONDecodeError:
        return []


def run_agent_with_memory(user_message: str, max_iterations: int = 5) -> str:
    relevant = search_memories(user_message)
    memory_block = ""
    if relevant:
        lines = "\n".join(f"- {m['text']}" for m in relevant)
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
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": "N/A"})

    if ai_response:
        for fact in extract_facts(user_message, ai_response):
            save_memory(fact)

    return ai_response


if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello"
    print(run_agent_with_memory(task))
