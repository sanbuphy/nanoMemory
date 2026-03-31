"""
nanoMemory Level 1: Persistent Fact Memory
~80 lines. File-based memory with LLM extraction and keyword matching.

The simplest cross-session memory. After each conversation, an LLM extracts
key facts into a JSONL file. Next conversation, relevant facts are injected
into the system prompt.

References:
  - Liang et al. (2023) SCM: Self-Controlled Memory Framework
    https://arxiv.org/abs/2304.0
  - Tekparmak & Kaya (2025) Markdown-based memory for agents
    https://arxiv.org/abs/2602.11243
"""
import os
import json
from datetime import datetime
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MEMORY_FILE = "memory_facts.jsonl"


# ── Memory Operations ──────────────────────────────────────────────────

def load_memories():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_memory(fact: str, source: str = "conversation"):
    memories = load_memories()
    entry = {
        "text": fact,
        "source": source,
        "timestamp": datetime.now().isoformat(),
    }
    memories.append(entry)
    with open(MEMORY_FILE, "w") as f:
        for m in memories[-200:]:  # keep last 200
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def extract_facts(user_input: str, ai_response: str) -> list[str]:
    """Ask the LLM: is there anything worth remembering?"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"""
Extract key facts worth remembering long-term from this conversation.
Ignore greetings, small talk, and opinions. Only extract facts, preferences, and decisions.

User: {user_input}
AI: {ai_response}

Return a JSON array of strings. Example: ["Alice likes Python", "Project uses React 18"]
If nothing is worth remembering, return []."""}],
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    try:
        facts = json.loads(text)
        return facts if isinstance(facts, list) else []
    except json.JSONDecodeError:
        return []


def search_memories(query: str, top_k: int = 10) -> list[dict]:
    """Keyword matching (Level 1 is deliberately simple)."""
    keywords = set(query.lower().split())
    memories = load_memories()
    scored = []
    for m in memories:
        words = set(m["text"].lower().split())
        overlap = len(keywords & words)
        if overlap > 0:
            scored.append((m, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scored[:top_k]]


# ── Agent with Memory ──────────────────────────────────────────────────

def run_agent_with_memory(user_message: str, max_iterations: int = 5) -> str:
    relevant = search_memories(user_message)
    memory_block = ""
    if relevant:
        lines = "\n".join(f"- {m['text']}" for m in relevant[:10])
        memory_block = f"\n\nWhat you already know about the user:\n{lines}"

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
            # delegate tool calls to agent.py style execution
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": "tool not available in memory demo"})

    # Extract and save facts after conversation
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
