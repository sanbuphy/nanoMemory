"""
nanoMemory Level 6: Hierarchical Memory
~100 lines. Multi-layer memory: raw messages → episodes → themes.

Memory is organized in a 3-level hierarchy:
  Level 0 (Raw):    Original conversation exchanges
  Level 1 (Episode): Compressed event summaries (LLM-generated)
  Level 2 (Theme):  High-level patterns and recurring topics

Retrieval starts from themes (cheapest) and drills down to raw (most detailed).
This mirrors how humans recall: you first remember the gist, then details.

References:
  - xMemory (2025) "Hierarchical Retrieval via Structured Components"
    Decouples memories into 4 levels: original → episodes → semantics → themes
    Uses sparsity-semantics objective for split/merge operations
    https://arxiv.org/abs/2502.13743
  - Park et al. (2023) "Generative Agents" — memory stream with retrieval
    and reflection for higher-level abstractions
    https://arxiv.org/abs/2304.03442 (UIST 2023)
  - A-MEM (Hou et al., 2024) "Zettelkasten-style note networks"
    Dynamic linking with hierarchical organization
    https://arxiv.org/abs/2502.12110
  - Survey: "Memory in the Age of AI Agents" §2.2 Forms — hierarchical
    token-level memory (pyramid/multi-layer structures)
    https://arxiv.org/abs/2512.13564
  - SCM (Liang et al., 2023) "Self-Controlled Memory Framework"
    Three-tier memory with self-evaluation
    https://arxiv.org/abs/2304.0
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
MEMORY_FILE = "memory_hierarchical.jsonl"
# Hierarchy levels: 0=raw, 1=episode, 2=theme
PROMOTE_EVERY = 3  # raw → episode after N entries
ABSTRACT_EVERY = 5  # episode → theme after N episodes


# ── Memory Operations ──────────────────────────────────────────────────

def load_memories():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_memory(text: str, level: int = 0, **kwargs):
    entry = {
        "text": text,
        "level": level,
        "timestamp": datetime.now().isoformat(),
        **kwargs,
    }
    with open(MEMORY_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    label = {0: "Raw", 1: "Episode", 2: "Theme"}.get(level, "?")
    print(f"[Memory:{label}] {text[:60]}")


def promote_to_episodes():
    """Compress recent raw memories into episode summaries.
    Inspired by xMemory's split-and-merge operations."""
    memories = load_memories()
    raw = [m for m in memories if m["level"] == 0]
    if len(raw) < PROMOTE_EVERY:
        return
    recent_raw = raw[-PROMOTE_EVERY:]
    texts = [m["text"] for m in recent_raw]
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"""
Compress these conversation snippets into 1-2 episode summaries.
Each episode captures WHAT happened in a concise sentence.

Snippets:
{chr(10).join(f'- {t}' for t in texts)}

Return JSON array of summary strings."""}],
        temperature=0,
    )
    try:
        episodes = json.loads(resp.choices[0].message.content.strip())
        for ep in episodes:
            save_memory(ep, level=1, source_count=len(recent_raw))
    except json.JSONDecodeError:
        pass


def abstract_to_themes():
    """Distill episodes into high-level themes/patterns.
    Inspired by Park et al. reflection + xMemory theme layer."""
    memories = load_memories()
    episodes = [m for m in memories if m["level"] == 1]
    if len(episodes) < ABSTRACT_EVERY:
        return
    recent_eps = episodes[-ABSTRACT_EVERY:]
    texts = [m["text"] for m in recent_eps]
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"""
Identify 1-2 high-level themes or patterns from these episode summaries.
These should be abstract insights, not raw facts.

Episodes:
{chr(10).join(f'- {t}' for t in texts)}

Return JSON array of theme strings."""}],
        temperature=0.3,
    )
    try:
        themes = json.loads(resp.choices[0].message.content.strip())
        for theme in themes:
            save_memory(theme, level=2)
    except json.JSONDecodeError:
        pass


def search_memories(query: str, top_k: int = 3) -> list[dict]:
    """Top-down retrieval: themes first, then episodes, then raw details.
    Mirrors human recall: gist → context → specifics."""
    keywords = set(query.lower().split())
    memories = load_memories()

    # Search each level separately, themes first
    results = []
    for level in [2, 1, 0]:
        level_mems = [m for m in memories if m["level"] == level]
        for m in level_mems:
            words = set(m["text"].lower().split())
            if keywords & words:
                results.append(m)
        if len(results) >= top_k:
            break
    return results[:top_k]


# ── Agent with Hierarchical Memory ──────────────────────────────────────

def extract_facts(user_input: str, ai_response: str) -> list[str]:
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
        label = {0: "Detail", 1: "Episode", 2: "Theme"}
        lines = "\n".join(
            f"- [{label.get(m['level'], '?')}] {m['text']}" for m in relevant
        )
        memory_block = f"\n\nRecalled memories:\n{lines}"

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
            save_memory(fact, level=0)  # always start as raw
        promote_to_episodes()   # raw → episode when threshold met
        abstract_to_themes()    # episode → theme when threshold met

    return ai_response


if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello"
    print(run_agent_with_memory(task))
