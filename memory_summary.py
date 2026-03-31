"""
nanoMemory Level 5: Summary Compression Memory
~80 lines. Instead of storing raw facts, compress conversations into summaries.

Core idea: raw conversation grows unbounded. Periodically compress older
exchanges into concise summaries, discarding detail but preserving intent.
This trades fidelity for capacity — a lossy but practical memory strategy.

References:
  - MemoryBank (Zhong et al., 2024) "MemoryBank: Enhancing Large Language Models
    with Long-Term Memory" — uses Ebbinghaus curve + summarization for forgetting
    https://arxiv.org/abs/2401.10917 (AAAI 2024)
  - "Compress to Impress: Unleashing the Potential of Compressive Memory
    in Real-World Long-Term Conversations" (Wang et al., 2025)
    https://arxiv.org/abs/2402.19285 (COLING 2025)
  - MemoChat (Lu et al., 2023) "MemoChat: Tuning LLMs to Use Memos
    for Consistent Long-Range Open-Domain Conversation"
    https://arxiv.org/abs/2308.07101
  - Survey: "Memory in the Age of AI Agents" §3.2 Token-level Memory —
    summarization as compression strategy
    https://arxiv.org/abs/2512.13564
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
MEMORY_FILE = "memory_summaries.jsonl"
COMPRESS_EVERY = 5  # compress after every N conversations


# ── Memory Operations ──────────────────────────────────────────────────

def load_memories():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_summary(summary: str, source_turns: int = 1):
    entry = {
        "summary": summary,
        "source_turns": source_turns,
        "timestamp": datetime.now().isoformat(),
        "type": "summary",
    }
    with open(MEMORY_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[Memory] Summary saved: {summary[:60]}...")


def compress_memories():
    """Merge all existing summaries into one condensed summary.
    Inspired by MemoryBank's periodic consolidation mechanism."""
    memories = load_memories()
    if len(memories) < COMPRESS_EVERY:
        return
    texts = [m["summary"] for m in memories]
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"""
Compress these memory summaries into ONE concise summary.
Keep all important facts, preferences, and decisions. Remove redundancy.

Summaries:
{chr(10).join(f'- {t}' for t in texts)}

Output a single paragraph summary."""}],
        temperature=0,
    )
    compressed = resp.choices[0].message.content.strip()
    total_turns = sum(m.get("source_turns", 1) for m in memories)
    # Replace all old summaries with one compressed summary
    with open(MEMORY_FILE, "w") as f:
        entry = {
            "summary": compressed,
            "source_turns": total_turns,
            "timestamp": datetime.now().isoformat(),
            "type": "compressed",
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[Memory] Compressed {len(memories)} summaries into 1")


def search_memories(query: str, top_k: int = 5) -> list[dict]:
    """Keyword search over summaries (deliberately simple for this level)."""
    keywords = set(query.lower().split())
    memories = load_memories()
    scored = []
    for m in memories:
        words = set(m["summary"].lower().split())
        overlap = len(keywords & words)
        if overlap > 0:
            scored.append((m, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scored[:top_k]]


# ── Agent with Summary Memory ────────────────────────────────────────────

def extract_summary(user_input: str, ai_response: str) -> str | None:
    """Ask the LLM to summarize the exchange into one sentence."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"""
Summarize this conversation exchange into ONE sentence capturing any
facts, preferences, or decisions worth remembering.
If nothing notable happened, output exactly: NONE

User: {user_input}
AI: {ai_response}"""}],
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    return None if text == "NONE" else text


def run_agent_with_memory(user_message: str, max_iterations: int = 5) -> str:
    relevant = search_memories(user_message)
    memory_block = ""
    if relevant:
        lines = "\n".join(f"- {m['summary']}" for m in relevant)
        memory_block = f"\n\nPast context:\n{lines}"

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
        summary = extract_summary(user_message, ai_response)
        if summary:
            save_summary(summary)
        compress_memories()  # auto-compress when threshold reached

    return ai_response


if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello"
    print(run_agent_with_memory(task))
