"""
nanoMemory Level 4: Knowledge Graph Memory
~150 lines. Structured (Subject, Predicate, Object) triples with temporal tracking.

Instead of flat text, memories are stored as graph triples.
Each triple has valid_from / valid_until for temporal reasoning.
Entities are normalized (e.g., "Alice" == "alice@corp.com").

References:
  - HippoRAG 2: "From RAG to Memory: Non-Parametric Continual Learning"
    https://openreview.net/forum?id=LWH8yn4HS2
  - Zep / Graphiti: Temporal Knowledge Graph for agents
    https://github.com/getzep/graphiti
  - KBLaM: "Knowledge Base Augmented Language Model"
    https://arxiv.org/abs/2410.10450 (ICLR 2025, Microsoft)
"""
import os
import json
import sqlite3
import numpy as np
from datetime import datetime
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
DB_PATH = "memory_graph.db"


# ── SQLite Knowledge Graph ───────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS triples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            source TEXT DEFAULT '',
            valid_from TEXT NOT NULL,
            valid_until TEXT,
            embedding TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_subject ON triples(subject);
        CREATE INDEX IF NOT EXISTS idx_predicate ON triples(predicate);
        CREATE INDEX IF NOT EXISTS idx_active ON triples(valid_until);
    """)
    conn.commit()
    return conn


def normalize_entity(entity: str) -> str:
    """Simple normalization: lowercase, strip whitespace."""
    return entity.strip().lower()


# ── Extract Triples from Conversation ────────────────────────────────────

def extract_triples(user_input: str, ai_response: str) -> list[dict]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"""
Extract factual triples from this conversation as (subject, predicate, object).
Include temporal info when possible (e.g., "moved to", "changed from X to Y").

User: {user_input}
AI: {ai_response}

Return JSON array: [{{"subject": "...", "predicate": "...", "object": "..."}}]
Return [] if no facts."""}],
        temperature=0,
    )
    try:
        triples = json.loads(resp.choices[0].message.content.strip())
        return [t for t in triples if all(k in t for k in ("subject", "predicate", "object"))]
    except json.JSONDecodeError:
        return []


def detect_contradiction(conn, sub, pred, obj):
    """Check if a new triple contradicts existing ones."""
    rows = conn.execute(
        "SELECT id, object FROM triples WHERE subject=? AND predicate=? AND valid_until IS NULL",
        (normalize_entity(sub), pred.lower()),
    ).fetchall()
    contradictions = []
    for row_id, old_obj in rows:
        if old_obj.lower() != obj.lower():
            contradictions.append(row_id)
    return contradictions


# ── Memory Operations ───────────────────────────────────────────────────

def add_triple(subject: str, predicate: str, object: str, source: str = "conversation"):
    conn = init_db()
    sub = normalize_entity(subject)
    pred = predicate.lower().strip()
    now = datetime.now().isoformat()

    # Detect contradictions → invalidate old triples
    contradictions = detect_contradiction(conn, sub, pred, object)
    for cid in contradictions:
        conn.execute("UPDATE triples SET valid_until=? WHERE id=?", (now, cid))
        print(f"[Memory] Invalidated old triple #{cid}")

    conn.execute(
        "INSERT INTO triples (subject, predicate, object, source, valid_from) VALUES (?,?,?,?,?)",
        (sub, pred, object.lower().strip(), source, now),
    )
    conn.commit()
    conn.close()
    print(f"[Memory] Triple: ({sub}, {pred}, {object})")


def query_triples(subject: str = None, predicate: str = None) -> list[dict]:
    """Query current valid triples. At least one of subject/predicate required."""
    conn = init_db()
    conditions = ["valid_until IS NULL"]
    params = []
    if subject:
        conditions.append("subject=?")
        params.append(normalize_entity(subject))
    if predicate:
        conditions.append("predicate=?")
        params.append(predicate.lower())
    rows = conn.execute(
        f"SELECT subject, predicate, object, valid_from FROM triples WHERE {' AND '.join(conditions)}",
        params,
    ).fetchall()
    conn.close()
    return [{"subject": r[0], "predicate": r[1], "object": r[2], "valid_from": r[3]} for r in rows]


def query_history(subject: str = None, predicate: str = None) -> list[dict]:
    """Query ALL triples including expired ones (temporal reasoning)."""
    conn = init_db()
    conditions = []
    params = []
    if subject:
        conditions.append("subject=?")
        params.append(normalize_entity(subject))
    if predicate:
        conditions.append("predicate=?")
        params.append(predicate.lower())
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = conn.execute(
        f"SELECT subject, predicate, object, valid_from, valid_until FROM triples {where} ORDER BY valid_from",
        params,
    ).fetchall()
    conn.close()
    return [{"subject": r[0], "predicate": r[1], "object": r[2],
             "valid_from": r[3], "valid_until": r[4]} for r in rows]


# ── Agent with Graph Memory ──────────────────────────────────────────────

def run_agent_with_memory(user_message: str, max_iterations: int = 5) -> str:
    # Extract entities from user message for graph lookup
    entities_resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"""
Extract entity names from this message. Return JSON array of strings or [].
Message: {user_message}"""}],
        temperature=0,
    )
    try:
        entities = json.loads(entities_resp.choices[0].message.content.strip())
    except json.JSONDecodeError:
        entities = []

    # Query graph for each entity
    graph_facts = []
    for entity in entities[:5]:
        triples = query_triples(subject=entity)
        for t in triples:
            graph_facts.append(f"{t['subject']} {t['predicate']} {t['object']}")

    memory_block = ""
    if graph_facts:
        memory_block = "\n\nKnown facts:\n" + "\n".join(f"- {f}" for f in graph_facts[:15])

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
        for t in extract_triples(user_message, ai_response):
            add_triple(t["subject"], t["predicate"], t["object"])

    return ai_response


if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello"
    print(run_agent_with_memory(task))
