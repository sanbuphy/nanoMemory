"""
LoCoMo Benchmark Evaluation for nanoMemory
Evaluates all Levels 0-7 against LoCoMo QA tasks.

Strategy: For each level, we simulate its memory approach:
- Level 0: No context (pure LLM)
- Level 1: Keyword-matched raw text chunks
- Level 2: LLM-based semantic retrieval (simulates embedding without API)
- Level 3: LLM retrieval + importance weighting
- Level 4: LLM extracts SPO triples, then answers from graph
- Level 5: Compressed summaries as context
- Level 6: Hierarchical: themes -> episodes -> raw
- Level 7: Agent decides what to remember via self-selection

Environment:
  export OPENAI_API_KEY='your-key'
  export OPENAI_BASE_URL='https://api.stepfun.com/v1'
  export OPENAI_MODEL='step-3.5-flash'

Usage:
  python eval/run_locomo.py
  python eval/run_locomo.py --results-only

References:
  - LoCoMo: Maharana et al., ACL 2024
    https://arxiv.org/abs/2402.10790
    https://github.com/snap-research/LoCoMo
"""
import os, sys, json, time, random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

LOCOMO_DATA = os.path.join(BASE_DIR, "eval", "locomo10.json")
RESULTS_PATH = os.path.join(BASE_DIR, "docs", "benchmark_results.json")

SAMPLES_PER_CAT = int(os.environ.get("LOCOMO_SAMPLES", 5))
SEED = 42

cat_names = {1: "single_hop", 2: "temporal", 3: "multi_hop",
             4: "open_domain", 5: "adversarial"}


def get_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                  base_url=os.environ.get("OPENAI_BASE_URL"))

def get_model():
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

def call_llm(prompt, temperature=0):
    client = get_client()
    model = get_model()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# ── LoCoMo Data ────────────────────────────────────────────────────────

def load_locomo():
    with open(LOCOMO_DATA) as f:
        return json.load(f)

def flatten_conversation(conv):
    lines = []
    conv_data = conv["conversation"]
    for k in sorted(conv_data.keys()):
        if k.startswith("session_") and not k.endswith("_date_time"):
            session = conv_data[k]
            if isinstance(session, list):
                for turn in session:
                    text = turn.get("text", "").strip()
                    if text:
                        lines.append(f"{turn.get('speaker', '?')}: {text}")
    return lines

def get_answer(q):
    return q.get("answer") or q.get("adversarial_answer", "")

def sample_questions(data, n=5):
    random.seed(SEED)
    by_cat = {}
    for conv in data:
        for q in conv.get("qa", []):
            cat = cat_names.get(q["category"], str(q["category"]))
            if get_answer(q):
                by_cat.setdefault(cat, []).append(q)
    return {cat: random.sample(qs, min(n, len(qs))) for cat, qs in by_cat.items()}


# ── Level Implementations ──────────────────────────────────────────────

STORE = {}

def index_all(conv_lines):
    """Pre-build all memory stores."""
    print("  Building memory stores...")

    # L1: raw text
    STORE["raw"] = conv_lines

    # L5: summaries (compress every 10 lines)
    summaries = []
    for i in range(0, len(conv_lines), 10):
        chunk = "\n".join(conv_lines[i:i+10])
        try:
            s = call_llm(f"Summarize in 1-2 sentences, keeping key facts, names, dates:\n\n{chunk}\n\nSummary:")
            summaries.append(s)
        except: pass
        time.sleep(0.2)
    STORE["summaries"] = summaries
    print(f"    Summaries: {len(summaries)}")

    # L6: episodes + themes
    episodes = []
    for i in range(0, len(conv_lines), 15):
        chunk = "\n".join(conv_lines[i:i+15])
        try:
            ep = call_llm(f"Extract 1-2 episode summaries (what happened):\n\n{chunk}\n\nReturn JSON array of strings.")
            try:
                parsed = json.loads(ep)
                episodes.extend(parsed if isinstance(parsed, list) else [parsed])
            except: episodes.append(ep[:120])
        except: pass
        time.sleep(0.2)
    STORE["episodes"] = episodes

    # L6 themes
    if len(episodes) >= 3:
        try:
            th = call_llm(f"Identify 2-3 high-level themes:\n\n{chr(10).join(f'- {e}' for e in episodes)}\n\nReturn JSON array of strings.")
            try:
                STORE["themes"] = json.loads(th)
            except: STORE["themes"] = []
        except: STORE["themes"] = []
    print(f"    Episodes: {len(episodes)}, Themes: {len(STORE.get('themes', []))}")

    # L4: SPO triples
    triples = []
    for i in range(0, min(len(conv_lines), 100), 5):
        chunk = "\n".join(conv_lines[i:i+5])
        try:
            t = call_llm(f"Extract factual (subject, predicate, object) triples from this text.\n\n{chunk}\n\nReturn JSON array: [{{\"s\":\"...\",\"p\":\"...\",\"o\":\"...\"}}]")
            try:
                parsed = json.loads(t)
                triples.extend(parsed if isinstance(parsed, list) else [])
            except: pass
        except: pass
        time.sleep(0.2)
    STORE["triples"] = triples
    print(f"    Triples: {len(triples)}")

    # L3: importance-scored memories (LLM rates each summary)
    scored = []
    for s in summaries:
        try:
            score_str = call_llm(f"Rate the importance of this memory 1-10 (10=critical fact, 1=trivial):\n\n{s}\n\nOutput just a number.")
            score = max(1, min(10, int(score_str.strip())))
        except: score = 5
        scored.append((s, score))
    STORE["scored"] = scored
    print(f"    Scored memories: {len(scored)}")

    # L7: agent selects what to remember
    selected = []
    try:
        sel = call_llm(f"""You are an agent deciding what to remember from this conversation.
Select the 20 most important facts/preferences/events to store long-term.

Conversation:
{chr(10).join(conv_lines[:80])}

Return JSON array of fact strings (max 20).""")
        try:
            selected = json.loads(sel)
            selected = selected[:20] if isinstance(selected, list) else []
        except: pass
    except: pass
    STORE["selected"] = selected
    print(f"    Agent-selected: {len(selected)}")


def retrieve_keyword(query, top_k=10):
    """L1: keyword overlap."""
    keywords = set(query.lower().split())
    scored = []
    for line in STORE.get("raw", []):
        words = set(line.lower().split())
        overlap = len(keywords & words)
        if overlap > 0:
            scored.append((line, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:top_k]]

def retrieve_semantic(query, top_k=10):
    """L2: LLM-based semantic retrieval (simulates embedding search)."""
    candidates = STORE.get("raw", [])
    # Batch: ask LLM to pick relevant lines
    if len(candidates) > 50:
        # Sample 50 candidates evenly
        step = max(1, len(candidates) // 50)
        candidates = candidates[::step][:50]
    try:
        result = call_llm(f"""Given this question, select the most relevant conversation lines.
Question: {query}

Lines:
{chr(10).join(f'{i}: {c[:100]}' for i, c in enumerate(candidates[:50]))}

Return JSON array of the 5 most relevant line numbers.""")
        try:
            indices = json.loads(result)
            return [candidates[i] for i in indices if 0 <= i < len(candidates)]
        except: pass
    except: pass
    return retrieve_keyword(query, top_k)  # fallback

def retrieve_scored(query, top_k=10):
    """L3: scored retrieval with importance weighting."""
    keywords = set(query.lower().split())
    scored = []
    for text, importance in STORE.get("scored", []):
        words = set(text.lower().split())
        overlap = len(keywords & words)
        score = overlap * 2 + importance  # weight by importance
        if overlap > 0:
            scored.append((text, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:top_k]]

def retrieve_graph(query, top_k=10):
    """L4: triple-based retrieval."""
    triples = STORE.get("triples", [])
    keywords = set(query.lower().split())
    results = []
    for t in triples:
        ttext = f"{t.get('s','')} {t.get('p','')} {t.get('o','')}".lower()
        if keywords & set(ttext.split()):
            results.append(f"{t.get('s','')} {t.get('p','')} {t.get('o','')}")
    return results[:top_k]

def retrieve_summary(query, top_k=5):
    """L5: keyword over summaries."""
    keywords = set(query.lower().split())
    scored = []
    for s in STORE.get("summaries", []):
        overlap = len(keywords & set(s.lower().split()))
        if overlap > 0:
            scored.append((s, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:top_k]]

def retrieve_hierarchical(query, top_k=10):
    """L6: top-down retrieval."""
    keywords = set(query.lower().split())
    results = []
    for key in ["themes", "episodes", "raw"]:
        for item in STORE.get(key, []):
            if keywords & set(item.lower().split()):
                results.append(item)
        if len(results) >= top_k:
            break
    return results[:top_k]

def retrieve_selected(query, top_k=10):
    """L7: search agent-selected memories."""
    keywords = set(query.lower().split())
    scored = []
    for s in STORE.get("selected", []):
        overlap = len(keywords & set(s.lower().split()))
        if overlap > 0:
            scored.append((s, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:top_k]]


def answer_with_context(question, context):
    if not context:
        context = "(no relevant memories found)"
    return call_llm(f"""Based on these memories, answer the question in one short sentence.

Memories:
{context}

Question: {question}
Answer:""")

def answer_no_memory(question):
    return call_llm(f"Answer briefly in one sentence: {question}")

def llm_judge(question, ground_truth, prediction):
    if not ground_truth or not prediction:
        return 0
    result = call_llm(f"""Is the predicted answer correct given the ground truth?
A prediction is correct if it contains the key information, even if worded differently.

Question: {question}
Ground truth: {ground_truth}
Prediction: {prediction}

Output exactly: CORRECT or INCORRECT""")
    text = result.upper()
    return 1 if "CORRECT" in text and "INCORRECT" not in text else 0


# ── Main ────────────────────────────────────────────────────────────────

def run_evaluation():
    print("Loading LoCoMo dataset...")
    data = load_locomo()
    sampled = sample_questions(data, n=SAMPLES_PER_CAT)

    total_qs = sum(len(qs) for qs in sampled.values())
    print(f"Total sampled questions: {total_qs}")
    for cat, qs in sampled.items():
        print(f"  {cat}: {len(qs)}")

    conv = data[0]
    conv_lines = flatten_conversation(conv)
    print(f"\nUsing conversation '{conv['sample_id']}' with {len(conv_lines)} turns")

    # Build all memory stores
    index_all(conv_lines)

    levels = [
        (0, "No Memory",          None),
        (1, "Text + Keyword",     retrieve_keyword),
        (2, "Semantic Retrieval",  retrieve_semantic),
        (3, "Scored Retrieval",    retrieve_scored),
        (4, "Knowledge Graph",     retrieve_graph),
        (5, "Summary Compression", retrieve_summary),
        (6, "Hierarchical",        retrieve_hierarchical),
        (7, "Agent Selected",      retrieve_selected),
    ]

    all_results = {}

    for level, name, retriever in levels:
        print(f"\n{'='*50}")
        print(f"Level {level}: {name}")
        print(f"{'='*50}")

        scores = {}
        for cat, questions in sampled.items():
            correct = 0
            total = len(questions)
            for q in questions:
                try:
                    gt = get_answer(q)
                    if level == 0:
                        pred = answer_no_memory(q["question"])
                    else:
                        ctx_items = retriever(q["question"], top_k=10)
                        ctx = "\n".join(ctx_items[:10]) if ctx_items else ""
                        pred = answer_with_context(q["question"], ctx)
                    judge = llm_judge(q["question"], gt, pred)
                    correct += judge
                except Exception as e:
                    print(f"    Error: {str(e)[:60]}")
                time.sleep(0.3)
            acc = round(correct / max(total, 1) * 100, 1)
            scores[cat] = {"correct": correct, "total": total, "accuracy": acc}
            print(f"  {cat}: {correct}/{total} = {acc}%")

        all_results[str(level)] = {"name": name, "scores": scores}

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_PATH}")
    return all_results


def print_results():
    if not os.path.exists(RESULTS_PATH):
        print("No results. Run: python eval/run_locomo.py")
        return
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    cats = ["single_hop", "temporal", "multi_hop", "open_domain", "adversarial"]
    print(f"\n{'Lvl':<4} {'Method':<22} {'Single':>7} {'Temporal':>9} {'Multi':>6} {'Open':>6} {'Advers':>7} {'Avg':>6}")
    print("-" * 75)
    for lid in sorted(results.keys(), key=int):
        r = results[lid]
        scores = r["scores"]
        if not scores:
            print(f"{lid:<4} {r['name']:<22}  (skipped)")
            continue
        vals = [scores.get(c, {}).get("accuracy", 0) for c in cats]
        avg = round(sum(vals) / max(len(vals), 1), 1)
        print(f"{lid:<4} {r['name']:<22} {vals[0]:>6.1f}% {vals[1]:>8.1f}% {vals[2]:>5.1f}% {vals[3]:>5.1f}% {vals[4]:>6.1f}% {avg:>5.1f}%")


if __name__ == "__main__":
    if "--results-only" in sys.argv:
        print_results()
    else:
        run_evaluation()
        print("\n")
        print_results()
