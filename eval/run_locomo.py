"""
LoCoMo Benchmark Evaluation for nanoMemory
Evaluates Levels 0-7 (where API allows) against LoCoMo QA tasks.

Environment:
  export OPENAI_API_KEY='your-key'
  export OPENAI_BASE_URL='https://api.stepfun.com/v1'     # optional
  export OPENAI_MODEL='step-3.5-flash'                    # optional
  export OPENAI_EMBED_MODEL='text-embedding-3-small'      # only if embedding API available

Usage:
  python eval/run_locomo.py                  # run eval
  python eval/run_locomo.py --results-only   # just print saved results

References:
  - LoCoMo: "Evaluating Very Long-Term Conversational Memory of LLM Agents"
    Maharana et al., ACL 2024
    https://arxiv.org/abs/2402.10790
    https://github.com/snap-research/LoCoMo
"""
import os
import sys
import json
import time
import random

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
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )


def get_model():
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


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
                        speaker = turn.get("speaker", "Unknown")
                        lines.append(f"{speaker}: {text}")
    return lines


def get_answer(q):
    """Get ground truth answer, handling adversarial questions."""
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


# ── Memory Indexing ────────────────────────────────────────────────────

MEMORY_STORE = {}  # in-memory store for evaluation


def index_keyword(conv_lines):
    """Level 1: keyword search over raw text."""
    MEMORY_STORE["keyword"] = conv_lines


def index_summary(conv_lines):
    """Level 5: compress every 10 lines into a summary via LLM."""
    summaries = []
    client = get_client()
    model = get_model()
    for i in range(0, len(conv_lines), 10):
        chunk = "\n".join(conv_lines[i:i+10])
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"""Summarize this conversation chunk in 1-2 sentences, keeping key facts, names, dates, and events.

{chunk}

Summary:"""}],
            temperature=0,
        )
        summaries.append(resp.choices[0].message.content.strip())
        time.sleep(0.2)
    MEMORY_STORE["summary"] = summaries


def index_hierarchical(conv_lines):
    """Level 6: raw -> episodes -> themes."""
    # Raw level
    MEMORY_STORE["hier_raw"] = conv_lines
    # Episodes: compress every 15 lines
    episodes = []
    client = get_client()
    model = get_model()
    for i in range(0, len(conv_lines), 15):
        chunk = "\n".join(conv_lines[i:i+15])
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"""Extract 1-2 episode summaries from this conversation chunk.

{chunk}

Return JSON array of summary strings."""}],
            temperature=0,
        )
        try:
            eps = json.loads(resp.choices[0].message.content.strip())
            episodes.extend(eps if isinstance(eps, list) else [eps])
        except json.JSONDecodeError:
            episodes.append(resp.choices[0].message.content.strip()[:100])
        time.sleep(0.2)
    MEMORY_STORE["hier_episodes"] = episodes
    # Themes: distill episodes
    if len(episodes) >= 3:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"""Identify 2-3 high-level themes from these episode summaries.

{chr(10).join(f'- {e}' for e in episodes)}

Return JSON array of theme strings."""}],
            temperature=0.3,
        )
        try:
            themes = json.loads(resp.choices[0].message.content.strip())
            MEMORY_STORE["hier_themes"] = themes if isinstance(themes, list) else [themes]
        except json.JSONDecodeError:
            MEMORY_STORE["hier_themes"] = []


# ── Retrieval ──────────────────────────────────────────────────────────

def retrieve_keyword(query, top_k=5):
    """Level 1: keyword overlap."""
    keywords = set(query.lower().split())
    scored = []
    for line in MEMORY_STORE.get("keyword", []):
        words = set(line.lower().split())
        overlap = len(keywords & words)
        if overlap > 0:
            scored.append((line, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:top_k]]


def retrieve_summary(query, top_k=3):
    """Level 5: keyword over summaries."""
    keywords = set(query.lower().split())
    scored = []
    for s in MEMORY_STORE.get("summary", []):
        words = set(s.lower().split())
        overlap = len(keywords & words)
        if overlap > 0:
            scored.append((s, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:top_k]]


def retrieve_hierarchical(query, top_k=5):
    """Level 6: top-down: themes -> episodes -> raw."""
    keywords = set(query.lower().split())
    results = []
    for level_key in ["hier_themes", "hier_episodes", "hier_raw"]:
        for item in MEMORY_STORE.get(level_key, []):
            words = set(item.lower().split())
            if keywords & words:
                results.append(item)
        if len(results) >= top_k:
            break
    return results[:top_k]


# ── Answering ──────────────────────────────────────────────────────────

def answer_with_context(question, context, level_name):
    """Generate answer using retrieved context."""
    client = get_client()
    model = get_model()
    if not context:
        context = "(no relevant memories found)"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"""Based on these memories, answer the question in one short sentence.

Memories:
{context}

Question: {question}
Answer:"""}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def answer_no_memory(question):
    """Level 0: answer without any memory."""
    client = get_client()
    model = get_model()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"Answer briefly in one sentence: {question}"}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ── LLM Judge ──────────────────────────────────────────────────────────

def llm_judge(question, ground_truth, prediction):
    """LLM-as-judge."""
    if not ground_truth or not prediction:
        return 0
    client = get_client()
    model = get_model()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"""Is the predicted answer correct given the ground truth?
A prediction is correct if it contains the key information, even if worded differently.

Question: {question}
Ground truth: {ground_truth}
Prediction: {prediction}

Output exactly one word: CORRECT or INCORRECT"""}],
        temperature=0,
    )
    text = resp.choices[0].message.content.strip().upper()
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

    # Use first conversation
    conv = data[0]
    conv_lines = flatten_conversation(conv)
    print(f"\nUsing conversation '{conv['sample_id']}' with {len(conv_lines)} turns")

    levels = [
        (0, "No Memory", None, None),
        (1, "Text + Keyword", index_keyword, retrieve_keyword),
        (5, "Summary Compression", index_summary, retrieve_summary),
        (6, "Hierarchical", index_hierarchical, retrieve_hierarchical),
    ]

    all_results = {}

    for level, name, indexer, retriever in levels:
        print(f"\n{'='*50}")
        print(f"Level {level}: {name}")
        print(f"{'='*50}")

        # Index
        if indexer:
            print(f"  Indexing {len(conv_lines)} lines...")
            try:
                indexer(conv_lines)
                print(f"  Done. Summaries: {len(MEMORY_STORE.get('summary', []))}, "
                      f"Episodes: {len(MEMORY_STORE.get('hier_episodes', []))}, "
                      f"Themes: {len(MEMORY_STORE.get('hier_themes', []))}")
            except Exception as e:
                print(f"  Indexing error: {e}")
                continue

        # Evaluate
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
                        ctx_items = retriever(q["question"], top_k=5)
                        ctx = "\n".join(ctx_items[:5])
                        pred = answer_with_context(q["question"], ctx, name)
                    judge = llm_judge(q["question"], gt, pred)
                    correct += judge
                except Exception as e:
                    err = str(e)[:60]
                    print(f"    Error: {err}")
                time.sleep(0.3)

            acc = round(correct / max(total, 1) * 100, 1)
            scores[cat] = {"correct": correct, "total": total, "accuracy": acc}
            print(f"  {cat}: {correct}/{total} = {acc}%")

        all_results[str(level)] = {"name": name, "scores": scores}

    # Add placeholder for levels that need embedding API (2, 3) or special handling (4, 7)
    for lid, lname in [(2, "Vector Embedding"), (3, "Cognitive Scoring"),
                       (4, "Knowledge Graph"), (7, "Agentic Lifecycle")]:
        all_results[str(lid)] = {"name": lname, "scores": {},
                                  "note": "Requires embedding API or special setup"}

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_PATH}")
    return all_results


def print_results():
    if not os.path.exists(RESULTS_PATH):
        print("No results found. Run: python eval/run_locomo.py")
        return
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    cats = ["single_hop", "temporal", "multi_hop", "open_domain", "adversarial"]
    print(f"\n{'Level':<5} {'Method':<25} {'Single':>7} {'Temporal':>9} {'Multi':>6} {'Open':>6} {'Advers':>7} {'Avg':>6}")
    print("-" * 78)
    for lid in sorted(results.keys(), key=int):
        r = results[lid]
        scores = r["scores"]
        if not scores:
            print(f"{lid:<5} {r['name']:<25}  (requires embedding API)")
            continue
        vals = [scores.get(c, {}).get("accuracy", 0) for c in cats]
        avg = round(sum(vals) / max(len(vals), 1), 1)
        print(f"{lid:<5} {r['name']:<25} {vals[0]:>6.1f}% {vals[1]:>8.1f}% {vals[2]:>5.1f}% {vals[3]:>5.1f}% {vals[4]:>6.1f}% {avg:>5.1f}%")


if __name__ == "__main__":
    if "--results-only" in sys.argv:
        print_results()
    else:
        run_evaluation()
        print("\n")
        print_results()
