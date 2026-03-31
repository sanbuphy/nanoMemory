"""
LoCoMo Benchmark Evaluation for nanoMemory
Evaluates Levels 1-7 against LoCoMo QA tasks.

Environment:
  export OPENAI_API_KEY='your-key'
  export OPENAI_BASE_URL='https://openrouter.ai/api/v1'   # optional
  export OPENAI_MODEL='stepfun/step-3.5-flash'            # optional
  export OPENAI_EMBED_MODEL='text-embedding-3-small'      # optional

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
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

LOCOMO_DATA = os.path.join(BASE_DIR, "eval", "locomo10.json")
RESULTS_PATH = os.path.join(BASE_DIR, "docs", "benchmark_results.json")

# Sample 5 questions per category for cost efficiency
SAMPLES_PER_CAT = int(os.environ.get("LOCOMO_SAMPLES", 5))
SEED = 42

cat_names = {1: "single_hop", 2: "temporal", 3: "multi_hop",
             4: "open_domain", 5: "adversarial"}


# ── LoCoMo Data Loading ────────────────────────────────────────────────

def load_locomo():
    with open(LOCOMO_DATA) as f:
        return json.load(f)


def flatten_conversation(conv):
    """Flatten all sessions into a single list of 'Speaker: text' strings."""
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


def sample_questions(data, n=5):
    """Sample N questions per category across all conversations."""
    random.seed(SEED)
    by_cat = {}
    for conv in data:
        for q in conv.get("qa", []):
            cat = cat_names.get(q["category"], str(q["category"]))
            by_cat.setdefault(cat, []).append(q)
    return {cat: random.sample(qs, min(n, len(qs))) for cat, qs in by_cat.items()}


# ── Memory Indexing ────────────────────────────────────────────────────

def index_level_1(conv_lines):
    """Level 1: JSONL + keyword matching."""
    from memory_file import MEMORY_FILE, save_memory
    # Clear old data
    open(MEMORY_FILE, "w").close()
    for line in conv_lines:
        save_memory(line)


def index_level_2(conv_lines):
    """Level 2: Vector embedding."""
    from memory_vector import MEMORY_FILE, save_memory
    open(MEMORY_FILE, "w").close()
    for line in conv_lines:
        save_memory(line)


def index_level_3(conv_lines):
    """Level 3: Cognitive scoring."""
    from memory_scored import MEMORY_FILE, save_memory
    open(MEMORY_FILE, "w").close()
    for line in conv_lines:
        save_memory(line)


def index_level_5(conv_lines):
    """Level 5: Summary compression."""
    from memory_summary import MEMORY_FILE, save_summary, compress_memories
    open(MEMORY_FILE, "w").close()
    for line in conv_lines:
        save_summary(line)
    compress_memories()


def index_level_6(conv_lines):
    """Level 6: Hierarchical memory."""
    from memory_hierarchical import MEMORY_FILE, save_memory, promote_to_episodes, abstract_to_themes
    open(MEMORY_FILE, "w").close()
    for line in conv_lines:
        save_memory(line, level=0)
    promote_to_episodes()
    abstract_to_themes()


# ── QA Answering ────────────────────────────────────────────────────────

def answer_with_memory(question, level):
    """Use the memory system to answer a question."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # Retrieve relevant memories
    context = ""
    if level == 1:
        from memory_file import search_memories
        results = search_memories(question, top_k=5)
        context = "\n".join(m["text"] for m in results[:5])
    elif level == 2:
        from memory_vector import search_memories
        results = search_memories(question, top_k=5)
        context = "\n".join(m["text"] for m in results[:5])
    elif level == 3:
        from memory_scored import search_memories
        results = search_memories(question, top_k=5)
        context = "\n".join(m["text"] for m in results[:5])
    elif level == 5:
        from memory_summary import search_memories
        results = search_memories(question, top_k=3)
        context = "\n".join(m["summary"] for m in results[:3])
    elif level == 6:
        from memory_hierarchical import search_memories
        results = search_memories(question, top_k=5)
        context = "\n".join(m["text"] for m in results[:5])

    if not context:
        context = "(no relevant memories found)"

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"""Based on these memories, answer the question briefly.

Memories:
{context}

Question: {question}
Answer:"""}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def answer_no_memory(question):
    """Level 0: No memory baseline."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"Answer briefly: {question}"}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ── LLM Judge ──────────────────────────────────────────────────────────

def llm_judge(question, ground_truth, prediction):
    """LLM-as-judge: CORRECT or INCORRECT."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
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


# ── Main Evaluation ────────────────────────────────────────────────────

def run_evaluation():
    print("Loading LoCoMo dataset...")
    data = load_locomo()
    sampled = sample_questions(data, n=SAMPLES_PER_CAT)

    total_qs = sum(len(qs) for qs in sampled.values())
    print(f"Total sampled questions: {total_qs}")
    for cat, qs in sampled.items():
        print(f"  {cat}: {len(qs)}")

    # Use first conversation for indexing (to keep cost reasonable)
    conv = data[0]
    conv_lines = flatten_conversation(conv)
    print(f"\nUsing conversation '{conv['sample_id']}' with {len(conv_lines)} turns")

    # Levels to evaluate
    levels = {
        0: ("No Memory", "answer_no_memory"),
        1: ("Text + Keyword", "answer_with_memory", index_level_1),
        2: ("Vector Embedding", "answer_with_memory", index_level_2),
        3: ("Cognitive Scoring", "answer_with_memory", index_level_3),
        5: ("Summary Compression", "answer_with_memory", index_level_5),
        6: ("Hierarchical", "answer_with_memory", index_level_6),
    }
    # Level 4 (graph) and Level 7 (agentic) require different approach

    all_results = {}

    for level, spec in levels.items():
        name = spec[0]
        print(f"\n{'='*50}")
        print(f"Level {level}: {name}")
        print(f"{'='*50}")

        # Index conversation into memory
        if len(spec) > 2:
            indexer = spec[2]
            print(f"  Indexing {len(conv_lines)} lines...")
            try:
                indexer(conv_lines)
                print(f"  Indexed.")
            except Exception as e:
                print(f"  Indexing error: {e}")
                continue

        # Evaluate each category
        scores = {}
        answer_fn = spec[1]

        for cat, questions in sampled.items():
            correct = 0
            total = len(questions)
            for q in questions:
                try:
                    if answer_fn == "answer_no_memory":
                        prediction = answer_no_memory(q["question"])
                    else:
                        prediction = answer_with_memory(q["question"], level)

                    judge = llm_judge(q["question"], q["answer"], prediction)
                    correct += judge
                except Exception as e:
                    print(f"    Error on '{q['question'][:40]}...': {e}")
                time.sleep(0.3)

            acc = round(correct / max(total, 1) * 100, 1)
            scores[cat] = {"correct": correct, "total": total, "accuracy": acc}
            print(f"  {cat}: {correct}/{total} = {acc}%")

        all_results[str(level)] = {"name": name, "scores": scores}

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_PATH}")
    return all_results


def print_results():
    """Print saved results as a table."""
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
