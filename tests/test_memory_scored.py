"""
Tests for Level 3: Hybrid Scoring + Reflection (memory_scored.py)
Mocks OpenAI API calls, tests three-factor scoring and reflection.
"""
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np


class TestMemoryScoredLevel3(unittest.TestCase):
    """Test Level 3: Three-factor scoring (relevance × recency × importance) + reflection."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.memory_file = os.path.join(self.tmpdir, "test_memory.jsonl")
        self.reflection_file = os.path.join(self.tmpdir, "test_reflections.jsonl")
        self.mock_client = MagicMock()
        self.client_patcher = patch("memory_scored.client", self.mock_client)
        self.client_patcher.start()
        self.file_patcher = patch("memory_scored.MEMORY_FILE", self.memory_file)
        self.file_patcher.start()
        self.refl_patcher = patch("memory_scored.REFLECTION_FILE", self.reflection_file)
        self.refl_patcher.start()

    def tearDown(self):
        self.client_patcher.stop()
        self.file_patcher.stop()
        self.refl_patcher.stop()
        for f in [self.memory_file, self.reflection_file]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(self.tmpdir)

    def _fake_embedding(self, seed=0):
        rng = np.random.RandomState(seed)
        return rng.randn(64).tolist()

    def _mock_embedding(self, seed=0):
        data = MagicMock()
        data.embedding = self._fake_embedding(seed)
        resp = MagicMock()
        resp.data = [data]
        return resp

    def _make_response(self, content):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = None
        resp = MagicMock()
        resp.choices = [MagicMock(message=msg)]
        return resp

    def test_recency_score_now(self):
        import memory_scored
        score = memory_scored.recency_score(datetime.now().isoformat())
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_recency_score_old(self):
        import memory_scored
        old = (datetime.now() - timedelta(days=30)).isoformat()
        score = memory_scored.recency_score(old)
        self.assertAlmostEqual(score, 0.5, places=2)

    def test_recency_score_very_old(self):
        import memory_scored
        old = (datetime.now() - timedelta(days=90)).isoformat()
        score = memory_scored.recency_score(old)
        # 0.5^(90/30) = 0.5^3 = 0.125
        self.assertAlmostEqual(score, 0.125, places=2)

    def test_score_importance(self):
        import memory_scored
        self.mock_client.chat.completions.create.return_value = self._make_response("8")
        score = memory_scored.score_importance("User is allergic to peanuts")
        self.assertEqual(score, 8)

    def test_score_importance_clamps(self):
        import memory_scored
        self.mock_client.chat.completions.create.return_value = self._make_response("15")
        score = memory_scored.score_importance("something")
        self.assertEqual(score, 10)

    def test_score_importance_invalid(self):
        import memory_scored
        self.mock_client.chat.completions.create.return_value = self._make_response("not a number")
        score = memory_scored.score_importance("something")
        self.assertEqual(score, 5)  # default

    def test_save_memory_stores_importance(self):
        import memory_scored

        def side_effect(**kwargs):
            if "embedding" in str(kwargs):
                return self._mock_embedding(1)
            return self._make_response("7")

        self.mock_client.embeddings.create.return_value = self._mock_embedding(1)
        self.mock_client.chat.completions.create.return_value = self._make_response("7")
        memory_scored.save_memory("Critical project deadline")
        memories = memory_scored.load_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["importance"], 7)
        self.assertIn("embedding", memories[0])

    def test_search_hybrid_scoring(self):
        import memory_scored
        # Memory A: similar, recent, important
        # Memory B: similar, old, less important
        now = datetime.now().isoformat()
        old = (datetime.now() - timedelta(days=60)).isoformat()

        with open(self.memory_file, "w") as f:
            f.write(json.dumps({
                "text": "Recent important fact",
                "embedding": self._fake_embedding(1),
                "importance": 9,
                "timestamp": now,
                "access_count": 0,
            }) + "\n")
            f.write(json.dumps({
                "text": "Old trivial fact",
                "embedding": self._fake_embedding(2),
                "importance": 2,
                "timestamp": old,
                "access_count": 0,
            }) + "\n")

        # Query with same embedding as memory A
        self.mock_client.embeddings.create.return_value = self._mock_embedding(1)
        results = memory_scored.search_memories("test", top_k=2)
        self.assertEqual(len(results), 2)
        # Memory A should score higher (more similar + more recent + more important)
        self.assertEqual(results[0]["text"], "Recent important fact")

    def test_reflect_requires_min_memories(self):
        import memory_scored
        with open(self.memory_file, "w") as f:
            f.write(json.dumps({"text": "only one memory"}) + "\n")
        insights = memory_scored.reflect()
        self.assertEqual(insights, [])

    def test_reflect_generates_insights(self):
        import memory_scored
        with open(self.memory_file, "w") as f:
            for i in range(5):
                f.write(json.dumps({"text": f"fact {i}"}) + "\n")
        self.mock_client.chat.completions.create.return_value = self._make_response(
            '["User prefers concise answers", "Project uses React"]'
        )
        insights = memory_scored.reflect()
        self.assertEqual(len(insights), 2)
        # Check reflection file was written
        with open(self.reflection_file) as f:
            lines = [l for l in f if l.strip()]
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
