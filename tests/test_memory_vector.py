"""
Tests for Level 2: Vector Semantic Retrieval (memory_vector.py)
Mocks OpenAI API calls, tests embedding-based retrieval.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np


class TestMemoryVectorLevel2(unittest.TestCase):
    """Test Level 2: Embedding-based semantic search with JSONL storage."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.memory_file = os.path.join(self.tmpdir, "test_memory.jsonl")
        self.mock_client = MagicMock()
        self.client_patcher = patch("memory_vector.client", self.mock_client)
        self.client_patcher.start()
        self.file_patcher = patch("memory_vector.MEMORY_FILE", self.memory_file)
        self.file_patcher.start()

    def tearDown(self):
        self.client_patcher.stop()
        self.file_patcher.stop()
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
        os.rmdir(self.tmpdir)

    def _fake_embedding(self, seed=0):
        rng = np.random.RandomState(seed)
        vec = rng.randn(64).tolist()
        return vec

    def _mock_embedding_response(self, seed=0):
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

    def test_save_memory_stores_embedding(self):
        import memory_vector
        self.mock_client.embeddings.create.return_value = self._mock_embedding_response(1)
        memory_vector.save_memory("Alice likes Python", {"source": "test"})
        memories = memory_vector.load_memories()
        self.assertEqual(len(memories), 1)
        self.assertIn("embedding", memories[0])
        self.assertEqual(memories[0]["text"], "Alice likes Python")
        self.assertEqual(memories[0]["metadata"]["source"], "test")

    def test_search_returns_scored_results(self):
        import memory_vector
        # Save memories with different embeddings
        self.mock_client.embeddings.create.side_effect = [
            self._mock_embedding_response(1),  # save "Python programming"
            self._mock_embedding_response(2),  # save "JavaScript frameworks"
            self._mock_embedding_response(1),  # query "Python" → same seed as #1 → high similarity
        ]
        memory_vector.save_memory("Python programming language")
        memory_vector.save_memory("JavaScript frameworks")

        results = memory_vector.search_memories("Python", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "Python programming language")
        self.assertTrue(results[0]["score"] > results[1]["score"])

    def test_cosine_sim_identical_vectors(self):
        import memory_vector
        v = self._fake_embedding(42)
        score = memory_vector.cosine_sim(v, v)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_cosine_sim_orthogonal_vectors(self):
        import memory_vector
        a = [1, 0, 0, 0]
        b = [0, 1, 0, 0]
        score = memory_vector.cosine_sim(a, b)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_search_empty_memories(self):
        import memory_vector
        results = memory_vector.search_memories("anything")
        self.assertEqual(results, [])

    def test_extract_facts(self):
        import memory_vector
        self.mock_client.chat.completions.create.return_value = self._make_response(
            '["User prefers dark mode"]'
        )
        facts = memory_vector.extract_facts("I like dark mode", "Noted!")
        self.assertEqual(facts, ["User prefers dark mode"])

    def test_search_skips_entries_without_embedding(self):
        import memory_vector
        # Manually write a memory without embedding
        with open(self.memory_file, "w") as f:
            f.write(json.dumps({"text": "old entry", "timestamp": "2025-01-01"}) + "\n")
        self.mock_client.embeddings.create.return_value = self._mock_embedding_response(1)
        results = memory_vector.search_memories("test")
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
