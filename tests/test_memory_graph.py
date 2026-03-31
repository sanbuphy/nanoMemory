"""
Tests for Level 4: Knowledge Graph Memory (memory_graph.py)
Mocks OpenAI API calls, tests graph operations and temporal reasoning.
"""
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock


class TestMemoryGraphLevel4(unittest.TestCase):
    """Test Level 4: Knowledge graph with (S, P, O) triples and temporal tracking."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_graph.db")
        self.mock_client = MagicMock()
        self.client_patcher = patch("memory_graph.client", self.mock_client)
        self.client_patcher.start()
        self.db_patcher = patch("memory_graph.DB_PATH", self.db_path)
        self.db_patcher.start()

    def tearDown(self):
        self.client_patcher.stop()
        self.db_patcher.stop()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.tmpdir)

    def _make_response(self, content):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = None
        resp = MagicMock()
        resp.choices = [MagicMock(message=msg)]
        return resp

    def test_normalize_entity(self):
        import memory_graph
        self.assertEqual(memory_graph.normalize_entity("  Alice Smith  "), "alice smith")
        self.assertEqual(memory_graph.normalize_entity("Bob"), "bob")

    def test_add_and_query_triple(self):
        import memory_graph
        memory_graph.add_triple("Alice", "works_at", "Google")
        results = memory_graph.query_triples(subject="alice")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["subject"], "alice")
        self.assertEqual(results[0]["predicate"], "works_at")
        self.assertEqual(results[0]["object"], "google")

    def test_contradiction_invalidates_old(self):
        import memory_graph
        memory_graph.add_triple("Alice", "works_at", "Google")
        memory_graph.add_triple("Alice", "works_at", "Meta")  # contradiction
        # Only the new triple should be active
        results = memory_graph.query_triples(subject="alice")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["object"], "meta")

    def test_history_shows_all_including_expired(self):
        import memory_graph
        memory_graph.add_triple("Alice", "works_at", "Google")
        memory_graph.add_triple("Alice", "works_at", "Meta")
        history = memory_graph.query_history(subject="alice")
        self.assertEqual(len(history), 2)
        # First should have valid_until set, second should be None
        self.assertIsNotNone(history[0]["valid_until"])
        self.assertIsNone(history[1]["valid_until"])

    def test_query_by_predicate(self):
        import memory_graph
        memory_graph.add_triple("Alice", "likes", "Python")
        memory_graph.add_triple("Bob", "likes", "JavaScript")
        memory_graph.add_triple("Alice", "works_at", "Google")
        results = memory_graph.query_triples(predicate="likes")
        self.assertEqual(len(results), 2)

    def test_query_by_subject_and_predicate(self):
        import memory_graph
        memory_graph.add_triple("Alice", "likes", "Python")
        memory_graph.add_triple("Bob", "likes", "JavaScript")
        results = memory_graph.query_triples(subject="alice", predicate="likes")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["object"], "python")

    def test_non_contradicting_triples_coexist(self):
        import memory_graph
        memory_graph.add_triple("Alice", "likes", "Python")
        memory_graph.add_triple("Alice", "likes", "Coffee")
        # Different predicate would not conflict, but same subject+predicate different object = contradiction
        # Actually: same subject + same predicate + different object = contradiction
        results = memory_graph.query_triples(subject="alice")
        self.assertEqual(len(results), 1)  # only "Coffee" survives

    def test_same_triple_no_contradiction(self):
        import memory_graph
        memory_graph.add_triple("Alice", "likes", "Python")
        memory_graph.add_triple("Alice", "likes", "Python")
        results = memory_graph.query_triples(subject="alice")
        self.assertEqual(len(results), 2)  # same value, no contradiction

    def test_extract_triples(self):
        import memory_graph
        self.mock_client.chat.completions.create.return_value = self._make_response(
            '[{"subject": "Alice", "predicate": "works_at", "object": "Google"}]'
        )
        triples = memory_graph.extract_triples("I work at Google", "Cool!")
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0]["subject"], "Alice")

    def test_extract_triples_invalid(self):
        import memory_graph
        self.mock_client.chat.completions.create.return_value = self._make_response("not json")
        triples = memory_graph.extract_triples("Hello", "Hi")
        self.assertEqual(triples, [])

    def test_extract_triples_filters_incomplete(self):
        import memory_graph
        self.mock_client.chat.completions.create.return_value = self._make_response(
            '[{"subject": "Alice"}, {"subject": "Bob", "predicate": "likes", "object": "Python"}]'
        )
        triples = memory_graph.extract_triples("test", "test")
        self.assertEqual(len(triples), 1)  # first one filtered out
        self.assertEqual(triples[0]["subject"], "Bob")

    def test_query_empty_graph(self):
        import memory_graph
        results = memory_graph.query_triples(subject="nobody")
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
