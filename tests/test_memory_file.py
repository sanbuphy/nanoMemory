"""
Tests for Level 1: Persistent Fact Memory (memory_file.py)
Mocks OpenAI API calls, tests local memory operations.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock


class TestMemoryFileLevel1(unittest.TestCase):
    """Test Level 1: JSONL fact memory with keyword matching."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.memory_file = os.path.join(self.tmpdir, "test_memory.jsonl")
        self.mock_client = MagicMock()
        self.patcher = patch("memory_file.client", self.mock_client)
        self.patcher.start()
        self.file_patcher = patch("memory_file.MEMORY_FILE", self.memory_file)
        self.file_patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.file_patcher.stop()
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
        os.rmdir(self.tmpdir)

    def _make_response(self, content):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = None
        resp = MagicMock()
        resp.choices = [MagicMock(message=msg)]
        return resp

    def test_save_and_load(self):
        import memory_file
        memory_file.save_memory("Alice likes Python")
        memory_file.save_memory("Bob uses React 18")
        memories = memory_file.load_memories()
        self.assertEqual(len(memories), 2)
        self.assertEqual(memories[0]["text"], "Alice likes Python")
        self.assertEqual(memories[1]["text"], "Bob uses React 18")

    def test_save_caps_at_200(self):
        import memory_file
        for i in range(210):
            memory_file.save_memory(f"fact {i}")
        memories = memory_file.load_memories()
        self.assertEqual(len(memories), 200)
        self.assertEqual(memories[0]["text"], "fact 10")

    def test_search_keyword_matching(self):
        import memory_file
        memory_file.save_memory("Alice likes Python programming")
        memory_file.save_memory("Bob prefers JavaScript")
        memory_file.save_memory("Charlie writes Python scripts")

        results = memory_file.search_memories("Python")
        self.assertEqual(len(results), 2)
        texts = [r["text"] for r in results]
        self.assertIn("Alice likes Python programming", texts)
        self.assertIn("Charlie writes Python scripts", texts)

    def test_search_returns_empty_when_no_match(self):
        import memory_file
        memory_file.save_memory("Alice likes Python")
        results = memory_file.search_memories("quantum physics")
        self.assertEqual(len(results), 0)

    def test_search_returns_empty_when_no_memories(self):
        import memory_file
        results = memory_file.search_memories("anything")
        self.assertEqual(len(results), 0)

    def test_extract_facts(self):
        import memory_file
        self.mock_client.chat.completions.create.return_value = self._make_response(
            '["Alice prefers dark mode", "Project uses FastAPI"]'
        )
        facts = memory_file.extract_facts("I like dark mode", "Noted!")
        self.assertEqual(facts, ["Alice prefers dark mode", "Project uses FastAPI"])

    def test_extract_facts_empty(self):
        import memory_file
        self.mock_client.chat.completions.create.return_value = self._make_response("[]")
        facts = memory_file.extract_facts("Hello", "Hi there!")
        self.assertEqual(facts, [])

    def test_extract_facts_invalid_json(self):
        import memory_file
        self.mock_client.chat.completions.create.return_value = self._make_response("not json")
        facts = memory_file.extract_facts("Hello", "Hi")
        self.assertEqual(facts, [])

    def test_run_agent_saves_facts(self):
        import memory_file
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self._make_response("Noted! Alice likes Python.")
            else:
                return self._make_response('["Alice likes Python"]')

        self.mock_client.chat.completions.create.side_effect = side_effect
        result = memory_file.run_agent_with_memory("I like Python")
        self.assertIn("Alice likes Python", result)
        memories = memory_file.load_memories()
        self.assertEqual(len(memories), 1)
        self.mock_client.chat.completions.create.side_effect = None


if __name__ == "__main__":
    unittest.main()
