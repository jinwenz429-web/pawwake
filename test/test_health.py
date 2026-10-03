import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import main


class HealthTests(unittest.TestCase):
    def test_public_get_and_head_need_no_database_or_model(self):
        client = TestClient(main.app)
        with patch.object(main, "GATEWAY_SECRET", "test-configured-secret"), \
             patch.object(main, "get_all_memories_count", side_effect=AssertionError("health touched database")), \
             patch.object(main, "get_system_prompt", side_effect=AssertionError("health loaded prompt")), \
             patch.object(main.httpx, "AsyncClient", side_effect=AssertionError("health called model")):
            for method in ("GET", "HEAD"):
                with self.subTest(method=method):
                    response = client.request(method, "/health")
                    self.assertEqual(response.status_code, 200)
                    if method == "GET":
                        self.assertEqual(response.json(), {"status": "ok"})
                    else:
                        self.assertEqual(response.content, b"")


if __name__ == "__main__":
    unittest.main()
