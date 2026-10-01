import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main


class MemoryRebuildApplyEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.token = "test-token-abcdefghijklmnopqrstuvwxyz-123456"

    def test_apply_fails_closed_without_server_secret(self):
        with patch.dict(os.environ, {"MEMORY_REBUILD_APPLY_TOKEN": ""}), \
             patch.object(main, "apply_memory_rebuild_plan", new_callable=AsyncMock) as apply:
            response = self.client.post(
                "/api/memories/rebuild/plan/1/apply",
                json={"confirmed_plan_id": 1},
            )
        self.assertEqual(response.status_code, 503)
        apply.assert_not_called()

    def test_apply_rejects_missing_or_wrong_secret(self):
        with patch.dict(os.environ, {"MEMORY_REBUILD_APPLY_TOKEN": self.token}), \
             patch.object(main, "apply_memory_rebuild_plan", new_callable=AsyncMock) as apply:
            response = self.client.post(
                "/api/memories/rebuild/plan/1/apply",
                headers={"X-Memory-Rebuild-Apply-Key": "wrong"},
                json={"confirmed_plan_id": 1},
            )
        self.assertEqual(response.status_code, 403)
        apply.assert_not_called()

    def test_apply_needs_matching_plan_id_and_secret(self):
        with patch.dict(os.environ, {"MEMORY_REBUILD_APPLY_TOKEN": self.token}), \
             patch.object(main, "apply_memory_rebuild_plan", new_callable=AsyncMock) as apply:
            apply.return_value = {"status": "applied", "plan_id": 1}
            response = self.client.post(
                "/api/memories/rebuild/plan/1/apply",
                headers={"X-Memory-Rebuild-Apply-Key": self.token},
                json={"confirmed_plan_id": 1},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "applied")
        apply.assert_awaited_once_with(1, 1)


if __name__ == "__main__":
    unittest.main()
