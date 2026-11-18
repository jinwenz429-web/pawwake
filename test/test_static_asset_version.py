import hashlib
import unittest

import httpx

import main


class StaticAssetVersionTests(unittest.IsolatedAsyncioTestCase):
    def test_version_is_the_file_content_hash(self):
        with open("static/js/dashboard.js", "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()[:10]
        self.assertEqual(main.static_asset("js/dashboard.js"), f"/static/js/dashboard.js?v={digest}")

    def test_missing_file_does_not_break_the_page(self):
        self.assertEqual(main.static_asset("js/nope.js"), "/static/js/nope.js?v=0")

    async def test_login_page_links_the_hashed_stylesheet(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as c:
            html = (await c.get("/dashboard/login")).text
        self.assertIn(main.static_asset("css/login.css"), html)

    def test_templates_have_no_hand_written_versions(self):
        for name in ("templates/dashboard.html", "templates/login.html"):
            with open(name, encoding="utf-8") as fh:
                self.assertNotIn("?v=", fh.read(), name)


if __name__ == "__main__":
    unittest.main()
