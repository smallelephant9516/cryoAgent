import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cryoagent.config.config_loader import CryoSPARCSettings
from cryoagent.core.base_react_agent import BaseReActAgent
from cryoagent.tools.cryosparc_forum_tools import fetch_cryosparc_forum_topic
from cryoagent.tools.cryosparc_guide_index import (
    GUIDE_INDEX_URL,
    GUIDE_SITEMAP_URL,
    bm25_rank,
    load_guide_index,
    parse_llms_index,
)
from cryoagent.tools.cryosparc_guide_tools import consult_cryosparc_guide
from cryoagent.tools.cryosparc_tools import CryoSPARCTools


class _Response:
    def __init__(self, content):
        self.content = content


class _LLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _Response(self.content)


class _Agent(BaseReActAgent):
    def _create_tools(self):
        return []

    def _get_react_system_prompt(self):
        return ""


def _bare_agent(llm=None):
    agent = _Agent.__new__(_Agent)
    agent.llm = llm
    agent._record_tool_execution = Mock()
    agent._recent_history_context = Mock(return_value="No previous actions")
    return agent


class GuideIndexTests(unittest.TestCase):
    def test_parse_complete_llms_catalog_and_canonicalize_urls(self):
        text = """# CryoSPARC
- [Topaz Train](https://guide.cryosparc.com/jobs/topaz-train.md): Train a picker.
- [External](https://example.com/not-guide.md): Ignore this.
- [Lookalike](https://guide.cryosparc.com.evil.test/no.md): Ignore this too.
"""
        entries = parse_llms_index(text)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["title"], "Topaz Train")
        self.assertEqual(entries[0]["url"], "https://guide.cryosparc.com/jobs/topaz-train")
        self.assertEqual(entries[0]["description"], "Train a picker.")

    def test_cache_and_stale_network_fallback(self):
        entries = [{"title": "Topaz", "url": "https://guide.cryosparc.com/topaz", "description": "Deep picking"}]
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "guide.json"
            calls = []

            def fetcher(url):
                calls.append(url)
                return "- [Topaz](https://guide.cryosparc.com/topaz.md): Deep picking"

            first = load_guide_index(fetcher, cache_path=cache)
            self.assertEqual(first, entries)
            second = load_guide_index(lambda _: self.fail("fresh cache should avoid network"), cache_path=cache)
            self.assertEqual(second, entries)

            data = json.loads(cache.read_text())
            data["fetched_at"] = time.time() - 3 * 24 * 60 * 60
            cache.write_text(json.dumps(data))
            stale = load_guide_index(lambda _: None, cache_path=cache)
            self.assertEqual(stale, entries)

    def test_sitemap_fallback(self):
        child = "https://guide.cryosparc.com/sitemap-pages.xml"
        responses = {
            GUIDE_INDEX_URL: None,
            GUIDE_SITEMAP_URL: f"<urlset><loc>{child}</loc></urlset>",
            child: "<urlset><loc>https://guide.cryosparc.com/jobs/reference-based-motion-correction</loc></urlset>",
        }
        entries = load_guide_index(responses.get, refresh=True, cache_path=Path(tempfile.mkdtemp()) / "cache.json")
        self.assertEqual(len(entries), 1)
        self.assertIn("Reference Based Motion Correction", entries[0]["title"])

    def test_bm25_discovers_uncurated_page(self):
        entries = [
            {"title": "Patch CTF Estimation", "url": "https://guide.cryosparc.com/ctf", "description": "Estimate defocus"},
            {"title": "Job: Topaz Train", "url": "https://guide.cryosparc.com/topaz-train", "description": "Train a neural network for deep particle picking"},
        ]
        hits = bm25_rank(entries, "deep learning picking Topaz", limit=2)
        self.assertEqual(hits[0]["url"], "https://guide.cryosparc.com/topaz-train")


class AdvisorTests(unittest.TestCase):
    @patch("cryoagent.tools.cryosparc_forum_tools.search_cryosparc_forum_multi")
    @patch("cryoagent.tools.cryosparc_guide_tools.load_guide_index")
    def test_uncurated_guide_and_forum_results_are_combined(self, load_index, forum_search):
        url = "https://guide.cryosparc.com/new/topaz-worker-installation"
        load_index.return_value = [
            {"title": "Topaz worker installation", "url": url, "description": "Configure exec_path for Topaz Train"}
        ]
        forum_search.return_value = {
            "success": True,
            "results": [{"title": "Worker cannot find Topaz", "url": "https://discuss.cryosparc.com/t/topaz/42", "blurb": "Set exec_path"}],
        }
        html = "<main>On this page Topaz Train requires a worker-accessible exec_path.</main>"
        with patch("cryoagent.tools.cryosparc_guide_tools._fetch", return_value=html):
            result = consult_cryosparc_guide(
                "How can a worker run Topaz Train?",
                queries=["Topaz Train exec_path", "Topaz Train exec_path"],
                max_pages=1,
            )
        self.assertTrue(result["success"])
        self.assertEqual(result["guide_index_size"], 1)
        self.assertEqual(result["queries"], ["Topaz Train exec_path"])
        self.assertEqual(result["pages"][0]["url"], url)
        self.assertEqual(result["forum_results"][0]["url"], "https://discuss.cryosparc.com/t/topaz/42")

    def test_focused_queries_parse_json_and_deduplicate(self):
        llm = _LLM('```json\n["Topaz Train worker exec_path", "Topaz Extract CUDA", "Topaz Train worker exec_path"]\n```')
        agent = _bare_agent(llm)
        question = ("The CTF experiments hurt and must not be searched again. " * 8 +
                    "The unresolved hypothesis is whether Topaz Train can recover cleaner picks on the worker.")
        queries = agent._focused_retrieval_queries(question)
        self.assertEqual(queries, ["Topaz Train worker exec_path", "Topaz Extract CUDA"])
        self.assertEqual(len(llm.calls), 1)

    def test_focused_queries_fall_back_to_original(self):
        agent = _bare_agent(_LLM("not JSON"))
        question = "x" * 250
        self.assertEqual(agent._focused_retrieval_queries(question), [question])

    @patch("cryoagent.tools.cryosparc_guide_tools.consult_cryosparc_guide")
    def test_high_level_wrapper_labels_guide_and_forum_evidence(self, consult):
        consult.return_value = {
            "success": True,
            "message": "Searched 268 guide pages; fetched 1. Forum: 1 discussion(s).",
            "pages": [{"url": "https://guide.cryosparc.com/topaz", "excerpt": "Official Topaz instructions"}],
            "forum_results": [{"title": "Topaz installation", "url": "https://discuss.cryosparc.com/t/topaz/42", "blurb": "Community workaround"}],
            "related_tutorials": [],
        }
        agent = _bare_agent()
        agent._focused_retrieval_queries = Mock(return_value=["Topaz worker installation"])
        output = agent._consult_cryosparc_guide_tool('{"question":"How do I install Topaz?"}')
        self.assertIn("Whole-guide search", output)
        self.assertIn("Official Topaz instructions", output)
        self.assertIn("separate community evidence", output)
        self.assertIn("forum_url=<URL>", output)
        self.assertTrue(consult.call_args.kwargs["include_forum"])

    @patch("cryoagent.tools.cryosparc_guide_tools.consult_cryosparc_guide")
    def test_forum_deep_read_is_condensed(self, consult):
        consult.return_value = {
            "success": True,
            "forum_thread": {
                "url": "https://discuss.cryosparc.com/t/topaz/42",
                "full_text": "Detailed discussion " * 200,
                "excerpt": "Detailed discussion",
            },
        }
        agent = _bare_agent()
        agent._condense_tutorial = Mock(return_value="Set exec_path to the worker-visible topaz binary.")
        output = agent._consult_cryosparc_guide_tool(
            '{"question":"Topaz worker setup","forum_url":"https://discuss.cryosparc.com/t/topaz/42"}'
        )
        self.assertIn("forum digest", output)
        self.assertIn("worker-visible topaz binary", output)
        agent._condense_tutorial.assert_called_once()


class _HTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class ForumTopicTests(unittest.TestCase):
    @patch("cryoagent.tools.cryosparc_forum_tools.urllib.request.urlopen")
    def test_fetch_full_discourse_thread(self, urlopen):
        payload = {
            "title": "Installing Topaz",
            "slug": "installing-topaz",
            "post_stream": {"posts": [
                {"post_number": 1, "username": "alice", "cooked": "<p>Set <code>exec_path</code> &amp; retry.</p>"},
                {"post_number": 2, "username": "bob", "cooked": "<script>ignore()</script><p>The worker can now run Topaz.</p>"},
            ]},
        }
        urlopen.return_value = _HTTPResponse(json.dumps(payload).encode())
        result = fetch_cryosparc_forum_topic("https://discuss.cryosparc.com/t/installing-topaz/42/3")
        self.assertTrue(result["success"])
        thread = result["forum_thread"]
        self.assertEqual(thread["posts_read"], 2)
        self.assertIn("exec_path & retry", thread["full_text"])
        self.assertNotIn("ignore", thread["full_text"])
        self.assertIn("/t/42.json", urlopen.call_args.args[0].full_url)

    def test_reject_non_cryosparc_host(self):
        result = fetch_cryosparc_forum_topic("https://example.com/t/topaz/42")
        self.assertFalse(result["success"])


class TopazConfigurationTests(unittest.TestCase):
    def test_settings_accept_worker_executable(self):
        settings = CryoSPARCSettings(topaz_exec_path="/opt/topaz/bin/topaz")
        self.assertEqual(settings.topaz_exec_path, "/opt/topaz/bin/topaz")

    def test_all_topaz_jobs_receive_executable_and_allow_override(self):
        tools = CryoSPARCTools.__new__(CryoSPARCTools)
        tools.settings = SimpleNamespace(topaz_exec_path="/opt/topaz/bin/topaz")
        tools._create_and_queue_job = Mock(return_value={"success": True})

        tools.topaz_train("P1", "W1", "J1", "J2")
        tools.topaz_extract("P1", "W1", "J3", "J1")
        tools.topaz_denoise("P1", "W1", "J1", params={"exec_path": "/custom/topaz"})

        calls = tools._create_and_queue_job.call_args_list
        self.assertEqual(calls[0].kwargs["params"]["exec_path"], "/opt/topaz/bin/topaz")
        self.assertEqual(calls[1].kwargs["params"]["exec_path"], "/opt/topaz/bin/topaz")
        self.assertEqual(calls[2].kwargs["params"]["exec_path"], "/custom/topaz")


if __name__ == "__main__":
    unittest.main()
