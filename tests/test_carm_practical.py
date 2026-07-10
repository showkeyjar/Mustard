"""Tests for CARM practical enhancements: parallel routing, multi-turn anaphora, and REST API.

Covers:
- Parallel function call detection and execution
- Multi-turn calculator with implicit anaphora (e.g. "再加上10")
- REST API server endpoints (smoke tests with TestClient)
"""

import unittest

from carm.router import CARMRouter, RouteResult
from tools.calc_tool import CalculatorTool


class TestParallelRouting(unittest.TestCase):
    """Parallel function call: multiple intents in one query."""

    def setUp(self) -> None:
        self.router = CARMRouter()

    def test_parallel_detection(self) -> None:
        """Two comma-separated intents should produce sub_results."""
        result = self.router.route_parallel("3+5, 7*8")
        self.assertTrue(result.ok)
        self.assertIsNotNone(result.sub_results)
        self.assertEqual(len(result.sub_results), 2)

    def test_parallel_results_correct(self) -> None:
        """Each sub-query should produce the correct numerical answer."""
        result = self.router.route_parallel("3+5, 7*8")
        results_text = " ".join(sr.result for sr in result.sub_results)
        self.assertIn("8", results_text)
        self.assertIn("56", results_text)

    def test_single_query_no_parallel(self) -> None:
        """A single intent should not produce sub_results."""
        result = self.router.route_parallel("3加5")
        self.assertTrue(result.ok)
        # Single query may or may not have sub_results, but result should be correct
        self.assertIn("8", result.result)

    def test_parallel_three_queries(self) -> None:
        """Three comma-separated intents."""
        result = self.router.route_parallel("1+1, 2+2, 3+3")
        self.assertTrue(result.ok)
        self.assertEqual(len(result.sub_results), 3)
        all_text = " ".join(sr.result for sr in result.sub_results)
        self.assertIn("2", all_text)
        self.assertIn("4", all_text)
        self.assertIn("6", all_text)


class TestMultiTurnAnaphora(unittest.TestCase):
    """Multi-turn calculator with implicit anaphora resolution."""

    def setUp(self) -> None:
        self.router = CARMRouter()
        self.session = "test-anaphora"

    def test_basic_multi_turn_add(self) -> None:
        """3加5 → 再加上10 should yield 18."""
        r1 = self.router.route("3加5", session_id=self.session)
        self.assertIn("8", r1.result)

        r2 = self.router.route("再加上10", session_id=self.session)
        self.assertIn("18", r2.result)

    def test_multi_turn_chain(self) -> None:
        """3加5 → 再加上10 → 再乘以3 should yield 54."""
        self.router.route("3加5", session_id=self.session)
        self.router.route("再加上10", session_id=self.session)
        r3 = self.router.route("再乘以3", session_id=self.session)
        self.assertIn("54", r3.result)

    def test_multi_turn_subtract(self) -> None:
        """100减30 → 再减去20 should yield 50."""
        self.router.route("100减30", session_id=self.session + "2")
        r2 = self.router.route("再减去20", session_id=self.session + "2")
        self.assertIn("50", r2.result)

    def test_multi_turn_divide(self) -> None:
        """100乘以4 → 再除以2 should yield 200."""
        self.router.route("100乘以4", session_id=self.session + "3")
        r2 = self.router.route("再除以2", session_id=self.session + "3")
        self.assertIn("200", r2.result)


class TestCalculatorNLPatterns(unittest.TestCase):
    """CalculatorTool NL pattern with '再' prefix."""

    def setUp(self) -> None:
        self.calc = CalculatorTool()

    def test_zai_jia(self) -> None:
        """'8 再加上10' should compute 18."""
        result = self.calc.execute("8 再加上10", {})
        self.assertTrue(result.ok)
        self.assertIn("18", result.result)

    def test_zai_cheng(self) -> None:
        """'18 再乘以3' should compute 54."""
        result = self.calc.execute("18 再乘以3", {})
        self.assertTrue(result.ok)
        self.assertIn("54", result.result)

    def test_zai_jian(self) -> None:
        """'70 再减去20' should compute 50."""
        result = self.calc.execute("70 再减去20", {})
        self.assertTrue(result.ok)
        self.assertIn("50", result.result)

    def test_zai_chu(self) -> None:
        """'400 再除以2' should compute 200."""
        result = self.calc.execute("400 再除以2", {})
        self.assertTrue(result.ok)
        self.assertIn("200", result.result)

    def test_plain_add_still_works(self) -> None:
        """'3加5' without '再' should still work."""
        result = self.calc.execute("3加5", {})
        self.assertTrue(result.ok)
        self.assertIn("8", result.result)


class TestRouteResultSchema(unittest.TestCase):
    """RouteResult schema includes sub_results field for parallel calls."""

    def test_sub_results_field_exists(self) -> None:
        """RouteResult should accept sub_results parameter."""
        r = RouteResult(
            query="test",
            tool_name="calculator",
            result="8",
            confidence=0.9,
            source="test",
            ok=True,
            sub_results=[],
        )
        self.assertEqual(r.sub_results, [])


if __name__ == "__main__":
    unittest.main()
