import json
import os
import unittest
from unittest.mock import MagicMock, patch

from tools.calc_tool import CalculatorTool
from tools.bigmodel_tool import BigModelProxyTool


class ToolTests(unittest.TestCase):
    def test_calculator_returns_result(self) -> None:
        tool = CalculatorTool()
        result = tool.execute("2 + 3 * 4", {})

        self.assertTrue(result.ok)
        self.assertIn("14", result.result)

    def test_calculator_prefers_explicit_formula_inside_natural_language(self) -> None:
        tool = CalculatorTool()
        result = tool.execute("套餐说明写着每席位 79 元，12 人团队按年付总价是多少？请按 79 * 12 * 12 精确计算。", {})

        self.assertTrue(result.ok)
        self.assertIn("11376", result.result)

    def test_bigmodel_proxy_distill_can_use_gemini_rest_when_key_present(self) -> None:
        tool = BigModelProxyTool()
        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "task_type": "calculate",
                                        "logic_skill": "tool_selection",
                                        "expected_tool": "calculator",
                                        "target_slot": "HYP",
                                        "plan_summary": "teacher",
                                        "plan_action_items": ["识别变量", "执行计算", "核对结果"],
                                        "plan_unknowns": ["税费未知"],
                                        "evidence_targets": ["精确数值"],
                                        "draft_summary": "输出结果",
                                        "quality_score": 0.95,
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        ]
                    }
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_payload, ensure_ascii=False).encode("utf-8")
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_response
        mock_context.__exit__.return_value = False

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-2.5-flash"}, clear=False):
            with patch("tools.bigmodel_tool.request.urlopen", return_value=mock_context) as mocked:
                result = tool.execute("请计算 1 + 2", {"mode": "distill"})

        self.assertTrue(result.ok)
        self.assertIn("calculator", result.result)
        self.assertIn("gemini-2.5-flash", mocked.call_args.args[0].full_url)


if __name__ == "__main__":
    unittest.main()
