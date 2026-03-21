import unittest

from tools.calc_tool import CalculatorTool


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


if __name__ == "__main__":
    unittest.main()
