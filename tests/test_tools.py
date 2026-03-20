import unittest

from tools.calc_tool import CalculatorTool


class ToolTests(unittest.TestCase):
    def test_calculator_returns_result(self) -> None:
        tool = CalculatorTool()
        result = tool.execute("2 + 3 * 4", {})

        self.assertTrue(result.ok)
        self.assertIn("14", result.result)


if __name__ == "__main__":
    unittest.main()
