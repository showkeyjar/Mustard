"""Safe arithmetic calculator — replaces eval()-based implementation.

Parses and evaluates arithmetic expressions using a proper recursive descent
parser instead of Python's eval(). Supports +, -, *, /, **, (), and numeric
literals (int, float). No function calls, no attribute access, no imports.
"""

from __future__ import annotations

import math
import re
from typing import Any

from carm.schemas import ToolResult


class CalculatorTool:
    name = "calculator"

    # Token types
    _NUM = "NUM"
    _OP = "OP"
    _LPAREN = "LPAREN"
    _RPAREN = "RPAREN"
    _EOF = "EOF"

    def execute(self, query: str, arguments: dict) -> ToolResult:
        # First try to parse natural-language arithmetic patterns
        expression = self._extract_nl_expression(query)

        if not expression:
            # Fall back to extracting explicit arithmetic expressions
            candidates = re.findall(r"[0-9\.\+\-\*\/\(\) ]+", query)
            expressions = [
                item.strip()
                for item in candidates
                if re.search(r"\d", item) and re.search(r"[\+\-\*\/]", item)
            ]
            expression = max(expressions, key=len, default="").strip()

        if not expression:
            # Try the whole query if it looks like a pure expression
            if re.search(r"\d", query) and re.search(r"[\+\-\*\/]", query):
                expression = query.strip()
            else:
                return ToolResult(
                    ok=True,
                    tool_name=self.name,
                    result="未找到可计算表达式，建议补充明确数字和算式。",
                    confidence=0.2,
                    source="tool/calculator",
                )

        try:
            tokens = self._tokenize(expression)
            parser = _Parser(tokens)
            value = parser.parse_expression()
            if parser.current_type() != self._EOF:
                raise ValueError(
                    f"Unexpected token after expression: {parser.current_value()}"
                )
            # Format result nicely
            if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
                value = int(value)
            result = f"计算结果: {expression} = {value}"
            confidence = 0.95
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            result = f"表达式计算失败: {exc}。建议检查算式格式。"
            confidence = 0.25
        except Exception:
            result = "找到的表达式不完整，建议补充标准算式。"
            confidence = 0.25

        return ToolResult(
            ok=True,
            tool_name=self.name,
            result=result,
            confidence=confidence,
            source="tool/calculator",
        )

    # ------------------------------------------------------------------
    # Natural language → arithmetic expression extraction
    # ------------------------------------------------------------------

    _NL_PATTERNS: list[tuple[str, re.Pattern]] = [
        # "N个席位每席位M元 按年/年预算/每年" → N * M * 12
        (
            r"(\d+)\s*个?(?:席位|人|用户|台|个).*?每(?:席位|人|用户|台|个)\s*(\d+)"
            r"(?:元|块)?(?:[/每](?:月|年))?.*?(?:按年|年预算|年费|每年|年预算)",
            lambda m: f"{m.group(1)} * {m.group(2)} * 12",
        ),
        # "N个席位每席位M元/月" → N * M (monthly)
        (
            r"(\d+)\s*个?(?:席位|人|用户|台|个).*?每(?:席位|人|用户|台|个)\s*(\d+)"
            r"(?:元|块)?[/每]月",
            lambda m: f"{m.group(1)} * {m.group(2)}",
        ),
        # "N的M次方" → N ** M
        (
            r"(\d+)\s*的\s*(\d+)\s*次方",
            lambda m: f"{m.group(1)} ** {m.group(2)}",
        ),
        # "N乘以/乘M" → N * M
        (
            r"(\d+(?:\.\d+)?)\s*(?:乘以|乘|×)\s*(\d+(?:\.\d+)?)",
            lambda m: f"{m.group(1)} * {m.group(2)}",
        ),
        # "N除以/除M" → N / M
        (
            r"(\d+(?:\.\d+)?)\s*(?:除以|除|÷)\s*(\d+(?:\.\d+)?)",
            lambda m: f"{m.group(1)} / {m.group(2)}",
        ),
        # "N加M" → N + M
        (
            r"(\d+(?:\.\d+)?)\s*(?:加上|加|加以)\s*(\d+(?:\.\d+)?)",
            lambda m: f"{m.group(1)} + {m.group(2)}",
        ),
        # "N减M" → N - M
        (
            r"(\d+(?:\.\d+)?)\s*(?:减去|减)\s*(\d+(?:\.\d+)?)",
            lambda m: f"{m.group(1)} - {m.group(2)}",
        ),
        # "N的平方根/开方" → sqrt(N)
        (
            r"(\d+(?:\.\d+)?)\s*的?(?:平方根|开方|开根号)",
            lambda m: f"{m.group(1)} ** 0.5",
        ),
        # "N的平方" → N ** 2
        (
            r"(\d+(?:\.\d+)?)\s*的?平方",
            lambda m: f"{m.group(1)} ** 2",
        ),
        # "N的M次方" (already covered above but with Chinese numerals)
        (
            r"(\d+(?:\.\d+)?)\s*(?:的)?(\d+(?:\.\d+)?)\s*次方",
            lambda m: f"{m.group(1)} ** {m.group(2)}",
        ),
        # "百分之N" → N / 100
        (
            r"百分之\s*(\d+(?:\.\d+)?)",
            lambda m: f"{m.group(1)} / 100",
        ),
    ]

    def _extract_nl_expression(self, query: str) -> str:
        """Convert natural language arithmetic descriptions into expressions."""
        for pattern, builder in self._NL_PATTERNS:
            m = re.search(pattern, query)
            if m:
                return builder(m)
        return ""

    def _tokenize(self, text: str) -> list[tuple[str, Any]]:
        """Tokenize arithmetic expression into (type, value) pairs."""
        tokens: list[tuple[str, Any]] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch.isspace():
                i += 1
                continue
            if ch.isdigit() or ch == ".":
                # Number literal
                start = i
                has_dot = False
                while i < len(text) and (
                    text[i].isdigit() or (text[i] == "." and not has_dot)
                ):
                    if text[i] == ".":
                        has_dot = True
                    i += 1
                num_str = text[start:i]
                try:
                    if has_dot:
                        tokens.append((self._NUM, float(num_str)))
                    else:
                        tokens.append((self._NUM, int(num_str)))
                except ValueError:
                    raise ValueError(f"Invalid number: {num_str}")
                continue
            if ch == "(":
                tokens.append((self._LPAREN, "("))
            elif ch == ")":
                tokens.append((self._RPAREN, ")"))
            elif ch in "+-":
                tokens.append((self._OP, ch))
            elif ch == "*":
                if i + 1 < len(text) and text[i + 1] == "*":
                    tokens.append((self._OP, "**"))
                    i += 1
                else:
                    tokens.append((self._OP, "*"))
            elif ch == "/":
                tokens.append((self._OP, "/"))
            else:
                raise ValueError(f"Unexpected character: {ch}")
            i += 1
        tokens.append((self._EOF, None))
        return tokens


class _Parser:
    """Recursive descent parser for arithmetic expressions.

    Grammar:
        expression = term (('+' | '-') term)*
        term       = power (('*' | '/') power)*
        power      = unary ('**' power)?
        unary      = ('+' | '-')? atom
        atom       = NUMBER | '(' expression ')'
    """

    def __init__(self, tokens: list[tuple[str, Any]]) -> None:
        self._tokens = tokens
        self._pos = 0

    def current_type(self) -> str:
        return self._tokens[self._pos][0]

    def current_value(self) -> Any:
        return self._tokens[self._pos][1]

    def _advance(self) -> None:
        if self._pos < len(self._tokens) - 1:
            self._pos += 1

    def parse_expression(self) -> float:
        """expression = term (('+' | '-') term)*"""
        result = self._parse_term()
        while self.current_type() == CalculatorTool._OP and self.current_value() in (
            "+",
            "-",
        ):
            op = self.current_value()
            self._advance()
            right = self._parse_term()
            if op == "+":
                result = result + right
            else:
                result = result - right
        return result

    def _parse_term(self) -> float:
        """term = power (('*' | '/') power)*"""
        result = self._parse_power()
        while self.current_type() == CalculatorTool._OP and self.current_value() in (
            "*",
            "/",
        ):
            op = self.current_value()
            self._advance()
            right = self._parse_power()
            if op == "*":
                result = result * right
            else:
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                result = result / right
        return result

    def _parse_power(self) -> float:
        """power = unary ('**' power)?"""
        base = self._parse_unary()
        if self.current_type() == CalculatorTool._OP and self.current_value() == "**":
            self._advance()
            exponent = self._parse_power()  # right-associative
            result = base**exponent
            if isinstance(result, complex):
                raise ValueError("Complex result not supported")
            return result
        return base

    def _parse_unary(self) -> float:
        """unary = ('+' | '-')? atom"""
        if self.current_type() == CalculatorTool._OP and self.current_value() in (
            "+",
            "-",
        ):
            op = self.current_value()
            self._advance()
            value = self._parse_atom()
            return value if op == "+" else -value
        return self._parse_atom()

    def _parse_atom(self) -> float:
        """atom = NUMBER | '(' expression ')'"""
        if self.current_type() == CalculatorTool._NUM:
            value = self.current_value()
            self._advance()
            return float(value)
        if self.current_type() == CalculatorTool._LPAREN:
            self._advance()
            result = self.parse_expression()
            if self.current_type() != CalculatorTool._RPAREN:
                raise ValueError("Missing closing parenthesis")
            self._advance()
            return result
        raise ValueError(
            f"Expected number or '(', got {self.current_type()}: {self.current_value()}"
        )
