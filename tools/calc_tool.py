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
        # ── Unit conversion patterns (must come before generic arithmetic) ──
        # "N公里等于多少米" → N * 1000
        (
            r"(\d+(?:\.\d+)?)\s*公里等于多少?米",
            lambda m: f"{m.group(1)} * 1000",
        ),
        # "N千米等于多少米" → N * 1000
        (
            r"(\d+(?:\.\d+)?)\s*千米等于多少?米",
            lambda m: f"{m.group(1)} * 1000",
        ),
        # "N小时等于多少分钟" → N * 60
        (
            r"(\d+(?:\.\d+)?)\s*小时等于多少?分钟",
            lambda m: f"{m.group(1)} * 60",
        ),
        # "N天等于多少小时" → N * 24
        (
            r"(\d+(?:\.\d+)?)\s*天等于多少?小时",
            lambda m: f"{m.group(1)} * 24",
        ),
        # "N吨等于多少千克" → N * 1000
        (
            r"(\d+(?:\.\d+)?)\s*吨等于多少?千克",
            lambda m: f"{m.group(1)} * 1000",
        ),
        # ── Simple word problem patterns ───────────────────────────────
        # Multi-purchase: "买了N1本书每本M1元又买了N2支笔每支M2元一共多少钱"
        # → N1 * M1 + N2 * M2
        # Uses .*? between unit and "每" to skip intervening nouns like "书/笔"
        (
            r"(?:买了?|采购了?)\s*(\d+(?:\.\d+)?)\s*(?:个?|只?|本?|件?|台?|瓶?|斤?|箱?|支?).*?"
            r"每(?:个?|只?|本?|件?|台?|瓶?|斤?|箱?|支?)\s*(\d+(?:\.\d+)?)\s*元"
            r".*?(?:又|再).*?"
            r"(?:买了?|采购了?)\s*(\d+(?:\.\d+)?)\s*(?:个?|只?|本?|件?|台?|瓶?|斤?|箱?|支?).*?"
            r"每(?:个?|只?|本?|件?|台?|瓶?|斤?|箱?|支?)\s*(\d+(?:\.\d+)?)\s*元"
            r".*?(?:一共|总共|共|多少|钱)",
            lambda m: f"{m.group(1)} * {m.group(2)} + {m.group(3)} * {m.group(4)}",
        ),
        # Multi-purchase variant: "N1个X M1元/个 又N2个Y M2元/个 一共"
        (
            r"(\d+(?:\.\d+)?)\s*(?:个?|只?|本?|件?|台?|瓶?|斤?|箱?|支?).*?"
            r"(?:每(?:个?|只?|本?|件?|台?|瓶?|斤?|箱?|支?))?\s*(\d+(?:\.\d+)?)\s*元"
            r".*?(?:又|再).*?"
            r"(\d+(?:\.\d+)?)\s*(?:个?|只?|本?|件?|台?|瓶?|斤?|箱?|支?).*?"
            r"(?:每(?:个?|只?|本?|件?|台?|瓶?|斤?|箱?|支?))?\s*(\d+(?:\.\d+)?)\s*元"
            r".*?(?:一共|总共|共|多少|钱)",
            lambda m: f"{m.group(1)} * {m.group(2)} + {m.group(3)} * {m.group(4)}",
        ),
        # "有N个XX又买了M个一共有多少" → N + M
        (
            r"(?:有|买了?|得到)\s*(\d+(?:\.\d+)?)\s*个?.*?(?:又|再).*?"
            r"(?:买了?|得到|拿)\s*(\d+(?:\.\d+)?)\s*个?.*?(?:一共|总共|共有)",
            lambda m: f"{m.group(1)} + {m.group(2)}",
        ),
        # "每X M元 买N 需要多少钱" → M * N
        (
            r"(?:每|1[个只本])\s*(?:个?|只?|本?)\s*(\d+(?:\.\d+)?)\s*元?"
            r".*?(?:买|要)\s*(\d+(?:\.\d+)?)\s*(?:个?|只?|本?|件?)"
            r".*?(?:需要|一共|总共|多少)",
            lambda m: f"{m.group(1)} * {m.group(2)}",
        ),
        # "一X M元 买N 需要多少钱" → M * N (e.g. "一本书15元买3本需要多少钱")
        (
            r"一?(?:个?|只?|本?|件?|台?|瓶?|斤?|箱?)"
            r"\s*(\d+(?:\.\d+)?)\s*元"
            r".*?(?:买|要)\s*(\d+(?:\.\d+)?)\s*(?:个?|只?|本?|件?|台?|瓶?|斤?|箱?)"
            r".*?(?:需要|一共|总共|多少|钱)",
            lambda m: f"{m.group(1)} * {m.group(2)}",
        ),
        # ── Enterprise pricing patterns ─────────────────────────────────
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
        # ── Percentage / discount patterns ───────────────────────────
        # "原价N元打M折" → N * M / 10
        (
            r"(?:原价|原)\s*(\d+(?:\.\d+)?)\s*元?\s*打\s*(\d+(?:\.\d+)?)\s*折",
            lambda m: f"{m.group(1)} * {m.group(2)} / 10",
        ),
        # "N元打M折" → N * M / 10
        (
            r"(\d+(?:\.\d+)?)\s*元?\s*打\s*(\d+(?:\.\d+)?)\s*折",
            lambda m: f"{m.group(1)} * {m.group(2)} / 10",
        ),
        # "N人/个M%(及格率/合格率等)" → N * M / 100
        # Must come BEFORE general percentage pattern
        (
            r"(\d+(?:\.\d+)?)\s*(?:个?|人|名|位)\s*(?:及格率|合格率|通过率|成功率|比例)\s*(\d+(?:\.\d+)?)\s*[%％]",
            lambda m: f"{m.group(1)} * {m.group(2)} / 100",
        ),
        # "N的M%是多少" → N * M / 100
        (
            r"(\d+(?:\.\d+)?)\s*的\s*(\d+(?:\.\d+)?)\s*[%％]",
            lambda m: f"{m.group(1)} * {m.group(2)} / 100",
        ),
        # "N元打M折" → N * M / 10
        (
            r"(\d+(?:\.\d+)?)\s*元?\s*打\s*(\d+(?:\.\d+)?)\s*折",
            lambda m: f"{m.group(1)} * {m.group(2)} / 10",
        ),
        # "NM%是多少" / "N的M%是多少" → N * M / 100
        (
            r"(\d+(?:\.\d+)?)\s*的?\s*(\d+(?:\.\d+)?)\s*[%％]",
            lambda m: f"{m.group(1)} * {m.group(2)} / 100",
        ),
        # "NM%" as standalone → N * M / 100
        (
            r"(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)\s*[%％]",
            lambda m: f"{m.group(1)} * {m.group(2)} / 100",
        ),
        # "N人M%是多少" / "及格率M%" → N * M / 100
        (
            r"(\d+(?:\.\d+)?)\s*(?:个?|人?|名?)\s*(?:及格率|合格率|通过率|比例)?\s*(\d+(?:\.\d+)?)\s*[%％]",
            lambda m: f"{m.group(1)} * {m.group(2)} / 100",
        ),
        # ── Speed / rate patterns ────────────────────────────────────
        # "相距N公里 M小时到达 平均速度" → N / M
        (
            r"(?:相距|距离|路程)\s*(\d+(?:\.\d+)?)\s*公里?\s*.*?(\d+(?:\.\d+)?)\s*小时.*?(?:平均|速度)",
            lambda m: f"{m.group(1)} / {m.group(2)}",
        ),
        # "N公里 M小时 速度" → N / M
        (
            r"(\d+(?:\.\d+)?)\s*公里?\s*.*?(\d+(?:\.\d+)?)\s*小时.*?(?:速度|每?小时)",
            lambda m: f"{m.group(1)} / {m.group(2)}",
        ),
        # "N的M次方" → N ** M
        (
            r"(\d+)\s*的\s*(\d+)\s*次方",
            lambda m: f"{m.group(1)} ** {m.group(2)}",
        ),
        # Chain arithmetic: "5加3乘2" → 5 + 3 * 2
        # Matches sequences like: N op M op N op M ...
        # Also supports negative numbers: "-3加5" → -3 + 5
        (
            r"((?:-?\d+(?:\.\d+)?\s*(?:加上?|减去?|乘以?|乘|除以?|除|×|÷)\s*)+-?\d+(?:\.\d+)?)",
            lambda m: _convert_chain(m.group(1)),
        ),
        # Chinese large numbers: "1万亿除以14亿" → 100000000000 / 1400000000
        (
            r"(\d+)\s*万亿",
            lambda m: str(int(m.group(1)) * 100000000000),
        ),
        (
            r"(\d+)\s*亿",
            lambda m: str(int(m.group(1)) * 100000000),
        ),
        (
            r"(\d+)\s*万",
            lambda m: str(int(m.group(1)) * 10000),
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
        # ── Geometry patterns ──────────────────────────────────────────
        # "圆的面积，半径是N" → pi * N ** 2
        (
            r"(?:圆|圆形).*?(?:半径|r)\s*[是为]?\s*(\d+(?:\.\d+)?)",
            lambda m: f"3.14159265 * {m.group(1)} ** 2",
        ),
        # "半径N的圆面积" → pi * N ** 2
        (
            r"半径\s*(\d+(?:\.\d+)?).*?(?:圆|面积)",
            lambda m: f"3.14159265 * {m.group(1)} ** 2",
        ),
        # "长A宽B的矩形面积" → A * B
        (
            r"长\s*(\d+(?:\.\d+)?)\s*宽\s*(\d+(?:\.\d+)?).*?(?:矩形|面积|的面积)",
            lambda m: f"{m.group(1)} * {m.group(2)}",
        ),
        # "矩形 长5 宽3 面积" → 5 * 3
        (
            r"(?:矩形|长方形).*?长\s*(\d+(?:\.\d+)?).*?宽\s*(\d+(?:\.\d+)?)",
            lambda m: f"{m.group(1)} * {m.group(2)}",
        ),
        # ── Negative number arithmetic ────────────────────────────────
        # "负3加5" → -3 + 5  (after preprocessing "负3" → "-3")
        (
            r"(-\d+(?:\.\d+)?)\s*(?:加上?|减去?|乘以?|乘|除以?|除|×|÷)\s*(\d+(?:\.\d+)?)",
            lambda m: (
                f"{m.group(1)} + {m.group(2)}"
                if "加" in (m.group(0))
                else f"{m.group(1)} - {m.group(2)}"
                if "减" in m.group(0)
                else f"{m.group(1)} * {m.group(2)}"
                if "乘" in m.group(0) or "×" in m.group(0)
                else f"{m.group(1)} / {m.group(2)}"
            ),
        ),
    ]

    def _extract_nl_expression(self, query: str) -> str:
        """Convert natural language arithmetic descriptions into expressions."""
        # Pre-process: expand Chinese large number units
        preprocessed = self._expand_chinese_numbers(query)
        # Pre-process: convert negative numbers ("负3" → "-3")
        preprocessed = re.sub(r"负\s*(\d+)", r"-\1", preprocessed)
        for pattern, builder in self._NL_PATTERNS:
            m = re.search(pattern, preprocessed)
            if m:
                return builder(m)
        return ""

    def _expand_chinese_numbers(self, text: str) -> str:
        """Replace Chinese large number units with their numeric values.

        '1万亿' → '100000000000', '14亿' → '1400000000', '5万' → '50000'
        """
        text = re.sub(r"(\d+)\s*万亿", lambda m: str(int(m.group(1)) * 10**12), text)
        text = re.sub(r"(\d+)\s*亿", lambda m: str(int(m.group(1)) * 10**8), text)
        text = re.sub(r"(\d+)\s*万", lambda m: str(int(m.group(1)) * 10**4), text)
        return text

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


def _convert_chain(text: str) -> str:
    """Convert a Chinese arithmetic chain like '5加3乘2' into '5 + 3 * 2'."""
    mapping = {
        "加上": "+",
        "加": "+",
        "减去": "-",
        "减": "-",
        "乘以": "*",
        "乘": "*",
        "除以": "/",
        "除": "/",
        "×": "*",
        "÷": "/",
    }
    result = text
    # Replace longest operators first to avoid partial matches
    for zh_op in sorted(mapping, key=len, reverse=True):
        result = result.replace(zh_op, f" {mapping[zh_op]} ")
    return result.strip()
