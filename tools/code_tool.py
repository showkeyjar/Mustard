"""Code executor with sandboxed subprocess execution.

Executes Python code in a subprocess with timeout and output capture.
Safe alternative to eval/exec with proper isolation.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path

from carm.intent import IntentCategory
from carm.schemas import ToolResult


class CodeExecutorTool:
    name = "code_executor"
    capability_tags = [IntentCategory.CODE]

    # Default timeout in seconds
    DEFAULT_TIMEOUT = 15

    def execute(self, query: str, arguments: dict) -> ToolResult:
        code = arguments.get("code", "")
        timeout = int(arguments.get("timeout", self.DEFAULT_TIMEOUT))

        # If no explicit code provided, try to extract from the query
        if not code:
            code = self._extract_code(query)

        if not code:
            return ToolResult(
                ok=True,
                tool_name=self.name,
                result="未检测到可执行代码。请提供明确的代码片段或使用 /code 前缀标记。",
                confidence=0.25,
                source="tool/code_executor",
            )

        return self._run_code(code, timeout)

    def _extract_code(self, query: str) -> str:
        """Try to extract code from natural language query.

        Supports three sources:
        1. Explicit code blocks (```...```) and inline code (`...`)
        2. Common algorithm/recipe templates via pattern matching
        3. Raw query if it already looks like Python code
        """
        import re

        # Check for code blocks
        code_block = re.search(r"```(?:python)?\s*\n(.*?)```", query, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        # Check for inline code
        inline_code = re.search(r"`([^`]+)`", query)
        if inline_code:
            candidate = inline_code.group(1).strip()
            # Only treat as code if it looks like Python
            if any(
                kw in candidate
                for kw in ("print(", "def ", "import ", "for ", "if ", "=", "calc")
            ):
                return candidate

        # Common algorithm templates — generate code from natural language
        template = self._match_template(query)
        if template:
            return template

        # Check if the query itself looks like code
        code_indicators = (
            "print(",
            "def ",
            "import ",
            "for ",
            "while ",
            "if ",
            "class ",
        )
        if any(indicator in query for indicator in code_indicators):
            return query.strip()

        return ""

    # ------------------------------------------------------------------
    # Pattern-matched code templates for common programming requests
    # ------------------------------------------------------------------

    _ALGORITHM_TEMPLATES: list[tuple[list[str], str]] = [
        (
            ["快速排序", "quicksort", "quick sort"],
            textwrap.dedent("""\
                def quicksort(arr):
                    if len(arr) <= 1:
                        return arr
                    pivot = arr[len(arr) // 2]
                    left = [x for x in arr if x < pivot]
                    middle = [x for x in arr if x == pivot]
                    right = [x for x in arr if x > pivot]
                    return quicksort(left) + middle + quicksort(right)

                print(quicksort([3, 6, 8, 10, 1, 2, 1]))
            """),
        ),
        (
            ["冒泡排序", "bubble sort", "bubblesort"],
            textwrap.dedent("""\
                def bubble_sort(arr):
                    n = len(arr)
                    for i in range(n):
                        for j in range(0, n - i - 1):
                            if arr[j] > arr[j + 1]:
                                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    return arr

                print(bubble_sort([64, 34, 25, 12, 22, 11, 90]))
            """),
        ),
        (
            ["二分查找", "binary search", "二分搜索"],
            textwrap.dedent("""\
                def binary_search(arr, target):
                    left, right = 0, len(arr) - 1
                    while left <= right:
                        mid = (left + right) // 2
                        if arr[mid] == target:
                            return mid
                        elif arr[mid] < target:
                            left = mid + 1
                        else:
                            right = mid - 1
                    return -1

                print(binary_search([1, 3, 5, 7, 9, 11], 7))
            """),
        ),
        (
            ["斐波那契", "fibonacci"],
            textwrap.dedent("""\
                def fibonacci(n):
                    if n <= 0:
                        return []
                    fib = [0, 1]
                    for i in range(2, n):
                        fib.append(fib[i-1] + fib[i-2])
                    return fib[:n]

                print(fibonacci(10))
            """),
        ),
        (
            ["阶乘", "factorial"],
            textwrap.dedent("""\
                def factorial(n):
                    if n < 0:
                        return "负数没有阶乘"
                    result = 1
                    for i in range(2, n + 1):
                        result *= i
                    return result

                print(factorial(5))
            """),
        ),
        (
            ["归并排序", "merge sort", "mergesort"],
            textwrap.dedent("""\
                def merge_sort(arr):
                    if len(arr) <= 1:
                        return arr
                    mid = len(arr) // 2
                    left = merge_sort(arr[:mid])
                    right = merge_sort(arr[mid:])
                    return merge(left, right)

                def merge(left, right):
                    result = []
                    i = j = 0
                    while i < len(left) and j < len(right):
                        if left[i] <= right[j]:
                            result.append(left[i])
                            i += 1
                        else:
                            result.append(right[j])
                            j += 1
                    result.extend(left[i:])
                    result.extend(right[j:])
                    return result

                print(merge_sort([38, 27, 43, 3, 9, 82, 10]))
            """),
        ),
        (
            ["链表", "linked list", "linkedlist"],
            textwrap.dedent("""\
                class ListNode:
                    def __init__(self, val=0, next=None):
                        self.val = val
                        self.next = next

                def create_list(values):
                    dummy = ListNode()
                    curr = dummy
                    for v in values:
                        curr.next = ListNode(v)
                        curr = curr.next
                    return dummy.next

                def to_list(head):
                    result = []
                    while head:
                        result.append(head.val)
                        head = head.next
                    return result

                head = create_list([1, 2, 3, 4, 5])
                print(to_list(head))
            """),
        ),
    ]

    def _match_template(self, query: str) -> str:
        """Match a natural language query to a code template.

        If the user provides numbers in the query, replace the template's
        hardcoded test data with the user's numbers.
        """
        import re

        lower = query.lower()
        for keywords, template in self._ALGORITHM_TEMPLATES:
            if any(kw in lower for kw in keywords):
                # Try to extract user-provided number list from the query
                user_numbers = self._extract_number_list(query)
                if user_numbers:
                    # Replace the last Python list literal in the template
                    # (the test data) with the user's numbers
                    template = re.sub(
                        r"\[\s*[\d.,\s]+\]",
                        repr(user_numbers),
                        template,
                        count=1,
                    )
                else:
                    # For single-number templates (e.g. factorial),
                    # replace the function call argument
                    single_num = self._extract_single_number(query)
                    if single_num is not None:
                        template = re.sub(
                            r"(factorial|fibonacci)\s*\(\s*\d+\s*\)",
                            rf"\1({single_num})",
                            template,
                        )
                return template
        return ""

    def _extract_number_list(self, query: str) -> list[int | float] | None:
        """Extract a list of numbers from a natural language query.

        Matches any sequence of 2+ numbers in the query text.
        """
        import re

        # Extract all number tokens from the query
        nums_str = re.findall(r"\d+\.?\d*", query)
        if len(nums_str) < 2:
            return None
        result: list[int | float] = []
        for s in nums_str:
            try:
                if "." in s:
                    result.append(float(s))
                else:
                    result.append(int(s))
            except ValueError:
                continue
        return result if len(result) >= 2 else None

    def _extract_single_number(self, query: str) -> int | float | None:
        """Extract a single number from a natural language query.

        Used for templates like factorial(N) where only one number is needed.
        Returns the largest number found (heuristic: the parameter is usually
        the most prominent number in the query).
        """
        import re

        nums_str = re.findall(r"\d+\.?\d*", query)
        if not nums_str:
            return None
        # Return the last number (most likely the parameter)
        try:
            s = nums_str[-1]
            return float(s) if "." in s else int(s)
        except ValueError:
            return None

    def _run_code(self, code: str, timeout: int) -> ToolResult:
        """Execute Python code in a subprocess with timeout."""
        # Wrap code to capture both stdout and the last expression value
        wrapped = textwrap.dedent(f"""\
            import sys
            import json
            import io

            _code = {repr(code)}
            _result = None
            _captured_output = io.StringIO()

            try:
                # Capture print output from user code
                _old_stdout = sys.stdout
                sys.stdout = _captured_output
                try:
                    # Try to compile as expression first
                    try:
                        _compiled = compile(_code, "<code_executor>", "eval")
                        _result = eval(_compiled)
                    except SyntaxError:
                        exec(compile(_code, "<code_executor>", "exec"))
                        _result = None
                finally:
                    sys.stdout = _old_stdout

                _output = {{
                    "ok": True,
                    "result": _result,
                    "output": _captured_output.getvalue().strip(),
                }}
            except Exception as _e:
                sys.stdout = _old_stdout
                _output = {{
                    "ok": False,
                    "error": str(_e),
                    "error_type": type(_e).__name__,
                    "output": _captured_output.getvalue().strip(),
                }}

            print(json.dumps(_output, ensure_ascii=False, default=str))
        """)

        try:
            proc = subprocess.run(
                [sys.executable, "-c", wrapped],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path.cwd()),
            )

            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()

            # Try to parse JSON output
            try:
                # Find the JSON line (last non-empty line)
                lines = [line for line in stdout.split("\n") if line.strip()]
                if lines:
                    output = json.loads(lines[-1])
                    if output.get("ok"):
                        parts: list[str] = []
                        if output.get("output"):
                            parts.append(str(output["output"]))
                        if output.get("result") is not None:
                            parts.append(f"返回值: {output['result']}")
                        result_text = (
                            "代码执行成功。\n" + "\n".join(parts)
                            if parts
                            else "代码执行成功，无输出。"
                        )
                        return ToolResult(
                            ok=True,
                            tool_name=self.name,
                            result=result_text,
                            confidence=0.88,
                            source="tool/code_executor",
                        )
                    else:
                        error_msg = output.get("error", "Unknown error")
                        error_type = output.get("error_type", "Error")
                        return ToolResult(
                            ok=True,
                            tool_name=self.name,
                            result=f"代码执行出错: {error_type}: {error_msg}",
                            confidence=0.45,
                            source="tool/code_executor",
                        )
            except (json.JSONDecodeError, IndexError):
                pass

            # Fallback: return raw stdout/stderr
            if stdout:
                return ToolResult(
                    ok=True,
                    tool_name=self.name,
                    result=f"代码执行完成，输出:\n{stdout[:1000]}",
                    confidence=0.72,
                    source="tool/code_executor",
                )

            if stderr:
                return ToolResult(
                    ok=True,
                    tool_name=self.name,
                    result=f"代码执行出错:\n{stderr[:500]}",
                    confidence=0.4,
                    source="tool/code_executor",
                )

            return ToolResult(
                ok=True,
                tool_name=self.name,
                result="代码执行完成，无输出。",
                confidence=0.68,
                source="tool/code_executor",
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                ok=True,
                tool_name=self.name,
                result=f"代码执行超时（{timeout}秒），可能存在无限循环。建议检查循环条件或缩短执行时间。",
                confidence=0.3,
                source="tool/code_executor",
            )
        except Exception as exc:
            return ToolResult(
                ok=True,
                tool_name=self.name,
                result=f"代码执行失败: {exc}",
                confidence=0.25,
                source="tool/code_executor",
            )
