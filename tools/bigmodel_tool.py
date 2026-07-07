from __future__ import annotations

import json
import os
from urllib import error, parse, request

from carm.intent import IntentCategory
from carm.schemas import ToolResult


class BigModelProxyTool:
    name = "bigmodel_proxy"
    capability_tags = [IntentCategory.CONSULT, IntentCategory.SEARCH]

    def execute(self, query: str, arguments: dict) -> ToolResult:
        # Path-C: extract CARM signal analysis if present
        carm_signals = arguments.get("carm_signals", "")

        # Strategy 1: Gemini API (cloud, highest quality)
        if self._gemini_enabled():
            live_result = self._execute_gemini(query, arguments, carm_signals)
            if live_result is not None:
                return live_result

        # Strategy 2: Ollama local LLM (zero-config, good quality)
        ollama_result = self._execute_ollama(query, arguments, carm_signals)
        if ollama_result is not None:
            return ollama_result

        # Strategy 3: Distill mode (structured teacher sample)
        if arguments.get("mode") == "distill":
            return ToolResult(
                ok=True,
                tool_name=self.name,
                result=json.dumps(self._distill_payload(query), ensure_ascii=False),
                confidence=0.88,
                source="tool/bigmodel_proxy:distill_fallback",
            )

        # Strategy 4: Honest fallback — tell user what's missing
        return ToolResult(
            ok=False,
            tool_name=self.name,
            result=(
                f"无可用大模型后端。请设置 GEMINI_API_KEY 环境变量连接 Gemini，"
                f"或启动 Ollama（ollama serve）连接本地模型。原问题: {query[:80]}"
            ),
            confidence=0.15,
            source="tool/bigmodel_proxy:no_backend",
        )

    # ── Gemini ──────────────────────────────────────────────────────────

    def _gemini_enabled(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY", "").strip())

    def _execute_gemini(
        self, query: str, arguments: dict, carm_signals: str = ""
    ) -> ToolResult | None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None

        model = (
            os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
            or "gemini-2.5-flash"
        )
        timeout_s = float(os.environ.get("GEMINI_TIMEOUT_S", "30") or 30)
        mode = str(arguments.get("mode", "")).strip().lower()
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{parse.quote(model, safe='')}:generateContent?key={parse.quote(api_key, safe='')}"
        )

        payload = self._build_gemini_payload(query, mode, carm_signals)
        req = request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-client": "mustard-carm/1.0",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
            return None

        text = self._extract_gemini_text(body)
        if not text:
            return None

        if mode == "distill":
            try:
                normalized = json.loads(text)
                text = json.dumps(normalized, ensure_ascii=False)
            except json.JSONDecodeError:
                return None

        return ToolResult(
            ok=True,
            tool_name=self.name,
            result=text,
            confidence=0.92 if mode == "distill" else 0.86,
            source=f"tool/bigmodel_proxy:{model}",
        )

    def _build_gemini_payload(
        self, query: str, mode: str, carm_signals: str = ""
    ) -> dict[str, object]:
        if mode == "distill":
            return {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": (
                                    "You are a teacher model for a small reasoning controller. "
                                    "Convert the task into a compact JSON training sample.\n\n"
                                    f"Task:\n{query}"
                                )
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseJsonSchema": self._distill_schema(),
                },
            }

        # Path-C: build system prompt with CARM signal context
        system_text = (
            "You are an external large model used by a small logic controller (CARM). "
            "Return a concise, practical answer grounded in the user's request. "
            "Prefer structure over flourish."
        )
        if carm_signals:
            system_text += (
                f"\n\nCARM detected these intent signals: [{carm_signals}]. "
                "Use these as priors to guide your response."
            )

        return {
            "systemInstruction": {
                "parts": [
                    {
                        "text": system_text,
                    }
                ]
            },
            "contents": [
                {
                    "parts": [
                        {
                            "text": query,
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
            },
        }

    def _extract_gemini_text(self, payload: dict[str, object]) -> str:
        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list):
            return ""
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content", {})
            if not isinstance(content, dict):
                continue
            parts = content.get("parts", [])
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    return str(part.get("text", "")).strip()
        return ""

    # ── Ollama local LLM ────────────────────────────────────────────────

    def _execute_ollama(
        self, query: str, arguments: dict, carm_signals: str = ""
    ) -> ToolResult | None:
        """Try Ollama local LLM (default http://localhost:11434)."""
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip(
            "/"
        )
        model = os.environ.get("OLLAMA_MODEL", "qwen3-coder")
        timeout_s = float(os.environ.get("OLLAMA_TIMEOUT_S", "120") or 120)

        endpoint = f"{base_url}/api/generate"
        mode = str(arguments.get("mode", "")).strip().lower()

        system_prompt = (
            "You are an external large model assisting a small logic controller (CARM). "
            "Return a concise, practical answer grounded in the user's request. "
            "Prefer structure over flourish. Respond in the same language as the query."
        )
        # Path-C: inject CARM signal analysis
        if carm_signals:
            system_prompt += (
                f"\n\nCARM detected these intent signals: [{carm_signals}]. "
                "Use these as priors to guide your response."
            )
        if mode == "distill":
            system_prompt = (
                "You are a teacher model for a small reasoning controller. "
                "Convert the task into a compact JSON training sample."
            )

        payload = {
            "model": model,
            "prompt": f"{system_prompt}\n\n{query}",
            "stream": False,
            "options": {"temperature": 0.2},
        }

        req = request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (
            error.URLError,
            error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
            OSError,
        ):
            return None

        text = str(body.get("response", "")).strip()
        if not text:
            return None

        # Extract the non-thinking part if the model uses <think> tags
        if "<think>" in text and "</think>" in text:
            after_think = text.split("</think>", 1)
            text = after_think[1].strip() if len(after_think) > 1 else text

        if mode == "distill":
            try:
                normalized = json.loads(text)
                text = json.dumps(normalized, ensure_ascii=False)
            except json.JSONDecodeError:
                return None

        return ToolResult(
            ok=True,
            tool_name=self.name,
            result=text,
            confidence=0.84 if mode == "distill" else 0.80,
            source=f"tool/bigmodel_proxy:ollama/{model}",
        )

        # ── Distill fallback ────────────────────────────────────────────────

    def _distill_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "task_type": {"type": "string"},
                "logic_skill": {"type": "string"},
                "expected_tool": {"type": "string"},
                "target_slot": {"type": "string"},
                "plan_summary": {"type": "string"},
                "plan_action_items": {"type": "array", "items": {"type": "string"}},
                "plan_unknowns": {"type": "array", "items": {"type": "string"}},
                "evidence_targets": {"type": "array", "items": {"type": "string"}},
                "draft_summary": {"type": "string"},
                "quality_score": {"type": "number"},
            },
            "required": [
                "task_type",
                "logic_skill",
                "expected_tool",
                "target_slot",
                "plan_summary",
                "plan_action_items",
                "plan_unknowns",
                "evidence_targets",
                "draft_summary",
                "quality_score",
            ],
        }

    def _distill_payload(self, query: str) -> dict[str, object]:
        lower = query.lower()
        task_type = "planning"
        logic_skill = "tool_selection"
        expected_tool = "search"
        target_slot = "PLAN"
        action_items = ["明确任务目标", "判断最合适的外部能力", "输出可执行结构"]
        unknowns = ["任务边界仍需澄清"]
        evidence_targets = ["外部事实", "执行约束"]
        draft_summary = "先形成结构化推理骨架，再根据外部结果收束结论。"

        if any(token in query for token in ("比较", "对比", "优缺点")):
            task_type = "compare"
            logic_skill = "comparison"
            expected_tool = "search"
            target_slot = "PLAN"
            action_items = ["确定比较对象", "列出比较维度", "基于证据给出取舍建议"]
            unknowns = ["场景约束可能不完整"]
            evidence_targets = ["成本", "性能", "维护复杂度", "生态成熟度"]
            draft_summary = "按维度整合比较证据并输出建议。"
        elif any(
            token in query
            for token in ("计算", "预算", "总价", "每席位", "按年", "分几批", "多少")
        ) or any(op in query for op in ("*", "/", "+", "-")):
            task_type = "calculate"
            logic_skill = "tool_selection"
            expected_tool = "calculator"
            target_slot = "HYP"
            action_items = ["识别数值变量", "调用精确计算", "核对结果与单位"]
            unknowns = ["是否存在额外计费口径"]
            evidence_targets = ["精确数值", "单位", "计费口径"]
            draft_summary = "基于精确计算结果输出简洁结论。"
        elif any(
            token in lower
            for token in ("python", "code", "script", "代码", "脚本", "报错")
        ):
            task_type = "coding"
            logic_skill = "tool_selection"
            expected_tool = "code_executor"
            target_slot = "PLAN"
            action_items = ["定位代码问题", "构造验证步骤", "确认修复方向"]
            unknowns = ["执行上下文可能不完整"]
            evidence_targets = ["异常信息", "输入输出", "执行结果"]
            draft_summary = "基于可执行验证输出修复建议。"
        elif any(
            token in query
            for token in ("负责人", "管理层", "正式", "简洁", "材料", "资料", "组织")
        ):
            task_type = "summarize"
            logic_skill = "result_integration"
            expected_tool = "bigmodel_proxy"
            target_slot = "DRAFT"
            action_items = ["归并多份材料", "保留关键证据与风险", "输出正式结论草稿"]
            unknowns = ["资料间可能仍有分歧"]
            evidence_targets = ["共同结论", "关键证据", "剩余风险"]
            draft_summary = "把多源材料整合成适合对外或对上沟通的正式结论。"
        elif any(token in query for token in ("核验", "验证", "可靠", "冲突", "过时")):
            task_type = "fact_check"
            logic_skill = (
                "evidence_judgment" if "冲突" not in query else "conflict_detection"
            )
            expected_tool = "search"
            target_slot = "HYP"
            action_items = ["拆出待核验陈述", "对照来源与时效", "确认冲突后再下结论"]
            unknowns = ["来源可信度可能不一致"]
            evidence_targets = ["来源可信度", "时间有效性", "冲突点"]
            draft_summary = "先核验证据质量，再输出可靠判断。"

        return {
            "task_type": task_type,
            "logic_skill": logic_skill,
            "expected_tool": expected_tool,
            "target_slot": target_slot,
            "plan_summary": f"围绕任务 `{task_type}` 构建 teacher 推理骨架。",
            "plan_action_items": action_items,
            "plan_unknowns": unknowns,
            "evidence_targets": evidence_targets,
            "draft_summary": draft_summary,
            "quality_score": 0.93,
        }
