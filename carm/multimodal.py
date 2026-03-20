from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from carm.desktop import DesktopDigest


@dataclass
class MultimodalSignal:
    source: str
    semantic_text: str
    tags: list[str] = field(default_factory=list)
    confidence: str = "low"
    modality_hints: list[str] = field(default_factory=list)
    suggested_tool: str = ""


class MultimodalAdapter:
    """Compress external or desktop observations into MVP-friendly semantic signals."""

    def from_desktop_digest(self, digest: DesktopDigest) -> MultimodalSignal:
        tags = list(digest.semantic_tags) or self._infer_tags_from_text(digest.semantic_summary, digest.top_apps)
        suggested_tool = ""
        if "coding" in tags or "research" in tags:
            suggested_tool = "search"
        elif "spreadsheet" in tags:
            suggested_tool = "calculator"
        return MultimodalSignal(
            source="desktop",
            semantic_text=digest.semantic_summary,
            tags=tags,
            confidence=digest.semantic_confidence,
            modality_hints=list(digest.modality_hints),
            suggested_tool=suggested_tool,
        )

    def _infer_tags_from_text(self, semantic_text: str, top_apps: list[str]) -> list[str]:
        lowered = f"{semantic_text} {' '.join(top_apps)}".lower()
        tags: list[str] = []
        if any(token in lowered for token in ("vs code", "code", "pycharm", "cursor", "代码")):
            tags.append("coding")
        if any(token in lowered for token in ("查资料", "检索", "浏览网页", "chrome", "edge", "firefox")):
            tags.append("research")
        if any(token in lowered for token in ("表格", "excel", "sheet")):
            tags.append("spreadsheet")
        if any(token in lowered for token in ("文档", "word", "notion", "obsidian")):
            tags.append("document")
        if any(token in lowered for token in ("聊天", "协作", "wechat", "feishu", "slack", "discord")):
            tags.append("communication")
        return tags or ["desktop_task"]


class ScreenObservationAdapter:
    def __init__(
        self,
        *,
        enabled: bool = False,
        capture_dir: str | Path = "data/desktop/screens",
        describe_command: list[str] | None = None,
        capture_fn: Callable[[], Path | None] | None = None,
        describe_fn: Callable[[Path, DesktopDigest], MultimodalSignal] | None = None,
    ) -> None:
        self.enabled = enabled
        self.capture_dir = Path(capture_dir)
        self.describe_command = list(describe_command or [])
        self.capture_fn = capture_fn
        self.describe_fn = describe_fn
        self._ocr_engine: Any | None = None

    def enrich_digest(self, digest: DesktopDigest) -> DesktopDigest:
        if not self.enabled:
            return digest
        image_path = self._capture_screen()
        if image_path is None:
            return digest
        signal = self._describe_image(image_path, digest)
        merged_tags = sorted(set(digest.semantic_tags + signal.tags))
        merged_hints = sorted(set(digest.modality_hints + signal.modality_hints))
        merged_multimodal_tags = sorted(set(digest.multimodal_tags + signal.tags))
        confidence = self._merge_confidence(digest.semantic_confidence, signal.confidence)
        return replace(
            digest,
            semantic_tags=merged_tags,
            semantic_confidence=confidence,
            modality_hints=merged_hints,
            multimodal_summary=signal.semantic_text,
            multimodal_tags=merged_multimodal_tags,
            multimodal_artifact_path=image_path.as_posix(),
        )

    def _capture_screen(self) -> Path | None:
        if self.capture_fn is not None:
            return self.capture_fn()
        try:
            from PIL import ImageGrab
        except Exception:
            return None
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.capture_dir / f"screen-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.png"
        try:
            image = ImageGrab.grab(all_screens=True)
            image.thumbnail((1440, 900))
            image.save(image_path, format="PNG")
            return image_path
        except Exception:
            return None

    def _describe_image(self, image_path: Path, digest: DesktopDigest) -> MultimodalSignal:
        if self.describe_fn is not None:
            return self.describe_fn(image_path, digest)
        if self.describe_command:
            try:
                result = subprocess.run(
                    [*self.describe_command, str(image_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=20,
                )
                if result.returncode == 0 and result.stdout.strip():
                    payload = json.loads(result.stdout)
                    if isinstance(payload, dict):
                        return MultimodalSignal(
                            source="screen",
                            semantic_text=str(payload.get("semantic_text", "") or "屏幕视觉工具已返回结果。"),
                            tags=[str(item) for item in payload.get("tags", []) if str(item).strip()],
                            confidence=str(payload.get("confidence", "medium") or "medium"),
                            modality_hints=[str(item) for item in payload.get("modality_hints", []) if str(item).strip()],
                            suggested_tool=str(payload.get("suggested_tool", "") or ""),
                        )
            except Exception:
                pass
        ocr_signal = self._describe_with_rapidocr(image_path, digest)
        if ocr_signal is not None:
            return ocr_signal
        return MultimodalSignal(
            source="screen",
            semantic_text=(
                "已捕获当前屏幕快照，可结合窗口上下文进一步理解视觉内容。"
                f" 当前桌面语义={digest.semantic_summary}"
            ),
            tags=["screen_context"],
            confidence="medium",
            modality_hints=["image", "desktop"],
            suggested_tool="bigmodel_proxy" if digest.top_apps else "",
        )

    def _describe_with_rapidocr(self, image_path: Path, digest: DesktopDigest) -> MultimodalSignal | None:
        try:
            if self._ocr_engine is None:
                from rapidocr_onnxruntime import RapidOCR

                self._ocr_engine = RapidOCR()
            result, _ = self._ocr_engine(str(image_path))
        except Exception:
            return None
        if not result:
            return None

        texts: list[str] = []
        confidences: list[float] = []
        for item in result:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            text = str(item[1]).strip()
            try:
                confidence = float(item[2])
            except Exception:
                confidence = 0.0
            if len(text) < 2 or confidence < 0.45:
                continue
            if text not in texts:
                texts.append(text)
                confidences.append(confidence)
            if len(texts) >= 8:
                break
        if not texts:
            return None

        joined_text = "；".join(texts[:5])
        inferred_tags = MultimodalAdapter()._infer_tags_from_text(
            f"{joined_text} {digest.semantic_summary}",
            digest.top_apps,
        )
        suggested_tool = ""
        if "coding" in inferred_tags or "research" in inferred_tags or "document" in inferred_tags:
            suggested_tool = "search"
        elif "spreadsheet" in inferred_tags:
            suggested_tool = "calculator"

        avg_conf = sum(confidences) / len(confidences)
        confidence = "high" if avg_conf >= 0.75 and len(texts) >= 3 else "medium"
        return MultimodalSignal(
            source="screen",
            semantic_text=(
                f"屏幕识别到文字：{joined_text}。"
                f" 结合当前桌面上下文，可能正在{digest.semantic_summary.strip('。')}。"
            ),
            tags=sorted(set(inferred_tags + ["ocr", "screen_context"])),
            confidence=confidence,
            modality_hints=["image", "text", "desktop"],
            suggested_tool=suggested_tool,
        )

    def _merge_confidence(self, left: str, right: str) -> str:
        order = {"low": 0, "medium": 1, "high": 2}
        return left if order.get(left, 0) >= order.get(right, 0) else right
