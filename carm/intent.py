"""Intent category taxonomy for CARM.

CARM's signal detection layer produces IntentCategory values, not tool names.
The mapping from IntentCategory → actual tool name is done at runtime by
ToolManager.find_by_capability(), which checks the registered tools'
capability_tags.

This decoupling allows users to register arbitrary tools without modifying
CARM's core routing logic.
"""

from __future__ import annotations

from enum import Enum


class IntentCategory(str, Enum):
    """Fixed taxonomy of intent categories that CARM can detect.

    Tools declare which categories they handle via capability_tags.
    Multiple tools can handle the same category; the first registered
    tool for a category is the default.
    """

    CALC = "calc"
    CODE = "code"
    SEARCH = "search"
    CONSULT = "consult"
    MULTI_INTENT = "multi_intent"
    MULTI_STEP = "multi_step"

    @property
    def is_composite(self) -> bool:
        """Whether this category represents a composite intent that
        requires multi-step execution (not a single tool call)."""
        return self in (IntentCategory.MULTI_INTENT, IntentCategory.MULTI_STEP)


# Default mapping: IntentCategory → conventional tool name
# Used for backward compatibility when no dynamic registration is done.
DEFAULT_TOOL_MAP: dict[IntentCategory, str] = {
    IntentCategory.CALC: "calculator",
    IntentCategory.CODE: "code_executor",
    IntentCategory.SEARCH: "search",
    IntentCategory.CONSULT: "bigmodel_proxy",
    IntentCategory.MULTI_INTENT: "multi_intent",
    IntentCategory.MULTI_STEP: "multi_step",
}
