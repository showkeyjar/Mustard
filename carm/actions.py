from __future__ import annotations

from enum import StrEnum


class Action(StrEnum):
    THINK = "THINK"
    READ_MEM = "READ_MEM"
    WRITE_MEM = "WRITE_MEM"
    CALL_TOOL = "CALL_TOOL"
    CALL_BIGMODEL = "CALL_BIGMODEL"
    VERIFY = "VERIFY"
    ROLLBACK = "ROLLBACK"
    ANSWER = "ANSWER"
