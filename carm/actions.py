from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class Action(StrEnum):
    THINK = "THINK"
    READ_MEM = "READ_MEM"
    WRITE_MEM = "WRITE_MEM"
    CALL_TOOL = "CALL_TOOL"
    CALL_BIGMODEL = "CALL_BIGMODEL"
    VERIFY = "VERIFY"
    ROLLBACK = "ROLLBACK"
    ANSWER = "ANSWER"
