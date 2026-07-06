"""CARM — Context-Aware Routing Model.

Usage::

    from carm import CARMRouter

    router = CARMRouter()          # auto-detect embedding availability
    result = router.route("3加5等于多少")
    print(result.tool_name, result.result)
"""

from carm.router import CARMRouter, RouteResult
from carm.intent import IntentCategory

__all__ = ["CARMRouter", "RouteResult", "IntentCategory"]
__version__ = "0.9.0"
