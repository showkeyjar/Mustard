from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("")
    payload = {
        "semantic_text": f"已收到屏幕截图 {image_path.name or 'unknown'}，可进一步接入外部视觉模型生成更具体的界面摘要。",
        "tags": ["screen_context", "image"],
        "confidence": "medium",
        "modality_hints": ["image", "desktop"],
        "suggested_tool": "bigmodel_proxy",
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
