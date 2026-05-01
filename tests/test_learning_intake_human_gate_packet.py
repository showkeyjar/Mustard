import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_learning_intake_human_gate_packet import build_learning_intake_human_gate_packet


class LearningIntakeHumanGatePacketTests(unittest.TestCase):
    def test_build_learning_intake_human_gate_packet_splits_approve_and_defer(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "artifacts").mkdir(parents=True, exist_ok=True)
            (root / "backlog" / "opportunities").mkdir(parents=True, exist_ok=True)

            (root / "artifacts" / "learning_intake_review_queue_latest.json").write_text(
                json.dumps(
                    {
                        "queue": [
                            {
                                "user_input": "样本 A",
                                "source_type": "learning_intake:learning_focus_stress",
                                "sample_id": "a",
                                "recommended_status": "accept",
                                "reasons": ["still failing"],
                            },
                            {
                                "user_input": "样本 B",
                                "source_type": "learning_intake:learning_focus_stress",
                                "sample_id": "b",
                                "recommended_status": "edit",
                                "reasons": ["baseline only fail"],
                            },
                            {
                                "user_input": "样本 C",
                                "source_type": "learning_intake:attention_gap",
                                "sample_id": "",
                                "recommended_status": "accept",
                                "reasons": ["top gap"],
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "artifacts" / "learning_intake_suggested_shadow_delta_latest.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "added_count": 2,
                            "deduped_import_count": 1,
                        },
                        "added_rows": [
                            {"user_input": "样本 A"},
                            {"user_input": "样本 B"},
                        ],
                        "deduped_import_rows": [
                            {"prompt": "样本 C"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            packet = build_learning_intake_human_gate_packet(root)

            self.assertEqual(packet["summary"]["recommend_approve_count"], 1)
            self.assertEqual(packet["summary"]["recommend_edit_count"], 1)
            self.assertEqual(packet["summary"]["recommend_defer_count"], 1)
            self.assertTrue((root / "artifacts" / "learning_intake_human_gate_packet_latest.json").exists())
            self.assertTrue((root / "backlog" / "opportunities" / "learning_intake_human_gate_packet.md").exists())


if __name__ == "__main__":
    unittest.main()
