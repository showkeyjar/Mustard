import unittest

from carm.memory import MemoryBoard, MemorySlot
from carm.state import AgentState


class MemoryBoardTests(unittest.TestCase):
    def test_goal_rewrite_is_stable(self) -> None:
        board = MemoryBoard(max_slots=4)
        board.write(MemorySlot("GOAL", "first", 0.7, "test"))
        board.write(MemorySlot("GOAL", "second", 0.8, "test"))

        goal = board.latest("GOAL")
        self.assertIsNotNone(goal)
        self.assertEqual(goal.content, "second")
        self.assertEqual(len(board.read("GOAL")), 1)

    def test_write_from_state_uses_candidate(self) -> None:
        board = MemoryBoard()
        state = AgentState(hidden={"candidate": "draft text"})
        board.write_from_state(state, "DRAFT", "unit")

        draft = board.latest("DRAFT")
        self.assertIsNotNone(draft)
        payload = board.parse_content(draft)
        self.assertEqual(payload.get("kind"), "draft")
        self.assertEqual(payload.get("summary"), "draft text")


if __name__ == "__main__":
    unittest.main()
