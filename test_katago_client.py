import unittest

from katago_client import build_score_query


class ScoreQueryTests(unittest.TestCase):
    def test_can_force_root_moves(self):
        query = build_score_query(
            [("B", "Q16")],
            allowed_player="W",
            allowed_moves=["D4", "Q4"],
            max_visits=32,
        )

        self.assertEqual(query["maxVisits"], 32)
        self.assertEqual(
            query["allowMoves"],
            [{"player": "W", "moves": ["D4", "Q4"], "untilDepth": 1}],
        )


if __name__ == "__main__":
    unittest.main()
