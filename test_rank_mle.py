import unittest

from rank_mle import (
    HUMAN_RANKS,
    _compute_improvements,
    _ordered_unique,
    _score_improvements,
    _top_policy_moves,
)
from sgf_loader import LoadedGame, gtp_to_index


class ImprovementScoringTests(unittest.TestCase):
    def test_improvement_queries_use_preaz_profiles(self):
        game = LoadedGame(
            sgf_path="game.sgf",
            board_size=(19, 19),
            komi=6.5,
            rules="japanese",
            initial_player="B",
            initial_stones=[],
            moves=[("B", "Q16")],
            players={},
        )

        class FakeClient:
            def __init__(self):
                self.profiles = []

            def is_alive(self):
                return True

            def send_query(self, query, cb, _err_cb):
                profile = query.get("overrideSettings", {}).get("humanSLProfile")
                if profile is not None:
                    self.profiles.append(profile)
                    policy = [0.0] * 362
                    policy[gtp_to_index("D4", game.board_size)] = 0.2
                    cb({"humanPolicy": policy})
                else:
                    cb({"moveInfos": [{"move": "Q16", "scoreLead": 0.0}]})

        client = FakeClient()
        _compute_improvements(game, client, {"B": {"rank": "3k"}, "W": {"rank": None}})

        self.assertEqual(client.profiles, ["preaz_3k", "preaz_1k"])

    def test_scores_alternative_that_improves_with_stronger_rank(self):
        game = LoadedGame(
            sgf_path="game.sgf",
            board_size=(19, 19),
            komi=6.5,
            rules="japanese",
            initial_player="B",
            initial_stones=[],
            moves=[("B", "Q16")],
            players={},
        )
        rank = "rank_3k"
        target = HUMAN_RANKS[HUMAN_RANKS.index(rank) + 2]
        pol_rank = [0.0] * 362
        pol_target = [0.0] * 362
        played_idx = gtp_to_index("Q16", game.board_size)
        alt_idx = gtp_to_index("D4", game.board_size)
        pol_rank[played_idx] = 0.30
        pol_target[played_idx] = 0.20
        pol_rank[alt_idx] = 0.05
        pol_target[alt_idx] = 0.25

        scored = _score_improvements(
            game,
            {"B": {"rank": "3k"}, "W": {"rank": None}},
            {0: {rank: pol_rank, target: pol_target}},
            {0: {"Q16": 0.0, "D4": 1.8}},
        )

        moves = scored["B"]["moves"]
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0]["played"], "Q16")
        self.assertEqual(moves[0]["alternative"], "D4")
        self.assertEqual(moves[0]["alternatives"][0]["move"], "D4")
        self.assertEqual(moves[0]["alternatives"][0]["kind"], "most_human_likely")
        self.assertAlmostEqual(moves[0]["played_gain_pp"], -10.0)
        self.assertAlmostEqual(moves[0]["alternatives"][0]["point_gain"], 1.8)
        self.assertAlmostEqual(moves[0]["alternatives"][0]["gain_pp"], 20.0)
        self.assertNotIn("fix_pp", moves[0])
        self.assertNotIn("alternative_p_rank", moves[0])

    def test_ranks_by_played_policy_drop(self):
        game = LoadedGame(
            sgf_path="game.sgf",
            board_size=(19, 19),
            komi=6.5,
            rules="japanese",
            initial_player="B",
            initial_stones=[],
            moves=[("B", "Q16"), ("W", "D4"), ("B", "Q4")],
            players={},
        )
        rank = "rank_3k"
        target = HUMAN_RANKS[HUMAN_RANKS.index(rank) + 2]
        policies = {}
        for move_idx, played, played_rank, played_target, alt, alt_rank, alt_target in (
            (0, "Q16", 0.25, 0.20, "D4", 0.02, 0.20),
            (2, "Q4", 0.30, 0.10, "K10", 0.08, 0.10),
        ):
            pol_rank = [0.0] * 362
            pol_target = [0.0] * 362
            pol_rank[gtp_to_index(played, game.board_size)] = played_rank
            pol_target[gtp_to_index(played, game.board_size)] = played_target
            pol_rank[gtp_to_index(alt, game.board_size)] = alt_rank
            pol_target[gtp_to_index(alt, game.board_size)] = alt_target
            policies[move_idx] = {rank: pol_rank, target: pol_target}

        scored = _score_improvements(
            game,
            {"B": {"rank": "3k"}, "W": {"rank": None}},
            policies,
            {0: {"Q16": 0.0, "D4": 1.5}, 2: {"Q4": 0.0, "K10": 1.5}},
        )

        self.assertEqual([m["move_num"] for m in scored["B"]["moves"]], [3, 1])

    def test_top_policy_moves_are_limited_and_ordered(self):
        policy = [0.0] * 362
        policy[gtp_to_index("D4", (19, 19))] = 0.2
        policy[gtp_to_index("Q16", (19, 19))] = 0.4
        policy[gtp_to_index("K10", (19, 19))] = 0.1

        self.assertEqual(_top_policy_moves(policy, (19, 19), 2), ["Q16", "D4"])
        self.assertEqual(_ordered_unique(["Q16", "D4", "Q16"]), ["Q16", "D4"])

    def test_top_policy_moves_cut_off_below_one_percent(self):
        policy = [0.0] * 362
        policy[gtp_to_index("Q16", (19, 19))] = 0.04
        policy[gtp_to_index("D4", (19, 19))] = 0.011
        policy[gtp_to_index("K10", (19, 19))] = 0.009

        self.assertEqual(_top_policy_moves(policy, (19, 19), 5), ["Q16", "D4"])


if __name__ == "__main__":
    unittest.main()
