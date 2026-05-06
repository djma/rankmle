import asyncio
import tempfile
import unittest
from pathlib import Path

from server import JOBS, JOBS_LOCK, Job, _annotated_sgf_bytes, start_improvements
from sgfmill import sgf


class AnnotatedSgfTests(unittest.TestCase):
    def test_adds_prediction_comment_before_first_move(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sgf_path = Path(tmp_dir) / "game.sgf"
            sgf_path.write_bytes(b"(;GM[1]FF[4]SZ[19]C[Original note];B[pd];W[dd])")
            job = Job(
                job_id="job",
                sgf_path=str(sgf_path),
                sgf_sha="abc123",
                status="done",
                result={
                    "players": {"B": {"name": "Black"}, "W": {"name": "White"}},
                    "prediction": {"B": {"rank": "3k"}, "W": {"rank": "1d"}},
                },
            )

            annotated = sgf.Sgf_game.from_bytes(_annotated_sgf_bytes(job))
            root = annotated.get_root()

        self.assertEqual(
            root.get("C"),
            "Predicted ranks:\n"
            "Black (Black): 3k\n"
            "White (White): 1d\n\n"
            "Original note",
        )
        self.assertEqual(annotated.get_main_sequence()[1].get_move()[0], "b")

    def test_adds_improvement_branch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sgf_path = Path(tmp_dir) / "game.sgf"
            sgf_path.write_bytes(b"(;GM[1]FF[4]SZ[19];B[pd];W[dd])")
            job = Job(
                job_id="job",
                sgf_path=str(sgf_path),
                sgf_sha="abc123",
                status="done",
                result={
                    "players": {"B": {"name": "Black"}, "W": {"name": "White"}},
                    "prediction": {"B": {"rank": "3k"}, "W": {"rank": "1d"}},
                    "improvements": {
                        "players": {
                            "B": {
                                "moves": [
                                    {
                                        "move_num": 1,
                                        "player": "B",
                                        "played": "Q16",
                                        "rank": "3k",
                                        "target_rank": "1k",
                                        "played_gain_pp": -2.0,
                                        "played_p_rank": 20.0,
                                        "played_p_target": 18.0,
                                        "alternatives": [
                                            {
                                                "move": "D4",
                                                "kind": "most_human_likely",
                                                "point_gain": 1.5,
                                                "p_rank": 5.0,
                                                "p_target": 15.3,
                                                "gain_pp": 10.3,
                                            },
                                            {
                                                "move": "K10",
                                                "kind": "biggest_human_gain",
                                                "point_gain": 1.2,
                                                "p_rank": 1.0,
                                                "p_target": 12.0,
                                                "gain_pp": 11.0,
                                            },
                                        ],
                                    }
                                ]
                            },
                            "W": {"moves": []},
                        }
                    },
                },
            )

            annotated_bytes = _annotated_sgf_bytes(job)
            annotated = sgf.Sgf_game.from_bytes(annotated_bytes)

        self.assertEqual(annotated.get_main_sequence()[1].get_move(), ("b", (15, 15)))
        self.assertIn(b"(;B[dp]", annotated_bytes)
        self.assertIn(b"(;B[jj]", annotated_bytes)
        self.assertIn(b"Rank MLE options before move 1", annotated_bytes)
        self.assertIn(b"Suggested: D4", annotated_bytes)
        self.assertIn(b"Suggested: K10", annotated_bytes)


class ImprovementJobTests(unittest.TestCase):
    def tearDown(self):
        with JOBS_LOCK:
            JOBS.clear()

    def test_start_improvements_returns_cached_results(self):
        job = Job(
            job_id="job",
            sgf_path="/tmp/game.sgf",
            sgf_sha="abc123",
            status="done",
            result={
                "players": {"B": {}, "W": {}},
                "prediction": {"B": {"rank": "3k"}, "W": {"rank": "1d"}},
                "improvements": {"players": {"B": {"moves": []}, "W": {"moves": []}}},
            },
        )
        with JOBS_LOCK:
            JOBS[job.job_id] = job

        response = asyncio.run(start_improvements(job.job_id))

        self.assertEqual(response["improvement_status"], "done")
        self.assertIs(response["result"], job.result)
        self.assertEqual(job.improvement_status, "done")


if __name__ == "__main__":
    unittest.main()
