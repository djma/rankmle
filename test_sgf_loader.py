import unittest

from sgf_loader import gtp_to_index, gtp_to_sgf_coord, index_to_gtp, load_sgf_bytes


class SgfEncodingTests(unittest.TestCase):
    def test_missing_charset_prefers_utf8_when_valid(self):
        game = load_sgf_bytes(
            "(;GM[1]FF[4]SZ[19]PB[axing11]BR[3级]PW[djma0]WR[3级])".encode(
                "utf-8"
            )
        )
        root = game.get_root()

        self.assertEqual(root.get("BR"), "3级")
        self.assertEqual(root.get("WR"), "3级")

    def test_declared_charset_is_respected(self):
        game = load_sgf_bytes(
            "(;GM[1]FF[4]CA[ISO-8859-1]SZ[19]PB[Andr\xe9])".encode(
                "latin-1"
            )
        )

        self.assertEqual(game.get_root().get("PB"), "Andr\xe9")


class CoordinateTests(unittest.TestCase):
    def test_gtp_policy_index_roundtrip(self):
        board_size = (19, 19)
        for gtp in ("A1", "T19", "Q16", "pass"):
            self.assertEqual(index_to_gtp(gtp_to_index(gtp, board_size), board_size), gtp)

    def test_gtp_to_sgf_coord(self):
        self.assertEqual(gtp_to_sgf_coord("A1"), (0, 0))
        self.assertEqual(gtp_to_sgf_coord("T19"), (18, 18))
        self.assertIsNone(gtp_to_sgf_coord("pass"))


if __name__ == "__main__":
    unittest.main()
