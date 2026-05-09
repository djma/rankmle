import unittest

from ogs import parse_ogs_url


class ParseOgsUrlTests(unittest.TestCase):
    def test_canonical_url(self):
        self.assertEqual(parse_ogs_url("https://online-go.com/game/86429302"), 86429302)

    def test_http_scheme(self):
        self.assertEqual(parse_ogs_url("http://online-go.com/game/12345"), 12345)

    def test_no_scheme(self):
        self.assertEqual(parse_ogs_url("online-go.com/game/12345"), 12345)

    def test_www_subdomain(self):
        self.assertEqual(parse_ogs_url("https://www.online-go.com/game/42"), 42)

    def test_view_segment(self):
        self.assertEqual(parse_ogs_url("https://online-go.com/game/view/77"), 77)

    def test_trailing_slash(self):
        self.assertEqual(parse_ogs_url("https://online-go.com/game/86429302/"), 86429302)

    def test_query_string(self):
        self.assertEqual(
            parse_ogs_url("https://online-go.com/game/86429302?move=10"), 86429302
        )

    def test_surrounding_whitespace(self):
        self.assertEqual(parse_ogs_url("  https://online-go.com/game/123  "), 123)

    def test_bare_id(self):
        self.assertEqual(parse_ogs_url("86429302"), 86429302)

    def test_non_ogs_url(self):
        self.assertIsNone(parse_ogs_url("https://example.com/game/123"))

    def test_other_ogs_path(self):
        self.assertIsNone(parse_ogs_url("https://online-go.com/user/view/123"))

    def test_sgf_text_is_not_an_url(self):
        self.assertIsNone(parse_ogs_url("(;GM[1]FF[4]SZ[19];B[pd];W[dd])"))

    def test_empty_string(self):
        self.assertIsNone(parse_ogs_url(""))


if __name__ == "__main__":
    unittest.main()
