import unittest

from solution import slugify


class SlugifyTest(unittest.TestCase):
    def test_words(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")

    def test_runs_collapse(self):
        self.assertEqual(slugify("a  --  b"), "a-b")

    def test_edges_are_stripped(self):
        self.assertEqual(slugify("  --Rig 2026--  "), "rig-2026")

    def test_nothing_left(self):
        self.assertEqual(slugify("!!!"), "")


if __name__ == "__main__":
    unittest.main()
