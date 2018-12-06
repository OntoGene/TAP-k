#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


import io
import re
import unittest
import itertools as it
from pathlib import Path

import tapk


TESTROOT = Path(__file__).parent


class ScoreTest(unittest.TestCase):
    def test_scores(self):
        for ref in (TESTROOT/'ref-scores').iterdir():
            self._score(ref)

    def _score(self, ref_path):
        case, *params = ref_path.stem.split('-')
        params = {p[0]: self._paramtype(p[1:]) for p in params}
        src_path = TESTROOT/'retlists'/'{}.tsv'.format(case)
        with self.subTest(case=case, **params):
            with ref_path.open() as ref, src_path.open() as src:
                scores = io.StringIO()
                tapk.run([src], output=scores, show_query_wise_result=True,
                         **params)
                scores.seek(0)
                self._compare(list(scores), list(ref))

    def _compare(self, scores, ref):
        self._compare_headers(scores[:2], ref[:2])
        self.assertEqual(scores[4:-1], ref[4:])  # -1: strip trailing blank line

    def _compare_headers(self, scores, ref):
        # Extract all numbers and convert to float.
        scores, ref = (map(float, re.findall(r'[\d.]+', ''.join(l)))
                       for l in (scores, ref))
        for s, r in it.zip_longest(scores, ref):
            self.assertAlmostEqual(s, r)

    @staticmethod
    def _paramtype(value):
        if not value:  # boolean flags have no value
            return True
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


if __name__ == '__main__':
    unittest.main()
