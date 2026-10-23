# -*- coding: utf-8 -*-
"""筹码预热状态测试（1800 天 / 累计换手 400%，只提示不阻断）。"""
import unittest

import numpy as np
import pandas as pd

from challenge_scan import chip_warmup_status


class TestChipWarmup(unittest.TestCase):
    def _df(self, n, turnover):
        return pd.DataFrame({
            'date': pd.date_range('2019-01-01', periods=n),
            'turnover': np.full(n, turnover),
        })

    def test_enough(self):
        ws = chip_warmup_status(self._df(1800, 0.003))  # 540%
        self.assertTrue(ws['ok'])
        self.assertEqual(ws['days'], 1800)
        self.assertGreaterEqual(ws['turnover'], 4.0)

    def test_insufficient_turnover(self):
        ws = chip_warmup_status(self._df(1800, 0.001))  # 180%
        self.assertFalse(ws['ok'])

    def test_short_history(self):
        ws = chip_warmup_status(self._df(500, 0.02))
        self.assertFalse(ws['ok'])
        self.assertEqual(ws['days'], 500)

    def test_missing_turnover(self):
        df = self._df(1800, 0.003).drop(columns=['turnover'])
        self.assertFalse(chip_warmup_status(df)['ok'])


if __name__ == '__main__':
    unittest.main()
