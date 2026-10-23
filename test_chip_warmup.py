# -*- coding: utf-8 -*-
"""筹码预热窗口测试：总窗口 1800 天 / 显示前累计换手 500%，二选一先到。"""
import unittest

import numpy as np
import pandas as pd

from challenge_scan import chip_warmup_window


class TestChipWarmup(unittest.TestCase):
    def _df(self, n, turnover):
        return pd.DataFrame({
            'date': pd.date_range('2019-01-01', periods=n),
            'turnover': np.full(n, turnover),
        })

    def test_max_days_ok(self):
        # 显示 150 天，窗口最多 1800 天：1650 天预热累计 495% < 500%，但总窗口先到 1800 天
        ws = chip_warmup_window(self._df(1800, 0.003), 150)
        self.assertTrue(ws['ok'])
        self.assertEqual(ws['total_days'], 1800)
        self.assertEqual(ws['start'], 0)
        self.assertAlmostEqual(ws['warmup_turnover'], 4.95, places=4)

    def test_turnover_stops_early(self):
        # 500 天内 4%/天：显示前 125 天累计 500%，初始日早于显示起点
        ws = chip_warmup_window(self._df(500, 0.04), 150)
        self.assertTrue(ws['ok'])
        self.assertEqual(ws['warmup_days'], 125)
        self.assertEqual(ws['total_days'], 275)
        self.assertEqual(ws['start'], 225)

    def test_insufficient(self):
        ws = chip_warmup_window(self._df(500, 0.001), 150)
        self.assertFalse(ws['ok'])
        self.assertEqual(ws['total_days'], 500)

    def test_total_window_cap(self):
        # 3000 天数据、低换手：1800 天总窗口先到，起点固定在 end-1800
        ws = chip_warmup_window(self._df(3000, 0.0001), 150)
        self.assertTrue(ws['ok'])
        self.assertEqual(ws['total_days'], 1800)
        self.assertEqual(ws['warmup_days'], 1650)
        self.assertEqual(ws['start'], 1200)

    def test_missing_turnover(self):
        df = self._df(1800, 0.003).drop(columns=['turnover'])
        self.assertFalse(chip_warmup_window(df, 150)['ok'])


if __name__ == '__main__':
    unittest.main()
