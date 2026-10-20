# -*- coding: utf-8 -*-
"""Grid Target P0 回归测试（2026-10-20）：
1) max_dev 由 levels[-1] + 1% 自动推导；
2) 连续 2 日超偏离才置空；
3) 置空后回到 max_dev - 2% 才恢复（滞回）；
4) 当日收盘只影响下一日目标（无未来函数）。
"""
import unittest

import numpy as np

from panic_reversal import compute_grid_target_price


def _run(closes, levels=(-0.09, -0.06, -0.03, 0.00, 0.03, 0.06, 0.09, 0.12),
         min_level=4, **kwargs):
    closes = np.asarray(closes, dtype=float)
    base = np.full(len(closes), 10.0, dtype=float)
    return compute_grid_target_price(closes, base, base, levels=levels,
                                     min_level=min_level, **kwargs)


class TestGridTargetP0(unittest.TestCase):
    def test_max_dev_auto_follows_top_level(self):
        # 顶档 +5% -> max_dev = 6%（若仍硬编码 0.13 则不会置空）
        closes = [10.0] * 5 + [10.55, 10.0, 10.65, 10.65, 10.0, 10.0, 10.0]
        target, _ = _run(closes, levels=(0.0, 0.05), min_level=0)
        # t=7 首日 6.5% -> 只计数；t=8 第二日 -> 次日(t=9)置空
        self.assertTrue(np.isfinite(target[8]))
        self.assertTrue(np.isnan(target[9]))
        # t=9 回到 0% <= 6%-2% -> 次日恢复
        self.assertTrue(np.isfinite(target[10]))

    def test_two_day_confirm_and_hysteresis(self):
        closes = [10.0] * 6 + [11.4, 11.4, 11.2, 10.5, 10.5, 10.5]
        target, lvl = _run(closes)
        # 单日 14% 只计数，第二日才置空
        self.assertTrue(np.isfinite(target[7]))
        self.assertTrue(np.isnan(target[8]))
        # 12% 仍在滞回区（> 13%-2%=11%），保持置空
        self.assertTrue(np.isnan(target[9]))
        # 5% 恢复到正常档位
        self.assertTrue(np.isfinite(target[10]))
        self.assertNotEqual(lvl[10], -1)

    def test_same_day_close_does_not_change_target(self):
        # 改变最后一天收盘价，不应改变最后一天或更早的目标价
        a = [10.0] * 6 + [10.2, 10.1]
        b = list(a)
        b[-1] = 11.0
        ta, _ = _run(a)
        tb, _ = _run(b)
        np.testing.assert_allclose(ta[:-1], tb[:-1], equal_nan=True)
        np.testing.assert_allclose(ta, tb, equal_nan=True)

    def test_truncation_consistency(self):
        closes = [10.0] * 6 + [10.2, 10.1, 10.0, 10.0]
        full, _ = _run(closes)
        for end in range(2, len(closes) + 1):
            part, _ = _run(closes[:end])
            np.testing.assert_allclose(part, full[:end], equal_nan=True)


if __name__ == '__main__':
    unittest.main()
