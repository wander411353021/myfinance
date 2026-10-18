# -*- coding: utf-8 -*-
"""Ehlers 平滑周期可调回归（2026-10-18）。

覆盖：
1. 不同 period 必须产生不同平滑结果；
2. 平滑严格因果（截断前缀重算 == 全量前缀）；
3. compute / scan_recent / run_segmentation / plot 均暴露 ehlers_period 参数。
"""
import inspect
import unittest

import numpy as np

from challenge_scan import ehlers_smooth, compute, scan_recent
from price_segmenter_v10 import run_segmentation, plot_price_segmentation_v10


class TestEhlersPeriod(unittest.TestCase):
    def setUp(self):
        x = np.linspace(10.0, 20.0, 200) + np.sin(np.arange(200) / 5.0)
        self.x = x

    def test_period_changes_smoothing(self):
        a = ehlers_smooth(self.x, 5)
        b = ehlers_smooth(self.x, 45)
        self.assertFalse(np.allclose(a, b, equal_nan=True))

    def test_causal_truncation(self):
        full = ehlers_smooth(self.x, 20)
        part = ehlers_smooth(self.x[:120], 20)
        np.testing.assert_allclose(full[:120], part, equal_nan=True)

    def test_signatures_expose_period(self):
        for fn in (compute, scan_recent, run_segmentation, plot_price_segmentation_v10):
            params = inspect.signature(fn).parameters
            self.assertIn('ehlers_period', params, fn.__name__)
            if fn in (run_segmentation, plot_price_segmentation_v10):
                self.assertIn('cost2_pct', params, fn.__name__)
                self.assertIn('cost2_ehlers', params, fn.__name__)


if __name__ == '__main__':
    unittest.main()
