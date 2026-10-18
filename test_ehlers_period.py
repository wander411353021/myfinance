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
import pandas as pd

import algo_config as cfg
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

    def test_signal_defaults_to_main_cost_b(self):
        # 2026-10-18 定版：V3/V3b/early 默认使用主力成本线 B（COST2_*）
        self.assertEqual(cfg.SIGNAL_COST_PCT, cfg.COST2_PCT)
        self.assertEqual(cfg.SIGNAL_COST_EHLERS, cfg.COST2_EHLERS)

        n = 600
        rng = np.random.default_rng(42)
        close = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        df = pd.DataFrame({
            'date': pd.bdate_range('2020-01-01', periods=n),
            'open': close * 1.001,
            'high': close * 1.01,
            'low': close * 0.99,
            'close': close,
            'volume': rng.integers(10000, 50000, n).astype(float),
            'turnover': np.full(n, 0.02),
        })
        default = compute(df)[4]
        b = compute(df, cost_pct=cfg.COST2_PCT, ehlers_period=cfg.COST2_EHLERS)[4]
        a = compute(df, cost_pct=cfg.COST_PCT, ehlers_period=cfg.COST_EHLERS)[4]
        np.testing.assert_allclose(default, b, equal_nan=True)
        self.assertFalse(np.allclose(default, a, equal_nan=True))


if __name__ == '__main__':
    unittest.main()
