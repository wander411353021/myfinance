# -*- coding: utf-8 -*-
"""B2-early 强势分支测试（2026-10-21）。"""
import unittest

import numpy as np

from challenge_scan import MIN_I, scan_series, v3b_signal_b2


class TestB2Early(unittest.TestCase):
    def setUp(self):
        n, self.i = 500, 450
        self.cl = np.full(n, 11.0)
        self.cl[390] = 9.5          # 回踩 < cA*0.98
        self.cl[self.i] = 13.0      # gap = 30%
        self.gt = np.full(n, 10.0)  # 红压线
        self.cA = np.full(n, 10.0)  # 主成本线 A
        self.vol = np.full(n, 100.0)
        self.vol[self.i] = 200.0

    def test_gap_branch(self):
        ok, _ = v3b_signal_b2(self.cl, self.gt, self.cA, self.vol, self.i)
        self.assertFalse(ok)  # 正式 B2 拒绝 gap=30%
        ok, _ = v3b_signal_b2(self.cl, self.gt, self.cA, self.vol, self.i,
                              strong_gap=True)
        self.assertTrue(ok)   # B2-early 接受 gap>20%

        self.cl[self.i] = 11.5  # gap=15%
        ok, _ = v3b_signal_b2(self.cl, self.gt, self.cA, self.vol, self.i)
        self.assertTrue(ok)   # 正式 B2 接受 5~20%
        ok, _ = v3b_signal_b2(self.cl, self.gt, self.cA, self.vol, self.i,
                              strong_gap=True)
        self.assertFalse(ok)  # B2-early 只接受 >20%

    def test_scan_series_type(self):
        hits = scan_series(self.cl, self.gt, self.gt, None, self.vol,
                           start_i=MIN_I, end_i=self.i + 1,
                           types=('v3b_b2_early',), mode='days', cool=0,
                           cA=self.cA)
        self.assertIn(('v3b_b2_early', self.i), hits)

    def test_rearm_days(self):
        n = 520
        cl = np.full(n, 11.0)
        cl[440] = 9.5          # 回踩
        cl[450] = 13.0         # 第一信号
        cl[480] = 13.0         # 30 日后第二信号
        gt = np.full(n, 10.0)
        cA = np.full(n, 10.0)
        c35 = np.full(n, 10.0)
        vol = np.full(n, 100.0)
        vol[450] = 200.0
        vol[480] = 200.0
        hits = scan_series(cl, gt, gt, c35, vol, start_i=MIN_I, end_i=481,
                           types=('v3b_b2_early',), mode='structure', cool=0,
                           cA=cA, rearm_days=30)
        self.assertEqual([i for _, i in hits], [450, 480])
        hits_no = scan_series(cl, gt, gt, c35, vol, start_i=MIN_I, end_i=481,
                              types=('v3b_b2_early',), mode='structure', cool=0,
                              cA=cA)
        self.assertEqual([i for _, i in hits_no], [450])


if __name__ == '__main__':
    unittest.main()
