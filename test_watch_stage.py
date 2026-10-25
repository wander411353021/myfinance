# -*- coding: utf-8 -*-
"""watch_stage 第一版：阶段与因果性测试。"""
import unittest
import numpy as np

from watch_stage import compute_watch_stages


def _synthetic():
    n = 100
    B = np.full(n, 10.0)
    A = np.full(n, 10.5)
    closes = np.full(n, 9.5)
    closes[60:70] = 10.2      # 小碎步上 B，仍在 A 下方
    closes[70:] = 11.0        # 放量上 A
    highs = closes * 1.01
    lows = closes * 0.99
    vol = np.full(n, 100.0)
    vol[70:] = 300.0          # 5/20 量比达到阈值
    turn = np.full(n, 0.01)
    turn[70:] = 0.03
    return closes, highs, lows, B, A, vol, turn


class TestWatchStage(unittest.TestCase):
    def test_stage_sequence(self):
        closes, highs, lows, B, A, vol, turn = _synthetic()
        st = compute_watch_stages(closes, highs, lows, B, A,
                                  volume=vol, turnover=turn)
        self.assertEqual(st[60], 1)
        self.assertEqual(st[71], 2)

    def test_causal_truncation(self):
        closes, highs, lows, B, A, vol, turn = _synthetic()
        full = compute_watch_stages(closes, highs, lows, B, A,
                                    volume=vol, turnover=turn)
        part = compute_watch_stages(closes[:71], highs[:71], lows[:71],
                                    B[:71], A[:71], volume=vol[:71],
                                    turnover=turn[:71])
        np.testing.assert_array_equal(part, full[:71])

    def test_s4_requires_s3(self):
        # 没有 S3（B 未转正）时，S2 后的缩量回踩不能直接标成 S4
        n = 100
        B = np.full(n, 10.0)
        A = np.full(n, 10.2)
        closes = np.full(n, 9.5)
        closes[60:70] = 10.1
        closes[70:75] = 10.5
        closes[75:] = 10.25
        highs = closes * 1.005
        lows = closes * 0.995
        vol = np.full(n, 100.0)
        vol[70:] = 300.0
        vol[75:] = 50.0
        turn = np.full(n, 0.01)
        turn[70:] = 0.03
        turn[75:] = 0.005
        st = compute_watch_stages(closes, highs, lows, B, A,
                                  volume=vol, turnover=turn)
        self.assertNotIn(4, st.tolist())
        self.assertIn(2, st.tolist())


if __name__ == '__main__':
    unittest.main()
