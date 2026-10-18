# -*- coding: utf-8 -*-
"""布局修复回归：信号画像底部留白按窗口内实际画像数计算。

2026-10-18 修复背景：300437 全历史 v3b 信号 8 个、tail=200 窗口内仅 2 个，
旧逻辑用 len(v3b_sig_dates) 预留 0.45 底部空白，把主图压到上半部分。
"""
import unittest

from price_segmenter_v10 import _profile_rect_bottom


class TestProfileRectBottom(unittest.TestCase):
    def test_typical_window(self):
        # 0 个画像 -> 仅 0.05 基础留白
        self.assertAlmostEqual(_profile_rect_bottom(0), 0.05)
        # 300437 修复场景：窗口内 2 个 -> 0.15（旧逻辑 8 个会算成 0.45）
        self.assertAlmostEqual(_profile_rect_bottom(2), 0.15)
        # 全历史 8 个不应再被直接传入；函数本身仍保持线性
        self.assertAlmostEqual(_profile_rect_bottom(8), 0.45)

    def test_negative_is_clamped(self):
        self.assertAlmostEqual(_profile_rect_bottom(-3), 0.05)


if __name__ == '__main__':
    unittest.main()
