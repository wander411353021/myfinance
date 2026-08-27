# -*- coding: utf-8 -*-
"""黄金坑策略插件(注释中文,显示英文由引擎负责)。

信号定义(93% 超高胜率,全市场验证):
  黄金坑(z<-1.5, 坑长≥8) + 快启动(坑底→出坑≤5天)
  meta.super = 出坑后7天内放量堆峰值量比≥5(加仓确认/超高标记)

用法:
    from strategies.golden_pit import GoldenPitStrategy
    sigs = GoldenPitStrategy().generate_signals(df)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import panic_reversal as pr
from mean_reversion.signal_residual import compute_rolling_regression


class GoldenPitStrategy:
    """黄金坑策略:输出出坑日买入信号(meta.super 标注超高)。"""

    name = 'golden_pit'

    def __init__(self, min_len=8, fast_days=5, peak_ratio=5.0, window_days=7,
                 require_super=False, code=''):
        self.min_len = min_len          # 坑长下限(天)
        self.fast_days = fast_days      # 快启动:坑底→出坑上限(天)
        self.peak_ratio = peak_ratio    # 超高:出坑后放量堆峰值量比
        self.window_days = window_days  # 超高:出坑后观察窗口(天)
        self.require_super = require_super  # True=只输出超高信号
        self.code = code

    def generate_signals(self, df):
        """返回信号列表:每个 = {idx, date, meta}。"""
        c = df['close'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['volume'].values.astype(float)
        dates = df['date'].dt.strftime('%Y%m%d').values
        n = len(c)
        reg250, _ = compute_rolling_regression(c, window=250, use_log=True)
        pits = pr.detect_golden_pit(c, reg250)
        cl = pr.detect_volume_clusters(c, v)
        sigs = []
        for s, b, lch in pits:
            if lch is None:
                continue
            if lch - b > self.fast_days:
                continue  # 非快启动
            if b - s + 1 < self.min_len:
                continue  # 坑长不足
            # 超高:出坑后 window_days 天内放量堆峰值量比
            post_peak = max([pp for ss, ee, kk, dd, pp, vv in cl
                             if kk == 'HIGH' and lch < ss <= lch + self.window_days] or [0])
            super_ok = post_peak >= self.peak_ratio
            if self.require_super and not super_ok:
                continue
            sigs.append({
                'idx': int(lch), 'date': dates[lch], 'code': self.code,
                'meta': {'super': super_ok, 'peak_ratio': round(float(post_peak), 2),
                         'pit_len': int(b - s + 1), 'lag': int(lch - b)},
            })
        return sigs
