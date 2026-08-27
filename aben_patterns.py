# -*- coding: utf-8 -*-
"""阿笨"股眼"形态量化识别(注释中文,无未来函数,只求识别不追求胜率)。

实现形态(阿笨文章归纳):
  1. 偏态走势(piantai):趋势向下(reg 斜率<0)中突然放量+大涨的 V 型波
  2. 金龙探海(golden_dragon):连续两根大阳线(堆量)之后被吃掉(吸筹型K线)
  3. 墨龙探海(ink_dragon):同金龙探海但表面为阴线(看量不看阴阳)

用法:
    from aben_patterns import detect_piantai, detect_golden_dragon, detect_ink_dragon
    pts = detect_piantai(closes, volumes, reg250)
    gd  = detect_golden_dragon(closes, volumes)
"""
import numpy as np
from mean_reversion.signal_residual import compute_rolling_regression


def _reg_slope(closes, win=120, at=None):
    """at 日 reg 斜率(年化),只依赖历史。"""
    if at is None:
        at = len(closes) - 1
    if at < win:
        return 0.0
    seg = closes[at - win + 1:at + 1]
    x = np.arange(win, dtype=float)
    y = np.log(seg)
    a, b = np.polyfit(x, y, 1)
    return (np.exp(a * 250) - 1)  # 年化斜率


def detect_piantai(closes, volumes, reg250=None, drop_days=40, up_pct=0.08,
                   vol_ratio=1.5, min_up_days=3):
    """偏态走势:趋势向下中的 V 型放量大涨。

    条件(全部只用 i 及之前):
      1) 前期趋势向下:reg250 斜率<0(或 60 日下跌)
      2) 突然放量大涨:单日涨幅>=up_pct 且 量比>=vol_ratio
      3) 之后延续:连续 min_up_days 天累计涨幅(确认 V 型)
    返回 [(起点idx, 启动idx, 涨幅)] 升序。
    """
    c = np.asarray(closes, dtype=float)
    v = np.asarray(volumes, dtype=float)
    n = len(c)
    if reg250 is None:
        reg250, _ = compute_rolling_regression(c, window=250, use_log=True)
    out = []
    i = drop_days
    while i < n - min_up_days:
        # 前期趋势向下:reg 斜率<0
        slope = _reg_slope(c, 120, i - 1)
        if slope > 0:
            i += 1
            continue
        # 突然大涨 + 放量
        if c[i] / c[i - 1] - 1 >= up_pct:
            vr = v[i] / (np.mean(v[max(0, i - 20):i]) + 1e-9)
            if vr >= vol_ratio:
                # 确认延续:后续 min_up_days 天累计涨幅
                fut = c[i + min_up_days - 1] / c[i] - 1
                if fut > 0.02:
                    out.append((i, i, fut))
                    i += min_up_days  # 去重
                    continue
        i += 1
    return out


def detect_golden_dragon(closes, volumes, big_pct=0.05, vol_ratio=1.3, eat_pct=0.5):
    """金龙探海(吸筹型):连续两根大阳线(堆量)+ 之后被吃掉一半。

    条件:
      1) 连续 2 根大阳线(涨幅>=big_pct)且放量(量比>=vol_ratio)
      2) 之后回调吃掉第一根大阳线的 eat_pct(回调确认,非追高)
    返回 [(双阳起点idx, 回调低点idx)]。
    """
    c = np.asarray(closes, dtype=float)
    v = np.asarray(volumes, dtype=float)
    n = len(c)
    out = []
    i = 25
    while i < n - 30:
        r1 = c[i] / c[i - 1] - 1
        r2 = c[i + 1] / c[i] - 1
        vr = np.mean(v[i:i + 2]) / (np.mean(v[max(0, i - 20):i]) + 1e-9)
        if r1 >= big_pct and r2 >= big_pct and vr >= vol_ratio:
            # 双阳起点 = i-1(第一根大阳线),之后找回调低点(吃掉部分)
            base = c[i - 1]
            j = i + 2
            while j < n - 5:
                if c[j] <= base * (1 - eat_pct * (base - c[i - 1]) / base) or c[j] <= base * (1 - 0.03):
                    out.append((i - 1, j))
                    break
                j += 1
        i += 1
    return out


def detect_ink_dragon(closes, volumes, big_pct=0.04, vol_ratio=1.3):
    """墨龙探海:连续两根大阴线但放量(看量不看阴阳,量堆吸筹)。

    条件:连续 2 根阴线(跌幅>=big_pct)且放量(量比>=vol_ratio),
    之后 10 天内收复(反转)。
    返回 [(双阴起点idx, 收复idx)]。
    """
    c = np.asarray(closes, dtype=float)
    v = np.asarray(volumes, dtype=float)
    n = len(c)
    out = []
    i = 25
    while i < n - 15:
        d1 = c[i - 1] / c[i] - 1
        d2 = c[i] / c[i + 1] - 1
        vr = np.mean(v[i:i + 2]) / (np.mean(v[max(0, i - 20):i]) + 1e-9)
        if d1 >= big_pct and d2 >= big_pct and vr >= vol_ratio:
            base = c[i + 1]  # 双阴后低点
            for j in range(i + 2, min(i + 12, n)):
                if c[j] > base:
                    out.append((i - 1, j))
                    break
        i += 1
    return out


def detect_guyan(closes, volumes, pre_win=60, post_win=60, min_trend=0.08, stop_ok=-0.03, mega=8.0):
    """股眼 = 改变走势的关键量堆(用户定义,2026-08-26)。

    识别:放量堆(仅用历史,无未来函数)+ 走势确认(用未来 post_win 日,标注用)。
    v2 调整:reversal 只需"止跌"(post>-3% 且 pre<-8%),识别巨量吸筹堆
      (300437 20210412 峰值16.5 前-19%→后+8% 被识别 = S2 吸筹股眼);
      新增 mega_absorb(峰值>8 巨量堆单独标吸筹)。
    类型: reversal=下跌→止跌/反转 / breakout=横盘→突破 / accel=上涨→加速 / mega_absorb=巨量吸筹。
    返回 [(堆起点, 堆终点, 前段涨跌, 后段涨跌, 类型, 峰值量比)]。
    案例:300437 20210412(+8%止跌)、20210721(→+370%)、20210830(→+205%);
         300251 20240924(吸筹)、20241028(→+164%);300204 20250407(→+33%)。
    """
    import panic_reversal as _pr
    c = np.asarray(closes, dtype=float)
    n = len(c)
    clusters = _pr.detect_volume_clusters(closes, volumes)
    out = []
    for s, e, kd, dr, pk, vr in clusters:
        if kd != 'HIGH' or s < pre_win + 10 or e + post_win >= n:
            continue
        pre = c[s] / c[s - pre_win] - 1
        post = c[e + post_win] / c[e] - 1
        if pre < -min_trend and post > stop_ok:
            out.append((s, e, pre, post, 'reversal', pk))
        elif abs(pre) < min_trend and post > min_trend:
            out.append((s, e, pre, post, 'breakout', pk))
        elif pre > min_trend and post > pre:
            out.append((s, e, pre, post, 'accel', pk))
        elif pk >= mega and post > stop_ok:
            out.append((s, e, pre, post, 'mega_absorb', pk))
    return out
