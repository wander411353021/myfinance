# -*- coding: utf-8 -*-
"""全局算法参数（唯一入口）— 2026-10-18

以后改成本线/筹码分位，只改这个文件（或用环境变量覆盖），不要在各脚本里散改。

主成本线（仅主图显示；信号判定见下方 SIGNAL_* 新标准）:
    COST_PCT        默认 0.70  → COST70

信号判定标准（v3 / v3b / early，2026-10-20 用户定版，全套新标准）:
    SIGNAL_COST_PCT      默认 0.35 → COST35
    SIGNAL_COST_EHLERS   默认 60
    SIGNAL_MIN_LEVEL     默认 4（+0% 档）
    SIGNAL_MIN_LEVEL_DOWN 默认 0（下跌段允许 -9%）
    SIGNAL_DYNAMIC_MIN_LEVEL 默认 True
    A/B 成本线（COST_PCT / COST2_PCT）仅用于显示，不参与信号判定。

板块/筹码三档（必须保持低<中<高；与主成本线独立）:
    COST_PCT_LOW    默认 0.35  → 下沿（受压/跌破）
    COST_PCT_MID    默认 0.50  → 中档（板块页成本中线）
    COST_PCT_HIGH   默认 0.75  → 上沿（强势/峰上沿）

筹码带宽/获利盘（signal_profile 的 band/profit）:
    CHIP_BAND_LO    默认 0.10
    CHIP_BAND_HI    默认 0.90

Ehlers SuperSmoother 平滑周期:
    COST_EHLERS      默认 90  → 主成本线 A
    COST_HIGH_EHLERS 默认 20  → COST75 筹码峰上沿

参考成本线 B（仅主图参考，不参与信号判定）:
    COST2_PCT        默认 0.35 → COST35
    COST2_EHLERS     默认 60

环境变量可临时覆盖，例如:
    COST_PCT=0.50 python -m streamlit run streamlit_demo.py
"""
import os


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _b(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in ('0', 'false', 'no', 'off', '')


COST_PCT = _f('COST_PCT', '0.70')
COST_PCT_LOW = _f('COST_PCT_LOW', '0.35')
COST_PCT_MID = _f('COST_PCT_MID', '0.50')
COST_PCT_HIGH = _f('COST_PCT_HIGH', '0.75')
CHIP_BAND_LO = _f('CHIP_BAND_LO', '0.10')
CHIP_BAND_HI = _f('CHIP_BAND_HI', '0.90')

# Ehlers SuperSmoother 平滑周期（2026-10-18 可调）
#   COST_EHLERS        主成本线（COST_PCT）的平滑周期，默认 90
#   COST_HIGH_EHLERS   COST75 筹码峰上沿的平滑周期，默认 20
COST_EHLERS = int(_f('COST_EHLERS', '90'))
COST_HIGH_EHLERS = int(_f('COST_HIGH_EHLERS', '20'))

# 参考成本线 B（2026-10-18 用户要求：独立分位/周期，主图红色线+红色填充，不参与信号）
COST2_PCT = _f('COST2_PCT', '0.35')
COST2_EHLERS = int(_f('COST2_EHLERS', '60'))

# 信号判定专用标准（2026-10-20 用户定版：v3/v3b/early 全套新标准）
#   不再跟随 A/B 显示线；A/B 只用于主图。
SIGNAL_COST_PCT = _f('SIGNAL_COST_PCT', '0.35')
SIGNAL_COST_EHLERS = int(_f('SIGNAL_COST_EHLERS', '60'))
SIGNAL_MIN_LEVEL = int(_f('SIGNAL_MIN_LEVEL', '4'))
SIGNAL_MIN_LEVEL_DOWN = int(_f('SIGNAL_MIN_LEVEL_DOWN', '0'))
SIGNAL_DYNAMIC_MIN_LEVEL = _b('SIGNAL_DYNAMIC_MIN_LEVEL', True)


def _label(p):
    return 'COST%d' % int(round(p * 100))


COST_LABEL = _label(COST_PCT)
COST_LOW_LABEL = _label(COST_PCT_LOW)
COST_MID_LABEL = _label(COST_PCT_MID)
COST_HIGH_LABEL = _label(COST_PCT_HIGH)
COST2_LABEL = _label(COST2_PCT)
