# -*- coding: utf-8 -*-
"""全局算法参数（唯一入口）— 2026-10-18

以后改成本线/筹码分位，只改这个文件（或用环境变量覆盖），不要在各脚本里散改。

主成本线（信号判定 / V10 深绿线 / 结构冷却 / early / v3b / v3）:
    COST_PCT        默认 0.50  → COST50

筹码/板块三档（必须保持低<中<高，否则分类失效）:
    COST_PCT_LOW    默认 0.35  → 下沿（受压/跌破）
    COST_PCT        默认 0.50  → 主成本线（中位）
    COST_PCT_HIGH   默认 0.75  → 上沿（强势/峰上沿）

筹码带宽/获利盘（signal_profile 的 band/profit）:
    CHIP_BAND_LO    默认 0.10
    CHIP_BAND_HI    默认 0.90

Ehlers SuperSmoother 平滑周期:
    COST_EHLERS      默认 45  → 主成本线
    COST_HIGH_EHLERS 默认 20  → COST75 筹码峰上沿

环境变量可临时覆盖，例如:
    COST_PCT=0.35 python -m streamlit run streamlit_demo.py
"""
import os


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


COST_PCT = _f('COST_PCT', '0.50')
COST_PCT_LOW = _f('COST_PCT_LOW', '0.35')
COST_PCT_HIGH = _f('COST_PCT_HIGH', '0.75')
CHIP_BAND_LO = _f('CHIP_BAND_LO', '0.10')
CHIP_BAND_HI = _f('CHIP_BAND_HI', '0.90')

# Ehlers SuperSmoother 平滑周期（2026-10-18 可调）
#   COST_EHLERS        主成本线（COST_PCT）的平滑周期，默认 45
#   COST_HIGH_EHLERS   COST75 筹码峰上沿的平滑周期，默认 20
COST_EHLERS = int(_f('COST_EHLERS', '45'))
COST_HIGH_EHLERS = int(_f('COST_HIGH_EHLERS', '20'))


def _label(p):
    return 'COST%d' % int(round(p * 100))


COST_LABEL = _label(COST_PCT)
COST_LOW_LABEL = _label(COST_PCT_LOW)
COST_HIGH_LABEL = _label(COST_PCT_HIGH)
