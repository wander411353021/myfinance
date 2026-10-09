# -*- coding: utf-8 -*-
"""概念板块筹码页面(streamlit 多页面, 2026-10-08 polo4111)。
用法: streamlit run app.py -> 侧边栏切换到「1 概念板块筹码」
功能: 选择通达信概念板块(8805xx-8809xx, 含名称) -> 拉指数日线(方案A换手率代理)
      -> V10风格 或 筹码带 画图 -> 显示筹码状态(收盘 vs COST35/75 位置)。
红线: 逐日递推 fengwo 只用当日及以前数据, 无未来函数。
"""
import os
import sys
import tempfile
from datetime import date as dt_date

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
from index_chip_plot import fetch_index as fetch_index_chip, plot_chip_band
from index_v10_plot import fetch_index as fetch_index_v10
from price_segmenter_v10 import run_segmentation
from tdx_index_names import display_label
from concept_chip_scan import turnover_proxy

st.set_page_config(page_title="概念板块筹码", layout="wide")


@st.cache_data(show_spinner=False)
def load_concepts():
    """8805xx-8809xx 概念板块 {code: name}, 升序。
    2026-10-09 reasonix: 排除 52 个"动态成分"板块(昨日涨停/跌停/连板/历史新高/重仓/次新/
    送转/解禁…), 其成分股按日或短期重构, 指数是动态组合, 筹码成本无持有者对应
    (实测"昨日跌停"显示 -38% 虚假超跌) —— 只保留 373 个真实题材/概念板块。"""
    try:
        m = json.load(open(os.path.join(BASE_DIR, 'tdx_block_names.json'), encoding='utf-8'))
    except Exception:
        return {}
    try:
        from concept_chip_scan import is_dynamic_block
        _dyn = is_dynamic_block
    except Exception:
        _dyn = lambda n: False
    return {c: m[c] for c in sorted(m)
            if 880500 <= int(c) <= 880999 and not _dyn(m[c])}


@st.cache_data(show_spinner=False)
def fetch_concept_kline(code, end_str, datalen=800):
    # 默认无锚定: 只拉最近 datalen 根(快); 结束日期早于2年前才锚定全量(保证窗口覆盖)
    anchor = False
    try:
        if (dt_date.today() - pd.Timestamp(end_str).date()).days > 730:
            anchor = True
    except Exception:
        anchor = False
    return fetch_index_v10('sh' + code, end_str, datalen=datalen, anchor=anchor)


st.title("概念板块筹码")
st.caption("通达信概念指数(88系列, 已排除动态成分板块) · 方案A换手率代理(turn=0.05×量比) · 逐日递推无未来函数")
st.caption("📈 2026-10-09 验证(363板块/32734样本): 深度跌破 COST35(<-8%) 后续 20/60 日胜率 76%; "
           "极深(<-15%) 88%; 轻微跌破/收>COST75 均无优势 —— 需按深度分档使用")

concepts = load_concepts()
if not concepts:
    st.error("板块映射 tdx_block_names.json 缺失，无法加载概念列表。")
    st.stop()

with st.sidebar:
    st.header("概念板块选择")
    options = ["%s %s" % (c, n) for c, n in concepts.items()]
    sel = st.selectbox("概念板块", options, index=0)
    code = sel.split(" ")[0]
    name = sel.split(" ", 1)[1]
    end_date = st.date_input("结束日期（含当日）", value=dt_date.today())
    tail_days = st.slider("显示窗口（最后 N 根）", 60, 500, 150, 10)
    style = st.radio("画图风格", ["V10 风格（含筹码线）", "筹码带（COST15-85 集中区）"], horizontal=True)
    go = st.button("拉取并出图", type="primary", width='stretch')

if not go:
    st.info("左侧选择概念板块 → 点「拉取并出图」")
    st.stop()

end_str = end_date.strftime("%Y%m%d")
with st.spinner(f"拉取概念指数 {code} {name} 日线（截止 {end_str}）…"):
    try:
        df = fetch_concept_kline(code, end_str)
        if df is None or len(df) < 300:
            st.error(f"数据不足：{code} 仅 {0 if df is None else len(df)} 根（可能为新概念指数）")
            st.stop()
    except Exception as e:
        st.error(f"取数失败：{e}")
        st.stop()

fd, tmp_png = tempfile.mkstemp(suffix=".png")
os.close(fd)

try:
    if style.startswith("V10"):
        df2 = df.copy()
        df2['turnover'] = turnover_proxy(df2['volume'].values.astype(float))
        run_segmentation(df2, tail_days=tail_days,
                         name=display_label(code), code=None, end_date=end_str,
                         show_chip=True, hide_overlay_lines=True,
                         save_path=tmp_png)
        summary = None
    else:
        summary = plot_chip_band(df, tail=tail_days, out=tmp_png, code=display_label(code))
except Exception as e:
    st.error(f"画图失败：{e}")
    st.stop()

st.image(tmp_png, width='stretch')

# 筹码状态摘要
V = df['volume'].values.astype(float)
turn = turnover_proxy(V)
from chip_panel import cost_series
H = df['high'].values.astype(float)
L = df['low'].values.astype(float)
C = df['close'].values.astype(float)
c35 = float(cost_series(H, L, V, turn, 0.35)[-1])
c50 = float(cost_series(H, L, V, turn, 0.50)[-1])
c75 = float(cost_series(H, L, V, turn, 0.75)[-1])
close = float(C[-1])

if summary is None:
    summary = dict(close=close, c35=c35, c50=c50, c75=c75)

above75 = close / summary['c75'] - 1
below35 = close / summary['c35'] - 1
# 2026-10-09 reasonix 验证修正: 概念板块(413个/37451样本/2020-2026)统计显示
#   "收<COST35 越深 → 后续 60 日反弹越强"(深度-8~-15%: +34%/胜率67%; <-15%: +89%/胜率72%),
#   而"收>COST75"后续平庸(+1.5%~+8%) —— 方向与"强势=好"直觉相反, 标签改为中性/超跌语义。
if below35 < -0.15:
    pos = "极深超跌（收盘 << COST35）"
    note = "统计: 60日 +89%/胜率72%(n=263, 均值受尾部驱动)"
elif below35 < -0.08:
    pos = "深度跌破（收盘 < COST35）"
    note = "统计: 60日 +34%/胜率67%(n=955)"
elif below35 < 0:
    pos = "轻度跌破（收盘 < COST35）"
    note = "统计: 60日 +5~11%(优势不明显)"
elif above75 > 0.08:
    pos = "高位（收盘 >> COST75）"
    note = "统计: 60日均值 +38% 但胜率仅 52%(尾部驱动)"
elif above75 > 0:
    pos = "突破筹码峰（收盘 > COST75）"
    note = "统计: 60日 +1.5%~+8%(偏平庸)"
else:
    pos = "中间（部分套牢/获利）"
    note = "统计: 无优势"

col1, col2, col3, col4 = st.columns(4)
col1.metric("收盘", f"{close:.2f}")
col2.metric("COST35（筹码峰下沿）", f"{summary['c35']:.2f}")
col3.metric("COST50（成本中线）", f"{summary['c50']:.2f}")
col4.metric("COST75（筹码峰上沿）", f"{summary['c75']:.2f}")
st.markdown(f"**板块筹码状态：{pos}**　偏离 COST35 `{below35*100:+.1f}%` · 偏离 COST75 `{above75*100:+.1f}%`")
st.caption(f"📊 {note}　·　注: 概念指数无股本, 换手率用代理(0.05×量/MA20), 绝对成本价不可当压力位; 统计存在样本重叠")

if style.startswith("V10"):
    with st.expander("说明：V10 图上的筹码线"):
        st.markdown(
            "- 深绿虚线 = COST35（Ehlers45 平滑，跌破段深绿填充）\n"
            "- 橙红虚线 = COST75（Ehlers20，放量段显示）\n"
            "- 深蓝实线 = COST50\n"
            "- 与 DOWN zone low 同色的绿色虚线是 V10 分段支撑线，非筹码线")
