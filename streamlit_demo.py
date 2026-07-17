# -*- coding: utf-8 -*-
r"""
Streamlit demo —— 本地跑通 V10 价格分段 4 面板图

数据源（全部来自 tdx_quant，无 pkl、无 adata）：
  · K 线：通达信直连  tdx_quant.get_daily_kline_from_tdx(code, end_date)（前复权）
  · 板块/成分：读取本地 result/blocks/*.csv（GBK，由 tdx_quant.update_all_block_info 生成）

左侧菜单提供「板块导航（CSV）」：选板块类型 → 选板块 → 看成分股 → 一键直连载入图表。

启动：
    <venv>\Scripts\streamlit.exe run streamlit_demo.py
或：
    <venv>\Scripts\python.exe -m streamlit run streamlit_demo.py
"""
import os
import tempfile
from datetime import date as dt_date

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from price_segmenter_v10 import run_segmentation
from tdx_quant import (load_block_csvs, load_stock_list, BLOCK_KEYS)

# ── 路径 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="V10 价格分段 Demo", layout="wide")

# 板块类型标签 → 文件键
BLOCK_LABELS = {v: k for k, v in BLOCK_KEYS.items()}  # {'行业/指数': 'block_zs', ...}


# ── 数据加载 ──
@st.cache_data(show_spinner=False)
def load_blocks_cached():
    """缓存读取本地板块 CSV（GBK），避免每次 rerun 重复解析。"""
    return load_block_csvs()


@st.cache_data(show_spinner=False)
def load_stock_names_cached():
    """读取 result/blocks/stock_list.csv，返回 {code: name} 映射。失败则返回空字典。"""
    try:
        df = load_stock_list()
        return dict(zip(df["code"].astype(str), df["name"].astype(str)))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def get_stock_items_cached():
    """[code, name] 列表，供代码输入框的联想组件使用。"""
    nm = load_stock_names_cached()
    return [[c, nm.get(c, "")] for c in nm]


@st.cache_data(show_spinner=False)
def fetch_tdx_kline(code, end_date):
    """通达信直连拉日线（前复权）。end_date 透传给 get_daily_kline_from_tdx 的 anchor_date。"""
    from tdx_quant import get_daily_kline_from_tdx
    df = get_daily_kline_from_tdx(code, end_date)
    if df is None or len(df) == 0:
        raise RuntimeError(f"通达信返回为空（代码 {code}）")
    # 标准化：时间升序、剔除空收盘，保证 run_segmentation 的 tail 取到最近 N 根
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return df


# ── 计算 + 存 session（绘图代码零改动）──
def compute_and_store(df, tail_days, name):
    fd, tmp_png = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    out = run_segmentation(df, tail_days=tail_days, name=name, save_path=tmp_png)
    st.session_state["last_png"] = tmp_png
    st.session_state["last_out"] = out
    st.session_state["last_name"] = name
    return out


# ── 显示（统一在此渲染）──
def display_last():
    out = st.session_state["last_out"]
    st.image(st.session_state["last_png"], use_container_width=True)

    c_result, bs_signal, bs_reason, bs_strength, all_levels = out
    tail_days = st.session_state.get("last_tail", 150)
    off = max(0, len(bs_signal) - tail_days)

    col1, col2, col3 = st.columns(3)
    col1.metric("窗口买入信号数", int((bs_signal[off:] > 0).sum()))
    col2.metric("窗口卖出信号数", int((bs_signal[off:] < 0).sum()))
    col3.metric("阻力/支撑位数（all_levels）", len(all_levels))

    with st.expander("窗口内信号明细", expanded=False):
        rows = []
        for i in range(off, len(bs_signal)):
            if bs_signal[i] != 0:
                rows.append({
                    "rel_bar": i - off,
                    "signal": int(bs_signal[i]),
                    "reason": str(bs_reason[i]),
                    "strength": round(float(bs_strength[i]), 3),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("窗口内无买卖信号。")

    with st.expander("阻力/支撑位生命周期（all_levels）", expanded=False):
        lv_rows = [{
            "price": round(lv["price"], 2),
            "kind": lv.get("kind"),
            "form_idx": lv.get("form_idx"),
            "touch_count": lv.get("touch_count"),
            "break_idx": lv.get("break_idx"),
            "break_strength": (round(lv["break_strength"], 3)
                               if lv.get("break_strength") is not None else None),
        } for lv in all_levels]
        st.dataframe(pd.DataFrame(lv_rows), use_container_width=True, hide_index=True)


# ── 页面 ──
st.title("V10 价格分段 · Streamlit Demo")

st.caption("数据源：通达信直连（tdx_quant.get_daily_kline_from_tdx，前复权）。"
           "输入代码（+ 结束日期）→ 直连拉日线 → 出图。"
           "板块/成分来自本地 result/blocks/*.csv（GBK），可经左侧『板块导航』一键载入成分股。")

# 代码输入框的权威会话值（双向组件的值滞后一帧，故用会话值兜底，避免被旧回传值覆盖）
if "code_box_value" not in st.session_state:
    st.session_state["code_box_value"] = "300437"
# 板块导航一键载入：经 pending_code 触发，组件在 force 时强制把代码显示进输入框
_pc = st.session_state.pop("pending_code", None)
_force_code = False
if _pc is not None:
    st.session_state["code_box_value"] = str(_pc).strip()
    _force_code = True

# 代码输入框组件（带「代码+名称」联想下拉）
_CODE_DIR = os.path.join(BASE_DIR, "_code_component")


@st.cache_resource(show_spinner=False)
def _get_code_component():
    return components.declare_component("code_input", path=_CODE_DIR)


with st.sidebar:
    st.header("行情查询")
    # 代码输入框：带「代码+名称」联想下拉的双向组件（↑/↓ 选择、Enter 或点击填入；不自动出图）
    stock_items = get_stock_items_cached()
    if "_code_stocks_sent" not in st.session_state:
        st.session_state["_code_stocks_sent"] = False
    stocks_arg = stock_items if not st.session_state["_code_stocks_sent"] else []
    comp_val = _get_code_component()(
        stocks=stocks_arg,
        value=st.session_state["code_box_value"],
        force=_force_code,
        key="code_box",
    )
    st.session_state["_code_stocks_sent"] = True

    # 组件回传：新版为 {v, src}（src="user" 真人手动改码 / "force" 板块或同行业点击注入）；
    # 兼容旧式纯字符串（按 "user" 处理）。src 用于区分"应清空同行业列表"与"仅同步输入框"。
    def _parse_code_val(v):
        if isinstance(v, dict):
            return str(v.get("v") or "").strip(), str(v.get("src") or "user")
        if v is None:
            return None, "user"
        return str(v).strip(), "user"

    _comp_v, _comp_src = _parse_code_val(comp_val)
    # 关键：comp_val 是「粘性」的——组件只在真人 select/change 时才 setValue，force 注入
    # （板块/同行业点击）不重发，故 comp_val 会长期停留在「上一次用户手动输入的值」。
    # 若直接拿这个粘性旧值与 code_box_value（已被 force 改成 peer）比较，就会在换股后的
    # 下一次普通 rerun 里把旧值误判为「用户又改了代码」→ 错误清空列表并把代码弹回旧值
    # （表现为：同行业列表里切一次没事、切两次被莫名清空）。
    # 修法：只对组件「真正发出的新值」(签名变化) 做反应，粘性旧值一律忽略。
    _cur_sig = None if comp_val is None else (_comp_v, _comp_src)
    if _force_code:
        # 由 pending_code 注入（板块导航 / 同行业列表点击）：code_box_value 已在顶部更新为 peer，
        # 忽略此帧的粘性 comp_val，绝不在此清空列表。
        pass
    elif _cur_sig is not None and _cur_sig != st.session_state.get("_code_last_seen"):
        # 组件发出了「新」值（真人操作）才处理；粘性旧值签名不变 → 跳过，杜绝误清空
        st.session_state["_code_last_seen"] = _cur_sig
        if _comp_src == "user" and _comp_v and _comp_v != st.session_state["code_box_value"]:
            # 真人在输入框手动改码 → 同步 + 清空同行业列表（需重新点按钮查询）
            st.session_state.pop("_related_df", None)
            st.session_state.pop("_related_for_code", None)
            st.session_state["code_box_value"] = _comp_v
    code = st.session_state["code_box_value"].strip()
    # 板块导航一键载入时会改写 code_box_value 并触发 auto_go
    end_date = st.date_input("结束日期（含当日）", value=dt_date.today())
    tail_days = st.slider(
        "显示窗口 tail_days（画最后 N 根）",
        min_value=60, max_value=2000, value=150, step=10,
    )
    go = st.button("拉取并出图", type="primary", use_container_width=True)

# 板块导航 / 同行业列表点行 / 手动「拉取并出图」都会走到这里
auto_go = st.session_state.pop("auto_go", False)
if go or auto_go:
    if not code or not code.isdigit() or len(code) != 6:
        st.error("请输入 6 位数字股票代码。")
        st.stop()
    st.session_state["last_tail"] = tail_days
    df_full = None
    with st.spinner(f"通达信直连拉取 {code} 日线中（截止 {end_date}）…"):
        try:
            df_full = fetch_tdx_kline(code, end_date)
        except Exception as e:
            # 取数失败（如该票退市/停牌/无行情）不致命：保留上一只图表，仅提示
            st.warning(f"⚠️ 通达信取数失败（代码 {code}）：{e}\n"
                       f"该股票可能已退市 / 停牌 / 无行情数据，已保留上一只股票的图表。")
    if df_full is not None and len(df_full) >= 2:
        st.success(f"已拉取 {len(df_full)} 根，日期 {df_full['date'].iloc[0].date()} ~ {df_full['date'].iloc[-1].date()}（前复权）")
        compute_and_store(df_full, tail_days, f"{code}")
        # 注意：同行业列表不再随主图加载自动刷新（避免卡顿与误刷新）。
        # 仅在输入框手动改码 / 板块导航点股票时清空，需用户手动点「刷新同行业个股」按钮重新查询。
    elif df_full is not None:
        st.error("取数不足 2 根，请检查代码或结束日期。")

# ── 双向组件声明（成分股列表）──
# 注意：st.components.v1.html 是静态 iframe，不会回传值；必须用 declare_component。
_CONS_DIR = os.path.join(BASE_DIR, "_cons_component")


@st.cache_resource(show_spinner=False)
def _get_cons_component():
    return components.declare_component("cons_list", path=_CONS_DIR)


# ── 板块导航（CSV，左侧菜单，始终可见）──
with st.sidebar.expander("板块导航（CSV）", expanded=False):
    st.caption("板块与成分均从 result/blocks/*.csv 读取（GBK）。点击列表，或 ↑↓ 移动高亮后按 Enter 载入并出图。")
    btype = st.radio("板块类型", list(BLOCK_LABELS.keys()), horizontal=True)
    key = BLOCK_LABELS[btype]
    blocks = load_blocks_cached()
    if key not in blocks:
        st.warning(f"未找到 {key}.csv，请先运行 tdx_quant.update_all_block_info()。")
    else:
        bdf = blocks[key]
        csel = st.selectbox("选择板块", bdf["name"].tolist())
        # 切换板块时清空载入记录，允许在新板块中重新载入同名代码
        board_changed = st.session_state.get("_loaded_board") != csel
        st.session_state["_loaded_board"] = csel
        row = bdf[bdf["name"] == csel].iloc[0]
        codes = [str(c) for c in row["codes"]]
        name_map = load_stock_names_cached()
        items = [[c, name_map.get(c, "")] for c in codes]
        st.caption(f"「{csel}」共 {len(codes)} 只成分股 — 点击列表，或 ↑↓ 选择后按 Enter 载入并出图")
        # 双向组件：每个板块用独立 key，切换板块即换实例（default=None，无跨板块残留）
        picked = _get_cons_component()(
            items=items,
            ident=f"{key}|{csel}",
            key=f"cons_{key}_{csel}",
            default=None,
            selected=code,  # 高亮跟随当前主图股票，避免切换后回到第一项
        )
        # 切换板块当次 rerun 不触发载入（避免刚切过来就误加载）
        if board_changed:
            st.session_state["_loaded_cons"] = None
        # 仅当组件回传了"新"代码才载入（Enter/点击触发），避免无关 rerun 重复加载
        if isinstance(picked, str) and picked and picked != st.session_state.get("_loaded_cons"):
            st.session_state["_loaded_cons"] = picked
            st.session_state["pending_code"] = picked
            st.session_state["auto_go"] = True
            # 板块导航点股票 → 切换股票，清空同行业列表（需手动重新点按钮刷新）
            st.session_state.pop("_related_df", None)
            st.session_state.pop("_related_for_code", None)
            st.rerun()

# ── 关联个股（同行业）；替代原「同题材可比股票」概念下拉流程 ──
with st.sidebar.expander("关联个股（同行业）", expanded=True):
    st.caption(f"展示当前股票（{code}）「同行业个股」的 [证券代码, 股票名称]。"
               "仅在点击下方按钮时查询；在输入框改码或板块导航点股票会清空本列表，需重新点按钮。"
               "点击列表中的某只即可载入它的主图（不清空本列表）。")
    if st.button(f"刷新同行业个股 ({code})", use_container_width=True, key="rel_query"):
        if not code or len(code) != 6 or not code.isdigit():
            st.error("请先在上方输入有效的 6 位股票代码。")
        else:
            try:
                from related_stocks import get_related_df
                with st.spinner(f"通达信 F10 关联个股查询中（{code} · 同行业）…"):
                    _rel_df = get_related_df(code, "2")
                st.session_state["_related_df"] = _rel_df
                st.session_state["_related_for_code"] = code
                st.session_state["_loaded_rel"] = None
            except Exception as e:
                st.error(f"查询失败：{e}")
                st.session_state.pop("_related_df", None)

    _rel_df = st.session_state.get("_related_df")
    _rel_for = st.session_state.get("_related_for_code")  # 当前展示的"同行业列表"所属股票
    # 只要已查过就展示（列表绑定 _related_for_code，即使当前主图已切换到某只同行业个股也保留）
    if _rel_df is not None:
        cols = [c for c in ("证券代码", "股票名称") if c in _rel_df.columns]
        if cols:
            _sec = _rel_df.attrs.get("section", "")
            _date = _rel_df.attrs.get("date")
            _scope = _rel_df.attrs.get("scope")
            st.caption(f"{_sec} · 截止 {_date} · {_scope}（共 {len(_rel_df)} 只）— 列表为 {_rel_for} 的同行业，"
                       f"点击 / ↑↓+Enter 仅切换主图，不刷新本列表")
            # 复用与板块导航相同的双向列表组件：↑↓ 移动高亮、Enter/点击载入并出图
            # key 绑定 _rel_for（而非当前 code），浏览同行业时列表与高亮保持稳定
            items = [[str(r["证券代码"]).zfill(6), str(r["股票名称"])]
                     for _, r in _rel_df.iterrows()]
            picked_rel = _get_cons_component()(
                items=items,
                ident=f"rel|{_rel_for}",
                key=f"rel_{_rel_for}",
                default=None,
                selected=code,  # 高亮跟随当前主图股票：点 peer 后主图切到该 peer，列表高亮也停在该行
            )
            # 仅当组件回传了"新"代码才载入（Enter/点击触发），避免无关 rerun 重复加载
            # 同行业列表内点击：只换主图，既不刷新也不清空本列表（保留清单供继续浏览）
            if isinstance(picked_rel, str) and picked_rel and picked_rel != st.session_state.get("_loaded_rel"):
                st.session_state["_loaded_rel"] = picked_rel
                st.session_state["pending_code"] = picked_rel
                st.session_state["auto_go"] = True
                st.rerun()
        else:
            st.warning("返回结果中未包含「证券代码 / 股票名称」列。")
    else:
        st.caption(f"尚未查询 {code} 的同行业个股（在上方输入框改码或板块导航点股票后已清空；点「刷新同行业个股」按钮查询）。")

# 统一显示：仅当 session 里已存图
if "last_png" in st.session_state:
    display_last()
else:
    st.info("输入股票代码与结束日期后，点击「拉取并出图」（或经左侧『板块导航』载入成分股）。")
