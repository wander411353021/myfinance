import os
import ast
import numpy as np
import pandas as pd

_PAGE = 800  # eltdx 单次返回上限(协议限制,count>800 报 ProtocolError)

def _historical_float_shares_records(code):
    """通达信历史流通股本记录: {日期: 流通股本(股)}。
    来源 corporate.capital_changes 中 category_raw in (5,9) 的 c3_value;
    无未来函数: 只取记录日期≤某天的最后一条, 第 i 天换手率只用 ≤i 天的股本。
    记录为空(次新股/无变动记录)时返回 {}, 调用方 fallback 当前快照并告警。
    """
    from eltdx import TdxClient
    recs = {}
    with TdxClient() as client:
        cc = client.corporate.capital_changes(code)
        for r in cc.records:
            # 流通股本只在这些类别中出现(c3字段=流通股本, 单位股):
            #   2=送配股上市 / 3=非流通股上市(全流通) / 5=股本变化 / 9=转配股上市
            #   category 1(除权除息)/14(送认沽权证) 的 c3 是比例系数(如6/1/16), 严禁混入
            if r.category_raw in (2, 3, 5, 9) and r.c3_value and r.c3_value > 0:
                d = pd.to_datetime(r.date).normalize()
                recs[d] = float(r.c3_value)
    return recs


def _current_float_shares(code):
    """当前流通股本快照(股) — 仅作历史记录缺失时的 fallback(会告警)。"""
    from eltdx import TdxClient
    with TdxClient() as client:
        rows = client.helpers.daily_share_capital([code]).rows
    if not rows:
        return None
    return float(rows[0].circulating_shares)


def add_turnover(df, code, end_date):
    """给日线 df 附加换手率列(无未来函数, 逐日历史股本):
      turnover    float  换手率 = 成交量(股)/当日流通股本(股), 小数(可>1, 如次新/暴涨日)
      circ_shares float  当日流通股本(股), 来自历史资本变动记录(≤当日), 无记录时 fallback 当前快照
    volume 单位是手(volume_lots), 需 ×100 转股。
    """
    recs = _historical_float_shares_records(code)
    dates = df['date'].values
    vol_shares = df['volume'].values.astype(float) * 100.0   # 手→股
    circ = np.empty(len(df), dtype=float)
    if recs:
        rec_dates = np.array(sorted(recs.keys()), dtype='datetime64[ns]')
        rec_vals = np.array([recs[pd.Timestamp(d)] for d in rec_dates], dtype=float)
        idx = np.searchsorted(rec_dates, dates, side='right') - 1
        circ[:] = rec_vals[np.clip(idx, 0, len(rec_dates)-1)]
    else:
        circ[:] = np.nan
    if np.isnan(circ).any():
        cur = _current_float_shares(code)
        if cur is None:
            raise RuntimeError(f'{code} 无历史股本记录且当前快照不可用, 无法计算换手率')
        if np.isnan(circ).all():
            print(f'⚠ {code} 无历史资本变动记录, 换手率用当前流通股本快照({cur/1e8:.2f}亿股)替代 (解禁/增发前会高估)')
        else:
            print(f'⚠ {code} 部分日期无股本记录({int(np.isnan(circ).sum())}天), 用当前快照替代')
        circ[np.isnan(circ)] = cur
    df = df.copy()
    df['circ_shares'] = circ
    df['turnover'] = vol_shares / circ
    return df


def get_daily_kline_from_tdx(code, end_date, datalen=800, with_turnover=True):
    """通达信直连拉日线(前复权)。列: date/open/high/low/close/volume
    若 with_turnover=True(默认), 额外附加 turnover(换手率)/circ_shares(当日流通股本),
    换手率用逐日历史股本计算(无未来函数), 见 add_turnover。

    datalen>800 时自动分页拼接(eltdx 单次 count>800 报 ProtocolError):
    start=0/800/1600... 逐页拉取,按时间升序拼接。datalen=2400 ≈ 10 年。
    """
    from eltdx import TdxClient
    n_pages = (datalen + _PAGE - 1) // _PAGE
    all_bars = []
    with TdxClient() as client:
        for pi in range(n_pages):
            adj = client.bars.get(code, period='day', adjust='qfq',
                                  anchor_date=end_date, start=pi * _PAGE, count=_PAGE)
            bars = [b for b in adj.bars if float(b.close) > 0]
            if not bars:
                break
            all_bars.extend(bars)
    if not all_bars:
        return None
    all_bars.sort(key=lambda b: b.time)  # 分页返回顺序须显式排序(防错位)
    df = pd.DataFrame({
        'date':   [b.time for b in all_bars],
        'open':   [float(b.open) for b in all_bars],
        'high':   [float(b.high) for b in all_bars],
        'low':    [float(b.low) for b in all_bars],
        'close':  [float(b.close) for b in all_bars],
        'volume': [float(b.volume_lots) for b in all_bars],
    })
    df = df[df['close'] > 0].reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()  # 去 15:00:00 收盘时间戳,统一为纯日期(00:00:00)
    df = df.drop_duplicates(subset='date').reset_index(drop=True)
    # 2026-09-03 防御(豆包报告偶发混入非交易日bar): 日线只应含周一~周五交易日
    df = df[df['date'].dt.weekday < 5].reset_index(drop=True)
    df = df.tail(datalen).reset_index(drop=True)
    if with_turnover:
        df = add_turnover(df, code, end_date)
    return df


def get_weekly_kline_from_tdx(code, end_date):
    """通达信直连拉周线(前复权)。列结构与日线一致(date/open/high/low/close/volume)。
    eltdx period='week' 返回约 800 根;用于周线级别探索主升前形态。
    """
    from eltdx import TdxClient
    with TdxClient() as client:
        adj = client.get_adjusted_kline(period='week', code=code, adjust='qfq', anchor_date=end_date)
    df = pd.DataFrame({
        'date':   [b.time for b in adj.bars],
        'open':   [float(b.open) for b in adj.bars],
        'high':   [float(b.high) for b in adj.bars],
        'low':    [float(b.low) for b in adj.bars],
        'close':  [float(b.close) for b in adj.bars],
        'volume': [float(b.volume_lots) for b in adj.bars],
    })
    df = df[df['close'] > 0].reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    return df


"""
DataFrame 列说明:
  name      str         板块名称（如"房地产"、"新能源车"、"央企改革"）
  category  int         板块分类编号（0=行业, 1=地域, 2=概念, 3=风格, 等）
  count     int         板块内包含的股票数量
  codes     list[str]   板块成分股代码列表（每个代码为 6 位数字字符串）

三个常用板块文件:
  'block_zs.dat'  -- 行业/指数板块（约 80 个，按申万行业分类）
  'block_gn.dat'  -- 概念板块（约 500+ 个，按市场热点主题分类）
  'block_fg.dat'  -- 风格板块（约 50 个，按市值/估值/地域等风格分类）
"""
def get_block_info(type="block_gn.dat"):
    from easy_tdx import TdxClient
    with TdxClient.from_best_host() as c:
        df = c.get_block_info(type)
        print(f"概念板块，共 {len(df)} 个:")
        return df

def update_all_block_info():
    from easy_tdx import TdxClient
    with TdxClient.from_best_host() as c:
        for type in ["block_zs", "block_gn", "block_fg"]:
            df = c.get_block_info(f"{type}.dat")
            df.to_csv(f"./result/blocks/{type}.csv", index=False, encoding="utf-8")
    return

# ── 本地 CSV 读取（GBK 编码，由 update_all_block_info 生成）──
def _parse_codes(cell):
    """codes 列是字符串化的 Python list，如 \"['600000', '600001']\"。
    优先 ast.literal_eval；失败则退化为按逗号拆分。"""
    if isinstance(cell, list):
        return cell
    s = str(cell).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    try:
        return ast.literal_eval("[" + s + "]") if s else []
    except Exception:
        return [c.strip().strip("'\"").strip('"\'')
                for c in s.split(",") if c.strip()]


BLOCK_KEYS = {
    "block_zs": "行业/指数",
    "block_gn": "概念",
    "block_fg": "风格",
}


def load_block_csvs(block_dir=None):
    """从 result/blocks/*.csv（GBK）读取板块与成分股。

    返回 dict: {块类型键: DataFrame(columns=[name, category, count, codes(list)])}
      - block_zs: 行业/指数板块
      - block_gn: 概念板块
      - block_fg: 风格板块
    codes 已解析为 6 位代码字符串列表。
    """
    if block_dir is None:
        base = os.path.dirname(os.path.abspath(__file__))
        block_dir = os.path.join(base, "result", "blocks")
    out = {}
    for key in BLOCK_KEYS:
        f = os.path.join(block_dir, f"{key}.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, encoding="utf-8")
        if "codes" in df.columns:
            df["codes"] = df["codes"].apply(_parse_codes)
        out[key] = df
    return out

def update_all_stck_list():
    from eltdx import TdxClient
    with TdxClient(timeout=3) as client:
        stock_list = client.get_a_share_codes_all()
        print("共有%d只股票" % len(stock_list))
        period = 200
        count = len(stock_list) / period
        left = len(stock_list) % period
        print("需要分%d次获取，每次%d只" % (count, period))
        print("最后一批%d只" % left)
        rows = tuple()
        for i in range(int(count)):
            table = client.helpers.stock_profile_table(stock_list[i*period:i*period+period])
            rows += table.rows
        if left > 0:
            table = client.helpers.stock_profile_table(stock_list[-left:])
            rows += table.rows

        stock_df = pd.DataFrame(columns=['code', 'name'])
        for row in rows:
            full_code = row.full_code
            short_code = full_code[2:]
            stock_df.loc[len(stock_df)] = [short_code, row.name]
        stock_df.to_csv(f"./result/blocks/stock_list.csv", index=False, encoding="utf-8")
        print("保存完毕")

def load_stock_list(stock_list_file=None):
    if stock_list_file is None:
        stock_list_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", "blocks", "stock_list.csv")
    stock_df = pd.read_csv(stock_list_file, encoding="utf-8", dtype=str)
    return stock_df

# ── 课比较股票列表 ──
def get_same_topic_stocks(code):
    from eltdx import TdxClient
    stocks_df = pd.DataFrame(columns=["full_code", "code", "name"])
    with TdxClient(timeout=3) as client:
        stocks = client.helpers.topic_stocks(code)
        for row in stocks.rows:
            stocks_df.loc[len(stocks_df)] = [row.full_code, row.code, row.name]
        return stocks_df

def _pick(row, *names):
    """容错取字段：依次尝试候选键名，返回首个存在且非空的；都不命中再返回首个存在的；否则 None。
    通达信 F10 不同接口的返回字段命名不统一（如 zqdm/code、zqjc/name、t001/id、t002/ztmc），
    用此函数避免因单一字段名缺失而 KeyError。"""
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    for n in names:
        if n in row:
            return row[n]
    return None


def get_stock_concepts(code):
    from eltdx import F10Client
    concept_df = pd.DataFrame(columns=["concept", "id"])
    f10 = F10Client(timeout=3)
    response = f10.hot_topics(code)
    for row in response.rows:
        # 原始字段可能为 id/t001/topic_id（概念ID）、ztmc/t002/topic_name（概念名）
        cid = _pick(row, "id", "t001", "topic_id")
        cname = _pick(row, "ztmc", "t002", "topic_name", "mc")
        if cid is None:
            continue
        concept_df.loc[len(concept_df)] = [str(cname or ""), str(cid)]
    return concept_df

def get_stock_concept_compare_by_id(code, concept_id):
    from eltdx import F10Client
    f10 = F10Client(timeout=3)
    stock_df = pd.DataFrame(columns=["code", "name"])
    profile = f10.topic_compare(code=code, topic_id=concept_id)
    for row in profile.rows:
        # 原始字段可能为 zqdm/code（代码）、zqjc/name（名称）
        c = _pick(row, "zqdm", "code", "gpdm")
        n = _pick(row, "zqjc", "name", "mc", "gpmc")
        if c is None:
            continue
        stock_df.loc[len(stock_df)] = [str(c), str(n or "")]
    return stock_df
    