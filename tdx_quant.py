import os
import ast
import pandas as pd

def get_daily_kline_from_tdx(code, end_date):
    from eltdx import TdxClient
    with TdxClient() as client:
        adj = client.get_adjusted_kline(period='day', code=code, adjust='qfq', anchor_date=end_date)
    df = pd.DataFrame({
        'date':   [b.time for b in adj.bars],
        'open':   [float(b.open) for b in adj.bars],
        'high':   [float(b.high) for b in adj.bars],
        'low':    [float(b.low) for b in adj.bars],
        'close':  [float(b.close) for b in adj.bars],
        'volume': [float(b.volume_lots) for b in adj.bars],
    })
    df = df[df['close'] > 0].reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
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
    