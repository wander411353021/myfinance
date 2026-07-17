"""稳定获取「关联个股 / 同行业个股」F10 数据。

背景：TDX 不同行情主机返回的 F10 结构不一样——
  - 旧版主机（如 180.153.18.170）：返回 ~16 个分类，含「关联个股」（内含
    同行业个股、同地域个股、集团关联、基金持仓等多张表）。
  - 新版主机（如 124.70.x）：只返回 1 条「最新提示」，没有关联个股栏。
`from_best_host()` 每次按延迟随机挑主机，因此会出现「刚刚有、现在没了」。

解决：重试若干次，直到连上返回多分类（含关联个股）的主机；也可用
KNOWN_OLD_HOSTS 直接指定已知旧版主机优先尝试。
"""

from easy_tdx import Market, TdxClient

CODE = "600519"
MARKET = Market.SH
RELATED_KW = "关联个股"      # 该分类内含「同行业个股」等表
MAX_RETRY = 20               # 随机重连上限
MIN_CATS = 10                # 判定为旧版多分类主机的阈值
# 已知含关联个股的旧版主机（优先直连，命中即省去重试）
KNOWN_OLD_HOSTS = ["180.153.18.170"]


def _read_related(c):
    """在已连接的 client 上读取关联个股分类完整内容；无该分类返回 None。"""
    cats = c.get_company_info_category(MARKET, CODE)
    if len(cats) < MIN_CATS:
        return None, cats
    row = cats[cats["name"].str.contains(RELATED_KW, na=False)]
    if row.empty:
        return None, cats
    r = row.iloc[0]
    content = c.get_company_info_content(
        MARKET, CODE, str(r["filename"]), int(r["start"]), int(r["length"]))
    return content, cats


def get_related_stocks(code=CODE, market=MARKET):
    """稳定返回「关联个股」分类全文（含同行业个股表）。连不到旧版主机则抛异常。"""
    # 1) 优先直连已知旧版主机
    for host in KNOWN_OLD_HOSTS:
        try:
            with TdxClient(host=host) as c:
                content, cats = _read_related(c)
                if content:
                    return content, host, cats
        except Exception:
            pass
    # 2) 回退：重试 from_best_host 直到连上多分类主机
    for _ in range(MAX_RETRY):
        try:
            with TdxClient.from_best_host() as c:
                content, cats = _read_related(c)
                if content:
                    host = getattr(c, "_host", "?")
                    return content, host, cats
        except Exception:
            continue
    raise RuntimeError(
        f"重试 {MAX_RETRY} 次仍未连上含「{RELATED_KW}」的旧版主机——"
        f"当前多为只返回「最新提示」的新版主机，稍后重试或补充 KNOWN_OLD_HOSTS。")


if __name__ == "__main__":
    content, host, cats = get_related_stocks()
    print(f"命中主机 host={host}  目录条数={len(cats)}")
    print("分类列表:", cats["name"].tolist())
    print("=" * 64)
    print(content)
