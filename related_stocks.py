"""按 code + 类型 获取「关联个股」的 DataFrame。

背景：通达信 F10 的「关联个股」分类内含四张表——
  【1.同大股东个股】【2.同行业个股】【3.股本相近个股】【4.同地域个股】
但 TDX 不同行情主机返回的 F10 结构不同：旧版主机含此分类，
新版主机（如 124.70.x）只返回「最新提示」、没有关联个股。
因此本模块优先直连已知旧版主机，连不到再随机重试。

用法：
    from related_stocks import get_related_df
    df = get_related_df("600199", "2.同行业个股")   # 或 "2" / "同行业个股"
    print(df.columns, df.attrs)   # df.attrs 含 date / scope / section / code
"""

import re
import threading
import pandas as pd
from easy_tdx import Market, TdxClient

RELATED_KW = "关联个股"
MIN_CATS = 10                       # 判定为含关联个股的旧版主机的阈值
KNOWN_OLD_HOSTS = ["180.153.18.170"]  # 已知含关联个股的旧版主机，优先直连
BEST_RETRY = 2                     # 仅当已知旧版主机全部失败时，再试 best-host 兜底（关联个股只在旧版主机上）
SOCK_TIMEOUT = 8.0                 # 单次连接超时：避免单主机 connect/recv 无限阻塞（默认 15s 太长）
OVERALL_TIMEOUT = 35.0             # 整体硬上限：超时即报错而非一直转圈（Streamlit 长驻进程尤需）
_query_lock = threading.Lock()     # 防止并发/重复点击堆叠后台线程

# 类型数字 -> 规范名称（用于模糊匹配）
_TYPE_ALIASES = {
    "1": "同大股东个股",
    "2": "同行业个股",
    "3": "股本相近个股",
    "4": "同地域个股",
}


def _infer_market(code):
    """根据 6 位代码前缀推断市场。"""
    code = str(code).zfill(6)
    head = code[:1]
    if head in ("6", "9"):
        return Market.SH
    if head in ("0", "3", "2"):
        return Market.SZ
    if head in ("8", "4"):
        return Market.BJ
    return Market.SH  # fallback


def _read_related(c, code, market):
    """在已连接 client 上读「关联个股」全文；无该分类返回 None。

    TDX 单次 get_company_info_content 有硬上限（实测声明 length 会远大于
    实际返回，约 2 万字封顶），因此按 chunk 递进多次读取拼全。
    """
    cats = c.get_company_info_category(market, code)
    if len(cats) < MIN_CATS:
        return None
    row = cats[cats["name"].str.contains(RELATED_KW, na=False)]
    if row.empty:
        return None
    r = row.iloc[0]
    start = int(r["start"])
    total = int(r["length"])
    return _read_full(c, market, code, str(r["filename"]), start, total)


def _read_full(c, market, code, filename, start, total):
    """分段多次读取拼出完整内容（绕过单次读取上限）。

    关键：每次请求「剩余全部」而非小 chunk。TDX 单次有硬上限（实测约 2 万
    字），请求剩余全部时服务器会返回尽量多的数据；若返回量 < 请求量，说明
    触达单次上限，offset 已推进，循环再补剩余部分即可拼全。
    """
    buf, off, guard = "", 0, 0
    while off < total and guard < 50:
        guard += 1
        n = total - off
        part = c.get_company_info_content(market, code, filename, start + off, n)
        if not part:
            break
        buf += part
        off += len(part)
        if len(part) >= n:   # 一次性拿全了
            break
        # 否则 len(part) < n：已达单次上限，循环继续补剩余
    return buf


def _fetch_content(code):
    """稳定拿到「关联个股」全文（旧版主机优先，少量 best-host 兜底）。

    TDX 不同主机/连接返回的 F10 完整度不同，故遍历已知旧版主机 +
    少量 from_best_host 兜底，**保留最长的一份** content 返回。若某次
    已拿到约 2 万字（接近单次上限、通常已是全量），提前结束以省时。
    每个连接都带显式 SOCK_TIMEOUT，死主机快速失败而非阻塞 15s。
    """
    market = _infer_market(code)
    best_content = ""
    pool = list(KNOWN_OLD_HOSTS) + ["__best__"] * BEST_RETRY
    for spec in pool:
        try:
            if spec == "__best__":
                with TdxClient.from_best_host(timeout=SOCK_TIMEOUT, ping_timeout=3.0) as c:
                    content = _read_related(c, code, market)
            else:
                with TdxClient(host=spec, timeout=SOCK_TIMEOUT) as c:
                    content = _read_related(c, code, market)
        except Exception:
            continue
        if content and len(content) > len(best_content):
            best_content = content
        if len(best_content) >= 20000:
            break
    if not best_content:
        raise RuntimeError(
            f"未能连上含「{RELATED_KW}」的旧版主机（code={code}）。"
            f"可补充 KNOWN_OLD_HOSTS 或稍后重试。")
    return best_content, market


def _split_sections(content):
    """按 【N.名称】 把全文切成 {num: {'title','text'}}。

    TDX 关联个股文本里 【N.名称】 标记会有「噪声」，必须处理两类：
      1) 开头一行「目录」(TOC)：【1.同大股东个股】【2.同行业个股】
         【3.股本相近个股】【4.同地域个股】★ —— 各标记紧挨、之间无内容。
      2) 文件末尾损坏的重复标记：真实章节之后又出现一次【3.…】【4.…】
         且其后紧跟乱码填充。
    策略：先跳过开头的 TOC 连续块，再对每个 num 只保留**第一次出现**
    的真实标记（损坏的重复标记在更后面，自然被去重丢掉）。
    """
    pat = re.compile(r"【\s*(\d+)\.\s*([^】]{1,40})】")
    ms = list(pat.finditer(content))
    if not ms:
        return {}

    # 1) 跳过开头 TOC：从首个标记起，若与下一个标记之间无换行且文本极短，
    #    视为同一 TOC 块，连续消耗；遇到有实质间隔的标记即停止。
    i = 0
    while i + 1 < len(ms):
        between = content[ms[i].end():ms[i + 1].start()]
        if "\n" not in between and len(between.strip()) < 3:
            i += 1
        else:
            break
    after_toc = ms[i + 1:] if i + 1 < len(ms) else []
    if not after_toc:          # 退化：完全没识别到真实章节，退回全量
        after_toc = ms

    # 2) 每个 num 只保留第一次出现（去掉文件末尾的损坏重复标记）
    seen, real = set(), []
    for m in after_toc:
        n = int(m.group(1))
        if n in seen:
            continue
        real.append(m)
        seen.add(n)

    sections = {}
    for ri, m in enumerate(real):
        num = int(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = real[ri + 1].start() if ri + 1 < len(real) else len(content)
        sections[num] = {"title": title, "text": content[start:end]}
    return sections


def _parse_table(text):
    """从某节文本解析表格，返回 (header, rows, meta)。

    一节内可能有**多张子表**（如同大股东个股下，每个大股东各一张表），
    因此这里收集所有「排名…证券代码」表头，逐张解析后合并，并追加「分组」
    列标明每张子表来源（取自该表头前最近的【…】（共N家）子标题）。
    无表头返回 (None, [], {})。
    """
    lines = text.splitlines()
    header_pos = [i for i, ln in enumerate(lines)
                  if "排名" in ln and "证券代码" in ln]
    if not header_pos:
        return None, [], {"date": None, "scope": None}

    def group_before(idx):
        """取该表头之前最近一个含「共N家」的子标题作为分组名。"""
        for k in range(idx - 1, -1, -1):
            sm = re.search(r"【?([^】]*?)】?[（(]共[^）)]*家[）)]", lines[k])
            if sm:
                return sm.group(1).strip() or lines[k].strip()
        return ""

    tables = []
    for hi, hpos in enumerate(header_pos):
        header = re.split(r"\s{2,}", lines[hpos].strip())
        sep = None
        for j in range(hpos + 1, min(hpos + 4, len(lines))):
            if "─" in lines[j]:
                sep = j
                break
        if sep is None:
            continue
        end = header_pos[hi + 1] if hi + 1 < len(header_pos) else len(lines)
        rows = []
        for ln in lines[sep + 1:end]:
            if "─" in ln:
                continue
            s = ln.strip()
            if not s:
                continue
            parts = re.split(r"\s{2,}", s)
            if parts and parts[0].isdigit():
                rows.append(parts)
        tables.append((header, rows, group_before(hpos)))

    if not tables:
        return None, [], {}

    base_header = tables[0][0]
    data = []
    for header, rows, grp in tables:
        for r in rows:
            if len(r) == len(base_header):
                data.append(r + [grp])
            elif len(r) > len(base_header):
                data.append(r[:len(base_header)] + [grp])
            else:
                data.append(r + [""] * (len(base_header) - len(r)) + [grp])
    full_header = base_header + ["分组"]

    # 元信息：截止日期 + 所有分组（；连接，反映「描述信息中有很多」）
    date = None
    for ln in lines:
        mm = re.search(r"截止日期[:：]\s*([\d\-]+)", ln)
        if mm:
            date = mm.group(1)
            break
    groups = sorted({t[2] for t in tables if t[2]})
    scope = "；".join(groups) if groups else None
    return full_header, data, {"date": date, "scope": scope}


def _resolve_type(rel_type):
    """把用户传入的类型规范化为数字。"""
    s = str(rel_type).strip()
    if s in _TYPE_ALIASES:
        return int(s), _TYPE_ALIASES[s]
    m = re.match(r"(\d+)", s)            # 取数字前缀，如 "2.同行业个股" -> 2
    if m:
        return int(m.group(1)), None
    for num, name in _TYPE_ALIASES.items():   # 子串匹配名称
        if name in s or s in name:
            return int(num), name
    raise ValueError(
        f"无法识别 rel_type={rel_type!r}，可选：{list(_TYPE_ALIASES.values())} 或 1~4")


def _get_related_df_inner(code, rel_type):
    """返回指定类型关联个股的 DataFrame（无整体超时，仅供 get_related_df 包一层）。

    参数
    ----
    code : str/int   6 位股票代码，如 "600199" / 600199
    rel_type : str      "1": "同大股东个股",
                        "2": "同行业个股",
                        "3": "股本相近个股",
                        "4": "同地域个股",

    返回
    ----
    pd.DataFrame，列=表头，行=各股票；df.attrs 含
        date   截止日期
        scope  范围标签（如「食品饮料--白酒Ⅱ--白酒Ⅲ（共21家）」/「安徽（共148家）」）
        section 节标题（如「同行业个股」）
        code   标准化后的 6 位代码
    """
    content, market = _fetch_content(code)
    sections = _split_sections(content)
    num, _ = _resolve_type(rel_type)
    if num not in sections:
        available = {n: s["title"] for n, s in sections.items()}
        raise KeyError(
            f"该股票关联个股中未找到第 {num} 类（{rel_type}）。可用：{available}")
    sec = sections[num]
    header, data, meta = _parse_table(sec["text"])
    if header is None:
        # 该节暂无数据（如「暂无数据」），返回空 DataFrame 而非抛错
        df = pd.DataFrame(columns=["排名", "证券代码", "股票名称", "分组"])
        df.attrs["empty"] = True
    else:
        df = pd.DataFrame(data, columns=header)
    if "排名" in df.columns:
        df["排名"] = pd.to_numeric(df["排名"], errors="coerce").astype("Int64")
    if "证券代码" in df.columns:
        df["证券代码"] = df["证券代码"].astype(str).str.zfill(6)
    df.attrs["date"] = meta["date"]
    df.attrs["scope"] = meta["scope"]
    df.attrs["section"] = sec["title"]
    df.attrs["code"] = str(code).zfill(6)
    return df


def get_related_df(code, rel_type):
    """对外入口：带整体硬超时的 get_related_df。

    背景：该函数会连通达信主机，而 TDX 服务器按来源 IP 限制连接数；在
    Streamlit 这类长期运行的服务里，偶发会连到被限速/暂时不可达的主机，
    导致 socket 长时间阻塞，且原重试逻辑（最多 20 次 from_best_host）会
    堆积成「几分钟的转圈」。这里用后台线程 + 整体超时兜底：超过
    OVERALL_TIMEOUT 秒仍未返回就直接抛 TimeoutError（由调用方捕获并提示），
    绝不让 UI 卡死。
    """
    if not _query_lock.acquire(blocking=False):
        raise RuntimeError("已有「关联个股」查询正在进行，请稍候再试。")
    try:
        box, err = {}, {}
        def _run():
            try:
                box["v"] = _get_related_df_inner(code, rel_type)
            except Exception as e:   # noqa: BLE001
                err["e"] = e
        th = threading.Thread(target=_run, daemon=True)
        th.start()
        th.join(OVERALL_TIMEOUT)
        if th.is_alive():
            raise TimeoutError(
                f"查询「关联个股」超时（>{OVERALL_TIMEOUT:.0f}s，code={code}）。"
                f"通达信主机可能暂时不可达或连接被限速，请稍后重试。")
        if "e" in err:
            raise err["e"]
        return box["v"]
    finally:
        _query_lock.release()