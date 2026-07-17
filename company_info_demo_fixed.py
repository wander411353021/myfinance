"""公司信息演示（适配新版通达信 F10：单条合并文档 + 【N.栏目名】分节）。

背景：通达信 F10 协议已改版。旧版 get_company_info_category 返回 16 条独立分类、
各自指向 .txt 不同 offset；新版只返回 1 条「最新提示」大文档，所有子栏目用
【1.最新提示】【2.互动问答】… 这类标记内嵌在正文里。

本脚本的做法：
  1. 取目录（兼容任意条数：旧版多条目 / 新版单条目都行）。
  2. 取每条目录项的正文全文。
  3. 用正则按【N.栏目名】把合并文档拆回子栏目，逐节打印。
  4. 若无分节标记，则整段打印（兜底）。
"""

import re
from easy_tdx import Market, TdxClient

CODE = "600519"
NAME = "贵州茅台"
MARKET = Market.SH

# 每节最多显示字数（设 0 表示全部显示，避免长表格被截断）
MAX_CHARS = 0


def split_subsections(text: str):
    """把合并后的 F10 大文档按【N.栏目名】拆成 [(栏目名, 内容), ...]。"""
    # 先剔除目录行（"★本栏包括【1.最新提示】…"），否则会被当成空分节
    lines = text.splitlines()
    body = "\n".join(ln for ln in lines if "本栏包括" not in ln)
    pat = re.compile(r"【\s*(\d+)\.\s*([^】]{1,30})】")
    matches = list(pat.finditer(body))
    if not matches:
        return [("全文", text)]
    out = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sec = body[start:end].strip()
        if not sec:
            continue
        out.append((m.group(2).strip(), sec))
    return out


with TdxClient.from_best_host() as c:
    cats = c.get_company_info_category(MARKET, CODE)
    print(f"{NAME} 公司信息目录 (共 {len(cats)} 条): {cats['name'].tolist()}")

    for _, r in cats.iterrows():
        content = c.get_company_info_content(
            MARKET, CODE, str(r["filename"]), int(r["start"]), int(r["length"])
        )
        print(f"\n########## 目录项: {r['name']} (共 {len(content)} 字) ##########")

        for sec_name, sec_text in split_subsections(content):
            body = sec_text.strip()
            shown = body if MAX_CHARS <= 0 or len(body) <= MAX_CHARS else body[:MAX_CHARS] + " ..."
            print(f"\n===== 【{sec_name}】 =====")
            print(shown)
