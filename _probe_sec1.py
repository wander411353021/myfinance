"""探测 600519 关联个股四节的实际长度，定位第1节为何为空。"""
import re
from easy_tdx import Market, TdxClient

CODE = "600519"
MARKET = Market.SH
with TdxClient(host="180.153.18.170") as c:
    cats = c.get_company_info_category(MARKET, CODE)
    print("cats len:", len(cats), "names:", cats["name"].tolist())
    row = cats[cats["name"].str.contains("关联个股", na=False)]
    print("matched rows:", len(row))
    r = row.iloc[0]
    print("filename/start/length:", r["filename"], r["start"], r["length"])
    content = c.get_company_info_content(
        MARKET, CODE, str(r["filename"]), int(r["start"]), int(r["length"]))
    print("content len:", len(content))

pat = re.compile(r"【\s*(\d+)\.\s*([^】]{1,40})】")
ms = list(pat.finditer(content))
print(f"\n找到 {len(ms)} 个分节标记:")
for m in ms:
    num = int(m.group(1))
    title = m.group(2)
    print(f"  pos={m.start()}  num={num}  title={title!r}")

print("\n===== 各节长度 =====")
for i, m in enumerate(ms):
    start = m.end()
    end = ms[i + 1].start() if i + 1 < len(ms) else len(content)
    print(f"  第{int(m.group(1))}节 [{m.group(2)}]: {end - start} 字")

# 直接看第一个 【1. 之后的前 300 字
m1 = ms[0]
print("\n===== 第1节开头 300 字 =====")
print(content[m1.end():m1.end()+300])
