# -*- coding: utf-8 -*-
"""通达信板块指数名称映射(2026-10-08 polo4111 抓取自 stockso/tqcenter get_sector_list 快照)
用法:
    from tdx_index_names import sector_name
    sector_name('sh881478')  -> '综合类'
    sector_name('881478')    -> '综合类'
    sector_name('sh880001')  -> None   # 缺失返回 None, 调用方自行降级
映射文件 tdx_block_names.json 587 条(880xxx 风格/概念/地区 + 881xxx 行业细分)。
881475/880001 等通达信统计/特殊指数不在列表内, 返回 None。
"""
import json
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAP_PATH = os.path.join(_THIS_DIR, 'tdx_block_names.json')

_cache = None


def _load():
    global _cache
    if _cache is None:
        try:
            with open(_MAP_PATH, encoding='utf-8') as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    return _cache


def sector_name(code: str):
    """code 形如 sh881478 / 881478 / 880948.SH, 返回名称或 None"""
    c = str(code).lower().strip()
    if c.startswith(('sh', 'sz', 'bj')):
        c = c[2:]
    c = c.split('.')[0]
    if not c.isdigit() or len(c) != 6:
        return None
    return _load().get(c)


def display_label(code: str) -> str:
    """画图标题/角标用: sh881478 -> 'sh881478 综合类'; 缺失 -> 'sh881478'"""
    c = str(code).lower().strip()
    c6 = c[2:] if c.startswith(('sh', 'sz', 'bj')) else c.split('.')[0]
    name = sector_name(c)
    return f"{c6} {name}" if name else c6
