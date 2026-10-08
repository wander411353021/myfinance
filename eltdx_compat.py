# -*- coding: utf-8 -*-
"""eltdx bars.get 兼容封装(2026-10-08 polo4111)。
问题1: modelscope 创空间等环境的 eltdx 版本较旧, bars.get 不支持 all_pages=True
      (报错: BarApi.get() got an unexpected keyword argument 'all_pages')。
问题2: all_pages 会拉全量历史(概念板块普遍 5000+ 根), 筹码逐日递推计算随长度变慢。
方案: 有 anchor_date(锚定历史, 回测/历史画图) -> 新版 all_pages, 旧版降级分页;
      无 anchor_date(页面/扫描默认, 取最近 N 根) -> 统一 start 分页精确拉 count 根,
      800 根只拉 1 页, 计算量可控。
统一返回 bars 列表(时间倒序, 调用方需按日期升序排序)。
"""
import math


def get_bars(cli, code, period='day', count=800, anchor_date=None, page_size=800):
    """兼容版 bars 拉取, 返回 bars 列表(倒序)。
    - anchor_date 非空: 锚定历史用 all_pages(新版), 旧版 TypeError 降级 start 分页
    - anchor_date 为空: 统一 start 分页, 精确取最近 count 根(最快)"""
    if anchor_date:
        try:
            ks = cli.bars.get(code, period=period, count=count,
                              anchor_date=anchor_date, all_pages=True,
                              page_size=page_size)
            return list(ks.bars)
        except TypeError as e:
            msg = str(e)
            if 'all_pages' not in msg and 'anchor_date' not in msg:
                raise
        # 旧版: 不支持 anchor_date/all_pages, 用 start 分页拉最近 count 根(无法锚定历史)
    out, start, pages = [], 0, 0
    max_pages = math.ceil(count / page_size) + 1
    while len(out) < count and pages < max_pages:
        ks = cli.bars.get(code, period=period, start=start,
                          count=min(page_size, count - len(out)))
        bars = list(ks.bars)
        if not bars:
            break
        out.extend(bars)
        pages += 1
        if len(bars) < page_size:
            break
        start += len(bars)
    return out
