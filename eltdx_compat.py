# -*- coding: utf-8 -*-
"""eltdx bars.get 兼容封装(2026-10-08 polo4111)。
问题: modelscope 创空间等环境的 eltdx 版本较旧, bars.get 不支持 all_pages=True
      (报错: BarApi.get() got an unexpected keyword argument 'all_pages')。
方案: 新版走 all_pages(一次拉全); 旧版抛 TypeError 时自动降级为 start 分页拉取。
统一返回 bars 列表(时间可能乱序, 调用方需按日期升序排序)。
"""


def get_bars(cli, code, period='day', count=800, anchor_date=None, page_size=800):
    """兼容版 bars 拉取。返回 bars 列表。
    - 新版 eltdx: all_pages=True 一次拉全
    - 旧版 eltdx: 不支持 all_pages/anchor_date 时, 用 start 分页(最多 max_pages 次)"""
    max_pages = 200
    try:
        ks = cli.bars.get(code, period=period, count=count,
                          anchor_date=anchor_date, all_pages=True,
                          page_size=page_size)
        return list(ks.bars)
    except TypeError as e:
        msg = str(e)
        if 'all_pages' not in msg and 'anchor_date' not in msg:
            raise
        # 旧版: 逐个去掉不支持的参数重试
        kw = dict(period=period, count=count, start=0)
        try:
            ks = cli.bars.get(code, **kw)
            return list(ks.bars)
        except TypeError as e2:
            if 'start' not in str(e2):
                raise
            out, start = [], 0
            pages = 0
            while start < count and pages < max_pages:
                ks = cli.bars.get(code, period=period, start=start,
                                  count=min(page_size, count - start))
                bars = list(ks.bars)
                if not bars:
                    break
                out.extend(bars)
                pages += 1
                if len(bars) < page_size:
                    break
                start += len(bars)
            return out
