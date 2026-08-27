# -*- coding: utf-8 -*-
"""策略插件包:每个策略实现 generate_signals(df) -> [Signal, ...]。

Signal = {idx, date, meta}
  - idx: 信号在日线中的位置(整数)
  - date: 'YYYYMMDD'
  - meta: 附加信息(如 super/peak_ratio),用于可视化标注
"""
