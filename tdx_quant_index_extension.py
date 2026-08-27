# -*- coding: utf-8 -*-
"""
tdx_quant.py 大盘指数接口扩展 — 可直接追加到 tdx_quant.py 末尾

基于 pytdx.hq.TdxHq_API.get_index_bars，不依赖 eltdx 版本差异。
返回格式与 get_daily_kline_from_tdx 完全一致：
    DataFrame(date, open, high, low, close, volume)
"""

# ============================================================
# 大盘指数代码映射（别名 → 数字代码）
# 调用时可传别名，也可直接传数字代码
# ============================================================
INDEX_CODE_MAP = {
    # 宽基指数
    'sh':      '999999',   # 上证指数
    'sz':      '399001',   # 深证成指
    'hs300':   '000300',   # 沪深300（最常用作大盘趋势门控）
    'zz500':   '000905',   # 中证500
    'zz1000':  '000852',   # 中证1000
    'sz50':    '000016',   # 上证50
    'zzall':   '000985',   # 中证全指
    # 深市指数
    'cyb':     '399006',   # 创业板指
    'sz100':   '399330',   # 深证100
    'zxb':     '399005',   # 中小板指
    'cyb50':   '399673',   # 创业板50
    # 行业/主题指数（常用）
    'khb':     '880472',   # 科创板（通达信行业代码，非标准，慎用）
}

# K线周期 → pytdx category
_PERIOD_MAP = {
    '5min':    0,
    '15min':   1,
    '30min':   2,
    '60min':   3,
    'daily':   4,   # 日K（默认）
    'weekly':  5,   # 周K
    'monthly': 6,   # 月K
}

# 通达信行情服务器列表（多节点容错，按延迟排序）
_TDX_HOSTS = [
    ('119.147.212.81', 7709),   # 深圳招商
    ('112.74.214.43',  7709),   # 深圳阿里云
    ('221.231.141.60', 7709),   # 南京电信
    ('101.227.73.20',  7709),   # 上海电信
    ('14.215.128.18',  7709),   # 广州电信
    ('59.173.18.140',  7709),   # 武汉电信
]


def _infer_index_market(code):
    """根据指数代码推断市场：0=深圳, 1=上海
    399xxx → 深市；其余(999xxx, 000xxx, 880xxx) → 沪市
    """
    code = str(code)
    if code.startswith(('399', '159')):
        return 0  # 深市
    return 1      # 沪市


def get_index_kline_from_tdx(index_code, end_date=None, period='daily', count=800):
    """获取大盘指数K线（不复权，指数本身无复权概念）。

    Args:
        index_code: 指数代码，支持两种格式：
                    - 别名: 'sh' / 'sz' / 'hs300' / 'cyb' / 'zz500' ...
                    - 数字代码: '999999' / '399001' / '000300' ...
        end_date:   截止日期 'YYYYMMDD'，返回该日期及之前的K线
                    None 则返回最新 count 根
        period:     'daily' / 'weekly' / 'monthly' / '5min' / '15min' ...
        count:      获取K线数量（默认800根日K，约3年；指数数据量大可设更大）

    Returns:
        pd.DataFrame: 列 = [date, open, high, low, close, volume]
                      date 已归一化为 00:00:00，按日期升序
        失败返回 None

    用法示例:
        >>> df = get_index_kline_from_tdx('hs300', end_date='20260826')
        >>> print(df.tail())
        >>> # 大盘趋势门控：沪深300在250日均线之上才开仓
        >>> hs300_ma250 = df['close'].rolling(250).mean()
        >>> bull_market = df['close'].iloc[-1] > hs300_ma250.iloc[-1]
    """
    from pytdx.hq import TdxHq_API

    # 1. 别名解析
    code = INDEX_CODE_MAP.get(str(index_code).lower(), str(index_code))

    # 2. 市场推断
    market = _infer_index_market(code)

    # 3. 周期转换
    category = _PERIOD_MAP.get(period, 4)

    api = TdxHq_API()
    try:
        # 4. 多服务器容错连接
        connected = False
        for host, port in _TDX_HOSTS:
            try:
                if api.connect(host, port, time_out=3):
                    connected = True
                    break
            except Exception:
                continue
        if not connected:
            print(f'[tdx_quant] 指数 {code}({index_code}) 连接行情服务器失败')
            return None

        # 5. 获取指数K线（start=0 从最新开始往前取 count 根）
        data = api.get_index_bars(category, market, code, 0, count)
        if not data:
            print(f'[tdx_quant] 指数 {code}({index_code}) 无K线数据')
            return None

        # 6. 构造 DataFrame（pytdx 返回 dict 列表，字段: open/high/low/close/volume/datetime）
        df = pd.DataFrame({
            'date':   [d['datetime'] for d in data],
            'open':   [float(d['open'])   for d in data],
            'high':   [float(d['high'])   for d in data],
            'low':    [float(d['low'])    for d in data],
            'close':  [float(d['close'])  for d in data],
            'volume': [float(d['volume']) for d in data],
        })
        # 清洗：剔除异常K线，日期归一化，按日期升序
        df = df[df['close'] > 0].reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        df = df.sort_values('date').reset_index(drop=True)

        # 7. 按 end_date 截断（pytdx 不支持按日期取数，在此过滤）
        if end_date:
            end_dt = pd.to_datetime(str(end_date))
            df = df[df['date'] <= end_dt].reset_index(drop=True)

        return df

    except Exception as e:
        print(f'[tdx_quant] 获取指数 {code}({index_code}) K线异常: {e}')
        return None
    finally:
        try:
            api.disconnect()
        except Exception:
            pass


# ============================================================
# 便捷函数：大盘趋势判断（用于开仓门控）
# ============================================================
def is_bull_market(index_code='hs300', ma_window=250, end_date=None):
    """判断大盘是否处于多头市场（指数收盘价在 N 日均线之上）。

    用于黄金坑策略的大盘环境过滤：熊市中关闭信号，规避2023/2026年的低胜率。

    Args:
        index_code: 基准指数，默认 'hs300'（沪深300）
        ma_window:  均线窗口，默认250日（年线）
        end_date:   截止日期

    Returns:
        tuple: (is_bull: bool, index_close: float, ma_value: float)
               is_bull=True 表示多头市场，可开仓
    """
    df = get_index_kline_from_tdx(index_code, end_date=end_date, count=ma_window + 50)
    if df is None or len(df) < ma_window:
        return (None, None, None)  # 数据不足，无法判断

    close = df['close'].iloc[-1]
    ma = df['close'].rolling(ma_window).mean().iloc[-1]
    is_bull = close > ma
    return (is_bull, close, ma)


# ============================================================
# 自测（直接运行此文件时执行）
# ============================================================
if __name__ == '__main__':
    print("=== tdx_quant 大盘指数接口自测 ===")
    for alias in ['hs300', 'sh', 'cyb', 'zz500']:
        df = get_index_kline_from_tdx(alias, count=10)
        if df is not None and len(df) > 0:
            print(f"  {alias}({INDEX_CODE_MAP.get(alias)}): {len(df)}根, "
                  f"最新={df['date'].iloc[-1].strftime('%Y-%m-%d')} "
                  f"收盘={df['close'].iloc[-1]:.2f}")
        else:
            print(f"  {alias}: 获取失败")

    print("\n=== 大盘趋势判断 ===")
    bull, close, ma = is_bull_market('hs300', ma_window=250)
    if bull is not None:
        print(f"  沪深300: 收盘={close:.2f}, 250日均线={ma:.2f}, "
              f"{'多头市场(可开仓)' if bull else '空头市场(暂停信号)'}")
    else:
        print("  数据不足，无法判断")
