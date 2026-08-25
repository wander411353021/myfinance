# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
import akshare as ak
import signal
import time
import efinance as ef
import os

# 可以直接提取数据，掘金终端需要打开，接口取数是通过网络请求的方式，效率一般，行情数据可通过subscribe订阅方式
# 设置token， 查看已有token ID,在用户-秘钥管理里获取
set_token('657134ea03be7e792951bcb3baaffc104205d0f1')


class TimeoutError(Exception):
    """超时异常"""
    pass


def timeout_handler(signum, frame):
    """超时信号处理器"""
    raise TimeoutError("网络请求超时")

#从掘金获取K线数据
#fqt=0表示不复权 1表示前复权，2表示后复权
def get_stock_klines_from_juejing(code,end_date="20250101",fqt=0, max_retries=3, timeout=30):
    """
    从掘金获取K线数据，带超时和重试机制
    
    参数:
        code: 股票代码
        end_date: 结束日期
        fqt: 复权类型 (0=不复权, 1=前复权, 2=后复权)
        max_retries: 最大重试次数
        timeout: 超时时间（秒）
    
    返回:
        DataFrame 或 None
    """
    symbol = ""
    if code[0] >= "8" or code[0] == "4":
        print(f"不支持非上交所和深交所的股票{code}")
        return None
    if code[0] == "6":
        symbol = f"SHSE.{code}"
    else:
        symbol = f"SZSE.{code}"
    end_time  = end_date[0:4] + "-" + end_date[4:6] + "-" + end_date[6:8]
    adjust=ADJUST_NONE
    if fqt == 1:
        adjust = ADJUST_PREV
    elif fqt == 2:
        adjust = ADJUST_POST
    
    # 带超时和重试的请求
    for attempt in range(max_retries):
        try:
            print(f"正在获取 {code} 的数据 (尝试 {attempt + 1}/{max_retries})...")
            
            # 使用线程和超时机制（Windows兼容）
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
            
            def fetch_data():
                return history(
                    symbol=symbol, 
                    frequency='1d', 
                    start_time='2015-07-28',  
                    end_time=end_time, 
                    fields='open, close, low, high, volume, eob', 
                    adjust=adjust, 
                    df=True
                )
            
            # 使用线程池执行，设置超时
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fetch_data)
                history_data = future.result(timeout=timeout)
            
            if history_data is None or len(history_data) == 0:
                print(f"获取 {code} 数据失败: 无数据返回")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                return None
            
            # 处理数据
            history_data['date'] = pd.to_datetime(history_data['eob'], unit='ns').dt.strftime('%Y-%m-%d')
            history_data['datetime'] = pd.to_datetime(history_data['date'])
            history_data.set_index('datetime', inplace=True)
            history_data['pct_chg'] = history_data['close'].pct_change() * 100
            
            print(f"成功获取 {code} 的 {len(history_data)} 条数据")
            return history_data
            
        except FutureTimeoutError:
            print(f"获取 {code} 数据超时 ({timeout}秒)")
            if attempt < max_retries - 1:
                print(f"等待 {2 ** attempt} 秒后重试...")
                time.sleep(2 ** attempt)  # 指数退避
                continue
            print(f"获取 {code} 数据失败: 超时（已重试 {max_retries} 次）")
            return None
            
        except Exception as e:
            print(f"获取 {code} 数据失败: {str(e)}")
            if attempt < max_retries - 1:
                print(f"等待 {2 ** attempt} 秒后重试...")
                time.sleep(2 ** attempt)  # 指数退避
                continue
            return None
    
    return None

def get_stock_15min_klines_from_juejing(code,end_date="20251231",fqt=0):
    symbol = ""
    if code[0] >= "8" or code[0] == "4":
        print(f"不支持非上交所和深交所的股票{code}")
        return None
    if code[0] == "6":
        symbol = f"SHSE.{code}"
    else:
        symbol = f"SZSE.{code}"
    end_time  = end_date[0:4] + "-" + end_date[4:6] + "-" + end_date[6:8]
    adjust=ADJUST_NONE
    if fqt == 1:
        adjust = ADJUST_PREV
    elif fqt == 2:
        adjust = ADJUST_POST
    
    history_data = history_n(symbol=symbol, count=1200, frequency='900s', end_time=end_time, fields='open, close, low, high, volume, eob', adjust=adjust, df= True)
    history_data['date'] = pd.to_datetime(history_data['eob'], unit='ns').dt.strftime('%Y-%m-%d')
    history_data['datetime'] = pd.to_datetime(history_data['date'])
    history_data.set_index('datetime', inplace=True)
    history_data['pct_chg'] = history_data['close'].pct_change() * 100
    return history_data

def get_industry():
    industry_df = ak.stock_board_industry_name_em()
    industry = industry_df[["板块名称", "板块代码", "涨跌幅", "上涨家数"]]
    return industry

def get_industry_stocks(symbol):
    # get_industry_stocks("半导体")
    industry_cons_df = ak.stock_board_industry_cons_em(symbol=symbol)
    industry_mem = industry_cons_df[["代码","名称","涨跌幅", "振幅", "换手率"]]
    return industry_mem


def update_all_stock_info():
    df = ef.stock.get_realtime_quotes()
    all_stocks = df[["股票代码", "股票名称"]]
    all_stocks.columns = ["code", "name"]
    # 1. 转为字符串并补足 6 位（避免少位数问题）
    all_stocks["code"] = all_stocks["code"].astype(str).str.zfill(6)
    # 2. 过滤：只保留以 00、30、60、68 开头的股票
    #    （即主板、创业板、科创板）
    mask = (all_stocks["code"].str.startswith(('00', '30', '60', '68')))
    all_stocks = all_stocks[mask].copy()
    # 3. 根据代码前缀设定涨跌幅限制
    def get_limit(code):
        if code.startswith(('00', '60')):
            return 10   # 主板 10%
        elif code.startswith(('30', '68')):
            return 20   # 创业板/科创板 20%
        else:
            return None # 理论上不会执行，因为已过滤
    all_stocks["limit"] = all_stocks["code"].apply(get_limit)
    # 4. 按 code 升序排序
    all_stocks = all_stocks.sort_values(by="code", ascending=True)
    # 5. 保存到 CSV（code 列会以字符串形式保存，带前导零）
    all_stocks.to_csv("./result/all_stocks.csv", index=False)

def get_all_stock_info():
    if not os.path.exists("./result/all_stocks.csv"):
        update_all_stock_info()
    # 读取时明确指定 code 为字符串，防止被解析为数字
    all_stocks = pd.read_csv("./result/all_stocks.csv", dtype={"code": str})
    return all_stocks