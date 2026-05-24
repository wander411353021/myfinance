import pandas as pd
import numpy as np
import time
import warnings
from juejing import get_stock_klines_from_juejing
from SuperSSF import ssf, compute_grid_coupling
warnings.filterwarnings('ignore', category=UserWarning)

def analyze_signals(code, end_date, tail_days, name):
    print(f'\n{"="*60}')
    print(f'正在分析 {name} (截止日期: {end_date})')
    print(f'{"="*60}')
    
    get_data_start = time.perf_counter()
    df = get_stock_klines_from_juejing(code, end_date, fqt=1)
    get_data_end = time.perf_counter()
    print(f'获取数据耗时: {get_data_end - get_data_start:.3f} 秒')
    print(f'数据总天数: {len(df)}')

    df = df[df['close'] > 0]

    df['pct_chg'] = df['close'].pct_change() * 100
    df['pct_chg'] = df['pct_chg'].round(2)
    df['ssf_s'] = ssf(close=df.close, length=30, poles=3)
    df['ssf_m'] = ssf(close=df.close, length=120, poles=3)
    df['ssf_m_20'] = df['ssf_m'] * 1.2
    df['ssf_l'] = ssf(close=df.close, length=250, poles=3)

    # 计算格栅线耦合
    grid_result = compute_grid_coupling(df)
    df['grid_coupling'] = grid_result

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    calc_end = time.perf_counter()
    print(f'计算指标耗时: {calc_end - get_data_end:.3f} 秒')
    
    # 分析信号
    plot_df = df.tail(tail_days)
    signals = plot_df[plot_df['grid_coupling'].notna()]
    
    print(f'\n信号统计（最近{tail_days}天）:')
    print(f'总信号数量: {len(signals)}')
    print(f'信号密度: {len(signals)/tail_days:.2%}')
    
    if len(signals) > 0:
        # 计算信号间隔
        signal_dates = signals.index
        intervals = []
        for i in range(1, len(signal_dates)):
            interval = (signal_dates[i] - signal_dates[i-1]).days
            intervals.append(interval)
        
        if intervals:
            print(f'信号间隔:')
            print(f'  平均间隔: {np.mean(intervals):.1f} 天')
            print(f'  最小间隔: {min(intervals)} 天')
            print(f'  最大间隔: {max(intervals)} 天')
        
        # 统计连续信号
        consecutive_groups = []
        if len(signals) > 0:
            current_group = [signal_dates[0]]
            for i in range(1, len(signal_dates)):
                if (signal_dates[i] - signal_dates[i-1]).days <= 5:  # 5天内算连续
                    current_group.append(signal_dates[i])
                else:
                    if current_group:
                        consecutive_groups.append(len(current_group))
                    current_group = [signal_dates[i]]
            if current_group:
                consecutive_groups.append(len(current_group))
        
        if consecutive_groups:
            print(f'连续信号组统计:')
            print(f'  组数: {len(consecutive_groups)}')
            print(f'  平均每组天数: {np.mean(consecutive_groups):.1f}')
            print(f'  最长连续: {max(consecutive_groups)} 天')
            print(f'  最短连续: {min(consecutive_groups)} 天')
        
        # 显示信号详情（前20个和后20个）
        print(f'\n信号详情（按时间）:')
        for idx, row in signals.head(20).iterrows():
            print(f'  {idx.date()}: 耦合值={row["grid_coupling"]:.2f}, 价格={row["close"]:.2f}, ssf_m={row["ssf_m"]:.2f}')
        
        if len(signals) > 20:
            print(f'  ... (省略中间 {len(signals)-40} 个信号) ...')
            for idx, row in signals.tail(20).iterrows():
                print(f'  {idx.date()}: 耦合值={row["grid_coupling"]:.2f}, 价格={row["close"]:.2f}, ssf_m={row["ssf_m"]:.2f}')
    
    # 保存详细数据
    output_file = f'{name}_signals_{end_date}.csv'
    signals.to_csv(output_file)
    print(f'\n信号数据已保存到: {output_file}')
    
    return signals, plot_df

# 分析600550在2025年2-10月期间
signals, plot_df = analyze_signals('600550', '20251010', 300, '600550')

print(f'\n{"="*60}')
print('分析完成！')
print(f'{"="*60}')