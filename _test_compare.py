"""对比 Oracle 和无未来函数版在多只股票上的表现"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from four_phase_visualizer import detect_four_phases_oracle, detect_four_phases

# 选近两年涨幅大的股票，取主升浪启动前约400天
tests = [
    # (code, end_date(主升前), reg_window, name)
    ('sz300059', '20240930', 120, '东方财富'),
    ('sz300124', '20240930', 120, '汇川技术'),
    ('sh600941', '20240830', 120, '中国移动'),
    ('sz000858', '20240930', 120, '五粮液'),
    ('sz300502', '20230131', 120, '新易盛'),
    ('sz002371', '20240930', 120, '北方华创'),
]

for code, end, rw, name in tests:
    print(f"\n{'='*70}")
    print(f"  {name} ({code})  end_date={end}")
    print(f"{'='*70}")
    
    o = detect_four_phases_oracle(code, end_date=end, reg_window=rw)
    n = detect_four_phases(code, end_date=end, reg_window=rw)
    
    if o:
        print(f"  Oracle:   S1={o['s1_start']}~{o['s1_end']}  "
              f"S2={o['s2_start']}~{o['s2_end']}({o['s2_high']})  "
              f"S3={o['s3_duration']}d  S4={o['s4_start']}")
    else:
        print(f"  Oracle:   NOT FOUND")
        
    if n:
        print(f"  NoFuture: S1={n['s1_start']}~{n['s1_end']}  "
              f"S2={n['s2_start']}~{n['s2_end']}({n['s2_high']})  "
              f"S3={n['s3_duration']}d  S4={n['s4_start']}")
    else:
        print(f"  NoFuture: NOT FOUND")
    
    if o and n:
        keys = ['s1_start','s1_end','s2_start','s2_end','s3_start','s3_end','s4_start']
        same = all(o[k]==n[k] for k in keys)
        print(f"  >>> 一致? {'YES - 无未来函数达到Oracle精度' if same else 'NO - 有差异'}")
        if not same:
            for k in keys:
                if o[k] != n[k]:
                    print(f"    差异 {k}: Oracle={o[k]}  NoFuture={n[k]}")
    elif o and not n:
        print(f"  >>> Oracle找到模式，但无未来函数版未检测到 (可能是假突破过滤)")
    elif not o and n:
        print(f"  >>> 无未来函数版找到模式，但Oracle过滤掉了")
