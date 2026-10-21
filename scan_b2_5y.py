# -*- coding: utf-8 -*-
"""v3b_B2 全市场5年回溯 + gap 多档敏感性 (2026-10-21 polo4111)
信号判定复用 challenge_scan.v3b_signal_b2 逻辑(自实现循环以支持多gap档+统一冷却链)。
用法: python3 scan_b2_5y.py --part N --nparts M [--gaps "3,20;5,20;3,25;5,25;0,30"]
产物: result/b2_scan_shard{part}.csv (每信号一行, 含 gap_band 列)"""
import sys, os, argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
try:
    import fengwo; fengwo.showMsg(False)
except Exception: pass
from challenge_scan import load, compute_eff, MIN_I, V3B_COOL, V3B_HIST_LO, V3B_HIST_HI, V3B_REC_LO, V3B_REC_HI, V3B_REC_DIP, V3B_VOL_MULT

END='20261016'; WIN_START='20211001'; WIN_END='20261016'
CACHE='result/scan_cache_v3b'
WATCH={'300171','003040','300204','300251','300437','600199','600519','600371','601579','002343','000530','000993','002460','300299','300253','600737','600721','600613','688233','688388','600865','301080','603778','600550','001696','300006','688018','600152','301487','300619','002866','603278','002923','002441','301087','300675','300389'}

ap=argparse.ArgumentParser()
ap.add_argument('--part',type=int,required=True); ap.add_argument('--nparts',type=int,required=True)
ap.add_argument('--gaps',default='3,20;5,20;3,25;5,25;0,30')
a=ap.parse_args()
gaps=[tuple(map(float,g.split(','))) for g in a.gaps.split(';')]
print('gap档:',gaps,'part',a.part,'/',a.nparts,flush=True)

allf=[f[:-4] for f in os.listdir(CACHE) if f.endswith('.csv')]
allf.sort()
mine=allf[a.part::a.nparts]
print('本片票数:',len(mine),flush=True)

def ma20_pre(vol):
    n=len(vol); out=np.full(n,np.nan)
    cum=np.concatenate([[0],np.cumsum(vol)])
    for i in range(n):
        lo=max(0,i-20); out[i]=(cum[i]-cum[lo])/max(1,(i-lo))
    return out

rows=[]
t0=__import__('time').time()
for k,code in enumerate(mine):
    fp=os.path.join(CACHE,f'{code}.csv')
    if not os.path.exists(fp): continue
    try:
        df=pd.read_csv(fp)
        df['date']=pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
        if len(df)<MIN_I+130: continue
        cl,darr,gt_p,gt2,gtp_e,gt2_e,c35s,base,cA=compute_eff(df)
        vol=df['volume'].values.astype(float)
    except Exception as e:
        print('ERR',code,e,flush=True); continue
    n=len(cl); ma20=ma20_pre(vol)
    # 2026-10-21 polo4111: 冷却口径改"结构武装"(对齐旧v3b武装冷却):
    # 跌破 c35s×0.98 重新武装; 站上期间触发一次后武装失效, 不逐日/间隔卡死连续站上大牛
    armed=True
    for i in range(MIN_I,n):
        pi=gtp_e[i] if np.isfinite(gtp_e[i]) else gt_p[i]
        if not (np.isfinite(pi) and pi>0 and np.isfinite(cA[i]) and cA[i]>0): continue
        if cl[i] < c35s[i]*0.98: armed=True
        if not armed: continue
        if cl[i] <= pi: continue          # 非站上红压线日
        if cl[i] <= cA[i]: continue   # 线上确认
        lo,hi=max(MIN_I,i-V3B_HIST_LO),i-V3B_HIST_HI
        h_touch=False
        if lo<hi:
            pw=gt_p[lo:hi]
            h_touch=bool(np.any(np.isfinite(pw)&(pw>0)&(cl[lo:hi]>=pw)))
        if not h_touch: continue
        rlo,rhi=max(0,i-V3B_REC_LO),i-V3B_REC_HI
        r_dip=False
        if rlo<rhi:
            cw=cA[rlo:rhi]
            r_dip=bool(np.any(np.isfinite(cw)&(cw>0)&(cl[rlo:rhi]<cw*V3B_REC_DIP)))
        if not r_dip: continue
        if not (np.isfinite(ma20[i]) and ma20[i]>0 and vol[i]>=ma20[i]*V3B_VOL_MULT): continue
        gap=(cl[i]/pi-1)*100; dev_a=(cl[i]/cA[i]-1)*100
        d=darr[i]
        if not (WIN_START<=d<=WIN_END): continue
        # 后验
        f60=cl[i+1:min(i+61,n)]; f120=cl[i+1:min(i+121,n)]
        f60mx=(f60.max()/cl[i]-1)*100 if len(f60)>=20 else np.nan
        f120mx=(f120.max()/cl[i]-1)*100 if len(f120)>=60 else np.nan
        s60=int(np.isfinite(f60mx) and f60mx>=10); s120=int(np.isfinite(f120mx) and f120mx>=50)
        fail=int(np.isfinite(f60mx) and f60mx<5)
        _matched=False
        for gk,(gmin,gmax) in enumerate(gaps):
            if gmin<=gap<gmax:
                rows.append((code,d,round(float(cl[i]),2),gk,round(gap,1),round(dev_a,1),
                             round(f60mx,1) if np.isfinite(f60mx) else None,
                             round(f120mx,1) if np.isfinite(f120mx) else None,
                             s60,s120,fail,int(code in WATCH)))
                _matched=True
        # 与 challenge_scan.v3b_signal_b2 一致: gap 出档视为判定不成立, 不消耗武装
        if _matched: armed=False
    if k%200==0:
        el=(__import__('time').time()-t0)/60
        print(f'  [{a.part}] {k}/{len(mine)} {el:.1f}min',flush=True)

out=f'result/b2_scan_shard{a.part}.csv'
pd.DataFrame(rows,columns=['code','sig_date','sig_close','gap_band','gap','dev_a','fut60_max','fut120_max','s60','s120','fail','watch']).to_csv(out,index=False)
print(f'[DONE part{a.part}] {len(rows)} signals -> {out}',flush=True)
