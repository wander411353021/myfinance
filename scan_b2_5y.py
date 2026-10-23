# -*- coding: utf-8 -*-
"""v3b_B2 全市场5年回溯 + B2-early 强势分支 (2026-10-21 polo4111, 合并 reasonix 3a7646a/156ad23)
信号判定复用 challenge_scan.v3b_signal_b2 逻辑(自实现循环以支持统一冷却链)。
正式 B2 严格口径: gap∈[5,20) + 史压 + 回踩 + 量能; B2-early 强势分支:
gap>20%(其余条件与正式B2相同, 对齐 reasonix strong_gap) 独立标记 kind='b2e'。
冷却: 正式B2 与 b2e 独立武装链 + 30日混合重武装(SIGNAL_REARM_DAYS)。
用法: python3 scan_b2_5y.py --part N --nparts M [--gaps "5,20"]
产物: result/b2_scan_shard{part}.csv (每信号一行, 含 gap_band/kind 列)"""
import sys, os, argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
try:
    import fengwo; fengwo.showMsg(False)
except Exception: pass
from challenge_scan import load, compute_eff, MIN_I, V3B_COOL, V3B_HIST_LO, V3B_HIST_HI, V3B_REC_LO, V3B_REC_HI, V3B_REC_DIP, V3B_VOL_MULT
from algo_config import SIGNAL_REARM_DAYS

END='20261016'; WIN_START='20211001'; WIN_END='20261016'
CACHE='result/scan_cache_v3b'
WATCH={'300171','003040','300204','300251','300437','600199','600519','600371','601579','002343','000530','000993','002460','300299','300253','600737','600721','600613','688233','688388','600865','301080','603778','600550','001696','300006','688018','600152','301487','300619','002866','603278','002923','002441','301087','300675','300389'}

ap=argparse.ArgumentParser()
ap.add_argument('--part',type=int,required=True); ap.add_argument('--nparts',type=int,required=True)
ap.add_argument('--gaps',default='5,20')
ap.add_argument('--max_pre_gain',type=float,default=999.0,
                help='信号日前20日低点→信号日收盘涨幅上限%%(超出即剔除), 默认999=不过滤(用户自判)')
a=ap.parse_args()
gaps=[tuple(map(float,g.split(','))) for g in a.gaps.split(';')]
b2e_gap_min=max(20.0, max(gmax for _, gmax in gaps))
print('gap档:',gaps,'b2e门槛:>%.1f%%'%b2e_gap_min,'part',a.part,'/',a.nparts,flush=True)

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
    # 冷却口径(2026-10-21 合并 reasonix 156ad23 + 独立链):
    # 正式 B2 与 B2-early 独立武装链(互不消耗, 对齐 reasonix scan_series 独立类型);
    # 跌破 c35s×0.98 或距上次同类信号>=SIGNAL_REARM_DAYS(30) 重新武装(混合重武装)
    armed_b2=True; armed_b2e=True; last_b2=-10**9; last_b2e=-10**9
    for i in range(MIN_I,n):
        pi=gtp_e[i] if np.isfinite(gtp_e[i]) else gt_p[i]
        if not (np.isfinite(pi) and pi>0 and np.isfinite(cA[i]) and cA[i]>0): continue
        if cl[i] < c35s[i]*0.98:
            armed_b2=True; armed_b2e=True
        else:
            if i-last_b2>=SIGNAL_REARM_DAYS: armed_b2=True
            if i-last_b2e>=SIGNAL_REARM_DAYS: armed_b2e=True
        if cl[i] <= pi: continue          # 非站上红压线日
        if cl[i] <= cA[i]: continue   # 线上确认
        gap=(cl[i]/pi-1)*100
        # 史压 [T-250,T-20] 曾站上红压线 (正式 B2 与 b2e 均要求, 对齐 reasonix strong_gap)
        lo,hi=max(MIN_I,i-V3B_HIST_LO),i-V3B_HIST_HI
        h_touch=False
        if lo<hi:
            pw=gt_p[lo:hi]
            h_touch=bool(np.any(np.isfinite(pw)&(pw>0)&(cl[lo:hi]>=pw)))
        # 回踩 [T-70,T-5] 跌破主成本线A×0.98 (正式 B2 与 b2e 均要求)
        rlo,rhi=max(0,i-V3B_REC_LO),i-V3B_REC_HI
        r_dip=False
        if rlo<rhi:
            cw=cA[rlo:rhi]
            r_dip=bool(np.any(np.isfinite(cw)&(cw>0)&(cl[rlo:rhi]<cw*V3B_REC_DIP)))
        # 量能确认(两类型均要求)
        if not (np.isfinite(ma20[i]) and ma20[i]>0 and vol[i]>=ma20[i]*V3B_VOL_MULT): continue
        dev_a=(cl[i]/cA[i]-1)*100
        pre_gain=(cl[i]/cl[max(0,i-20):i].min()-1)*100 if i>=20 and cl[max(0,i-20):i].min()>0 else np.nan
        d=darr[i]
        # 命中判定(先消耗武装: 窗口外命中同样消耗, 与 scan_series 全历史推进一致)
        hit_bands=[gk for gk,(gmin,gmax) in enumerate(gaps) if gmin<=gap<gmax]
        hit_b2 = armed_b2 and h_touch and r_dip and bool(hit_bands)
        hit_b2e = armed_b2e and h_touch and r_dip and gap>b2e_gap_min
        if hit_b2: armed_b2=False; last_b2=i
        if hit_b2e: armed_b2e=False; last_b2e=i
        if not (WIN_START<=d<=WIN_END): continue
        # 涨幅过大剔除(2026-10-22 polo4111: 前20日已涨>max_pre_gain 的信号不列出——追高不可买)
        if np.isfinite(pre_gain) and pre_gain>a.max_pre_gain: continue
        # 后验(公共计算一次)
        f60=cl[i+1:min(i+61,n)]; f120=cl[i+1:min(i+121,n)]
        f60mx=(f60.max()/cl[i]-1)*100 if len(f60)>=20 else np.nan
        f120mx=(f120.max()/cl[i]-1)*100 if len(f120)>=60 else np.nan
        s60=int(np.isfinite(f60mx) and f60mx>=10); s120=int(np.isfinite(f120mx) and f120mx>=50)
        fail=int(np.isfinite(f60mx) and f60mx<5)
        # 正式 B2: 每个命中的 gap 档各输出一行（支持 --gaps 多档敏感性）
        if hit_b2:
            for gk in hit_bands:
                rows.append((code,d,round(float(cl[i]),2),gk,round(gap,1),round(dev_a,1),
                             round(pre_gain,1) if np.isfinite(pre_gain) else None,
                             'b2', round(f60mx,1) if np.isfinite(f60mx) else None,
                             round(f120mx,1) if np.isfinite(f120mx) else None,
                             s60,s120,fail,int(code in WATCH)))
        # B2-early: gap>max(20, 最大档上界) + 史压 + 回踩 (独立武装链, 不与正式 B2 重复)
        if hit_b2e:
            rows.append((code,d,round(float(cl[i]),2),-1,round(gap,1),round(dev_a,1),
                         round(pre_gain,1) if np.isfinite(pre_gain) else None,
                         'b2e', round(f60mx,1) if np.isfinite(f60mx) else None,
                         round(f120mx,1) if np.isfinite(f120mx) else None,
                         s60,s120,fail,int(code in WATCH)))
    if k%200==0:
        el=(__import__('time').time()-t0)/60
        print(f'  [{a.part}] {k}/{len(mine)} {el:.1f}min',flush=True)

out=f'result/b2_scan_shard{a.part}.csv'
pd.DataFrame(rows,columns=['code','sig_date','sig_close','gap_band','gap','dev_a','pre_gain','kind','fut60_max','fut120_max','s60','s120','fail','watch']).to_csv(out,index=False)
print(f'[DONE part{a.part}] {len(rows)} signals -> {out}',flush=True)
