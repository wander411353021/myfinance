"""
Price Segmentation V10 — 逐层突破
==================================

V10 基于 V9，核心变化：
  - 买点改为"逐层突破"：追踪所有未突破的前UP区高点，每个被突破都发信号
  - 卖点对称：追踪所有未跌破的前DOWN区低点
  - 废弃 V9 的 max(last_3) 逻辑（下降趋势中远古高点卡住信号的问题）

信号体系:
  +1 (买入): BrkLvl — 突破一个前UP区高点（逐层突破，带 0~1 突破分量评分）
  +1 (买入): BrkRes — 突破之前触碰过的压力位
  +1 (买入): PullSup — UP待定区中回踩支撑
   0 (中性)
  -1 (卖出): BrkLow — 跌破一个前DOWN区低点（对称，带分量评分）
  -1 (卖出): BrkSup — 跌破之前触碰过的支撑位
  -1 (卖出): BncRes — DOWN待定区中反弹到阻力
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 共享工具函数（从 V9 沿用）
# ============================================================
def _compute_rolling_percentile(log_vol, ground_pct, sky_pct, rolling_window):
    log_s = pd.Series(log_vol)
    ground_thresh = (log_s.rolling(rolling_window, min_periods=20).quantile(ground_pct / 100).values.copy())
    sky_thresh = (log_s.rolling(rolling_window, min_periods=20).quantile(sky_pct / 100).values.copy())
    fv_g = np.where(~np.isnan(ground_thresh))[0]; fv_s = np.where(~np.isnan(sky_thresh))[0]
    if len(fv_g) > 0: ground_thresh[:fv_g[0]] = ground_thresh[fv_g[0]]
    if len(fv_s) > 0: sky_thresh[:fv_s[0]] = sky_thresh[fv_s[0]]
    return ground_thresh, sky_thresh

def _build_price_result(close, smooth, phase_id, phase_name, pivots,
                        is_pending=None, pending_confidence=None, vol_annotation=None):
    n = len(close)
    is_pivot = np.zeros(n, dtype=bool); pivot_type = np.array([""] * n, dtype='U8')
    for p in pivots:
        if p[0] < n: is_pivot[p[0]] = True; pivot_type[p[0]] = p[1]
    if is_pending is None: is_pending = np.zeros(n, dtype=bool)
    if pending_confidence is None: pending_confidence = np.zeros(n, dtype=float)
    if vol_annotation is None: vol_annotation = np.array(["NEUTRAL"] * n, dtype='U14')
    result = pd.DataFrame({"close":close,"smooth":smooth,"phase":phase_name,"phase_id":phase_id,
        "is_pivot":is_pivot,"pivot_type":pivot_type,"is_pending":is_pending,
        "pending_confidence":pending_confidence,"vol_annotation":vol_annotation,
        "touch_signal":np.zeros(n,dtype=int),"touch_source":np.array([""]*n,dtype='U20')})
    result.attrs["pivots"] = list(pivots)
    return result

class FutureLookingPriceSegmenter:
    """上帝视角分段器 — 有未来函数，仅基准对照。"""
    def __init__(self, sg_window=11, sg_poly=3, peak_distance=3, min_reversal_pct=0.02):
        self.sg_window=sg_window; self.sg_poly=sg_poly
        self.peak_distance=peak_distance; self.min_reversal_pct=min_reversal_pct
    def segment(self, close):
        close=np.asarray(close,float); n=len(close)
        smooth = close.copy() if n<self.sg_window else savgol_filter(close,self.sg_window,self.sg_poly)
        peaks,_=find_peaks(smooth,distance=self.peak_distance); troughs,_=find_peaks(-smooth,distance=self.peak_distance)
        pivots=[]
        for idx in peaks: pivots.append((idx,"PEAK"))
        for idx in troughs: pivots.append((idx,"TROUGH"))
        pivots.sort(key=lambda x:x[0])
        filtered=[]
        for p in pivots:
            if not filtered: filtered.append(p); continue
            li,lt=filtered[-1]; ci,ct=p
            if ct==lt:
                if ct=="PEAK" and smooth[ci]>smooth[li]: filtered[-1]=p
                elif smooth[ci]<smooth[li]: filtered[-1]=p; continue
            rev=abs(smooth[ci]-smooth[li])/smooth[li]
            if rev>=self.min_reversal_pct: filtered.append(p)
            elif ct=="PEAK" and smooth[ci]>smooth[li]: filtered[-1]=p
            elif ct=="TROUGH" and smooth[ci]<smooth[li]: filtered[-1]=p
        pivots=filtered
        pid=np.zeros(n,int); pnm=np.array(["NEUTRAL"]*n,dtype='U8')
        if not pivots: return _build_price_result(close,smooth,pid,pnm,pivots)
        ft=pivots[0][1]
        if ft=="PEAK": pid[:pivots[0][0]]=1; pnm[:pivots[0][0]]="UP"
        else: pid[:pivots[0][0]]=-1; pnm[:pivots[0][0]]="DOWN"
        for i in range(len(pivots)-1):
            si,ei=pivots[i][0],pivots[i+1][0]
            if pivots[i][1]=="TROUGH" and pivots[i+1][1]=="PEAK": pid[si:ei+1]=1; pnm[si:ei+1]="UP"
            elif pivots[i][1]=="PEAK" and pivots[i+1][1]=="TROUGH": pid[si:ei+1]=-1; pnm[si:ei+1]="DOWN"
            else:
                mid=(si+ei)//2
                if smooth[ei]>smooth[si]: pid[si:mid+1]=1; pnm[si:mid+1]="UP"; pid[mid+1:ei+1]=-1; pnm[mid+1:ei+1]="DOWN"
                else: pid[si:mid+1]=-1; pnm[si:mid+1]="DOWN"; pid[mid+1:ei+1]=1; pnm[mid+1:ei+1]="UP"
        lt=pivots[-1][1]
        if lt=="TROUGH": pid[pivots[-1][0]:]=1; pnm[pivots[-1][0]:]="UP"
        else: pid[pivots[-1][0]:]=-1; pnm[pivots[-1][0]:]="DOWN"
        return _build_price_result(close,smooth,pid,pnm,pivots)

class CausalIncrementalPriceSegmenter:
    """因果增量式分段器 — 无未来函数。"""
    def __init__(self, lookback=15, min_reversal_pct=0.02, confirm_bars=3, ema_span=15,
                 ground_pct=20, sky_pct=85, rolling_window=120, same_type_merge_gap=20):
        self.lookback=lookback; self.min_reversal_pct=min_reversal_pct
        self.confirm_bars=confirm_bars; self.ema_span=ema_span
        self.ground_pct=ground_pct; self.sky_pct=sky_pct; self.rolling_window=rolling_window
        self.same_type_merge_gap=same_type_merge_gap

    def segment(self, close, volume=None, high=None, low=None, opn=None):
        close=np.asarray(close,float); n=len(close)
        volume=np.asarray(volume,float) if volume is not None else None
        high=np.asarray(high,float) if high is not None else close
        low=np.asarray(low,float) if low is not None else close
        opn=np.asarray(opn,float) if opn is not None else close
        candidates=self._detect_candidates(close)
        confirmed_pivots=self._confirm_pivots(close,candidates)
        phase_id,phase_name,is_pending,pending_confidence=self._assign_phases(n,confirmed_pivots,close)
        vol_annotation=np.array(["NEUTRAL"]*n,dtype='U14')
        if volume is not None: vol_annotation=self._annotate_volume(volume)
        smooth=self._ema_close(close)
        touch_signal,touch_source=self._compute_touch_signal(close,high,low,opn,volume,n,confirmed_pivots)
        is_pivot=np.zeros(n,bool); pivot_type=np.array([""]*n,dtype='U8')
        for p in confirmed_pivots:
            if p[0]<n: is_pivot[p[0]]=True; pivot_type[p[0]]=p[1]
        result=pd.DataFrame({"close":close,"smooth":smooth,"phase":phase_name,"phase_id":phase_id,
            "is_pivot":is_pivot,"pivot_type":pivot_type,"is_pending":is_pending,
            "pending_confidence":pending_confidence,"vol_annotation":vol_annotation,
            "touch_signal":touch_signal,"touch_source":touch_source})
        result.attrs["pivots"]=list(confirmed_pivots)
        return result

    def _ema_close(self, close):
        n=len(close); s=np.zeros(n); a=2.0/(self.ema_span+1); s[0]=close[0]
        for i in range(1,n): s[i]=a*close[i]+(1-a)*s[i-1]
        return s

    def _detect_candidates(self, close):
        n=len(close); candidates=[]
        for t in range(self.lookback,n):
            w=close[t-self.lookback:t+1]
            if close[t]==w.max() and close[t]>close[t-1]:
                if not candidates or candidates[-1][1]!="PEAK" or candidates[-1][0]<t-1: candidates.append((t,"PEAK"))
                elif close[t]>=close[candidates[-1][0]]: candidates[-1]=(t,"PEAK")
            if close[t]==w.min() and close[t]<close[t-1]:
                if not candidates or candidates[-1][1]!="TROUGH" or candidates[-1][0]<t-1: candidates.append((t,"TROUGH"))
                elif close[t]<=close[candidates[-1][0]]: candidates[-1]=(t,"TROUGH")
        return candidates

    def _confirm_pivots(self, close, candidates):
        n=len(close); confirmed=[]
        for ci,ct in candidates:
            cft=None
            for t in range(ci+self.confirm_bars,n):
                if ct=="PEAK" and close[t]<=close[ci]*(1-self.min_reversal_pct): cft=t; break
                if ct=="TROUGH" and close[t]>=close[ci]*(1+self.min_reversal_pct): cft=t; break
            if cft is not None: confirmed.append((ci,ct,cft))
        filtered=[]
        for p in confirmed:
            filtered.append(p)
            # 反复合并：look-back 查找同类型 pivot，即使中间夹着异类型的也穿透合并
            while True:
                nf=len(filtered)
                if nf<2: break
                merged=False
                for j in range(nf-2,-1,-1):
                    if filtered[j][1]!=filtered[-1][1]:
                        continue
                    if abs(filtered[-1][0]-filtered[j][0])>self.same_type_merge_gap:
                        break  # 最近的同类型已超阈值，更远的只会更大
                    # 找到 ≤gap 的同类型：保留更极端的，移除 j 之后所有 pivot
                    ct=filtered[-1][1]
                    if ct=="PEAK":
                        if close[filtered[-1][0]]>close[filtered[j][0]]:
                            filtered=filtered[:j]+[filtered[-1]]
                        else:
                            filtered=filtered[:j+1]
                    else:  # TROUGH
                        if close[filtered[-1][0]]<close[filtered[j][0]]:
                            filtered=filtered[:j]+[filtered[-1]]
                        else:
                            filtered=filtered[:j+1]
                    merged=True
                    break
                if not merged:
                    break
        return filtered

    def _assign_phases(self, n, pivots, close):
        pid=np.zeros(n,int); pnm=np.array(["NEUTRAL"]*n,dtype='U8')
        ip=np.zeros(n,bool); pc=np.zeros(n,float)
        if not pivots: return pid,pnm,ip,pc
        sbi=sorted(pivots,key=lambda x:x[0])
        for t in range(n):
            vis=[(pi,pt) for pi,pt,pc in sbi if pc<=t]; vis.sort(key=lambda x:x[0])
            if not vis: continue
            if len(vis)==1:
                pi,pt=vis[0]
                if pt=="TROUGH": pid[t]=1; pnm[t]="UP"
                else: pid[t]=-1; pnm[t]="DOWN"
                if t<vis[0][0]: ip[t]=True; continue
            assigned=False
            for si in range(len(vis)-1):
                sidx,st=vis[si]; eidx,et=vis[si+1]
                if sidx<=t<=eidx:
                    if st=="TROUGH" and et=="PEAK": pid[t]=1; pnm[t]="UP"
                    elif st=="PEAK" and et=="TROUGH": pid[t]=-1; pnm[t]="DOWN"
                    else:
                        mid=(sidx+eidx)//2
                        if close[eidx]>close[sidx]:
                            pid[t]=1 if t<=mid else -1; pnm[t]="UP" if t<=mid else "DOWN"
                        else: pid[t]=-1 if t<=mid else 1; pnm[t]="DOWN" if t<=mid else "UP"
                    assigned=True; break
            if not assigned:
                lp=vis[-1]; pi,pt=lp; lpp=close[pi]
                # 死区缓冲：用 min_reversal_pct 做阈值。幅度不足最小反转的穿越
                # 不应触发 phase 翻转，避免 pending 区域产生 1-2 天的碎片 zone
                buf = self.min_reversal_pct
                upper, lower = lpp * (1 + buf), lpp * (1 - buf)
                if pt=="PEAK":
                    if close[t]>upper: pid[t]=1; pnm[t]="UP"
                    elif close[t]<lower: pid[t]=-1; pnm[t]="DOWN"
                    elif t>0: pid[t]=pid[t-1]; pnm[t]=pnm[t-1]
                    else: pid[t]=0; pnm[t]="NEUTRAL"
                else:
                    if close[t]<lower: pid[t]=-1; pnm[t]="DOWN"
                    elif close[t]>upper: pid[t]=1; pnm[t]="UP"
                    elif t>0: pid[t]=pid[t-1]; pnm[t]=pnm[t-1]
                    else: pid[t]=0; pnm[t]="NEUTRAL"
                ip[t]=True
                move=(close[t]-lpp)/lpp if pnm[t]=="UP" else (lpp-close[t])/lpp
                pc[t]=min(1.0,max(0.0,move/self.min_reversal_pct))
        return pid,pnm,ip,pc

    def _annotate_volume(self, volume):
        n=len(volume); lv=np.log1p(volume.astype(float)); a=2.0/(self.ema_span+1)
        sm=np.zeros(n); sm[0]=lv[0]
        for i in range(1,n): sm[i]=a*lv[i]+(1-a)*sm[i-1]
        gt,_=_compute_rolling_percentile(lv,self.ground_pct,self.sky_pct,self.rolling_window)
        ann=np.array(["NEUTRAL"]*n,dtype='U14')
        for i in range(1,n):
            if sm[i]>sm[i-1] and lv[i]>gt[i]: ann[i]="VOL_EXPANDING"
            elif sm[i]<sm[i-1] or lv[i]<=gt[i]: ann[i]="VOL_SHRINKING"
        return ann

    def _compute_touch_signal(self, close, high, low, opn, volume, n, confirmed_pivots):
        tt=0.005; at=0.05; gma=10
        ts=np.zeros(n,int); tsrc=np.array([""]*n,dtype='U20')
        if not confirmed_pivots: return ts,tsrc
        sp=sorted(confirmed_pivots,key=lambda x:x[0])
        all_gaps=[]
        for k in range(1,n):
            if low[k]>high[k-1]: all_gaps.append((k,low[k],high[k-1],True))
            if high[k]<low[k-1]: all_gaps.append((k,low[k-1],high[k],False))
        gap_fills={}
        for gi,gt,gb,iu in all_gaps:
            fb=n
            for k in range(gi+1,n):
                if iu and low[k]<=gb: fb=k; break
                if not iu and high[k]>=gt: fb=k; break
            gap_fills[gi]=fb
        def _fkc(s,e,ib):
            bi,bs=None,-1
            for k in range(s,e+1):
                bp=(close[k]-opn[k])/opn[k]*100 if opn[k]>0 else 0
                if (ib and bp<=0) or (not ib and bp>=0): continue
                vw=volume[k]/max(volume[s:e+1].mean(),1) if volume is not None else 1.0
                sc=abs(bp)*vw; lp=20.0 if opn[k]>=50 else 10.0
                if (ib and bp>=lp*0.9) or (not ib and bp<=-lp*0.9): sc*=10
                if sc>bs: bs=sc; bi=k
            return bi
        for t in range(n):
            vis=[(pi,pt) for pi,pt,pc in sp if pc<=t]; vis.sort(key=lambda x:x[0])
            if len(vis)<2: continue
            zones=[]
            for vi in range(len(vis)-1):
                si,st=vis[vi]; ei,et=vis[vi+1]
                if st=="TROUGH" and et=="PEAK": zones.append((si,ei,"UP"))
                elif st=="PEAK" and et=="TROUGH": zones.append((si,ei,"DOWN"))
            if not zones: continue
            cur_zi=None
            for zi,(zs,ze,zp) in enumerate(zones):
                if zs<=t<=ze: cur_zi=zi; break
            if cur_zi is None:
                lpi,lpt=vis[-1]
                if t>=lpi and zones[-1][1]==lpi:
                    cur_zi=len(zones)
                    if zones[-1][2]=="UP": zones.append((zones[-1][1],t,"DOWN"))
                    else: zones.append((zones[-1][1],t,"UP"))
            if cur_zi is None or cur_zi==0: continue
            czs,cze,czp=zones[cur_zi]; pzs,pze,pzp=zones[cur_zi-1]
            resistances=[]
            if pzp=="UP":
                zh=high[pzs:pze+1].max(); cm=high[czs:min(t+1,cze+1)].max()
                if zh>close[t] and cm<zh: resistances.append((zh,"UP_HIGH"))
            elif pzp=="DOWN":
                hg=False
                for gi,gt,gb,iu in all_gaps:
                    if gi<pzs or gi>pze or not iu or t-gi<gma: continue
                    if gap_fills.get(gi,n)<=t: continue
                    if gt>close[t]:
                        cm=high[czs:min(t+1,cze+1)].max()
                        if cm<gt: resistances.append((gt,"GAP")); hg=True; break
                if not hg:
                    ki=_fkc(pzs,pze,False)
                    if ki is not None:
                        kh=high[ki]
                        if kh>close[t]:
                            cm=high[czs:min(t+1,cze+1)].max()
                            if cm<kh: resistances.append((kh,"KEY"))
            cm=high[czs:min(t+1,cze+1)].max(); em=max((lv for lv,_ in resistances),default=cm)
            for zi in range(cur_zi-2,-1,-1):
                zs,ze,zp=zones[zi]
                if zp!="UP": continue
                zh=high[zs:ze+1].max()
                if zh<=em or cm>=zh or zh<=close[t]: continue
                resistances.append((zh,f"UP_HIGH+{cur_zi-zi}")); em=zh
            resistances.sort(key=lambda x:x[0])
            supports=[]
            if pzp=="DOWN":
                zl=low[pzs:pze+1].min(); cmn=low[czs:min(t+1,cze+1)].min()
                if zl<close[t] and cmn>zl: supports.append((zl,"DN_LOW"))
            elif pzp=="UP":
                hg=False
                for gi,gt,gb,iu in all_gaps:
                    if gi<pzs or gi>pze or iu or t-gi<gma: continue
                    if gap_fills.get(gi,n)<=t: continue
                    if gb<close[t]:
                        cmn=low[czs:min(t+1,cze+1)].min()
                        if cmn>gb: supports.append((gb,"GAP")); hg=True; break
                if not hg:
                    ki=_fkc(pzs,pze,True)
                    if ki is not None:
                        kl=low[ki]
                        if kl<close[t]:
                            cmn=low[czs:min(t+1,cze+1)].min()
                            if cmn>kl: supports.append((kl,"KEY"))
            cmn=low[czs:min(t+1,cze+1)].min(); emn=min((lv for lv,_ in supports),default=cmn)
            for zi in range(cur_zi-2,-1,-1):
                zs,ze,zp=zones[zi]
                if zp!="DOWN": continue
                zl=low[zs:ze+1].min()
                if zl>=emn or cmn<=zl or zl>=close[t]: continue
                supports.append((zl,f"DN_LOW+{cur_zi-zi}")); emn=zl
            supports.sort(key=lambda x:-x[0])
            for lv,src in resistances:
                if high[t]>=lv*(1-tt): ts[t]=2; tsrc[t]=src; break
                elif close[t]>=lv*(1-at): ts[t]=1; tsrc[t]=src; break
            if ts[t]==0:
                for lv,src in supports:
                    if low[t]<=lv*(1+tt): ts[t]=-2; tsrc[t]=src; break
                    elif close[t]<=lv*(1+at): ts[t]=-1; tsrc[t]=src; break
        return ts,tsrc


# ============================================================
# 三、买卖点信号 — V10: 逐层突破（核心改动）
# ============================================================

def compute_buy_sell_signals(df_ohlc, result, dur_horizon=120, touch_norm=3,
                             W_DUR=0.7, W_TOUCH=0.3, tt=0.005):
    """
    V10 买卖点 — 逐层突破（带突破分量评分）。


    +1 BrkLvl: 突破一个前UP区高点。追踪所有未突破的前高，每突破一个发一次信号；
               分量 strength = W_DUR·g(压制时长) + W_TOUCH·h(上影线测试次数)，0~1。
    -1 BrkLow: 跌破一个前DOWN区低点（对称，同样带分量）。

    保留 V9 信号:
      +1 BrkRes, +1 PullSup, -1 BrkSup, -1 BncRes

    废弃 V9 的 BrkHigh/BrkLow (max of last 3) — 逐层突破更及时、更密集。
    """
    n = len(df_ohlc); close = df_ohlc['close'].values
    high = df_ohlc['high'].values; low = df_ohlc['low'].values; volume = df_ohlc['volume'].values

    bs_signal = np.zeros(n, dtype=int)
    bs_reason = np.array([''] * n, dtype='U50')
    bs_strength = np.zeros(n, dtype=float)
    pivots = result.attrs.get('pivots', [])

    # ── 从 phase 分段预计算 UP/DOWN 区间（绝对坐标，与 plot 中 ax0 的 intervals 同源）──
    # 目的：让阻力/支撑位追踪也覆盖“无 TROUGH 回踩的短 UP run”（如 300437 的 10.34 尖峰）。
    # 这些短 UP run 在 pivot 配对（TROUGH→PEAK）里组不成 zone，故原逻辑漏登记；
    # 并入后它们会进入 unbroken_highs/completed_highs，从而在 ax3 显示并参与 BrkLvl/BrkLow。
    # 与 pivot 区间靠 round(price,2) 去重，互不重复。
    _phase_arr = result['phase'].values
    zones_phase = []
    _pi = 0
    while _pi < n:
        _pj = _pi
        while _pj < n and _phase_arr[_pj] == _phase_arr[_pi]:
            _pj += 1
        if _phase_arr[_pi] in ('UP', 'DOWN') and _pj - 1 > _pi:
            zones_phase.append((_pi, _pj - 1, _phase_arr[_pi]))
        _pi = _pj


    # ── 追踪未突破的前高 / 未跌破的前低（带形成位置与测试计数）──
    unbroken_highs = {}    # key=round(price,2) -> {price, form_idx, touch_count, tests, kind}
    unbroken_lows = {}     # 对称
    completed_highs = []   # 已突破的阻力位（含 break_idx / break_strength / tests / kind）
    completed_lows = []    # 已跌破的支撑位
    processed_ends = set()    # 已处理的 zone 结束位置（去重）

    pending_resistance_level = None; pending_support_level = None

    # 预提取数组
    _phase = result['phase'].values; _is_pending = result['is_pending'].values
    _touch_signal = result['touch_signal'].values; _touch_source = result['touch_source'].values

    for t in range(n):
        # ── 构建 bar t 的区间 ──
        sp = sorted(pivots, key=lambda x: x[2])
        visible = [(p_idx, p_type) for p_idx, p_type, p_confirm in sp if p_confirm <= t]
        visible.sort(key=lambda x: x[0])
        zones = []
        for vi in range(len(visible) - 1):
            s_idx, s_type = visible[vi]; e_idx, e_type = visible[vi + 1]
            if s_type == 'TROUGH' and e_type == 'PEAK': zones.append((s_idx, e_idx, 'UP'))
            elif s_type == 'PEAK' and e_type == 'TROUGH': zones.append((s_idx, e_idx, 'DOWN'))

        # ── 更新未突破前高/前低集合（带形成位置与测试计数）──
        # 合并 pivot 区间与 phase 区间：phase 区间覆盖 pivot 配对漏掉的短 UP/DOWN run
        _all_zones = zones + zones_phase
        if _all_zones:
            # 用 ze < t 判断 zone 是否已结束（比 zones[:-1] 更准确）
            for zs, ze, zp in _all_zones:
                if ze < t and ze not in processed_ends:
                    processed_ends.add(ze)
                    if zp == 'UP':
                        seg = high[zs:ze + 1]; fi = int(np.argmax(seg)) + zs; zh = float(seg.max())
                        if zh > close[t]:
                            key = round(zh, 2)
                            if key in unbroken_highs:
                                unbroken_highs[key]['price'] = max(unbroken_highs[key]['price'], zh)
                                unbroken_highs[key]['form_idx'] = min(unbroken_highs[key]['form_idx'], fi)
                            else:
                                unbroken_highs[key] = {'price': zh, 'form_idx': fi,
                                                        'touch_count': 0, 'tests': [], 'kind': 'RES'}
                    elif zp == 'DOWN':
                        seg = low[zs:ze + 1]; fi = int(np.argmin(seg)) + zs; zl = float(seg.min())
                        if zl < close[t]:
                            key = round(zl, 2)
                            if key in unbroken_lows:
                                unbroken_lows[key]['price'] = min(unbroken_lows[key]['price'], zl)
                                unbroken_lows[key]['form_idx'] = min(unbroken_lows[key]['form_idx'], fi)
                            else:
                                unbroken_lows[key] = {'price': zl, 'form_idx': fi,
                                                       'touch_count': 0, 'tests': [], 'kind': 'SUP'}

        # ── 计数“上影线/下影线测试”：未破期间，影线蹭到价位但收盘未站上/跌破 ──
        for _lv, _m in unbroken_highs.items():
            if high[t] >= _m['price'] * (1 - tt) and close[t] < _m['price']:
                if not (t > 0 and high[t - 1] >= _m['price'] * (1 - tt) and close[t - 1] < _m['price']):
                    _m['touch_count'] += 1
                    _m['tests'].append(t)
        for _lv, _m in unbroken_lows.items():
            if low[t] <= _m['price'] * (1 + tt) and close[t] > _m['price']:
                if not (t > 0 and low[t - 1] <= _m['price'] * (1 + tt) and close[t - 1] > _m['price']):
                    _m['touch_count'] += 1
                    _m['tests'].append(t)

        # ────────────────────────────────────────────────
        # +1 买入: BrkLvl — 逐层突破前高（带分量评分）
        # ────────────────────────────────────────────────
        if (zones or zones_phase) and unbroken_highs:
            # 按价格从低到高排列，突破最近（最低）的优先
            broken = [lv for lv in unbroken_highs if close[t] > lv]
            if broken:
                target = min(broken)            # 取最近的一个（价格最低的未突破高点）
                meta = unbroken_highs[target]
                dur = t - meta['form_idx']
                dscore = min(1.0, dur / dur_horizon)
                tscore = min(1.0, meta['touch_count'] / touch_norm)
                strength = W_DUR * dscore + W_TOUCH * tscore   # 时长优先 7:3
                bs_signal[t] = 1
                bs_reason[t] = f'BrkLvl({target:.2f},s={strength:.2f})'
                bs_strength[t] = strength
                cc = unbroken_highs.pop(target)
                cc['break_idx'] = t
                cc['break_strength'] = strength
                completed_highs.append(cc)
                continue

        # ────────────────────────────────────────────────
        # -1 卖出: BrkLow — 逐层跌破前低
        # ────────────────────────────────────────────────
        if (zones or zones_phase) and unbroken_lows:
            broken = [lv for lv in unbroken_lows if close[t] < lv]
            if broken:
                target = max(broken)  # 取最近的一个（价格最高的未跌破低点）
                meta = unbroken_lows[target]
                dur = t - meta['form_idx']
                dscore = min(1.0, dur / dur_horizon)
                tscore = min(1.0, meta['touch_count'] / touch_norm)
                strength = W_DUR * dscore + W_TOUCH * tscore
                bs_signal[t] = -1
                bs_reason[t] = f'BrkLow({target:.2f},s={strength:.2f})'
                bs_strength[t] = strength
                cc = unbroken_lows.pop(target)
                cc['break_idx'] = t
                cc['break_strength'] = strength
                completed_lows.append(cc)
                continue

        if bs_signal[t] != 0: continue

        # ── 以下保留 V9 信号 ──
        phase = _phase[t]; is_pend = _is_pending[t]
        touch_sig = _touch_signal[t]; touch_src = _touch_source[t]

        if touch_sig == 2: pending_resistance_level = high[t]
        elif touch_sig == -2: pending_support_level = low[t]

        if pending_resistance_level is not None:
            if close[t] > pending_resistance_level and phase in ('UP', 'NEUTRAL'):
                bs_signal[t] = 1; bs_reason[t] = f'BrkRes({pending_resistance_level:.2f})'
                pending_resistance_level = None; continue
            if phase == 'DOWN' and not is_pend: pending_resistance_level = None

        if is_pend and phase == 'UP' and touch_sig <= -1:
            bs_signal[t] = 1; bs_reason[t] = f'PullSup({touch_src})'; continue

        if pending_support_level is not None:
            if close[t] < pending_support_level and phase in ('DOWN', 'NEUTRAL'):
                bs_signal[t] = -1; bs_reason[t] = f'BrkSup({pending_support_level:.2f})'
                pending_support_level = None; continue
            if phase == 'UP' and not is_pend: pending_support_level = None

        if is_pend and phase == 'DOWN' and touch_sig >= 1:
            bs_signal[t] = -1; bs_reason[t] = f'BncRes({touch_src})'; continue

        # ────────────────────────────────────────────────
    # 汇总阻力/支撑位生命周期（未突破的 + 已突破的），供画图用
    all_levels = []
    for m in list(unbroken_highs.values()) + completed_highs:
        d = dict(m); d.setdefault('break_idx', None); d.setdefault('break_strength', 0.0)
        all_levels.append(d)
    for m in list(unbroken_lows.values()) + completed_lows:
        d = dict(m); d.setdefault('break_idx', None); d.setdefault('break_strength', 0.0)
        all_levels.append(d)
    return bs_signal, bs_reason, bs_strength, all_levels


# ============================================================
# 四、图表 — 4面板（同 V9）
# ============================================================
def plot_price_segmentation_v10(df_ohlc, result, bs_signal, bs_reason,
                                tail_days=200, name="", save_path=None,
                                bs_strength=None, all_levels=None,
                                reg_preds=None, reg_preds_long=None,
                                hide_ma=True,
                                reg_win=120, reg_win_long=250,
                                panic_info=None,
                                strength_win=10,
                                dir_atr=2.0,
                                despeckle=False,
                                hide_mid_panels=True):
    """5面板: K线 + 成交量 + 买卖信号(柱高=突破分量) + 阻力/支撑位生命周期 + 极速杀跌反转signal。
    panic_info: panic_reversal.signal() 返回的 dict(含 signal 状态与门控明细),None 则面板显示提示。"""
    ohlc = df_ohlc.tail(tail_days).copy().reset_index(drop=True)
    n = len(ohlc); x = np.arange(n); offset = len(df_ohlc) - n  # ohlc 是 df_ohlc 末尾 n 行，offset 为其在原序列中的起始下标（恒 >=0）

    if hide_mid_panels:
        fig, axes = plt.subplots(4, 1, figsize=(22, 14),
                                 sharex=True,
                                 gridspec_kw={'height_ratios': [4, 1.3, 0.55, 0.6]})
    else:
        fig, axes = plt.subplots(6, 1, figsize=(22, 18),
                                 sharex=True,
                                 gridspec_kw={'height_ratios': [4, 1.3, 0.45, 0.9, 0.55, 0.6]})
    fig.suptitle(f'{name}  Price Segmentation V10 (Level Breakout)', fontsize=14, fontweight='bold')

    ax0 = axes[0]; opens = ohlc['open'].values; highs = ohlc['high'].values
    lows = ohlc['low'].values; closes = ohlc['close'].values; vols = ohlc['volume'].values
    bar_w = 0.6

    ph = result['phase'].values[offset:offset + n]
    intervals = []; i = 0
    while i < n:
        j = i
        while j < n and ph[j] == ph[i]: j += 1
        intervals.append((i, j - 1, ph[i], False)); i = j
    if intervals: intervals[-1] = (intervals[-1][0], intervals[-1][1], intervals[-1][2], True)

    for s, e, p, pend in intervals:
        if pend:
            ax0.axvspan(s - 0.5, e + 0.5, alpha=0.04, color='orange' if p == "UP" else 'cyan', zorder=0)
            zl = lows[s:e + 1].min(); zh = highs[s:e + 1].max(); mg = (zh - zl) * 0.03
            ax0.add_patch(plt.Rectangle((s - 0.5, zl - mg), e - s + 1, (zh - zl) + 2 * mg,
                facecolor='none', edgecolor='#FF6F00' if p == "UP" else '#00695C',
                linewidth=1.5, linestyle='--', zorder=4))
            ax0.text((s + e) / 2, zh + mg, f"PENDING {p}", fontsize=7, fontweight='bold',
                     color='#FF6F00' if p == "UP" else '#00695C', ha='center', va='bottom', zorder=5)
        else:
            ax0.axvspan(s - 0.5, e + 0.5, alpha=0.10 if p == "UP" else 0.08,
                        color='red' if p == "UP" else 'green', zorder=0)
            if e > s:
                zl = lows[s:e + 1].min(); zh = highs[s:e + 1].max(); mg = (zh - zl) * 0.03
                ax0.text((s + e) / 2, zh + mg, p, fontsize=7, fontweight='bold',
                         color='#B71C1C' if p == "UP" else '#1B5E20', ha='center', va='bottom', zorder=5)

    for i in range(n):
        c = '#ef5350' if closes[i] >= opens[i] else '#26a69a'
        ax0.plot([x[i], x[i]], [lows[i], highs[i]], color=c, linewidth=0.5)
        bl = min(opens[i], closes[i]); bh = max(opens[i], closes[i])
        ax0.add_patch(plt.Rectangle((x[i] - bar_w / 2, bl), bar_w, bh - bl,
                                    facecolor=c, edgecolor=c, linewidth=0.4))

    fc = df_ohlc['close'].values
    if not hide_ma:
        ma120 = pd.Series(fc).rolling(120, min_periods=1).mean().values[-tail_days:]
        ax0.plot(x, ma120, color='#7B1FA2', linewidth=1.2, alpha=0.8, label='MA120')
        sm = result['smooth'].values[offset:offset + n]
        ax0.plot(x, sm, color='#1565C0', linewidth=1.0, alpha=0.6, label='EMA')
    # 长周期回归线(250日):按趋势方向分段着色——20日斜率>0 绿色(reg 上行) / <0 红色(reg 下行)
    # 2026-08-27:黄金坑+reg上行验证(94.9% vs 下行86.0%),着色直观显示方向
    if reg_preds_long is not None:
        from matplotlib.collections import LineCollection as _LC
        rpl = reg_preds_long[offset:offset + n]
        _segs = []; _cols = []
        for _i in range(len(rpl) - 1):
            if np.isfinite(rpl[_i]) and np.isfinite(rpl[_i + 1]):
                _segs.append([(x[_i], rpl[_i]), (x[_i + 1], rpl[_i + 1])])
                if _i >= 20 and np.isfinite(rpl[_i]) and np.isfinite(rpl[_i - 20]):
                    _cols.append('#2E7D32' if rpl[_i] > rpl[_i - 20] else '#C62828')
                else:
                    _cols.append('#90A4AE')
        if _segs:
            _lc = _LC(_segs, colors=_cols, linewidth=1.8, alpha=0.9)
            ax0.add_collection(_lc)
            ax0.plot([], [], color='#2E7D32', lw=1.8, label=f'Reg Long up ({reg_win_long}d)')
            ax0.plot([], [], color='#C62828', lw=1.8, label=f'Reg Long down ({reg_win_long}d)')
    # 中周期回归线(120日):分段着色(细线,辅助)——与 250 日同绿 = 最强置信区(reg120+250双上行 97.7%)
    if reg_preds is not None:
        from matplotlib.collections import LineCollection as _LC2
        rp = reg_preds[offset:offset + n]
        _segs2 = []; _cols2 = []
        for _i in range(len(rp) - 1):
            if np.isfinite(rp[_i]) and np.isfinite(rp[_i + 1]):
                _segs2.append([(x[_i], rp[_i]), (x[_i + 1], rp[_i + 1])])
                if _i >= 20 and np.isfinite(rp[_i]) and np.isfinite(rp[_i - 20]):
                    _cols2.append('#1B5E20' if rp[_i] > rp[_i - 20] else '#B71C1C')
                else:
                    _cols2.append('#B0BEC5')
        if _segs2:
            _lc2 = _LC2(_segs2, colors=_cols2, linewidth=1.0, alpha=0.65)
            ax0.add_collection(_lc2)
            ax0.plot([], [], color='#1B5E20', lw=1.0, label=f'Reg up ({reg_win}d)')
            ax0.plot([], [], color='#B71C1C', lw=1.0, label=f'Reg down ({reg_win}d)')

    # 阴柱期转阳目标价线(仅阴柱日有值,阳柱/无柱日 NaN——阶梯线,突破该价次日转阳)
    try:
        import panic_reversal as _pr
        _fc = df_ohlc['close'].values.astype(np.float64)
        _fh = df_ohlc['high'].values.astype(np.float64)
        _fl = df_ohlc['low'].values.astype(np.float64)
        _fo = df_ohlc['open'].values.astype(np.float64)
        _frg = None
        if reg_preds_long is not None:
            _frg = np.asarray(reg_preds_long, dtype=np.float64)
        elif reg_preds is not None:
            _frg = np.asarray(reg_preds, dtype=np.float64)
        _tp = _pr.compute_turn_positive_prices(_fc, _fh, _fl, opens=_fo, reg_preds=_frg)
        tp_win = _tp[offset:offset + n]
        ax0.plot(x, tp_win, drawstyle='steps-post', color='#1565C0', linewidth=2.6,
                 linestyle='--', alpha=0.95, label='Turn-Up Target')
    except Exception:
        pass

    for si, (s, e, p, _) in enumerate(intervals):
        if p == "UP" and e > s:
            ax0.hlines(highs[s:e + 1].max(), s - 0.5, e + 0.5, colors='#B71C1C',
                       linewidths=1.0, linestyles='--', alpha=0.7, zorder=3)
        if p == "DOWN" and e > s:
            ax0.hlines(lows[s:e + 1].min(), s - 0.5, e + 0.5, colors='#1B5E20',
                       linewidths=1.0, linestyles='--', alpha=0.7, zorder=3)

    GAP_COLOR = '#1B5E20'; GAP_LINE_COLOR = '#2E7D32'; all_gaps = []
    for k in range(1, n):
        if lows[k] > highs[k - 1]: all_gaps.append((k, lows[k], highs[k - 1], True))
        if highs[k] < lows[k - 1]: all_gaps.append((k, lows[k - 1], highs[k], False))
    for gi, gt, gb, iug in all_gaps:
        fb = n
        for k in range(gi + 1, n):
            if iug and lows[k] <= gb: fb = k; break
            if not iug and highs[k] >= gt: fb = k; break
        ax0.add_patch(plt.Rectangle((x[gi - 1] + bar_w / 2, gb), x[gi] - x[gi - 1] - bar_w,
            gt - gb, facecolor=GAP_COLOR, alpha=0.30, edgecolor=GAP_COLOR,
            linewidth=1.0, linestyle='-', zorder=4))
        le = fb - 0.5 if fb < n else n - 0.5
        ax0.add_patch(plt.Rectangle((gi - 0.5, gb), le - gi + 1.0, gt - gb,
            facecolor=GAP_COLOR, alpha=0.10, edgecolor='none', zorder=2))
        ax0.hlines(gt, gi - 0.5, le, colors=GAP_LINE_COLOR, linewidths=0.8, linestyles=':', alpha=0.7, zorder=3)
        ax0.hlines(gb, gi - 0.5, le, colors=GAP_LINE_COLOR, linewidths=0.8, linestyles=':', alpha=0.7, zorder=3)
        lx = le + 0.3; gtype = '▲' if iug else '▼'
        ax0.text(lx, gt, f'{gtype} {gt:.2f}', fontsize=5.5, color=GAP_COLOR, va='center', ha='left',
                 zorder=5, bbox=dict(boxstyle='round,pad=0.08', facecolor='white', edgecolor=GAP_COLOR, alpha=0.75, linewidth=0.4))
        ax0.text(lx, gb, f'{gtype} {gb:.2f}', fontsize=5.5, color=GAP_COLOR, va='center', ha='left',
                 zorder=5, bbox=dict(boxstyle='round,pad=0.08', facecolor='white', edgecolor=GAP_COLOR, alpha=0.75, linewidth=0.4))

    def _fkc(s, e, ib):
        bi, bs = None, -1
        for k in range(s, e + 1):
            bp = (closes[k] - opens[k]) / opens[k] * 100 if opens[k] > 0 else 0
            if (ib and bp <= 0) or (not ib and bp >= 0): continue
            vw = vols[k] / max(vols[s:e+1].mean(), 1); sc = abs(bp) * vw
            lp = 20.0 if opens[k] >= 50 else 10.0
            if (ib and bp >= lp * 0.9) or (not ib and bp <= -lp * 0.9): sc *= 10
            if sc > bs: bs = sc; bi = k
        return bi
    def _zhug(s, e, iug, cs):
        for gg, ggt, ggb, ig in all_gaps:
            if gg < s or gg > e or ig != iug: continue
            fbg = n
            for k in range(gg + 1, n):
                if ig and lows[k] <= ggb: fbg = k; break
                if not ig and highs[k] >= ggt: fbg = k; break
            if fbg > cs: return True
        return False
    KCC = '#E65100'
    for si, (s, e, p, _) in enumerate(intervals):
        if si == 0: continue
        ps, pe, pp, _ = intervals[si - 1]
        if pe < ps: continue
        if p == "UP":
            if _zhug(ps, pe, False, s): continue
            ki = _fkc(ps, pe, False)
            if ki is not None:
                kh = highs[ki]; kl = lows[ki]
                ax0.hlines(kh, s - 0.5, e + 0.5, colors=KCC, linewidths=0.8, linestyles=':', alpha=0.6, zorder=3)
                ax0.hlines(kl, s - 0.5, e + 0.5, colors=KCC, linewidths=0.8, linestyles=':', alpha=0.6, zorder=3)
                ax0.text(e + 1.0, kh, f'Key H {kh:.2f}', fontsize=6, color=KCC, va='center', zorder=5)
                bl = min(opens[ki], closes[ki]); bh = max(opens[ki], closes[ki])
                ax0.add_patch(plt.Rectangle((x[ki] - bar_w / 2 - 0.15, bl - 0.1), bar_w + 0.3, bh - bl + 0.2,
                    facecolor='none', edgecolor=KCC, linewidth=2.0, linestyle='-', zorder=5))
        elif p == "DOWN":
            if _zhug(ps, pe, True, s): continue
            ki = _fkc(ps, pe, True)
            if ki is not None:
                kh = highs[ki]; kl = lows[ki]
                ax0.hlines(kh, s - 0.5, e + 0.5, colors=KCC, linewidths=0.8, linestyles=':', alpha=0.6, zorder=3)
                ax0.hlines(kl, s - 0.5, e + 0.5, colors=KCC, linewidths=0.8, linestyles=':', alpha=0.6, zorder=3)
                ax0.text(e + 1.0, kh, f'Key H {kh:.2f}', fontsize=6, color=KCC, va='center', zorder=5)
                bl = min(opens[ki], closes[ki]); bh = max(opens[ki], closes[ki])
                ax0.add_patch(plt.Rectangle((x[ki] - bar_w / 2 - 0.15, bl - 0.1), bar_w + 0.3, bh - bl + 0.2,
                    facecolor='none', edgecolor=KCC, linewidth=2.0, linestyle='-', zorder=5))

    for i in range(n):
        gi = offset + i
        if result['is_pivot'].values[gi]:
            pt = result['pivot_type'].values[gi]
            if pt == "PEAK": ax0.plot(x[i], highs[i] * 1.01, 'rv', markersize=6, alpha=0.7)
            elif pt == "TROUGH": ax0.plot(x[i], lows[i] * 0.99, 'g^', markersize=6, alpha=0.7)

    uzh = []
    for si2, (s2, e2, p2, _) in enumerate(intervals):
        if p2 == "UP" and e2 > s2: uzh.append((e2, highs[s2:e2 + 1].max(), s2))
    for si, (s, e, p, _) in enumerate(intervals):
        if p != "UP" or e <= s: continue
        phs = [(zh, s2, e2) for (e2, zh, s2) in uzh if e2 < s]
        if not phs: continue
        pss = sorted(phs, key=lambda x: x[2], reverse=True)[:5]; pss.sort(key=lambda x: x[0])
        for rk, (zh, s2, e2) in enumerate(pss):
            ax0.hlines(zh, e2 + 0.5, e + 0.5, colors='#FF1744', linewidths=0.8,
                       linestyles='-.', alpha=min(0.3 + 0.08 * rk, 0.85), zorder=3)
            ax0.text(e + 1.2, zh, f'R {zh:.2f}', fontsize=6, color='#FF1744', va='center', ha='left',
                     zorder=5, bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='#FF1744', alpha=0.8, linewidth=0.5))

    # ── 价格轴聚焦：以最近可见收盘价为锚，排除把轴撑爆的远端历史高低点 ──
    # 关键修复：轴范围必须纳入叠加曲线（MA120 / EMA / 回归线），否则这些线在窗口左侧
    # 回看 pre-window 历史价时会低于可见 K 线最低价，被裁切到 y 轴外看不见。
    # 仅当极值属“离群”（>4× 或 <0.12× 近期价）才封顶；健康区间（如 5-30）保持原样不被过度收窄。
    # ax3 复用该区间，使两轴范围一致。
    _last_cv = closes[-1]
    # 收集所有可见曲线的 y 值（K线极值 + 叠加均线/回归线），统一参与轴范围计算
    _overlay = [highs, lows]
    if not hide_ma:
        _overlay += [np.asarray(ma120), np.asarray(sm)]
    if reg_preds is not None:
        _overlay.append(np.asarray(rp))
    if reg_preds_long is not None:
        _overlay.append(np.asarray(rpl))
    _all_y = np.concatenate([a[np.isfinite(a)] for a in _overlay])
    _y_hi = _all_y.max()
    _y_lo = _all_y.min()
    if _y_hi > _last_cv * 4:
        _near = _all_y[_all_y <= _last_cv * 4]
        _y_hi = _near.max() if _near.size else _last_cv * 4
    if _y_lo < _last_cv * 0.12:
        _near = _all_y[_all_y >= _last_cv * 0.12]
        _y_lo = _near.min() if _near.size else _last_cv * 0.12
    # 上下各留 5% 余量，避免曲线贴边（兼顾均线与回归线在窗口边缘的可见性）
    _pad = (_y_hi - _y_lo) * 0.05
    _focus_lo, _focus_hi = _y_lo - _pad, _y_hi + _pad
    ax0.set_ylim(_focus_lo, _focus_hi)

    ax0.set_ylabel('Price', fontsize=10); ax0.grid(True, alpha=0.3)
    # 放宽 x 轴边界：左留 2 根 K 线宽 + 右留 1 根，避免首尾 K 线/成交量柱被窗口边缘裁切
    ax0.set_xlim(-2, n + 0.5)
    ax0.legend(handles=[
        Patch(facecolor='red', alpha=0.15, label='UP (confirmed)'),
        Patch(facecolor='green', alpha=0.12, label='DOWN (confirmed)'),
        Patch(facecolor='orange', alpha=0.15, label='UP (pending)'),
        Patch(facecolor='cyan', alpha=0.15, label='DOWN (pending)'),
        Line2D([0], [0], color='#B71C1C', linewidth=1.0, linestyle='--', label='UP zone high'),
        Line2D([0], [0], color='#1B5E20', linewidth=1.0, linestyle='--', label='DOWN zone low'),
        Line2D([0], [0], color='#d62728', linewidth=2.0, linestyle='--', label=f'Reg ({reg_win}d)'),
        Line2D([0], [0], color='#1565C0', linewidth=1.5, linestyle='-', label=f'Reg Long ({reg_win_long}d)'),
        Line2D([0], [0], marker='v', color='r', linestyle='None', markersize=6, label='PEAK'),
        Line2D([0], [0], marker='^', color='g', linestyle='None', markersize=6, label='TROUGH'),
        Patch(facecolor='#1B5E20', alpha=0.30, label='Gap'),
        Line2D([0], [0], color='#2E7D32', linewidth=0.8, linestyle=':', label='Gap line'),
        Line2D([0], [0], color='#E65100', linewidth=0.8, linestyle=':', label='Key candle'),
        Line2D([0], [0], color='#FF1744', linewidth=0.9, linestyle='-.', label='Resistance'),
        Line2D([0], [0], color='#d62728', linewidth=2.0, linestyle='--', label='Regression'),
    ], loc='upper left', fontsize=7, ncol=4)

    ax1 = axes[1]; vol = ohlc['volume'].values; va = result['vol_annotation'].values[offset:offset + n]
    vc = {"VOL_EXPANDING": "#ef5350", "VOL_SHRINKING": "#26a69a", "NEUTRAL": "#9E9E9E"}
    ax1.bar(x, vol, width=bar_w, color=[vc.get(va[k], '#9E9E9E') for k in range(n)], alpha=0.8)
    # 成交量堆背景(MAD z-score,无未来函数):放量堆淡红 / 缩量堆淡绿
    # ohlc 是窗口切片,堆索引即窗口坐标,直接用 x 画(勿加 offset!)
    try:
        import panic_reversal as _pr
        # ⚠️ 堆必须用完整历史算(基准60日/20日中位量),窗口只影响显示——
        # 否则 150 天小窗口会把窗口外的放量堆截断(300251 9/24 堆被 150 天窗口截掉,用户看到 10/29 起点)
        _vcl = _pr.detect_volume_clusters(df_ohlc['close'].values, df_ohlc['volume'].values)
        for _s, _e, _kd, _dr, _zp, _vr in _vcl:
            if _kd != 'HIGH':
                continue  # 只标放量堆(缩量堆不标记,用户 2026-08-21 要求)
            _s0 = _s - offset; _e0 = _e - offset  # 全序列索引 → 窗口坐标
            if _e0 < 0 or _s0 >= n:
                continue
            _s0 = max(0, _s0); _e0 = min(n - 1, _e0)
            ax1.axvspan(_s0 - 0.5, _e0 + 0.5, color='#ef5350', alpha=0.14, zorder=0)
    except Exception:
        pass
    # 股眼标注(阿笨:改变走势的关键量堆)— 色块 + 类型文字,叠加在 volume 上
    try:
        import aben_patterns as _ab
        _gys = _ab.detect_guyan(df_ohlc['close'].values.astype(np.float64),
                                df_ohlc['volume'].values.astype(np.float64))
        _gycol = {'reversal': '#E65100', 'breakout': '#1565C0',
                  'accel': '#6A1B9A', 'mega_absorb': '#C62828'}
        _vmax = vol.max() if len(vol) > 0 and vol.max() > 0 else 1.0
        for _gs, _ge, _pre, _post, _typ, _pk in _gys:
            _xs = _gs - offset
            _xe = _ge - offset
            if _xe < 0 or _xs >= n:
                continue
            _xs = max(0, _xs); _xe = min(n - 1, _xe)
            _c = _gycol.get(_typ, '#333333')
            ax1.axvspan(_xs - 0.5, _xe + 0.5, color=_c, alpha=0.30, zorder=1)
            ax1.text((_xs + _xe) / 2, _vmax * 1.06, _typ, ha='center', va='bottom',
                     fontsize=6.5, color=_c, fontweight='bold', zorder=5)
    except Exception:
        pass
    # 量轴：取窗口内实际最大值 * 1.2，完整显示 + 20% 顶空
    if len(vol) > 0 and vol.max() > 0:
        ax1.set_ylim(0, vol.max() * 1.2)
    ax1.set_ylabel('Volume', fontsize=9); ax1.grid(True, alpha=0.2)
    ax1.set_title('Volume (red=expanding, green=shrinking; bg=cluster)', fontsize=9, loc='left', pad=2)

    ax2 = axes[2] if not hide_mid_panels else None
    if ax2 is not None:
        # ── 信号面板（ax2）：柱高编码突破分量；顶端圆点大小随分量增大 ──
        bsl = bs_signal[offset:offset + n]; brl = bs_reason[offset:offset + n]
        bstr = (bs_strength[offset:offset + n] if bs_strength is not None
                else np.zeros(n, dtype=float))
        bc2 = {1: '#42A5F5', 0: 'none', -1: '#FF7043'}
        heights = np.array([0.0 if bsl[k] == 0
                            else (bsl[k] * (0.4 + 0.6 * bstr[k]) if bstr[k] > 0 else bsl[k] * 0.5)
                            for k in range(n)])
        ax2.bar(x, heights, width=bar_w * 2,
                color=[bc2.get(bsl[k], 'none') for k in range(n)], alpha=0.9)
        for i in range(n):
            s = bsl[i]
            if s == 0: continue
            st = bstr[i]
            if st > 0:
                tip = s * (0.4 + 0.6 * st)
                ax2.scatter(i, tip + 0.06 * s, s=20 + 90 * st,
                            color=bc2.get(s, 'gray'), alpha=0.9, zorder=6)
            ax2.text(i, s * 1.28, brl[i], fontsize=5.0, color=bc2.get(s, 'gray'),
                     ha='center', va='top', rotation=90, zorder=5)
        ax2.set_ylim(-1.45, 1.45)
        ax2.set_title('Buy/Sell Signals (V10: Level Breakout) — bar height & dot size = breakout conviction (strength)',
                      fontsize=8, loc='left', pad=2)

    ax3 = axes[3] if not hide_mid_panels else None
    if ax3 is not None:
        # ── 阻力/支撑位生命周期面板（ax3）：★形成 · ▽/▲测试 · ●突破 ──
        # 只显示落在 ax0 聚焦价格区间内的价位（_focus_lo/_focus_hi），
        # 使 ax3 的 y 轴与 ax0 一致，过滤掉远古未突破的高/低位（如把轴撑到 50 的阻力）
        _lo, _hi = _focus_lo, _focus_hi
        if all_levels:
            for lv in all_levels:
                fp = lv['price']; kind = lv.get('kind', 'RES')
                if fp < _lo or fp > _hi:
                    continue
                fi = lv['form_idx'] - offset
                bi = lv.get('break_idx')
                end = (bi - offset) if bi is not None else (n - 1)
                if end < 0 or fi > n - 1: continue
                xs = max(fi, 0); xe = min(end, n - 1)
                col = '#B71C1C' if kind == 'RES' else '#1B5E20'
                ax3.hlines(fp, xs - 0.5, xe + 0.5, colors=col,
                           linewidths=1.0, linestyles='-' if bi is not None else '--',
                           alpha=0.55, zorder=2)
                for tt in lv.get('tests', []):
                    ttx = tt - offset
                    if 0 <= ttx < n:
                        ax3.plot(ttx, fp, marker=('v' if kind == 'RES' else '^'),
                                 color=col, markersize=3, alpha=0.55, zorder=3)
                if 0 <= fi < n:
                    ax3.plot(fi, fp, marker='*', color=col, markersize=8, zorder=4)
                if bi is not None and 0 <= (bi - offset) < n:
                    bix = bi - offset
                    ax3.plot(bix, fp, marker='o', color=col, markersize=6, zorder=5)
                    ax3.annotate(f"s={lv.get('break_strength', 0):.2f}", (bix, fp),
                                 textcoords='offset points', xytext=(4, 2),
                                 fontsize=5, color=col, zorder=6)
        ax3.plot(x, fc[offset:offset + n], color='#37474F', linewidth=0.9, alpha=0.6, zorder=1)
        ax3.set_ylim(_focus_lo, _focus_hi)
        ax3.set_ylabel('Level Price', fontsize=9); ax3.grid(True, alpha=0.2)
        ax3.set_title('Resistance/Support Level Lifecycle  (★ formed · ▽ upper-shadow test · ▲ lower-shadow test · ● broken w/ strength)',
                      fontsize=8, loc='left', pad=2)


    # ── Panic-Reversal Signal Panel (5th panel: per-stock 5d drop bars, share x-axis with K-line) ──
    ax4 = axes[2] if hide_mid_panels else axes[4]
    ax5 = axes[3] if hide_mid_panels else axes[5]  # 黄金坑 0/1 方波面板(最底)
    ax4.set_facecolor('#FAFAFA')
    # v4 strength 柱(死区滤波:波幅/ATR + 收盘位移方向,±30 饱和压缩)
    try:
        import panic_reversal as _pr
        _sc4 = df_ohlc['close'].values.astype(np.float64)
        _sh4 = df_ohlc['high'].values.astype(np.float64)
        _sl4 = df_ohlc['low'].values.astype(np.float64)
        _rg4 = None
        # 回归线门控优先用 250 日(reg_preds_long)——低于长期回归线时反转不可信
        if reg_preds_long is not None:
            _rg4 = np.asarray(reg_preds_long, dtype=np.float64)
        elif reg_preds is not None:
            _rg4 = np.asarray(reg_preds, dtype=np.float64)
        _so4 = df_ohlc['open'].values.astype(np.float64)
        strength4 = _pr.compute_strength(_sc4, _sh4, _sl4, win=strength_win, dir_atr=dir_atr,
                                         reg_preds=_rg4, opens=_so4)
        # despeckle_strength 用到右侧(未来)柱段判断,存在未来函数,默认关闭,
        # 仅用于事后可视化参考,绝不用于 signal()/实盘判定。
        if despeckle:
            strength4 = _pr.despeckle_strength(strength4)   # 在完整序列上同化后再切片
        strength4 = strength4[offset:offset + n]
        has_str = True
    except Exception:
        strength4 = np.zeros(n); has_str = False
    for _i in range(n):
        if has_str and np.isfinite(strength4[_i]) and abs(strength4[_i]) >= 1e-9:
            ax4.bar(x[_i], strength4[_i], width=0.65,
                    color='#E8403F' if strength4[_i] > 0 else '#2ECC40', alpha=0.85)
    ax4.axhline(0, color='#AAAAAA', lw=0.8)
    # 事件日/确认日标注(单只股票,来自 signal())
    _pi4 = panic_info or {}
    _dts = ohlc['date'].dt.strftime('%Y%m%d').values
    pt4, cf4 = _pi4.get('panic_t'), _pi4.get('confirm')
    if pt4 and has_str:
        _i = np.where(_dts == str(pt4))[0]
        if len(_i):
            ax4.scatter([_i[0]], [min(strength4[_i[0]], -2.0)], color='#C0392B', s=80, zorder=6, marker='v')
    if cf4 and has_str:
        _i = np.where(_dts == str(cf4))[0]
        if len(_i):
            ax4.scatter([_i[0]], [min(strength4[_i[0]], -2.0)], color='#2ECC40', s=70, zorder=6, marker='o')
    sig4 = _pi4.get('signal', 'N/A')
    _y4 = float(strength4[-1]) if has_str and np.isfinite(strength4[-1]) else 0.0
    _st4v = _pi4.get('strength_t')
    _pt4s = _pi4.get('panic_t')
    _suf = ''
    if _pt4s:
        _suf += ' @' + str(_pt4s)[-8:]
    if _st4v is not None:
        _suf += ' str=%.1f' % float(_st4v)
    if sig4 == 'BUY':
        ax4.annotate('BUY' + _suf, xy=(n - 1, _y4), xytext=(8, 8),
                     textcoords='offset points', fontsize=13, fontweight='bold', color='#27AE60')
    elif sig4 == 'WATCH':
        ax4.annotate('WATCH' + _suf, xy=(n - 1, _y4), xytext=(8, 8),
                     textcoords='offset points', fontsize=11, fontweight='bold', color='#E67E22')
    elif sig4 == 'NONE':
        ax4.text(n - 1, -28.0, 'NONE', fontsize=10, fontweight='bold',
                 color='#7F8C8D', ha='right')
    ax4.set_ylim(-30, 30)
    ax4.set_ylabel('Strength', fontsize=9)
    ax4.grid(True, alpha=0.2)
    ax4.set_title('Panic-Reversal Signal (deadzone-filtered strength v4, +/-30; red=up, green=down; v=event, o=confirm)',
                  fontsize=9, loc='left', pad=2)

    # ── 第6面板:黄金坑 0/1 方波(坑内=1,其他=0)──
    try:
        import panic_reversal as _pr
        _fc = df_ohlc['close'].values.astype(np.float64)
        _frg2 = None
        if reg_preds_long is not None:
            _frg2 = np.asarray(reg_preds_long, dtype=np.float64)
        elif reg_preds is not None:
            _frg2 = np.asarray(reg_preds, dtype=np.float64)
        if _frg2 is None:
            from mean_reversion.signal_residual import compute_rolling_regression as _crr
            _frg2, _ = _crr(_fc, window=250, use_log=True)  # res 不含 reg,缺失时自算
        if _frg2 is not None:
            _fv = df_ohlc['volume'].values.astype(np.float64) if 'volume' in df_ohlc.columns else None
            _pits = _pr.detect_golden_pit(_fc, _frg2)
            _qual = _pr.compute_pit_quality(_pits, _fc, _fv) if _fv is not None else None
            _highpos = _pr.mark_high_pos(_pits, _fc)  # 高位坑软标注(坑前250日涨幅>150%)
            _super = _pr.mark_super_pits(_pits, _fc, _fv) if _fv is not None else None  # 超高胜率坑(93%)
            # 质量: 高度区分(strong=1.0 / normal=0.7 / weak=0.4) + 颜色辅助
            _qh = {'strong': 1.0, 'normal': 0.7, 'weak': 0.4}
            # 颜色对比拉开: strong=橙 / normal=蓝 / weak=灰(快启动=红,四种互不混淆)
            _qcolor = {'strong': '#E65100', 'normal': '#1565C0', 'weak': '#BDBDBD'}
            pit_mask = np.zeros(len(_fc))
            ax5.set_facecolor('#F5F5F5')
            for _k, (_s, _b, _lch) in enumerate(_pits):
                _end = _lch if _lch is not None else _b  # 坑画到出坑日(启动日);未出坑画到坑底
                _q = _qual[_k][2] if _qual is not None else 'normal'
                _lv = _qh.get(_q, 0.7)
                pit_mask[_s:_end + 1] = _lv
                _fast = _lch is not None and _lch - _b <= 5  # 快启动坑(坑底→出坑 ≤5 天,2026-08-15 由3放宽)
                _hp = _highpos[_k] if _highpos is not None and _k < len(_highpos) else False  # 高位坑软标注
                _sp = _super[_k] if _super is not None and _k < len(_super) else False  # 超高胜率坑(93%)
                _col = '#E53935' if _fast else _qcolor.get(_q, '#1565C0')
                # ⚠️ 必须用窗口坐标(0..n-1):折线 x 是窗口坐标,fill 用全局坐标(+offset)会画到窗口外被裁剪
                _x0w = _s - offset
                _x1w = _end - offset + 1  # steps-post 需右闭边界
                ax5.fill_between(np.arange(_x0w, _x1w + 1), 0, _lv, step='post',
                                 color=_col, alpha=0.85 if _fast else 0.60,
                                 hatch='//' if _hp else None)  # 高位坑加斜纹标注
                if _sp:  # 加仓确认(出坑后7天巨量):金色★标在 K 线加仓位(放量堆起点)
                    _vcl = _pr.detect_volume_clusters(_fc, _fv)
                    _add = None  # 加仓日(窗口坐标)
                    for _ss, _ee, _kk, _dd, _pp, _vv in _vcl:
                        if _kk == 'HIGH' and _lch is not None and _lch < _ss <= _lch + 7:
                            _add = _ss - offset
                            break
                    if _add is not None and 0 <= _add < n:
                        ax0.plot(_add, highs[_add] * 1.05, '*', color='#FF8F00', markersize=15,
                                 zorder=9, markeredgewidth=0.5, markeredgecolor='#B26A00')
                        ax0.annotate('加仓', (_add, highs[_add] * 1.05), textcoords='offset points',
                                     xytext=(0, 8), ha='center', fontsize=7, color='#B26A00',
                                     fontweight='bold', zorder=9)
            pit_win = pit_mask[offset:offset + n]
            ax5.plot(x, pit_win, drawstyle='steps-post', color='#0D47A1', linewidth=1.2)
            ax5.set_ylim(-0.1, 1.15)
            ax5.set_yticks([0, 0.4, 0.7, 1.0])
            ax5.set_yticklabels(['0', 'weak', 'normal', 'strong'], fontsize=6)
            ax5.set_ylabel('GOLD PIT', fontsize=8)
            ax5.grid(axis='y', alpha=0.3)
            ax5.legend(loc='upper left', fontsize=7)
    except Exception:
        pass

    # ── 统一 x 轴日期刻度（落在最底层面板）──
    ts2 = max(1, n // 12); dates = ohlc['date'].values
    tp = list(range(0, n, ts2)); tl = [str(dates[i])[:10] for i in tp]
    axes[-1].set_xticks(tp); axes[-1].set_xticklabels(tl, rotation=45, fontsize=7)
    # tight_layout 在渲染前基于最终内容计算间距，比 constrained_layout 更稳定可靠
    fig.tight_layout(pad=1.5)
    if save_path:
        plt.savefig(save_path, dpi=120); plt.close()
    else:
        plt.show(); plt.close()


# ============================================================
# 便捷入口
# ============================================================
def run_segmentation(df_ohlc, tail_days=200, name="",
                     lookback=15, min_reversal_pct=0.02, confirm_bars=3,
                     save_path=None, fast_mode=False, same_type_merge_gap=20,
                     dur_horizon=120, touch_norm=3,
                     reg_window=120, reg_window_long=250,
                     hide_ma=True,
                     code=None, end_date=None, panic_index=None):
    """fast_mode: True=跳过画图，返回 bool（最后一天有买入信号）。
    返回 (c_result, bs_signal, bs_reason, bs_strength, all_levels)；
    bs_strength 为 BrkLvl/BrkLow 的 0~1 分量评分；all_levels 为阻力/支撑位生命周期列表。

    reg_window : int, default 120
        中期回归窗口（交易日数），红色虚线。=0 不显示。
    reg_window_long : int, default 250
        长期回归窗口（交易日数），蓝色虚线。=0 不显示。
    hide_ma : bool, default True
        是否隐藏 MA120 和 EMA 线。"""
    close = df_ohlc['close'].values; volume = df_ohlc['volume'].values
    high = df_ohlc['high'].values; low = df_ohlc['low'].values; opn = df_ohlc['open'].values

    c_seg = CausalIncrementalPriceSegmenter(lookback=lookback, min_reversal_pct=min_reversal_pct,
                                            confirm_bars=confirm_bars,
                                            same_type_merge_gap=same_type_merge_gap)
    c_result = c_seg.segment(close, volume, high=high, low=low, opn=opn)
    bs_signal, bs_reason, bs_strength, all_levels = compute_buy_sell_signals(
        df_ohlc, c_result, dur_horizon=dur_horizon, touch_norm=touch_norm)

    if fast_mode: return bs_signal[-1] > 0

    # 滚动回归线
    reg_preds = None; reg_preds_long = None
    if reg_window > 0:
        from mean_reversion.signal_residual import compute_rolling_regression
        reg_preds, _ = compute_rolling_regression(close, window=reg_window)
    if reg_window_long > 0:
        from mean_reversion.signal_residual import compute_rolling_regression
        reg_preds_long, _ = compute_rolling_regression(close, window=reg_window_long)

    # 极速杀跌反转 signal(可选:传 code 自动计算,panic_index 可预传缓存)
    panic_info = None
    if code is None and name:
        import re
        m = re.search(r'(s[hz]\d{6}|\d{6})', str(name))   # 从 name 解析代码
        if m:
            c = m.group(1)
            if c.isdigit():                                # 纯6位数字 → 推断交易所前缀
                c = ('sh' if c[0] in '69' else 'sz') + c
            code = c
    if code is not None:
        try:
            from panic_reversal import signal as _panic_signal
            panic_info = _panic_signal(code, end_date=end_date)
        except Exception as _e:
            panic_info = {'code': code, 'signal': 'ERR', 'date': str(end_date or ''),
                          'reason': 'signal calc failed: %s' % _e}

    plot_price_segmentation_v10(df_ohlc, c_result, bs_signal, bs_reason,
                                tail_days=tail_days, name=name, save_path=save_path,
                                bs_strength=bs_strength, all_levels=all_levels,
                                reg_preds=reg_preds, reg_preds_long=reg_preds_long,
                                hide_ma=hide_ma,
                                reg_win=reg_window, reg_win_long=reg_window_long,
                                panic_info=panic_info)
    return c_result, bs_signal, bs_reason, bs_strength, all_levels
