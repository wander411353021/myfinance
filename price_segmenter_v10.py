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
                                reg_preds=None):
    """4面板: K线 + 成交量 + 买卖信号(柱高=突破分量) + 阻力/支撑位生命周期。"""
    ohlc = df_ohlc.tail(tail_days).copy().reset_index(drop=True)
    n = len(ohlc); x = np.arange(n); offset = len(df_ohlc) - n  # ohlc 是 df_ohlc 末尾 n 行，offset 为其在原序列中的起始下标（恒 >=0）

    fig, axes = plt.subplots(4, 1, figsize=(22, 15),
                             sharex=True,
                             gridspec_kw={'height_ratios': [4, 1.4, 0.9, 1.4]})
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
    ma120 = pd.Series(fc).rolling(120, min_periods=1).mean().values[-tail_days:]
    ax0.plot(x, ma120, color='#7B1FA2', linewidth=1.2, alpha=0.8, label='MA120')
    sm = result['smooth'].values[offset:offset + n]
    ax0.plot(x, sm, color='#1565C0', linewidth=1.0, alpha=0.6, label='EMA')
    if reg_preds is not None:
        rp = reg_preds[offset:offset + n]
        ax0.plot(x, rp, color='#d62728', linewidth=2.0, linestyle='--',
                 alpha=0.85, label='Regression')

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
    # 仅当极值属“离群”（>4× 或 <0.12× 近期价）才封顶；健康区间（如 5-30）保持原样不被过度收窄。
    # ax3 复用该区间，使两轴范围一致。
    _last_cv = closes[-1]
    _y_hi = highs.max()
    if _y_hi > _last_cv * 4:
        _near = highs[highs <= _last_cv * 4]
        _y_hi = _near.max() if _near.size else _last_cv * 4
    _y_lo = lows.min()
    if _y_lo < _last_cv * 0.12:
        _near = lows[lows >= _last_cv * 0.12]
        _y_lo = _near.min() if _near.size else _last_cv * 0.12
    _focus_lo, _focus_hi = _y_lo, _y_hi
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
        Line2D([0], [0], color='#7B1FA2', linewidth=1.2, label='MA120'),
        Line2D([0], [0], color='#1565C0', linewidth=1.0, label='EMA'),
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
    # 量轴：取窗口内实际最大值 * 1.2，完整显示 + 20% 顶空
    if len(vol) > 0 and vol.max() > 0:
        ax1.set_ylim(0, vol.max() * 1.2)
    ax1.set_ylabel('Volume', fontsize=9); ax1.grid(True, alpha=0.2)
    ax1.set_title('Volume (red=expanding, green=shrinking)', fontsize=9, loc='left', pad=2)

    # ── 信号面板（ax2）：柱高编码突破分量；顶端圆点大小随分量增大 ──
    ax2 = axes[2]
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

    # ── 阻力/支撑位生命周期面板（ax3）：★形成 · ▽/▲测试 · ●突破 ──
    ax3 = axes[3]
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
                     reg_window=0):
    """fast_mode: True=跳过画图，返回 bool（最后一天有买入信号）。
    返回 (c_result, bs_signal, bs_reason, bs_strength, all_levels)；
    bs_strength 为 BrkLvl/BrkLow 的 0~1 分量评分；all_levels 为阻力/支撑位生命周期列表。

    reg_window : int, default 0
        滚动回归窗口（交易日数）。>0 时在面板0叠加红色虚线回归线。
        推荐 120（平滑），调大(180-250) 更平滑，调小(40-60) 更敏感。"""
    close = df_ohlc['close'].values; volume = df_ohlc['volume'].values
    high = df_ohlc['high'].values; low = df_ohlc['low'].values; opn = df_ohlc['open'].values

    c_seg = CausalIncrementalPriceSegmenter(lookback=lookback, min_reversal_pct=min_reversal_pct,
                                            confirm_bars=confirm_bars,
                                            same_type_merge_gap=same_type_merge_gap)
    c_result = c_seg.segment(close, volume, high=high, low=low, opn=opn)
    bs_signal, bs_reason, bs_strength, all_levels = compute_buy_sell_signals(
        df_ohlc, c_result, dur_horizon=dur_horizon, touch_norm=touch_norm)

    if fast_mode: return bs_signal[-1] > 0

    # 滚动回归线（reg_window > 0 时启用）
    reg_preds = None
    if reg_window > 0:
        from mean_reversion.signal_residual import compute_rolling_regression
        reg_preds, _ = compute_rolling_regression(close, window=reg_window)

    # if save_path is None:
    #     save_path = f'E:\\\\chip_analyzer_ui\\\\new_algo\\\\result\\\\{name}_price_v10.png'
    plot_price_segmentation_v10(df_ohlc, c_result, bs_signal, bs_reason,
                                tail_days=tail_days, name=name, save_path=save_path,
                                bs_strength=bs_strength, all_levels=all_levels,
                                reg_preds=reg_preds)
    return c_result, bs_signal, bs_reason, bs_strength, all_levels
