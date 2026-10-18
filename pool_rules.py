# -*- coding: utf-8 -*-
"""每日股票池的纯状态机与分级规则（2026-10-18）

设计目标
--------
把"出池五规则"从 IO/行情计算里剥离出来，做成无依赖纯函数：
  · daily_pipeline.py 负责取数、画像、写 CSV；
  · 本模块只负责"给定今天的指标，池状态应该变成什么"；
  · 这样规则可以在没有 eltdx / 行情数据的环境里用合成数据单测。

五条出池规则（golden-pit-strategy skill 定版）
----------------------------------------------
1. 质量甄别：入池（early 日）后 WATCH_DAYS=20 个交易日内未出现大阳确认
   → 归档（quality_no_bigline_20d）。入池当天即参与，不延迟入场；
   大阳只能做事后质量甄别，不能作为买入前提。
2. 量能证伪：大阳确认后满 10 日，且后 10 日均量 / 大阳前 20 日均量 < 1.15
   → 归档（volume_weakened）。
3. 结构破坏：收盘跌破 COST35(Ehlers45) 且前一日也在 COST35 下方（2 日未收回）
   → 归档（cost35_broken）。
4. 到期归档：大阳确认后满 TRACK_DAYS=60 个交易日 → 归档（expired_60d）。
5. 人工标记：manual_status == 'skip' → 归档（manual）。

优先级（重要）
--------------
manual > 已归档 > 到期 > 结构破坏 > 量能证伪 > 质量甄别。
到期是强制规则，不应被其他条件掩盖；结构破坏优先于量能，因为它是硬风险。
"""

WATCH_DAYS = 20          # 入池后等待大阳确认的交易日数
TRACK_DAYS = 60          # 大阳确认后的最长跟踪交易日数
VOL_WEAK = 1.15          # 大阳后量能证伪阈值
VOL_CHECK_DAYS = 10      # 大阳后检查量能的交易日数
BIG_PCT = 0.05           # 大阳线单日涨幅下限
BIG_PCT_MAX = 0.21       # 上限（防除权/错价）

PHASE_WATCHING = 'watching'
PHASE_TRACKING = 'tracking'
PHASE_ARCHIVED = 'archived'

REASON_TEXT = {
    'manual': '人工标记不关注',
    'quality_no_bigline_20d': '入池20日未出大阳（质量甄别）',
    'volume_weakened': '大阳后10日量能<1.15（延续段缩量）',
    'cost35_broken': '连续2日跌破COST35（结构破坏）',
    'expired_60d': '确认后满60个交易日（到期）',
    'data_missing': '行情数据缺失/信号日不在数据中',
}


def next_phase(phase, days_in_pool, big_found, days_since_big=None,
               post_big_vol_ratio=None, close=None, cost35=None,
               prev_close=None, prev_cost35=None, manual_skip=False):
    """返回 (new_phase, archive_reason 或 None, note 或 None)。

    参数全部来自"截至今天"的数据；本函数不取数、不看未来。
    """
    if phase == PHASE_ARCHIVED:
        return PHASE_ARCHIVED, None, 'already_archived'
    if manual_skip:
        return PHASE_ARCHIVED, 'manual', None

    # 4) 到期（强制，优先级最高）
    if phase == PHASE_TRACKING and days_since_big is not None and days_since_big >= TRACK_DAYS:
        return PHASE_ARCHIVED, 'expired_60d', None

    # 3) 结构破坏（硬风险）——只在已确认(tracking)阶段生效：
    #    watching 阶段的回踩是 v3b 的正常路径（突破后深回踩再启动），
    #    若 early 入池即用 COST35 出池会误杀二次买点。
    if phase == PHASE_TRACKING and _both_below(close, cost35, prev_close, prev_cost35):
        return PHASE_ARCHIVED, 'cost35_broken', None

    if phase == PHASE_WATCHING:
        # 1) 质量甄别 / 确认
        if big_found:
            return PHASE_TRACKING, None, 'bigline_confirmed'
        if days_in_pool is not None and days_in_pool >= WATCH_DAYS:
            return PHASE_ARCHIVED, 'quality_no_bigline_20d', None
        return PHASE_WATCHING, None, None

    if phase == PHASE_TRACKING:
        # 2) 量能证伪（大阳后满 10 日才判定）
        if (days_since_big is not None and days_since_big >= VOL_CHECK_DAYS
                and post_big_vol_ratio is not None
                and post_big_vol_ratio < VOL_WEAK):
            return PHASE_ARCHIVED, 'volume_weakened', None
        return PHASE_TRACKING, None, None

    # 未知 phase：保守地当作观察层
    return PHASE_WATCHING, None, 'unknown_phase_reset'


def _both_below(close, cost35, prev_close, prev_cost35):
    vals = (close, cost35, prev_close, prev_cost35)
    if any(v is None for v in vals):
        return False
    try:
        vals = [float(v) for v in vals]
    except (TypeError, ValueError):
        return False
    if any(v != v for v in vals):   # NaN
        return False
    return vals[0] < vals[1] and vals[2] < vals[3]


def grade_of(vol_ratio=None, big_pct=None, cool_days=None, profit=None,
             dev35=None, has_big=False):
    """A/B/C 质量分层（不硬过滤，只标注）。

    A 档（需要大阳确认）: vol_ratio>=1.5 且 big_pct>=7% 且 cool_days>=40
                          且 profit>=0.80 且 dev35∈[5,35]
    B 档（基础达标）:     vol_ratio>=1.2 且 cool_days>=20 且 profit>=0.60
                          且（无大阳 或 big_pct>=5%）
    C 档: 其余（缩量大阳/贴线/高位追涨/获利低/冷却短）
    A 是 B 的子集。
    """
    def num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if f != f else f

    vr, bp, cd, pf, dv = (num(vol_ratio), num(big_pct), num(cool_days),
                          num(profit), num(dev35))
    if has_big and None not in (vr, bp, cd, pf, dv):
        if vr >= 1.5 and bp >= 7.0 and cd >= 40 and pf >= 0.80 and 5.0 <= dv <= 35.0:
            return 'A'
    if None not in (vr, cd, pf):
        if vr >= 1.2 and cd >= 20 and pf >= 0.60 and (not has_big or (bp is not None and bp >= 5.0)):
            return 'B'
    return 'C'


def reason_text(reason):
    return REASON_TEXT.get(reason, reason or '')
