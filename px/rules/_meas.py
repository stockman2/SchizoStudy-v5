# -*- coding: utf-8 -*-
"""Сводка по блоку и кривые. meas() ПЕРЕНЕСЕНА ИЗ mass_core6.py (разд. 2) слово в слово, кроме
одного: ND передаётся аргументом, а не берётся из глобальной области. Любая правка здесь меняет
числа сразу у девяти правил, поэтому править её можно только вместе с регрессионным тестом.
"""
import math
import numpy as np
from scipy.stats import rankdata


def meas(rr, ndays, mask=None):
    mk = np.ones(len(rr['nb']), bool) if mask is None else mask
    rr = {kk: (vv[mk] if isinstance(vv, np.ndarray) and vv.shape[:1] == mk.shape else vv) for kk, vv in rr.items()}
    ep = rr['epn'] > 0; ev = rr['everdam'] | rr['open_ep']; started = rr['nb'] + rr['open_ep'] * 1.0
    dur = np.where(ep, rr['epsum'] / np.maximum(rr['epn'], 1), np.nan)
    epm_ = rr['epn_m'] > 0; durm = np.where(epm_, rr['epsum_m'] / np.maximum(rr['epn_m'], 1), np.nan)
    return dict(n_sets=int(mk.sum()), share_ever=float(ev.mean()), share_break=float((rr['nb'] > 0).mean()),
                dur=float(np.nanmedian(dur)) if ep.any() else float('nan'),
                dur_arr=dur, n_ep=int(rr['epn'].sum()), n_ep_sets=int(ep.sum()),
                dur_m=float(np.nanmedian(durm)) if epm_.any() else float('nan'),
                short_m=float(np.nanmedian(np.where(epm_, rr['epshort_m'] / np.maximum(rr['epn_m'], 1), np.nan))) if epm_.any() else float('nan'),
                n_ep_m=int(rr['epn_m'].sum()),
                dur_all=float(np.nanmedian(np.where(ev, rr['tdam'] * ndays / np.maximum(started, 1), np.nan))) if (started > 0).any() else float('nan'),
                short=float(np.nanmedian(np.where(ep, rr['epshort'] / np.maximum(rr['epn'], 1), np.nan))) if ep.any() else float('nan'),
                short_pool=float(rr['epshort'].sum() / max(rr['epn'].sum(), 1)),
                ep_min=float(np.min(rr['epmin'][np.isfinite(rr['epmin'])])) if np.isfinite(rr['epmin']).any() else float('nan'),
                ep_min_med=float(np.median(rr['epmin'][np.isfinite(rr['epmin'])])) if np.isfinite(rr['epmin']).any() else float('nan'),
                sleep_end=float(rr['ep_sleep'].sum() / max(rr['epn'].sum(), 1)),
                sleep_kills=float(rr['ep_sleep'].sum() / max(rr['sl_in_ep'].sum(), 1)),
                n_sl_in_ep=float(rr['sl_in_ep'].sum()),
                late=float(rr['nb_late'].sum() / max(rr['nb_aw'].sum(), 1)), n_aw=float(rr['nb_aw'].sum()),
                trig=float(rr['nb_trig'].sum() / max(rr['nb'].sum(), 1)), n_br=float(rr['nb'].sum()),
                n_sr=float(rr['n_sr'].mean()),
                open_ep=float(rr['open_ep'].mean()), stuck=float((rr['wklong'] > 0).mean()), stuck15=float((rr['wk15'] > 0).mean()),
                wkmax=float(np.median(rr['wkmax'])), wkmean=float(np.median(np.where(rr['wkn'] > 0, rr['wksum'] / np.maximum(rr['wkn'], 1), np.nan))),
                per=float(np.nanmedian(rr['per'])), taw=float(np.median(rr['taw'])),
                tdam_ev=float(np.median(rr['tdam'][ev])) if ev.any() else 0.0,
                nday=float(np.median(rr['nday'][rr['nb'] > 0])) if (rr['nb'] > 0).any() else 0.0,
                nday_pool=float(rr['nb'].sum() / max(len(rr['nb']) * float(ndays), 1)),
                Dbar=float(np.median(rr['Dbar'])), Dmbar=float(np.median(rr['Dmbar'])),
                Ebar=float(np.median(rr['Ebar'])), Lbar=float(np.median(rr['Lbar'])),
                gap_mean=float(np.nansum(rr['gap_sum']) / max(np.nansum(rr['gap_n']), 1)), n_gap=int(rr['gap_n'].sum()),
                Dmin=float(np.median(rr['Dmin'])), Dmax=float(np.median(rr['Dmax'])),
                chronic=float((rr['Dmin'] > 0.1).mean()), swings=float((rr['Dmax'] - rr['Dmin'] > 0.2).mean()))


def dbin_curve(rr, mask):
    """Ф4(а): длина бодрствования по ячейкам доли повреждённых. Левая мера — все бодрствования
    в общей куче (вердикт по ней), правая — каждое поделено на СОБСТВЕННЫЙ здоровый уровень набора."""
    ds2, dn2 = [z[mask] for z in rr['dbin']]; hbs, hbn = [z[mask] for z in rr['hbase']]
    ds_, dnn = ds2.sum(0), dn2.sum(0)
    cur = np.where(dnn > 0, ds_ / np.maximum(dnn, 1), np.nan)
    own = np.where(hbn >= 3, hbs / np.maximum(hbn, 1), np.nan); okn = hbn >= 3
    rel = np.where((dn2 > 0) & okn[:, None], ds2 / np.maximum(dn2, 1) / np.maximum(own[:, None], 1e-9), np.nan)
    with np.errstate(invalid='ignore'):
        return cur, dnn, np.nanmean(rel, 0), (~np.isnan(rel)).sum(0), rel, ds2, dn2


def boot_ratio(ds2, dn2, j_hi, j_lo, nrep=2000, seed=17):
    """SE ПУЛОВОГО отношения двух ячеек, загрузкой ПО НАБОРАМ. Так и надо: ячейка 0 в долях
    собственного уровня равна единице у каждого набора по построению, поэтому разброс по этому
    столбцу равен нулю и SE из него получается нулевой — ровно эта ошибка и была здесь 14.10."""
    rng = np.random.default_rng(seed); n = ds2.shape[0]
    if n < 5: return float('nan')
    def rat(ix):
        a = ds2[ix, j_hi].sum() / max(dn2[ix, j_hi].sum(), 1e-9)
        b = ds2[ix, j_lo].sum() / max(dn2[ix, j_lo].sum(), 1e-9)
        return a / b if b > 0 else float('nan')
    bs = np.array([rat(rng.integers(0, n, n)) for _ in range(nrep)], float)
    bs = bs[np.isfinite(bs)]
    return float(np.std(bs)) if len(bs) > 10 else float('nan')


def lead_curve(rr, mask):
    """Ф4(б): то же по суткам до срыва, абсолютная кривая."""
    ls2, ln2 = [z[mask] for z in rr['lbin']]
    ls_, ln = ls2.sum(0), ln2.sum(0)
    return np.where(ln > 0, ls_ / np.maximum(ln, 1), np.nan), ln


def curve6(rr, mask=None):
    """Ф6 и Ф18: кривая предвестника — длина бодрствования в долях СВОЕГО здорового уровня,
    по суткам до срыва. Перенесено из mass_core6.py (curve6 в разд. 4в) дословно."""
    ls3, ln3 = rr['lbin']; hs3, hn3 = rr['hbase']
    if mask is not None:
        ls3, ln3, hs3, hn3 = ls3[mask], ln3[mask], hs3[mask], hn3[mask]
    own3 = np.where(hn3 >= 3, hs3 / np.maximum(hn3, 1), np.nan); ok3 = hn3 >= 3
    rel3 = np.where((ln3 > 0) & ok3[:, None], ls3 / np.maximum(ln3, 1) / np.maximum(own3[:, None], 1e-9), np.nan)
    with np.errstate(invalid='ignore'):
        return np.nanmean(rel3, 0), (~np.isnan(rel3)).sum(0), rel3


def rank_corr(y):
    """Ранговая связь величины с номером ячейки по заполненным ячейкам."""
    y = np.asarray(y, float); f = np.where(np.isfinite(y))[0]
    if len(f) < 4 or np.ptp(y[f]) == 0: return float('nan'), len(f)
    return float(np.corrcoef(rankdata(f), rankdata(y[f]))[0, 1]), len(f)


def col_se(rel, j):
    """Стандартная ошибка значения кривой в ячейке j: по наборам, которые в неё попали."""
    v = np.asarray(rel, float)[:, j]; v = v[np.isfinite(v)]
    return (float(np.std(v, ddof=1) / np.sqrt(len(v))), len(v)) if len(v) >= 3 else (float('nan'), len(v))


def mid_time(rr, ndays, nsets, dur, NHD=20, lo=0.3, hi=0.7):
    """Доля времени в середине шкалы (гистограммы hist накапливаются НА ВЕСЬ БЛОК, а не по наборам,
    поэтому ни маску полосы, ни разброс по наборам к ним применить нельзя).
    SE считается по ЭФФЕКТИВНОМУ числу независимых отсчётов: доля повреждённых держится примерно
    длительность эпизода, поэтому независимых отсчётов около nsel*ndays/dur, а не nsel*шагов.
    Это ОЦЕНКА СВЕРХУ по точности (то есть SE снизу), и так она и подписана в отчёте."""
    import math
    hD = np.asarray(rr['hist'][0], float); hDm = np.asarray(rr['hist'][1], float)
    hD = hD / max(hD.sum(), 1); hDm = hDm / max(hDm.sum(), 1)
    sl = slice(int(lo * NHD), int(hi * NHD))
    p = float(hD[sl].sum())
    neff = float(nsets) * float(ndays) / max(float(dur) if dur == dur else 1.0, 1e-3)
    se = math.sqrt(max(p * (1 - p), 1e-12) / max(neff, 1.0))
    return p, float(hDm[sl].sum()), se, neff, hD, hDm


def atten(rel, minb=4):
    """Коэффициент ослабления ранговой связи у ОДНОГО набора против сводной кривой.

    ЗАЧЕМ. Ранговая связь по сводной кривой считается на числе ЯЧЕЕК, а их около восьми: ошибка
    1/sqrt(7) = 0.378, MDE = 0.940, и порог −0,50 неразличим при любом числе наборов. Мельчить
    ячейки нельзя — кривая индексируется бодрствованиями, их примерно одно в сутки, а глубина
    просмотра ограничена межэпизодным промежутком. То есть это информационный предел, а не сетка.

    Выход — считать наклон У КАЖДОГО набора по его собственным ячейкам и проверять распределение
    наклонов по наборам: тогда ошибка берётся из разброса МЕЖДУ наборами и n — это сотни.

    НО ПОРОГ ПРИ ЭТОМ ДРУГОЙ, и взять прежние −0,50 было бы подлогом: кривая одного набора шумнее
    сводной, поэтому его связь ослаблена. Классическое ослабление при независимом шуме в ячейке:
        rho_набора = rho_сводной / sqrt(1 + lam),   lam = дисперсия шума ячейки / дисперсия сигнала,
    где дисперсия сигнала — разброс сводной кривой ПО ячейкам, а дисперсия шума — разброс значений
    в ячейке ПО наборам. Обе оцениваются из тех же данных ДО того, как мы посмотрели на наклоны,
    поэтому порог остаётся назначенным заранее, а не подогнанным.

    Возвращает (lam, множитель 1/sqrt(1+lam), число годных наборов)."""
    r = np.asarray(rel, float)
    ok = np.isfinite(r).sum(1) >= minb
    if ok.sum() < 3: return float('nan'), float('nan'), int(ok.sum())
    rr_ = r[ok]
    y = np.nanmean(rr_, 0)                                   # сводная кривая
    f = np.isfinite(y)
    if f.sum() < minb or np.nanstd(y[f]) == 0: return float('nan'), float('nan'), int(ok.sum())
    v_sig = float(np.nanvar(y[f], ddof=1))
    v_noi = float(np.nanmean([np.nanvar(rr_[:, j], ddof=1) for j in np.where(f)[0]
                              if np.isfinite(rr_[:, j]).sum() >= 3]))
    if not (v_sig > 0) or not np.isfinite(v_noi): return float('nan'), float('nan'), int(ok.sum())
    lam = v_noi / v_sig
    return lam, 1.0 / math.sqrt(1.0 + lam), int(ok.sum())


def rank_corr_sets(rel, minb=4):
    """Ранговая связь ПО КАЖДОМУ набору отдельно: среднее, ошибка из разброса между наборами, n.

    Ошибка считается по наборам, а не по ячейкам, поэтому n — сотни, а не восемь. Наборы, у которых
    заполнено меньше minb ячеек или кривая постоянна, выбрасываются и считаются отдельно: это не
    пропуск, а отсутствие информации, и число выброшенных возвращается."""
    r = np.asarray(rel, float); out = []
    for i in range(r.shape[0]):
        y = r[i]; f = np.where(np.isfinite(y))[0]
        if len(f) < minb or np.ptp(y[f]) == 0: continue
        out.append(float(np.corrcoef(rankdata(f), rankdata(y[f]))[0, 1]))
    a = np.asarray(out, float)
    if len(a) < 3: return float('nan'), float('nan'), len(a), int(r.shape[0])
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a))), len(a), int(r.shape[0])
