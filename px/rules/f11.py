# -*- coding: utf-8 -*-
"""Ф11: закрывает ли индукция эпизод. Сила индукции beta/N0 разыграна по наборам, поэтому
связь «сильнее индукция — короче эпизод» проверяется прямо по ансамблю, без перебора.
Связь со сном выключена (k = 0): меряется чистое действие защиты."""
import numpy as np
from scipy.stats import rankdata
from ..consts import TAU_M, TAU_S
from . import _blocks as B

NAME = 'Ф11'
BINS = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]


def needs(cfg): return [B.base_block(cfg)]


def run(ctx):
    S = ctx.S; rr = ctx.b('base'); bf = S['bfac']; band = S['band']
    epm = np.where(rr['epn'] > 0, rr['epsum'] / np.maximum(rr['epn'], 1), np.nan)
    sh = np.where(rr['epn'] > 0, rr['epshort'] / np.maximum(rr['epn'], 1), np.nan)
    okb = band & (rr['epn'] > 0)
    c_a = float(np.corrcoef(rankdata(bf[okb]), rankdata(epm[okb]))[0, 1]) if okb.sum() >= 10 else float('nan')
    hi = okb & (bf >= 2.0); lo = okb & (bf < 1.0)
    dur_hi = float(np.nanmedian(epm[hi])) if hi.any() else float('nan')
    dur_lo = float(np.nanmedian(epm[lo])) if lo.any() else float('nan')
    sh_hi = float(np.nanmedian(sh[hi])) if hi.any() else float('nan')
    sh_hi_pool = float(rr['epshort'][hi].sum() / max(rr['epn'][hi].sum(), 1)) if hi.any() else float('nan')
    emin = rr['epmin'][hi]; emin = emin[np.isfinite(emin)]
    flick = band & (rr['tdam'] > 0.5) & ((rr['Dmax'] - rr['Dmin']) > 0.2)
    va = bool(c_a == c_a and c_a <= -0.40)
    vb = bool(dur_hi == dur_hi and dur_hi <= 2.0 and sh_hi >= 0.30)
    vd = bool(float(flick.mean()) <= 0.25)
    vt = lambda v: 'выполнено' if v is True else 'не выполнено'
    rows = []
    for a_, b_ in BINS:
        m = okb & (bf >= a_) & (bf < b_); n = int(m.sum())
        rows.append(dict(lo=a_, hi=b_, n=n,
                         dur=float(np.nanmedian(epm[m])) if n else float('nan'),
                         short=float(np.nanmedian(sh[m])) if n else float('nan'),
                         short_pool=float(rr['epshort'][m].sum() / max(rr['epn'][m].sum(), 1)) if n else float('nan')))
    md = ['\n## %s. Закрывает ли индукция эпизод\n' % NAME,
          'Блок %s: k = 0, рабочая память защиты %.0f ч. Сила индукции разыграна по наборам '
          '(beta = N0·U(0; 3)), поэтому отклик на дозу виден прямо в ансамбле.\n' % (ctx.key('base'), TAU_S * 24),
          '\n| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|',
          '| а: длительность падает с силой индукции | ранговая связь %s | не слабее −0,40 | %s |'
          % (('%+.2f' % c_a) if c_a == c_a else '—', vt(va)),
          '| б: при измеренной силе (кратность ≥ 2) эпизод короткий | %.2f сут; коротких по наборам %.3f, '
          'по ВСЕМ эпизодам %.3f; самый короткий %s сут | не больше 2 сут и не меньше 0,30 | %s |'
          % (dur_hi, sh_hi, sh_hi_pool, ('%.2f' % emin.min()) if len(emin) else '—', vt(vb)),
          '| г: без вырождения в дребезг | %.3f наборов полосы | не больше 0,25 | %s |' % (float(flick.mean()), vt(vd)),
          '\nВердикт части (б) выносится по МЕДИАНЕ ПО НАБОРАМ — так записано правило. Эта статистика '
          'обнуляется, едва короткие эпизоды есть меньше чем у половины наборов, поэтому рядом стоит доля '
          'по всем эпизодам. Вердикт не пересматривается, но читать надо оба числа.',
          '\nУ наборов со слабой индукцией (кратность меньше 1) длительность %.2f суток.' % dur_lo,
          '\n| кратность индукции beta/N0 | наборов | длительность, сут | коротких: по наборам / по всем |\n|---|---|---|---|']
    for z in rows:
        md.append('| %.1f–%.1f | %d | %s | %s |' % (z['lo'], z['hi'], z['n'],
                  ('%.2f' % z['dur']) if z['n'] else '—',
                  ('%.3f / %.3f' % (z['short'], z['short_pool'])) if z['n'] else '—'))
    out = dict(a=dict(corr=c_a, thresh=-0.40, verdict=va),
               b=dict(dur_hi=dur_hi, dur_lo=dur_lo, short_hi=sh_hi, short_hi_pool=sh_hi_pool,
                      ep_min=float(emin.min()) if len(emin) else float('nan'), verdict=vb),
               d=dict(flicker=float(flick.mean()), verdict=vd), by_bfac=rows,
               tau_m_h=TAU_M * 24, tau_s_h=TAU_S * 24)
    ctx.verdict(NAME, 'а', c_a, va, note='ранговая связь силы индукции и длительности')
    ctx.verdict(NAME, 'б', dur_hi, vb, note='вердикт по медиане по наборам; по всем эпизодам %.3f' % sh_hi_pool)
    ctx.verdict(NAME, 'г', float(flick.mean()), vd, note='доля дребезжащих наборов полосы')
    return out, md
