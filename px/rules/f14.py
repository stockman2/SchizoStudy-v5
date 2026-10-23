# -*- coding: utf-8 -*-
"""Ф14: есть ли у защиты собственная память. Перебор силы медленного шума на длину памяти
защиты, десять блоков. Те же блоки просят Ф15 и Ф16, поэтому считаются они один раз.

ОГОВОРКА: статистика частей (а) и (б) — та же негодная доля коротких промежутков, что в Ф12
(равна 1 + CV²(частот)). Годная версия — Ф15 на тех же блоках. Правило считается в записанном
виде; вердикты не пересматриваются.
Часть (в) считается в записанном виде — СТРОГАЯ монотонность, — а рядом без вердикта печатается
та же проверка с допуском в одну SE: 14.10 выяснилось, что строгая монотонность от величины с
SE около 3 % требует невозможного, и просадка в 0,4 SE была зачтена как нарушение.
"""
import math
import numpy as np
from ..consts import TAU_M, TAU_S
from .. import stats as st
from . import _blocks as B

NAME = 'Ф14'
DEFECT = 'статистика негодная (1 + CV²(частот)); годная версия — Ф15'


def needs(cfg): return B.defence_scan(cfg)


def row_stats(rr, mask, bf):
    gp = np.asarray(rr['gaps'], float); gt = float(gp.sum())
    gm = float(np.nansum(rr['gap_sum']) / max(np.nansum(rr['gap_n']), 1))
    ob = float(gp[:2].sum() / max(gt, 1)); ex = float(1 - math.exp(-2.0 / max(gm, 1e-9)))
    epm = np.where(rr['epn'] > 0, rr['epsum'] / np.maximum(rr['epn'], 1), np.nan)
    hi = mask & (rr['epn'] > 0) & (bf >= 2.0)
    return dict(n_gap=int(gt), gap_mean=gm, obs=ob, exp=ex, ratio=ob / max(ex, 1e-9),
                dur_hi=float(np.nanmedian(epm[hi])) if hi.any() else float('nan'), n_hi=int(hi.sum()))


def run(ctx):
    S = ctx.S; mask = S['band']; bf = S['bfac']; rows = []
    for sg in B.SIGS:
        for ts in B.TAUS:
            tg = B.tag_ds(sg, ts); rr = ctx.b(tg)
            z = row_stats(rr, mask, bf); z.update(sigs=sg, tau_s_h=ts * 24, tag=tg, block=ctx.key(tg))
            rows.append(z)
    pick = lambda sg, h: next((z for z in rows if abs(z['sigs'] - sg) < 1e-12 and abs(z['tau_s_h'] - h) < 0.6), None)
    m0, c0 = pick(B.SIGS[0], 63.), pick(B.SIGS[0], TAU_M * 24)
    mn, cn = pick(B.SIGS[1], 63.), pick(B.SIGS[1], TAU_M * 24)
    ok = all(z is not None and z['n_gap'] >= 50 for z in (m0, c0, mn, cn))
    va = (bool(m0['ratio'] <= 0.70 and c0['ratio'] >= 0.90) if ok else 'непроверяемо')
    vb = (bool(abs(mn['ratio'] - cn['ratio']) <= 0.15) if ok else 'непроверяемо')
    gm_samp = lambda rr: (np.asarray(rr['gap_sum'], float)[mask & (np.asarray(rr['gap_n'], float) > 0)]
                          / np.maximum(np.asarray(rr['gap_n'], float)[mask & (np.asarray(rr['gap_n'], float) > 0)], 1))
    ses = [st.boot_se(gm_samp(ctx.b(B.tag_ds(sg, ts)))) for sg in B.SIGS for ts in B.TAUS]
    ses = [x for x in ses if x == x]
    SE = float(max(ses)) if ses else 0.034      # SE СВОЕЙ величины, загрузкой в этом же прогоне
    def mono(seq, tol=0.0):
        return all(seq[i + 1] >= seq[i] * (1 - tol) - 1e-12 for i in range(len(seq) - 1))
    g0 = [z['gap_mean'] for z in sorted([z for z in rows if abs(z['sigs'] - B.SIGS[0]) < 1e-12], key=lambda z: z['tau_s_h'])]
    g1 = [z['gap_mean'] for z in sorted([z for z in rows if abs(z['sigs'] - B.SIGS[1]) < 1e-12], key=lambda z: z['tau_s_h'])]
    grow = (ok and m0['gap_mean'] >= 1.30 * c0['gap_mean'] and mn['gap_mean'] >= 1.30 * cn['gap_mean'])
    vc = (bool(mono(g0) and mono(g1) and grow) if ok else 'непроверяемо')
    vc_tol = (bool(mono(g0, SE) and mono(g1, SE) and grow) if ok else 'непроверяемо')
    vt = lambda v: 'выполнено' if v is True else ('не выполнено' if v is False else str(v))
    md = ['\n## %s. Есть ли у защиты собственная память\n' % NAME,
          'Десять блоков: %d силы медленного шума на %d длины памяти защиты, связь со сном выключена. '
          'Те же блоки просят Ф15 и Ф16, поэтому счёт один на три правила.\n' % (len(B.SIGS), len(B.TAUS)),
          '**Части (а) и (б) опираются на негодную статистику** (доля коротких промежутков = 1 + CV²(частот)); '
          'годная версия — Ф15 на этих же блоках. Правило считается в записанном виде.\n',
          '\n| sigma_м | память, ч | блок | промежутков | средний промежуток, сут | коротких | у случайности | отношение | длительность при кратности ≥ 2 |\n|---|---|---|---|---|---|---|---|---|']
    for z in rows:
        md.append('| %.2f | %.0f%s | %s | %d | %.2f | %.3f | %.3f | %.2f | %s |' % (
            z['sigs'], z['tau_s_h'], ' (контроль)' if abs(z['tau_s_h'] - TAU_M * 24) < 0.6 else '',
            z['block'], z['n_gap'], z['gap_mean'], z['obs'], z['exp'], z['ratio'],
            ('%.2f' % z['dur_hi']) if z['dur_hi'] == z['dur_hi'] else '—'))
    md += ['\n| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|']
    if ok:
        md += ['| а: при sigma_м = %.2f провал появляется | модель %.2f, контроль %.2f | модель не выше 0,70, контроль не ниже 0,90 | %s |'
               % (B.SIGS[0], m0['ratio'], c0['ratio'], vt(va)),
               '| б: при sigma_м = %.2f шум маскирует память | %.2f против %.2f, разница %.2f | разница не больше 0,15 | %s |'
               % (B.SIGS[1], mn['ratio'], cn['ratio'], abs(mn['ratio'] - cn['ratio']), vt(vb)),
               '| в: промежуток растёт с памятью | %.2f → %.2f сут (в %.2f раза) и %.2f → %.2f (в %.2f раза) | '
               'монотонно СТРОГО и не меньше чем в 1,3 раза | %s |'
               % (c0['gap_mean'], m0['gap_mean'], m0['gap_mean'] / max(c0['gap_mean'], 1e-9),
                  cn['gap_mean'], mn['gap_mean'], mn['gap_mean'] / max(cn['gap_mean'], 1e-9), vt(vc)),
               '\nБез вердикта, для чтения: та же часть (в) с допуском в одну SE (%.1f %%) на монотонность — **%s**. '
               'Строгая монотонность от величины с таким шумом требует невозможного, но вердикт стоит по '
               'записанному правилу и не пересматривается.' % (100 * SE, vt(vc_tol))]
    else:
        md.append('| все части | промежутков меньше 50 хотя бы в одной нужной строке | не меньше 50 | непроверяемо |')
    out = dict(rows=rows, a=dict(verdict=va, defect=DEFECT), b=dict(verdict=vb, defect=DEFECT),
               c=dict(verdict=vc, verdict_with_se_tolerance=vc_tol, se=SE), checkable=ok)
    ctx.verdict(NAME, 'а', m0['ratio'] if ok else float('nan'), va, note=DEFECT)
    ctx.verdict(NAME, 'б', abs(mn['ratio'] - cn['ratio']) if ok else float('nan'), vb, note=DEFECT)
    ctx.verdict(NAME, 'в', m0['gap_mean'] / max(c0['gap_mean'], 1e-9) if ok else float('nan'), vc,
                note='строгая монотонность; с допуском 1 SE было бы %s' % vt(vc_tol))
    return out, md
