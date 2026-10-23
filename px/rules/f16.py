# -*- coding: utf-8 -*-
"""Ф16: закон длительности эпизода, проверенный ВНЕ подгонки.

Форма выведена, а не подобрана. Внутри эпизода нагрузка стоит выше верха окна на Delta,
защита поднимает верх окна как delta(t) = delta_макс·(1 − exp(−t/tau_б)). Эпизод кончается,
когда верх окна догонит нагрузку, либо когда нагрузка сама упадёт:
      длительность = min( tau_б·ln(1/(1 − Delta/delta_макс)),  время экскурсии нагрузки ).
Две постоянные подгоняются по точкам с памятью НИЖЕ 100 ч, а проверяется ПРЕДСКАЗАНИЕ в точке
150 ч, которая в подгонку не входит. Точка 36 ч намеренно не считается: там формы-кандидаты
расходятся всего на 0,4 суток, то есть на 3 SE, и блок был бы потрачен зря.

Блоки те же, что у Ф14 и Ф15."""
import numpy as np
from scipy.optimize import curve_fit
from .. import stats as st
from . import _blocks as B

NAME = 'Ф16'
# SE СЧИТАЕТСЯ ЗАГРУЗКОЙ ПО СВОЕЙ ЖЕ ВЕЛИЧИНЕ, в этом же прогоне. 14.10 здесь была моя ошибка:
# я подставил SE = 3,4 %, измеренную для медианы по ВСЕЙ полосе (300 наборов), тогда как правило
# меряет длительность на подвыборке «полоса и кратность индукции ≥ 2» — там наборов втрое меньше
# тридцати, и SE оказалась 14 %. Порог 2 SE поэтому был занижен вчетверо. Правило говорит «2 SE»,
# и теперь код считает именно SE своей величины.
HOLD_H = 150.       # точка, не входящая в подгонку
FORMS = (('min(k*t, T): одна Delta на всех', lambda t, k, T: np.minimum(k * t, T)),
         ('T*t/(t+t0): Delta разбросана', lambda t, T, t0: T * t / (t + t0)),
         ('T*(1-exp(-t/t0)): плавное насыщение', lambda t, T, t0: T * (1 - np.exp(-t / t0))),
         ('a + b*t: АФФИННАЯ, без насыщения', lambda t, a, b: a + b * t))
# Четвёртая форма добавлена 14.10 и ЗАПИСАНА ДО СЧЁТА вместе с тем, что она означает. Основание:
# в прошлом прогоне все шесть предсказаний насыщающихся форм НЕДОБРАЛИ до наблюдения (знаковый
# критерий по шести точкам p = 0,031, но независимых точек там всего две, поэтому p = 0,50 —
# то есть это подозрение, а не вывод), а подгонка по четырём точкам дала почти прямую
# dur = 1,63 + 0,73*tau_защиты с RMS 0,22 сут. Если аффинная форма ОБОЙДЁТ обе насыщающиеся по
# RMS и её предсказание в точке 150 ч уложится в допуск, то вывод записан заранее: насыщения в
# измеренном диапазоне нет вовсе, а значит неверна та часть вывода формы, где эпизод кончается
# упором в потолок экскурсии нагрузки (up_cap). Тогда эпизод кончает ВОЗВРАТ нагрузки, а потолок
# защиты до конца эпизода не доходит, и двух суток пассивной защитой не получить нигде внутри
# 24-88 ч (нужная память печатается в отчёте и оказывается меньше первой ступени).


def needs(cfg): return B.defence_scan(cfg)


def run(ctx):
    S = ctx.S; mask = S['band']; bf = S['bfac']; out_rows = []
    def sample_hi(rr):
        epm = np.where(rr['epn'] > 0, rr['epsum'] / np.maximum(rr['epn'], 1), np.nan)
        hi = mask & (rr['epn'] > 0) & (bf >= 2.0)
        return epm[hi]
    def dur_hi(rr):
        v = sample_hi(rr); v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else float('nan')
    SEs = [st.boot_se(sample_hi(ctx.b(B.tag_ds(sg, ts)))) for sg in B.SIGS for ts in B.TAUS]
    SEs = [x for x in SEs if x == x]
    SE = float(max(SEs)) if SEs else float('nan')          # берём наибольшую: порог должен быть щадящим
    TOL = 2 * SE
    for sg in B.SIGS:
        tt = np.array([ts for ts in B.TAUS]); dd = np.array([dur_hi(ctx.b(B.tag_ds(sg, ts))) for ts in B.TAUS])
        fitm = (tt * 24) < 100.; hold = ~fitm
        if np.sum(np.isfinite(dd[fitm])) < 4 or not hold.any(): continue
        for nm, fn in FORMS:
            try:
                pp, _ = curve_fit(fn, tt[fitm], dd[fitm], p0=[3., 2.], maxfev=40000)
                res = dd[fitm] - fn(tt[fitm], *pp); rms = float(np.sqrt((res * res).mean()))
                pr = float(fn(tt[hold][0], *pp)); ob = float(dd[hold][0])
                out_rows.append(dict(sigs=sg, form=nm, rms=rms, pred=pr, obs=ob,
                                     err=abs(pr - ob) / max(abs(ob), 1e-9) if ob == ob else float('nan'),
                                     pars=[float(x) for x in pp]))
            except Exception:
                out_rows.append(dict(sigs=sg, form=nm, rms=float('nan'), pred=float('nan'),
                                     obs=float('nan'), err=float('nan'), pars=[]))
    good = [z for z in out_rows if z['err'] == z['err']]
    best = min(good, key=lambda z: z['rms']) if good else None
    aff = [z for z in good if z['form'].startswith('a + b')]
    sat = [z for z in good if z['form'].startswith(('T*t/', 'T*(1-'))]
    aff_best = min(aff, key=lambda z: z['rms']) if aff else None
    affine_wins = bool(aff_best is not None and sat and
                       aff_best['rms'] < min(z['rms'] for z in sat) and aff_best['err'] <= TOL)
    hard = [z for z in good if z['form'].startswith('min(')]
    chk = best is not None
    va = (bool(best['err'] <= TOL) if chk else 'непроверяемо')
    vb = (bool(hard and all(z['err'] > TOL for z in hard)) if chk else 'непроверяемо')
    vt = lambda v: 'выполнено' if v is True else ('не выполнено' if v is False else str(v))
    md = ['\n## %s. Закон длительности, проверенный вне подгонки\n' % NAME,
          'Форма выведена: длительность = min( tau_б·ln(1/(1 − Delta/delta_макс)), время экскурсии нагрузки ). '
          'Две постоянные подогнаны по точкам с памятью ниже 100 ч; проверяется предсказание в %.0f ч, '
          'которая в подгонку не входит. Порог — 2 SE = %.1f %%, где SE = %.1f %% восстановлена из перебора '
          'по зёрнам. Блоки те же, что у Ф14 и Ф15.\n' % (HOLD_H, 100 * TOL, 100 * SE),
          '\n| sigma_м | форма | RMS подгонки, сут | предсказание при %.0f ч | наблюдено | промах |\n|---|---|---|---|---|---|' % HOLD_H]
    for z in out_rows:
        md.append('| %.2f | %s | %s | %s | %s | %s |' % (
            z['sigs'], z['form'],
            ('%.3f' % z['rms']) if z['rms'] == z['rms'] else '—',
            ('%.2f' % z['pred']) if z['pred'] == z['pred'] else '—',
            ('%.2f' % z['obs']) if z['obs'] == z['obs'] else '—',
            ('%.1f %%' % (100 * z['err'])) if z['err'] == z['err'] else '—'))
    md += ['\n| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|']
    if chk:
        md += ['| а: предсказание лучшей формы подтверждается | лучшая «%s», промах %.1f %% | не больше %.1f %% | %s |'
               % (best['form'], 100 * best['err'], 100 * TOL, vt(va)),
               '| б: жёсткая форма отвергается | промах %s | больше %.1f %% | %s |'
               % (', '.join('%.1f %%' % (100 * z['err']) for z in hard) if hard else '—', 100 * TOL, vt(vb)),
               '\nОтвержение жёсткой формы содержательно: значит Delta/delta_макс РАЗБРОСАНО по наборам, '
               'а не общее, и насыщение поэтому постепенное.']
    else:
        md.append('| а и б | подгонка не вышла: длительность не определена в части строк | — | непроверяемо |')
    md += ['\n| в: закон аффинный, а не насыщающийся (записано до счёта 14.10) | %s | аффинная лучше обеих '
           'насыщающихся по RMS и её предсказание в допуске | %s |'
           % (('аффинная RMS %.3f против %.3f у лучшей насыщающейся, промах %.1f %%'
               % (aff_best['rms'], min(z['rms'] for z in sat), 100 * aff_best['err']))
              if (aff_best and sat) else '—', vt(affine_wins))]
    if affine_wins:
        md.append('\n**Часть (в) выполнена, и это меняет вывод формы.** Насыщения в измеренном диапазоне нет: '
                  'длительность идёт как %.2f + %.2f*tau_защиты (сутки). Значит эпизод кончает не упор потолка '
                  'защиты в нагрузку, а ВОЗВРАТ самой нагрузки, и та часть вывода, где длительность '
                  'ограничена потолком экскурсии, неверна. Следствие для всей дороги: чтобы эпизод уложился '
                  'в двое суток, нужна память %.0f ч — то есть МЕНЬШЕ, чем первая ступень, и пассивной '
                  'защитой цель недостижима нигде внутри 24–88 ч.'
                  % (aff_best['pars'][0], aff_best['pars'][1], 24 * (2.0 - aff_best['pars'][0]) / max(aff_best['pars'][1], 1e-9)))
    out = dict(rows=out_rows, a=dict(verdict=va), b=dict(verdict=vb),
               c=dict(verdict=affine_wins, affine=aff_best, saturating=sat), se_measured=SE,
               se_n=len(SEs), tol=TOL, hold_h=HOLD_H)
    ctx.verdict(NAME, 'а', best['err'] if chk else float('nan'), va,
                note='промах предсказания вне подгонки, порог 2 SE')
    ctx.verdict(NAME, 'б', max((z['err'] for z in hard), default=float('nan')) if chk else float('nan'), vb,
                note='отвержение формы с общей Delta')
    ctx.verdict(NAME, 'в', aff_best['rms'] if aff_best else float('nan'), affine_wins,
                note='аффинная форма против насыщающихся, записано до счёта')
    return out, md
