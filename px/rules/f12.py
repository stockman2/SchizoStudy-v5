# -*- coding: utf-8 -*-
"""Ф12: рефрактерный период, в ТОМ ВИДЕ, как правило было записано 12.10.

ВАЖНАЯ ОГОВОРКА, без которой это правило читать нельзя. Статистика частей (а) и (в) —
доля коротких промежутков, поделённая на «случайность без памяти», — ПРИЗНАНА НЕГОДНОЙ
14.10. Она разбирается аналитически до конца и равна 1 + CV²(частот эпизодов по наборам),
то есть измеряет разнородность ансамбля, а не физику. Поэтому вердикты по ней не являются
доводом ни в какую сторону. Годная версия той же проверки — правило Ф15 (CV² промежутков
внутри набора). Само правило здесь оставлено и считается по-прежнему: вердикты после счёта
мы не пересматриваем, негодность отмечается рядом.
"""
import math
import numpy as np
from ..consts import TAU_M, TAU_S
from . import _blocks as B

NAME = 'Ф12'
DEFECT = ('статистика признана негодной 14.10: равна 1 + CV²(частот), физики не видит; '
          'годная версия — Ф15')


def needs(cfg): return [B.base_block(cfg), B.no_memory_block(cfg)]


def _ratio(rr):
    gp = np.asarray(rr['gaps'], float); gt = float(gp.sum())
    gm = float(np.nansum(rr['gap_sum']) / max(np.nansum(rr['gap_n']), 1))
    ob = float(gp[:2].sum() / max(gt, 1)); ex = float(1 - math.exp(-2.0 / max(gm, 1e-9)))
    return dict(hist=gp, n=gt, mean=gm, obs=ob, exp=ex, ratio=ob / max(ex, 1e-9))


def run(ctx):
    rr = ctx.b('base'); ns = ctx.b('nomem')
    m = _ratio(rr); c = _ratio(ns); NGB = int(rr['NGB']); NSB = int(rr['NSB'])
    sAs, sAn = rr['s_after']; sHs, sHn = rr['s_health']
    sAs = np.asarray(sAs, float).sum(0) if np.asarray(sAs).ndim > 1 else np.asarray(sAs, float)
    sAn = np.asarray(sAn, float).sum(0) if np.asarray(sAn).ndim > 1 else np.asarray(sAn, float)
    sh0 = float(np.nansum(sHs) / max(np.nansum(sHn), 1))
    scur = np.where(sAn > 0, sAs / np.maximum(sAn, 1), np.nan) / max(sh0, 1e-9)
    above = [j for j in range(NSB) if sAn[j] > 0 and scur[j] >= 1.5]
    va = bool(m['n'] >= 50 and m['exp'] > 0 and m['ratio'] <= 0.5)
    vb = bool(above and max(above) >= 1)
    vc = bool(m['n'] >= 50 and c['n'] >= 50 and c['ratio'] >= 1.5 * m['ratio'] and (c['ratio'] - m['ratio']) >= 0.15)
    vt = lambda v: 'выполнено' if v is True else 'не выполнено'
    md = ['\n## %s. Рефрактерный период (правило в записанном виде)\n' % NAME,
          '**Статистика частей (а) и (в) признана негодной.** Она равна 1 + CV²(частот эпизодов по '
          'наборам), то есть измеряет разнородность ансамбля, а не память защиты. Годная версия — Ф15. '
          'Правило оставлено и считается по-прежнему: вердикты после счёта не пересматриваются, '
          'негодность отмечается рядом.\n',
          'Блоки: %s (рабочий) и %s (парный контроль, вторая ступень без длинной памяти, %.0f ч вместо %.0f).\n'
          % (ctx.key('base'), ctx.key('nomem'), TAU_M * 24, TAU_S * 24),
          '\n| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|',
          '| а: провал в начале распределения промежутков | %.3f против %.3f, отношение %.2f | отношение не больше 0,50 | %s |'
          % (m['obs'], m['exp'], m['ratio'], vt(va)),
          '| б: защита остаётся повышенной не меньше полутора суток | %s | уровень не ниже 1,5 от здорового | %s |'
          % (('до %d–%d суток' % (max(above), max(above) + 1)) if above else 'не держится', vt(vb)),
          '| в: провал пропадает без длинной памяти | отношение %.2f в модели против %.2f в контроле | '
          'в контроле не меньше чем в 1,5 раза больше И не меньше чем на 0,15 выше | %s |'
          % (m['ratio'], c['ratio'], vt(vc)),
          '\n- промежутков: %d в модели (средний %.2f сут), %d в контроле (средний %.2f сут)'
          % (int(m['n']), m['mean'], int(c['n']), c['mean']),
          '\n| суток между эпизодами | ' + ' | '.join('%d–%d' % (j, j + 1) for j in range(NGB)) + ' |\n|' + '---|' * (NGB + 1),
          '| доля, модель | ' + ' | '.join('%.3f' % (m['hist'][j] / max(m['n'], 1)) for j in range(NGB)) + ' |',
          '| доля, контроль | ' + ' | '.join('%.3f' % (c['hist'][j] / max(c['n'], 1)) for j in range(NGB)) + ' |',
          '\n| суток после эпизода | ' + ' | '.join('%d–%d' % (j, j + 1) for j in range(NSB)) + ' |\n|' + '---|' * (NSB + 1),
          '| защита, в долях здорового уровня | ' + ' | '.join(
              ('%.2f' % scur[j]) if sAn[j] > 0 else '—' for j in range(NSB)) + ' |']
    out = dict(a=dict(obs=m['obs'], exp=m['exp'], ratio=m['ratio'], n=int(m['n']), verdict=va, defect=DEFECT),
               b=dict(curve=[None if sAn[j] == 0 else float(scur[j]) for j in range(NSB)], verdict=vb),
               c=dict(ratio_model=m['ratio'], ratio_control=c['ratio'], n_control=int(c['n']),
                      gap_mean_model=m['mean'], gap_mean_control=c['mean'], verdict=vc, defect=DEFECT),
               gap_hist=[float(z) for z in m['hist']], gap_hist_control=[float(z) for z in c['hist']])
    ctx.verdict(NAME, 'а', m['ratio'], va, note=DEFECT)
    ctx.verdict(NAME, 'б', float(max(above) + 1) if above else 0.0, vb, note='уровень защиты после эпизода')
    ctx.verdict(NAME, 'в', c['ratio'], vc, note=DEFECT)
    return out, md
