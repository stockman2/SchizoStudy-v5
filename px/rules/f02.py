# -*- coding: utf-8 -*-
"""Ф2: суточный член нагрузки. Три части, и у каждой свой ужесточённый двойник.

ЧТО ПЕРЕНЕСЕНО КАК ЗАПИСАНО
  (а) доля срывов во второй половине бодрствования > 0,5;  (б) есть порог по размаху A*:
  доля повреждавшихся растёт не плавно, а с переломом (рост не меньше 4 раз и один разрыв
  выделяется);  (в) сон редко кончает эпизод: доля эпизодов, кончающихся во сне, < 0,3.

ВЫКЛАДКИ, ИЗ КОТОРЫХ СДЕЛАНЫ ДВОЙНИКИ (подробно в px/predict.py)
  (а′) Порог 0,5 — это ровно равномерность срывов по бодрствованию, то есть НУЛЕВОЕ значение,
       а не ожидание механизма. Ожидание считается: нагрузка к концу бодрствования выше на
       L = ln(day_pk), барьер ближе на L/sigma сигм (печатается в отчёте), частота идёт как exp(-d^2/2), откуда
       late = int_{w/2}^{w} h / int_0^{w} h. Это и есть порог двойника.
  (б′) В модели опасности ворот нет ПО ПОСТРОЕНИЮ: доля повреждавшихся идёт гладко как
       1 - exp(-lam*H(a)), H(a) — интеграл опасности по суткам при размахе a*L. Предэкспонента
       снимается одной привязкой в рабочей точке a = 1. Если наблюдённая кривая ложится на
       предсказанную гладкую, то порога нет, и прежний отрицательный результат подтверждён
       численно. Проверяется наибольшее отклонение против 2 SE доли.
  (в′) Порог 0,3 почти совпадает с равномерным ожиданием 1 - w (доля сна в сутках): если концы эпизодов
       разбросаны по суткам как попало, во сне их окажется ровно доля сна. Значит правило как
       записано почти не имеет хода и ничего не различает. Двойник сравнивает с 1 - w и говорит
       одно из трёх: во сне эпизоды кончаются РЕЖЕ случайного (как ждало правило), ЧАЩЕ
       (сон их обрывает) или неразличимо.
"""
import numpy as np
from .. import predict as P
from .. import stats as st
from . import _blocks as B
from ._meas import meas

NAME = 'Ф2'


def needs(cfg): return [B.base_block(cfg)] + B.swing_scan(cfg)


def run(ctx):
    S = ctx.S; cfg = ctx.cfg; nd = cfg.d['run']['days']; band = S['band']
    r0 = meas(ctx.b('base'), nd); r0b = meas(ctx.b('base'), nd, band)
    pr = P.summary(S, cfg)
    rows = [dict(af=af, **{k: v for k, v in meas(ctx.b(B.tag_a(af)), nd).items() if k != 'dur_arr'}) for af in B.ASCAN]
    sh = [z['share_ever'] for z in rows]

    # ---- (а) и (а′) ----
    se_a = st.prop_se(r0['late'], r0['n_aw'])
    a = st.adjudicate(r0['late'], 0.5, se_a, 'ge', n=int(r0['n_aw']))
    pred_a = float(np.nanmedian(P.late_share(S)[band]))
    se_ab = st.prop_se(r0b['late'], r0b['n_aw'])
    ap = st.adjudicate(r0b['late'], pred_a, se_ab, 'ge', null=0.5, n=int(r0b['n_aw']))

    # ---- (б) и (б′) ----
    gro = max(sh) / max(min(sh), 1e-9)
    dif = np.diff(sh); kink = bool(len(dif) and max(dif) > 2.5 * np.median(np.abs(dif)))
    se_sh = st.prop_se(float(np.mean(sh)), S['nsel'])
    bq = st.adjudicate(gro, 4.0, st.mde(se_sh) / max(min(sh), 1e-9) / st.MDE_K, 'ge', null=1.0, n=S['nsel'])
    bq['verdict'] = ('непроверяемо' if bq['verdict'] == 'непроверяемо' else bool(bq['raw_verdict'] and kink))
    anc = [z for z in rows if abs(z['af'] - 1.0) < 1e-9][0]['share_ever']
    pcur = P.share_ever_pred(S, B.ASCAN, nd, anc)
    dev = {af: abs(pcur[af] - sh[i]) for i, af in enumerate(B.ASCAN)}
    worst = max(dev, key=lambda k: dev[k])
    bp = st.adjudicate(dev[worst], st.eq_tol(se_sh), se_sh, 'le', null=0.0, n=S['nsel'])

    # ---- (в) и (в′) ----
    se_c = st.prop_se(r0['sleep_end'], r0['n_ep'])
    c = st.adjudicate(r0['sleep_end'], 0.3, se_c, 'le', n=int(r0['n_ep']))
    nul_c = pr['sleep_end_null']
    # Размер эффекта записан заранее и не зависит от нашей SE: если сон эпизод НЕ кончает,
    # концы должны быть реже случайных хотя бы на четверть. Четверть — наименьшее отличие,
    # которое в литературе по суточному распределению событий считают содержательным.
    cp = st.adjudicate(r0['sleep_end'], 0.75 * nul_c, se_c, 'le', null=nul_c, n=int(r0['n_ep']), lim=(0., 1.))
    enrich = bool(r0['sleep_end'] >= nul_c + st.mde(se_c))

    vt = lambda v: 'выполнено' if v is True else ('не выполнено' if v is False else str(v))
    md = ['\n## %s. Суточный член нагрузки\n' % NAME,
          'Блок %s (k = 0, рабочий шум) и перебор размаха %s. Зазор до барьера у медианного набора '
          'полосы %.2f сигмы, суточный подъём %.2f сигмы — из этих двух чисел и сделаны все ожидания ниже.\n'
          % (ctx.key('base'), ', '.join(ctx.key(B.tag_a(af))[:6] for af in B.ASCAN if af != 1.0),
             pr['gap_sigma'], pr['day_rise_sigma']),
          '\n| часть | получено | порог | нулевое значение | SE | MDE | вывод |\n|---|---|---|---|---|---|---|',
          '| а: срывы во второй половине бодрствования (как записано) | %.3f (%d срывов) | больше 0,5 | 0,5 (равномерно) | %.3f | %.3f | %s |'
          % (r0['late'], int(r0['n_aw']), se_a, a['mde'], vt(a['verdict'])),
          '| а′: не ниже аналитического ожидания | %.3f по полосе | %.3f (выкладка) | 0,5 | %.3f | %.3f | %s |'
          % (r0b['late'], pred_a, se_ab, ap['mde'], vt(ap['verdict'])),
          '| б: есть порог по размаху (как записано) | рост %.2f раза, перелом %s | рост не меньше 4 и перелом | 1,0 | %.4f на долю | — | %s |'
          % (gro, 'есть' if kink else 'нет', se_sh, vt(bq['verdict'])),
          '| б′: кривая ложится на гладкое предсказание без ворот | худшее отклонение %.4f при a = %.2f | не больше %.4f (3 SE, допуск равносильности) | 0 | %.4f | %.4f | %s |'
          % (dev[worst], worst, st.eq_tol(se_sh), se_sh, bp['mde'], vt(bp['verdict'])),
          '| в: сон редко кончает эпизод (как записано) | %.3f (%d эпизодов) | меньше 0,3 | 0,3 | %.3f | %.3f | %s |'
          % (r0['sleep_end'], int(r0['n_ep']), se_c, c['mde'], vt(c['verdict'])),
          '| в′: реже случайного хотя бы на четверть | %.3f | не больше %.3f | %.3f (доля сна в сутках) | %.3f | %.3f | %s |'
          % (r0['sleep_end'], 0.75 * nul_c, nul_c, se_c, cp['mde'], vt(cp['verdict'])),
          '\n### Перебор размаха суточной нагрузки (k = 0)\n',
          '| множитель при A_H | доля повреждавшихся | предсказание без ворот | отклонение | срывов во второй половине | эпизодов короче суток (все эпизоды) |\n|---|---|---|---|---|---|']
    for i, z in enumerate(rows):
        md.append('| %.2f%s | %.4f | %.4f | %+.4f | %.3f | %.3f |' % (
            z['af'], ' (привязка)' if abs(z['af'] - 1) < 1e-9 else '', z['share_ever'],
            pcur[z['af']], sh[i] - pcur[z['af']], z['late'], z['short_pool']))
    md += ['\nПривязка — одно число: предэкспонента опасности взята из точки a = 1. Все остальные точки '
           'кривой предсказаны, а не подогнаны.\n']
    if enrich:
        md.append('\n**Часть (в) переворачивается двойником.** Во сне эпизоды кончаются ЧАЩЕ, чем вышло бы '
                  'при случайном разбросе концов (%.3f против %.3f, зазор %.1f SE). То есть сон эпизод не '
                  'пропускает, а ОБРЫВАЕТ, и правило как записано спрашивало не то. Вторая мера, доля ночей '
                  'внутри эпизода, которые его оборвали: %.3f на %d ночах.'
                  % (r0['sleep_end'], nul_c, abs(r0['sleep_end'] - nul_c) / max(se_c, 1e-9),
                     r0['sleep_kills'], int(r0['n_sl_in_ep'])))
    out = dict(a=a, a_tight=dict(pred=pred_a, **ap), b=dict(growth=gro, kink=kink, shares=sh, **bq),
               b_tight=dict(pred=pcur, dev=dev, worst_af=worst, **bp),
               c=c, c_tight=dict(null_uniform=nul_c, enriched=enrich, sleep_kills=r0['sleep_kills'], **cp),
               rows=rows, analytic=pr)
    ctx.verdict(NAME, 'а', r0['late'], a['verdict'], note='нулевое значение 0,5 — равномерность по бодрствованию')
    ctx.verdict(NAME, 'а′', r0b['late'], ap['verdict'], note='порог — аналитическое ожидание %.3f' % pred_a)
    ctx.verdict(NAME, 'б', gro, bq['verdict'], note='рост и перелом, как записано')
    ctx.verdict(NAME, 'б′', dev[worst], bp['verdict'], note='согласие с гладким предсказанием без ворот')
    ctx.verdict(NAME, 'в', r0['sleep_end'], c['verdict'], note='порог 0,3 почти равен равномерному %.3f' % nul_c)
    ctx.verdict(NAME, 'в′', r0['sleep_end'], cp['verdict'],
                note='равномерное ожидание %.3f; обогащение: %s' % (nul_c, 'да' if enrich else 'нет'))
    return out, md
