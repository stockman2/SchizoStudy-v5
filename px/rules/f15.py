# -*- coding: utf-8 -*-
"""Ф15: рефрактерность, измеренная ГОДНОЙ статистикой — CV² промежутков внутри набора.

Вывод: при глухом периоде t_r и пуассоновском риске промежуток = t_r + Exp, откуда
CV² = (1 − t_r/g)², то есть t_r/g = 1 − CV; без глухого периода CV² = 1 РОВНО при любой
частоте. Величина безразмерна и считается ПО НАБОРУ, поэтому разнородность ансамбля в неё
не попадает. Нулевое значение при НАШЕМ n считается розыгрышем, а не берётся равным единице:
выборочная CV² при малом числе промежутков занижена.

Блоки те же, что у Ф14 и Ф16."""
import numpy as np
from ..consts import TAU_M
from .. import stats as st
from . import _blocks as B

NAME = 'Ф15'
NQ, NPER, NREP = 60, 8, 200


def needs(cfg): return B.defence_scan(cfg)


def run(ctx):
    S = ctx.S; mask = S['band']; rng = np.random.default_rng(555); rows = []
    for sg in B.SIGS:
        for ts in B.TAUS:
            tg = B.tag_ds(sg, ts); rr = ctx.b(tg)
            z = st.cv2_gaps(rr['gap_n'], rr['gap_sum'], rr['gap_sq'], mask)
            z.update(sigs=sg, tau_s_h=ts * 24, tag=tg, block=ctx.key(tg))
            rows.append(z)
    # контроли (б) и (в) идут через ТУ ЖЕ функцию, что и настоящее измерение
    def fake(ng, nper, gmean=4.0, trf=0.0):
        tr = trf * gmean; sc = gmean - tr
        g = tr + rng.exponential(sc, (ng, nper))
        return np.full(ng, float(nper)), g.sum(1), (g * g).sum(1)
    mk1 = np.ones(NQ, bool)
    nul = st.cv2_gaps(*fake(NQ, NPER), mask=mk1, nrep=400)['null']
    ctl_b = float(np.median([st.cv2_gaps(*fake(NQ, NPER), mask=mk1, nrep=1)['cv2'] for _ in range(NREP)]))
    ctl_c = float(np.median([st.cv2_gaps(*fake(NQ, NPER, trf=0.25), mask=mk1, nrep=1)['cv2'] for _ in range(NREP)]))
    work = next((z for z in rows if abs(z['sigs'] - B.SIGS[0]) < 1e-12 and abs(z['tau_s_h'] - 63.) < 0.6), None)
    chk = bool(work is not None and work['cv2'] == work['cv2'])
    va = (bool((work['null'] - work['cv2']) >= 0.10) if chk else 'непроверяемо')
    vb = bool(abs(ctl_b - nul) <= 0.03)
    vc = bool(ctl_c <= 0.50)
    vt = lambda v: 'выполнено' if v is True else ('не выполнено' if v is False else str(v))
    md = ['\n## %s. Рефрактерность годной статистикой\n' % NAME,
          'CV² = (1 − t_r/g)², нуль равен 1 ровно при любой частоте, глубина читается прямо: t_r/g = 1 − CV. '
          'Нулевое значение считается при НАШЕМ наборе n розыгрышем. Блоки те же, что у Ф14 и Ф16.\n',
          '\n### Правило, записано до счёта\n',
          '(а) Рефрактерность есть, если медиана CV² ниже нулевого значения не меньше чем на 0,10.\n'
          '(б) Контроль на ошибку: та же функция на экспоненциальных промежутках даёт свой нуль с точностью 0,03.\n'
          '(в) Контроль на мощность: с вшитым t_r = 0,25·g даёт не выше 0,50 (расчёт обещал около 0,42).\n'
          'Берутся наборы с не менее чем пятью промежутками.\n',
          '\n| sigma_м | память, ч | блок | наборов | CV² | нуль при том же n | разность | t_r/g |\n|---|---|---|---|---|---|---|---|']
    for z in rows:
        md.append('| %.2f | %.0f%s | %s | %d | %s | %s | %s | %s |' % (
            z['sigs'], z['tau_s_h'], ' (контроль)' if abs(z['tau_s_h'] - TAU_M * 24) < 0.6 else '',
            z['block'], z['n_sets'],
            ('%.3f' % z['cv2']) if z['cv2'] == z['cv2'] else '—',
            ('%.3f' % z['null']) if z['null'] == z['null'] else '—',
            ('%+.3f' % (z['cv2'] - z['null'])) if z['cv2'] == z['cv2'] else '—',
            ('%.2f' % z['tr_over_g']) if z['tr_over_g'] == z['tr_over_g'] else '—'))
    md += ['\n| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|',
           '| б: статистика без ошибки | та же функция даёт %.3f (медиана по %d повторам) против своего нуля %.3f | расходятся не больше 0,03 | %s |'
           % (ctl_b, NREP, nul, vt(vb)),
           '| в: статистика обладает мощностью | с вшитым t_r = 0,25·g даёт %.3f | не выше 0,50 | %s |' % (ctl_c, vt(vc)),
           '| а: рефрактерность есть | %s | ниже нуля не меньше чем на 0,10 | %s |'
           % (('CV² = %.3f против нуля %.3f, разность %+.3f' % (work['cv2'], work['null'], work['cv2'] - work['null']))
              if chk else 'наборов с пятью и более промежутками не нашлось', vt(va)),
           '\nЧасти (б) и (в) читаются первыми: если хоть одна не выполнена, в самом измерении ошибка, и строка (а) '
           'ничего не значит.']
    out = dict(rows=rows, a=dict(verdict=va), b=dict(ctl=ctl_b, null=nul, verdict=vb),
               c=dict(ctl=ctl_c, verdict=vc))
    ctx.verdict(NAME, 'а', work['cv2'] if chk else float('nan'), va, note='CV² промежутков внутри набора')
    ctx.verdict(NAME, 'б', ctl_b, vb, note='контроль на ошибку, через ту же функцию')
    ctx.verdict(NAME, 'в', ctl_c, vc, note='контроль на мощность, вшит t_r = 0,25·g')
    return out, md
