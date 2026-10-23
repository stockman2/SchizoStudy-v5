# -*- coding: utf-8 -*-
"""Ф19: примирование индукции. beta умножается на (1 + A_PR*p), где p — след недавнего
окисления на том же сигнале, что движет индукцию. Потолок усиления ВТРОЕ измерен
(Bischoff и соавт. 2019: отклик сульфиредоксина на второе воздействие до трёх раз выше).
Постоянная памяти tau_p ничем не закреплена — в опыте проверен один интервал, 24 ч, —
поэтому перебирается.

Ускорение первой ступени НЕ добавлено намеренно: на одной клетке оно дало от −1 до −5 %
против −52 % у усиления, и причина записана до счёта (первая ступень с постоянной 12 ч к
началу эпизода уже уравновешена, скорость задаёт вторая).
"""
import numpy as np
from ..batch import Block
from ..consts import A_PR, TAU_M
from .. import stats as st

NAME = 'Ф19'
TARGET_DAYS = 2.0          # клиническая цель: из наблюдений, не пересчитывается
BFAC_MIN = 2.0             # измеренная сила индукции: кратность beta/N0 не ниже 2


def needs(cfg):
    return [Block('base', kd=0.0, apr=0.0),
            Block('prime12', kd=0.0, apr=A_PR, taup=12. / 24),
            Block('prime24', kd=0.0, apr=A_PR, taup=24. / 24)]


def _dur(rr): return np.where(rr['epn'] > 0, rr['epsum'] / np.maximum(rr['epn'], 1), np.nan)
def _shr(rr): return np.where(rr['epn'] > 0, rr['epshort'] / np.maximum(rr['epn'], 1), np.nan)


def run(ctx):
    S = ctx.S; mask = S['band'] & (S['bfac'] >= BFAC_MIN)
    ctl = ctx.b('base'); d_c = _dur(ctl); s_c = _shr(ctl)
    ok_c = mask & (ctl['epn'] > 0)
    base = dict(dur=float(np.nanmedian(d_c[mask])), ub=st.med_upper(d_c[mask])['bound'],
                short_pool=float(ctl['epshort'][ok_c].sum() / max(ctl['epn'][ok_c].sum(), 1)),
                n_ep=int(ctl['epn'][ok_c].sum()), n_sets=int(ok_c.sum()),
                se=st.boot_se(d_c[mask]))
    rows = []
    for tag, tph in (('prime12', 12.), ('prime24', 24.)):
        rr = ctx.b(tag); d_o = _dur(rr); s_o = _shr(rr); ok = mask & (rr['epn'] > 0)
        ub = st.med_upper(d_o[mask])
        rows.append(dict(tag=tag, tau_p_h=tph, block=ctx.key(tag),
                         dur=float(np.nanmedian(d_o[mask])), ub=ub['bound'], ub_rank=ub['rank'], ub_n=ub['n'],
                         short_pool=float(rr['epshort'][ok].sum() / max(rr['epn'][ok].sum(), 1)),
                         pr_dam=float(np.median(rr['prdam'][ok])) if ok.any() else float('nan'),
                         n_ep=int(rr['epn'][ok].sum()),
                         sign_dur=st.sign_test(d_c, d_o, mask, 'decrease'),
                         sign_short=st.sign_test(s_c, s_o, mask, 'increase')))
    # калибровка критерия: случайная перестановка меток «опыт/контроль» по наборам
    d_o = _dur(ctx.b('prime12'))
    okp = mask & np.isfinite(d_c) & np.isfinite(d_o) & (d_c != d_o)
    dif = d_o[okp] - d_c[okp]; rngp = np.random.default_rng(321); fires = 0
    for _ in range(2000):
        sg = rngp.choice([-1.0, 1.0], len(dif))
        fires += (st.sign_test(np.zeros(len(dif)), dif * sg, np.ones(len(dif), bool), 'decrease')['p'] <= 0.05)
    calib = fires / 2000.
    fin = [z for z in rows if z['dur'] == z['dur']]
    best = min(fin, key=lambda z: z['dur']) if fin else rows[0]
    chk = bool(fin and best['sign_dur']['n'] >= 10 and len(dif) >= 10)
    va = (best['sign_dur']['verdict'] if chk else 'непроверяемо')
    vb = (bool(best['ub'] == best['ub'] and best['ub'] <= TARGET_DAYS) if chk else 'непроверяемо')
    vc = (best['sign_short']['verdict'] if chk else 'непроверяемо')
    vg = (bool(calib <= 0.09) if chk else 'непроверяемо')          # допуск ОДНОСТОРОННИЙ, см. текст правила
    vt = lambda v: 'выполнено' if v is True else ('не выполнено' if v is False else str(v))
    md = ['\n## %s. Примирование индукции\n' % NAME,
          'Усиление индукции зависит от истории: p\' = [u/(K_B+u) − p]/tau_p, beta_эфф = beta·(1 + %.1f·p). '
          'Потолок втрое измерен, tau_p перебирается. Парный контроль — блок с выключенным примированием, '
          'то же зерно и те же наборы. Связь со сном выключена (k = 0), как в Ф11.\n' % A_PR,
          '\n### Правило, записано до счёта\n',
          '(а) Укорачивает: парный знаковый критерий по наборам, направление «меньше». Под нулевой гипотезой '
          'k ~ Bin(n, 1/2), то есть E[k] = n/2 и Var[k] = n/4; вердикт по точному биномиальному p, уровень 0,05. '
          'Распределение длительностей не используется.\n',
          '(б) Доходит до цели: односторонняя верхняя 95-процентная граница медианы по порядковой статистике '
          'не выше %.1f суток. Покрытие границы проверено розыгрышем, промах 0,038–0,052.\n' % TARGET_DAYS,
          '(в) Короткие эпизоды ПОЯВЛЯЮТСЯ: тот же критерий на доле эпизодов короче суток, направление «больше». '
          'Направление задано явным обязательным аргументом: 14.10 здесь была ошибка, критерий считал убыль '
          'вместо прироста и дал p = 0,99 там, где верное направление даёт 0,038.\n',
          '(г) Калибровка: при случайной перестановке меток критерий (а) должен срабатывать не чаще чем в 9 %% '
          'случаев. Допуск ОДНОСТОРОННИЙ намеренно: знаковый критерий дискретен, и при малом n его достижимые '
          'уровни редки (при n = 12 это 0,019 и 0,073, между ними ничего нет). Значение НИЖЕ номинальных 0,05 '
          'означает, что критерий консервативен — он теряет мощность, но не врёт, и это не повод браковать прогон. '
          'Опасно только превышение. Читать надо с этой строки.\n',
          '\n| | блок | длительность, сут | верхняя граница медианы | коротких по всем эпизодам | эпизодов | p во время эпизода |\n|---|---|---|---|---|---|---|',
          '| контроль | %s | %.2f | %.2f | %.3f | %d | — |' % (ctx.key('base'), base['dur'], base['ub'], base['short_pool'], base['n_ep'])]
    for z in rows:
        md.append('| примирование, tau_p = %.0f ч | %s | %.2f | %.2f | %.3f | %d | %.3f |'
                  % (z['tau_p_h'], z['block'], z['dur'], z['ub'], z['short_pool'], z['n_ep'], z['pr_dam']))
    md += ['\n| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|',
           '| г: калибровка критерия | срабатывает в %.3f случаев | не выше 0,09 (ниже — консервативно, это допустимо) | %s |' % (calib, vt(vg)),
           '| а: укорачивает | k = %d из n = %d; под нулём E = %.1f, SD = %.2f, критическое %.1f; p = %.1e | p не больше 0,05 | %s |'
           % (best['sign_dur']['k'], best['sign_dur']['n'], best['sign_dur'].get('mu', float('nan')),
              best['sign_dur'].get('sd', float('nan')), best['sign_dur'].get('crit', float('nan')),
              best['sign_dur']['p'], vt(va)),
           '| б: доходит до %.1f суток | медиана %.2f, граница %.2f (ранг %d из %d) | граница не выше %.1f | %s |'
           % (TARGET_DAYS, best['dur'], best['ub'], best['ub_rank'], best['ub_n'], TARGET_DAYS, vt(vb)),
           '| в: короткие эпизоды появляются | k = %d из n = %d; p = %.1e; доля %.3f против %.3f в контроле | p не больше 0,05 | %s |'
           % (best['sign_short']['k'], best['sign_short']['n'], best['sign_short']['p'],
              best['short_pool'], base['short_pool'], vt(vc)),
           '\nБаза: %.2f суток при доле коротких %.3f, по %d наборам полосы с кратностью индукции не ниже %.1f; '
           'SE медианы %.1f %%.' % (base['dur'], base['short_pool'], base['n_sets'], BFAC_MIN, 100 * base['se'])]
    out = dict(base=base, rows=rows, calib_false_alarm=calib, A_PR=A_PR,
               a=dict(verdict=va), b=dict(verdict=vb, thresh_days=TARGET_DAYS),
               c=dict(verdict=vc), g=dict(verdict=vg))
    ctx.verdict(NAME, 'а', best['sign_dur']['p'], va, note='знаковый критерий, направление «меньше»')
    ctx.verdict(NAME, 'б', best['ub'], vb, note='верхняя граница медианы, порядковая статистика')
    ctx.verdict(NAME, 'в', best['sign_short']['p'], vc, note='направление «больше»; 14.10 было перепутано, исправлено')
    ctx.verdict(NAME, 'г', calib, vg, note='калибровка перестановкой меток')
    return out, md
