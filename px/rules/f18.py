# -*- coding: utf-8 -*-
"""Ф18: чем на самом деле сделан предвестник сна.

КАК ЗАПИСАНО (до счёта, в mass_core6.py, разд. 4в-2)
  (а) Предвестник не нуждается в медленном шуме: при sigma_м = 0 последние сутки не ниже 1,05 и
      ранговая связь не выше −0,50. Это ПОДТВЕРЖДЕНИЕ уже увиденного, а не предсказание.
  (б) ГЛАВНАЯ ЧАСТЬ: если предвестник делает медленная переменная ЗАЩИТЫ, то при sigma_м = 0 и
      отнятой длинной памяти (вторая ступень на 12 ч вместо 63) он должен ПРОПАСТЬ — последние
      сутки ниже 1,05 ИЛИ связь выше −0,50.
  (в) Если (б) не выполнено, вывод записан заранее: в модели есть третья медленная величина,
      которую я не вижу, и следующий шаг — искать её, а не достраивать имеющиеся.

ПОЧЕМУ НУЖНЫ ДВОЙНИКИ
  Часть (б) как записана выполняется ДИЗЪЮНКЦИЕЙ: достаточно, чтобы связь вышла выше −0,50. Но
  связь считается по восьми ячейкам, её SE равна 1/sqrt(7) = 0,38, и различимый порог 0,94 — то
  есть эта половина условия непроверяема в принципе (см. Ф6(б′)). Остаётся половина про высоту,
  и она одна решает вердикт. Поэтому:
  (б′) Двойник сравнивает высоту предвестника ПАРНО по наборам: те же наборы, то же зерно, разница
       только в памяти защиты. Направление задано заранее и обязательно: если предвестник делает
       память защиты, то при её отнятии высота у набора должна ПАДАТЬ (direction = decrease).
       Это тот самый случай, где 14.10 в Ф19(в) направление было перепутано молча; теперь
       перепутать его нельзя — аргумент обязателен, и критерий проверяет сам себя на синтетике.
  (б″) Отдельно печатается, насколько высота ПАДАЕТ в среднем: если приписка верна, превышение
       над собственным уровнем должно уйти не частично, а почти целиком, потому что других
       медленных величин в модели нет. Порог — превышение не больше MDE от единицы.
"""
import numpy as np
from .. import stats as st
from ..consts import TAU_M, TAU_S
from . import _blocks as B
from ._meas import meas, curve6, col_se, rank_corr, atten, rank_corr_sets

NAME = 'Ф18'
NLB = 8


def needs(cfg): return B.slow_scan(cfg)


def run(ctx):
    S = ctx.S; cfg = ctx.cfg; nd = cfg.d['run']['days']; band = S['band']
    cur, cnt, rel = curve6(ctx.b(B.tag_s(B.SSCAN[0])), band)
    nm, nmn, nmrel = curve6(ctx.b('s_nomem'), band)
    ctl = [s for s in B.SSCAN if s > 0][0]
    c0, n0, _ = curve6(ctx.b(B.tag_s(ctl)), band)
    se0, ns0 = col_se(rel, 0); sen, nsn = col_se(nmrel, 0)
    rho, nb_ = rank_corr(np.where(cnt > 0, cur, np.nan))
    rho_nm, nbn = rank_corr(np.where(nmn > 0, nm, np.nan))
    rse = st.rho_se(nb_)
    # (а′) и (б‴): наклон по НАБОРАМ. Части (а) и (б) непроверяемы по информационному пределу —
    # ячеек около восьми, порог −0,50 при MDE около 0,94. Наклон у каждого набора по его ячейкам
    # даёт n в сотнях; порог ослаблен множителем из дисперсий (_meas.atten), оценённым заранее.
    lamA, fatA, _ = atten(rel); mrA, msA, nkA, ntA = rank_corr_sets(rel)
    thrA = -0.5 * fatA if fatA == fatA else float('nan')
    spA = st.adjudicate(mrA, thrA, msA, 'le', null=0.0, n=nkA)
    lamN, fatN, _ = atten(nmrel); mrN, msN, nkN, ntN = rank_corr_sets(nmrel)
    thrN = -0.5 * fatN if fatN == fatN else float('nan')
    # Без памяти защиты предвестник должен ПРОПАСТЬ, то есть наклон не круче порога: проверка
    # на отсутствие, поэтому направление обратное и нулевое значение — сам порог.
    spN = st.adjudicate(mrN, thrN, msN, 'ge', null=0.0, n=nkN)

    a_h = st.adjudicate(float(cur[0]), 1.05, se0, 'ge', null=1.0, n=ns0)
    a_s = st.adjudicate(rho, -0.5, rse, 'le', null=0.0, n=nb_)
    a = ('непроверяемо' if (a_h['verdict'] == 'непроверяемо' or a_s['verdict'] == 'непроверяемо')
         else bool(a_h['raw_verdict'] and a_s['raw_verdict']))

    nm_ok = bool(nmn[0] >= 10)
    b_raw = ('непроверяемо' if not nm_ok else bool(not (nm[0] >= 1.05 and rho_nm == rho_nm and rho_nm <= -0.5)))
    b_h = st.adjudicate(float(nm[0]), 1.05, sen, 'le', null=1.0, n=nsn)       # «ниже 1,05» — половина дизъюнкции
    b = ('непроверяемо' if (b_h['verdict'] == 'непроверяемо' or a_s['verdict'] == 'непроверяемо') else b_raw)

    v1 = rel[:, 0]; v2 = nmrel[:, 0]
    ok = np.isfinite(v1) & np.isfinite(v2)
    sg = st.sign_adjudicate(st.sign_test(v1, v2, ok, 'decrease'))
    drop = float(np.nanmean(v1[ok] - v2[ok])) if ok.sum() else float('nan')
    se_d = float(np.nanstd(v1[ok] - v2[ok], ddof=1) / np.sqrt(max(ok.sum(), 1))) if ok.sum() >= 3 else float('nan')
    # Размер эффекта записан заранее: если предвестник делает память защиты, превышение должно
    # уйти не частично, а хотя бы на три четверти от того, что есть с памятью.
    exc0 = float(cur[0]) - 1.0
    gone = st.adjudicate(float(nm[0]) - 1.0, 0.25 * exc0, sen, 'le', null=exc0, n=nsn)

    vt = lambda v: 'выполнено' if v is True else ('не выполнено' if v is False else str(v))
    md = ['\n## %s. Чем сделан предвестник\n' % NAME,
          'Три блока: рабочий %s (шума нет, память %.0f ч), без памяти %s (память %.0f ч) и прежнее '
          'рабочее %s (шум %.2f). Наборы во всех трёх ОДНИ И ТЕ ЖЕ, зерно одно, поэтому доступно '
          'парное сравнение по наборам.\n'
          % (ctx.key(B.tag_s(B.SSCAN[0])), TAU_S * 24, ctx.key('s_nomem'), TAU_M * 24,
             ctx.key(B.tag_s(ctl)), ctl),
          '\n| суток до срыва | шума нет, память %.0f ч | наборов | шума нет, памяти нет | наборов | шум %.2f, память %.0f ч |\n|---|---|---|---|---|---|'
          % (TAU_S * 24, ctl, TAU_S * 24)]
    for j in range(NLB):
        md.append('| %d–%d | %s | %d | %s | %d | %s |' % (j, j + 1,
                  ('%.4f' % cur[j]) if cnt[j] > 0 else '—', int(cnt[j]),
                  ('%.4f' % nm[j]) if nmn[j] > 0 else '—', int(nmn[j]),
                  ('%.4f' % c0[j]) if n0[j] > 0 else '—'))
    md += ['\n| часть | получено | порог | нулевое значение | SE | MDE | вывод |\n|---|---|---|---|---|---|---|',
           '| а: предвестник без медленного шума есть (как записано) | высота %.4f, наклон %s | не ниже 1,05 И не выше −0,50 | 1,0 и 0 | %.4f / %s | %.4f / %s | %s |'
           % (cur[0], ('%+.2f' % rho) if rho == rho else '—', se0, ('%.3f' % rse) if rse == rse else '—',
              a_h['mde'], ('%.3f' % a_s['mde']) if a_s['mde'] == a_s['mde'] else '—', vt(a)),
           '| б: без памяти защиты предвестник пропадает (как записано) | высота %.4f, наклон %s, наборов %d | ниже 1,05 ИЛИ выше −0,50 | — | %.4f | %.4f | %s |'
           % (nm[0], ('%+.2f' % rho_nm) if rho_nm == rho_nm else '—', int(nmn[0]), sen, b_h['mde'], vt(b)),
           '| б′: парно по наборам — падает ли высота при отнятии памяти | %d из %d наборов вниз, p = %s | p не больше 0,05 | 1/2 | — | мощность %.2f | %s |'
           % (sg['k'], sg['n'], ('%.3f' % sg['p']) if sg['p'] == sg['p'] else '—', sg['power'], vt(sg['verdict'])),
           '| б″: превышение уходит хотя бы на три четверти | без памяти %+.4f (с памятью %+.4f, среднее падение %+.4f) | не больше %.4f | %.4f | %.4f | %.4f | %s |'
           % (nm[0] - 1, exc0, drop, 0.25 * exc0, exc0, sen, gone['mde'], vt(gone['verdict']))]
    if b_raw is False and sg['verdict'] is not True:
        md.append('\n**Часть (в) правила включается.** Предвестник не делает ни медленный шум (Ф6(в′)), ни '
                  'память защиты: при её отнятии высота у наборов не падает достоверно. Значит в модели есть '
                  'третья медленная величина, и записанный заранее вывод требует её ИСКАТЬ, а не достраивать '
                  'имеющиеся. Где искать: единственные медленные времена в модели, кроме памяти защиты, — '
                  'восстановление x (его пол) и c_база с постоянной T. Их и надо мерить по отдельности.')
    elif sg['verdict'] is True:
        md.append('\n**Приписка подтверждена парным критерием.** Высота предвестника падает у %d из %d '
                  'наборов при отнятии длинной памяти защиты (p = %.3g). Это и есть ответ на вопрос, зачем '
                  'вводился медленный шум нагрузки: задачу «медленная величина на масштабе суток-недели» '
                  'решает вторая ступень защиты, и шум из модели можно убирать совсем.'
                  % (sg['k'], sg['n'], sg['p']))
    out = dict(a=dict(height=float(cur[0]), rank_corr=rho, verdict=a, height_part=a_h, slope_part=a_s,
                      note='подтверждение прошлого прогона, не предсказание'),
               b=dict(height=float(nm[0]), rank_corr=rho_nm, sets=int(nmn[0]), verdict=b, raw=b_raw,
                      height_part=b_h, tau_control_h=TAU_M * 24),
               b_paired=sg, b_gone=dict(excess=float(nm[0]) - 1, mean_drop=drop, se_drop=se_d, **gone),
               curve=[None if cnt[j] == 0 else float(cur[j]) for j in range(NLB)],
               curve_no_memory=[None if nmn[j] == 0 else float(nm[j]) for j in range(NLB)])
    ctx.verdict(NAME, 'а', float(cur[0]), a, note='подтверждение, не предсказание')
    ctx.verdict(NAME, 'а′', mrA, spA['verdict'],
                note='наклон по наборам: %d из %d годны, lam = %.2f, порог −0,50 ослаблен до %.3f'
                     % (nkA, ntA, lamA, thrA))
    ctx.verdict(NAME, 'б', float(nm[0]), b, note='дизъюнкция; половина про наклон непроверяема')
    ctx.verdict(NAME, 'б′', float(sg['k']), sg['verdict'], note='парный критерий, направление decrease')
    ctx.verdict(NAME, 'б″', float(nm[0]) - 1, gone['verdict'], note='уходит ли превышение целиком')
    ctx.verdict(NAME, 'б‴', mrN, spN['verdict'],
                note='наклон по наборам без памяти защиты: %d из %d годны, порог −0,50 ослаблен до '
                     '%.3f, проверка НА ОТСУТСТВИЕ' % (nkN, ntN, thrN))
    return out, md
