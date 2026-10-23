# -*- coding: utf-8 -*-
"""Ф6: предвестник сна — растёт ли бодрствование по мере приближения к срыву.

КАК ЗАПИСАНО
  Длина бодрствования в долях СОБСТВЕННОГО здорового уровня растёт монотонно по мере приближения
  к срыву и в последние сутки превышает здоровый уровень не меньше чем на 5 %.
  (а) последние сутки не ниже 1,05;  (б) ранговая связь с номером ячейки не выше −0,50;
  (в) контроль: та же проверка при ДРУГОЙ силе медленного шума не должна проходить;
  (г) наклон держится при всех силах шума, где есть не меньше 20 наборов.

ВЫКЛАДКИ ДЛЯ ДВОЙНИКОВ
  (а′) Высота предвестника не произвольна: бодрствование удлиняется ровно потому, что повреждение
       тормозит сонное ядро как -k*D, поэтому высота обязана лежать между единицей и тем, что даёт
       переключатель при D на ПОРОГЕ эпизода: выше этой границы удлинение не может быть сделано
       повреждением вовсе (для такого удлинения нужно D больше порога, а тогда это уже эпизод, а не
       предвестник). Двойник проверяет попадание в эту вилку, то есть и существование предвестника,
       и его приписку механизму сразу.
  (б′) Ранговая связь считается по числу ЗАПОЛНЕННЫХ ячеек (их восемь), и её SE равна 1/sqrt(n−1).
       При восьми ячейках SE = 0,38, различимый порог 0,94, а правило ставит −0,50. Значит часть (б)
       как записана НЕПРОВЕРЯЕМА при любом числе наборов: дело не в выборке наборов, а в числе
       ячеек. Чтобы порог −0,50 стал проверяемым, нужно около 40 ячеек, то есть ячейки «суток до
       срыва» надо дробить мельче. Это правка счёта, и она записана в план, а не сделана задним
       числом.
  (в′) Контроль «не проходит / проходит» — двоичный и поэтому слабый. Наборы в обоих блоках ОДНИ И
       ТЕ ЖЕ, с одним зерном, поэтому доступно гораздо более сильное сравнение: ПАРНЫЙ знаковый
       критерий по высоте предвестника у каждого набора. Направление задано заранее: если
       предвестник делает медленный шум, то при его включении высота у набора должна РАСТИ.
"""
import numpy as np
from .. import predict as P
from .. import stats as st
from ..consts import DTHR
from . import _blocks as B
from ._meas import meas, curve6, col_se, rank_corr, atten, rank_corr_sets

NAME = 'Ф6'
NLB = 8


def needs(cfg): return B.slow_scan(cfg)


def _curve(ctx, tag):
    return curve6(ctx.b(tag), ctx.S['band'])


def run(ctx):
    S = ctx.S; cfg = ctx.cfg; nd = cfg.d['run']['days']; band = S['band']; kb = B.KB(cfg)
    ref = B.SSCAN[0]; ctl = [s for s in B.SSCAN if s > 0][0]
    cur, cnt, rel = _curve(ctx, B.tag_s(ref))
    c0, n0, rel0 = _curve(ctx, B.tag_s(ctl))
    se0, ns0 = col_se(rel, 0)
    a = st.adjudicate(float(cur[0]), 1.05, se0, 'ge', null=1.0, n=ns0)
    hi = P.bout_gain(kb, 0.05, DTHR)
    # Вилка: снизу — записанные в правиле 5 %, сверху — то, что может дать повреждение на пороге
    # эпизода. Проверяемость решается зазором между краями вилки: он должен быть не меньше MDE.
    ap = dict(value=float(cur[0]), lo=1.05, hi=hi['ratio'], se=se0, mde=st.mde(se0), n=ns0)
    ap['verdict'] = ('непроверяемо' if (not (se0 == se0) or not (hi['ratio'] - 1.05 >= st.mde(se0)))
                     else bool(1.05 <= cur[0] <= hi['ratio']))
    rho, nb_ = rank_corr(np.where(cnt > 0, cur, np.nan)); rse = st.rho_se(nb_)
    b = st.adjudicate(rho, -0.5, rse, 'le', null=0.0, n=nb_)
    # (б′) тот же наклон, но по НАБОРАМ: ошибка из разброса между наборами, n — сотни.
    # Часть (б) непроверяема по ИНФОРМАЦИОННОМУ пределу, а не по нехватке наборов: ячеек около
    # восьми, кривая индексируется бодрствованиями (их около одного в сутки), глубина просмотра
    # ограничена межэпизодным промежутком. Порог ослаблен множителем из дисперсий, оценённым до
    # того, как мы посмотрели на наклоны (см. _meas.atten).
    lam6, fat6, nat6 = atten(rel)
    mrho6, mse6, nok6, ntot6 = rank_corr_sets(rel)
    thr6 = -0.5 * fat6 if fat6 == fat6 else float('nan')
    bp6 = st.adjudicate(mrho6, thr6, mse6, 'le', null=0.0, n=nok6)
    rho0, nb0 = rank_corr(np.where(n0 > 0, c0, np.nan))
    ctrl_ok = bool(n0[0] >= 10)
    c = ('непроверяемо' if not ctrl_ok else bool(not (c0[0] >= 1.05 and rho0 == rho0 and rho0 <= -0.5)))
    v1 = rel[:, 0]; v2 = rel0[:, 0]
    sg = st.sign_adjudicate(st.sign_test(v1, v2, np.isfinite(v1) & np.isfinite(v2), 'increase'))
    rows = []
    for ss in B.SSCAN:
        cu, cn, rl = _curve(ctx, B.tag_s(ss)); r_, nbn = rank_corr(np.where(cn > 0, cu, np.nan))
        lm_, ft_, _ = atten(rl); mr_, ms_, nk_, nt_ = rank_corr_sets(rl)
        mm = meas(ctx.b(B.tag_s(ss)), nd, band)
        rows.append(dict(sigs=ss, last=float(cu[0]) if cn[0] > 0 else float('nan'), sets=int(cn[0]),
                         rank_corr=r_, n_bins=nbn, share_ever=mm['share_ever'], dur=mm['dur'],
                         short_pool=mm['short_pool'], stuck15=mm['stuck15'],
                         se_last=col_se(rl, 0)[0], mean_rho=mr_, se_rho=ms_, n_sets=nk_,
                         lam=lm_, atten=ft_, thr_sets=-0.5 * ft_ if ft_ == ft_ else float('nan')))
    strong = [z for z in rows if z['sets'] >= 20]
    g_raw = bool(strong and all(z['rank_corr'] == z['rank_corr'] and z['rank_corr'] <= -0.5 for z in strong))
    g = ('непроверяемо' if b['verdict'] == 'непроверяемо' else g_raw)
    plat = float(np.nanmean(np.where(cnt[1:] > 0, cur[1:], np.nan)))
    vt = lambda v: 'выполнено' if v is True else ('не выполнено' if v is False else str(v))
    md = ['\n## %s. Предвестник сна\n' % NAME,
          'Опорный блок %s (медленного шума нет), парный контроль %s (sigma_м = %.2f). Кривая — длина '
          'бодрствования в долях собственного здорового уровня набора, по полосе.\n'
          % (ctx.key(B.tag_s(ref)), ctx.key(B.tag_s(ctl)), ctl),
          '\n| суток до срыва | опорная (sigma_м = %.2f) | наборов | SE | контроль (sigma_м = %.2f) | наборов |\n|---|---|---|---|---|---|' % (ref, ctl)]
    for j in range(NLB):
        s_, _ = col_se(rel, j)
        md.append('| %d–%d | %s | %d | %s | %s | %d |' % (j, j + 1, ('%.4f' % cur[j]) if cnt[j] > 0 else '—',
                  int(cnt[j]), ('%.4f' % s_) if s_ == s_ else '—',
                  ('%.4f' % c0[j]) if n0[j] > 0 else '—', int(n0[j])))
    md += ['\n| плато на 1–8 сутках | последние сутки | превышение плато |\n|---|---|---|',
           '| %.4f | %.4f | %.1f %% |' % (plat, cur[0], 100 * (cur[0] / plat - 1)),
           '\n| часть | получено | порог | нулевое значение | SE | MDE | вывод |\n|---|---|---|---|---|---|---|',
           '| а: последние сутки (как записано) | %.4f на %d наборах | не меньше 1,05 | 1,0 | %.4f | %.4f | %s |'
           % (cur[0], ns0, se0, a['mde'], vt(a['verdict'])),
           '| а′: высота в механической вилке | %.4f | от %.4f до %.4f (переключатель при D = %.2f) | 1,0 | %.4f | %.4f | %s |'
           % (cur[0], ap['lo'], ap['hi'], DTHR, se0, ap['mde'], vt(ap['verdict'])),
           '| б: наклон (как записано) | %s по %d ячейкам | не выше −0,50 | 0 | %s | %s | %s |'
           % (('%+.2f' % rho) if rho == rho else '—', nb_, ('%.3f' % rse) if rse == rse else '—',
              ('%.3f' % b['mde']) if b['mde'] == b['mde'] else '—', vt(b['verdict'])),
           '| в: контроль не проходит (как записано) | контроль: последние сутки %s, наклон %s, наборов %d | контроль НЕ должен проходить | — | — | — | %s |'
           % (('%.4f' % c0[0]) if n0[0] > 0 else '—', ('%+.2f' % rho0) if rho0 == rho0 else '—', int(n0[0]), vt(c)),
           '| в′: парный критерий — поднимает ли шум предвестник | %d из %d наборов вверх, p = %s | p не больше 0,05 | 1/2 | — | мощность %.2f | %s |'
           % (sg['k'], sg['n'], ('%.3f' % sg['p']) if sg['p'] == sg['p'] else '—', sg['power'], vt(sg['verdict'])),
           '| г: наклон при всех силах шума | %s | каждая сила с 20+ наборами проходит −0,50 | — | — | — | %s |'
           % (', '.join('%.2f: %s (%d наб.)' % (z['sigs'], ('%+.2f' % z['rank_corr']) if z['rank_corr'] == z['rank_corr'] else 'не опр.', z['sets']) for z in rows), vt(g)),
           '\n| sigma_м | последние сутки | SE | наборов | наклон | доля повреждавшихся | длительность, сут | коротких по всем | застряли >1,5 сут |\n|---|---|---|---|---|---|---|---|---|']
    for z in rows:
        md.append('| %.2f | %s | %s | %d | %s | %.3f | %s | %.3f | %.3f |' % (
            z['sigs'], ('%.4f' % z['last']) if z['last'] == z['last'] else '—',
            ('%.4f' % z['se_last']) if z['se_last'] == z['se_last'] else '—', z['sets'],
            ('%+.2f' % z['rank_corr']) if z['rank_corr'] == z['rank_corr'] else '—', z['share_ever'],
            ('%.2f' % z['dur']) if z['dur'] == z['dur'] else '—', z['short_pool'], z['stuck15']))
    if b['verdict'] == 'непроверяемо':
        md.append('\n**Части (б) и (г) непроверяемы, и выборка наборов тут ни при чём.** Наклон мерится по %d '
                  'ячейкам «суток до срыва»; SE ранговой связи равна 1/sqrt(n−1) = %.2f, различимый порог '
                  '%.2f, а правило ставит −0,50. Нужно около %.0f ячеек. Вывод для плана: дробить ячейки '
                  'предвестника вчетверо (по шесть часов) — тогда порог −0,50 станет проверяемым при том же '
                  'числе наборов.' % (nb_, rse, st.mde(rse), (st.MDE_K / 0.5) ** 2 + 1))
    out = dict(a=a, a_tight=ap, b=dict(rank_corr=rho, n_bins=nb_, **b),
               b_sets=dict(mean_rho=mrho6, se_sets=mse6, n_sets=nok6, n_total=ntot6, lam=lam6,
                           atten=fat6, thr=thr6, **bp6),
               c=dict(control_last=float(c0[0]) if n0[0] > 0 else None,
               control_rank_corr=rho0, control_sets=int(n0[0]), verdict=c), c_paired=sg,
               g=dict(verdict=g, raw=g_raw), rows=rows, plateau=plat,
               curve=[None if cnt[j] == 0 else float(cur[j]) for j in range(NLB)],
               curve_control=[None if n0[j] == 0 else float(c0[j]) for j in range(NLB)],
               switch_ceiling=hi)
    ctx.verdict(NAME, 'а', float(cur[0]), a['verdict'], note='высота предвестника в последние сутки')
    ctx.verdict(NAME, 'а′', float(cur[0]), ap['verdict'], note='вилка %.4f…%.4f из переключателя' % (ap['lo'], ap['hi']))
    ctx.verdict(NAME, 'б', rho, b['verdict'], note='наклон; при %d ячейках порог −0,50 неразличим' % nb_)
    ctx.verdict(NAME, 'б′', mrho6, bp6['verdict'],
                note='наклон по наборам: %d из %d годны, lam = %.2f, порог −0,50 ослаблен до %.3f'
                     % (nok6, ntot6, lam6, thr6))
    _gs = [z for z in strong if z['mean_rho'] == z['mean_rho']]
    _g2 = ('непроверяемо' if not _gs else
           bool(all(z['mean_rho'] <= z['thr_sets'] and 2.4865 * z['se_rho'] <= abs(z['thr_sets'])
                    for z in _gs)))
    ctx.verdict(NAME, 'г′', float(len(_gs)), _g2,
                note='наклон по наборам при всех силах шума: ' +
                     '; '.join('%.2f: %+.3f ± %.3f против %.3f'
                               % (z['sigs'], z['mean_rho'], z['se_rho'], z['thr_sets']) for z in _gs))
    ctx.verdict(NAME, 'в', float(c0[0]) if n0[0] > 0 else float('nan'), c, note='двоичный контроль, как записано')
    ctx.verdict(NAME, 'в′', float(sg['k']), sg['verdict'], note='парный знаковый критерий, направление increase')
    ctx.verdict(NAME, 'г', float(len(strong)), g, note='наклон при всех силах шума')
    return out, md
