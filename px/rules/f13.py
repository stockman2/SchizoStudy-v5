# -*- coding: utf-8 -*-
"""Ф13: зависит ли ответ модели от числа клеток J.

КАК ЗАПИСАНО
  (а) На ЗАКРЕПЛЁННОЙ сетке уязвимости три величины — доля повреждавшихся, длительность эпизода по
      непрерывной переменной и Ф5 (доля времени в середине шкалы) — меняются по всему перебору J не
      больше чем на 10 % от своего среднего.
  (б) На ПРЕЖНЕЙ сетке хотя бы одна из трёх меняется больше чем на 10 %. Контроль наоборот: без
      него часть (а) ничего не значит, потому что могла бы пройти от нечувствительности меры.
  (в) При J = 32 обе сетки дают расхождение РОВНО нуль (переанкеровка не сдвинула прежние прогоны).

ВЫКЛАДКА ДЛЯ ДВОЙНИКА — это главное, что здесь добавлено
  Размах трёх значений по J есть СЛУЧАЙНАЯ величина, даже если J ни на что не влияет: каждое
  значение — медиана по выборке наборов, и у неё есть своя ошибка. Для трёх независимых значений с
  относительной ошибкой r математическое ожидание размаха равно 1,693*r (ожидание размаха трёх
  стандартных нормальных). Значит порог 0,10 имеет смысл ТОЛЬКО если 1,693*r < 0,10, то есть при
  r < 0,059. Относительная ошибка каждой величины меряется в этом же прогоне загрузкой по наборам.
  Если потолок шума выше записанного порога — часть (а) НЕПРОВЕРЯЕМА при этом числе наборов, и
  рядом печатается, сколько наборов нужно; никакого «выполнено» в этом случае не ставится, даже
  если размах формально уложился.
  Оговорка в нашу пользу: блоки разных J считаются на ОДНИХ И ТЕХ ЖЕ наборах с одним зерном, то
  есть поток быстрого шума у них общий, и часть ошибки сокращается. Поэтому 1,693*r — ПОТОЛОК шума,
  а не его точное значение: размах ВЫШЕ потолка означает настоящую зависимость от J, размах ниже
  потолка не означает ничего.
"""
import numpy as np
from scipy.stats import norm
from .. import stats as st
from ..consts import DTHR
from . import _blocks as B
from ._meas import meas, mid_time

NAME = 'Ф13'
E_RANGE3 = 1.6926                 # ожидание размаха трёх стандартных нормальных
KK = (('share_ever', 'доля повреждавшихся'), ('dur_m', 'длительность по непрерывной переменной, сут'),
      ('mid', 'Ф5: доля времени в середине шкалы'))
NHD = 20


def needs(cfg): return B.j_scan(cfg)


def _one(ctx, Jx, vf, nd, band):
    rr = ctx.b(B.tag_j(Jx, vf)); mm = meas(rr, nd, band); mall = meas(rr, nd)
    midD, _, se_mid, neff, _, _ = mid_time(rr, nd, len(rr['nb']), mall['dur'], NHD)
    d = {k: mm[k] for k, _ in KK if k in mm}
    d['mid'] = midD; d['J'] = Jx; d['grid'] = 'закреплённая' if vf else 'прежняя'
    d['thr_eff'] = (int(Jx * DTHR) + 1) / float(Jx); d['n_ep_m'] = mm['n_ep_m']
    d['se_share'] = st.prop_se(mm['share_ever'], int(band.sum())) / max(mm['share_ever'], 1e-9)
    v = mm['dur_arr'][np.isfinite(mm['dur_arr'])]
    d['se_dur_m'] = st.boot_se(v) if len(v) >= 5 else float('nan')
    d['se_mid'] = se_mid / max(d['mid'], 1e-9); d['n_eff'] = neff
    return d


def run(ctx):
    S = ctx.S; cfg = ctx.cfg; nd = cfg.d['run']['days']; band = S['band']
    rows = [_one(ctx, Jx, vf, nd, band) for Jx in B.JS13 for vf in (1, 0)]
    new = [z for z in rows if z['grid'] == 'закреплённая']; old = [z for z in rows if z['grid'] == 'прежняя']
    spr = lambda rws, k: st.spread([z[k] for z in rws])
    sp_n = {k: spr(new, k) for k, _ in KK}; sp_o = {k: spr(old, k) for k, _ in KK}
    sek = dict(share_ever='se_share', dur_m='se_dur_m', mid='se_mid')
    rel = {k: float(np.nanmax([z[sek[k]] for z in new])) for k, _ in KK}
    floor = {k: E_RANGE3 * rel[k] for k, _ in KK}
    parts = {}
    for k, nm in KK:
        parts[k] = st.adjudicate(sp_n[k], 0.10, rel[k] * E_RANGE3 / st.MDE_K, 'le', null=0.0,
                                 n=int(band.sum()))
        parts[k]['floor'] = floor[k]
        if not (floor[k] == floor[k]) or floor[k] > 0.10:
            parts[k]['verdict'] = 'непроверяемо'
            parts[k]['n_required'] = (float(band.sum()) * (E_RANGE3 * rel[k] / 0.10) ** 2
                                      if rel[k] == rel[k] else float('nan'))
    a = ('непроверяемо' if any(parts[k]['verdict'] == 'непроверяемо' for k, _ in KK)
         else bool(all(parts[k]['verdict'] is True for k, _ in KK)))
    real = {k: bool(sp_n[k] == sp_n[k] and floor[k] == floor[k] and sp_n[k] > floor[k] + st.mde(rel[k]))
            for k, _ in KK}
    ap = ('непроверяемо' if any(not (floor[k] == floor[k]) for k, _ in KK)
          else bool(not any(real[k] for k, _ in KK)))
    b = bool(any(sp_o[k] == sp_o[k] and sp_o[k] > 0.10 for k, _ in KK))
    j32 = [z for z in rows if z['J'] == 32]
    e13 = float(max(abs(j32[0][k] - j32[1][k]) for k, _ in KK)) if len(j32) == 2 else float('nan')
    c = bool(len(j32) == 2 and e13 == 0.0)

    vt = lambda v: 'выполнено' if v is True else ('не выполнено' if v is False else str(v))
    md = ['\n## %s. Зависит ли ответ от числа клеток\n' % NAME,
          'Перебор J = %s на двух сетках уязвимости, k = 0. Блок J = 32 на закреплённой сетке — это тот же '
          'блок, что «base» у Ф2, Ф7 и Ф12 (%s), поэтому перебор стоит на один блок меньше.\n'
          % (', '.join(str(j) for j in B.JS13), ctx.key(B.tag_j(32, 1))),
          '\n| сетка | J | настоящий порог эпизода | доля повреждавшихся | длительность по непрерывной, сут | эпизодов в счёте | Ф5: середина шкалы | SE: доля / длит. / середина |\n|---|---|---|---|---|---|---|---|']
    for z in rows:
        md.append('| %s | %d | %.4f | %.4f | %s | %d | %.4f | %.3f / %s / %s |' % (
            z['grid'], z['J'], z['thr_eff'], z['share_ever'],
            ('%.2f' % z['dur_m']) if z['dur_m'] == z['dur_m'] else '—', z['n_ep_m'], z['mid'],
            z['se_share'], ('%.3f' % z['se_dur_m']) if z['se_dur_m'] == z['se_dur_m'] else '—',
            ('%.3f' % z['se_mid']) if z['se_mid'] == z['se_mid'] else '—'))
    md += ['\n| величина | размах на закреплённой | потолок шума 1,693·SE | размах на прежней | вывод по (а) | наборов нужно |\n|---|---|---|---|---|---|']
    for k, nm in KK:
        md.append('| %s | %.4f | %.4f | %.4f | %s | %s |' % (
            nm, sp_n[k], floor[k], sp_o[k], vt(parts[k]['verdict']),
            ('%.0f' % parts[k]['n_required']) if parts[k].get('n_required') == parts[k].get('n_required') else '—'))
    md += ['\n| часть | получено | порог | вывод |\n|---|---|---|---|',
           '| а: на закреплённой сетке ответ от J не зависит (как записано) | %s | каждый размах не больше 0,10 | %s |'
           % ('; '.join('%s %.3f' % (nm, sp_n[k]) for k, nm in KK), vt(a)),
           '| а′: размах не превышает потолка шума | %s | ни один размах не выше своего потолка | %s |'
           % ('; '.join('%s %.3f против %.3f' % (nm, sp_n[k], floor[k]) for k, nm in KK), vt(ap)),
           '| б: на прежней сетке зависит (контроль наоборот) | %s | хотя бы один больше 0,10 | %s |'
           % ('; '.join('%s %.3f' % (nm, sp_o[k]) for k, nm in KK), vt(b)),
           '| в: при J = 32 сетки совпадают | расхождение %.3e | ровно нуль | %s |'
           % (e13, 'выполнено' if c else 'НЕ ВЫПОЛНЕНО')]
    bad = [nm for k, nm in KK if parts[k]['verdict'] == 'непроверяемо']
    if bad:
        md.append('\n**Непроверяемо, и вот сколько наборов нужно.** У величин «%s» потолок шума размаха выше '
                  'записанного порога 0,10, то есть размах такой величины может превысить порог и при полной '
                  'независимости от J. Нужно наборов: %s. Это и есть тот случай, для которого мы договорились '
                  'ставить «непроверяемо» вместо вердикта: вердикт здесь был бы вердиктом о выборке, а не о '
                  'модели.'
                  % (', '.join(bad), '; '.join('%s — %.0f' % (nm, parts[k]['n_required'])
                                               for k, nm in KK if parts[k]['verdict'] == 'непроверяемо'
                                               and parts[k]['n_required'] == parts[k]['n_required'])))
    if any(real[k] for k, _ in KK):
        md.append('\n**Настоящая зависимость от J найдена** у величин «%s»: их размах выше потолка шума. '
                  'Значит J влез в модель третьим путём, которого я не вижу, и его надо искать, а не '
                  'замазывать. Два известных пути — хвост распределения уязвимости (закрыт закреплением '
                  'сетки) и квантование порога эпизода (столбец «настоящий порог»; обойдён мерой по '
                  'непрерывной переменной).' % ', '.join(nm for k, nm in KK if real[k]))
    out = dict(rows=rows, spread_fixed=sp_n, spread_old=sp_o, se_rel=rel, noise_floor=floor,
               a=dict(verdict=a, parts=parts), a_tight=dict(verdict=ap, real_dependence=real),
               b=dict(verdict=b), c=dict(err=e13, verdict=c), thresh=0.10)
    ctx.verdict(NAME, 'а', float(max(v for v in sp_n.values() if v == v)), a,
                note='порог 0,10 против потолка шума размаха')
    ctx.verdict(NAME, 'а′', float(max(v for v in sp_n.values() if v == v)), ap,
                note='размах против собственного потолка шума')
    ctx.verdict(NAME, 'б', float(max(v for v in sp_o.values() if v == v)), b, note='контроль наоборот')
    ctx.verdict(NAME, 'в', e13, c, note='совпадение сеток при J = 32: ровно нуль')
    return out, md
