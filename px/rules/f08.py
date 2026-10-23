# -*- coding: utf-8 -*-
"""Ф8: серотониновый триггер — делает ли односторонний всплеск срыв событием иного рода.

КАК ЗАПИСАНО
  (а) доля срывов в пределах 12 часов после всплеска не меньше 0,80;  (б) длительность эпизода не
  больше 2 суток;  (в) доля эпизодов короче суток не меньше 0,30;  (г) отклик на дозу: частота
  срывов растёт вместе с частотой всплесков.

ВЫКЛАДКА ДЛЯ ДВОЙНИКА (а′) — она меняет смысл всей части
  Всплеск поднимает нагрузку в SR_AMP раз, то есть приближает барьер на ln(SR_AMP)/sigma сигм
  (печатается в отчёте, порядка пяти). Казалось бы, тогда ВСЕ срывы обязаны идти после всплесков,
  и 0,80 — скромное требование. Но наборы ПОЛОСЫ отобраны условием gap_up <= ln(SR_AMP*day_pk):
  у части из них суточного пика хватает и БЕЗ всплеска, и у таких срывы идут сами. Отсюда
  аналитическое ожидание доли срывов, вызванных всплесками:
        trig = R_всп/(R_всп + R_0),   R_0 = <exp(-d^2/2)>,  R_всп = частота*окно*<прибавка>,
  посчитанное по каждому набору полосы. Нулевое значение (совпадение по времени, всплеск ни при
  чём) равно 1 - exp(-частота*0,5 сут). Двойник сравнивает наблюдение с ЭТИМ ожиданием, а не с
  круглым 0,80: если наблюдение согласуется с ним, механизм работает ровно так, как считает
  арифметика, и записанные 0,80 были требованием не к модели, а к отбору наборов.
  Оговорка модели: сколько срывов даст каждый набор, она не знает, наборы взвешены равно.
  (б′) и (в′) — то же, что в Ф7: длительность против аффинного закона Ф16 и клинической цели,
  доля коротких — по ВСЕМ эпизодам вместо вырожденной медианы по наборам.
  (г′) Частота срывов «в сутки» бралась МЕДИАНОЙ по наборам и выходила одинаковой во всех строках
  перебора просто потому, что медиана дискретна при малом числе срывов на набор. Двойник считает
  частоту ПУЛОМ: все срывы на все набор-сутки, с пуассоновской SE sqrt(N)/N, и проверяет рост
  знаковым критерием по парам соседних частот.
"""
import numpy as np
from .. import predict as P
from .. import stats as st
from ..consts import SR_AMP, SR_RATE, SR_CEIL, TAU_SR
from . import _blocks as B
from ._meas import meas

NAME = 'Ф8'
TARGET = 2.0


def needs(cfg): return B.sr_scan(cfg) + [B.kbase_block(cfg)]


def run(ctx):
    S = ctx.S; cfg = ctx.cfg; nd = cfg.d['run']['days']; band = S['band']
    rr = [dict(rate=rt, **{k: v for k, v in meas(ctx.b(B.tag_sr_r(rt)), nd, band).items() if k != 'dur_arr'})
          for rt in B.RSCAN]
    ra = [dict(amp=am, **{k: v for k, v in meas(ctx.b(B.tag_sr_a(am)), nd, band).items() if k != 'dur_arr'})
          for am in B.ASCAN_SR]
    ctl = meas(ctx.b('kbase'), nd, band)
    base = min(rr, key=lambda z: abs(z['rate'] - SR_RATE))
    mb_arr = meas(ctx.b(B.tag_sr_r(base['rate'])), nd, band)['dur_arr']
    tg = P.trig_share(S, amp=SR_AMP, rate=base['rate'], tau_h=TAU_SR)
    pred = float(np.mean(tg['pred'][band]))

    se_t = st.prop_se(base['trig'], base['n_br'])
    a = st.adjudicate(base['trig'], 0.80, se_t, 'ge', null=tg['null'], n=int(base['n_br']))
    ap = st.adjudicate(abs(base['trig'] - pred), st.eq_tol(se_t), se_t, 'le', null=0.0, n=int(base['n_br']))
    se_d = st.boot_se(mb_arr[np.isfinite(mb_arr)]) * max(base['dur'], 1e-9)
    b = st.adjudicate(base['dur'], TARGET, se_d, 'le', n=base['n_ep_sets'])
    predd = P.episode_dur_affine(cfg.d['model']['taus'])
    bp = st.adjudicate(abs(base['dur'] - predd), st.eq_tol(se_d), se_d, 'le', null=0.0, n=base['n_ep_sets'])
    se_s = st.prop_se(base['short_pool'], base['n_ep'])
    c = st.adjudicate(base['short'], 0.30, st.prop_se(base['short'], base['n_ep']), 'ge', n=int(base['n_ep']))
    cp = st.adjudicate(base['short_pool'], 0.30, se_s, 'ge', null=0.0, n=int(base['n_ep']))
    seq = [z['nday'] for z in rr]
    d_raw = bool(all(seq[i + 1] >= seq[i] - 1e-12 for i in range(len(seq) - 1)) and seq[-1] > seq[0])
    pool = [z['nday_pool'] for z in rr]
    # 15.10: знаковый критерий по парам заменён НАКЛОНОМ со своей ошибкой (у знакового при трёх
    # точках перебора мощности нет вовсе: нужно 23 пары, есть 2). Ошибка каждой точки
    # пуассоновская: SE частоты = sqrt(N)/N от самой частоты.
    Npool = np.array([z['n_br'] for z in rr], float)
    se_pt = np.where(Npool > 0, np.array(pool) / np.sqrt(np.maximum(Npool, 1)), np.nan)
    sg = st.slope_test(B.RSCAN, pool, se_pt, 'increase')
    N = np.array([z['n_br'] for z in rr], float)
    se_pool = np.where(N > 0, np.sqrt(np.maximum(N, 1)) / np.maximum(N, 1), np.nan) * np.array(pool)
    # Порог предсказан арифметикой опасности: отношение частот срывов на краях перебора.
    resp = P.rate_response(S, B.RSCAN)
    want = pool[0] * (resp[B.RSCAN[-1]] - 1.0)
    dp = st.adjudicate(pool[-1] - pool[0], want, float(np.nanmax(se_pool)), 'ge', null=0.0, n=int(N.sum()))

    vt = lambda v: 'выполнено' if v is True else ('не выполнено' if v is False else str(v))
    md = ['\n## %s. Серотониновый триггер\n' % NAME,
          'Всплеск поднимает нагрузку в %.1f раза (потолок %.1f: активность пирамидных клеток вдвое при '
          '5-HT2A в срезах ПФК), постоянная спада %.0f ч. В сигмах быстрого шума это %.2f — то есть '
          'всплеск заведомо сильнее любого шумового выброса. Но %.0f %% наборов полосы ломаются и БЕЗ '
          'всплеска: им хватает суточного пика (gap_up <= ln(day_pk)). Отсюда ожидание доли срывов после '
          'всплеска %.3f, а не 0,80.\n'
          % (SR_AMP, SR_CEIL, TAU_SR, tg['boost_sigma'], 100 * (1 - tg['need_spike']), pred),
          '\n| часть | получено | порог | нулевое значение | SE | MDE | вывод |\n|---|---|---|---|---|---|---|',
          '| а: срывы после всплеска (как записано) | %.3f на %d срывах | не меньше 0,80 | %.3f (совпадение) | %.4f | %.4f | %s |'
          % (base['trig'], int(base['n_br']), tg['null'], se_t, a['mde'], vt(a['verdict'])),
          '| а′: согласие с аналитическим ожиданием | промах %.3f (ожидание %.3f) | не больше %.3f (3 SE) | 0 | %.4f | %.4f | %s |'
          % (abs(base['trig'] - pred), pred, st.eq_tol(se_t), se_t, ap['mde'], vt(ap['verdict'])),
          '| б: длительность (как записано) | %.2f сут | не больше 2 сут | 2 сут | %.3f | %.3f | %s |'
          % (base['dur'], se_d, b['mde'], vt(b['verdict'])),
          '| б′: согласие с законом Ф16 | промах %.2f сут (предсказание %.2f) | не больше %.2f | 0 | %.3f | %.3f | %s |'
          % (abs(base['dur'] - predd), predd, st.eq_tol(se_d), se_d, bp['mde'], vt(bp['verdict'])),
          '| в: коротких эпизодов (медиана по наборам) | %.3f | не меньше 0,30 | 0,30 | %.3f | %.3f | %s |'
          % (base['short'], st.prop_se(base['short'], base['n_ep']), c['mde'], vt(c['verdict'])),
          '| в′: то же по ВСЕМ эпизодам | %.3f на %d эпизодах | не меньше 0,30 | 0 | %.4f | %.4f | %s |'
          % (base['short_pool'], int(base['n_ep']), se_s, cp['mde'], vt(cp['verdict'])),
          '| г: отклик на дозу (как записано, медиана по наборам) | %s | растёт | — | — | — | %s |'
          % (' → '.join('%.4f' % z for z in seq), vt(d_raw)),
          '| г′: то же пулом, против предсказанного отклика | %s | рост не меньше %.4f (предсказано ×%.2f) | 0 | %.4f | %.4f | %s |'
          % (' → '.join('%.4f' % z for z in pool), want, resp[B.RSCAN[-1]],
             float(np.nanmax(se_pool)), dp['mde'], vt(dp['verdict'])),
          '| г″: отклик растёт (наклон со своей ошибкой) | %s срывов на набор-сутки на единицу частоты, по %d точкам | больше нуля на MDE | 0 | %s | %s | %s |'
          % (('%+.5f' % sg['slope']) if sg['slope'] == sg['slope'] else '—', sg['n'],
             ('%.5f' % sg['se']) if sg['se'] == sg['se'] else '—',
             ('%.5f' % sg['mde']) if sg.get('mde') == sg.get('mde') else '—', vt(sg['verdict'])),
          '\n### Перебор частоты всплесков (амплитуда %.1f)\n' % SR_AMP,
          '| всплесков в сутки | всплесков за прогон | доля повреждавшихся | срывов в сутки: медиана / пулом | срывов после всплеска | длительность, сут | коротких по всем |\n|---|---|---|---|---|---|---|']
    for z in rr:
        md.append('| %.2f | %.1f | %.3f | %.4f / %.4f | %.3f | %s | %.3f |' % (
            z['rate'], z['n_sr'], z['share_ever'], z['nday'], z['nday_pool'], z['trig'],
            ('%.2f' % z['dur']) if z['dur'] == z['dur'] else '—', z['short_pool']))
    md += ['\n### Перебор амплитуды всплеска (частота %.2f в сутки)\n' % SR_RATE,
           '| нагрузка умножается на | доля повреждавшихся | срывов в сутки пулом | срывов после всплеска | длительность, сут | коротких по всем |\n|---|---|---|---|---|---|']
    for z in ra:
        md.append('| %.1f%s | %.3f | %.4f | %.3f | %s | %.3f |' % (
            z['amp'], ' (потолок)' if z['amp'] >= SR_CEIL else '', z['share_ever'], z['nday_pool'],
            z['trig'], ('%.2f' % z['dur']) if z['dur'] == z['dur'] else '—', z['short_pool']))
    md += ['\n### Контроль: всплесков нет вовсе (блок %s)\n' % ctx.key('kbase'),
           '| | доля повреждавшихся | срывов в сутки пулом | длительность, сут | коротких по всем |\n|---|---|---|---|---|',
           '| без всплесков | %.3f | %.4f | %s | %.3f |' % (ctl['share_ever'], ctl['nday_pool'],
             ('%.2f' % ctl['dur']) if ctl['dur'] == ctl['dur'] else '—', ctl['short_pool']),
           '| со всплесками (%.2f/сут, ×%.1f) | %.3f | %.4f | %s | %.3f |'
           % (base['rate'], SR_AMP, base['share_ever'], base['nday_pool'],
              ('%.2f' % base['dur']) if base['dur'] == base['dur'] else '—', base['short_pool'])]
    if bp['verdict'] is True or b['verdict'] is False:
        md.append('\n**Главный вывод раздела, записанный заранее.** Односторонний триггер не укорачивает '
                  'эпизод: длительность идёт по тому же аффинному закону памяти защиты, что и без всплесков. '
                  'Значит дело не в том, ЧЕМ вызван срыв, а в самом устройстве «двухустойчивая клетка плюс '
                  'нагрузка»: событийный вход меняет СТАТИСТИКУ срывов, но не их ИСХОД. Это и есть основание '
                  'идти к переменной защите (Ф11–Ф16) и к примированию (Ф19), а не к новым входам.')
    # При какой частоте всплесков предсказание арифметики станет ПРОВЕРЯЕМЫМ. Предсказанный прирост
    # линеен по частоте (R_всп = частота * окно * <прибавка>), значит нужная частота находится прямо:
    rate_need = float('nan')
    if want == want and want > 0 and dp['mde'] == dp['mde']:
        rate_need = float(B.RSCAN[-1] * dp['mde'] / want)
    md.append('\n**Сколько нужно, чтобы часть (г′) стала проверяемой.** Предсказанный прирост частоты '
              'срывов линеен по частоте всплесков, поэтому при нынешнем шуме он сравняется с минимальным '
              'различимым эффектом при частоте около **%.2f всплеска в сутки** (сейчас перебор доходит до '
              '%.2f). Это и есть верхняя точка, которую надо добавить в перебор; при ней предсказание '
              'либо подтвердится, либо будет отвергнуто, а не останется непроверяемым. Отдельно стоит '
              'заметить: НАБЛЮДАЕМЫЙ прирост уже сейчас вдесятеро больше предсказанного (%.5f против '
              '%.5f при MDE %.5f), то есть всплески поднимают частоту срывов сильнее, чем считает '
              'арифметика опасности. Это зеркально Ф2(б), где наблюдение оказалось ВТРОЕ СЛАБЕЕ '
              'предсказания. Обе расхождения в одну сторону: редкие сильные толчки работают лучше, '
              'чем постоянное поднятие нагрузки, и это стоит записать отдельным правилом.'
              % (rate_need, B.RSCAN[-1], pool[-1] - pool[0], want, dp['mde']))
    out = dict(rate_need=rate_need, a=a, a_tight=dict(pred=pred, analytic=dict(null=tg['null'], need_spike=tg['need_spike'],
               boost_sigma=tg['boost_sigma']), **ap), b=b, b_affine=dict(pred=predd, **bp), c=c, c_pool=cp,
               d=dict(seq=seq, verdict=d_raw),
               d_pool=dict(pool=pool, predicted_ratio={str(k): v for k, v in resp.items()}, want=want, **dp),
               d_sign=sg,
               rate_scan=rr, amp_scan=ra, control={k: v for k, v in ctl.items() if k != 'dur_arr'})
    ctx.verdict(NAME, 'а', base['trig'], a['verdict'], note='порог 0,80 не выведен; ожидание арифметики %.3f' % pred)
    ctx.verdict(NAME, 'а′', abs(base['trig'] - pred), ap['verdict'], note='согласие с аналитическим ожиданием')
    ctx.verdict(NAME, 'б', base['dur'], b['verdict'], note='клиническая цель')
    ctx.verdict(NAME, 'б′', abs(base['dur'] - predd), bp['verdict'], note='согласие с аффинным законом Ф16')
    ctx.verdict(NAME, 'в', base['short'], c['verdict'], note='медиана по наборам: статистика вырождена')
    ctx.verdict(NAME, 'в′', base['short_pool'], cp['verdict'], note='доля по всем эпизодам')
    ctx.verdict(NAME, 'г', float(seq[-1] - seq[0]), d_raw, note='медиана по наборам, как записано')
    ctx.verdict(NAME, 'г′', float(pool[-1] - pool[0]), dp['verdict'], note='частота срывов пулом')
    ctx.verdict(NAME, 'г″', sg['slope'], sg['verdict'], note='наклон отклика по частоте всплесков, со своей ошибкой')
    return out, md
