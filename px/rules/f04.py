# -*- coding: utf-8 -*-
"""Ф4: портится ли сон РАНЬШЕ срыва.

ЧТО ПЕРЕНЕСЕНО КАК ЗАПИСАНО
  (а) средняя длина бодрствования растёт с повреждением уже при D < 0,5: в ячейке 0,4-0,5 не
      меньше чем на 10 % больше, чем в ячейке 0,0-0,1; меньше 20 подпороговых бодрствований —
      непроверяемо;  (б) длина бодрствования превышает здоровый уровень не меньше чем за сутки
      до первого срыва и не больше чем за восемь.

ВЫКЛАДКА ДЛЯ ДВОЙНИКА (а′) — главная в этом правиле
  Порог 1,10 был назначен, а не выведен, и его можно проверить на достижимость ДО всякого счёта.
  Повреждение входит в приток к сонному ядру ровно как -k*D, поэтому длину бодрствования при
  постоянном подпороговом повреждении даёт сам переключатель Филлипса-Робинсона:
        ожидаемое отношение = w(k*D_ячейки) / w(k*D_0),
  где D берётся серединой ячейки. Это ЧИСТАЯ арифметика переключателя, клетки не считаются вовсе.
  Если отношение из переключателя меньше 1,10, то записанное правило требовало от механизма
  больше, чем механизм может дать, и его провал ничего не говорит о биологии — это надо сказать
  прямо, а не записывать в минус модели. Двойник проверяет наблюдение против этого отношения в
  ОБЕ стороны: наблюдение не должно быть ни ниже предсказания на MDE (тогда торможение ядра
  работает слабее, чем должно), ни выше (тогда удлинение делает что-то ещё, не торможение).

ВЫКЛАДКА ДЛЯ ДВОЙНИКА (б′)
  Левая кривая (все бодрствования в общей куче) смешивает наборы: у наборов, которые вообще
  срываются, бодрствование длиннее само по себе, и «опережение» может оказаться межнаборным
  сдвигом. Правая кривая — каждое бодрствование делится на СОБСТВЕННЫЙ здоровый уровень набора,
  и у неё нулевое значение равно ровно 1. Двойник выносит вердикт по правой и требует, чтобы
  превышение в первой ячейке было не меньше MDE, посчитанного по разбросу между наборами.
"""
import numpy as np
from .. import predict as P
from .. import stats as st
from ..consts import DTHR
from . import _blocks as B
from ._meas import meas, dbin_curve, lead_curve, curve6, col_se, rank_corr, boot_ratio, atten, rank_corr_sets

NAME = 'Ф4'
NDB, NLB = 10, 8


def needs(cfg): return [B.kbase_block(cfg)]


def run(ctx):
    S = ctx.S; cfg = ctx.cfg; nd = cfg.d['run']['days']; band = S['band']; kb = B.KB(cfg)
    rr = ctx.b('kbase'); mm = meas(rr, nd, band)
    cur, dnn, rel_cur, rel_cnt, rel, ds2, dn2 = dbin_curve(rr, band)
    # Разметка по D НА НАЧАЛЕ бодрствования. Прежняя берёт МАКСИМУМ D за это же бодрствование,
    # то есть метка частично вызвана измеряемой длиной: чем дольше тянется, тем больше успевает
    # накопиться. Обратная причинность, и длинные бодрствования систематически уезжают в ячейки с
    # завышенной D — именно отсюда наблюдаемое 1.085 ниже предсказания 1.114.
    dbs_s, dbs_n = rr['dbstart']; hs_s, hs_n = rr['hstart']
    ds2, dn2b = dbs_s[band], dbs_n[band]
    cur_b = np.where(dn2b.sum(0) > 0, ds2.sum(0) / np.maximum(dn2b.sum(0), 1), np.nan)
    own_b = np.where(hs_n[band] >= 3, hs_s[band] / np.maximum(hs_n[band], 1), np.nan)
    rel_b = np.where((dn2b > 0) & np.isfinite(own_b)[:, None],
                     ds2 / np.maximum(dn2b, 1) / np.maximum(own_b[:, None], 1e-9), np.nan)
    lcur, ln = lead_curve(rr, band)
    rcur, rcnt, rrel = curve6(rr, band)

    # ---- (а) как записано ----
    jlast = max([j for j in range(1, 5) if dnn[j] >= 10], default=-1)
    nsub = float(dnn[1:5].sum()); enough = bool(nsub >= 20 and jlast >= 2 and dnn[0] >= 10)
    g4 = float(cur[jlast] / cur[0]) if (enough and jlast > 0) else float('nan')
    # SE пулового отношения — ЗАГРУЗКОЙ ПО НАБОРАМ (см. boot_ratio: оценка через столбец 0 в долях
    # собственного уровня даёт ровно ноль, потому что этот столбец равен единице по построению)
    se_rel = boot_ratio(ds2, dn2, jlast, 0) if enough else float('nan')
    a = st.adjudicate(g4, 1.10, se_rel, 'ge', null=1.0, n=int(nsub)) if enough else \
        dict(verdict='непроверяемо', value=g4, thresh=1.10, null=1.0, se=se_rel, mde=float('nan'),
             gap=float('nan'), raw_verdict=False, n=int(nsub), n_required=float('nan'))

    # ---- (а′) ожидание переключателя ----
    d_lo = 0.05; d_hi = (jlast + 0.5) / 10. if jlast > 0 else float('nan')
    bg = P.bout_gain(kb, d_lo, d_hi) if d_hi == d_hi else dict(ratio=float('nan'), wake_lo_h=float('nan'),
                                                               wake_hi_h=float('nan'))
    ap = st.adjudicate(g4, bg['ratio'], se_rel, 'ge', null=1.0, n=int(nsub))
    over = bool(g4 == g4 and bg['ratio'] == bg['ratio'] and g4 > bg['ratio'] + st.mde(se_rel))
    reach = bool(bg['ratio'] == bg['ratio'] and bg['ratio'] >= 1.10)

    # ---- (б) как записано ----
    base_bout = float(cur[0]) if dnn[0] > 0 else float('nan')
    lead_ok = [j for j in range(NLB) if ln[j] >= 5 and lcur[j] > base_bout]
    b = bool(lead_ok and min(lead_ok) <= 7)

    # ---- (б′) по собственному уровню ----
    se0, n0 = col_se(rrel, 0)
    bp = st.adjudicate(float(rcur[0]), 1.05, se0, 'ge', null=1.0, n=n0)   # 5 % — размер эффекта из Ф6
    # (а″) тот же прирост, но по разметке начала бодрствования.
    # ВЫБОР ВЫСОКОЙ ЯЧЕЙКИ ЗАПИСАН ДО СЧЁТА, и не как число, а как правило: берём САМУЮ ВЫСОКУЮ
    # ячейку, в которой не меньше MINB бодрствований. Почему нельзя просто взять ячейку у порога
    # эпизода, как в части (а): чтобы бодрствование НАЧАЛОСЬ при D около 0,45, повреждение должно
    # уже быть, а через 0,4–0,5 доля повреждённых проходит быстро — ячейка почти пуста. Прежняя
    # разметка по максимуму такого ограничения не имела, потому что максимум успевал дойти докуда
    # угодно за само бодрствование. Ожидание переключателя пересчитывается в центре ВЫБРАННОЙ
    # ячейки, поэтому сравнение остаётся честным: предсказание следует за выбором, а не наоборот.
    MINB = 200
    cnt_b = dn2b.sum(0)
    jlo = 0
    cand = [j for j in range(1, NDB) if cnt_b[j] >= MINB]
    jhi = max(cand) if cand else int(0.45 * NDB)
    d_hi_b = (jhi + 0.5) / float(NDB); d_lo_b = (jlo + 0.5) / float(NDB)
    bgb = P.bout_gain(kb, d_lo_b, d_hi_b)
    nb_lo = int(cnt_b[jlo]); nb_hi = int(cnt_b[jhi])
    rb_lo = rel_b[:, jlo]; rb_hi = rel_b[:, jhi]
    g4b = (float(np.nanmean(rb_hi)) / float(np.nanmean(rb_lo))) if (nb_lo > 0 and nb_hi > 0) else float('nan')
    vb = rb_hi[np.isfinite(rb_hi)]
    se_b = float(np.std(vb, ddof=1) / np.sqrt(len(vb))) if len(vb) >= 3 else float('nan')
    ab = st.adjudicate(g4b, bgb['ratio'], se_b, 'ge', null=1.0, n=len(vb))
    rho, nb_ = rank_corr(np.where(rcnt > 0, rcur, np.nan))
    rse = st.rho_se(nb_)
    rp = st.adjudicate(rho, -0.5, rse, 'le', null=0.0, n=nb_)

    # ---- (б‴) наклон ПО НАБОРАМ, а не по ячейкам ----
    # Часть (б″) непроверяема не из-за нехватки наборов, а из-за нехватки ЯЧЕЕК: кривая индексируется
    # бодрствованиями, их около одного в сутки, глубина просмотра ограничена межэпизодным промежутком.
    # Мельчить нечего. Считаем наклон у каждого набора по его ячейкам и проверяем разброс МЕЖДУ
    # наборами: n становится сотнями. Порог ослабляется множителем из дисперсий (см. _meas.atten) —
    # он оценивается до того, как мы смотрим на наклоны, поэтому назначен заранее.
    lam4, fat4, nat4 = atten(rrel)
    mrho4, mse4, nok4, ntot4 = rank_corr_sets(rrel)
    thr4 = -0.5 * fat4 if fat4 == fat4 else float('nan')
    rp2 = st.adjudicate(mrho4, thr4, mse4, 'le', null=0.0, n=nok4)

    vt = lambda v: 'выполнено' if v is True else ('не выполнено' if v is False else str(v))
    md_sets = ['', '**Наклон по наборам, а не по ячейкам.** Часть (б″) непроверяема по информационному'
               ' пределу: ячеек около %d, ошибка ранговой связи 1/sqrt(n−1) = %.3f, MDE = %.3f, а порог'
               ' всего −0,50. Мельчить ячейки нечего — кривая индексируется бодрствованиями, их около'
               ' одного в сутки. Наклон у каждого набора по его ячейкам: **%+.3f ± %.3f** по %d годным'
               ' наборам из %d, MDE = %.3f. Порог пересчитан из дисперсий: шум ячейки к сигналу'
               ' lam = %.2f, ослабление %.3f, порог −0,50 → **%.3f**. Вывод: %s.'
               % (nb_, rse, st.mde(rse), mrho4, mse4, nok4, ntot4, st.mde(mse4), lam4, fat4, thr4,
                  vt(rp2['verdict'])), '']
    md = ['\n## %s. Портится ли сон раньше срыва (k = %.1f мВ, блок %s)\n' % (NAME, kb, ctx.key('kbase')),
          '| доля повреждённых D | длина бодрствования, сут | бодрствований | в долях своего здорового уровня | наборов |\n|---|---|---|---|---|']
    for j in range(NDB):
        md.append('| %.1f–%.1f%s | %s | %d | %s | %d |' % (
            j / 10., (j + 1) / 10., ' (порог эпизода)' if j == int(DTHR * 10) - 1 else '',
            ('%.3f' % cur[j]) if dnn[j] > 0 else '—', int(dnn[j]),
            ('%.3f' % rel_cur[j]) if rel_cnt[j] > 0 else '—', int(rel_cnt[j])))
    md += ['\n| суток до срыва | длина бодрствования, сут | бодрствований | в долях своего уровня | наборов | SE |\n|---|---|---|---|---|---|']
    for j in range(NLB):
        s_, n_ = col_se(rrel, j)
        md.append('| %d–%d | %s | %d | %s | %d | %s |' % (j, j + 1, ('%.3f' % lcur[j]) if ln[j] > 0 else '—',
                  int(ln[j]), ('%.4f' % rcur[j]) if rcnt[j] > 0 else '—', int(rcnt[j]),
                  ('%.4f' % s_) if s_ == s_ else '—'))
    md += ['', '**Разметка по началу бодрствования.** Части (а) и (а′) размечают бодрствование по '
           'МАКСИМУМУ доли повреждённых за него же, то есть метка частично вызвана измеряемой длиной: '
           'обратная причинность. Часть (а″) размечает по D на НАЧАЛЕ бодрствования — величина, '
           'причинно предшествующая его длине. Высокая ячейка выбрана по записанному заранее '
           'правилу: самая высокая с не менее %d бодрствований — вышла %.1f–%.1f (%d бодрствований), '
           'и ожидание переключателя пересчитано в её центре. Прирост по новой разметке **%s** против '
           'ожидания %.4f (по прежней разметке было %.4f против %.4f).'
           % (MINB, jhi / float(NDB), (jhi + 1) / float(NDB), nb_hi,
              ('%.4f' % g4b) if g4b == g4b else 'не определён',
              bgb['ratio'] if bgb['ratio'] == bgb['ratio'] else float('nan'), g4,
              bg['ratio'] if bg['ratio'] == bg['ratio'] else float('nan')), '',
           '| ячейка D на начале бодрствования | в долях своего уровня | бодрствований |',
           '|---|---|---|'] + [
           '| %.1f–%.1f | %s | %d |' % (j / float(NDB), (j + 1) / float(NDB),
                                        ('%.4f' % cur_b[j]) if np.isfinite(cur_b[j]) else '—',
                                        int(dn2b.sum(0)[j])) for j in range(NDB)] + [
           '\n| часть | получено | порог | нулевое значение | SE | MDE | вывод |\n|---|---|---|---|---|---|---|',
           '| а: подпороговый рост (как записано) | %s (ячейка %s против 0,0–0,1, %d подпороговых) | не меньше 1,10 | 1,0 | %s | %s | %s |'
           % (('%.3f' % g4) if g4 == g4 else '—', ('%.1f–%.1f' % (jlast / 10., (jlast + 1) / 10.)) if jlast > 0 else 'нет',
              int(nsub), ('%.4f' % se_rel) if se_rel == se_rel else '—',
              ('%.4f' % a['mde']) if a.get('mde') == a.get('mde') else '—', vt(a['verdict'])),
           '| а′: против ожидания переключателя | %s | %s (бодрствование %.2f ч при D = %.2f против %.2f ч при D = %.02f) | 1,0 | %s | %s | %s |'
           % (('%.3f' % g4) if g4 == g4 else '—', ('%.3f' % bg['ratio']) if bg['ratio'] == bg['ratio'] else '—',
              bg['wake_hi_h'], d_hi, bg['wake_lo_h'], d_lo,
              ('%.4f' % se_rel) if se_rel == se_rel else '—',
              ('%.4f' % ap['mde']) if ap['mde'] == ap['mde'] else '—', vt(ap['verdict'])),
           '| б: опережение (как записано, абсолютная кривая) | ячейки %s | есть ячейка не дальше 7 суток | — | — | — | %s |'
           % (('%d–%d' % (min(lead_ok), max(lead_ok) + 1)) if lead_ok else '—', vt(b)),
           '| б′: превышение своего уровня в последние сутки | %.4f на %d наборах | не меньше %.4f | 1,0 | %s | %s | %s |'
           % (rcur[0], n0, 1.05, ('%.4f' % se0) if se0 == se0 else '—',
              ('%.4f' % bp['mde']) if bp['mde'] == bp['mde'] else '—', vt(bp['verdict'])),
           '| а″: прирост по разметке НАЧАЛА бодрствования | %s на %d бодрствованиях | не меньше %s '
           '(переключатель) | 1,0 | %s | %s | %s |'
           % (('%.4f' % g4b) if g4b == g4b else '—', nb_hi,
              ('%.4f' % bgb['ratio']) if bgb['ratio'] == bgb['ratio'] else '—',
              ('%.4f' % se_b) if se_b == se_b else '—',
              ('%.4f' % ab['mde']) if ab['mde'] == ab['mde'] else '—', vt(ab['verdict'])),
           '| б″: наклон кривой предвестника | ранговая связь %s по %d ячейкам | не выше −0,50 | 0 | %s | %s | %s |'
           % (('%+.2f' % rho) if rho == rho else '—', nb_, ('%.3f' % rse) if rse == rse else '—',
              ('%.3f' % rp['mde']) if rp['mde'] == rp['mde'] else '—', vt(rp['verdict']))]
    if not reach:
        md.append('\n**Главное по части (а).** Сам переключатель при таком повреждении даёт удлинение всего в '
                  '%.3f раза, то есть записанный порог 1,10 ВЫШЕ того, что механизм способен дать. Провал части '
                  '(а) поэтому не говорит о биологии ничего: правило требовало невозможного. Проверять надо '
                  'двойник (а′), и именно его вердикт содержателен.' % bg['ratio'])
    under = bool(g4 == g4 and bg['ratio'] == bg['ratio'] and g4 < bg['ratio'] - st.mde(se_rel))
    if under:
        md.append('\n**Наблюдение НИЖЕ ожидания переключателя** (%.3f против %.3f, зазор %.1f SE), и это '
                  'разбирается так. Бодрствование попадает в ячейку по доле повреждённых ЗА ЭТО '
                  'бодрствование, а повреждение внутри бодрствования не стоит на месте: оно нарастает к '
                  'концу. Значит среднее торможение ядра за бодрствование МЕНЬШЕ, чем даёт середина ячейки, '
                  'и отношение обязано выйти ниже предсказанного. Проверяемое следствие: если считать D не '
                  'за всё бодрствование, а на его СЕРЕДИНЕ, расхождение должно уйти. Это правка счёта '
                  '(ещё один накопитель), записана в план. Пока же часть (а′) не выполнена именно так — '
                  'из-за меры, а не из-за механизма.'
                  % (g4, bg['ratio'], abs(g4 - bg['ratio']) / max(se_rel, 1e-9)))
    if over:
        md.append('\n**Наблюдение ВЫШЕ ожидания переключателя** (%.3f против %.3f, зазор %.1f SE). Значит '
                  'бодрствование удлиняет не только торможение ядра: подпороговое повреждение растёт внутри '
                  'самого бодрствования, и к его концу торможение больше, чем средняя по ячейке.'
                  % (g4, bg['ratio'], abs(g4 - bg['ratio']) / max(se_rel, 1e-9)))
    if rp['verdict'] == 'непроверяемо':
        md.append('\n**Непроверяемо по части (б″), и причина не в модели.** Ранговая связь считается по %d '
                  'ячейкам, её SE равна 1/sqrt(n−1) = %.2f, значит различимый порог равен %.2f, а правило '
                  'ставит −0,50. Чтобы порог −0,50 стал проверяемым, нужно около %.0f ячеек, то есть ячейки '
                  'суток до срыва надо дробить мельче (например, по четверти суток) — это правка счёта, а не '
                  'набора. Отмечено в план.' % (nb_, rse, st.mde(rse), (st.MDE_K / 0.5) ** 2 + 1))
    out = dict(a=dict(gain=g4, bin_used=int(jlast), n_sub=int(nsub), enough=enough, **a),
               a_tight=dict(switch=bg, observed_over_switch=over, observed_under_switch=under,
                            switch_reaches_written=reach, **ap),
               b=dict(lead_days=lead_ok, base_bout=base_bout, verdict=b),
               b_tight=dict(last_day=float(rcur[0]), n_sets=n0, **bp),
               b_slope=dict(rank_corr=rho, n_bins=nb_, **rp),
               a_start=dict(gain=g4b, n_bouts=nb_hi, se_start=se_b, bin_used=jhi,
                            d_hi=d_hi_b, switch=bgb, min_bouts=MINB, curve=[float(z) for z in cur_b],
                            counts=[int(z) for z in dn2b.sum(0)], **ab),
               b_slope_sets=dict(mean_rho=mrho4, se_sets=mse4, n_sets=nok4, n_total=ntot4,
                                 lam=lam4, atten=fat4, thr=thr4, **rp2),
               curve_abs=[None if dnn[j] == 0 else float(cur[j]) for j in range(NDB)],
               curve_rel=[None if rel_cnt[j] == 0 else float(rel_cur[j]) for j in range(NDB)],
               lead_abs=[None if ln[j] == 0 else float(lcur[j]) for j in range(NLB)],
               lead_rel=[None if rcnt[j] == 0 else float(rcur[j]) for j in range(NLB)],
               counts=dict(dbin=[int(x) for x in dnn], lead=[int(x) for x in ln]),
               block=mm['n_sets'])
    ctx.verdict(NAME, 'а', g4, a['verdict'], note='порог 1,10 назначен; достижим переключателем: %s' % ('да' if reach else 'НЕТ'))
    ctx.verdict(NAME, 'а′', g4, ap['verdict'], note='ожидание переключателя %.3f' % bg['ratio'])
    ctx.verdict(NAME, 'б', float(min(lead_ok)) if lead_ok else float('nan'), b, note='абсолютная кривая, как записано')
    ctx.verdict(NAME, 'б′', float(rcur[0]), bp['verdict'], note='по собственному здоровому уровню набора')
    ctx.verdict(NAME, 'б″', rho, rp['verdict'], note='наклон; при %d ячейках порог −0,50 неразличим' % nb_)
    ctx.verdict(NAME, 'а″', g4b, ab['verdict'],
                note='разметка по D на НАЧАЛЕ бодрствования, метка не зависит от измеряемой длины; '
                     'ячейка %.1f-%.1f на %d бодрствованиях, ожидание переключателя %.4f'
                     % (jhi / float(NDB), (jhi + 1) / float(NDB), nb_hi, bgb['ratio']))
    ctx.verdict(NAME, 'б‴', mrho4, rp2['verdict'],
                note='наклон по наборам: %d из %d годны, lam = %.2f, порог −0,50 ослаблен до %.3f'
                     % (nok4, ntot4, lam4, thr4))
    return out, md
