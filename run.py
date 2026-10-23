# -*- coding: utf-8 -*-
"""Точка входа пайплайна.

    python run.py --rules f19                      посчитать только Ф19
    python run.py --rules f19,f15 --config config/noslow.toml
    python run.py --list                           какие правила есть
    python run.py --ledger                         показать журнал вердиктов
    python run.py --rules all --days 20 --sets 40  быстрая проверка, что всё доходит до конца
    python run.py --rules all --dry                оценка: сколько блоков и сколько времени, без счёта
    python run.py --rules f13 --nocache            посчитать заново, не читая кэш блоков

РЕЗУЛЬТАТ КАЖДОГО БЛОКА КЭШИРУЕТСЯ в cache/blocks/. Повторный прогон того же блока — чтение
файла вместо счёта. Ключ кэша включает хеш исходного кода модели, поэтому правка px/model.py
и соседей сама обесценивает кэш: старые блоки к новым не подмешаются.

Считаются ТОЛЬКО блоки, нужные выбранным правилам. Одинаковые блоки разных правил
склеиваются и считаются один раз; хеш блока печатается рядом с числами правила.
"""
import argparse, json, os, sys, time
import numpy as np
from px import conf, rules
from px.batch import Runner
from px.report import Report, Ledger
from px import stats as st

T0 = time.time()
def stage(m): print('[%6.1f с] %s' % (time.time() - T0, m), flush=True)


# Постоянные стоимости счёта, ИЗМЕРЕННЫЕ на этой машине 14.10 (два ядра):
#   один блок, 40 наборов, 76 суток по 1600 шагов, один процесс  -> 60,5 с
#   шесть блоков по 300 наборов, 26 суток, два процесса          -> 99,5 с
# Отсюда стоимость шага батча: T_A + T_B * (строк в процессе), строка = набор * J/32.
# Калибровка 14.10 сделана на numpy. Замер 16.10 на слитом ядре: 48 блоков за 4198 с, то есть
# 87 с на блок против 271 с по этим константам — множитель 3.11. Поэтому при включённом быстром
# пути оценка делится на него. Одна точка замера на две неизвестные, поэтому масштабируются обе
# константы, а не выводятся заново: это оценка времени, а не измерение.
def _fast_on():
    """Включён ли быстрый путь — от этого зависит оценка времени втрое."""
    try:
        from px import fast
        return fast.enabled()
    except Exception:
        return False


FAST_GAIN = 3.11
T_A, T_B = 4.09e-4, 2.20e-6


def estimate(runner, cfg, S, nproc, per_rule):
    """Оценка числа блоков и времени ДО счёта: главное, что надо знать перед прогоном на тысячах
    наборов. Считается по измеренным постоянным, поэтому это оценка, а не гарантия."""
    groups = {}
    for k, b in runner.want.items(): groups.setdefault(b.group(cfg), []).append(k)
    nsel = S['nsel']; tot = 0.0; L = ['', 'ОЦЕНКА ПРОГОНА (счёт не запускался)', '']
    L.append('| группа: суток / шагов / J / разгон | блоков | строк | процессов | оценка, мин |')
    L.append('|---|---|---|---|---|')
    for g, ks in sorted(groups.items(), key=lambda z: str(z[0])):
        ndays, M, J, warm = g; NB = len(ks)
        npr = max(1, min(nproc, NB)); rows = nsel * NB * (J / 32.) / npr
        t = int((ndays + warm) * M) * (T_A + T_B * rows)
        if _fast_on(): t /= FAST_GAIN
        tot += t
        L.append('| %d / %d / %d / %d | %d | %d | %d | %.1f |' % (ndays, M, J, warm, NB, nsel * NB, npr, t / 60))
    L += ['', 'всего блоков %d, оценка счёта %.0f мин (%.1f ч)' % (runner.n_blocks + len(runner.want), tot / 60, tot / 3600), '']
    L.append('| правило | блоков просит | из них своих (не нужны другим) |')
    L.append('|---|---|---|')
    own = [w for w, kk in per_rule.items() if not kk]
    if own:
        L.append('| %s | 0 — свой счёт, в смету выше не входит | — |' % ', '.join(sorted(own)))
    cnt = {}
    for w, kk in per_rule.items():
        for k in kk: cnt[k] = cnt.get(k, 0) + 1
    for w, kk in sorted(per_rule.items()):
        L.append('| %s | %d | %d |' % (w, len(kk), sum(1 for k in kk if cnt[k] == 1)))
    L += ['', 'Наборов %d. Время растёт прямо пропорционально числу наборов и числу суток, поэтому'
          ' половина наборов — половина времени.' % nsel, '']
    return '\n'.join(L)


class Ctx:
    """То, что видит правило: наборы, конфиг, свои блоки по тегу, журнал."""
    def __init__(self, S, cfg, runner, keys, ledger):
        self.S = S; self.cfg = cfg; self._r = runner; self._k = keys; self._led = ledger
    def b(self, tag): return self._r.get(self._k[tag])
    def key(self, tag): return self._k[tag]
    def verdict(self, rule, part, value, verdict, note=''):
        self._led.record(rule, part, value, verdict, sorted(set(self._k.values())), note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config'); ap.add_argument('--rules', default='')
    ap.add_argument('--out'); ap.add_argument('--nproc', type=int)
    ap.add_argument('--days', type=int); ap.add_argument('--sets', type=int)
    # Сид ШУМА, не сид наборов. Нужен для независимой реализации: популяция та же,
    # случайный ход другой. Правило, уже увидевшее ответ на одном сиде, на нём же
    # считает подтверждение, а не предсказание.
    ap.add_argument('--seed', type=int)
    ap.add_argument('--m', type=int); ap.add_argument('--j', type=int)
    ap.add_argument('--list', action='store_true'); ap.add_argument('--ledger', action='store_true')
    ap.add_argument('--dry', action='store_true'); ap.add_argument('--nocache', action='store_true')
    a = ap.parse_args()
    if a.list: print('правила:', ', '.join(rules.available())); return
    if a.ledger: print(Ledger().table()); return
    # Без --config берём config/base.toml, если он есть. Прежде брались ВСТРОЕННЫЕ значения по
    # умолчанию, и прогон молча шёл не на том, что лежит в конфиге (300 наборов вместо 2000).
    if not a.config and os.path.exists(os.path.join('config', 'base.toml')):
        a.config = os.path.join('config', 'base.toml')
    cfg = conf.load(a.config)
    stage('конфиг: %s' % (a.config or 'встроенные значения по умолчанию'))
    if a.days: cfg.d['run']['days'] = a.days
    if a.sets: cfg.d['sets']['nsel'] = a.sets
    if a.seed: cfg.d['run']['seed'] = a.seed
    if a.m: cfg.d['run']['M'] = a.m
    if a.j: cfg.d['run']['J'] = a.j
    if a.out: cfg.d['run']['out'] = a.out
    want = rules.available() if a.rules in ('all', '') else [x.strip() for x in a.rules.split(',')]
    bad = [w for w in want if w not in rules.available()]
    assert not bad, 'нет таких правил: %s' % bad
    nproc = a.nproc or cfg.d['run']['nproc'] or (os.cpu_count() or 1)

    if a.dry:
        S = cfg.sets(log=stage)
        runner = Runner(S, cfg, nproc=nproc, log=stage)
        mods = {w: rules.load(w) for w in want}
        per = {}
        for w, m in mods.items():
            kk = [runner.request(b) for b in m.needs(cfg)]
            per[w] = kk
        print(estimate(runner, cfg, S, nproc, per))
        return

    stage('проверка самих критериев на синтетике с известным ответом')
    sc = st.self_test(); pw = st.power_selftest()
    print('   мощность: ' + json.dumps({k: v for k, v in pw.items() if k.endswith('_ok') or k == 'all_ok'}, ensure_ascii=False))
    assert pw['all_ok'], 'машинерия мощности не прошла самопроверку — вердикты «непроверяемо» считать нельзя'
    print('   ' + json.dumps({k: v for k, v in sc.items() if k.endswith('_ok') or k == 'all_ok'}, ensure_ascii=False))
    assert sc['all_ok'], 'критерии не прошли самопроверку — считать нельзя'

    S = cfg.sets(log=stage)
    rep = Report(cfg.d['run']['out'], 'Редокс-модель PV: прогон правил %s' % ', '.join(want))
    led = Ledger()
    rep.md('Конфиг: %s. Правила: %s. Наборов %d, суток %d, шагов %d, J %d, разгон %d, зерно %d.\n'
           % (a.config or 'по умолчанию', ', '.join(want), S['nsel'], cfg.d['run']['days'],
              cfg.d['run']['M'], cfg.d['run']['J'], cfg.d['run']['warm'], cfg.d['run']['seed']),
           'Самопроверка критериев: ложная тревога знакового критерия %.3f, промах границы медианы %.3f, '
           'CV² при вшитом глухом периоде %.3f — все три в допуске.\n'
           % (sc['sign_false_alarm'], sc['med_bound_miss'], sc['cv2_with_refractory']))
    rep.json('selftest', sc)
    rep.json('setup', dict(rules=want, nsel=S['nsel'], **cfg.d['run'], band=int(S['band'].sum()),
                           Wstar=S['Wstar'], budget=S['budget'], A_H=S['A_H']))

    runner = Runner(S, cfg, nproc=nproc, log=stage, cache=not a.nocache)
    mods = {w: rules.load(w) for w in want}
    keys = {}
    for w, m in mods.items():
        keys[w] = {b.tag: runner.request(b) for b in m.needs(cfg)}
    stage('запрошено блоков: %d, после склейки %d%s' % (sum(len(v) for v in keys.values()), len(runner.want),
          '' if a.nocache else ', версия кода модели %s' % runner.code_ver))
    runner.run_all()
    stage('блоков из кэша %d, посчитано %d' % (runner.n_hit, runner.n_blocks))

    for w, m in mods.items():
        stage('правило %s' % w)
        out, md = m.run(Ctx(S, cfg, runner, keys[w], led))
        rep.json(w, out); rep.md(*md)
    rep.json('cache', dict(hits=runner.n_hit, computed=runner.n_blocks, code_version=runner.code_ver,
                           dir=runner.cache_dir))
    led.run(dict(rules=want, blocks=runner.n_blocks, cached=runner.n_hit, t_calc=runner.t_calc,
                 config=a.config or 'default', date=time.strftime('%Y-%m-%d %H:%M')))
    led.save()
    rep.md('\n## Журнал вердиктов (накопленный)\n', led.table())
    rep.json('time_s', round(time.time() - T0, 1))
    rep.save()
    stage('готово: %s, блоков посчитано %d, счёт %.0f с' % (cfg.d['run']['out'], runner.n_blocks, runner.t_calc))


if __name__ == '__main__':
    main()
