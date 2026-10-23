# -*- coding: utf-8 -*-
"""РЕГРЕССИЯ ПЕРЕЕЗДА.

Проверяется три вещи, по убыванию авторитета:
 2. px.model.simulate против mass_core6.simulate НА ЭТОЙ ЖЕ МАШИНЕ — требуется ПОБИТОВОЕ
    совпадение. Это и есть ответ на вопрос «не изменил ли переезд счёт»;
 1. выборка наборов в px.conf даёт те же наборы, что цикл отбора в mass_core6.py;
 3. то же против эталонного датасета в ref/, чтобы дальше не гонять mass_core6 каждый раз.

ПРО ПЕРЕНОСИМОСТЬ ЭТАЛОНА. Пункт 3 побитов ТОЛЬКО на той машине, где эталон сделан. На другом
процессоре numpy берёт другой путь для exp, pow и sqrt, последние биты расходятся, и
времяусреднённые итоги (taw, Dbar, Dmbar, Ebar, Lbar, Gbar_, prbar, tdam) расходятся в
пятнадцатом знаке. Измерено прямо: сдвиг ОДНОГО параметра на ОДИН последний бит даёт
расхождение 3e-16...3e-15 за 14 суток счёта, а счётные и крайние величины (nb, epn, epshort,
wkmax) остаются побитовыми. То есть усиления ошибки нет, система в этом режиме не хаотична.
Поэтому эталон несёт отпечаток машины, и при его несовпадении пункт 3 сверяется ПО ДОПУСКУ
1e-12, а не побитово, и печатает настоящее расхождение.

Запуск:  python tests/test_regression.py            сверка
         python tests/test_regression.py --make     сделать эталон НА ЭТОЙ машине (нужно один раз)
"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_sv = sys.argv; sys.argv = [_sv[0]]
import mass_core6 as old
sys.argv = _sv
from px import conf, core, model
from px.batch import Block, Runner

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ref')
TOL = 1e-12          # допуск для сверки МЕЖДУ машинами: измеренный разброс 3e-15, так что это с запасом


def fingerprint():
    """Отпечаток машины: платформа, numpy и три трансцендентных значения, на которых
    процессоры и расходятся. Нужен, чтобы отличить «переезд сломал счёт» от «другой процессор»."""
    import platform
    v = np.array([np.exp(0.1), np.power(10.0, 0.3), np.sqrt(2.0), np.exp(-37.5), np.power(0.7, 1.3)])
    return '%s|%s|%s' % (platform.machine(), np.__version__, '_'.join('%.17g' % x for x in v))
# закреплённые конфигурации: по одной на вариацию параметров, которые мы правда гоняем
PINS = {
    'base':    dict(nsel=40, days=14, M=400, J=8, warm=4, sigs=0.0, apr=0.0),
    'slownoise': dict(nsel=40, days=14, M=400, J=8, warm=4, sigs=0.08, apr=0.0),
    'priming': dict(nsel=40, days=14, M=400, J=8, warm=4, sigs=0.0, apr=2.0, taup=0.5),
}


def pinned_sets(nsel, seed=83, K=200000):
    """Выборка наборов ТЕМ ЖЕ порядком, что в mass_core6.main."""
    rng = np.random.default_rng(seed); Sm = old.sample(K, rng); sel = []
    for k in range(K):
        p = {n: float(Sm[n][k, 0]) for n in old.NAMES}; w = old.window(old.curve(p)[0])
        if not w or w[1] / w[0] <= 1.05: continue
        st = old.states_at(p, math.sqrt(w[0] * w[1]))
        if len(st) < 2: continue
        sel.append(dict(k=k, lo=w[0], hi=w[1], xlo=st[0], xhi=st[-1], r=float(rng.uniform(2, 4))))
        if len(sel) >= nsel: break
    P = {n: np.array([float(Sm[n][d['k'], 0]) for d in sel]) for n in old.NAMES}
    P['r'] = np.array([d['r'] for d in sel])
    g = lambda f: np.array([d[f] for d in sel])
    return dict(P=P, nsel=len(sel), lo=g('lo'), hi=g('hi'), xlo=g('xlo'), xhi=g('xhi'),
                thr=0.5 * (g('xlo') + g('xhi')), bfac=np.array([float(Sm['bfac'][d['k'], 0]) for d in sel]))


def check_sets():
    cfg = conf.load(); cfg.d['sets']['nsel'] = 40; cfg.d['cache']['dir'] = 'cache_test'
    A = cfg._make(log=lambda *a: None); B = pinned_sets(40)
    bad = []
    for f in ('lo', 'hi', 'xlo', 'xhi', 'thr', 'bfac'):
        if not np.array_equal(A[f], B[f]): bad.append(f)
    for n in sorted(A['P']):
        if not np.array_equal(A['P'][n], B['P'][n]): bad.append('P.' + n)
    return (A['nsel'] == B['nsel'] and not bad), bad


def check_band():
    """Полоса и закон сохранения разрывов: арифметика px.conf против mass_core6 (разд. 1в).
    Добавлено 14.10 после того, как я набрал её по памяти вместо извлечения и получил полосу
    из 80 наборов вместо 158."""
    from px import conf
    cfg = conf.load(); cfg.d['sets']['nsel'] = 300
    S = cfg.sets(log=lambda *a: None)
    E0, qw, Gb, g0, g1, A_H, _, _ = old.calibrate(M=1600, nd=60)
    gg = np.linspace(1., 1 - old.GDEP, 5)
    gfb = float(np.array([old.wc_fix(g, old.PE0 + old.PE1)[0] for g in gg])[[-1, 0]][0]
                / old.wc_fix(1.0, old.PE0 + old.PE1)[0])
    bud = 2 * cfg.d['noise']['sigma'] + cfg.d['noise']['sigs']; q = cfg.d['sets']['q']
    day_pk = 1 + A_H * (g1 - Gb)
    gu = np.log((S['hi'] / S['lo']) ** (1 - q) / day_pk)
    gd = np.log((S['hi'] / S['lo']) ** q * gfb * (2 - day_pk))
    uc = math.log(old.SR_AMP * day_pk)
    band = (gu <= uc) & (gd <= bud)
    ok = bool(np.array_equal(S['band'], band) and abs(S['A_H'] - A_H) < 1e-15 and abs(S['gain_fb'] - gfb) < 1e-12)
    return ok, int(S['band'].sum()), int(band.sum())


def run_pair(S, pin):
    """Один и тот же блок старым и новым simulate, аргументы слово в слово."""
    E0, qw, Gb, g0, g1, A_H, _, _ = old.calibrate(M=pin['M'], nd=60)
    kw = dict(kd=0.0, aH=A_H, sigv=0.10, E0=E0, qw=qw, Gbar=Gb, sigma=0.08, tauc_rel=0.3,
              ndays=pin['days'], M=pin['M'], J=pin['J'], warm=pin['warm'], sigs=pin['sigs'],
              apr=pin.get('apr', 0.0), taup=pin.get('taup', 1.0))
    num = S['lo'] * (S['hi'] / S['lo']) ** 0.10
    a = old.simulate(S['P'], num, S['thr'], S['xhi'], S['xlo'], rng=np.random.default_rng(4242), **kw)
    b = model.simulate(S['P'], num, S['thr'], S['xhi'], S['xlo'], rng=np.random.default_rng(4242), **kw)
    return a, b


def cmp_res(a, b):
    bad = []
    for k in sorted(a):
        va, vb = a[k], b[k]
        if isinstance(va, tuple):
            for i, (x, y) in enumerate(zip(va, vb)):
                if not np.array_equal(np.asarray(x), np.asarray(y)): bad.append('%s[%d]' % (k, i))
        elif isinstance(va, np.ndarray):
            if not np.array_equal(va, vb): bad.append(k)
        elif va != vb: bad.append(k)
    return bad


def main():
    make = '--make' in sys.argv
    ok1, bad1 = check_sets()
    print('1) выборка наборов совпадает с mass_core6: %s%s' % ('ДА' if ok1 else 'НЕТ', '' if ok1 else ' (%s)' % bad1))
    ok0, nb_new, nb_ref = check_band()
    print('0) полоса и закон разрывов совпадают с mass_core6: %s (наборов %d против %d)'
          % ('ДА' if ok0 else 'НЕТ', nb_new, nb_ref))
    S = pinned_sets(40); allok = ok1 and ok0
    os.makedirs(REF, exist_ok=True)
    for name, pin in PINS.items():
        a, b = run_pair(S, pin)
        bad = cmp_res(a, b)
        print('2) %-10s px.model против mass_core6: %s%s' % (name, 'ПОБИТОВО СОВПАЛО' if not bad else 'РАСХОЖДЕНИЕ', '' if not bad else ' в %s' % bad[:6]))
        allok = allok and not bad
        fp = os.path.join(REF, 'ref_%s.npz' % name)
        flat = {}
        for k, v in b.items():
            if isinstance(v, tuple):
                for i, x in enumerate(v): flat['%s__%d' % (k, i)] = np.asarray(x)
            else: flat[k] = np.asarray(v)
        if make or not os.path.exists(fp):
            np.savez_compressed(fp, _fp=np.array([fingerprint()]), **flat)
            print('   эталон записан НА ЭТОЙ машине: %s (%.0f КБ)' % (fp, os.path.getsize(fp) / 1024.))
        else:
            z = np.load(fp)
            same = ('_fp' in z.files) and (str(z['_fp'][0]) == fingerprint())
            if same:
                bad2 = [k for k in flat if k not in z.files or not np.array_equal(flat[k], z[k])]
                print('3) %-10s против эталона (та же машина, побитово): %s%s' % (
                    name, 'СОВПАЛО' if not bad2 else 'РАСХОЖДЕНИЕ', '' if not bad2 else ' в %s' % bad2[:6]))
                allok = allok and not bad2
            else:
                worst = 0.0; who = ''
                for k in flat:
                    if k not in z.files: continue
                    x = np.asarray(flat[k], float); y = np.asarray(z[k], float)
                    if x.shape != y.shape: worst = float('inf'); who = k + ' (форма)'; break
                    d = np.abs(x - y) / np.maximum(np.abs(y), 1e-30)
                    if np.nanmax(d) > worst: worst = float(np.nanmax(d)); who = k
                ok2 = worst <= TOL
                print('3) %-10s эталон сделан на ДРУГОЙ машине, сверка по допуску %g: %s '
                      '(наибольшее расхождение %.1e в %s)' % (name, TOL, 'в допуске' if ok2 else 'ВНЕ ДОПУСКА', worst, who))
                print('   это ожидаемо: последние биты exp/pow/sqrt у процессоров разные. '
                      'Сделайте свой эталон: python tests/test_regression.py --make')
                allok = allok and ok2
    print('\nИТОГ: %s' % ('ПЕРЕЕЗД ПРИНЯТ: пункт 2 побитовый, остальное сошлось'
                            if allok else 'ЕСТЬ РАСХОЖДЕНИЯ. Смотрите пункт 2: если он побитовый, '
                                          'переезд в порядке, а расходится только эталон с чужой машины'))
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
