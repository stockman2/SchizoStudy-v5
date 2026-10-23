# -*- coding: utf-8 -*-
"""Сверка слитого ядра с эталоном: px/fast.simulate_fast против px/model.simulate.

Требование: все СЧЁТНЫЕ величины совпадают точно, непрерывные — до 1e-12. Пока сверка не
пройдена, быстрый путь не включается нигде. Гоняется на трёх точках: без связи, со связью,
с праймингом."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from px import conf, fast
from px.model import simulate

INT = {'nb', 'nb_aw', 'nb_late', 'epn', 'epshort', 'epn_m', 'epshort_m', 'ep_sleep', 'sl_in_ep',
       'everdam', 'open_ep', 'n_sr', 'nb_trig', 'wkn', 'wklong', 'wk15', 'nlead', 'gap_n',
       'NGB', 'NSB', 'nbase', 'NB'}
PINS = (('без связи', dict(kd=0.0)), ('со связью', dict(kd=2.0)),
        ('с праймингом', dict(kd=0.0, apr=2.0, taup=0.5)),
        ('асимметрия', dict(kd=2.0, apr=2.0, taup=0.5, tausd=3.0 * 2.625)),
        ('асимметрия наоборот', dict(kd=2.0, tausd=0.4 * 2.625)),
        ('потеря ёмкости', dict(kd=2.0, a_n=0.3, taun_dn=1.0, taun_up=30.0)),
        ('потеря + прайминг', dict(kd=2.0, apr=2.0, taup=0.5, a_n=0.5, taun_dn=0.5, taun_up=60.0)),
        # ЭТИХ ДВУХ ПИНОВ НЕ БЫЛО, и поэтому ядро полгода целилось в сырое P['N0'] вместо
        # уменьшенной ёмкости. Дефект проявлялся ТОЛЬКО при n0_mul < 1, а ни один пин
        # его не трогал: пины на петлю (a_n > 0) шли при n0_mul = 1, где N0c = N0 и
        # расхождения нет. Второй пин обязателен отдельно от первого: статика и петля
        # перемножаются в ntgt, и именно там ошибка была невидима.
        ('статическая ёмкость', dict(kd=2.0, n0_mul=0.7)),
        ('статическая ёмкость + петля', dict(kd=2.0, n0_mul=0.7, a_n=0.5, taun_dn=1.0, taun_up=30.0)))


def main(nsel=80, nd=8, M=400):
    cfg = conf.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'config', 'base.toml'))
    cfg.d['sets']['nsel'] = nsel
    S = cfg.sets(log=lambda m: None); n = cfg.d['noise']; m = cfg.d['model']; q = cfg.d['sets']['q']
    nu0 = S['lo'] * (S['hi'] / S['lo']) ** q
    if not fast.available():
        print('numba недоступна — быстрый путь выключен, сверять нечего.'); return 0
    ok_all = True
    for name, extra in PINS:
        kw = dict(aH=S['A_H'], sigv=m['sigv'], E0=S['E0'], qw=S['qw'], Gbar=S['Gbar'],
                  sigma=n['sigma'], tauc_rel=n['tauc'], ndays=nd, M=M, J=cfg.d['run']['J'],
                  warm=2, sigs=n['sigs'], slpn=n['slpn'], usedm=m['usedm'], taus=m['taus'],
                  vfix=m['vfix'])
        kw.update(extra); kw.setdefault('kd', 0.0)
        a = simulate(S['P'], nu0, S['thr'], S['xhi'], S['xlo'], rng=np.random.default_rng(7), **kw)
        b = fast.simulate_fast(S['P'], nu0, S['thr'], S['xhi'], S['xlo'], rng=np.random.default_rng(7), **kw)
        worst = 0.0; bad = []; nbit = 0; ntot = 0
        def cmp(k, p, r):
            nonlocal worst, nbit, ntot
            p = np.asarray(p, float); r = np.asarray(r, float); ntot += 1
            if np.array_equal(p, r): nbit += 1; return
            base = k.split('[')[0]
            if base in INT: bad.append('%s: счётная величина разошлась' % k); return
            f = np.isfinite(p) & np.isfinite(r)
            d = float((np.abs(p - r)[f] / np.maximum(np.abs(p[f]), 1e-30)).max()) if f.any() else 0.0
            worst = max(worst, d)
            if d > 1e-12: bad.append('%s: %.2e' % (k, d))
        for k in a:
            if isinstance(a[k], tuple):
                for i, (p, r) in enumerate(zip(a[k], b[k])): cmp('%s[%d]' % (k, i), p, r)
            else: cmp(k, a[k], b[k])
        ok = not bad; ok_all &= ok
        print('%-14s полей %d, побитово %d, наибольшее расхождение %.2e — %s'
              % (name, ntot, nbit, worst, 'СОШЛОСЬ' if ok else 'РАСХОЖДЕНИЕ: ' + '; '.join(bad[:4])))
    print('\nИТОГ: %s' % ('быстрый путь принят' if ok_all else 'БЫСТРЫЙ ПУТЬ НЕ ПРИНЯТ'))
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
