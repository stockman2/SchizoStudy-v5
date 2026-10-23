# -*- coding: utf-8 -*-
"""Чтение конфига (TOML, стандартный tomllib, без зависимостей) и подготовка наборов.

Выборка наборов детерминирована зерном и стоит около 30 секунд. Она КЭШИРУЕТСЯ в файл по
хешу тех полей конфига, которые на неё влияют, и переиспользуется: на двухминутных прогонах
это четверть времени.
"""
import hashlib, json, math, os, time
import numpy as np
try: import tomllib
except ModuleNotFoundError: import tomli as tomllib
from . import core as C
from .consts import SLPN, TSLOW, USE_DM, VFIX, TAU_S, DSWING, GDEP, SR_AMP
from .sleep import pr_free, calibrate
from .cortex import wc_fix

BASE = dict(sets=dict(K=200000, nsel=300, seed=83, q=0.10),
            run=dict(days=60, M=1600, J=32, warm=16, seed=4242, nproc=0, out='out'),
            noise=dict(sigma=0.08, tauc=0.3, sigs=0.0, tslow=TSLOW, slpn=SLPN),
            model=dict(sigv=0.10, kd=2.0, usedm=1.0 if USE_DM else 0.0, taus=TAU_S, vfix=VFIX),
            cache=dict(dir='cache'))


def load(path=None):
    cfg = {k: dict(v) for k, v in BASE.items()}
    if path:
        with open(path, 'rb') as f: raw = tomllib.load(f)
        for sec, vals in raw.items():
            cfg.setdefault(sec, {}).update(vals)
    return Config(cfg, path)


class Config:
    def __init__(self, d, path=None): self.d = d; self.path = path

    def __getitem__(self, k): return self.d[k]
    def get(self, sec, key, dflt=None): return self.d.get(sec, {}).get(key, dflt)

    def block_defaults(self):
        """Значения построчных полей по умолчанию: их задаёт конфиг, а блок может переопределить."""
        n = self.d['noise']; m = self.d['model']; r = self.d['run']
        return dict(kd=m['kd'], sigv=m['sigv'], sigma=n['sigma'], tauc_rel=n['tauc'],
                    sigs=n['sigs'], tslow=n['tslow'], slpn=n['slpn'], usedm=m['usedm'],
                    taus=m['taus'], vfix=m['vfix'], ndays=r['days'], M=r['M'], J=r['J'],
                    warm=r['warm'], q=self.d['sets']['q'], seed=r['seed'])

    def key(self):
        """Хеш полей, влияющих на выборку наборов (и только их)."""
        s = self.d['sets']
        return hashlib.sha1(json.dumps([s['K'], s['nsel'], s['seed']], sort_keys=True).encode()).hexdigest()[:12]

    def sets(self, log=print):
        cdir = self.d['cache']['dir']; os.makedirs(cdir, exist_ok=True)
        fp = os.path.join(cdir, 'sets_%s.npz' % self.key())
        if os.path.exists(fp):
            z = np.load(fp, allow_pickle=True); log('наборы из кэша: %s' % fp)
            S = {k: z[k] for k in z.files if k not in ('P_keys',)}
            S['P'] = {k: S.pop('P_' + k) for k in [str(x) for x in z['P_keys']]}
            S['nsel'] = int(S['nsel']); S['bfac'] = S['bfac']
        else:
            S = self._make(log); np.savez_compressed(
                fp, P_keys=np.array(sorted(S['P'])), nsel=S['nsel'],
                **{('P_' + k): v for k, v in S['P'].items()},
                **{k: v for k, v in S.items() if k not in ('P', 'nsel')})
            log('наборы посчитаны и положены в кэш: %s' % fp)
        self._layers(S, log)
        return S

    def _make(self, log):
        s = self.d['sets']; K = int(s['K']); need = int(s['nsel'])
        rng = np.random.default_rng(int(s['seed'])); Sm = C.sample(K, rng); sel = []
        t0 = time.time()
        for k in range(K):
            p = {n: float(Sm[n][k, 0]) for n in C.NAMES}; w = C.window(C.curve(p)[0])
            if not w or w[1] / w[0] <= 1.05: continue
            st = C.states_at(p, math.sqrt(w[0] * w[1]))
            if len(st) < 2: continue
            sel.append((k, w[0], w[1], st[0], st[-1], float(rng.uniform(2, 4))))
            if len(sel) >= need: break
        log('выборка: %d наборов с двумя состояниями из %d, %.0f с' % (len(sel), K, time.time() - t0))
        idx = np.array([z[0] for z in sel])
        P = {n: np.array([float(Sm[n][i, 0]) for i in idx]) for n in C.NAMES}
        P['r'] = np.array([z[5] for z in sel])
        return dict(P=P, nsel=len(sel), lo=np.array([z[1] for z in sel]), hi=np.array([z[2] for z in sel]),
                    xlo=np.array([z[3] for z in sel]), xhi=np.array([z[4] for z in sel]),
                    thr=0.5 * (np.array([z[3] for z in sel]) + np.array([z[4] for z in sel])),
                    bfac=np.array([float(Sm['bfac'][i, 0]) for i in idx]))

    def _layers(self, S, log):
        """Величины слоёв сна и коры: считаются быстро, в кэш не идут."""
        E0, qw, Gb, g0, g1, A_H, ew, es = calibrate(M=self.d['run']['M'], nd=60)
        wH, sH, pH = pr_free(M=self.d['run']['M'], nd=60)[:3]
        S['qw'] = qw; S['Gbar'] = Gb; S['E0'] = E0; S['g0'] = g0; S['g1'] = g1; S['A_H'] = A_H
        S['sigv'] = self.d['model']['sigv']; S['sigma'] = self.d['noise']['sigma']
        S['tauc_rel'] = self.d['noise']['tauc']
        gg = np.linspace(1.0, 1 - GDEP, 5); Ew = np.array([wc_fix(g, 2.4)[0] for g in gg])
        S['gain_fb'] = float(Ew[-1] / Ew[0]); S['wake_h'] = wH; S['sleep_h'] = sH; S['period_h'] = pH
        # ЗАКОН СОХРАНЕНИЯ РАЗРЫВОВ И ПОЛОСА. Перенесено из mass_core6.py (разд. 1в) ДОСЛОВНО.
        # 14.10 здесь была ошибка переезда: я набрал это по памяти, а не извлёк текстом, и получил
        # (1) up_cap без множителя SR_AMP, то есть W* меньше в 1,5 раза, и (2) условие полосы по
        # ширине окна вместо условия по ДВУМ РАЗРЫВАМ по отдельности. Полоса вышла другая, и числа
        # по полосе разошлись с прошлыми прогонами, хотя simulate совпадал побитово.
        n = self.d['noise']; q = self.d['sets']['q']
        sig_tot = math.sqrt(n['sigma'] ** 2 + n['sigs'] ** 2); bud = 2 * n['sigma'] + n['sigs']
        day_pk = 1 + S['A_H'] * (g1 - Gb)
        gap_up = np.log((S['hi'] / S['lo']) ** (1 - q) / day_pk)
        gap_dn = np.log((S['hi'] / S['lo']) ** q * S['gain_fb'] * (2 - day_pk))
        up_cap = math.log(SR_AMP * day_pk)          # разрыв вверх перекрывает всплеск, а не шум
        S['budget'] = bud; S['sigma_total'] = sig_tot; S['day_peak'] = day_pk
        S['gap_up'] = gap_up; S['gap_dn'] = gap_dn; S['up_cap'] = up_cap
        S['Wstar'] = math.exp(up_cap + bud) * day_pk / ((2 - day_pk) * S['gain_fb'])
        S['band'] = (gap_up <= up_cap) & (gap_dn <= bud)
        log('полоса: %d наборов из %d (W* = %.3f, бюджет шума %.3f)'
            % (int(S['band'].sum()), S['nsel'], S['Wstar'], S['budget']))
