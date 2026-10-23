# -*- coding: utf-8 -*-
"""Ядро клетки: кривая стационаров, окно нагрузки, выборка наборов, разброс уязвимости."""
# ПЕРЕНЕСЕНО ИЗ mass_core6.py ДОСЛОВНО, автоматически, не перенабрано руками.
# Любая правка здесь обязана проходить регрессионный тест tests/test_regression.py,
# который требует ПОБИТОВОГО совпадения с эталонным датасетом в ref/.
import math
import numpy as np
from .consts import *

# ---------- Ядро клетки: перенесено из sleep_core2.py дословно ----------
def curve(P, u=UG, a=1.0, frozen=False):
    c = P['N0'] * a + 0 * u if frozen else P['N0'] * a / (1 + u / P['KN']) + P['beta'] * u / (P['KB'] + u)
    cm = c ** P['m']; x = cm / (cm + P['mu0'] * u * u / (1 + u)); return P['e0'] * c ** P['gam'] * u + x * u / (1 + u), x, c

def jac(P, u, x, c, a=1.0):
    eps = P['e0'] * c ** P['gam']; Gu = eps + x / (1 + u) ** 2; ux = -(u / (1 + u)) / Gu; uc = -(P['gam'] * P['e0'] * c ** (P['gam'] - 1) * u) / Gu
    h = u * u / (1 + u); hp = u * (2 + u) / (1 + u) ** 2; q = -P['N0'] * a / (P['KN'] * (1 + u / P['KN']) ** 2) + P['beta'] * P['KB'] / (P['KB'] + u) ** 2
    f1x = -c ** P['m'] - P['mu0'] * h - P['mu0'] * x * hp * ux; f1c = P['m'] * c ** (P['m'] - 1) * (1 - x) - P['mu0'] * x * hp * uc; f2x = q * ux / P['T']; f2c = (q * uc - 1) / P['T']
    return f1x + f2c, f1x * f2c - f1c * f2x

def rhs(t, y, p, nu, a=1.0):
    x, c = y[0], max(y[1], 1e-9); eps = p['e0'] * c ** p['gam']; b = eps + x - nu; s = math.sqrt(b * b + 4 * eps * nu); u = 2 * nu / (b + s) if b > 0 else (s - b) / (2 * eps)
    return [c ** p['m'] * (1 - x) - p['mu0'] * x * u * u / (1 + u), (p['N0'] * a / (1 + u / p['KN']) + p['beta'] * u / (p['KB'] + u) - c) / p['T']]

def window(nu):
    i = np.where(np.diff(nu) < 0)[0]
    if not len(i): return None
    return float(nu[i[0]:].min()), float(nu[i[0]])

def sample(K, rng):
    """beta теперь задаётся не сама по себе, а кратностью к N0: beta = N0 * U(0, BFMAX).
    Причина — измерение: индукция белка сульфиредоксина достигает 2,85 раза, а прежняя выборка
    beta из [0, 1] при N0 из [0,3; 1] такой кратности почти не допускала."""
    S = {}
    for k, (lo, hi, kind) in RANGES.items(): S[k] = (10 ** rng.uniform(np.log10(lo), np.log10(hi), K) if kind == 'log' else rng.uniform(lo, hi, K))[:, None]
    S['bfac'] = rng.uniform(0, BFMAX, K)[:, None]; S['beta'] = S['N0'] * S['bfac']
    return S

def states_at(p, nuv):
    nuc, xc, cc = curve(p); tr, dt_ = jac(p, UG, xc, cc); a0 = nuc[:-1] - nuv; a1 = nuc[1:] - nuv; cr = np.where(a0 * a1 < 0)[0]; out = []
    for i in cr:
        w = a0[i] / (a0[i] - a1[i])
        if (dt_[i] + w * (dt_[i + 1] - dt_[i])) > 0 and (tr[i] + w * (tr[i + 1] - tr[i])) < 0: out.append(float(xc[i] + w * (xc[i + 1] - xc[i])))
    return sorted(out)

# ---------- Счёт ----------
NDB = 10; NLB = 8; RING = 10; NHD = 20
# Потолок числа записанных эпизодов на набор. За 100 суток при длительности около 7 суток и
# промежутке около 10 их выходит 5–6, так что 24 — запас втрое; переполнение считается.
NEPR = 24          # ячейки: по повреждению, по суткам до срыва, память бодрствований, гистограмма D

def vuln(J, sigv, fixr=VFIX):
    """Множители нагрузки на клетку, mean = 1 ровно, при любой сетке и любом J.
    Две сетки вероятностей, и выбор между ними — главный вопрос раздела 13:
      fixr = 0  «равновероятные квантили»: p_j = (j + 0,5)/J. Диапазон ПОЛЗЁТ с J: чем больше клеток,
                тем более уязвима самая уязвимая. Тогда J — параметр модели, а не разрешение счёта.
      fixr = 1  «закреплённый диапазон, края»: p_j равномерно от P0V до 1 − P0V, крайние точки закреплены.
                Диапазон от J не зависит вовсе, J — только мелкость разбиения. При J = J0V = 32 эта сетка
                СОВПАДАЕТ с прежней до последнего знака, поэтому прежние прогоны при J = 32 в силе.
      fixr = 2  «закреплённый диапазон, середины»: p_j = P0V + (1 − 2*P0V)*(j + 0,5)/J. По смыслу выглядит
                естественнее (обрезанная логнормаль, выбранная равновероятно), но опора при этом всё равно
                ползёт с J, и сходится оно хуже. Остаётся как диагностика, в счёте не используется.
    sigv и fixr — скаляры или массивы по строкам; возвращается (число строк, J)."""
    sv = np.atleast_1d(np.asarray(sigv, float)); fr = np.atleast_1d(np.asarray(fixr, float))
    if sv.size < fr.size: sv = np.repeat(sv, fr.size // sv.size)
    if fr.size < sv.size: fr = np.repeat(fr, sv.size // fr.size)
    if J == 1: return np.ones((len(sv), 1))
    from scipy.stats import norm
    G = np.stack([(np.arange(J) + 0.5) / J,
                  P0V + (1 - 2 * P0V) * np.arange(J) / max(J - 1, 1),
                  P0V + (1 - 2 * P0V) * (np.arange(J) + 0.5) / J])
    p = G[np.rint(fr).astype(int)]
    v = np.exp(sv[:, None] * norm.ppf(p)); return v / v.mean(1, keepdims=True)
