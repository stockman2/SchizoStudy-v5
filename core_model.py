# -*- coding: utf-8 -*-
"""
core_model.py — ядро модели: активный фермент x, ёмкость защиты c, рецептор N (алгебраически), перекись u (алгебраически).

    nu = eps*u + x*u/(1 + u),   eps = e0*c^gam          баланс перекиси (выведено из модели Selvaggio и соавт.)
    N  = N0/(1 + u/KN)                                   рецептор, обратимо подавляется перекисью (форма — допущение)
    x' = c^m*(1 - x) - mu0*x*u^2/(1 + u)                 время в единицах времени восстановления фермента (выведено)
    c' = [N*a + beta*u/(KB + u) - c]/T                   защита следует за активностью рецептора; beta — индукция стрессом

  Параметры (9): e0, mu0 — быстрый слой; m, gam — как восстановление и обходная очистка зависят от защиты;
  KN, N0 — рецептор; beta, KB — индукция защиты стрессом (отрицательная связь); T — отношение масштабов времени.
  Нагрузка nu и активность a — внешние.

ПРАВИЛА, ЗАФИКСИРОВАННЫЕ ДО СЧЁТА
  П1. Оценка «порог нагрузки падает как N0^(m/2)» принимается, если при gam = 0 измеренный наклон отличается от m/2
      не больше чем на 0,1 во всех строках, где порог лежит при u < 0,3.
  П2. «Второй, медленный уровень двух состояний есть», если не менее 5 % случайной выборки параметров — наборы,
      где быстрый слой с замороженной защитой не даёт двух состояний ни при какой нагрузке, а ядро даёт.
  П3. «Ядро даёт несимметрию потери и возврата», если медиана отношения (время возврата)/(время потери) не меньше 30.
  П4. Замедление возврата у границы окна идёт по закону «призрака» (показатель -1/2), если наклон лежит в -0,5 ± 0,1.

Как считается
  При заданной перекиси u стационарное состояние единственно: N(u), c(u), x(u), nu(u). Поэтому все стационарные состояния
  при всех нагрузках лежат на одной кривой, а число состояний при данной нагрузке равно числу решений nu(u) = nu.
  Устойчивость — по следу и определителю матрицы Якоби (она выписана аналитически и сверена с численной).
  Классы набора параметров — по числу УСТОЙЧИВЫХ стационарных состояний при одной нагрузке:
    «два состояния» — при какой-то нагрузке их два; «колебания» — при какой-то нагрузке нет ни одного
    (система на плоскости ограничена, поэтому без устойчивого состояния она обязана выйти на цикл).
  Классификация сверяется прямым интегрированием из случайных начальных точек (раздел 2, «прямая проверка»).

  Правка после первой прогонки. В первой редакции «колебания» искались по признаку «состояние одно и неустойчиво»,
  а единственность проверялась с ошибкой у вершины складки; прямое интегрирование подтверждало колебания лишь в 5 случаях из 12.
  Теперь число устойчивых состояний считается при каждой нагрузке напрямую. Правило П2 считается по строгому определению
  (два устойчивых состояния), порог 5 % не менялся; число по старому определению (наличие складки) печатается рядом.
  В разделе 4 наборы с неустойчивыми ветвями исключены из расчёта времени потери и возврата.

Проекции девятимерного пространства параметров: параметры выбираются случайно, каждый набор получает класс поведения.
  fig2 — одномерные: доля каждого класса вдоль каждого параметра;
  fig3, fig4 — двумерные карты доли наборов с двумя состояниями и с колебаниями по четырём самым влиятельным параметрам;
  fig5 — проекция на две оси, лучше всего разделяющие классы (линейный дискриминант), и на осмысленные сочетания параметров.
  Само пространство состояний двумерно (x, c), проекции ему не нужны: фазовая плоскость — на fig1.

Запуск:  python core_model.py            быстрый вариант, около минуты
         python core_model.py --full     выборка в 10 раз больше и больше прямых проверок, несколько минут
Нужны numpy, scipy, matplotlib. Результат: папка core_out/ — report.md, results.json и пять рисунков fig1…fig5.
"""
import os, sys, json, time, math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

T0 = time.time(); FULL = '--full' in sys.argv
def stage(m): print('[%5.1f с] %s' % (time.time() - T0, m), flush=True)

UG = np.logspace(-4, 4, 641)
NAMES = ['e0', 'mu0', 'm', 'gam', 'KN', 'N0', 'beta', 'KB', 'T']
REF = dict(e0=0.02, mu0=0.3, m=2.0, gam=0.5, KN=3.0, N0=1.0, beta=0.1, KB=10.0, T=50.0)
RANGES = dict(e0=(0.005, 0.1, 'log'), mu0=(0.01, 10, 'log'), m=(1, 2, 'lin'), gam=(0, 1, 'lin'), KN=(0.1, 100, 'log'), N0=(0.3, 1, 'lin'), beta=(0, 1, 'lin'), KB=(1, 100, 'log'), T=(10, 300, 'log'))
CLS = ['нет скачка', 'два состояния', 'колебания', 'два состояния и колебания']
GRP = ['колебания', 'неустойчивое рядом с устойчивым', 'два состояния, ветви устойчивы', 'нет скачка']
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']; MRK = ['o', 's', '^', 'D']

def curve(P, u=UG, a=1.0, frozen=False):
    """Стационарная кривая. frozen=True — защита заморожена на здоровом уровне N0*a (быстрый слой один)."""
    c = P['N0'] * a + 0 * u if frozen else P['N0'] * a / (1 + u / P['KN']) + P['beta'] * u / (P['KB'] + u)
    cm = c ** P['m']; x = cm / (cm + P['mu0'] * u * u / (1 + u)); return P['e0'] * c ** P['gam'] * u + x * u / (1 + u), x, c

def jac(P, u, x, c, a=1.0):
    """След и определитель матрицы Якоби системы (x, c) в точке стационарной кривой."""
    eps = P['e0'] * c ** P['gam']; Gu = eps + x / (1 + u) ** 2; ux = -(u / (1 + u)) / Gu; uc = -(P['gam'] * P['e0'] * c ** (P['gam'] - 1) * u) / Gu
    h = u * u / (1 + u); hp = u * (2 + u) / (1 + u) ** 2; q = -P['N0'] * a / (P['KN'] * (1 + u / P['KN']) ** 2) + P['beta'] * P['KB'] / (P['KB'] + u) ** 2
    f1x = -c ** P['m'] - P['mu0'] * h - P['mu0'] * x * hp * ux; f1c = P['m'] * c ** (P['m'] - 1) * (1 - x) - P['mu0'] * x * hp * uc; f2x = q * ux / P['T']; f2c = (q * uc - 1) / P['T']
    return f1x + f2c, f1x * f2c - f1c * f2x

def U_of(x, c, nu, e0, gam):
    eps = e0 * c ** gam; b = eps + x - nu; s = np.sqrt(b * b + 4 * eps * nu); return np.where(b > 0, 2 * nu / (b + s), (s - b) / (2 * eps))

def rhs(t, y, p, nu, a=1.0):
    x, c = y[0], max(y[1], 1e-9); eps = p['e0'] * c ** p['gam']; b = eps + x - nu; s = math.sqrt(b * b + 4 * eps * nu); u = 2 * nu / (b + s) if b > 0 else (s - b) / (2 * eps)
    return [c ** p['m'] * (1 - x) - p['mu0'] * x * u * u / (1 + u), (p['N0'] * a / (1 + u / p['KN']) + p['beta'] * u / (p['KB'] + u) - c) / p['T']]

def integ(p, nu, y0, tend, ev=None, a=1.0, dense=False):
    return solve_ivp(rhs, [0, tend], y0, args=(p, nu, a), method='LSODA', rtol=1e-8, atol=1e-11, events=ev, dense_output=dense)

def threshold(P, a=1.0, frozen=False):
    """Порог нагрузки: складка, если она есть, иначе точка наибольшей крутизны. Возвращает (nu, u, есть ли складка)."""
    nu = curve(P, UG, a, frozen)[0]; i = np.where(np.diff(nu) < 0)[0]
    if len(i): return float(nu[i[0]]), float(UG[i[0]]), True
    s = np.gradient(np.log(UG), np.log(nu)); j = int(np.argmax(s)); return float(nu[j]), float(UG[j]), False

def window(nu):
    """Окно двух состояний по строке nu(u): (нижняя граница, верхняя) или None."""
    i = np.where(np.diff(nu) < 0)[0]
    if not len(i): return None
    return float(nu[i[0]:].min()), float(nu[i[0]])

def sample(K, rng):
    S = {}
    for k, (lo, hi, kind) in RANGES.items(): S[k] = (10 ** rng.uniform(np.log10(lo), np.log10(hi), K) if kind == 'log' else rng.uniform(lo, hi, K))[:, None]
    return S

def classify_chunk(S, NL=160):
    """Класс поведения каждого набора. Число и устойчивость состояний считаются прямо: при каждой нагрузке из сетки ищутся
    все пересечения стационарной кривой с этой нагрузкой, след и определитель в пересечении — линейной интерполяцией."""
    u = UG[None, :]; lu = np.log(UG); nu, x, c = curve(S, u); nuf = curve(S, u, frozen=True)[0]; tr, det = jac(S, u, x, c); K = nu.shape[0]
    d = np.diff(nu, axis=1); fold = (d < 0).any(1); foldf = (np.diff(nuf, axis=1) < 0).any(1); uns = (det > 0) & (tr > 0); hasu = uns.any(1)
    two = fold.copy(); osc = np.zeros(K, bool); coex = np.zeros(K, bool); wr = np.full(K, np.nan); wrf = np.full(K, np.nan); wfold = np.full(K, np.nan); Losc = np.full(K, np.nan); Lco = np.full(K, np.nan); uco = np.full(K, np.nan)
    for k in range(K):
        w = window(nu[k]); wf = window(nuf[k])
        if wf: wrf[k] = wf[1] / wf[0]
        if w: wfold[k] = w[1] / w[0]; wr[k] = wfold[k]
        if not hasu[k]: continue
        nk = nu[k]; cand = nk[uns[k]]; lo = cand.min(); hi = cand.max()
        if w: lo = min(lo, w[0]); hi = max(hi, w[1])
        Lg = np.exp(np.linspace(np.log(lo) - 0.03, np.log(hi) + 0.03, NL))[:, None]; a0 = nk[None, :-1] - Lg; a1 = nk[None, 1:] - Lg; cr = a0 * a1 < 0
        with np.errstate(divide='ignore', invalid='ignore'): t = np.where(cr, a0 / (a0 - a1), 0.0)
        trc = tr[k][None, :-1] + t * np.diff(tr[k])[None, :]; dtc = det[k][None, :-1] + t * np.diff(det[k])[None, :]; ns = (cr & (dtc > 0) & (trc < 0)).sum(1); nun_ = cr & (dtc > 0) & (trc > 0); cx = (ns >= 1) & (nun_.sum(1) >= 1)
        two[k] = (ns >= 2).any(); osc[k] = (ns == 0).any(); coex[k] = cx.any(); wr[k] = np.nan
        if two[k]: g = Lg[ns >= 2, 0]; wr[k] = g.max() / g.min()
        if osc[k]: g = Lg[ns == 0, 0]; Losc[k] = g[len(g) // 2]
        if coex[k]: j = np.where(cx)[0]; j = j[len(j) // 2]; q = np.where(nun_[j])[0][0]; Lco[k] = Lg[j, 0]; uco[k] = np.exp(lu[q] + t[j, q] * (lu[q + 1] - lu[q]))
    sg = np.sign(d); sd = np.sign(det[:, :-1]); agree = float((sg == sd).mean())
    cls = np.where(two & osc, 3, np.where(osc, 2, np.where(two, 1, 0)))
    return dict(cls=cls, fold=fold, foldf=foldf, two=two, osc=osc, coex=coex, hasu=hasu, wr=wr, wrf=wrf, wfold=wfold, Losc=Losc, Lco=Lco, uco=uco, agree=np.full(K, agree))

def classify(S, step=10000):
    """То же по частям, чтобы большая выборка не занимала много памяти."""
    K = len(S['e0']); parts = [classify_chunk({n: S[n][i:i + step] for n in NAMES}) for i in range(0, K, step)]; C = {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}; C['agree'] = float(C['agree'].mean()); return C

def Phi(mu):
    """Порог быстрого слоя: два состояния у него есть тогда и только тогда, когда eps < Phi(mu)."""
    ug = np.logspace(-5, 7, 1500)[None, :]; mu = np.atleast_1d(mu)[:, None]; return ((mu * ug ** 2 - 1) / (1 + ug + mu * ug ** 2) ** 2).max(1)

def states_at(p, L):
    """Все стационарные состояния набора p при нагрузке L: список (u, x, c, след, определитель)."""
    ug = np.logspace(-4, 4, 6401); f = curve(p, ug)[0] - L; out = []
    for j in np.where(f[:-1] * f[1:] < 0)[0]:
        u = brentq(lambda z: float(curve(p, z)[0]) - L, ug[j], ug[j + 1]); _, x, c = curve(p, u); tr, det = jac(p, u, x, c); out.append((u, float(x), float(c), float(tr), float(det)))
    return out

def healthy_state(p, nu_b):
    """Здоровое состояние при нагрузке nu_b: точка на первой, нижней по перекиси, ветви."""
    nu = curve(p)[0]; i = np.where(np.diff(nu) < 0)[0]; iu = i[0] if len(i) else len(UG) - 1
    u = brentq(lambda z: curve(p, z)[0] - nu_b, UG[0] * 1e-3, UG[iu]); _, x, c = curve(p, u); return float(x), float(c), u

def pulse(p, nu_b, nu_p, extra, tmax_rec=400.0):
    """Импульс нагрузки nu_p из здорового состояния при nu_b. Потеря: x < x_h/2. Импульс держится ещё extra после потери."""
    xh, ch, _ = healthy_state(p, nu_b); thr = 0.5 * xh
    dn = lambda t, y, *a: y[0] - thr; dn.terminal = True; dn.direction = -1; up = lambda t, y, *a: y[0] - thr; up.terminal = True; up.direction = 1
    s1 = integ(p, nu_p, [xh, ch], 50 * p['T'], ev=dn)
    if not len(s1.t_events[0]): return np.nan, np.nan, None
    tl = float(s1.t_events[0][0]); s2 = integ(p, nu_p, s1.y[:, -1], extra) if extra > 0 else None; y1 = s2.y[:, -1] if s2 is not None else s1.y[:, -1]
    s3 = integ(p, nu_b, y1, tmax_rec * p['T'], ev=up); tr_ = float(s3.t_events[0][0]) if len(s3.t_events[0]) else np.inf
    return tl, tr_, (s1, s2, s3)

def main():
    os.makedirs('core_out', exist_ok=True); rng = np.random.default_rng(11); R = {}; L = ['# Ядро модели: активный фермент, ёмкость защиты, рецептор\n', 'Режим: %s.\n' % ('полный' if FULL else 'быстрый')]

    # ---------- 0. Проверки ----------
    stage('проверки'); S0 = sample(400, rng); u = UG[None, ::8]; nu, x, c = curve(S0, u); tr, det = jac(S0, u, x, c); err = 0.0
    for k in range(0, 400, 20):
        p = {n: float(S0[n][k, 0]) for n in NAMES}
        for j in range(0, u.shape[1], 9):
            y = np.array([x[k, j], c[k, j]]); f0 = np.array(rhs(0, y, p, nu[k, j])); J = np.zeros((2, 2))
            for q in range(2):
                h = 1e-6 * max(abs(y[q]), 1e-3); yp = y.copy(); yp[q] += h; ym = y.copy(); ym[q] -= h; J[:, q] = (np.array(rhs(0, yp, p, nu[k, j])) - np.array(rhs(0, ym, p, nu[k, j]))) / (2 * h)
            sc = abs(J).max(); err = max(err, abs(np.trace(J) - tr[k, j]) / sc, abs(np.linalg.det(J) - det[k, j]) / sc ** 2, abs(f0).max())
    pf = dict(REF, KN=1e12, beta=0.0, gam=0.0, N0=1.0); ok = 0; ntest = 2000
    from math import inf
    Ugr = np.logspace(-6, 8, 6001)
    for _ in range(ntest):
        e0 = 10 ** rng.uniform(-3, -0.3); mu0 = 10 ** rng.uniform(-3, 3); phi = ((mu0 * Ugr ** 2 - 1) / (1 + Ugr + mu0 * Ugr ** 2) ** 2).max()
        if abs(e0 / phi - 1) < 0.02: ok += 1; continue
        ok += ((np.diff(curve(dict(pf, e0=e0, mu0=mu0), Ugr, frozen=True)[0]) < 0).any() == (e0 < phi))
    ys = integ(REF, 0.05, [1.0, 1.0], 40 * REF['T']).y[:, -1]; xh, ch, uh = healthy_state(REF, 0.05)
    L += ['## 0. Проверки\n', '- матрица Якоби, аналитическая против численной, и невязка стационарной кривой (20 наборов × 9 точек): наибольшее относительное расхождение %.1e' % err,
          '- предел без медленных связей: критерий e0 < Phi(mu0) совпал с наличием складки в %d из %d случайных наборов' % (ok, ntest),
          '- опорный набор, нагрузка 0,05: интегрирование даёт x = %.5f, c = %.5f; стационарная кривая — x = %.5f, c = %.5f\n' % (ys[0], ys[1], xh, ch)]

    # ---------- 1. Запас и рецептор ----------
    stage('порог и рецептор'); L += ['## 1. Как порог нагрузки зависит от рецептора (связи через перекись выключены: KN → ∞, beta = 0)\n', 'Оценка: порог ∝ N0^(m/2). Наклон измерен по N0 от 0,3 до 1; e0 = 0,02.\n', '| mu0 | m | gam | измеренный наклон | m/2 | перекись на пороге при N0 = 1 | складка |\n|---|---|---|---|---|---|---|']
    N0s = np.linspace(0.3, 1, 12); T1 = []
    for mu0 in (0.03, 0.3, 3.0, 30.0, 300.0):
        for m in (1.0, 1.5, 2.0):
            for gam in (0.0, 0.5, 1.0):
                th = [threshold(dict(REF, mu0=mu0, m=m, gam=gam, N0=n, KN=1e12, beta=0.0), frozen=True) for n in N0s]; sl = float(np.polyfit(np.log(N0s), np.log([t[0] for t in th]), 1)[0]); T1.append((mu0, m, gam, sl, th[-1][1], th[-1][2]))
                L.append('| %g | %.1f | %.1f | %.2f | %.2f | %.3g | %s |' % (mu0, m, gam, sl, m / 2, th[-1][1], 'да' if th[-1][2] else 'нет'))
    rows = [t for t in T1 if t[2] == 0.0 and t[4] < 0.3]; dev = max(abs(t[3] - t[1] / 2) for t in rows) if rows else np.nan; v1 = bool(rows) and dev <= 0.1
    L.append('\n**П1:** строк с gam = 0 и порогом при u < 0,3: %d; наибольшее отклонение наклона от m/2: %.2f. Вывод: %s.' % (len(rows), dev, 'оценка принята' if v1 else 'оценка не принята')); R['P1'] = dict(rows=len(rows), max_dev=dev, verdict=bool(v1))
    L += ['\nСвязанный случай: нагрузка nu = nu1·a и защита N0·a растут с одной и той же активностью a. Какую активность выдерживает клетка:\n', '| m | gam | наклон ln a* по ln N0 | оценка m/(2 − m) | a* при N0 = 1 |\n|---|---|---|---|---|']; ag = np.logspace(-2.5, 2.5, 500)
    for m in (1.0, 1.25, 1.5, 1.75):
        for gam in (0.0, 0.5):
            base = dict(REF, mu0=30.0, m=m, gam=gam, KN=1e12, beta=0.0); nu1 = threshold(dict(base, N0=1.0), a=1.0, frozen=True)[0] / 3; As = []
            for n in N0s:
                thr = np.array([threshold(dict(base, N0=n), a=a_, frozen=True)[0] for a_ in ag]); j = np.where(nu1 * ag >= thr)[0]; As.append(ag[j[0]] if len(j) and j[0] > 0 else np.nan)
            As = np.array(As); g = np.isfinite(As); sl = float(np.polyfit(np.log(N0s[g]), np.log(As[g]), 1)[0]) if g.sum() > 3 else np.nan
            L.append('| %.2f | %.1f | %.2f | %.2f | %.3g |' % (m, gam, sl, m / (2 - m), As[-1]))
    L += ['\nЧто добавляет обратная связь «перекись подавляет рецептор» (опорный набор, меняется только KN):\n', '| KN | порог нагрузки | отношение к порогу без связи | окно двух состояний: отношение границ |\n|---|---|---|---|']; th0 = threshold(dict(REF, KN=1e12))[0]
    for KN in (1e12, 100.0, 30.0, 10.0, 3.0, 1.0, 0.3):
        P = dict(REF, KN=KN); t = threshold(P); w = window(curve(P)[0]); L.append('| %s | %.4f | %.2f | %s |' % ('∞' if KN > 1e9 else '%g' % KN, t[0], t[0] / th0, '%.2f' % (w[1] / w[0]) if w else 'нет'))

    # ---------- 2. Случайная выборка параметров ----------
    K = 200000 if FULL else 20000; stage('случайная выборка: %d наборов' % K); S = sample(K, rng); C = classify(S); cls = C['cls']; share = [float((cls == i).mean()) for i in range(4)]
    only_slow = C['two'] & ~C['foldf']; only_fast = C['foldf'] & ~C['two']; both = C['two'] & C['foldf'] & np.isfinite(C['wr']); hb = C['hasu'] & C['fold'] & C['two'] & np.isfinite(C['wr'])
    L += ['\n## 2. Случайная выборка параметров (%d наборов; диапазоны — в конце отчёта)\n' % K, '- знак определителя совпал со знаком наклона стационарной кривой в %.2f %% точек (складка = слияние состояний)' % (100 * C['agree']),
          '', 'Классы — по числу устойчивых стационарных состояний: «два состояния» — при какой-то нагрузке их два; «колебания» — при какой-то нагрузке нет ни одного.\n', '| класс | доля выборки |\n|---|---|'] + ['| %s | %.3f |' % (CLS[i], share[i]) for i in range(4)]
    L += ['', '| | доля выборки |\n|---|---|', '| быстрый слой один даёт два состояния | %.3f |' % C['foldf'].mean(), '| ядро даёт два устойчивых состояния | %.3f |' % C['two'].mean(), '| два состояния только благодаря медленной связи | %.3f |' % only_slow.mean(), '| два состояния у быстрого слоя, но не у ядра | %.3f |' % only_fast.mean(),
          '| у ядра есть складка (без учёта устойчивости ветвей) | %.3f |' % C['fold'].mean(), '| складка есть, а двух устойчивых состояний нет ни при какой нагрузке | %.3f |' % (C['fold'] & ~C['two']).mean(), '| ветвь теряет устойчивость раньше своей складки (неустойчивое состояние рядом с устойчивым) | %.3f |' % C['coex'].mean(),
          '', '| ширина окна двух состояний (отношение границ) | медиана | 10-й и 90-й процентили |\n|---|---|---|', '| быстрый слой один | %.2f | %.2f – %.2f |' % tuple(np.nanpercentile(C['wrf'], [50, 10, 90])), '| ядро | %.2f | %.2f – %.2f |' % tuple(np.nanpercentile(C['wr'], [50, 10, 90])),
          '| наборы, где окно есть у обоих: показатель, в который окно ядра шире (ln/ln) | %.2f | %.2f – %.2f |' % tuple(np.nanpercentile((np.log(C['wr']) / np.log(C['wrf']))[both], [50, 10, 90]))]
    if hb.sum() > 10: L.append('| наборы, где ветвь теряет устойчивость до складки: доля окна по складкам, где устойчивы обе ветви (ln/ln) | %.2f | %.2f – %.2f |' % tuple(np.nanpercentile((np.log(C['wr']) / np.log(C['wfold']))[hb], [50, 10, 90])))
    v2 = only_slow.mean() >= 0.05; L.append('\n**П2:** доля наборов, где два состояния даёт только медленная связь: %.3f (по одному наличию складки, как в первой редакции: %.3f). Вывод: %s.' % (only_slow.mean(), (C['fold'] & ~C['foldf']).mean(), 'второй уровень есть' if v2 else 'второй уровень не подтверждён'))
    R['P2'] = dict(share=share, fold_fast=float(C['foldf'].mean()), two_core=float(C['two'].mean()), fold_core=float(C['fold'].mean()), only_slow=float(only_slow.mean()), only_fast=float(only_fast.mean()), coex=float(C['coex'].mean()), verdict=bool(v2))
    imp = {}
    for n in NAMES:
        v = S[n][:, 0]; q = np.quantile(v, np.linspace(0, 1, 9)); b = np.clip(np.searchsorted(q, v, side='right') - 1, 0, 7); fr = np.array([[(cls[b == j] == i).mean() for j in range(8)] for i in range(4)]); imp[n] = (fr, 0.5 * (q[1:] + q[:-1]), float((fr.max(1) - fr.min(1)).max()))
    order = sorted(NAMES, key=lambda n: -imp[n][2]); L += ['\nВлияние параметров (наибольший перепад доли какого-либо класса вдоль параметра):\n', '| параметр | перепад доли |\n|---|---|'] + ['| %s | %.2f |' % (n, imp[n][2]) for n in order]
    # проекция на оси, лучше всего разделяющие классы (линейный дискриминант), и на осмысленные сочетания параметров
    Z = np.column_stack([np.log10(S[n][:, 0]) if RANGES[n][2] == 'log' else S[n][:, 0] for n in NAMES]); Z = (Z - Z.mean(0)) / Z.std(0); Sw = np.zeros((9, 9)); Sb = np.zeros((9, 9))
    for i in range(4):
        g = cls == i
        if g.sum() > 9: zc = Z[g]; mc = zc.mean(0); Sw += (zc - mc).T @ (zc - mc); Sb += g.sum() * np.outer(mc, mc)
    ev, evec = np.linalg.eig(np.linalg.solve(Sw, Sb)); oi = np.argsort(-ev.real)[:2]; W = evec[:, oi].real; W /= np.linalg.norm(W, axis=0); W *= np.sign(W[np.argmax(abs(W), axis=0), [0, 1]]); PR = Z @ W; sep = ev.real[oi] / ev.real.sum()
    L += ['\nОси, лучше всего разделяющие классы (параметры приведены к единому масштабу; логарифмические — по логарифму). Доля разделения: ось 1 — %.2f, ось 2 — %.2f.\n' % (sep[0], sep[1]), '| параметр | вес на оси 1 | вес на оси 2 |\n|---|---|---|'] + ['| %s | %+.2f | %+.2f |' % (NAMES[j], W[j, 0], W[j, 1]) for j in np.argsort(-abs(W[:, 0]))]
    marg = np.concatenate([np.log10(Phi(S['mu0'][i:i + 20000, 0] / S['N0'][i:i + 20000, 0] ** S['m'][i:i + 20000, 0])) for i in range(0, K, 20000)]) - np.log10(S['e0'][:, 0] * S['N0'][:, 0] ** S['gam'][:, 0]); ind = S['beta'][:, 0] / S['N0'][:, 0]
    L += ['\nОсмысленные сочетания: запас быстрого слоя M = lg[Phi(mu0/N0^m) / (e0·N0^gam)] (при M > 0 быстрый слой сам даёт два состояния) и относительная индукция beta/N0.\n', '| | M < −0,3 | −0,3…0 | 0…0,3 | M > 0,3 |\n|---|---|---|---|---|']
    mb = [(-np.inf, -0.3), (-0.3, 0), (0, 0.3), (0.3, np.inf)]
    for i in range(4): L.append('| доля класса «%s» | ' % CLS[i] + ' | '.join('%.2f' % (cls[(marg >= lo_) & (marg < hi_)] == i).mean() if ((marg >= lo_) & (marg < hi_)).sum() else '—' for lo_, hi_ in mb) + ' |')
    L.append('| наборов в столбце | ' + ' | '.join('%d' % ((marg >= lo_) & (marg < hi_)).sum() for lo_, hi_ in mb) + ' |')

    # ---------- 2а. Прямая проверка классификации интегрированием ----------
    stage('прямая проверка классификации'); npair = 60 if FULL else 20; pool = [np.where(C['osc'])[0], np.where(C['coex'])[0], np.where(C['two'] & ~C['hasu'])[0], np.where(cls == 0)[0]]; tested = 0; bad = 0; slow_ = 0; cyc_co = 0; tab = {}
    for q in range(npair):
        g = pool[q % 4]
        if not len(g): continue
        k = int(g[rng.integers(len(g))]); p = {n: float(S[n][k, 0]) for n in NAMES}; kind = q % 4
        Lq = float(C['Losc'][k]) if kind == 0 else float(C['Lco'][k]) if kind == 1 else float(np.sqrt(np.prod(window(curve(p)[0])))) if kind == 2 else float(curve(p, 10 ** rng.uniform(-1, 1.5))[0])
        st_ = [z for z in states_at(p, Lq) if z[4] > 0 and z[3] < 0]; ends = set(); cyc = False
        for _ in range(4):
            y0 = [rng.uniform(0.01, 1), rng.uniform(0.05, p['N0'] + p['beta'])]; s_ = integ(p, Lq, y0, 300 * p['T'], dense=True); yy = s_.sol(np.linspace(220 * p['T'], 300 * p['T'], 3000)); ye = yy[:, -1]
            if np.ptp(yy[0]) > 0.01 or np.ptp(yy[1]) > 0.01 * yy[1].mean(): cyc = True; continue
            dist = [abs(ye[0] - z[1]) + abs(ye[1] - z[2]) for z in st_]
            if dist and min(dist) < 2e-3: ends.add(int(np.argmin(dist)))
            elif max(abs(np.array(rhs(0, ye, p, Lq))) * np.array([1.0, p['T']])) > 1e-4: slow_ += 1
            else: bad += 1
        tested += 1; bad += int(len(st_) == 0 and not cyc); cyc_co += int(cyc and len(st_) > 0); key = (kind, len(st_), ('цикл' if not ends else 'цикл и %d' % len(ends)) if cyc else '%d' % len(ends)); tab[key] = tab.get(key, 0) + 1
    L += ['\n### Прямая проверка классификации\n', 'Случайные пары (набор, нагрузка) из четырёх групп: колебания; неустойчивое состояние рядом с устойчивым; два состояния без неустойчивых ветвей; нет скачка. Из четырёх случайных начальных точек система интегрируется 300 T; конечные состояния сверяются с предсказанными устойчивыми.\n',
          '| группа | предсказано устойчивых состояний | найдено интегрированием | пар |\n|---|---|---|---|'] + ['| %s | %d | %s | %d |' % (GRP[k_[0]], k_[1], k_[2], v_) for k_, v_ in sorted(tab.items())]
    L.append('\n- пар проверено: %d; траекторий, пришедших в состояние, которого нет среди предсказанных, либо затухших при предсказании «нет устойчивых»: %d; траекторий, не успевших сойтись за 300 T: %d; пар, где цикл найден рядом с устойчивым состоянием: %d' % (tested, bad, slow_, cyc_co))
    R['check_classes'] = dict(tested=tested, mismatches=bad, not_converged=slow_, cycle_with_state=cyc_co)

    # ---------- 3. Колебания ----------
    stage('колебания: прямая проверка'); idx = np.where(C['osc'])[0]; nchk = 40 if FULL else 12; L += ['\n## 3. Колебания\n', '- наборов, где при какой-то нагрузке нет ни одного устойчивого стационарного состояния: %d (%.3f выборки)' % (len(idx), len(idx) / K)]; per = []; duty = []; conf = 0; ex = None
    for k in idx[:nchk]:
        p = {n: float(S[n][k, 0]) for n in NAMES}; nu_ = float(C['Losc'][k]); s = integ(p, nu_, [1.0, p['N0']], 240 * p['T'], dense=True); tt = np.linspace(160 * p['T'], 240 * p['T'], 12000); yy = s.sol(tt); xx = yy[0]
        if np.ptp(xx) > 0.01 or np.ptp(yy[1]) > 0.01 * yy[1].mean():
            conf += 1; mid = 0.5 * (xx.max() + xx.min()); up = np.where((xx[:-1] < mid) & (xx[1:] >= mid))[0]; duty.append(float((xx < mid).mean()))
            if len(up) > 2: per.append(float(np.mean(np.diff(tt[up])) / p['T']))
            if ex is None or (np.ptp(xx) > np.ptp(ex[3][0]) and len(up) > 4): ex = (p, nu_, tt, yy)
    L.append('- прямое интегрирование первых %d таких наборов в середине диапазона нагрузок без устойчивого состояния: незатухающие колебания в %d (ожидалось во всех: на плоскости ограниченная система без устойчивого стационарного состояния обязана выйти на цикл)' % (min(len(idx), nchk), conf))
    if per: L.append('- период, в единицах медленного времени T: медиана %.2f, от %.2f до %.2f' % (np.median(per), min(per), max(per)))
    if duty: L.append('- доля периода, которую фермент проводит ниже середины своего размаха («повреждённая» фаза): медиана %.2f, от %.2f до %.2f' % (np.median(duty), min(duty), max(duty)))
    if len(idx): L.append('- медианы параметров у наборов с колебаниями против всей выборки: beta %.2f против %.2f; KB %.2f против %.2f; KN %.2f против %.2f; mu0 %.2f против %.2f; e0 %.3f против %.3f' % (np.median(S['beta'][idx]), np.median(S['beta']), np.median(S['KB'][idx]), np.median(S['KB']), np.median(S['KN'][idx]), np.median(S['KN']), np.median(S['mu0'][idx]), np.median(S['mu0']), np.median(S['e0'][idx]), np.median(S['e0'])))
    jdx = np.where(C['coex'])[0]; ncyc = 0; nst_ = 0; nco = 300 if FULL else 30; side = [0, 0]
    for k in jdx[:nco]:
        p = {n: float(S[n][k, 0]) for n in NAMES}; nu_ = float(C['Lco'][k]); x0, c0 = curve(p, float(C['uco'][k]))[1:]; cyc = False; sts = [z for z in states_at(p, nu_) if z[4] > 0 and z[3] < 0]
        if sts: side[int(sts[0][0] > float(C['uco'][k]))] += 1
        for y0 in ([min(float(x0) * 1.01, 1.0), float(c0)], [1.0, p['N0']], [0.02, p['N0'] + p['beta']]):
            yy = integ(p, nu_, y0, 300 * p['T'], dense=True).sol(np.linspace(220 * p['T'], 300 * p['T'], 3000)); cyc = cyc or np.ptp(yy[0]) > 0.01 or np.ptp(yy[1]) > 0.01 * yy[1].mean()
        ncyc += int(cyc); nst_ += int(not cyc)
    L += ['- наборов, где неустойчивое состояние соседствует с устойчивым (ветвь теряет устойчивость раньше своей складки): %d (%.3f выборки). Из первых %d: устойчивым остаётся здоровое состояние — %d, повреждённое — %d' % (len(jdx), len(jdx) / K, min(len(jdx), nco), side[0], side[1]),
          '- те же %d наборов, интегрирование из трёх начальных точек (у неустойчивого состояния, из здорового угла, из повреждённого угла): все траектории пришли в устойчивое состояние — %d наборов; найден цикл рядом с устойчивым состоянием («покой и колебания при одной нагрузке») — %d' % (min(len(jdx), nco), nst_, ncyc)]
    R['osc'] = dict(n=int(len(idx)), checked=int(min(len(idx), nchk)), confirmed=conf, periods=per, duty=duty, coex_n=int(len(jdx)), coex_checked=int(min(len(jdx), nco)), coex_to_state=nst_, coex_with_cycle=ncyc, coex_stable_healthy=side[0], coex_stable_damaged=side[1])

    # ---------- 4. Потеря и возврат ----------
    stage('потеря и возврат'); cand = np.where((cls == 1) & ~C['hasu'] & (C['wr'] > 1.05))[0][:(600 if FULL else 150)]; ratio = []; TL = []; TRr = []
    for k in cand:
        p = {n: float(S[n][k, 0]) for n in NAMES}; w = window(curve(p)[0])
        try: tl, tr_, _ = pulse(p, 0.8 * w[0], 2 * w[1], p['T'])
        except Exception: continue
        if np.isfinite(tl) and np.isfinite(tr_): ratio.append(tr_ / tl); TL.append(tl); TRr.append(tr_ / p['T'])
    ratio = np.array(ratio); L += ['\n## 4. Потеря и возврат (наборы с двумя состояниями и без неустойчивых ветвей; исходная нагрузка 0,8 нижней границы окна; импульс вдвое выше верхней границы, держится ещё одно медленное время T после потери)\n',
          '- наборов просчитано: %d' % len(ratio), '- время потери, в единицах времени восстановления фермента: медиана %.2f (10–90 %%: %.2f – %.2f)' % tuple(np.percentile(TL, [50, 10, 90])),
          '- время возврата, в единицах медленного времени T: медиана %.2f (10–90 %%: %.2f – %.2f)' % tuple(np.percentile(TRr, [50, 10, 90])), '- отношение времени возврата ко времени потери: медиана %.1f (10–90 %%: %.1f – %.1f)' % tuple(np.percentile(ratio, [50, 10, 90]))]
    v3 = np.median(ratio) >= 30; L.append('\n**П3:** медиана отношения %.1f. Вывод: %s.' % (np.median(ratio), 'несимметрия есть' if v3 else 'несимметрия не подтверждена')); R['P3'] = dict(n=int(len(ratio)), median=float(np.median(ratio)), verdict=bool(v3))
    w = window(curve(REF)[0]); L += ['\nОпорный набор: окно двух состояний от %.4f до %.4f (отношение %.2f).' % (w[0], w[1], w[1] / w[0]), '\nДлительность импульса и возврат (исходная нагрузка 0,8 нижней границы, импульс 2 верхних):\n', '| импульс держится после потери ещё, в T | время потери | время возврата, в T |\n|---|---|---|']; DUR = []
    for ex_ in (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        tl, tr_, sol = pulse(REF, 0.8 * w[0], 2 * w[1], ex_ * REF['T']); DUR.append((ex_, tl, tr_ / REF['T'], sol)); L.append('| %g | %.2f | %.3f |' % (ex_, tl, tr_ / REF['T']))
    L += ['\nЗамедление у границы окна (импульс держится 3T; исходная нагрузка = нижняя граница × (1 − delta)):\n', '| delta | время возврата, в T |\n|---|---|']; dl = np.array([0.3, 0.1, 0.03, 0.01, 0.003, 0.001]); GT = []
    for d_ in dl: GT.append(pulse(REF, w[0] * (1 - d_), 2 * w[1], 3 * REF['T'], tmax_rec=20000)[1] / REF['T']); L.append('| %g | %.2f |' % (d_, GT[-1]))
    GT = np.array(GT); g = np.isfinite(GT) & (dl <= 0.03); sl = float(np.polyfit(np.log(dl[g]), np.log(GT[g]), 1)[0]) if g.sum() >= 3 else np.nan; v4 = abs(sl + 0.5) <= 0.1
    L.append('\n**П4:** наклон по точкам delta ≤ 0,03: %.2f. Вывод: %s.' % (sl, 'закон призрака подтверждён' if v4 else 'закон призрака не подтверждён')); R['P4'] = dict(slope=sl, verdict=bool(v4))
    L += ['\n## Диапазоны выборки\n', '| параметр | от | до | шкала |\n|---|---|---|---|'] + ['| %s | %g | %g | %s |' % (n, RANGES[n][0], RANGES[n][1], 'логарифмическая' if RANGES[n][2] == 'log' else 'линейная') for n in NAMES]
    text = '\n'.join(L) + '\n'; open(os.path.join('core_out', 'report.md'), 'w', encoding='utf-8').write(text); json.dump(R, open(os.path.join('core_out', 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print('\n' + text)

    # ---------- Рисунки ----------
    stage('рисунки'); import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import matplotlib.ticker
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'font.size': 9})
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5)); a = ax[0][0]; nu, x, c = curve(REF); nuf = curve(REF, frozen=True)[0]; tr, det = jac(REF, UG, x, c); st = (det > 0) & (tr < 0)
    a.loglog(nuf, UG, color=COL[0], lw=2, label='быстрый слой один'); a.loglog(np.where(st, nu, np.nan), UG, color=COL[1], lw=2, label='ядро, устойчивые'); a.loglog(np.where(~st, nu, np.nan), UG, color=COL[1], lw=1, ls=':', label='ядро, неустойчивые')
    a.set_xlim(1e-3, 10); a.set_ylim(1e-3, 1e3); a.set_xlabel('нагрузка nu'); a.set_ylabel('перекись u'); a.set_title('Опорный набор: стационарные состояния'); a.legend(frameon=False)
    a = ax[0][1]; nub = float(np.sqrt(w[0] * w[1])); X, Cc = np.meshgrid(np.linspace(0.002, 1, 300), np.linspace(0.02, 1.25, 300)); Uu = U_of(X, Cc, nub, REF['e0'], REF['gam'])
    F1 = Cc ** REF['m'] * (1 - X) - REF['mu0'] * X * Uu ** 2 / (1 + Uu); F2 = REF['N0'] / (1 + Uu / REF['KN']) + REF['beta'] * Uu / (REF['KB'] + Uu) - Cc
    a.contour(X, Cc, F1, levels=[0], colors=[COL[0]], linewidths=2); a.contour(X, Cc, F2, levels=[0], colors=[COL[1]], linewidths=2); a.plot([], [], color=COL[0], lw=2, label="x' = 0"); a.plot([], [], color=COL[1], lw=2, label="c' = 0")
    for y0 in ([0.98, 1.2], [0.5, 1.2], [0.3, 0.9], [0.15, 0.6], [0.6, 0.3], [0.05, 1.0], [0.9, 0.5], [0.02, 0.1]): s = integ(REF, nub, y0, 30 * REF['T']); a.plot(s.y[0], s.y[1], color='#52514e', lw=0.7); a.plot(s.y[0][-1], s.y[1][-1], 'o', color='#0b0b0b', ms=5)
    a.set_xlabel('активный фермент x'); a.set_ylabel('ёмкость защиты c'); a.set_title('Фазовая плоскость при нагрузке в середине окна (nu = %.3f)' % nub); a.legend(frameon=False)
    a = ax[0][2]
    for i, (ex_, tl, trr, sol) in enumerate([d for d in DUR if d[0] in (0.0, 0.3, 3.0)]):
        t0 = 0.0
        for s in sol:
            if s is None: continue
            a.plot((t0 + s.t) / REF['T'], s.y[0], color=COL[i], lw=1.8, label=('импульс после потери ещё %g T' % ex_) if s is sol[0] else None); t0 += s.t[-1]
    a.set_xscale('symlog', linthresh=0.05); a.set_xlabel('время, в единицах T'); a.set_ylabel('активный фермент x'); a.set_title('Потеря и возврат при разной длительности импульса'); a.legend(frameon=False)
    a = ax[1][0]; a.loglog(dl, GT, 'o-', color=COL[0], ms=7); a.loglog(dl, GT[-1] * (dl / dl[-1]) ** -0.5, color='#52514e', lw=1, ls='--', label='наклон −1/2'); a.set_xlabel('delta — расстояние до нижней границы окна'); a.set_ylabel('время возврата, в T'); a.set_title('Замедление у границы окна'); a.legend(frameon=False)
    a = ax[1][1]; a.hist(np.log10(ratio), bins=30, color=COL[0], edgecolor='#fcfcfb'); a.axvline(np.log10(30), color='#52514e', ls='--', lw=1); a.set_xlabel('lg (время возврата / время потери)'); a.set_ylabel('число наборов'); a.set_title('Несимметрия по выборке; пунктир — порог правила П3')
    a = ax[1][2]
    if ex is not None: p_, nu_, tt, yy = ex; a.plot((tt - tt[0]) / p_['T'], yy[0], color=COL[0], lw=1.6, label='активный фермент x'); a.plot((tt - tt[0]) / p_['T'], yy[1], color=COL[1], lw=1.6, label='ёмкость защиты c'); a.set_xlim(0, 25); a.set_xlabel('время, в единицах T'); a.set_title('Пример колебаний (beta = %.2f, KB = %.2g, нагрузка %.3g)' % (p_['beta'], p_['KB'], nu_)); a.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    else: a.text(0.5, 0.5, 'колебаний в выборке не найдено', ha='center', transform=a.transAxes)
    fig.suptitle('Ядро модели: опорный набор и динамика', fontsize=11); fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join('core_out', 'fig1_reference.png'), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(3, 3, figsize=(15, 11))
    for a, n in zip(ax.ravel(), NAMES):
        fr, mid, _ = imp[n]
        for i in range(4): a.plot(mid, fr[i], color=COL[i], marker=MRK[i], ms=6, lw=1.8, label=CLS[i])
        if RANGES[n][2] == 'log': a.set_xscale('log'); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_ylim(0, 1); a.set_xlabel(n); a.set_ylabel('доля наборов'); a.set_title('вдоль параметра %s' % n)
    ax[0][0].legend(frameon=False, fontsize=8); fig.suptitle('Одномерные проекции: доля каждого класса поведения вдоль каждого из девяти параметров', fontsize=11); fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join('core_out', 'fig2_projections_1d.png'), dpi=110); plt.close(fig)
    from matplotlib.colors import LinearSegmentedColormap
    top = order[:4]; pairs = [(top[i], top[j]) for i in range(4) for j in range(i + 1, 4)]
    def edges(n): lo_, hi_, kd = RANGES[n]; return np.logspace(np.log10(lo_), np.log10(hi_), 15) if kd == 'log' else np.linspace(lo_, hi_, 15)
    for fn, mask, col, ttl in (('fig3_projections_2d_two_states.png', (cls == 1) | (cls == 3), (COL[0], '#0d3566'), 'доля наборов с двумя устойчивыми состояниями'), ('fig4_projections_2d_oscillations.png', cls >= 2, (COL[2], '#07503a'), 'доля наборов с колебаниями')):
        cm = LinearSegmentedColormap.from_list('c', ['#f3f3f1', col[0], col[1]], N=256); fig, ax = plt.subplots(2, 3, figsize=(16, 9.5)); vmax = None
        maps = []
        for (p1, p2) in pairs:
            e1 = edges(p1); e2 = edges(p2); H = np.histogram2d(S[p1][:, 0], S[p2][:, 0], bins=[e1, e2])[0]; Hm = np.histogram2d(S[p1][mask, 0], S[p2][mask, 0], bins=[e1, e2])[0]; maps.append((e1, e2, np.where(H > 0, Hm / np.maximum(H, 1), np.nan)))
        vmax = max(0.05, float(np.nanmax([np.nanmax(m_[2]) for m_ in maps])))
        for a, (p1, p2), (e1, e2, F) in zip(ax.ravel(), pairs, maps):
            im = a.pcolormesh(e1, e2, F.T, cmap=cm, vmin=0, vmax=vmax, edgecolors='#fcfcfb', linewidth=0.4); a.grid(False)
            if RANGES[p1][2] == 'log': a.set_xscale('log')
            if RANGES[p2][2] == 'log': a.set_yscale('log')
            a.set_xlabel(p1); a.set_ylabel(p2); fig.colorbar(im, ax=a, label='доля')
        fig.suptitle('Двумерные проекции: %s; оси — четыре самых влиятельных параметра (%s), остальные пять случайны' % (ttl, ', '.join(top)), fontsize=11); fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join('core_out', fn), dpi=110); plt.close(fig)
    sub = rng.choice(K, min(K, 6000), replace=False); fig, ax = plt.subplots(1, 2, figsize=(16, 7.2))
    for a, (X_, Y_, xl, yl, tl) in zip(ax, ((PR[:, 0], PR[:, 1], 'ось 1', 'ось 2', 'Проекция на две оси, лучше всего разделяющие классы'), (marg, ind, 'запас быстрого слоя M = lg[Phi(mu0/N0^m) / (e0·N0^gam)]', 'относительная индукция beta/N0', 'Проекция на осмысленные сочетания параметров'))):
        for i in (1, 0, 3, 2): mk = sub[cls[sub] == i]; a.scatter(X_[mk], Y_[mk], s=9 if i < 2 else 20, color=COL[i], marker=MRK[i], alpha=0.45 if i < 2 else 0.9, linewidths=0, label=CLS[i])
        a.set_xlabel(xl); a.set_ylabel(yl); a.set_title(tl)
    ax[1].axvline(0, color='#52514e', ls='--', lw=1); h_, l_ = ax[0].get_legend_handles_labels(); o_ = [l_.index(c_) for c_ in CLS if c_ in l_]; fig.legend([h_[i] for i in o_], [l_[i] for i in o_], loc='lower center', ncol=4, frameon=False, markerscale=2.2)
    fig.suptitle('Девять параметров на плоскости: каждый значок — один случайный набор', fontsize=11); fig.tight_layout(rect=(0, 0.06, 1, 0.96)); fig.savefig(os.path.join('core_out', 'fig5_projection_axes.png'), dpi=110); plt.close(fig)
    stage('готово: папка core_out')

if __name__ == '__main__':
    main()
