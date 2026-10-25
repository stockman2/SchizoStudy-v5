# -*- coding: utf-8 -*-
"""
daily_core.py — суточный член в ядре: подводит ли бодрствование клетку к границе, на какой фазе происходит срыв
                и обрывает ли эпизод ближайший сон.

ЗАЧЕМ
  Ядро даёт два устойчивых состояния («защита работает» и «защита выключена») и окно нагрузок, где они сосуществуют.
  Рабочая гипотеза: за часы бодрствования нагрузка медленно растёт и подводит клетку к верхней границе окна,
  а случайный толчок довершает переход; сон снижает нагрузку и может вернуть клетку обратно.
  Проверяется правило Ф2 (ниже). Новых переменных не вводится: добавлены только периодический член в нагрузке и шум.

МОДЕЛЬ — ядро без изменений (функции curve, jac, U_of, rhs, window, sample, classify перенесены из core_model.py дословно)
    nu = eps*u + x*u/(1 + u),  eps = e0*c^gam;   N = N0/(1 + u/KN)
    x' = c^m*(1 - x) - mu0*x*u^2/(1 + u);        c' = [N + beta*u/(KB + u) - c]/T

СУТОЧНЫЙ ЧЛЕН (добавлено здесь)
    nu(t) = nu_mean * (1 + A*S(t)) / (1 + A*<S>) * exp(eta(t))
  S(t) — волна «процесса S» из литературы о сне: в бодрствовании растёт экспоненциально с постоянной времени 18,2 ч,
      во сне спадает с постоянной 4,2 ч (Daan, Beersma, Borbely, 1984, человек; подтверждено повторно в 2025 г.).
      Сутки = 16 ч бодрствования + 8 ч сна. Волна приведена к размаху от 0 (подъём, начало бодрствования) до 1
      (конец бодрствования); деление на (1 + A*<S>) держит СРЕДНЮЮ нагрузку постоянной при любом A.
  A — относительный размах нагрузки за сутки. Основное значение 0,2; разброс 0,1–0,4.
      Откуда: захват глюкозы корой мыши во сне ниже на 16 %; потребление кислорода у человека при засыпании ниже на 21 %;
      метка окисления ДНК в PV-клетках крысы после 6 ч нарушенного сна выше на 25 %. Три линии сходятся на отношении ~1,2.
      Оговорка: переокисленный пероксиредоксин — главная переменная модели — по времени суток в коре никем не измерен,
      поэтому A опирается только на косвенные величины.
  Длительность суток в единицах модели: P = r*T, r из [2, 4]. Основание: единица времени — время починки фермента,
      T — время жизни защитных белков в этих единицах, а 24 ч / (время жизни 6–12 ч) = 2–4.
  eta(t) — шум: процесс Орнштейна–Уленбека на логарифме нагрузки, среднеквадратичное отклонение sigma, время корреляции
      tauc = 0,3*T. Медленный шум выбран потому, что в прошлом расчёте быстрый шум правдоподобной силы оказался слишком слаб.

ПРАВИЛО, ЗАФИКСИРОВАННОЕ ДО СЧЁТА
  Ф2. В наборах, у которых средняя нагрузка стоит внутри окна около верхней границы:
    (а) ФАЗА. Срывы приходятся на бодрствование, а не распределены равномерно по суткам:
        медианная доля срывов во второй половине бодрствования (фаза от 1/3 до 2/3 суток) выше 0,5.
        Ожидание до счёта: около 0,7. При равномерном распределении было бы 0,33.
    (б) ПОРОГ ПО РАЗМАХУ. Существует амплитуда A*, ниже которой частота срывов при той же силе шума падает
        больше чем на порядок по сравнению с A = 0,2.
    (в) СОН КАК ВОЗВРАТ. Доля наборов, где начавшийся эпизод обрывается ближайшим сном, мала: меньше 0,3.
        Считается двумя мерами: строгой (суточный минимум нагрузки опускается ниже нижней границы окна — тогда
        возврат неизбежен) и наблюдаемой (доля эпизодов, закончившихся в пределах одних суток).
  Эпизод: повреждённая доля d = 1 - x выше середины между двумя устойчивыми ветвями, взятой в середине окна.
  Часть (в) — самая рискованная: клиническое представление говорит, что эпизод обрывается сном, и отрицательный
      результат здесь будет означать, что одного суточного члена мало либо что окно должно быть уже.

ЧИСЛЕННАЯ СХЕМА
  Шаг — экспоненциальный Эйлер: на шаге нагрузка и перекись заморожены, а x, c и eta подвигаются точными формулами
  для линейного уравнения. Такая схема устойчива при любом шаге, что важно: скорость x доходит до десятков.
  Шаг свой у каждого набора, dt = P/M, так что все наборы всегда находятся на одном и том же числе суток.

Запуск:  python daily_core.py            обычный вариант, около 1–2 минут
         python daily_core.py --full     втрое больше наборов и вдвое дольше, около 3–5 минут
         python daily_core.py --smoke    несколько наборов, только проверить, что скрипт доходит до конца
Нужны numpy, scipy, matplotlib. Результат: папка daily_out/ — report.md (в конце раздел для машинного чтения),
results.json, fig1…fig4.
"""
import os, sys, json, time, math
import numpy as np
from scipy.integrate import solve_ivp

T0 = time.time(); FULL = '--full' in sys.argv; SMOKE = '--smoke' in sys.argv; OUT = 'daily_out'
def stage(m): print('[%6.1f с] %s' % (time.time() - T0, m), flush=True)
UG = np.logspace(-4, 4, 641)
NAMES = ['e0', 'mu0', 'm', 'gam', 'KN', 'N0', 'beta', 'KB', 'T']
# T сужено против core_model.py (там 10–300): при большом T сутки длинны, и шаг счёта пришлось бы дробить без пользы.
RANGES = dict(e0=(0.005, 0.1, 'log'), mu0=(0.01, 10, 'log'), m=(1, 2, 'lin'), gam=(0, 1, 'lin'), KN=(0.1, 100, 'log'), N0=(0.3, 1, 'lin'), beta=(0, 1, 'lin'), KB=(1, 100, 'log'), T=(20, 80, 'log'))
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']; INK = '#52514e'
TAU_R, TAU_D, FW = 18.2 / 24., 4.2 / 24., 2. / 3.          # постоянные времени роста и спада в долях суток; доля бодрствования

# ---------- Ядро: перенесено из core_model.py дословно ----------
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
    """Перекись из алгебраического баланса: быстрый слой."""
    eps = e0 * c ** gam; b = eps + x - nu; s = np.sqrt(b * b + 4 * eps * nu); return np.where(b > 0, 2 * nu / (b + s), (s - b) / (2 * eps))

def rhs(t, y, p, nu, a=1.0):
    x, c = y[0], max(y[1], 1e-9); eps = p['e0'] * c ** p['gam']; b = eps + x - nu; s = math.sqrt(b * b + 4 * eps * nu); u = 2 * nu / (b + s) if b > 0 else (s - b) / (2 * eps)
    return [c ** p['m'] * (1 - x) - p['mu0'] * x * u * u / (1 + u), (p['N0'] * a / (1 + u / p['KN']) + p['beta'] * u / (p['KB'] + u) - c) / p['T']]

def window(nu):
    """Окно двух состояний по строке nu(u): (нижняя граница, верхняя) или None."""
    i = np.where(np.diff(nu) < 0)[0]
    if not len(i): return None
    return float(nu[i[0]:].min()), float(nu[i[0]])

def sample(K, rng):
    S = {}
    for k, (lo, hi, kind) in RANGES.items(): S[k] = (10 ** rng.uniform(np.log10(lo), np.log10(hi), K) if kind == 'log' else rng.uniform(lo, hi, K))[:, None]
    return S

# ---------- Суточная волна ----------
def wave(M):
    """Волна процесса S на сетке из M точек за сутки, приведённая к размаху 0…1. Возвращает (массив, среднее)."""
    W, Z = FW, 1 - FW; E = math.exp(-Z / TAU_D - W / TAU_R)
    s0 = math.exp(-Z / TAU_D) * (1 - math.exp(-W / TAU_R)) / (1 - E)            # уровень в начале бодрствования
    s1 = 1 - (1 - s0) * math.exp(-W / TAU_R)                                    # уровень в конце бодрствования
    ph = np.arange(M) / M; s = np.where(ph < W, 1 - (1 - s0) * np.exp(-ph / TAU_R), s1 * np.exp(-(ph - W) / TAU_D))
    S = (s - s0) / (s1 - s0); return S, float(S.mean()), s0, s1

# ---------- Устойчивые состояния при заданной нагрузке ----------
def states_at(p, nuv):
    """Устойчивые значения x при нагрузке nuv: пересечения стационарной кривой с проверкой следа и определителя."""
    nuc, xc, cc = curve(p); tr, dt_ = jac(p, UG, xc, cc); a0 = nuc[:-1] - nuv; a1 = nuc[1:] - nuv; cr = np.where(a0 * a1 < 0)[0]; out = []
    for i in cr:
        w = a0[i] / (a0[i] - a1[i])
        if (dt_[i] + w * (dt_[i + 1] - dt_[i])) > 0 and (tr[i] + w * (tr[i + 1] - tr[i])) < 0: out.append(float(xc[i] + w * (xc[i + 1] - xc[i])))
    return sorted(out)

# ---------- Счёт ----------
def simulate(P, nu_mean, thr, A, sigma, tauc_rel, ndays, M, Sg, sm, rng, warm=6, nbins=24):
    """Траектории всех наборов сразу, шаг — экспоненциальный Эйлер. Возвращает статистику по каждому набору.
    У каждого набора свой шаг dt = P_сут/M, поэтому после одинакового числа шагов все стоят на одном числе суток."""
    K = len(nu_mean); dt = P['r'] * P['T'] / M; tauc = tauc_rel * P['T']; nu0 = nu_mean / (1 + A * sm)
    ea = np.exp(-dt / tauc); sd = sigma * np.sqrt(1 - ea * ea); ec = np.exp(-dt / P['T'])
    x = np.ones(K); c = P['N0'] * 1.0; eta = np.zeros(K)
    dam = np.zeros(K, bool); nd = np.zeros(K, int); nup = np.zeros(K, int); bins = np.zeros((K, nbins), int)
    tdam = np.zeros(K); epst = np.zeros(K, int); epsum = np.zeros(K); epn = np.zeros(K, int); epshort = np.zeros(K, int); epsleep = np.zeros(K, int)
    nst = int((warm + ndays) * M); w0 = int(warm * M)
    for i in range(nst):
        S = Sg[i % M]; nu = nu0 * (1 + A * S) * np.exp(eta)
        eps = P['e0'] * c ** P['gam']; b = eps + x - nu; s = np.sqrt(b * b + 4 * eps * nu); u = np.where(b > 0, 2 * nu / (b + s), (s - b) / (2 * np.maximum(eps, 1e-30)))
        aa = c ** P['m']; bb = aa + P['mu0'] * u * u / (1 + u); xinf = aa / bb; x = xinf + (x - xinf) * np.exp(-bb * dt)
        cinf = P['N0'] / (1 + u / P['KN']) + P['beta'] * u / (P['KB'] + u); c = cinf + (c - cinf) * ec
        eta = eta * ea + sd * rng.standard_normal(K)
        nd_ = x < thr; down = nd_ & ~dam; up = (~nd_) & dam
        if down.any(): epst[down] = i                                           # начало эпизода помечается всегда, и на разгоне тоже
        if i >= w0:
            j = int((i % M) * nbins / M)
            if down.any(): nd[down] += 1; bins[down, j] += 1
            if up.any():
                nup[up] += 1; dur = (i - epst[up]) / float(M); epsum[up] += dur; epn[up] += 1
                epshort[up] += (dur < 1.0); epsleep[up] += (1 if (i % M) >= FW * M else 0)
            tdam += nd_ * dt
        dam = nd_
    return dict(nd=nd, nup=nup, bins=bins, tdam=tdam / (ndays * P['r'] * P['T']), epsum=epsum, epn=epn, epshort=epshort, epsleep=epsleep, nday=nd / float(ndays))

# ---------- Основной расчёт ----------
def main():
    K, NSET, ND, M = (2000, 12, 8, 800) if SMOKE else ((120000, 600, 80, 10000) if FULL else (40000, 250, 40, 6000))
    A0, SIG, TAUC, QPOS = 0.2, 0.08, 0.3, 0.75                                  # основной размах, шум, время корреляции, место средней нагрузки в окне
    ASCAN = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
    os.makedirs(OUT, exist_ok=True); rng = np.random.default_rng(71); R = {}
    L = ['# Суточный член в ядре: фаза срыва и возврат через сон\n', 'Режим: %s. Выборка %d наборов; просчитывается до %d наборов с двумя состояниями; %d суток на набор; %d шагов в сутки.\n' % ('пробный' if SMOKE else ('полный' if FULL else 'обычный'), K, NSET, ND, M)]
    L += ['Волна: рост в бодрствовании с постоянной 18,2 ч, спад во сне с постоянной 4,2 ч, сутки = 16 ч + 8 ч; сутки в единицах модели P = r*T, r из [2, 4].',
          'Средняя нагрузка держится постоянной при любом размахе A; место средней нагрузки в окне (в логарифме) q = %.2f; шум: сигма = %.2f, время корреляции %.1f*T.\n' % (QPOS, SIG, TAUC)]

    # ---------- 0. Проверки ----------
    stage('проверки'); Mw = 4000; Sg, sm, s0, s1 = wave(Mw)
    ph = np.arange(Mw) / Mw; iw = int(FW * Mw)
    # Постоянные времени снимаются обратно из построенной волны по её собственным асимптотам:
    # в бодрствовании волна стремится к (1 - s0)/(s1 - s0), во сне — к -s0/(s1 - s0); отклонение от асимптоты — чистая экспонента.
    inf_w = (1 - s0) / (s1 - s0); inf_s = -s0 / (s1 - s0)
    a_r = np.polyfit(ph[1:iw], np.log(np.maximum(inf_w - Sg[1:iw], 1e-12)), 1)[0]
    a_d = np.polyfit(ph[iw + 1:] - ph[iw], np.log(np.maximum(Sg[iw + 1:] - inf_s, 1e-12)), 1)[0]
    L += ['## 0. Проверки\n', '- постоянные времени, снятые обратно из построенной волны: рост %.1f ч (заложено 18,2), спад %.1f ч (заложено 4,2)' % (-24 / a_r, -24 / a_d),
          '- волна: наименьшее %.3f в начале бодрствования, наибольшее %.3f в конце бодрствования, среднее %.3f' % (Sg.min(), Sg.max(), sm)]
    # сверка шага с точным решателем на одном наборе без шума
    Sg_m, sm_m = wave(M)[:2]; pk = {'e0': 0.02, 'mu0': 0.3, 'm': 2.0, 'gam': 0.5, 'KN': 3.0, 'N0': 0.9, 'beta': 0.1, 'KB': 10.0, 'T': 50.0}
    wk = window(curve(pk)[0]); nuM = math.sqrt(wk[0] * wk[1]) if wk else 0.4; st = states_at(pk, nuM); Pd = 3 * pk['T']
    Pk = {n: np.array([pk[n]]) for n in NAMES}; Pk['r'] = np.array([3.0])
    rr = simulate(Pk, np.array([nuM]), np.array([-1.0]), A0, 0.0, TAUC, 3, M, Sg_m, sm_m, np.random.default_rng(1), warm=0)
    def nut(t): return nuM / (1 + A0 * sm_m) * (1 + A0 * Sg_m[int((t / Pd % 1.0) * M)])
    sol = solve_ivp(lambda t, y: rhs(t, y, pk, nut(t)), [0, 3 * Pd], [1.0, pk['N0']], method='LSODA', rtol=1e-9, atol=1e-12, max_step=Pd / 200.)
    # повторяем шаг вручную, чтобы сравнить конечную точку
    xx, cc_, et = 1.0, pk['N0'], 0.0; dtq = Pd / M
    for i in range(3 * M):
        nv = nuM / (1 + A0 * sm_m) * (1 + A0 * Sg_m[i % M]); e_ = pk['e0'] * cc_ ** pk['gam']; b_ = e_ + xx - nv; s_ = math.sqrt(b_ * b_ + 4 * e_ * nv); uu = 2 * nv / (b_ + s_) if b_ > 0 else (s_ - b_) / (2 * e_)
        a_ = cc_ ** pk['m']; bq = a_ + pk['mu0'] * uu * uu / (1 + uu); xi = a_ / bq; xx = xi + (xx - xi) * math.exp(-bq * dtq)
        ci = pk['N0'] / (1 + uu / pk['KN']) + pk['beta'] * uu / (pk['KB'] + uu); cc_ = ci + (cc_ - ci) * math.exp(-dtq / pk['T'])
    dx = abs(xx - sol.y[0, -1]); dc = abs(cc_ - sol.y[1, -1])
    L += ['- шаг против точного решателя (трое суток, без шума): расхождение по x %.2e, по c %.2e' % (dx, dc),
          '- проверочный набор: окно от %.3f до %.3f, устойчивых состояний в его середине: %d (должно быть 2)' % (wk[0], wk[1], len(st))]
    R['checks'] = dict(tau_rise_h=float(-24 / a_r), tau_decay_h=float(-24 / a_d), S_mean=sm, dx_vs_solver=float(dx), dc_vs_solver=float(dc))

    # ---------- 1. Отбор наборов ----------
    stage('выборка %d наборов, отбор наборов с двумя состояниями' % K); S = sample(K, rng); idx = []
    for k in range(K):
        p = {n: float(S[n][k, 0]) for n in NAMES}; w = window(curve(p)[0])
        if w and w[1] / w[0] > 1.05: idx.append((k, w[0], w[1]))
        if len(idx) >= NSET * 3: break
    sel = []
    for k, lo, hi in idx:
        p = {n: float(S[n][k, 0]) for n in NAMES}; num = lo * (hi / lo) ** QPOS; st = states_at(p, math.sqrt(lo * hi))
        if len(st) < 2: continue
        sel.append(dict(k=k, lo=lo, hi=hi, num=num, thr=0.5 * (st[0] + st[-1]), r=float(rng.uniform(2, 4))))
        if len(sel) >= NSET: break
    nsel = len(sel)
    L += ['\n## 1. Наборы\n', '- просмотрено наборов: %d; с окном двух состояний: %d; взято в расчёт: %d' % (min(K, idx[-1][0] + 1 if idx else K), len(idx), nsel)]
    if nsel < 3:
        L.append('\nНаборов слишком мало, правило проверить нельзя.'); open(os.path.join(OUT, 'report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n'); print('\n'.join(L)); return
    P = {n: np.array([float(S[n][d['k'], 0]) for d in sel]) for n in NAMES}; P['r'] = np.array([d['r'] for d in sel])
    lo = np.array([d['lo'] for d in sel]); hi = np.array([d['hi'] for d in sel]); num = np.array([d['num'] for d in sel]); thr = np.array([d['thr'] for d in sel])
    wr = hi / lo; q = lambda v, f='%.3f': (f + ' (10–90 %%: ' + f + ' – ' + f + ')') % tuple(np.percentile(v, [50, 10, 90]))
    L += ['- отношение границ окна: ' + q(wr, '%.2f'), '- период суток в единицах T: ' + q(P['r'], '%.2f'), '- T: ' + q(P['T'], '%.0f')]

    # ---------- 2. Основной прогон ----------
    stage('основной прогон, A = %.2f' % A0); Sg, sm = wave(M)[:2]
    res = simulate(P, num, thr, A0, SIG, TAUC, ND, M, Sg, sm, rng)
    act = res['nd'] > 0; nact = int(act.sum())
    L += ['\n## 2. Срывы: сколько и когда\n', '- наборов, где за %d суток был хотя бы один срыв: %d из %d (%.2f)' % (ND, nact, nsel, nact / nsel),
          '- срывов в сутки на набор (по наборам со срывами): ' + (q(res['nday'][act], '%.3f') if nact else 'нет'),
          '- доля времени в повреждённом состоянии: ' + (q(res['tdam'][act], '%.3f') if nact else 'нет')]

    # (а) фаза
    nb = res['bins'].shape[1]; iw2a, iw2b = nb // 3, 2 * nb // 3
    sh = np.full(nsel, np.nan)
    if nact: sh[act] = res['bins'][act, iw2a:iw2b].sum(1) / res['nd'][act]
    shw = np.full(nsel, np.nan)
    if nact: shw[act] = res['bins'][act, :iw2b].sum(1) / res['nd'][act]
    med = float(np.nanmedian(sh)) if nact else float('nan'); va = bool(nact and med > 0.5)
    L += ['\n### (а) Фаза срыва\n', '| мера | медиана (10–90 %) | при равномерном распределении |\n|---|---|---|',
          '| доля срывов во второй половине бодрствования | %s | 0,33 |' % (q(sh[act], '%.3f') if nact else '—'),
          '| доля срывов в бодрствовании целиком | %s | 0,67 |' % (q(shw[act], '%.3f') if nact else '—'),
          '\n**Ф2(а):** медиана %.3f (порог 0,5; ожидание до счёта — около 0,7). Вывод: %s.' % (med, 'правило выполнено' if va else 'правило не выполнено')]

    # ---------- 3. (б) Порог по размаху ----------
    stage('перебор размаха A'); sc = []
    for A in ASCAN:
        rr = simulate(P, num, thr, A, SIG, TAUC, max(ND // 2, 4), M, Sg, sm, np.random.default_rng(100 + int(A * 100)))
        sc.append(float(np.mean(rr['nday']))); stage('  A = %.2f: срывов в сутки на набор %.4f' % (A, sc[-1]))
    base = sc[ASCAN.index(A0)]; cand = [ASCAN[i] for i in range(len(ASCAN)) if ASCAN[i] < A0 and sc[i] < base / 10.]; Astar = max(cand) if cand else None
    vb = Astar is not None
    L += ['\n## 3. (б) Порог по суточному размаху\n', '| A | срывов в сутки на набор | к уровню при A = 0,2 |\n|---|---|---|']
    for A, s_ in zip(ASCAN, sc): L.append('| %.2f | %.4f | %s |' % (A, s_, ('%.3f' % (s_ / base)) if base > 0 else '—'))
    L += ['\n**Ф2(б):** %s. Вывод: %s.' % ('наибольшее A, при котором частота срывов ниже уровня A = 0,2 более чем на порядок: %.2f' % Astar if vb else 'падения более чем на порядок ни при каком A из перебора нет', 'правило выполнено' if vb else 'правило не выполнено')]

    # ---------- 4. (в) Сон как возврат ----------
    nu_min = num / (1 + A0 * sm); nu_max = nu_min * (1 + A0)
    struct = nu_min < lo; pk_in = nu_max > hi
    ended = res['epn'] > 0; shortsh = np.full(nsel, np.nan); sleepsh = np.full(nsel, np.nan); dur = np.full(nsel, np.nan)
    if ended.any():
        shortsh[ended] = res['epshort'][ended] / res['epn'][ended]; sleepsh[ended] = res['epsleep'][ended] / res['epn'][ended]; dur[ended] = res['epsum'][ended] / res['epn'][ended]
    s_str = float(struct.mean()); s_obs = float(np.nanmedian(shortsh)) if ended.any() else 0.0
    vc = (s_str < 0.3) and (s_obs < 0.3)
    L += ['\n## 4. (в) Обрывает ли эпизод ближайший сон\n', '| мера | значение | ожидание до счёта |\n|---|---|---|',
          '| строгая: суточный минимум нагрузки ниже нижней границы окна | %.3f | меньше 0,3 |' % s_str,
          '| наблюдаемая: медианная доля эпизодов, закончившихся за сутки | %s | меньше 0,3 |' % (('%.3f' % s_obs) if ended.any() else 'эпизодов с возвратом нет'),
          '| доля наборов, где суточный максимум выше верхней границы окна | %.3f | — |' % float(pk_in.mean()),
          '| доля возвратов, пришедшихся на фазу сна | %s | — |' % (q(sleepsh[ended], '%.3f') if ended.any() else '—'),
          '| средняя длительность эпизода, сутки | %s | — |' % (q(dur[ended], '%.2f') if ended.any() else '—'),
          '\n**Ф2(в):** %s. Вывод: %s.' % ('строгая мера %.3f, наблюдаемая %.3f' % (s_str, s_obs), 'правило выполнено' if vc else 'правило не выполнено')]
    R['F2'] = dict(a=dict(median=med, expect=0.7, thresh=0.5, verdict=va), b=dict(scan=dict(zip(map(str, ASCAN), sc)), A_star=Astar, verdict=vb),
                   c=dict(structural=s_str, observed=s_obs, peak_above_upper=float(pk_in.mean()), verdict=vc))
    R['setup'] = dict(mode='smoke' if SMOKE else ('full' if FULL else 'default'), n_sets=nsel, n_sampled=K, ndays=ND, steps_per_day=M, A=A0, sigma=SIG, tauc_rel=TAUC, q_pos=QPOS, S_mean=sm, T_range=list(RANGES['T'][:2]), r_range=[2, 4])
    R['sets'] = dict(window_ratio_median=float(np.median(wr)), T_median=float(np.median(P['T'])), r_median=float(np.median(P['r'])),
                     share_with_breakdown=nact / nsel, breakdowns_per_day_median=float(np.median(res['nday'][act])) if nact else 0.0,
                     damaged_time_median=float(np.median(res['tdam'][act])) if nact else 0.0)

    # ---------- 5. Связь с параметрами ----------
    from scipy.stats import rankdata
    L += ['\n## 5. С чем связана частота срывов: ранговая корреляция\n', '| величина | корреляция с числом срывов в сутки |\n|---|---|']
    cors = {}
    for qn in NAMES + ['r']:
        cc2 = float(np.corrcoef(rankdata(P[qn]), rankdata(res['nday']))[0, 1]); cors[qn] = cc2; L.append('| %s | %+.2f |' % (qn, cc2))
    for nm, vv in (('запас до верхней границы, log', np.log(hi / nu_max)),):                                 # отношение границ окна здесь не приводится: при фиксированном q оно связано с запасом взаимно однозначно
        cc2 = float(np.corrcoef(rankdata(vv), rankdata(res['nday']))[0, 1]); cors[nm] = cc2; L.append('| %s | %+.2f |' % (nm, cc2))
    R['corr'] = cors

    # ---------- 6. Для машинного чтения ----------
    R['time_s'] = round(time.time() - T0, 1)
    L += ['\n## 6. Для машинного чтения\n', 'Тот же набор чисел, что выше, одним блоком; полная копия — в results.json.\n', '```json', json.dumps(R, ensure_ascii=False, indent=1), '```']
    L.append('\nВремя счёта: %.0f с.' % (time.time() - T0)); text = '\n'.join(L) + '\n'
    open(os.path.join(OUT, 'report.md'), 'w', encoding='utf-8').write(text); json.dump(R, open(os.path.join(OUT, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print('\n' + text)

    # ---------- Рисунки ----------
    stage('рисунки'); import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'font.size': 9})
    j = int(np.argmax(res['nd'])); t3 = np.linspace(0, 3, 3 * M); S3 = np.tile(Sg, 3); nu3 = num[j] / (1 + A0 * sm) * (1 + A0 * S3)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True); a = ax[0]
    a.plot(t3, S3, color=COL[0], lw=2); a.set_ylabel('волна S'); a.set_title('Суточная волна и нагрузка (набор с наибольшим числом срывов)', fontsize=10, loc='left')
    for d_ in range(3): a.axvspan(d_ + FW, d_ + 1, color='#e4e4e1', lw=0, zorder=0)
    a = ax[1]; a.plot(t3, nu3, color=COL[1], lw=2, label='нагрузка'); a.axhline(lo[j], color=INK, ls='--', lw=1, label='нижняя граница окна'); a.axhline(hi[j], color=INK, ls=':', lw=1.2, label='верхняя граница окна')
    for d_ in range(3): a.axvspan(d_ + FW, d_ + 1, color='#e4e4e1', lw=0, zorder=0)
    a.set_xlabel('время, сутки'); a.set_ylabel('нагрузка'); a.legend(frameon=False, fontsize=8, ncol=3); fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig1_wave.png'), dpi=110); plt.close(fig)

    fig, a = plt.subplots(1, 1, figsize=(9, 4.6)); tot = res['bins'][act].sum(0) if nact else np.zeros(nb); cen = (np.arange(nb) + 0.5) / nb
    a.bar(cen, tot / max(tot.sum(), 1), width=1. / nb * 0.9, color=COL[0], edgecolor='#fcfcfb'); a.axvspan(FW, 1, color='#e4e4e1', lw=0, zorder=0, label='сон')
    a.axvspan(1 / 3., 2 / 3., color='#d9d8d4', alpha=0.7, lw=0, zorder=0, label='вторая половина бодрствования'); a.axhline(1. / nb, color=INK, ls='--', lw=1, label='равномерное распределение')
    a.set_xlabel('фаза суток (0 — пробуждение, %.2f — засыпание)' % FW); a.set_ylabel('доля срывов'); a.set_title('Когда происходит срыв', fontsize=10, loc='left'); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2_phase.png'), dpi=110); plt.close(fig)

    fig, a = plt.subplots(1, 1, figsize=(8, 4.6)); a.plot(ASCAN, np.maximum(sc, 1e-6), 'o-', color=COL[0], lw=2)
    a.axhline(max(base, 1e-6) / 10, color=COL[1], ls='--', lw=1.2, label='в десять раз ниже уровня A = 0,2'); a.axvline(A0, color=INK, ls=':', lw=1)
    a.set_yscale('log'); a.set_xlabel('суточный размах нагрузки A'); a.set_ylabel('срывов в сутки на набор'); a.set_title('Порог по размаху', fontsize=10, loc='left'); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3_amplitude.png'), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6)); a = ax[0]
    if ended.any(): a.hist(dur[ended], bins=20, color=COL[0], edgecolor='#fcfcfb'); a.axvline(1.0, color=COL[1], lw=2, label='одни сутки'); a.legend(frameon=False, fontsize=8)
    a.set_xlabel('средняя длительность эпизода, сутки'); a.set_ylabel('число наборов'); a.set_title('Сколько длится эпизод', fontsize=10, loc='left')
    a = ax[1]; a.scatter(np.log(hi / nu_max), res['nday'], s=16, color=COL[0], alpha=0.65, linewidths=0); a.axvline(0, color=COL[1], lw=1.5, label='суточный максимум на верхней границе')
    a.set_xlabel('запас до верхней границы окна, логарифм'); a.set_ylabel('срывов в сутки'); a.set_title('Срывы и запас до границы', fontsize=10, loc='left'); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4_episodes.png'), dpi=110); plt.close(fig)
    stage('готово: папка %s' % OUT)

if __name__ == '__main__':
    main()
