# -*- coding: utf-8 -*-
"""
sleep_core2.py — то же, что sleep_core.py, но с исправленным измерением и с прямой проверкой «сон портится раньше срыва».

ЧТО ИЗМЕНЕНО ПРОТИВ sleep_core.py (модель та же, менялось только измерение)
  1. Перебор по k идёт с ОДНИМ зерном случайных чисел: сравнение парное. Прежде у каждого k было своё зерно,
     и разброс от зерна (около 13 % по длительности эпизода) был сравним с измеряемым эффектом.
  2. Отдельно считается погрешность: пара k = 0 и k = k_база прогоняется с несколькими зёрнами.
  3. Эпизоды, не успевшие кончиться за время счёта, больше не теряются: печатается их доля и мера длительности,
     учитывающая незакрытые эпизоды. Вердикт по-прежнему выносится по прежней мере, чтобы сравнение с прошлым прогоном было честным.
  4. Добавлена средняя длина бодрствования (прежде была только наибольшая). Модель от этого не меняется: это ещё одна мера, а не новый член уравнений.
  5. В переборе по положению нагрузки медианы берутся по наборам, где срывы были, а не по всем.
  6. В раздел 0 выводится, при каком k порог засыпания упирается в потолок: полка на графиках создана потолком, а не моделью.
  7. НОВОЕ ИЗМЕРЕНИЕ: длина бодрствования в зависимости от повреждения во время этого бодрствования, и длина
     бодрствования в зависимости от того, за сколько суток до срыва оно случилось.

ЗАЧЕМ
  В daily_core.py сон был задан извне: 16 часов бодрствования, 8 часов сна, что бы с клеткой ни происходило.
  Это дыра: в действительности повреждённое состояние само мешает заснуть, и получается круг
  «клетка повреждена -> заснуть трудно -> нагрузка не падает -> эпизод держится».
  Здесь расписание заменено настоящим двухпроцессным переключателем сна, и добавлена одна связь: повреждение
  поднимает порог засыпания. Длительность суток перестаёт быть заданной и становится результатом счёта.

МОДЕЛЬ — ядро без изменений (curve, jac, U_of, rhs, window, sample, states_at перенесены из daily_core.py дословно)
    nu = eps*u + x*u/(1 + u),  eps = e0*c^gam;   N = N0/(1 + u/KN)
    x' = c^m*(1 - x) - mu0*x*u^2/(1 + u);        c' = [N + beta*u/(KB + u) - c]/T

СОН (переделано здесь)
  Давление сна S живёт само по себе: в бодрствовании растёт к 1 с постоянной времени 18,2 ч,
      во сне спадает к 0 с постоянной 4,2 ч (Daan, Beersma, Borbely, 1984, человек).
  Переключение по порогам той же работы: бодрствование кончается, когда S доходит до верхнего порога H;
      сон кончается, когда S опускается до нижнего порога L. H = 0,67, L = 0,17, оба с суточной модуляцией
      амплитуды 0,12 от внутренних часов, идущих со своим периодом независимо от состояния сна.
  НОВАЯ СВЯЗЬ (единственная): H_эфф = H + k*d, где d — повреждённая доля, приведённая к размаху между
      двумя устойчивыми ветвями. Чем сильнее повреждение, тем выше порог засыпания и тем дольше не наступает сон.
      Параметр один: k. При k = 0 всё сводится к прежнему поведению, поэтому сравнение честное.
      Потолок H_эфф = 0,98: иначе при сильном повреждении сон не наступал бы никогда, а это заведомо неверно.
  Сам сон считается одинаково восстановительным у здоровых и у больных: постоянная спада 4,2 ч не меняется.
      Это допущение, принятое сознательно; обратный вариант (сон менее восстановителен) здесь не рассматривается.

НАГРУЗКА
    nu(t) = nu_mean * (1 + A*S(t)) / (1 + A*S0_ср) * exp(eta(t)),  A = 0,2
  S0_ср — среднее давление сна за свободный цикл при k = 0, считается в разделе 0. При k > 0 бодрствование
  удлиняется, среднее S растёт, и средняя нагрузка поднимается сама — это и есть смысл переделки, а не ошибка.
  eta — шум Орнштейна–Уленбека на логарифме нагрузки: сигма 0,08, время корреляции 0,3*T.
  Единица времени: час = r*T/24, r из [2, 4]; то есть внутренние часы имеют период r*T, как и раньше.

ПРАВИЛО, ЗАФИКСИРОВАННОЕ ДО СЧЁТА
  Ф3. При переборе силы связи k = 0; 0,1; 0,2; 0,3:
    (а) ДЛИТЕЛЬНОСТЬ. Средняя длительность эпизода при k = 0,2 не меньше чем в 1,5 раза больше, чем при k = 0.
    (б) ЗАСТРЕВАНИЕ. Появляется режим «бодрствует и повреждён»: доля наборов, у которых есть непрерывное
        бодрствование дольше двух суток, при k > 0 больше нуля и растёт с k.
    (в) ВОЗВРАТ. Доля эпизодов, кончающихся за одни сутки, падает ниже 0,50, измеренных при расписании.
        Ожидание до счёта: 0,2–0,3.
  Без вердикта печатаются: длительность цикла сна и бодрствования как результат, доля времени бодрствования,
    фаза срыва, перебор по положению средней нагрузки в окне.

  Ф4 (новое правило, записано до счёта). Повреждение, ещё НЕ дошедшее до порога срыва, уже удлиняет бодрствование,
    и это удлинение опережает срыв на срок от одних до восьми суток.
    (а) Средняя длина бодрствования растёт с повреждением d уже в области d < 0,5, то есть до порога срыва:
        в последней подпороговой ячейке (d от 0,4 до 0,5) она не меньше чем на 10 % больше, чем в первой (d от 0 до 0,1).
        Если в подпороговых ячейках окажется меньше 20 бодрствований, правило считается непроверяемым, а не выполненным:
        в системе с двумя состояниями промежуточное повреждение может оказаться слишком редким.
    (б) Среднее опережение: длина бодрствования, усреднённая по суткам до первого срыва, превышает уровень здорового
        состояния (ячейка d от 0 до 0,1) не меньше чем за сутки до срыва, и это опережение не больше восьми суток.
        Уровень здорового состояния, а не средний по набору: средний загрязнён длинными бодрствованиями самого эпизода.
    Повреждение d здесь приведено к размаху между ветвями, поэтому d = 0,5 — ровно порог срыва.
    Основание ожиданий: в литературе нарушение сна предшествует переходу в психоз и предсказывает ухудшение
    на 1–8 суток вперёд (Reeve и соавт. 2015; Meyer и соавт. 2022). Эти работы при построении модели не использовались.
    Оговорка о честности: намёк на (а) был взят из прошлого прогона (длинных бодрствований оказалось больше, чем срывов),
    но при проверке этот довод оказался негодным — он сравнивал долю наборов с длинными бодрствованиями с долей наборов,
    где срыв ПРОИЗОШЁЛ во время записи, тогда как набор, сорвавшийся ещё на разгоне и не вернувшийся, в срывы не попадал.
    Поэтому обе части Ф4 — предсказания, а не проверки согласия: ни одна из двух кривых до сих пор не вычислялась.

ЧИСЛЕННАЯ СХЕМА
  Шаг — экспоненциальный Эйлер: на шаге перекись заморожена, а x, c, S и шум двигаются точными формулами.
  Схема устойчива при любом шаге. Шаг свой у каждого набора: dt = r*T/M, то есть M шагов на период внутренних часов.

Запуск:  python sleep_core2.py           обычный вариант, около 5 минут
         python sleep_core2.py --full    больше наборов, суток и точек перебора, около 12 минут
         python sleep_core2.py --smoke   несколько наборов, только проверить, что скрипт доходит до конца
Нужны numpy, scipy, matplotlib. Результат: папка sleep2_out/ — report.md (в конце раздел для машинного чтения),
results.json, fig1…fig5.
"""
import os, sys, json, time, math
import numpy as np
from scipy.integrate import solve_ivp

T0 = time.time(); FULL = '--full' in sys.argv; SMOKE = '--smoke' in sys.argv; OUT = 'sleep2_out'
def stage(m): print('[%6.1f с] %s' % (time.time() - T0, m), flush=True)
UG = np.logspace(-4, 4, 641)
NAMES = ['e0', 'mu0', 'm', 'gam', 'KN', 'N0', 'beta', 'KB', 'T']
RANGES = dict(e0=(0.005, 0.1, 'log'), mu0=(0.01, 10, 'log'), m=(1, 2, 'lin'), gam=(0, 1, 'lin'), KN=(0.1, 100, 'log'), N0=(0.3, 1, 'lin'), beta=(0, 1, 'lin'), KB=(1, 100, 'log'), T=(20, 80, 'log'))
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']; INK = '#52514e'
TAU_R, TAU_D = 18.2, 4.2                      # постоянные времени роста и спада давления сна, часы
H_M, L_M, C_A, TH0 = 0.67, 0.17, 0.12, 0.17   # пороги засыпания и пробуждения, амплитуда суточной модуляции, её фаза
H_CAP = 0.98                                  # потолок порога засыпания

# ---------- Ядро: перенесено из daily_core.py дословно ----------
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

def states_at(p, nuv):
    """Устойчивые значения x при нагрузке nuv: пересечения стационарной кривой с проверкой следа и определителя."""
    nuc, xc, cc = curve(p); tr, dt_ = jac(p, UG, xc, cc); a0 = nuc[:-1] - nuv; a1 = nuc[1:] - nuv; cr = np.where(a0 * a1 < 0)[0]; out = []
    for i in cr:
        w = a0[i] / (a0[i] - a1[i])
        if (dt_[i] + w * (dt_[i + 1] - dt_[i])) > 0 and (tr[i] + w * (tr[i + 1] - tr[i])) < 0: out.append(float(xc[i] + w * (xc[i + 1] - xc[i])))
    return sorted(out)

# ---------- Свободный цикл сна при k = 0: нужен для нормировки и для проверки ----------
def free_cycle(M=20000, ncyc=40):
    """Прогон одного переключателя сна без клетки: средние длительности бодрствования, сна, периода (часы) и среднее S."""
    dt = 24. / M; S = L_M; awake = True; th = 0.0; t = 0.0
    wk, sl, acc, n = [], [], 0.0, 0; ws = 0.0; ss = 0.0; start = 0.0
    for i in range(int(ncyc * 36 * M / 24)):
        C = math.cos(2 * math.pi * (th - TH0)); H = H_M + C_A * C; L = L_M + C_A * C
        S = 1 + (S - 1) * math.exp(-dt / TAU_R) if awake else S * math.exp(-dt / TAU_D)
        acc += S * dt; n += 1; t += dt; th = (th + dt / 24.) % 1.0
        if awake and S >= H: wk.append(t - start); start = t; awake = False
        elif (not awake) and S <= L: sl.append(t - start); start = t; awake = True
    w = float(np.median(wk[len(wk) // 2:])) if len(wk) > 4 else float('nan'); s_ = float(np.median(sl[len(sl) // 2:])) if len(sl) > 4 else float('nan')
    return w, s_, w + s_, acc / (n * dt)

# ---------- Счёт ----------
NDB = 10          # ячеек по повреждению для кривой «длина бодрствования против повреждения»
NLB = 8           # ячеек по суткам до первого срыва
RING = 8          # сколько последних бодрствований помнить, чтобы потом отнести их к сроку до срыва

def simulate(P, nu_mean, thr, xhi, xlo, kslp, A, sigma, tauc_rel, ndays, M, Sbar, rng, warm=8):
    """Траектории всех наборов сразу. Сон переключается по порогам; повреждение поднимает порог засыпания.
    Длительности в сутках, где сутки = период внутренних часов r*T. Повреждение d приведено к размаху между
    устойчивыми ветвями, так что d = 0,5 — ровно порог срыва."""
    K = len(nu_mean); day = P['r'] * P['T']; dt = day / M; hr = day / 24.
    tauc = tauc_rel * P['T']; nu0 = nu_mean / (1 + A * Sbar)
    ea = np.exp(-dt / tauc); sdn = sigma * np.sqrt(1 - ea * ea); ec = np.exp(-dt / P['T'])
    er = np.exp(-dt / (TAU_R * hr)); ed = np.exp(-dt / (TAU_D * hr))
    x = np.ones(K); c = P['N0'] * 1.0; eta = np.zeros(K); S = np.full(K, L_M); awake = np.ones(K, bool); th = np.zeros(K)
    dam = np.zeros(K, bool); nb = np.zeros(K, int); nb_aw = np.zeros(K, int); hsum = np.zeros(K)
    epst = np.zeros(K, int); epsum = np.zeros(K); epn = np.zeros(K, int); epshort = np.zeros(K, int); everdam = np.zeros(K, bool)
    wkst = np.zeros(K, int); wksum = np.zeros(K); wkn = np.zeros(K, int); wkmax = np.zeros(K); wklong = np.zeros(K, int); wk15 = np.zeros(K, int)
    taw = np.zeros(K); tdam = np.zeros(K); cycn = np.zeros(K, int); cyc1 = np.zeros(K); cyc2 = np.zeros(K)
    bmax = np.zeros(K)                                                                  # наибольшее повреждение за текущее бодрствование
    dbin_s = np.zeros(NDB); dbin_n = np.zeros(NDB)                                      # кривая: длина бодрствования против повреждения
    lb_s = np.zeros(NLB); lb_n = np.zeros(NLB)                                          # кривая: длина бодрствования против срока до первого срыва
    ring_l = np.zeros((K, RING)); ring_t = np.full((K, RING), -1.0); ptr = np.zeros(K, int); first_done = np.zeros(K, bool)
    nmid = [0.0]; nall = [0.0]                                                          # бодрствования с промежуточным повреждением и всего
    nst = int((warm + ndays) * M); w0 = int(warm * M)
    for i in range(nst):
        dn = np.clip((xhi - x) / np.maximum(xhi - xlo, 1e-9), 0, 1)
        bmax = np.maximum(bmax, dn)
        C = np.cos(2 * math.pi * (th - TH0)); H = np.minimum(H_M + C_A * C + kslp * dn, H_CAP); L = L_M + C_A * C
        nu = nu0 * (1 + A * S) * np.exp(eta)
        eps = P['e0'] * c ** P['gam']; b = eps + x - nu; s_ = np.sqrt(b * b + 4 * eps * nu); u = np.where(b > 0, 2 * nu / (b + s_), (s_ - b) / (2 * np.maximum(eps, 1e-30)))
        aa = c ** P['m']; bb = aa + P['mu0'] * u * u / (1 + u); xinf = aa / bb; x = xinf + (x - xinf) * np.exp(-bb * dt)
        cinf = P['N0'] / (1 + u / P['KN']) + P['beta'] * u / (P['KB'] + u); c = cinf + (c - cinf) * ec
        S = np.where(awake, 1 + (S - 1) * er, S * ed); eta = eta * ea + sdn * rng.standard_normal(K); th = (th + dt / day) % 1.0
        fall = awake & (S >= H); wake = (~awake) & (S <= L)
        if fall.any():
            idf = np.where(fall)[0]; d_ = (i - wkst[idf]) / float(M)
            if i >= w0:
                wkn[idf] += 1; wksum[idf] += d_; wkmax[idf] = np.maximum(wkmax[idf], d_); wklong[idf] += (d_ > 2.0); wk15[idf] += (d_ > 1.5)
                jb = np.clip((bmax[idf] * NDB).astype(int), 0, NDB - 1)                 # ячейка по наибольшему повреждению за это бодрствование
                np.add.at(dbin_s, jb, d_); np.add.at(dbin_n, jb, 1.0)
                nmid[0] += float(((bmax[idf] > 0.3) & (bmax[idf] < 0.7)).sum()); nall[0] += len(idf)
                ring_l[idf, ptr[idf]] = d_; ring_t[idf, ptr[idf]] = i / float(M); ptr[idf] = (ptr[idf] + 1) % RING
            bmax[idf] = 0.0
        if wake.any():
            wkst[wake] = i
            if i >= w0:
                cycn[wake] += 1; fst = cycn[wake] == 1
                cyc1[wake] = np.where(fst, i / float(M), cyc1[wake]); cyc2[wake] = i / float(M)
        awake = np.where(fall, False, np.where(wake, True, awake))
        nd_ = x < thr; down = nd_ & ~dam; up = (~nd_) & dam
        if down.any(): epst[down] = i
        if i >= w0:
            if down.any():
                nb[down] += 1; everdam[down] = True; aw = down & awake; nb_aw[aw] += 1; hsum[aw] += (i - wkst[aw]) / float(M) * 24.
                fb = np.where(down & ~first_done)[0]                                     # первый срыв набора: разносим запомненные бодрствования по сроку до него
                if len(fb):
                    tb = i / float(M)
                    for jj in range(RING):
                        lead = tb - ring_t[fb, jj]; ok = (ring_t[fb, jj] >= 0) & (lead > 0) & (lead < NLB)
                        if ok.any():
                            jl = lead[ok].astype(int); np.add.at(lb_s, jl, ring_l[fb[ok], jj]); np.add.at(lb_n, jl, 1.0)
                    first_done[fb] = True
            if up.any():
                dur_ = (i - epst[up]) / float(M); epsum[up] += dur_; epn[up] += 1; epshort[up] += (dur_ < 1.0)
            taw += awake * dt; tdam += nd_ * dt
        dam = nd_
    tot = ndays * day
    per = np.where(cycn > 1, (cyc2 - cyc1) / np.maximum(cycn - 1, 1) * 24., np.nan)
    return dict(nb=nb, nb_aw=nb_aw, hsum=hsum, epn=epn, epsum=epsum, epshort=epshort, everdam=everdam, open_ep=dam,
                wkn=wkn, wksum=wksum, wkmax=wkmax, wklong=wklong, wk15=wk15, taw=taw / tot, tdam=tdam / tot, per=per,
                nday=nb / float(ndays), dbin=(dbin_s, dbin_n), lbin=(lb_s, lb_n), bmid=(nmid[0] / nall[0] if nall[0] else 0.0))

def trace(p, nu_mean, thr, xhi, xlo, kslp, A, sigma, tauc_rel, ndays, M, Sbar, rng):
    """Одна траектория для рисунка: возвращает время в сутках, давление сна, признак бодрствования, нагрузку и повреждение."""
    P = {n: np.array([p[n]]) for n in NAMES}; P['r'] = np.array([p['r']])
    day = p['r'] * p['T']; dt = day / M; hr = day / 24.; tauc = tauc_rel * p['T']; nu0 = nu_mean / (1 + A * Sbar)
    ea = math.exp(-dt / tauc); sdn = sigma * math.sqrt(1 - ea * ea); ec = math.exp(-dt / p['T']); er = math.exp(-dt / (TAU_R * hr)); ed = math.exp(-dt / (TAU_D * hr))
    x, c, eta, S, aw, th = 1.0, p['N0'], 0.0, L_M, True, 0.0; out = []
    for i in range(int(ndays * M)):
        dn = min(max((xhi - x) / max(xhi - xlo, 1e-9), 0), 1); C = math.cos(2 * math.pi * (th - TH0)); H = min(H_M + C_A * C + kslp * dn, H_CAP); L = L_M + C_A * C
        nu = nu0 * (1 + A * S) * math.exp(eta)
        e_ = p['e0'] * c ** p['gam']; b = e_ + x - nu; s_ = math.sqrt(b * b + 4 * e_ * nu); u = 2 * nu / (b + s_) if b > 0 else (s_ - b) / (2 * e_)
        a_ = c ** p['m']; bq = a_ + p['mu0'] * u * u / (1 + u); xi = a_ / bq; x = xi + (x - xi) * math.exp(-bq * dt)
        ci = p['N0'] / (1 + u / p['KN']) + p['beta'] * u / (p['KB'] + u); c = ci + (c - ci) * ec
        S = 1 + (S - 1) * er if aw else S * ed; eta = eta * ea + sdn * rng.standard_normal(); th = (th + dt / day) % 1.0
        if aw and S >= H: aw = False
        elif (not aw) and S <= L: aw = True
        if i % 10 == 0: out.append((i / float(M), S, 1.0 if aw else 0.0, nu, dn))
    return np.array(out)

# ---------- Основной расчёт ----------
def main():
    K, NSET, ND, M = (2000, 12, 8, 800) if SMOKE else ((200000, 600, 60, 8000) if FULL else (40000, 250, 40, 6000))
    KSCAN = [0.0, 0.1, 0.2, 0.4] if SMOKE else ([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] if FULL else [0.0, 0.1, 0.2, 0.3, 0.4])
    QSCAN = [0.75, 0.85] if SMOKE else ([0.75, 0.8, 0.85, 0.9, 0.95] if FULL else [0.75, 0.8, 0.85, 0.9])
    NREP = 2 if SMOKE else (3 if FULL else 3)
    A0, SIG, TAUC, QPOS, KBASE, SEED = 0.2, 0.08, 0.3, 0.75, 0.2, 4242
    os.makedirs(OUT, exist_ok=True); rng = np.random.default_rng(83); R = {}
    L = ['# Сон как следствие: исправленное измерение и проверка «сон портится раньше срыва»\n',
         'Режим: %s. Выборка %d наборов; просчитывается до %d наборов с двумя состояниями; %d суток на набор; %d шагов на период внутренних часов.\n' % ('пробный' if SMOKE else ('полный' if FULL else 'обычный'), K, NSET, ND, M),
         'Модель та же, что в sleep_core.py. Изменено измерение: перебор по k идёт с одним зерном (сравнение парное), пара k = 0 и k = %.1f прогоняется с %d зёрнами, незакрытые эпизоды больше не теряются.\n' % (KBASE, NREP),
         'Сон: пороги H = 0,67, L = 0,17, суточная модуляция 0,12; H_эфф = H + k*d, потолок %.2f. Постоянные времени 18,2 и 4,2 ч. Нагрузка: A = %.2f, шум сигма = %.2f, время корреляции %.1f*T, положение в окне q = %.2f.\n' % (H_CAP, A0, SIG, TAUC, QPOS)]

    # ---------- 0. Проверки ----------
    stage('проверки'); wH, sH, pH, Sbar = free_cycle(); wmax_h = -TAU_R * math.log((1 - H_CAP) / (1 - (L_M - C_A)))
    tofH = lambda H: -TAU_R * math.log(max(1 - H, 1e-12) / (1 - L_M))                    # часы бодрствования при пороге H
    kcap0 = H_CAP - H_M; kcap1 = H_CAP - H_M - C_A                                       # k, при котором порог упирается в потолок: без модуляции и в худшей фазе
    pk = {'e0': 0.02, 'mu0': 0.3, 'm': 2.0, 'gam': 0.5, 'KN': 3.0, 'N0': 0.9, 'beta': 0.1, 'KB': 10.0, 'T': 50.0}
    wk_ = window(curve(pk)[0]); nuM = math.sqrt(wk_[0] * wk_[1]); st = states_at(pk, nuM)
    xx, cc_ = 1.0, pk['N0']; dtq = 3 * pk['T'] / M
    for i in range(M):
        e_ = pk['e0'] * cc_ ** pk['gam']; b_ = e_ + xx - nuM; s_ = math.sqrt(b_ * b_ + 4 * e_ * nuM); uu = 2 * nuM / (b_ + s_) if b_ > 0 else (s_ - b_) / (2 * e_)
        a_ = cc_ ** pk['m']; bq = a_ + pk['mu0'] * uu * uu / (1 + uu); xi = a_ / bq; xx = xi + (xx - xi) * math.exp(-bq * dtq)
        ci = pk['N0'] / (1 + uu / pk['KN']) + pk['beta'] * uu / (pk['KB'] + uu); cc_ = ci + (cc_ - ci) * math.exp(-dtq / pk['T'])
    sol = solve_ivp(lambda t, y: rhs(t, y, pk, nuM), [0, M * dtq], [1.0, pk['N0']], method='LSODA', rtol=1e-10, atol=1e-13)
    dx = abs(xx - sol.y[0, -1]); dc = abs(cc_ - sol.y[1, -1])
    L += ['## 0. Проверки\n', '- свободный цикл сна при k = 0: бодрствование %.1f ч, сон %.1f ч, период %.1f ч (должен выйти около 24 — проверка порогов, а не подгонка)' % (wH, sH, pH),
          '- среднее давление сна за свободный цикл: %.3f (нормировка нагрузки)' % Sbar,
          '- шаг против точного решателя при постоянной нагрузке: расхождение по x %.2e, по c %.2e' % (dx, dc),
          '- проверочный набор: окно от %.3f до %.3f, устойчивых состояний в его середине: %d (должно быть 2)' % (wk_[0], wk_[1], len(st)),
          '\n**Откуда берутся ступеньки на графиках по k.** Порог засыпания упирается в потолок %.2f при k = %.2f (в худшей фазе суток уже при k = %.2f); тогда бодрствование скачком удлиняется до %.1f ч, то есть %.1f суток. Полка на кривых создана этим потолком, а не моделью.\n' % (H_CAP, kcap0, kcap1, wmax_h, wmax_h / 24),
          '| нужная длина бодрствования | требуемый порог H | k при полном повреждении, без модуляции | то же в худшей фазе |\n|---|---|---|---|']
    for hh in (24., 36., 48., wmax_h):
        Hn = 1 - (1 - L_M) * math.exp(-hh / TAU_R)
        L.append('| %.1f ч (%.2f сут) | %.3f | %.2f | %.2f |' % (hh, hh / 24, Hn, max(Hn - H_M, 0), max(Hn - H_M - C_A, 0)))
    R['checks'] = dict(wake_h=wH, sleep_h=sH, period_h=pH, S_mean_free=Sbar, max_wake_h=float(wmax_h), k_cap=float(kcap0), k_cap_worst=float(kcap1),
                       dx_vs_solver=float(dx), dc_vs_solver=float(dc), n_states_check=len(st))

    # ---------- 1. Отбор наборов ----------
    stage('выборка %d наборов' % K); S = sample(K, rng); sel = []
    for k in range(K):
        p = {n: float(S[n][k, 0]) for n in NAMES}; w = window(curve(p)[0])
        if not w or w[1] / w[0] <= 1.05: continue
        stt = states_at(p, math.sqrt(w[0] * w[1]))
        if len(stt) < 2: continue
        sel.append(dict(k=k, lo=w[0], hi=w[1], xlo=stt[0], xhi=stt[-1], r=float(rng.uniform(2, 4))))
        if len(sel) >= NSET: break
    nsel = len(sel)
    L += ['\n## 1. Наборы\n', '- взято в расчёт наборов с двумя состояниями: %d' % nsel]
    if nsel < 3:
        L.append('\nНаборов слишком мало.'); open(os.path.join(OUT, 'report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n'); print('\n'.join(L)); return
    P = {n: np.array([float(S[n][d['k'], 0]) for d in sel]) for n in NAMES}; P['r'] = np.array([d['r'] for d in sel])
    lo = np.array([d['lo'] for d in sel]); hi = np.array([d['hi'] for d in sel])
    xlo = np.array([d['xlo'] for d in sel]); xhi = np.array([d['xhi'] for d in sel]); thr = 0.5 * (xlo + xhi)
    q = lambda v, f='%.3f': (f + ' (10–90 %%: ' + f + ' – ' + f + ')') % tuple(np.nanpercentile(v, [50, 10, 90]))
    L += ['- отношение границ окна: ' + q(hi / lo, '%.2f'), '- T: ' + q(P['T'], '%.0f'), '- период внутренних часов в единицах T: ' + q(P['r'], '%.2f')]

    # ---------- 2. Парный перебор по силе связи ----------
    num = lo * (hi / lo) ** QPOS; rows = []; keep = {}
    def meas(rr, mask=None):
        """Сводка одного прогона; mask — подмножество наборов, по которому считать."""
        mk = np.ones(len(rr['nb']), bool) if mask is None else mask
        rr = {kk: (vv[mk] if isinstance(vv, np.ndarray) and vv.shape[:1] == mk.shape else vv) for kk, vv in rr.items()}
        ep = rr['epn'] > 0; ev = rr['everdam'] | rr['open_ep']
        dur = np.where(ep, rr['epsum'] / np.maximum(rr['epn'], 1), np.nan)
        started = rr['nb'] + rr['open_ep'] * 1.0                                        # незакрытый эпизод тоже считается начатым
        dur_all = np.where(ev, rr['tdam'] * ND / np.maximum(started, 1), np.nan)
        return dict(share_ever=float(ev.mean()), n_sets=int(mk.sum()),dur=float(np.nanmedian(dur)) if ep.any() else float('nan'), dur_arr=dur,
                    dur_all=float(np.nanmedian(dur_all)) if (started > 0).any() else float('nan'),
                    short=float(np.nanmedian(np.where(ep, rr['epshort'] / np.maximum(rr['epn'], 1), np.nan))) if ep.any() else float('nan'),
                    open_ep=float(rr['open_ep'].mean()), stuck=float((rr['wklong'] > 0).mean()), stuck15=float((rr['wk15'] > 0).mean()),
                    wkmax=float(np.median(rr['wkmax'])), wkmean=float(np.median(np.where(rr['wkn'] > 0, rr['wksum'] / np.maximum(rr['wkn'], 1), np.nan))),
                    per=float(np.nanmedian(rr['per'])), taw=float(np.median(rr['taw'])),
                    tdam_ev=float(np.median(rr['tdam'][ev])) if ev.any() else 0.0, share_break=float((rr['nb'] > 0).mean()),
                    nday=float(np.median(rr['nday'][rr['nb'] > 0])) if (rr['nb'] > 0).any() else 0.0,
                    bmid=float(rr.get('bmid', 0.0)))
    for ks in KSCAN:
        stage('связь k = %.2f (парно, зерно %d)' % (ks, SEED))
        rr = simulate(P, num, thr, xhi, xlo, ks, A0, SIG, TAUC, ND, M, Sbar, np.random.default_rng(SEED))
        m = meas(rr); m['k'] = ks; rows.append(m); keep[ks] = rr
        stage('  длительность %.2f сут; незакрытых %.3f; застряли дольше 1,5 сут %.3f' % (m['dur'], m['open_ep'], m['stuck15']))
    L += ['\n## 2. Что даёт связь «повреждение мешает заснуть» (одно зерно на весь перебор)\n',
          '| k | доля повреждавшихся | доля со срывом во время записи | длительность эпизода, сут | то же с учётом незакрытых | доля незакрытых эпизодов | доля эпизодов короче суток | застряли дольше 2 сут | дольше 1,5 сут | бодрствование: среднее / наибольшее, сут | период цикла, ч | доля времени бодрствования | время в повреждении среди повреждавшихся | срывов в сутки |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r_ in rows: L.append('| %.2f | %.3f | %.3f | %.2f | %.2f | %.3f | %.3f | %.3f | %.3f | %.2f / %.2f | %.1f | %.3f | %.3f | %.3f |' % (
        r_['k'], r_['share_ever'], r_['share_break'], r_['dur'], r_['dur_all'], r_['open_ep'], r_['short'], r_['stuck'], r_['stuck15'], r_['wkmean'], r_['wkmax'], r_['per'], r_['taw'], r_['tdam_ev'], r_['nday']))

    # ---------- 3. Погрешность по зёрнам и правило Ф3 ----------
    stage('погрешность по зёрнам на паре k = 0 и k = %.1f' % KBASE); pairs = []
    for rep in range(NREP):
        sd = SEED if rep == 0 else SEED + 1000 * rep
        if rep == 0: d0, db = rows[KSCAN.index(0.0)]['dur'], rows[KSCAN.index(KBASE)]['dur']
        else:
            r0 = meas(simulate(P, num, thr, xhi, xlo, 0.0, A0, SIG, TAUC, ND, M, Sbar, np.random.default_rng(sd)))
            rb = meas(simulate(P, num, thr, xhi, xlo, KBASE, A0, SIG, TAUC, ND, M, Sbar, np.random.default_rng(sd)))
            d0, db = r0['dur'], rb['dur']
        pairs.append(dict(seed=sd, dur0=d0, durb=db, ratio=db / d0 if d0 > 0 else float('nan')))
        stage('  зерно %d: %.2f -> %.2f, отношение %.2f' % (sd, d0, db, pairs[-1]['ratio']))
    rts = [p_['ratio'] for p_ in pairs]; rmed = float(np.median(rts)); va = bool(rmed >= 1.5)
    st_seq = [r_['stuck'] for r_ in rows]; vb = bool(all(st_seq[i + 1] >= st_seq[i] - 1e-12 for i in range(len(st_seq) - 1)) and max(st_seq[1:]) > 0)
    vc = bool(rows[KSCAN.index(KBASE)]['short'] < 0.50)
    L += ['\n## 3. Правило Ф3 на парном сравнении\n', '| зерно | длительность при k = 0 | при k = %.1f | отношение |\n|---|---|---|---|' % KBASE]
    for p_ in pairs: L.append('| %d | %.2f | %.2f | %.2f |' % (p_['seed'], p_['dur0'], p_['durb'], p_['ratio']))
    L += ['\n| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|',
          '| а | отношение %.2f (по зёрнам: %s) | не меньше 1,5 | %s |' % (rmed, ', '.join('%.2f' % r_ for r_ in rts), 'выполнено' if va else 'не выполнено'),
          '| б | %s | растёт с k | %s |' % (' → '.join('%.3f' % s_ for s_ in st_seq), 'выполнено' if vb else 'не выполнено'),
          '| в | %.3f | ниже 0,50; ожидал 0,2–0,3 | %s |' % (rows[KSCAN.index(KBASE)]['short'], 'выполнено' if vc else 'не выполнено'),
          '\nПрежний непарный прогон давал по части (а) отношение 1,30. Мера длительности не менялась: считаются только закончившиеся эпизоды, чтобы сравнение с прошлым прогоном осталось честным. Столбец «то же с учётом незакрытых» приведён рядом без вердикта.']
    R['F3'] = dict(a=dict(ratio_median=rmed, ratios=rts, thresh=1.5, verdict=va), b=dict(stuck_by_k=dict(zip(map(str, KSCAN), st_seq)), verdict=vb),
                   c=dict(short_share=rows[KSCAN.index(KBASE)]['short'], verdict=vc), pairs=pairs, prev_unpaired_ratio=1.30)
    R['kscan'] = [{kk: vv for kk, vv in r_.items() if kk != 'dur_arr'} for r_ in rows]

    # ---------- 4. Правило Ф4: портится ли сон раньше срыва ----------
    rb_ = keep[KBASE]; ds, dnn = rb_['dbin']; ls, ln = rb_['lbin']
    dcur = np.where(dnn > 0, ds / np.maximum(dnn, 1), np.nan); lcur = np.where(ln > 0, ls / np.maximum(ln, 1), np.nan)
    base_bout = float(dcur[0]) if dnn[0] > 0 else float('nan')                          # уровень здорового состояния: ячейка d от 0 до 0,1
    i0, i4 = 0, 4; nsub = float(dnn[3] + dnn[4])                                         # сколько бодрствований в подпороговых ячейках 0,3–0,5
    enough = nsub >= 20
    g = dcur[i4] / dcur[i0] if (enough and dnn[i4] > 0 and dcur[i0] > 0) else float('nan')
    v4a = bool(enough and g == g and g >= 1.10)
    lead_ok = [j for j in range(NLB) if ln[j] >= 5 and lcur[j] > base_bout]
    v4b = bool(lead_ok and min(lead_ok) <= 7)
    L += ['\n## 4. Правило Ф4: портится ли сон раньше срыва (k = %.1f)\n' % KBASE,
          '### (а) Длина бодрствования против повреждения во время него\n', '| повреждение d | длина бодрствования, сут | число бодрствований |\n|---|---|---|']
    for j in range(NDB): L.append('| %.1f–%.1f%s | %s | %d |' % (j / 10., (j + 1) / 10., ' (порог срыва)' if j == 4 else '', ('%.2f' % dcur[j]) if dnn[j] > 0 else '—', int(dnn[j])))
    L += ['\n- бодрствований с промежуточным повреждением (d от 0,3 до 0,7): %.3f от всех — чем меньше, тем ближе повреждение к двоичному' % rows[KSCAN.index(KBASE)]['bmid'],
          '- бодрствований в подпороговых ячейках 0,3–0,5: %d' % int(nsub),
          '\n**Ф4(а):** %s Вывод: %s.' % (('подпороговый рост в %.2f раза (ячейка 0,4–0,5 против 0,0–0,1; порог 1,10).' % g) if enough else 'в подпороговых ячейках меньше 20 бодрствований, правило непроверяемо: промежуточное повреждение в системе с двумя состояниями слишком редко.', 'выполнено' if v4a else ('непроверяемо' if not enough else 'не выполнено')),
          '\n### (б) Длина бодрствования в зависимости от срока до первого срыва\n', 'Уровень здорового состояния (повреждение ниже 0,1): %.2f сут.\n' % base_bout, '| суток до срыва | длина бодрствования, сут | число бодрствований |\n|---|---|---|']
    for j in range(NLB): L.append('| %d–%d | %s | %d |' % (j, j + 1, ('%.2f' % lcur[j]) if ln[j] > 0 else '—', int(ln[j])))
    L += ['\n**Ф4(б):** бодрствование выше здорового уровня за %s суток до срыва (учитываются ячейки, где не меньше пяти бодрствований). Вывод: %s.' % (('—' if not lead_ok else '%d–%d' % (min(lead_ok), max(lead_ok) + 1)), 'выполнено' if v4b else 'не выполнено')]
    R['F4'] = dict(a=dict(d_curve=[None if dnn[j] == 0 else float(dcur[j]) for j in range(NDB)], d_counts=[int(v) for v in dnn], gain=float(g), thresh=1.10, verdict=v4a),
                   b=dict(lead_curve=[None if ln[j] == 0 else float(lcur[j]) for j in range(NLB)], lead_counts=[int(v) for v in ln], base_bout=base_bout, lead_days=lead_ok, verdict=v4b),
                   enough_subthreshold=bool(enough), n_subthreshold=int(nsub), mid_share=rows[KSCAN.index(KBASE)]['bmid'])

    # ---------- 4б. Наборы, где суточный максимум остаётся внутри окна ----------
    nu_peak = num * (1 + A0) / (1 + A0 * Sbar); inside = nu_peak < hi
    L += ['\n## 4б. То же на наборах, где суточный максимум нагрузки остаётся внутри окна (без вердикта)\n',
          'Таких наборов %d из %d (%.3f). У остальных максимум выходит за верхнюю границу, и они срываются сразу, ещё на разгоне; в общих таблицах они завышают долю повреждённого времени.\n' % (int(inside.sum()), nsel, float(inside.mean())),
          '| k | доля повреждавшихся | длительность эпизода, сут | доля эпизодов короче суток | застряли дольше 1,5 сут | время в повреждении среди повреждавшихся |\n|---|---|---|---|---|---|']
    ins_rows = []
    for ks in KSCAN:
        mi = meas(keep[ks], inside); mi['k'] = ks; ins_rows.append(mi)
        L.append('| %.2f | %.3f | %.2f | %.3f | %.3f | %.3f |' % (ks, mi['share_ever'], mi['dur'], mi['short'], mi['stuck15'], mi['tdam_ev']))
    R['inside'] = dict(share=float(inside.mean()), rows=[{kk: vv for kk, vv in r_.items() if kk != 'dur_arr'} for r_ in ins_rows])

    # ---------- 5. Перебор положения в окне ----------
    stage('перебор положения средней нагрузки в окне'); qrows = []
    for qp in QSCAN:
        rr = simulate(P, lo * (hi / lo) ** qp, thr, xhi, xlo, KBASE, A0, SIG, TAUC, max(ND // 2, 4), M, Sbar, np.random.default_rng(SEED))
        m = meas(rr); m['q'] = qp; qrows.append(m)
    L += ['\n## 5. Насколько глубоко посажена средняя нагрузка (без вердикта; медианы по наборам со срывами)\n',
          '| q | доля повреждавшихся | время в повреждении среди повреждавшихся | срывов в сутки | застряли дольше 1,5 сут | доля эпизодов короче суток |\n|---|---|---|---|---|---|']
    for r_ in qrows: L.append('| %.2f | %.3f | %.3f | %.3f | %.3f | %.3f |' % (r_['q'], r_['share_ever'], r_['tdam_ev'], r_['nday'], r_['stuck15'], r_['short']))
    R['qscan'] = [{kk: vv for kk, vv in r_.items() if kk != 'dur_arr'} for r_ in qrows]

    # ---------- 6. Связь с параметрами ----------
    from scipy.stats import rankdata
    L += ['\n## 6. С чем связано застревание: ранговая корреляция с наибольшим бодрствованием\n', '| величина | корреляция |\n|---|---|']
    cors = {}
    for qn in NAMES + ['r']:
        cv = float(np.corrcoef(rankdata(P[qn]), rankdata(rb_['wkmax']))[0, 1]); cors[qn] = cv; L.append('| %s | %+.2f |' % (qn, cv))
    for nm, vv in (('запас до верхней границы, log', np.log(hi / (num * (1 + A0) / (1 + A0 * Sbar)))), ('доля времени в повреждении', rb_['tdam'])):
        cv = float(np.corrcoef(rankdata(vv), rankdata(rb_['wkmax']))[0, 1]); cors[nm] = cv; L.append('| %s | %+.2f |' % (nm, cv))
    R['corr'] = cors

    # ---------- 7. Для машинного чтения ----------
    R['setup'] = dict(mode='smoke' if SMOKE else ('full' if FULL else 'default'), n_sets=nsel, n_sampled=K, ndays=ND, steps_per_day=M, seed=SEED, n_rep=NREP,
                      A=A0, sigma=SIG, tauc_rel=TAUC, q_pos=QPOS, k_scan=KSCAN, k_base=KBASE, q_scan=QSCAN,
                      H=H_M, Lw=L_M, C_amp=C_A, H_cap=H_CAP, tau_rise_h=TAU_R, tau_decay_h=TAU_D, T_range=list(RANGES['T'][:2]))
    R['time_s'] = round(time.time() - T0, 1)
    L += ['\n## 7. Для машинного чтения\n', '```json', json.dumps(R, ensure_ascii=False, indent=1), '```']
    L.append('\nВремя счёта: %.0f с.' % (time.time() - T0)); text = '\n'.join(L) + '\n'
    open(os.path.join(OUT, 'report.md'), 'w', encoding='utf-8').write(text); json.dump(R, open(os.path.join(OUT, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print('\n' + text)

    # ---------- Рисунки ----------
    stage('рисунки'); import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'font.size': 9})
    cen = (np.arange(NDB) + 0.5) / NDB
    fig, a = plt.subplots(1, 1, figsize=(8.5, 4.8)); ok = dnn > 0
    a.plot(cen[ok], dcur[ok], 'o-', color=COL[0], lw=2); a.axvline(0.5, color=COL[1], lw=2, label='порог срыва')
    a.axvspan(0, 0.5, color='#e4e4e1', lw=0, zorder=0, label='повреждение ниже порога'); a.axhline(base_bout, color=INK, ls='--', lw=1, label='средняя длина бодрствования')
    a.set_xlabel('наибольшее повреждение d за это бодрствование'); a.set_ylabel('длина бодрствования, сутки')
    a.set_title('Ф4(а): бодрствование удлиняется уже до порога срыва', fontsize=10, loc='left'); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig1_bout_vs_damage.png'), dpi=110); plt.close(fig)

    fig, a = plt.subplots(1, 1, figsize=(8.5, 4.8)); okl = ln > 0; xs = np.arange(NLB) + 0.5
    a.bar(xs[okl], lcur[okl], width=0.85, color=COL[0], edgecolor='#fcfcfb'); a.axhline(base_bout, color=COL[1], lw=2, label='собственный средний уровень')
    a.set_xlabel('суток до первого срыва'); a.set_ylabel('длина бодрствования, сутки'); a.invert_xaxis()
    a.set_title('Ф4(б): опережение — за сколько суток сон начинает портиться', fontsize=10, loc='left'); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2_lead.png'), dpi=110); plt.close(fig)

    fig, a = plt.subplots(1, 1, figsize=(8.5, 4.8)); ks_ = [r_['k'] for r_ in rows]
    a.plot(ks_, [r_['dur'] for r_ in rows], 'o-', color=COL[0], lw=2, label='только закончившиеся эпизоды')
    a.plot(ks_, [r_['dur_all'] for r_ in rows], 's--', color=COL[2], lw=2, label='с учётом незакрытых')
    a.axhline(rows[0]['dur'] * 1.5, color=COL[1], ls='--', lw=1.2, label='в 1,5 раза выше уровня k = 0')
    a.axvline(H_CAP - H_M - C_A, color=INK, ls=':', lw=1.2, label='порог упирается в потолок (худшая фаза)')
    a.set_yscale('log'); a.set_xlabel('сила связи k'); a.set_ylabel('длительность эпизода, сутки')
    a.set_title('Длительность эпизода: парное сравнение', fontsize=10, loc='left'); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3_duration.png'), dpi=110); plt.close(fig)

    fig, a = plt.subplots(1, 1, figsize=(8.5, 4.8))
    a.plot(ks_, [r_['stuck15'] for r_ in rows], '^-', color=COL[2], lw=2, label='бодрствование дольше 1,5 суток')
    a.plot(ks_, [r_['stuck'] for r_ in rows], 'o-', color=COL[1], lw=2, label='дольше 2 суток')
    a.plot(ks_, [r_['short'] for r_ in rows], 's-', color=COL[0], lw=2, label='эпизоды короче суток')
    a.axvline(H_CAP - H_M - C_A, color=INK, ls=':', lw=1.2, label='порог упирается в потолок (худшая фаза)')
    a.axvline(H_CAP - H_M, color=INK, ls='-.', lw=1.2, label='то же без модуляции')
    a.set_xlabel('сила связи k'); a.set_ylabel('доля'); a.set_title('Застревание и возврат; пунктиры — где работает потолок', fontsize=10, loc='left'); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4_stuck.png'), dpi=110); plt.close(fig)

    j = int(np.argmax(rb_['wkmax'])); pj = {n: float(P[n][j]) for n in NAMES}; pj['r'] = float(P['r'][j]); nd_tr = min(12, ND)
    fig, ax = plt.subplots(2, 2, figsize=(13, 6), sharex='col', sharey='row')
    for col, ks in enumerate([0.0, KBASE]):
        tr_ = trace(pj, num[j], thr[j], xhi[j], xlo[j], ks, A0, SIG, TAUC, nd_tr, M, Sbar, np.random.default_rng(11))
        t_, Sv, aw_, nuv, dnv = tr_[:, 0], tr_[:, 1], tr_[:, 2], tr_[:, 3], tr_[:, 4]
        for a in (ax[0, col], ax[1, col]):
            sl = np.where(aw_ < 0.5)[0]
            if len(sl):
                for bb in np.split(sl, np.where(np.diff(sl) > 1)[0] + 1):
                    if len(bb): a.axvspan(t_[bb[0]], t_[bb[-1]], color='#e4e4e1', lw=0, zorder=0)
        ax[0, col].plot(t_, Sv, color=COL[0], lw=1.6); ax[0, col].set_title('k = %.1f' % ks, fontsize=10, loc='left')
        ax[1, col].plot(t_, dnv, color=COL[1], lw=1.6); ax[1, col].axhline(0.5, color=INK, ls='--', lw=1); ax[1, col].set_xlabel('время, сутки')
    ax[0, 0].set_ylabel('давление сна S'); ax[1, 0].set_ylabel('повреждение d')
    fig.suptitle('Серым — сон, пунктир — порог срыва. Слева связи нет, справа повреждение мешает заснуть', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(OUT, 'fig5_traces.png'), dpi=110); plt.close(fig)
    stage('готово: папка %s' % OUT)

if __name__ == '__main__':
    main()
