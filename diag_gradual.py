# -*- coding: utf-8 -*-
"""
diag_gradual.py — диагностика: есть ли в модели хоть что-нибудь градуальное, и что именно.

ЗАЧЕМ
  В прошлом прогоне повреждённая доля d оказалась почти двоичной: промежуточные значения занимают около 1 % бодрствований.
  Это не ошибка, а содержание модели: x — быстрая переменная, и в системе с двумя устойчивыми состояниями она сидит
  на одной из двух ветвей, проскакивая середину. Для ОДНОЙ клетки это, вероятно, и есть правда.
  Ошибка в другом: двоичное состояние одной клетки было связано напрямую с градуальной величиной целого организма —
  со способностью заснуть. Нужен мост от клетки к организму. Кандидаты: популяция клеток либо медленная переменная c,
  которая и несёт PV и GAD67, то есть силу торможения.

  Этот скрипт ничего в модели не меняет и никаких правил не проверяет. Он только измеряет, насколько градуальны
  сами переменные и есть ли перед срывом наклон у чего-нибудь, кроме d. ВЕРДИКТОВ ЗДЕСЬ НЕТ.

ЧТО ИЗМЕРЯЕТСЯ
  1. Распределения по времени: повреждение d и дефицит защиты cn — двувершинные ли и насколько.
     d и cn приведены к размаху между устойчивыми ветвями в середине окна: 0 — здоровое состояние, 1 — повреждённое.
  2. Переходы: сколько времени система проводит в середине (d от 0,2 до 0,8) и сколько длится один переход.
  3. Что меняется перед срывом: d, cn, запас до верхней границы окна и длина бодрствования по суткам до срыва,
     за 15 суток. Засчитывается срыв, которому предшествовало не меньше трёх суток здорового состояния, и предыстория
     берётся только внутри этого здорового промежутка: иначе форму кривой задавал бы сам отбор, а не механизм.
     Если наклон есть, то видно, у какой величины.
  4. То же перед первым длинным бодрствованием (дольше 1,5 суток).
  5. Годится ли c в мост: суточный размах cn в здоровом состоянии против расстояния между ветвями.
     Если ветви по c почти совпадают, c в мост не годится — это тоже ответ.

МОДЕЛЬ — без изменений (функции перенесены из sleep_core2.py дословно; сон переключается по порогам,
  H_эфф = H + k*d, потолок 0,98). Прогоняются два значения связи: k = 0 и k = 0,2, одним зерном.

Запуск:  python diag_gradual.py            обычный вариант, около 2,5 минут
         python diag_gradual.py --full     больше наборов и суток, около 5 минут
         python diag_gradual.py --smoke    несколько наборов, только проверить, что скрипт доходит до конца
Нужны numpy, scipy, matplotlib. Результат: папка diag_out/ — report.md (в конце раздел для машинного чтения),
results.json, fig1…fig4.
"""
import os, sys, json, time, math
import numpy as np
from scipy.integrate import solve_ivp

T0 = time.time(); FULL = '--full' in sys.argv; SMOKE = '--smoke' in sys.argv; OUT = 'diag_out'
def stage(m): print('[%6.1f с] %s' % (time.time() - T0, m), flush=True)
UG = np.logspace(-4, 4, 641)
NAMES = ['e0', 'mu0', 'm', 'gam', 'KN', 'N0', 'beta', 'KB', 'T']
RANGES = dict(e0=(0.005, 0.1, 'log'), mu0=(0.01, 10, 'log'), m=(1, 2, 'lin'), gam=(0, 1, 'lin'), KN=(0.1, 100, 'log'), N0=(0.3, 1, 'lin'), beta=(0, 1, 'lin'), KB=(1, 100, 'log'), T=(20, 80, 'log'))
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']; INK = '#52514e'
TAU_R, TAU_D = 18.2, 4.2
H_M, L_M, C_A, TH0 = 0.67, 0.17, 0.12, 0.17
H_CAP = 0.98

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

# ---------- Устойчивые состояния: нужны и x, и c ----------
def states_xc(p, nuv):
    """Устойчивые состояния при нагрузке nuv: список пар (x, c), по возрастанию x."""
    nuc, xc, cc = curve(p); tr, dt_ = jac(p, UG, xc, cc); a0 = nuc[:-1] - nuv; a1 = nuc[1:] - nuv; cr = np.where(a0 * a1 < 0)[0]; out = []
    for i in cr:
        w = a0[i] / (a0[i] - a1[i])
        if (dt_[i] + w * (dt_[i + 1] - dt_[i])) > 0 and (tr[i] + w * (tr[i + 1] - tr[i])) < 0:
            out.append((float(xc[i] + w * (xc[i + 1] - xc[i])), float(cc[i] + w * (cc[i + 1] - cc[i]))))
    return sorted(out)

# ---------- Счёт ----------
NH = 20           # ячеек в распределениях d и cn
NLD = 15          # суток в кривых «что меняется перед событием»
RSTEP = 10        # точек хранения на сутки для этих кривых

def run(P, nu_mean, xhi, xlo, chi, clo, hi, kslp, A, sigma, tauc_rel, ndays, M, Sbar, rng, warm=NLD + 3):
    """Один прогон. Возвращает распределения, времена переходов и кривые «перед событием». Ничего не оценивает."""
    K = len(nu_mean); day = P['r'] * P['T']; dt = day / M; hr = day / 24.
    tauc = tauc_rel * P['T']; nu0 = nu_mean / (1 + A * Sbar)
    ea = np.exp(-dt / tauc); sdn = sigma * np.sqrt(1 - ea * ea); ec = np.exp(-dt / P['T'])
    er = np.exp(-dt / (TAU_R * hr)); ed = np.exp(-dt / (TAU_D * hr))
    x = np.ones(K); c = P['N0'] * 1.0; eta = np.zeros(K); S = np.full(K, L_M); awake = np.ones(K, bool); th = np.zeros(K)
    thr = 0.5 * (xhi + xlo); dam = np.zeros(K, bool)
    hd = np.zeros(NH); hc = np.zeros(NH)                                                # распределения по времени
    tmid_d = np.zeros(K); tmid_c = np.zeros(K); ncross = np.zeros(K)                    # время в середине и число переходов
    cs_h = np.zeros(K); cs_h2 = np.zeros(K); cs_hn = np.zeros(K)                        # cn в здоровом состоянии: сумма, сумма квадратов, счёт
    cs_d = np.zeros(K); cs_dn = np.zeros(K)                                             # cn в повреждённом состоянии
    wkst = np.zeros(K, int); bout_last = np.zeros(K)                                    # длина последнего закончившегося бодрствования
    ring = np.zeros((K, NLD * RSTEP, 4)); rt = np.full((K, NLD * RSTEP), -1.0); ptr = np.zeros(K, int)
    lb_s = np.zeros((NLD, 4)); lb_n = np.zeros(NLD)                                     # перед первым срывом
    lw_s = np.zeros((NLD, 4)); lw_n = np.zeros(NLD)                                     # перед первым длинным бодрствованием
    done_b = np.zeros(K, bool); done_w = np.zeros(K, bool); lastdam = np.zeros(K); nqual = np.zeros(K, int)
    HEAL = 3.0                                                                           # сколько суток набор должен быть здоров перед засчитываемым срывом
    nst = int((warm + ndays) * M); w0 = int(warm * M); rs = max(int(M / RSTEP), 1); hs = max(int(M / 400), 1)
    for i in range(nst):
        dn = np.clip((xhi - x) / np.maximum(xhi - xlo, 1e-9), 0, 1)
        cn = np.clip((chi - c) / np.where(np.abs(chi - clo) > 1e-9, chi - clo, np.nan), 0, 1)
        C = np.cos(2 * math.pi * (th - TH0)); H = np.minimum(H_M + C_A * C + kslp * dn, H_CAP); L = L_M + C_A * C
        nu = nu0 * (1 + A * S) * np.exp(eta)
        eps = P['e0'] * c ** P['gam']; b = eps + x - nu; s_ = np.sqrt(b * b + 4 * eps * nu); u = np.where(b > 0, 2 * nu / (b + s_), (s_ - b) / (2 * np.maximum(eps, 1e-30)))
        aa = c ** P['m']; bb = aa + P['mu0'] * u * u / (1 + u); xinf = aa / bb; x = xinf + (x - xinf) * np.exp(-bb * dt)
        cinf = P['N0'] / (1 + u / P['KN']) + P['beta'] * u / (P['KB'] + u); c = cinf + (c - cinf) * ec
        S = np.where(awake, 1 + (S - 1) * er, S * ed); eta = eta * ea + sdn * rng.standard_normal(K); th = (th + dt / day) % 1.0
        fall = awake & (S >= H); wake_ = (~awake) & (S <= L)
        if fall.any():
            idf = np.where(fall)[0]; bout_last[idf] = (i - wkst[idf]) / float(M)
            if i >= w0:
                lng = idf[(bout_last[idf] > 1.5) & ~done_w[idf]]                         # первое длинное бодрствование: разносим запомненное
                if len(lng):
                    tb = i / float(M)
                    for jj in range(NLD * RSTEP):
                        lead = tb - rt[lng, jj]; ok = (rt[lng, jj] >= 0) & (lead > 0) & (lead < NLD)
                        if ok.any():
                            jl = lead[ok].astype(int); w_ = ring[lng[ok], jj, :]
                            for cc_ in range(4): np.add.at(lw_s[:, cc_], jl, w_[:, cc_])
                            np.add.at(lw_n, jl, 1.0)
                    done_w[lng] = True
        if wake_.any(): wkst[wake_] = i
        awake = np.where(fall, False, np.where(wake_, True, awake))
        nd_ = x < thr; down = nd_ & ~dam; up = (~nd_) & dam
        qual = (i / float(M) - lastdam) >= HEAL                                          # был ли набор здоров достаточно долго ДО этого шага
        if i % rs == 0:                                                                  # память заполняется всегда, и на разгоне тоже
            sl = ptr % (NLD * RSTEP); ar = np.arange(K); ring[ar, sl, 0] = dn; ring[ar, sl, 1] = np.nan_to_num(cn)
            ring[ar, sl, 2] = np.log(hi / nu); ring[ar, sl, 3] = bout_last; rt[ar, sl] = i / float(M); ptr += 1
        if i >= w0:
            if i % hs == 0:                                                              # распределения: подвыборка, 400 точек в сутки
                np.add.at(hd, np.clip((dn * NH).astype(int), 0, NH - 1), 1.0)
                okc = cn == cn
                if okc.any(): np.add.at(hc, np.clip((cn[okc] * NH).astype(int), 0, NH - 1), 1.0)
            tmid_d += ((dn > 0.2) & (dn < 0.8)) * dt; tmid_c += ((cn > 0.2) & (cn < 0.8)) * dt
            hlth = dn < 0.1; dmg = dn > 0.9
            cs_h += np.where(hlth, np.nan_to_num(cn), 0); cs_h2 += np.where(hlth, np.nan_to_num(cn) ** 2, 0); cs_hn += hlth
            cs_d += np.where(dmg, np.nan_to_num(cn), 0); cs_dn += dmg
            ncross += down + up
            fb = np.where(down & ~done_b & qual)[0]                                   # срыв после здорового промежутка, иначе ждём следующего
            if len(fb):
                tb = i / float(M); ld = lastdam[fb]                                      # предыстория берётся только внутри здорового промежутка,
                for jj in range(NLD * RSTEP):                                            # иначе форму кривой задаёт сам отбор, а не механизм
                    lead = tb - rt[fb, jj]; ok = (rt[fb, jj] >= 0) & (lead > 0) & (lead < NLD) & (rt[fb, jj] > ld)
                    if ok.any():
                        jl = lead[ok].astype(int); w_ = ring[fb[ok], jj, :]
                        for cc_ in range(4): np.add.at(lb_s[:, cc_], jl, w_[:, cc_])
                        np.add.at(lb_n, jl, 1.0)
                done_b[fb] = True; nqual[fb] = 1
        lastdam = np.where(nd_, i / float(M), lastdam)                                   # отметка последнего повреждения, обновляется и на разгоне
        dam = nd_
    tot = ndays * day
    mh = np.where(cs_hn > 0, cs_h / np.maximum(cs_hn, 1), np.nan); sh = np.sqrt(np.maximum(np.where(cs_hn > 0, cs_h2 / np.maximum(cs_hn, 1), np.nan) - mh ** 2, 0))
    md = np.where(cs_dn > 0, cs_d / np.maximum(cs_dn, 1), np.nan)
    return dict(hd=hd, hc=hc, tmid_d=tmid_d / tot, tmid_c=tmid_c / tot, ncross=ncross,
                cn_h_mean=mh, cn_h_sd=sh, cn_d_mean=md, n_h=cs_hn, n_d=cs_dn,
                lb=(lb_s, lb_n), lw=(lw_s, lw_n), nqual=nqual, ndays=ndays, day=day)

# ---------- Основной расчёт ----------
def main():
    K, NSET, ND, M = (8000, 60, 25, 1500) if SMOKE else ((200000, 600, 90, 8000) if FULL else (40000, 300, 60, 6000))
    A0, SIG, TAUC, QPOS = 0.2, 0.08, 0.3, 0.75; KS = [0.0, 0.2]; SEED = 4242
    os.makedirs(OUT, exist_ok=True); rng = np.random.default_rng(83); R = {}
    L = ['# Диагностика: что в модели градуально\n', 'Режим: %s. Выборка %d наборов; в расчёте до %d наборов с двумя состояниями; %d суток; %d шагов на период внутренних часов.\n' % ('пробный' if SMOKE else ('полный' if FULL else 'обычный'), K, NSET, ND, M),
         '**Вердиктов в этом отчёте нет.** Модель не менялась; это только измерение. d — повреждённая доля, cn — дефицит защиты; обе приведены к размаху между устойчивыми ветвями в середине окна: 0 — здоровое состояние, 1 — повреждённое.\n']

    # ---------- 0. Проверки ----------
    stage('проверки'); wH, sH, pH, Sbar = free_cycle()
    pk = {'e0': 0.02, 'mu0': 0.3, 'm': 2.0, 'gam': 0.5, 'KN': 3.0, 'N0': 0.9, 'beta': 0.1, 'KB': 10.0, 'T': 50.0}
    wk_ = window(curve(pk)[0]); nuM = math.sqrt(wk_[0] * wk_[1]); sxc = states_xc(pk, nuM)
    L += ['## 0. Проверки\n', '- свободный цикл сна при k = 0: бодрствование %.1f ч, сон %.1f ч, период %.1f ч' % (wH, sH, pH),
          '- среднее давление сна за свободный цикл: %.3f' % Sbar,
          '- проверочный набор: устойчивых состояний в середине окна %d; x на ветвях %.3f и %.3f; c на ветвях %.3f и %.3f' % (len(sxc), sxc[0][0], sxc[-1][0], sxc[0][1], sxc[-1][1])]
    R['checks'] = dict(wake_h=wH, sleep_h=sH, period_h=pH, S_mean_free=Sbar, n_states=len(sxc))

    # ---------- 1. Отбор наборов ----------
    stage('выборка %d наборов' % K); S = sample(K, rng); sel = []
    for k in range(K):
        p = {n: float(S[n][k, 0]) for n in NAMES}; w = window(curve(p)[0])
        if not w or w[1] / w[0] <= 1.05: continue
        stt = states_xc(p, math.sqrt(w[0] * w[1]))
        if len(stt) < 2: continue
        sel.append(dict(k=k, lo=w[0], hi=w[1], xlo=stt[0][0], xhi=stt[-1][0], clo=stt[0][1], chi=stt[-1][1], r=float(rng.uniform(2, 4))))
        if len(sel) >= NSET: break
    nsel = len(sel)
    if nsel < 3:
        L.append('\nНаборов слишком мало.'); open(os.path.join(OUT, 'report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n'); print('\n'.join(L)); return
    P = {n: np.array([float(S[n][d['k'], 0]) for d in sel]) for n in NAMES}; P['r'] = np.array([d['r'] for d in sel])
    lo = np.array([d['lo'] for d in sel]); hi = np.array([d['hi'] for d in sel])
    xlo = np.array([d['xlo'] for d in sel]); xhi = np.array([d['xhi'] for d in sel])
    clo = np.array([d['clo'] for d in sel]); chi = np.array([d['chi'] for d in sel])
    gapc = np.abs(chi - clo) / np.maximum(np.abs(chi), 1e-12); gapx = np.abs(xhi - xlo)
    q = lambda v, f='%.3f': (f + ' (10–90 %%: ' + f + ' – ' + f + ')') % tuple(np.nanpercentile(v, [50, 10, 90]))
    L += ['\n## 1. Наборы и расстояние между ветвями\n', '- взято наборов: %d' % nsel, '- отношение границ окна: ' + q(hi / lo, '%.2f'),
          '- расстояние между ветвями по x: ' + q(gapx, '%.3f'), '- то же по c, в долях от здорового уровня: ' + q(gapc, '%.3f'),
          '- доля наборов, у которых ветви по c расходятся меньше чем на 5 %%: %.3f (у них c в мост не годится)' % float((gapc < 0.05).mean())]
    R['sets'] = dict(n=nsel, gap_x_median=float(np.median(gapx)), gap_c_median=float(np.median(gapc)), share_gap_c_small=float((gapc < 0.05).mean()))

    num = lo * (hi / lo) ** QPOS; res = {}
    for ks in KS:
        stage('прогон k = %.2f' % ks)
        res[ks] = run(P, num, xhi, xlo, chi, clo, hi, ks, A0, SIG, TAUC, ND, M, Sbar, np.random.default_rng(SEED))

    # ---------- 2. Распределения ----------
    L += ['\n## 2. Распределения по времени: насколько двувершинны d и cn\n', '| ячейка | d при k = 0 | d при k = 0,2 | cn при k = 0 | cn при k = 0,2 |\n|---|---|---|---|---|']
    nrm = lambda h: h / max(h.sum(), 1)
    for j in range(NH):
        L.append('| %.2f–%.2f | %.4f | %.4f | %.4f | %.4f |' % (j / NH, (j + 1) / NH, nrm(res[0.0]['hd'])[j], nrm(res[0.2]['hd'])[j], nrm(res[0.0]['hc'])[j], nrm(res[0.2]['hc'])[j]))
    mid = lambda h: float(nrm(h)[int(0.2 * NH):int(0.8 * NH)].sum())
    L += ['\n- доля времени в середине (от 0,2 до 0,8): d — %.4f при k = 0 и %.4f при k = 0,2; cn — %.4f и %.4f' % (mid(res[0.0]['hd']), mid(res[0.2]['hd']), mid(res[0.0]['hc']), mid(res[0.2]['hc'])),
          '- то же по траекториям, а не по подвыборке: d — ' + q(res[0.2]['tmid_d'], '%.4f') + '; cn — ' + q(res[0.2]['tmid_c'], '%.4f')]
    for ks in KS:
        nc = res[ks]['ncross']; tm = res[ks]['tmid_d'] * res[ks]['ndays']; act = nc > 0
        tt = float(np.median(tm[act] / nc[act] * 24)) if act.any() else float('nan')
        L.append('- длительность одного перехода при k = %.1f (время в середине, делённое на число переходов): %.2f ч' % (ks, tt))
        R.setdefault('transition_h', {})[str(ks)] = tt
    R['mid_share'] = dict(d_k0=mid(res[0.0]['hd']), d_kb=mid(res[0.2]['hd']), cn_k0=mid(res[0.0]['hc']), cn_kb=mid(res[0.2]['hc']))

    # ---------- 3. Что меняется перед срывом ----------
    def curve_tab(pack, title, note):
        s_, n_ = pack; out = ['\n%s\n' % title, note, '\n| суток до события | d | дефицит защиты cn | запас до границы, log | длина бодрствования, сут | число точек |\n|---|---|---|---|---|---|']
        vals = []
        for j in range(NLD):
            if n_[j] > 0:
                v = s_[j] / n_[j]; vals.append(v); out.append('| %d–%d | %.3f | %.3f | %.3f | %.3f | %d |' % (j, j + 1, v[0], v[1], v[2], v[3], int(n_[j])))
            else: out.append('| %d–%d | — | — | — | — | 0 |' % (j, j + 1)); vals.append([np.nan] * 4)
        return out, np.array(vals)
    o1, v1 = curve_tab(res[0.2]['lb'], '## 3. Что меняется перед срывом (k = 0,2)', 'Отсчёт назад от первого срыва, которому предшествовало не меньше трёх суток здорового состояния: таких наборов %d из %d. Учитывается только предыстория внутри самого здорового промежутка, поэтому число точек с удалением от срыва падает. Если у какой-то величины есть наклон, а у d его нет, значит градуальность живёт в ней.' % (int(res[0.2]['nqual'].sum()), nsel))
    L += o1
    def slope(v):
        ok = ~np.isnan(v); 
        return float(np.polyfit(np.arange(NLD)[ok], v[ok], 1)[0]) if ok.sum() >= 6 else float('nan')
    sl = [slope(v1[:, c_]) for c_ in range(4)]
    L += ['\n- наклон по суткам (знак минус — величина растёт по мере приближения к срыву): d %+.4f, cn %+.4f, запас %+.4f, бодрствование %+.4f' % tuple(sl),
          '- значение за 0–1 сутки против 8–15 суток: d %s, cn %s, запас %s, бодрствование %s' % tuple(
              ('%.3f / %.3f' % (v1[0, c_], np.nanmean(v1[8:, c_]))) if not np.isnan(v1[0, c_]) else '—' for c_ in range(4))]
    o2, v2 = curve_tab(res[0.2]['lw'], '\n## 4. То же перед первым длинным бодрствованием (дольше 1,5 суток)', 'Отсчёт назад от первого бодрствования дольше 1,5 суток.')
    L += o2
    R['before_break'] = dict(rows=[[None if np.isnan(v1[j, c_]) else float(v1[j, c_]) for c_ in range(4)] for j in range(NLD)], counts=[int(v) for v in res[0.2]['lb'][1]], slopes=sl)
    R['before_longbout'] = dict(rows=[[None if np.isnan(v2[j, c_]) else float(v2[j, c_]) for c_ in range(4)] for j in range(NLD)], counts=[int(v) for v in res[0.2]['lw'][1]])

    # ---------- 5. Годится ли c в мост ----------
    rb = res[0.2]; okh = rb['n_h'] > 0; okd = rb['n_d'] > 0; both = okh & okd
    swing = rb['cn_h_sd'][okh]; gap = (rb['cn_d_mean'] - rb['cn_h_mean'])[both]
    ratio = (rb['cn_h_sd'][both] / np.maximum(np.abs(gap), 1e-9))
    L += ['\n## 5. Годится ли медленная переменная c в мост (k = 0,2)\n',
          '- суточный разброс дефицита защиты cn в здоровом состоянии: ' + (q(swing, '%.3f') if okh.any() else '—'),
          '- расстояние между здоровым и повреждённым состоянием по cn: ' + (q(gap, '%.3f') if both.any() else '—'),
          '- отношение разброса к расстоянию: ' + (q(ratio, '%.3f') if both.any() else '—') + ' — чем больше, тем плавнее cn ходит внутри здорового состояния',
          '- наборов, где оба состояния встретились: %d из %d' % (int(both.sum()), nsel)]
    R['bridge_c'] = dict(swing_median=float(np.nanmedian(swing)) if okh.any() else None, gap_median=float(np.nanmedian(gap)) if both.any() else None,
                         ratio_median=float(np.nanmedian(ratio)) if both.any() else None, n_both=int(both.sum()))

    # ---------- 6. Для машинного чтения ----------
    R['setup'] = dict(mode='smoke' if SMOKE else ('full' if FULL else 'default'), n_sets=nsel, n_sampled=K, ndays=ND, steps_per_day=M,
                      A=A0, sigma=SIG, tauc_rel=TAUC, q_pos=QPOS, k_values=KS, seed=SEED, NH=NH, NLD=NLD)
    R['time_s'] = round(time.time() - T0, 1)
    L += ['\n## 6. Для машинного чтения\n', '```json', json.dumps(R, ensure_ascii=False, indent=1), '```', '\nВремя счёта: %.0f с.' % (time.time() - T0)]
    text = '\n'.join(L) + '\n'; open(os.path.join(OUT, 'report.md'), 'w', encoding='utf-8').write(text)
    json.dump(R, open(os.path.join(OUT, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print('\n' + text)

    # ---------- Рисунки ----------
    stage('рисунки'); import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'font.size': 9})
    cen = (np.arange(NH) + 0.5) / NH
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for a, (h0, hb, nm) in zip(ax, [(res[0.0]['hd'], res[0.2]['hd'], 'повреждение d'), (res[0.0]['hc'], res[0.2]['hc'], 'дефицит защиты cn')]):
        a.bar(cen - 0.012, nrm(h0), width=0.022, color=COL[0], label='k = 0'); a.bar(cen + 0.012, nrm(hb), width=0.022, color=COL[1], label='k = 0,2')
        a.axvspan(0.2, 0.8, color='#e4e4e1', lw=0, zorder=0, label='середина'); a.set_xlabel(nm); a.set_yscale('log'); a.legend(frameon=False, fontsize=8)
    ax[0].set_ylabel('доля времени'); fig.suptitle('Распределения по времени: насколько переменные двувершинны', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(OUT, 'fig1_distributions.png'), dpi=110); plt.close(fig)

    for fn, vv, nn, ttl in (('fig2_before_break.png', v1, res[0.2]['lb'][1], 'перед первым срывом'), ('fig3_before_longbout.png', v2, res[0.2]['lw'][1], 'перед первым длинным бодрствованием')):
        fig, ax = plt.subplots(1, 4, figsize=(15, 3.8)); xs = np.arange(NLD) + 0.5
        for a, c_, nm in zip(ax, range(4), ['повреждение d', 'дефицит защиты cn', 'запас до границы, log', 'длина бодрствования, сут']):
            ok = nn > 0; a.plot(xs[ok], vv[ok, c_], 'o-', color=COL[c_ % 4], lw=2); a.invert_xaxis(); a.set_xlabel('суток до события'); a.set_title(nm, fontsize=9.5, loc='left')
        fig.suptitle('Что меняется %s (k = 0,2)' % ttl, fontsize=10.5); fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(OUT, fn), dpi=110); plt.close(fig)

    fig, a = plt.subplots(1, 1, figsize=(8, 4.6))
    if both.any():
        a.scatter(np.abs(gap), rb['cn_h_sd'][both], s=16, color=COL[0], alpha=0.6, linewidths=0)
        lim = np.nanpercentile(np.abs(gap), 99); xs_ = np.linspace(0, lim, 50); a.plot(xs_, xs_, color=INK, ls='--', lw=1, label='разброс равен расстоянию')
        a.set_xlim(0, lim)
    a.set_xlabel('расстояние между состояниями по cn'); a.set_ylabel('суточный разброс cn в здоровом состоянии')
    a.set_title('Годится ли c в мост: чем ближе точки к диагонали, тем плавнее', fontsize=10, loc='left'); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4_bridge_c.png'), dpi=110); plt.close(fig)
    stage('готово: папка %s' % OUT)

if __name__ == '__main__':
    main()
