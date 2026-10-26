# -*- coding: utf-8 -*-
"""figs6.py — два контрастных эпизода полной модели: типичный и тяжёлый.

Запуск:     python latex/figs6.py           (из корня проекта)
Результат:  latex/pics/fig_episode.png

Строятся ДВА случая на одном наборе (#735 из принятой выборки), без поиска:

  A. ТИПИЧНЫЙ  — k_d = 2 мВ, без петли ёмкости: редкий короткий эпизод (около 1,5 сут)
     с ясным продромом (сон портится до срыва) и быстрым возвратом.

  B. ТЯЖЁЛЫЙ   — k_d = 4 мВ и петля ёмкости a_n = 0.3 (потеря ёмкости защиты, возврат
     30 сут): популяция сваливается в длинный эпизод (десятки суток) с D = 1, сон
     сходит почти на нет (пропущенные ночи), возврат медленный.

Ряды: положение нагрузки в окне q(t); доля повреждённых D и средняя доля рабочего
фермента x; защита c, наработанный белок s и сигнал индукции m; сон (ч/сутки) и
сомноген G (точные значения). Ось времени — сутки от конца разгона, общая для обоих.
"""
import math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from px.conf import load
from px.core import vuln
from px.consts import (NAMES, Q, F, TAUV, TAUM, NVM, NMV, A_M, NVC, ALPHA, TR, TD,
                       B0, AG, C1, C2, C3, C4, AE_S, THE, AI_S, THI, PE0, PE1,
                       GDEP, NWC, LAM, NSUB, TAU_M, TAU_S, SLPN, TAUSL, USE_DM,
                       COL, INK)

M = 800                                   # шагов на сутки (средние сходятся уже при M = 800)
WARM, NDAYS = 16, 60                      # разгон и горизонт, сутки
SIGMA, TAU_REL, SIGV, J = 0.08, 0.3, 0.10, 32
MIN_DUR = 0.3                             # минимальная длительность эпизода, сутки
PROD, AFTER = 10.0, 10.0                  # окно до/после эпизода для ТИПИЧНОГО случая

plt.rcParams.update({'font.size': 9.5, 'font.family': 'DejaVu Sans', 'savefig.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.22})


def run_trace(p, nu0, thr, xhi, xlo, aH, Gbar, qw, seed, kd=2.0,
              a_n=0.0, taun_dn=1.0, taun_up=30.0, nu_mul=1.0, t_boost=0.0, boost_dur=0.0):
    """Полный шаг модели (как px.model.trace, sigs = 0), с записью главных переменных.

    Возвращает словарь кривых и первый эпизод (onset, end в сутках от конца разгона,
    released — закончился ли он до горизонта), или None, если эпизода нет."""
    day = p['r'] * p['T']; dt = day / M; dth = 24. / M; ds = dth / NSUB
    v = vuln(J, SIGV)[0]; tauL = 0.1 * p['T']
    ea = math.exp(-dt / (TAU_REL * p['T'])); sdn = SIGMA * math.sqrt(1 - ea * ea)
    ec = math.exp(-dt / p['T']); eL = math.exp(-dt / tauL)
    av = 1 - math.exp(-ds / TAUV); am = 1 - math.exp(-ds / TAUM)
    er = math.exp(-dth / TR); ed = math.exp(-dth / TD)
    evz = math.exp(-dth / TAUSL); sdz = SLPN * math.sqrt(1 - evz * evz)
    em = math.exp(-dt / (TAU_M * day)); es = math.exp(-dt / (TAU_S * day))
    e_ndn = math.exp(-dt / (taun_dn * day)); e_nup = math.exp(-dt / (taun_up * day))
    x = np.ones(J); cb = np.full(J, p['N0']); mR = np.zeros(J); sP = np.zeros(J)
    c = np.full(J, p['N0']); nc = np.full(J, p['N0']); eta = 0.0; L = nu0; zz = 0.0
    rng = np.random.default_rng(seed); rngz = np.random.default_rng(99)
    Vv, Vm, G, E, I, Er, Ir = -5., A_M, 0.3, 0.3, 0.2, 0.3, 0.2
    n = int((WARM + NDAYS) * M)
    TT = np.empty(n); AW = np.empty(n); LS = np.empty(n); DS = np.empty(n)
    XS = np.empty(n); CS = np.empty(n); SS = np.empty(n); MS = np.empty(n)
    US = np.empty(n); GS = np.empty(n); ES_ = np.empty(n); ERS = np.empty(n)
    for i in range(n):
        dn = np.clip((xhi - x) / max(xhi - xlo, 1e-9), 0, 1)
        D = float((x < thr).mean()); Dm = float(dn.mean())
        t = i * dth
        Dv = B0 + NVC * math.cos(2 * math.pi * (t - ALPHA) / 24.) + AG * G - kd * (Dm if USE_DM else D) + zz
        zz = zz * evz + sdz * rngz.standard_normal()
        for _ in range(NSUB):
            Vv += (NVM * Q(Vm) + Dv - Vv) * av; Vm += (A_M + NMV * Q(Vv) - Vm) * am
        qm = Q(Vm); aw = qm > 1.
        G = 1 + (G - 1) * er if aw else G * ed
        g = 1. - GDEP * (Dm if USE_DM else D); Pe = PE0 + PE1 * min(qm / qw, 1.0)
        for _ in range(NWC):
            E += LAM * (F(C1 * E - C2 * g * I + Pe, AE_S, THE) - E)
            I += LAM * (F(C3 * E - C4 * g * I, AI_S, THI) - I)
            Er += LAM * (F(C1 * Er - C2 * Ir + Pe, AE_S, THE) - Er)
            Ir += LAM * (F(C3 * Er - C4 * Ir, AI_S, THI) - Ir)
        td = i / M - WARM
        mul = nu_mul if (t_boost <= td < t_boost + boost_dur) else 1.0
        Linf = nu0 * mul * (1 + aH * (G - Gbar)) * (E / max(Er, 1e-9)) * math.exp(eta)
        L = Linf + (L - Linf) * eL
        nu = np.maximum(L * v, 1e-12)
        eps = p['e0'] * c ** p['gam']; b = eps + x - nu; s_ = np.sqrt(b * b + 4 * eps * nu)
        u = np.where(b > 0, 2 * nu / (b + s_), (s_ - b) / (2 * np.maximum(eps, 1e-30)))
        aa = c ** p['m']; bb = aa + p['mu0'] * u * u / (1 + u); xi = aa / bb
        x = xi + (x - xi) * np.exp(-bb * dt)
        minf = p['beta'] * u / (p['KB'] + u); mR = minf + (mR - minf) * em
        sP = mR + (sP - mR) * es
        ntgt = np.where(x < thr, p['N0'] * (1 - a_n), p['N0'])
        nc = ntgt + (nc - ntgt) * np.where(ntgt < nc, e_ndn, e_nup)
        cbi = nc / (1 + u / p['KN']); cb = cbi + (cb - cbi) * ec
        c = np.maximum(cb + sP, 1e-9)
        eta = eta * ea + sdn * rng.standard_normal()
        TT[i] = td; AW[i] = aw; LS[i] = L; DS[i] = D
        XS[i] = x.mean(); CS[i] = c.mean(); SS[i] = sP.mean(); MS[i] = mR.mean()
        US[i] = float(np.log(np.maximum(u, 1e-30)).mean()); GS[i] = G; ES_[i] = E; ERS[i] = Er
    # первый эпизод: непрерывный участок D > 1/2 длиной не меньше MIN_DUR
    runs = []; i0 = None
    for i in range(len(DS)):
        if DS[i] > 0.5:
            if i0 is None: i0 = i
        elif i0 is not None:
            if (i - i0) / M >= MIN_DUR:
                runs.append((i0, i))
            i0 = None
    if i0 is not None and (len(DS) - i0) / M >= MIN_DUR:
        runs.append((i0, len(DS)))
    if not runs:
        return None
    i0, i1 = runs[0]
    return dict(TT=TT, AW=AW, LS=LS, DS=DS, XS=XS, CS=CS, SS=SS, MS=MS, US=US,
                GS=GS, ES=ES_, ERS=ERS, onset=i0 / M - WARM, end=i1 / M - WARM,
                released=(i1 < len(DS)))


def main():
    cfg = load(os.path.join(ROOT, 'config', 'base.toml'))
    cfg.d['cache']['dir'] = os.path.join(ROOT, 'cache')
    S = cfg.sets(log=print)
    q = cfg.d['sets']['q']

    k = 735                                                        # принятый набор (см. print ниже)
    p = {nm: float(S['P'][nm][k]) for nm in NAMES + ['r']}
    nu0 = float(S['lo'][k] * (S['hi'][k] / S['lo'][k]) ** q)
    thr, xhi, xlo = float(S['thr'][k]), float(S['xhi'][k]), float(S['xlo'][k])
    aH, Gbar, qw = float(S['A_H']), float(S['Gbar']), float(S['qw'])

    # A. ТИПИЧНЫЙ: k_d = 2 мВ, нет петли ёмкости; короткий эпизод с продромом.
    r1 = run_trace(p, nu0, thr, xhi, xlo, aH, Gbar, qw, seed=3, kd=2.0)
    # B. ТЯЖЁЛЫЙ: k_d = 4 мВ + петля ёмкости; длинный эпизод, пропущенные ночи.
    r2 = run_trace(p, nu0, thr, xhi, xlo, aH, Gbar, qw, seed=0, kd=4.0, a_n=0.3)

    cases = [dict(k=k, p=p, res=r1, kd=2.0, a_n=0.0, tag='typical'),
             dict(k=k, p=p, res=r2, kd=4.0, a_n=0.3, tag='severe, skipped nights')]

    for c in cases:
        r = c['res']; dur = r['end'] - r['onset']
        print('  набор #%d (%s): k_d = %.0f мВ, a_n = %.1f, эпизод: день %.1f, %.2f сут, released %s'
              % (c['k'], c['tag'], c['kd'], c['a_n'], r['onset'], dur, r['released']))
        print('    параметры: %s' % {nm: '%.3g' % c['p'][nm] for nm in NAMES})

    _plot(cases, S, q)
    out = os.path.join(ROOT, 'latex', 'pics', 'fig_episode.png')
    plt.savefig(out, dpi=170, bbox_inches='tight', pad_inches=0.08)
    print('готово: %s' % out)


def _daily_sleep(res, m):
    dabs = np.floor(res['TT'][m]).astype(int)
    days = np.unique(dabs)
    sleep = np.array([24. - 24. * res['AW'][m][dabs == d].mean() for d in days])
    return days + 0.5, sleep


def _plot(cases, S, q):
    n = len(cases)
    fig, axes = plt.subplots(4, n, figsize=(6.4 * n, 9.8), squeeze=False)
    fig.suptitle('One episode, two situations: typical vs. severe (same parameter set, different couplings)',
                 fontsize=11, y=0.995)
    for col, c in enumerate(cases):
        r = c['res']; lo = float(S['lo'][c['k']]); hi = float(S['hi'][c['k']])
        ep0, ep1 = r['onset'], r['end']
        if c['tag'] == 'typical':
            x0, x1 = ep0 - PROD, ep1 + AFTER
            prod = (ep0 - PROD, ep0)
        else:
            x0, x1 = -1.0, min(ep1 + 6.0, NDAYS)
            prod = None
            ep0 = max(ep0, 0.0)
        m = (r['TT'] >= x0) & (r['TT'] <= x1)
        tt = r['TT'][m]
        dur = ep1 - ep0 if c['tag'] == 'typical' else ep1

        for ax in axes[:, col]:
            if prod is not None:
                ax.axvspan(prod[0], prod[1], color='#f2c078', alpha=0.35, zorder=0)   # продром
            ax.axvspan(ep0, ep1, color='#e9897e', alpha=0.35, zorder=0)              # эпизод
        axes[0, col].set_title('Case %s (%s): set #%d, k_d = %.0f mV, episode %.1f d'
                               % ('AB'[col], c['tag'], c['k'], c['kd'], ep1 - ep0), fontsize=9)

        ax = axes[0, col]                                              # нагрузка в окне
        qp = (np.log(r['LS']) - math.log(lo)) / math.log(hi / lo)
        ax.axhspan(0, 1, color=INK, alpha=0.06, zorder=0)
        ax.plot(tt, qp[m], color=COL[0], lw=1.1)
        ax.axhline(q, color=INK, lw=0.7, ls=':')
        ax.set_ylabel('load position q(t)\n(0 = lower, 1 = upper boundary)')
        ax.set_ylim(-1.0, 1.8)
        ax.set_xlim(x0, x1)

        ax = axes[1, col]                                              # популяция
        ax.plot(tt, r['DS'][m], color=COL[1], lw=1.3, label='damaged share D')
        ax.plot(tt, r['XS'][m], color=COL[2], lw=1.3, label='working enzyme mean $\\bar{x}$')
        ax.axhline(0.5, color=INK, lw=0.7, ls=':')
        ax.set_ylabel('population state')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc='lower left', frameon=True, framealpha=0.88, edgecolor='none')

        ax = axes[2, col]                                              # защита
        ax.plot(tt, r['CS'][m], color=COL[0], lw=1.3, label='total defence $\\bar{c}$')
        ax.plot(tt, r['SS'][m], color=COL[1], lw=1.3, label='induced protein $\\bar{s}$')
        ax.plot(tt, r['MS'][m], color=COL[2], lw=1.0, ls='--', label='induction signal $\\bar{m}$')
        ax.set_ylabel('defence (means over cells)')
        ax.legend(fontsize=8, loc='lower right', frameon=True, framealpha=0.88, edgecolor='none')

        ax = axes[3, col]                                              # сон
        xd, sleep = _daily_sleep(r, m)
        ax.plot(xd, sleep, color=COL[3], lw=1.4, label='sleep, h/day')
        ax.set_ylabel('sleep, h/day'); ax.set_ylim(0, 24)
        ax2 = ax.twinx(); ax2.grid(False); ax2.spines['top'].set_visible(False)
        ax2.plot(tt, r['GS'][m], color=COL[0], lw=0.8, alpha=0.9, label='somnogen $G$')
        ax2.set_ylabel('somnogen $G$', color=COL[0]); ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', colors=COL[0])
        ax2.legend(fontsize=8, loc='upper left', frameon=True, framealpha=0.88, edgecolor='none')
        ax.legend(fontsize=8, loc='upper right', frameon=True, framealpha=0.88, edgecolor='none')
        ax.set_xlabel('days from end of warm-up')

    fig.tight_layout(pad=0.9, rect=[0, 0, 1, 0.985])


if __name__ == '__main__':
    main()
