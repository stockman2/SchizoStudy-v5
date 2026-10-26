# -*- coding: utf-8 -*-
"""figs6.py — эпизод, продром и последствие в полной модели; одна картинка, два набора.

Запуск:     python latex/figs6.py           (из корня проекта)
Результат:  latex/pics/fig_episode.png

Что делает. Берёт принятые (бистабильные) наборы из кэша выборки, находит два набора
с небольшим зазором вверх (эпизоды случаются за время прогона), прогоняет траекторию
полной модели — тот же шаг, что px.model.trace, с записью главных переменных, — и
рисует окно [10 сут до, эпизод, 10 сут после] вокруг ПЕРВОГО эпизода каждого набора.

Ряды: положение нагрузки в окне q(t); доля повреждённых D и средняя доля рабочего
фермента x; защита c, наработанный белок s и сигнал индукции m; суточное
бодрствование и возбуждение коры E/E_r. Заливка: оранжевая — продром (10 сут до
срыва), красная — сам эпизод (D > 1/2).
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
SIGMA, TAU_REL, KD, SIGV, J = 0.08, 0.3, 2.0, 0.10, 32
PROD, AFTER = 10.0, 10.0                  # сколько суток рисовать до и после эпизода
MIN_DUR = 0.3                             # минимальная длительность эпизода, сутки

plt.rcParams.update({'font.size': 9.5, 'font.family': 'DejaVu Sans', 'savefig.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.22})


def run_trace(p, nu0, thr, xhi, xlo, aH, Gbar, qw, seed):
    """Тот же шаг, что px.model.trace (полная модель, sigs = 0, без примирования),
    но с записью всех главных переменных на каждом шаге. Возвращает словарь кривых
    и (onset, end) первого эпизода в сутках от конца разгона, или None."""
    day = p['r'] * p['T']; dt = day / M; dth = 24. / M; ds = dth / NSUB
    v = vuln(J, SIGV)[0]; tauL = 0.1 * p['T']
    ea = math.exp(-dt / (TAU_REL * p['T'])); sdn = SIGMA * math.sqrt(1 - ea * ea)
    ec = math.exp(-dt / p['T']); eL = math.exp(-dt / tauL)
    av = 1 - math.exp(-ds / TAUV); am = 1 - math.exp(-ds / TAUM)
    er = math.exp(-dth / TR); ed = math.exp(-dth / TD)
    evz = math.exp(-dth / TAUSL); sdz = SLPN * math.sqrt(1 - evz * evz)
    em = math.exp(-dt / (TAU_M * day)); es = math.exp(-dt / (TAU_S * day))
    x = np.ones(J); cb = np.full(J, p['N0']); mR = np.zeros(J); sP = np.zeros(J)
    c = np.full(J, p['N0']); eta = 0.0; L = nu0; zz = 0.0
    rng = np.random.default_rng(seed); rngz = np.random.default_rng(99)
    Vv, Vm, G, E, I, Er, Ir = -5., A_M, 0.3, 0.3, 0.2, 0.3, 0.2
    n = int((WARM + NDAYS) * M)
    TT = np.empty(n); AW = np.empty(n); LS = np.empty(n); DS = np.empty(n)
    XS = np.empty(n); CS = np.empty(n); SS = np.empty(n); MS = np.empty(n)
    US = np.empty(n); ES_ = np.empty(n); ERS = np.empty(n)
    for i in range(n):
        dn = np.clip((xhi - x) / max(xhi - xlo, 1e-9), 0, 1)
        D = float((x < thr).mean()); Dm = float(dn.mean())
        t = i * dth
        Dv = B0 + NVC * math.cos(2 * math.pi * (t - ALPHA) / 24.) + AG * G - KD * (Dm if USE_DM else D) + zz
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
        Linf = nu0 * (1 + aH * (G - Gbar)) * (E / max(Er, 1e-9)) * math.exp(eta)
        L = Linf + (L - Linf) * eL
        nu = np.maximum(L * v, 1e-12)
        eps = p['e0'] * c ** p['gam']; b = eps + x - nu; s_ = np.sqrt(b * b + 4 * eps * nu)
        u = np.where(b > 0, 2 * nu / (b + s_), (s_ - b) / (2 * np.maximum(eps, 1e-30)))
        aa = c ** p['m']; bb = aa + p['mu0'] * u * u / (1 + u); xi = aa / bb
        x = xi + (x - xi) * np.exp(-bb * dt)
        minf = p['beta'] * u / (p['KB'] + u); mR = minf + (mR - minf) * em
        sP = mR + (sP - mR) * es
        cbi = p['N0'] / (1 + u / p['KN']); cb = cbi + (cb - cbi) * ec
        c = np.maximum(cb + sP, 1e-9)
        eta = eta * ea + sdn * rng.standard_normal()
        TT[i] = i / M - WARM; AW[i] = aw; LS[i] = L; DS[i] = D
        XS[i] = x.mean(); CS[i] = c.mean(); SS[i] = sP.mean(); MS[i] = mR.mean()
        US[i] = float(np.log(np.maximum(u, 1e-30)).mean()); ES_[i] = E; ERS[i] = Er
    # первый эпизод: непрерывный участок D > 1/2 длиной не меньше MIN_DUR
    i0 = None
    for i in range(len(DS)):
        if DS[i] > 0.5:
            if i0 is None: i0 = i
        elif i0 is not None:
            if (i - i0) / M >= MIN_DUR:
                break
            i0 = None
    if i0 is None or i >= len(DS) - 1:
        return None
    onset = i0 / M - WARM; end = i / M - WARM
    if onset < PROD or end + AFTER > NDAYS:
        return None
    return dict(TT=TT, AW=AW, LS=LS, DS=DS, XS=XS, CS=CS, SS=SS, MS=MS, US=US,
                ES=ES_, ERS=ERS, onset=onset, end=end)


def main():
    cfg = load(os.path.join(ROOT, 'config', 'base.toml'))
    cfg.d['cache']['dir'] = os.path.join(ROOT, 'cache')       # кэш независимо от cwd
    S = cfg.sets(log=print)
    q = cfg.d['sets']['q']
    gap = S['gap_up'] / SIGMA                                  # зазор вверх в сигмах (у суточного пика)
    order = np.argsort(gap)
    print('наборов: %d; зазор вверх: min %.2f med %.2f max %.2f сигм' %
          (S['nsel'], gap[order[0]], np.median(gap), gap[order[-1]]))

    cases = []
    for k in order:                                            # проход 1: два разных режима
        if gap[k] < 1.2: continue
        if cases and gap[k] < cases[0]['gap'] + 0.6: continue
        if len(cases) >= 2: break
        cases += _try(k, S, gap, q)
    if len(cases) < 2:                                         # проход 2: любой второй режим
        for k in order:
            if gap[k] < 1.2 or any(c['k'] == k for c in cases): continue
            cases += _try(k, S, gap, q)
            if len(cases) >= 2: break

    if not cases:
        raise SystemExit('ни один набор не дал эпизода за горизонт: ослабьте отбор')
    print('случаев для рисунка: %d' % len(cases))
    for c in cases:
        print('  набор #%d, зазор %.2f сигм, W = %.2f, эпизод: день %.1f, %.2f сут, сид %d, %s'
              % (c['k'], c['gap'], c['W'], c['res']['onset'], c['res']['end'] - c['res']['onset'],
                 c['seed'], {n: '%.3g' % c['p'][n] for n in NAMES}))

    _plot(cases, S, q)
    out = os.path.join(ROOT, 'latex', 'pics', 'fig_episode.png')
    plt.savefig(out, dpi=170, bbox_inches='tight', pad_inches=0.08)
    print('готово: %s' % out)


def _try(k, S, gap, q):
    p = {n: float(S['P'][n][k]) for n in NAMES + ['r']}
    nu0 = float(S['lo'][k] * (S['hi'][k] / S['lo'][k]) ** q)
    for seed in range(4):
        res = run_trace(p, nu0, float(S['thr'][k]), float(S['xhi'][k]), float(S['xlo'][k]),
                        float(S['A_H']), float(S['Gbar']), float(S['qw']), seed)
        if res is not None:
            return [dict(k=k, gap=float(gap[k]), W=float(S['hi'][k] / S['lo'][k]),
                         p=p, seed=seed, res=res)]
    return []


def _plot(cases, S, q):
    n = len(cases)
    fig, axes = plt.subplots(4, n, figsize=(6.4 * n, 9.8), squeeze=False)
    fig.suptitle('One episode of the full model: prodrome, episode, aftermath '
                 '(two accepted parameter sets; English labels)',
                 fontsize=11, y=0.995)
    for col, c in enumerate(cases):
        r = c['res']; lo = float(S['lo'][c['k']]); hi = float(S['hi'][c['k']])
        dur = r['end'] - r['onset']
        m = (r['TT'] >= r['onset'] - PROD) & (r['TT'] <= r['end'] + AFTER)
        tt = r['TT'][m] - r['onset']
        for ax in axes[:, col]:
            ax.axvspan(-PROD, 0, color='#f2c078', alpha=0.35, zorder=0)   # продром
            ax.axvspan(0, dur, color='#e9897e', alpha=0.35, zorder=0)     # эпизод
        axes[0, col].set_title('Case %s: set #%d, W = %.2f, upward gap = %.1fσ, episode %.2f d'
                               % ('ABCD'[col], c['k'], c['W'], c['gap'], dur), fontsize=9)

        ax = axes[0, col]                                              # нагрузка в окне
        qp = (np.log(r['LS']) - math.log(lo)) / math.log(hi / lo)
        ax.axhspan(0, 1, color=INK, alpha=0.06, zorder=0)
        ax.plot(tt, qp[m], color=COL[0], lw=1.1)
        ax.axhline(q, color=INK, lw=0.7, ls=':')
        ax.set_ylabel('load position q(t)\n(0 = lower, 1 = upper window boundary)')
        ax.set_ylim(-0.5, 1.8)

        ax = axes[1, col]                                              # популяция
        ax.plot(tt, r['DS'][m], color=COL[1], lw=1.3, label='damaged share D')
        ax.plot(tt, r['XS'][m], color=COL[2], lw=1.3, label='working enzyme mean $\\bar{x}$')
        ax.axhline(0.5, color=INK, lw=0.7, ls=':')
        ax.set_ylabel('population state')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=False, fontsize=8, loc='upper left')

        ax = axes[2, col]                                              # защита
        ax.plot(tt, r['CS'][m], color=COL[0], lw=1.3, label='total defence $\\bar{c}$')
        ax.plot(tt, r['SS'][m], color=COL[1], lw=1.3, label='induced protein $\\bar{s}$')
        ax.plot(tt, r['MS'][m], color=COL[2], lw=1.0, ls='--', label='induction signal $\\bar{m}$')
        ax.set_ylabel('defence (means over cells)')
        ax.legend(frameon=False, fontsize=8, loc='upper left')

        ax = axes[3, col]                                              # сон и кора
        tday = np.floor(r['TT'][m] + r['onset']).astype(int)
        days = np.unique(tday)
        wake_h = np.array([24. * r['AW'][m][tday == d].mean() for d in days])
        ax.plot(days - r['onset'], wake_h, color=COL[3], lw=1.4, label='wake, h/day')
        ax.set_ylabel('wake, h/day'); ax.set_ylim(0, 24)
        ax2 = ax.twinx(); ax2.grid(False); ax2.spines['top'].set_visible(False)
        ax2.plot(tt, r['ES'][m] / np.maximum(r['ERS'][m], 1e-9), color=COL[0], lw=1.1,
                 label='cortex $E/E_r$')
        ax2.set_ylabel('cortex $E/E_r$', color=COL[0]); ax2.tick_params(axis='y', colors=COL[0])
        ax2.legend(frameon=False, fontsize=8, loc='lower right')
        ax.legend(frameon=False, fontsize=8, loc='upper right')
        ax.set_xlabel('days relative to episode onset')

    fig.tight_layout(pad=0.9, rect=[0, 0, 1, 0.985])


if __name__ == '__main__':
    main()
