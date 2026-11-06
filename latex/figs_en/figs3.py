# -*- coding: utf-8 -*-
"""Further figures for the brochure: the sleep switch, the population, the conservation law, a summary of results."""
import numpy as np, math, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, SURF, PALE = '#0b0b0b', '#52514e', '#fcfcfb', '#f3f3f1'
LB, LO, LA, LY = '#dbe9f9', '#fbe2d8', '#d6f1e6', '#fbedc9'
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'savefig.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.22})
def save(fig, name): fig.savefig(name, dpi=170, bbox_inches='tight', pad_inches=0.08); plt.close(fig)

# ---- parameters of the wake switch (Phillips & Robinson 2008 / Skeldon 2014) ----
QMAX, THETA, SGM = 100., 10., 3.
NVM, NMV, A_M, NVC, B0, AG = -2.1, -1.8, 1.3, -2.9, -1.253, 6.0
TR, TD = 18.2, 4.2
Q = lambda V: QMAX / (1. + np.exp(-(V - THETA) / SGM))

# ============ Fig. A. The sleep switch: folds and sawtooth ============
def fig_sleepswitch():
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.3))
    a = ax[0]
    Vv = np.linspace(-26, 22, 40001); Dv = Vv - NVM * Q(A_M + NMV * Q(Vv)); qm = Q(A_M + NMV * Q(Vv))
    d1 = np.diff(Dv); s = np.where(d1[:-1] * d1[1:] < 0)[0]; folds = sorted(float(Dv[i + 1]) for i in s)
    up = qm > 2.5; dn = qm < 1.0; mid = (~up) & (~dn)
    a.plot(Dv[up], qm[up], color=BLUE, lw=2.6)
    a.plot(Dv[dn], qm[dn], color=BLUE, lw=2.6)
    a.plot(Dv[mid], qm[mid], color=MUTED, lw=1.6, ls=(0, (4, 3)))
    for f in folds: a.axvline(f, color=MUTED, lw=1.0, ls=':')
    def roots(dv):                                                   # all activity values at a given input
        g = Dv - dv; i = np.where(g[:-1] * g[1:] < 0)[0]; return sorted(float(qm[j]) for j in i)
    rT = roots(folds[1] - 1e-4); rB = roots(folds[0] + 1e-4)
    a.annotate('', xy=(folds[1], min(rT)), xytext=(folds[1], max(rT)), arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=2.4))
    a.annotate('', xy=(folds[0], max(rB)), xytext=(folds[0], min(rB)), arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=2.4))
    a.text(folds[1] + 0.12, 0.5 * (min(rT) + max(rT)), 'sleep onset', color=ORANGE, fontsize=9, va='center')
    a.text(folds[0] - 0.12, 0.5 * (min(rB) + max(rB)), 'waking', color=ORANGE, fontsize=9, va='center', ha='right')
    a.text(4.6, 4.55, 'awake', color=BLUE, fontsize=9.5, ha='right')
    a.text(4.6, 0.25, 'asleep', color=BLUE, fontsize=9.5, ha='right')
    a.text(1.30, 1.55, 'unstable branch', color=MUTED, fontsize=8.5, ha='right')
    a.set_xlim(-1.2, 4.8); a.set_ylim(-0.25, 5.3)
    a.set_xlabel('input to the sleep nucleus, mV'); a.set_ylabel('activity of the wake nuclei, imp/s')
    a.set_title('Folds: %.3f and %.3f mV\n(1.45 and 2.46 in the paper)' % (folds[0], folds[1]), fontsize=10, loc='left', color=INK)

    b = ax[1]
    W_, S_ = 15.5, 8.5; M = 4000; dt = (W_ + S_) / M                 # the same schedule for both curves
    def saw(tr, td, nd=3):
        G = 0.3; t, Gs, aw = [], [], []
        for c in range(nd):
            for i in range(M):
                tt = c * 24. + i * dt; ph = i * dt
                a2 = ph < W_
                G = 1 + (G - 1) * math.exp(-dt / tr) if a2 else G * math.exp(-dt / td)
                t.append(tt / 24.); Gs.append(G); aw.append(a2)
        G = np.array(Gs); k = len(G) - M                      # normalised on the last complete cycle
        return np.array(t), (G - G[k:].min()) / max(np.ptp(G[k:]), 1e-12), np.array(aw)
    t1, G1, aw1 = saw(TR, TD); t2, G2, _ = saw(45., 45.)
    sl = np.where(~aw1)[0]
    for bb in np.split(sl, np.where(np.diff(sl) > 1)[0] + 1):
        if len(bb): b.axvspan(t1[bb[0]], t1[bb[-1]], color=PALE, lw=0, zorder=0)
    b.plot(t1, G1, color=BLUE, lw=2.4, label='18.2 h up / 4.2 h down (Daan)')
    b.plot(t2, G2, color=YELLOW, lw=2.0, ls='--', label='a single constant of 45 h (as before)')
    b.set_xlim(1.0, 3.0); b.set_ylim(-0.06, 1.42); b.set_yticks([0, 0.5, 1.0])
    b.set_xlabel('time, days'); b.set_ylabel('accumulated sleep pressure')
    b.legend(frameon=False, fontsize=8.5, loc='upper center', ncol=1)
    b.set_title('The same schedule, a different shape.\nGrey is sleep: the decay must be faster than the rise', fontsize=10, loc='left', color=INK)
    fig.tight_layout(); save(fig, 'fig_sleepswitch.png')
    return folds

# ============ Fig. B. From one cell to a population ============
def fig_population():
    fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.9))
    a = ax[0]
    u = np.linspace(0, 1, 500); lo, hi = 0.30, 0.70
    xs = np.linspace(0, 1, 400)
    def branch(nu, lo, hi):
        return np.where(nu < lo, 1.0, np.nan), np.where(nu > hi, 0.0, np.nan)
    nu = np.linspace(0.05, 0.95, 400)
    top = np.where(nu < hi, 1 - 0.25 * (nu - 0.05) / 0.9, np.nan)
    bot = np.where(nu > lo, 0.12 + 0.05 * (0.95 - nu) / 0.9, np.nan)
    midb = np.linspace(lo, hi, 100); midv = np.linspace(0.78, 0.17, 100)
    a.plot(nu, top, color=AQUA, lw=2.6); a.plot(nu, bot, color=ORANGE, lw=2.6)
    a.plot(midb, midv, color=MUTED, lw=1.5, ls=(0, (4, 3)))
    a.axvspan(lo, hi, color=LY, lw=0, zorder=0)
    a.text(0.16, 0.93, 'healthy', color=AQUA, fontsize=9.5)
    a.text(0.62, 0.05, 'damaged', color=ORANGE, fontsize=9.5)
    a.text(0.50, 0.33, 'window of\ntwo states', color=MUTED, fontsize=8.5, ha='center')
    a.set_xlabel('load on the cell'); a.set_ylabel('reserve of defence'); a.set_ylim(-0.05, 1.08)
    a.set_title('One cell: two states', fontsize=10, loc='left', color=INK)

    b = ax[1]
    rng = np.random.default_rng(3); J = 24
    v = np.sort(np.exp(0.18 * np.sort(rng.standard_normal(J))))
    thr = lo + (hi - lo) * (v - v.min()) / np.ptp(v) * 0.9 + 0.03
    b.axvspan(lo, hi, color=LY, lw=0, zorder=0)
    for i, th in enumerate(thr):
        b.plot([0.05, th], [i, i], color=LB, lw=3.0, solid_capstyle='butt', zorder=2)
        b.plot([th], [i], 'o', color=BLUE, ms=5, zorder=3)
    b.set_xlim(0.05, 0.95); b.set_ylim(-1.0, J + 0.8); b.set_yticks([])
    b.set_xlabel('load the cell withstands'); b.set_ylabel('24 cells, in order of resilience')
    b.text(0.5, J - 0.3, 'spread of vulnerability: about ±20 %', fontsize=8.5, color=MUTED, ha='center')
    b.set_title('The cells do not break all at once', fontsize=10, loc='left', color=INK)

    c = ax[2]
    nn = np.linspace(0.05, 0.95, 600)
    frac_up = np.array([(thr <= z).mean() for z in nn])
    frac_dn = np.array([(thr <= z + 0.10).mean() for z in nn])
    c.step(nn, frac_up, where='post', color=ORANGE, lw=2.4, label='load rising')
    c.step(nn, frac_dn, where='post', color=AQUA, lw=2.4, ls='--', label='load falling')
    c.axvspan(lo, hi, color=LY, lw=0, zorder=0)
    c.set_xlim(0.05, 0.95); c.set_ylim(-0.04, 1.06)
    c.set_xlabel('load on the population'); c.set_ylabel('fraction of damaged cells')
    c.legend(frameon=False, fontsize=8.5, loc='upper left')
    c.set_title('The population: smooth, but the loop is intact', fontsize=10, loc='left', color=INK)
    fig.tight_layout(); save(fig, 'fig_population.png')

# ============ Fig. C. The conservation law for the gaps ============
def fig_gaps():
    W, FB, PK, SIGF, SIGS = 2.14, 1.192, 1.102, 0.08, 0.08
    TRO = 2 - PK; bud = 2 * SIGF + SIGS
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.2))
    a = ax[0]
    qs = [0.15, 0.50, 0.85]
    for i, q in enumerate(qs):
        x = i * 1.0
        a.add_patch(plt.Rectangle((x - 0.26, 0), 0.52, math.log(W), facecolor=LY, edgecolor=MUTED, lw=1.0))
        y0 = q * math.log(W)
        a.plot([x - 0.26, x + 0.26], [y0, y0], color=BLUE, lw=2.6)
        gu = math.log(W ** (1 - q) / PK); gd = math.log(W ** q * FB * TRO)
        a.annotate('', xy=(x, math.log(W)), xytext=(x, y0), arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=1.6))
        a.annotate('', xy=(x, 0), xytext=(x, y0), arrowprops=dict(arrowstyle='<->', color=AQUA, lw=1.6))
        a.text(x + 0.30, (y0 + math.log(W)) / 2, '%.1f σ' % (gu / SIGF), color=ORANGE, fontsize=9, va='center')
        a.text(x + 0.30, y0 / 2, '%.1f σ' % (gd / SIGF), color=AQUA, fontsize=9, va='center')
        a.text(x, -0.12, 'the load\nis low' if i == 0 else ('in the middle' if i == 1 else 'high'), ha='center', va='top', fontsize=9, color=MUTED)
    a.text(-0.34, math.log(W) * 0.97, 'top of the window', fontsize=8.5, color=MUTED, va='center', ha='right')
    a.text(-0.34, 0.03, 'bottom of the window', fontsize=8.5, color=MUTED, va='center', ha='right')
    a.text(1.0, math.log(W) + 0.10, 'orange: to collapse;  green: to return', ha='center', fontsize=9, color=INK)
    a.set_xlim(-1.55, 2.75); a.set_ylim(-0.42, math.log(W) + 0.22); a.axis('off')
    a.set_title('Shift the load where you will, the sum is the same', fontsize=10, loc='left', color=INK)

    b = ax[1]
    qq = np.linspace(0.02, 0.98, 200)
    gu = np.log(W ** (1 - qq) / PK) / SIGF; gd = np.log(W ** qq * FB * TRO) / SIGF
    b.plot(qq, gu, color=ORANGE, lw=2.4, label='to collapse')
    b.plot(qq, gd, color=AQUA, lw=2.4, label='to return')
    b.plot(qq, gu + gd, color=BLUE, lw=2.6, ls='--', label='the sum is constant')
    b.axhspan(0, 3.0, color=LA, lw=0, zorder=0)
    b.text(0.52, 1.4, 'what the noise can manage', fontsize=9, color=MUTED, ha='center')
    b.annotate('needed for the illness:\n3 σ up and 1.5 σ down,\n4.5 in all', xy=(0.62, 4.6), xytext=(0.70, 7.2),
               fontsize=9, color=INK, ha='center', arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.4))
    b.set_xlabel('where in the window the mean load sits'); b.set_ylabel('size of the jump, in units of the noise (σ)')
    b.set_ylim(0, 11.6); b.legend(frameon=False, fontsize=8.5, loc='upper left', ncol=1)
    b.set_title('The sum is about 9 σ. It does not fit into 4.5', fontsize=10, loc='left', color=INK)
    fig.tight_layout(); save(fig, 'fig_gaps.png')

if __name__ == '__main__':
    f = fig_sleepswitch(); print('folds:', [round(z, 3) for z in f])
    fig_population(); fig_gaps(); print('done')

# ============ Fig. D. Summary of results ============
def fig_results():
    fig, ax = plt.subplots(2, 2, figsize=(12.0, 7.0))
    # (a) where the smoothness comes from
    a = ax[0, 0]
    lab = ['one\ncell', 'population,\nspread\n±5 %', 'spread\n±10 %', 'spread\n±15 %', 'spread\n±25 %']
    val = [1.1, 1.0, 4.0, 10.9, 32.8]; cols = [ORANGE] + [BLUE] * 4
    a.bar(np.arange(5), val, 0.62, color=cols, edgecolor='white')
    for i2, v in enumerate(val): a.text(i2, v + 0.8, '%.1f %%' % v, ha='center', fontsize=9, color=INK)
    a.set_xticks(np.arange(5)); a.set_xticklabels(lab, fontsize=8.5)
    a.set_ylim(0, 38); a.set_ylabel('fraction of time in the middle of the scale, %')
    a.set_title('a. The smoothness comes from the spread of cell\nresilience, not from the population itself', fontsize=10, loc='left', color=INK)
    # (b) the staircase of waking
    b = ax[0, 1]
    kk = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]; wh = [15.50, 16.59, 17.72, 18.87, 19.97, 20.89, 21.60, 22.51]
    b.plot(kk, wh, 'o-', color=BLUE, lw=2.4, ms=6)
    b.axhline(15.5, color=AQUA, lw=1.6, ls='--'); b.text(5.1, 15.5, 'normal, 15.5 h', color=AQUA, fontsize=9, ha='right', va='bottom')
    b.axhline(24, color=ORANGE, lw=1.6, ls=':'); b.text(0.05, 24.3, 'one day', color=ORANGE, fontsize=9)
    b.annotate('beyond this, sleep\nnever comes at all', xy=(4.6, 22.9), xytext=(2.5, 25.6), fontsize=9, color=INK,
               ha='center', arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.3))
    b.set_xlim(-0.2, 5.2); b.set_ylim(14, 27.5)
    b.set_xlabel('how strongly the damage impedes falling asleep, mV'); b.set_ylabel('length of waking, h')
    b.set_title('b. A damaged cell prevents sleep:\n15.5 h → 22.5 h, then insomnia', fontsize=10, loc='left', color=INK)
    # (c) when during the day a collapse occurs
    c = ax[1, 0]
    sw = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]; lt = [0.447, 0.533, 0.658, 0.758, 0.800, 0.806, 0.796]
    c.plot(np.array(sw) * 100, lt, 'o-', color=BLUE, lw=2.4, ms=6)
    c.axhline(0.5, color=MUTED, lw=1.4, ls='--')
    c.text(2, 0.465, 'at random it would be 0.50', color=MUTED, fontsize=9)
    c.set_ylim(0.38, 0.92); c.set_xlabel('daily range of the load, %')
    c.set_ylabel('fraction of collapses in the second half of the day')
    c.set_title('c. Collapses accumulate towards the evening, and\nthe more so the larger the daily range', fontsize=10, loc='left', color=INK)
    # (d) the precursor
    d = ax[1, 1]
    days = np.arange(8) + 0.5
    cur = np.array([1.0562, 1.0322, 1.0272, 1.0257, 1.0215, 1.0183, 1.0177, 1.0193])
    d.bar(days, (cur - 1) * 100, width=0.82, color=BLUE, edgecolor='white')
    d.axhline(0, color=MUTED, lw=1.2)
    d.set_xticks(days); d.set_xticklabels(['0–1', '1–2', '2–3', '3–4', '4–5', '5–6', '6–7', '7–8'], fontsize=8.5)
    d.invert_xaxis(); d.set_ylim(0, 7)
    d.set_xlabel('days before the collapse'); d.set_ylabel('waking longer than usual, %')
    d.set_title('d. Before a collapse waking is longer:\n+2 % over the week, +5.6 % on the last night', fontsize=10, loc='left', color=INK)
    fig.tight_layout(); save(fig, 'fig_results.png')
