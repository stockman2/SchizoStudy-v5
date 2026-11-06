# -*- coding: utf-8 -*-
"""English rebuild of the version 4 and 5 result figures (fig_slow, fig_v5).

Layout follows the original Russian figures. Every value is either printed in the
brochure or recovered from the original figure; nothing is simulated. Panel (c) of
fig_slow is redrawn from the boundary-refinement run itself (tongues.csv).
"""
import csv, itertools, sys
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, SURF, PALE = '#0b0b0b', '#52514e', '#fcfcfb', '#f3f3f1'
LB, LO, LA, LY = '#dbe9f9', '#fbe2d8', '#d6f1e6', '#fbedc9'
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'savefig.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.22})
def save(fig, name): fig.savefig(name, dpi=170, bbox_inches='tight', pad_inches=0.08); plt.close(fig)

TAU_M, TAU_S = 12.0, 63.0
SRC = 'tongues.csv'

def staircase(src):
    rows = list(csv.DictReader(open(src)))
    for r in rows:
        r['kd'] = float(r['kd']); r['rho'] = float(r['rho']) if r['rho'] not in ('', 'nan') else np.nan
    rows.sort(key=lambda r: r['kd'])
    seg = []
    for key, g in itertools.groupby(rows, key=lambda r: (r['n'], r['m'])):
        g = list(g)
        if key[0] == '': continue
        seg.append(dict(a=g[0]['kd'], b=g[-1]['kd'], rho=g[0]['rho'], nm='%s:%s' % key,
                        pat=max(int(x['pattern'] or 0) for x in g), tail=g[0]['tail']))
    lost = min([r['kd'] for r in rows if r['n_sleep'] == '0'], default=np.nan)
    return rows, seg, lost

# ===================== fig_slow (version 4) =====================
def fig_slow():
    fig, ax = plt.subplots(2, 2, figsize=(12.2, 7.6))

    # (a) the two stages in series — closed form
    a = ax[0, 0]
    t = np.linspace(0, 240, 3000)
    m = 1 - np.exp(-t / TAU_M)
    s = 1 - (TAU_S * np.exp(-t / TAU_S) - TAU_M * np.exp(-t / TAU_M)) / (TAU_S - TAU_M)
    a.plot(t / 24, m, color=BLUE, lw=2.6, label='induction signal, 12 h')
    a.plot(t / 24, s, color=ORANGE, lw=2.6, label='protective protein, 63 h')
    half = t[np.argmin(np.abs(s - 0.5))]
    a.axhline(0.5, color=MUTED, lw=1.0, ls=':')
    a.plot([half / 24], [0.5], 'o', ms=8, color=ORANGE, mec='white', zorder=5)
    a.annotate('half the level only by hour %.0f' % half, xy=(half / 24, 0.5),
               xytext=(4.0, 0.33), fontsize=9, color=MUTED,
               arrowprops=dict(arrowstyle='-', color=MUTED, lw=1.0))
    a.set_xlim(0, 10); a.set_ylim(-0.03, 1.05)
    a.set_xlabel('days after the load is switched on'); a.set_ylabel('fraction of the steady level')
    a.legend(frameon=False, fontsize=9, loc='lower right')
    a.set_title('a. Two stages of defence: the signal is fast, the protein slow',
                fontsize=10, loc='left', color=INK)

    # (b) the duration law, affine against two saturating forms
    b = ax[0, 1]
    xf = np.array([0.50, 1.00, 2.60, 3.70])         # fit points, memory below 100 h
    yf = np.array([1.75, 2.25, 3.80, 4.80])
    xt, yt = 150 / 24.0, 6.60                        # test point outside the fit
    x = np.linspace(0.30, 7.0, 400)
    hyp = lambda z, T, t0: T * z / (z + t0)
    exps = lambda z, T, t0: T * (1 - np.exp(-z / t0))
    ph, _ = curve_fit(hyp, xf, yf, p0=[8.0, 4.0], maxfev=20000)
    pe, _ = curve_fit(exps, xf, yf, p0=[6.0, 2.0], maxfev=20000)
    b.plot(x, 1.28 + 0.97 * x, color=ORANGE, lw=2.8, label='affine:  $1.28+0.97\\,\\tau$')
    b.plot(x, hyp(x, *ph), color=BLUE, lw=2.0, ls=(0, (6, 3)), label='saturating:  $Tx/(x+t_0)$')
    b.plot(x, exps(x, *pe), color=AQUA, lw=2.0, ls=(0, (2, 2)), label='saturating:  $T(1-e^{-x/t_0})$')
    b.plot(xf, yf, 'o', ms=8, color=INK, zorder=5, label='fit points (memory < 100 h)')
    b.plot([xt], [yt], 's', ms=11, mfc='none', mec=ORANGE, mew=2.2, zorder=5,
           label='test outside the fit: 150 h')
    b.axhline(2.0, color=MUTED, lw=1.0, ls=':')
    b.text(7.15, 2.1, 'clinical target: 2 days', fontsize=8.8, color=MUTED, ha='right')
    b.set_xlim(0, 7.3); b.set_ylim(1.0, 8.6)
    b.set_xlabel('memory of the defence $\\tau_{\\rm d}$, days'); b.set_ylabel('episode duration, days')
    b.legend(frameon=False, fontsize=8.4, loc='upper left')
    b.set_title('b. No saturation: the defence lengthens the episode',
                fontsize=10, loc='left', color=INK)

    # (c) the locking staircase, from the refinement run
    c = ax[1, 0]
    rows, seg, lost = staircase(SRC)
    un = [r for r in rows if r['n'] == '' and r['n_sleep'] != '0']
    c.plot([r['kd'] for r in un], [r['rho'] for r in un], ls='none', marker='.', ms=3,
           color='#b9b8b4', zorder=2)
    for s in seg:
        c.plot([s['a'], s['b']], [s['rho']] * 2, color=ORANGE if s['pat'] > 1 else BLUE,
               lw=3.2, solid_capstyle='butt', zorder=3)
    if np.isfinite(lost):
        c.axvspan(lost, 5.20, color=PALE, lw=0, zorder=0)
        c.text(lost + 0.004, 4.30, 'no sleep\nat all', fontsize=9, color=MUTED, va='top')
    CL, CH = 4.70, 5.20
    for nm, side, dy, col in (('1:1', 'l', 0.10, INK), ('5:4', 'r', -0.20, ORANGE),
                              ('3:2', 'r', 0.09, ORANGE), ('2:1', 'l', 0.10, INK),
                              ('3:1', 'r', 0.10, INK), ('4:1', 'r', 0.10, INK)):
        ss = [s for s in seg if s['nm'] == nm]
        if not ss:
            continue
        s0 = max(ss, key=lambda z: min(z['b'], CH) - max(z['a'], CL))
        if side == 'l':
            c.text(max(s0['a'], CL) + 0.006, s0['rho'] + dy, nm, fontsize=8.8, color=col,
                   ha='left', clip_on=True, fontweight='bold' if col == ORANGE else 'normal')
        else:
            c.text(min(s0['b'], CH) - 0.006, s0['rho'] + dy, nm, fontsize=8.8, color=col,
                   ha='right', clip_on=True, fontweight='bold' if col == ORANGE else 'normal')
    c.set_xlim(4.70, 5.20); c.set_ylim(0.6, 4.7)
    c.set_xlabel('inhibition of the sleep nucleus by damage, $kD$, mV')
    c.set_ylabel('rotation number: days of forcing per cycle')
    c.set_title('c. The locking staircase: the rhythm does not break, it jumps',
                fontsize=10, loc='left', color=INK)
    s54 = next((s for s in seg if s['nm'] == '5:4' and s['tail']), None)
    if s54:
        tail = [float(z) for z in s54['tail'].split()]
        ins = c.inset_axes([0.42, 0.10, 0.55, 0.33], facecolor='white', zorder=6)
        for sp in ins.spines.values(): sp.set_visible(True); sp.set_color(MUTED); sp.set_linewidth(0.8)
        pat = (tail * 3)[:9]
        ins.bar(np.arange(len(pat)), pat, 0.70,
                color=[ORANGE if v > 36 else BLUE for v in pat], edgecolor='white')
        ins.axhline(24, color=MUTED, lw=0.8, ls=':')
        ins.set_ylim(0, 58); ins.set_xticks([]); ins.set_yticks([24, 48])
        ins.set_yticklabels(['24', '48'], fontsize=7.4); ins.tick_params(length=0); ins.grid(False)
        ins.set_title('the 5:4 pattern at $kD=%.3f$ mV: %s h'
                      % (s54['a'], ', '.join('%.0f' % z for z in tail)),
                      fontsize=7.8, color=ORANGE, loc='left', pad=3)

    # (d) J as an effective spread
    d = ax[1, 1]
    J = np.array([8, 16, 32, 64]); sd = np.array([0.1226, 0.1060, 0.0980, 0.0943]); lim = 0.0910
    d.plot(J, sd, 'o-', color=ORANGE, lw=2.4, ms=7, label='realised spread  std$(\\ln v)$')
    d.axhline(0.100, color=BLUE, lw=1.8, ls='--')
    d.text(9, 0.1007, 'specified $\\sigma_v = 0.10$', color=BLUE, fontsize=9, va='bottom')
    d.axhline(lim, color=MUTED, lw=1.2, ls=':')
    d.text(9, lim + 0.0008, 'limit as $J\\to\\infty$: %.4f' % lim, color=MUTED, fontsize=9, va='bottom')
    for x0, y0 in zip(J, sd):
        d.text(x0, y0 + 0.0020, 'J = %d' % x0, ha='center', fontsize=8.6, color=ORANGE)
    d.text(64, 0.1215, 'the ends of the grid are pinned and barely drift:\n'
                       'smallest multiplier 0.8002 at $J=8$, 0.8029 at $J=512$',
           fontsize=8.6, color=AQUA, ha='right', va='top')
    d.set_xscale('log', base=2); d.set_xticks(J); d.set_xticklabels([str(z) for z in J])
    d.set_xlim(7.0, 74); d.set_ylim(0.0885, 0.1265)
    d.set_xlabel('number of cells in the population $J$')
    d.set_ylabel('spread of the multipliers, std$(\\ln v)$')
    d.set_title('d. The third route: $J$ enters as an effective spread',
                fontsize=10, loc='left', color=INK)
    fig.tight_layout(); save(fig, 'fig_slow.png')

if __name__ == '__main__':
    fig_slow(); print('fig_slow ok')

# ===================== fig_v5 (version 5) =====================
def fig_v5():
    fig, ax = plt.subplots(2, 2, figsize=(12.2, 7.6))

    # (a) asymmetric defence: w(r) against the measured level
    a = ax[0, 0]
    r = np.array([0.33, 0.50, 1.00, 2.00, 3.00])
    w = np.array([0.441, 0.544, 0.705, 0.827, 0.878])      # derivation, f = 0.705
    lv = np.array([0.0575, 0.0644, 0.0792, 0.0958, 0.1065])  # measured level of the defence
    err = np.full_like(lv, 0.0075)
    k = (lv[-1] - lv[0]) / (w[-1] - w[0]); b0 = lv[0] - k * w[0]
    xx = np.linspace(0.40, 0.92, 50)
    a.plot(xx, k * xx + b0, color=MUTED, lw=1.6, ls=(0, (6, 4)),
           label='derivation: the level is linear in $w(r)$')
    a.errorbar(w, lv, yerr=err, fmt='o', ms=8, color=ORANGE, ecolor=ORANGE,
               elinewidth=1.6, capsize=4, mec='white', label='measured level of the defence')
    for i in (0, 4):
        a.plot([w[i]], [lv[i]], 'o', ms=18, mfc='none', mec=INK, mew=1.6, zorder=6)
    for x0, y0, rr in zip(w, lv, r):
        a.text(x0 + 0.012, y0 - 0.0045, '$r$ = %.2f' % rr, fontsize=8.8, color=MUTED)
    a.text(0.615, 0.0655, 'circled: the two anchor points;\nthe three middle ones are predicted',
           fontsize=8.6, color=MUTED)
    a.set_xlim(0.395, 0.945); a.set_ylim(0.047, 0.121)
    a.set_xlabel('predicted position  $w(r) = f r/(f r + 1 - f)$')
    a.set_ylabel("level of the defence, own healthy units")
    a.legend(frameon=False, fontsize=8.6, loc='upper left')
    a.set_title('a. Asymmetric defence: the ordering agreed exactly ($\\rho=+1$, $p=0.008$)',
                fontsize=10, loc='left', color=INK)

    # (b) burstiness: the answer depends on WHAT is measured
    b = ax[0, 1]
    lab = ['0.2', '0.5', '1', '2.6', '5', '10']
    i = np.arange(6)
    rate = np.array([1.000, 0.948, 0.862, 0.735, 0.659, 0.596])   # collapse rate, relative
    frac = np.array([1.000, 1.040, 1.045, 1.047, 1.025, 0.982])   # fraction ever damaged
    b.axvspan(1, 3, color=LY, lw=0, zorder=0)
    b.axhline(1.0, color=MUTED, lw=1.0, ls=':')
    b.plot(i, rate, 'o-', color=ORANGE, lw=2.6, ms=7, label='collapse rate (a fit measure)')
    b.plot(i, frac, 's--', color=BLUE, lw=2.2, ms=6, label='fraction ever damaged (saturates)')
    b.text(2, 0.84, 'between the constants\nof the defence: 12 h and 63 h',
           fontsize=8.8, color=MUTED, ha='center')
    b.set_xticks(i); b.set_xticklabels(lab)
    b.set_xlim(-0.3, 5.3); b.set_ylim(0.56, 1.09)
    b.set_xlabel('correlation time of the load, days (mean and variance pinned)')
    b.set_ylabel('in units of the value at 0.2 days')
    b.legend(frameon=False, fontsize=8.8, loc='lower left')
    b.set_title('b. Burstiness of the load: the answer depends on WHAT is measured',
                fontsize=10, loc='left', color=INK)

    # (c) three phases: the band width is not a function of D
    c = ax[1, 0]
    d1 = np.array([0.100, 0.109, 0.125, 0.143, 0.253, 0.288, 0.345])
    b1 = np.array([0.0133, 0.0105, 0.0073, 0.0047, 0.0032, 0.0026, 0.0021])
    d2 = np.array([0.065, 0.077, 0.100, 0.163])
    b2 = np.array([0.0095, 0.0104, 0.0135, 0.0191])
    c.plot(d1, b1, 'o-', color=ORANGE, lw=2.4, ms=6.5,
           label='$D$ changed by the sleep coupling  ($\\rho=-1$)')
    c.plot(d2, b2, 's-', color=BLUE, lw=2.4, ms=6.0,
           label='$D$ changed by the daily range  ($\\rho=+1$)')
    c.plot([0.100], [0.0134], 'o', ms=17, mfc='none', mec=INK, mew=1.6, zorder=6)
    c.annotate('common point', xy=(0.104, 0.0138), xytext=(0.145, 0.0168),
               fontsize=8.8, color=MUTED, arrowprops=dict(arrowstyle='-', color=MUTED, lw=1.0))
    c.set_xlim(0.045, 0.365); c.set_ylim(0.0010, 0.0212)
    c.set_xlabel('fraction of damaged cells $D$'); c.set_ylabel('width of the rescued band')
    c.legend(frameon=False, fontsize=8.6, loc='upper right')
    c.set_title('c. Three phases: the band width is not a function of $D$ — the signs are opposite',
                fontsize=10, loc='left', color=INK)

    # (d) why the 3.3-fold speed-up came from fusing the kernel
    d = ax[1, 1]
    rows = np.array([2000, 5250, 12000])
    npy = np.array([7.4, 14.7, 14.6]); fused = np.array([6.3, 5.9, 5.9])
    d.axvspan(2000, 5250, color=PALE, lw=0, zorder=0)
    d.plot(rows, npy, 'o-', color=BLUE, lw=2.6, ms=7, label='one operation at a time (numpy)')
    d.plot(rows, fused, 's-', color=ORANGE, lw=2.6, ms=6.5, label='fused kernel')
    for x0, y0 in zip(rows, npy):
        d.text(x0, y0 + 0.55, '%.1f' % y0, ha='center', fontsize=8.8, color=BLUE)
    for x0, y0 in zip(rows, fused):
        d.text(x0, y0 - 1.15, '%.1f' % y0, ha='center', fontsize=8.8, color=ORANGE)
    d.text(3625, 1.3, 'the working set stopped\nfitting in cache', fontsize=8.8,
           color=MUTED, ha='center')
    d.set_xlim(1200, 12900); d.set_ylim(0, 17.2)
    d.set_xlabel('rows in the process'); d.set_ylabel('cost per row, ms per step')
    d.legend(frameon=False, fontsize=8.8, loc='center right')
    d.set_title('d. Why the 3.3-fold speed-up came precisely from fusing',
                fontsize=10, loc='left', color=INK)
    fig.tight_layout(); save(fig, 'fig_v5.png')
