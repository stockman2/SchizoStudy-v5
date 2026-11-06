# -*- coding: utf-8 -*-
"""The devil's staircase of the sleep switch, rebuilt from the boundary-refinement run (tongues.csv)."""
import csv, itertools, sys
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, SURF, PALE = '#0b0b0b', '#52514e', '#fcfcfb', '#f3f3f1'
LB, LO, LA, LY = '#dbe9f9', '#fbe2d8', '#d6f1e6', '#fbedc9'
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'savefig.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.22})

SRC = sys.argv[1] if len(sys.argv) > 1 else 'tongues.csv'

def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r['kd'] = float(r['kd'])
        r['rho'] = float(r['rho']) if r['rho'] not in ('', 'nan') else np.nan
        r['pat'] = int(r['pattern'] or 0)
        r['nsl'] = int(r['n_sleep'] or 0)
        r['lock'] = r['n'] != ''
    rows.sort(key=lambda r: r['kd'])
    return rows

def segments(rows):
    """Maximal contiguous intervals carrying one and the same rotation number."""
    out = []
    for key, g in itertools.groupby(rows, key=lambda r: (r['n'], r['m'])):
        g = list(g)
        if key[0] == '':
            continue
        out.append(dict(nm='%s:%s' % key, a=g[0]['kd'], b=g[-1]['kd'],
                        rho=g[0]['rho'], pat=max(x['pat'] for x in g)))
    return out

def draw(rows, name='fig_stairs.png'):
    seg = segments(rows)
    alt = [s for s in seg if s['pat'] > 1]
    lost = min([r['kd'] for r in rows if r['nsl'] == 0], default=np.nan)
    fig, ax = plt.subplots(2, 1, figsize=(11.2, 7.4),
                           gridspec_kw=dict(height_ratios=[1.0, 1.15], hspace=0.34))

    # ---------------- whole interval ----------------
    a = ax[0]
    un = [r for r in rows if not r['lock'] and r['nsl'] > 0]
    a.plot([r['kd'] for r in un], [r['rho'] for r in un], ls='none', marker='.',
           ms=3.2, color='#b9b8b4', zorder=2)
    for s in seg:
        a.plot([s['a'], s['b']], [s['rho']] * 2, color=BLUE, lw=3.4,
               solid_capstyle='butt', zorder=3)
    if np.isfinite(lost):
        a.axvspan(lost, max(r['kd'] for r in rows), color=PALE, lw=0, zorder=0)
        a.text(lost + 0.004, 7.4, 'no sleep', fontsize=9, color=MUTED, va='top')
        a.axvline(lost, color=MUTED, lw=1.0, ls=':')
    for nm, y, dx in (('1:1', 1, -0.02), ('2:1', 2, -0.02), ('3:1', 3, 0.004),
                      ('4:1', 4, 0.004), ('5:1', 5, 0.004), ('6:1', 6, 0.004)):
        ss = [s for s in seg if s['nm'] == nm]
        if ss:
            a.text(ss[0]['b'] + dx, y + 0.18, nm, fontsize=9, color=BLUE,
                   ha='right' if dx < 0 else 'left')
    a.set_xlim(4.40, 5.30); a.set_ylim(0.5, 8.2)
    a.set_xlabel('coupling $kD$, mV'); a.set_ylabel('rotation number, days per cycle')
    a.set_title('The whole interval: the rotation number stands at rational values and jumps',
                fontsize=10.5, loc='left', color=INK)

    # ---------------- close-up ----------------
    b = ax[1]
    LOW, HIGH = 4.78255, 4.78855
    inw = [s for s in seg if s['b'] >= LOW and s['a'] <= HIGH]
    un2 = [r for r in rows if LOW <= r['kd'] <= HIGH and not r['lock'] and r['nsl'] > 0]
    b.plot([r['kd'] for r in un2], [r['rho'] for r in un2], ls='none', marker='.',
           ms=4.5, color='#b9b8b4', zorder=2, label='no repeating pattern')
    first_b, first_o = True, True
    for s in inw:
        c = ORANGE if s['pat'] > 1 else BLUE
        lab = None
        if s['pat'] > 1 and first_o:
            lab, first_o = 'cycle lengths alternate', False
        if s['pat'] == 1 and first_b:
            lab, first_b = 'all cycles equal', False
        w = max(s['b'] - s['a'], 2.5e-5)
        b.plot([max(s['a'] - w * 0.5, LOW), min(s['b'] + w * 0.5, HIGH)], [s['rho']] * 2,
               color=c, lw=4.0, solid_capstyle='butt', zorder=3, label=lab, clip_on=True)
    for nm, dy in (('3:2', 0.035), ('5:4', -0.060), ('4:3', 0.035), ('11:8', -0.060),
                   ('6:5', 0.035), ('2:1', 0.035)):
        ss = [s for s in inw if s['nm'] == nm]
        if ss:
            s = max(ss, key=lambda z: min(z['b'], HIGH) - max(z['a'], LOW))
            xc = 0.5 * (max(s['a'], LOW) + min(s['b'], HIGH))   # centre inside the window only
            b.text(xc, s['rho'] + dy, nm, fontsize=8.6, color=INK, clip_on=True,
                   ha='center', va='bottom' if dy > 0 else 'top')
    b.set_xlim(LOW, HIGH); b.set_ylim(1.10, 2.52)
    b.set_xlabel('coupling $kD$, mV'); b.set_ylabel('rotation number, days per cycle')
    b.legend(frameon=False, fontsize=8.6, loc='upper left', ncol=3, columnspacing=1.4)
    b.set_title('Close-up of the densest region: %d steps in all, %d of them with alternating cycle lengths'
                % (len(seg), len(alt)), fontsize=10.5, loc='left', color=INK)

    # ---------------- inset: the 3:2 pattern ----------------
    row32 = next((r for r in rows if (r['n'], r['m']) == ('3', '2') and r['kd'] > 4.78), None)
    if row32 and row32['tail']:
        tail = [float(z) for z in row32['tail'].split()]
        ins = b.inset_axes([0.645, 0.705, 0.335, 0.255], facecolor='white', zorder=6)
        for sp in ins.spines.values(): sp.set_visible(True); sp.set_color(MUTED); sp.set_linewidth(0.8)
        pat = (tail * 4)[:6]
        ins.bar(np.arange(len(pat)), pat, 0.66, color=[ORANGE if v > 36 else BLUE for v in pat],
                edgecolor='white')
        for i, v in enumerate(pat):
            ins.text(i, v + 1.6, '%.0f' % v, ha='center', fontsize=7.6, color=INK)
        ins.set_ylim(0, 66); ins.set_xticks([]); ins.set_yticks([24, 48])
        ins.set_yticklabels(['24', '48'], fontsize=7.6)
        ins.tick_params(length=0); ins.grid(alpha=0.18)
        ins.set_title('the 3:2 pattern: cycle length, h', fontsize=7.8, color=INK, loc='left', pad=3)
    fig.savefig(name, dpi=170, bbox_inches='tight', pad_inches=0.08); plt.close(fig)
    return len(seg), len(alt), lost

if __name__ == '__main__':
    rows = load(SRC)
    n, a, lost = draw(rows)
    print('steps %d, alternating %d, sleep lost at kD = %.4f mV' % (n, a, lost))
