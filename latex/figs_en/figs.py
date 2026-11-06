# -*- coding: utf-8 -*-
"""Figures for the brochure: our own diagrams (not copies of anyone else's figures)."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Ellipse, Rectangle, Arc

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, SURF, PALE = '#0b0b0b', '#52514e', '#fcfcfb', '#f3f3f1'
LB, LO, LA, LY = '#dbe9f9', '#fbe2d8', '#d6f1e6', '#fbedc9'
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'savefig.facecolor': 'white'})

def box(ax, x, y, w, h, text, fc=PALE, ec=MUTED, fs=10, tc=INK, lw=1.2, ls='-', bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.0,rounding_size=0.12', fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=tc, zorder=3, fontweight='bold' if bold else 'normal', linespacing=1.25)

def arr(ax, p1, p2, color=MUTED, inhibit=False, rad=0.0, lw=1.8, ls='-', z=1):
    st = '-[, widthB=0.45, lengthB=0.0' if inhibit else '-|>, head_length=0.6, head_width=0.3'
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=st, connectionstyle='arc3,rad=%g' % rad, color=color, lw=lw, ls=ls, zorder=z, shrinkA=2, shrinkB=2, mutation_scale=12))

def canvas(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, xl); ax.set_ylim(0, yl); ax.set_aspect('equal'); ax.axis('off'); return fig, ax

def save(fig, name):
    fig.savefig(name, dpi=170, bbox_inches='tight', pad_inches=0.08); plt.close(fig)

# ---------------- Fig. 1. Time scales ----------------
def fig_times():
    H, D, W, M, Y = 3600.0, 86400.0, 604800.0, 2.63e6, 3.156e7
    rows = [('neuronal impulse', 0.8e-3, 2e-3, BLUE, ''), ('inhibition via a PV cell (delay)', 1e-3, 2e-3, BLUE, ''), ('one period of the gamma rhythm (30–80 Hz)', 0.0125, 0.033, BLUE, ''),
            ('spontaneous fluctuations of brain activity', 2e-3, 330, BLUE, 'no preferred scale'),
            ('return of ordinarily oxidised peroxiredoxin', 1, 10, AQUA, ''), ('a hallucination episode', 1, 600, YELLOW, ''),
            ('repair of hyperoxidised peroxiredoxin', 180, 4.7 * H, AQUA, ''), ('GAD67 and PV halved under ketamine', 4 * H, 8 * H, ORANGE, ''),
            ('day: sleep and waking,\nrhythm of the oxidative state', 0.85 * D, 1.18 * D, AQUA, 'external rhythm'),
            ('glutathione-synthesis enzyme: −50 %\nafter receptor blockade', 0.8 * D, 1.25 * D, ORANGE, ''), ('glutathione turnover in the brain (half-life)', 60 * H, 80 * H, AQUA, ''),
            ('acute onset of psychosis', 1 * D, 7 * D, YELLOW, ''), ('PV loss after two days of ketamine:\nnot resolved by day 10', 8 * D, 12 * D, ORANGE, ''),
            ('salience → paranoia (lead time)', 1.6 * W, 2.4 * W, YELLOW, ''), ('rise of dopamine synthesis before psychosis', 3 * M, 3 * Y, YELLOW, ''),
            ('prodrome (before the first episode)', 4 * M, 100 * M, YELLOW, ''), ('maturation of inhibitory cells and the nets\naround them (adolescence)', 2 * Y, 10 * Y, ORANGE, '')]
    fig, ax = plt.subplots(figsize=(11, 7.2)); n = len(rows)
    for i, (lab, a, b, c, note) in enumerate(rows):
        y = n - 1 - i; ax.barh(y, np.log10(b) - np.log10(a), left=np.log10(a), height=0.58, color=c, edgecolor='white', lw=1.5, zorder=3)
        ax.text(np.log10(a) - 0.15, y, lab, ha='right', va='center', fontsize=8.8, color=INK, linespacing=1.1)
        if note: ax.text(np.log10(b) + 0.15, y, note, ha='left', va='center', fontsize=8, color=MUTED, style='italic')
    tk = [(1e-3, '1 ms', 0), (1, '1 s', 0), (60, '1 min', 0), (H, '1 h', 0), (D, '1 day', 0), (W, '1 week', 1), (M, '1 month', 0), (Y, '1 year', 1), (10 * Y, '10 years', 0)]
    ax.set_xticks([np.log10(t) for t, _, _ in tk]); ax.set_xticklabels([]); ax.set_yticks([]); ax.set_xlim(-10.3, 12.2); ax.set_ylim(-1.9, n - 0.3)
    for t, l, lev in tk: ax.plot([np.log10(t)] * 2, [-0.75, n - 0.5], color='#d9d8d4', lw=0.7, zorder=1); ax.text(np.log10(t), -1.05 - 0.55 * lev, l, ha='center', va='top', fontsize=8.8, color=INK)
    for sp in ('top', 'right', 'left', 'bottom'): ax.spines[sp].set_visible(False)
    ax.plot([-3, np.log10(10 * Y)], [-0.75, -0.75], color=INK, lw=1); ax.tick_params(axis='x', length=0)
    hs = [Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, AQUA, ORANGE, YELLOW)]
    ax.legend(hs, ['network activity', 'oxidative defence of the cell', 'protein expression', 'symptoms and course of the illness'], frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.06), fontsize=9, handlelength=1.2)
    save(fig, 'fig_times.png')

# ---------------- Fig. 2. Neuron and synapse ----------------
def fig_neuron():
    fig, ax = canvas(11, 4.6, 22, 9.2)
    # neuron on the left
    sx, sy = 3.2, 5.0
    for ang in (100, 140, 180, 220, 255, 60):
        a = np.deg2rad(ang); x1, y1 = sx + 2.3 * np.cos(a), sy + 2.3 * np.sin(a); ax.plot([sx, x1], [sy, y1], color=BLUE, lw=3, solid_capstyle='round', zorder=1)
        for da in (-25, 25): b = np.deg2rad(ang + da); ax.plot([x1, x1 + 0.9 * np.cos(b)], [y1, y1 + 0.9 * np.sin(b)], color=BLUE, lw=1.8, solid_capstyle='round', zorder=1)
    ax.add_patch(Circle((sx, sy), 0.85, fc=LB, ec=BLUE, lw=2, zorder=2)); ax.add_patch(Circle((sx, sy), 0.3, fc=BLUE, ec='none', zorder=3))
    ax.plot([sx + 0.85, 9.0], [sy - 0.1, 3.3], color=BLUE, lw=2.6, zorder=1); ax.add_patch(Circle((9.15, 3.25), 0.3, fc=BLUE, ec='none', zorder=2))
    ax.text(0.7, 8.3, 'dendrites:\nsignals arrive here', fontsize=9, color=INK, ha='left'); ax.text(5.2, 1.5, 'cell body\n(nucleus, protein synthesis)', fontsize=9, color=INK, ha='left', va='center'); ax.plot([3.6, 5.1], [4.3, 1.9], color=MUTED, lw=0.8, ls=':')
    ax.text(6.0, 4.75, 'axon: the electrical\nimpulse leaves along it', fontsize=9, color=INK, ha='left'); ax.text(9.2, 2.45, 'terminal\n(synapse)', fontsize=9, color=INK, ha='center', va='top')
    ax.add_patch(Rectangle((8.55, 2.65), 1.2, 1.2, fc='none', ec=MUTED, lw=1, ls='--')); ax.plot([9.75, 12.2], [3.85, 8.6], color=MUTED, lw=0.8, ls='--'); ax.plot([9.75, 12.2], [2.65, 0.6], color=MUTED, lw=0.8, ls='--')
    # synapse on the right (enlarged)
    ax.add_patch(Rectangle((12.2, 0.6), 9.5, 8.0, fc=SURF, ec=MUTED, lw=1, ls='--', zorder=0))
    ax.add_patch(FancyBboxPatch((13.6, 5.4), 6.7, 2.8, boxstyle='round,pad=0,rounding_size=0.9', fc=LB, ec=BLUE, lw=2)); ax.text(16.95, 7.7, 'axon terminal (transmitting cell)', fontsize=8, ha='center', color=INK)
    for vx, vy in ((14.9, 6.5), (16.2, 6.15), (17.6, 6.45), (18.9, 6.2), (15.6, 7.0), (18.2, 7.05)):
        ax.add_patch(Circle((vx, vy), 0.33, fc='white', ec=BLUE, lw=1.3)); [ax.add_patch(Circle((vx + dx, vy + dy), 0.06, fc=ORANGE, ec='none')) for dx, dy in ((-0.1, 0.08), (0.1, 0.05), (0.0, -0.12))]
    for dx, dy in ((15.4, 5.0), (16.3, 4.75), (17.1, 5.05), (17.9, 4.7), (18.6, 5.0), (16.8, 4.55)): ax.add_patch(Circle((dx, dy), 0.08, fc=ORANGE, ec='none'))
    ax.text(20.4, 4.85, 'transmitter\nmolecules', fontsize=8.5, color=ORANGE, ha='center', va='center')
    ax.add_patch(FancyBboxPatch((13.6, 1.0), 6.7, 3.0, boxstyle='round,pad=0,rounding_size=0.9', fc=LA, ec=AQUA, lw=2)); ax.text(16.95, 1.35, 'receiving cell', fontsize=9, ha='center', color=INK)
    for rx, col, nm in ((15.3, ORANGE, 'AMPA'), (17.0, YELLOW, 'NMDA'), (18.7, ORANGE, 'AMPA')):
        ax.add_patch(Rectangle((rx - 0.38, 3.55), 0.3, 0.9, fc=col, ec=INK, lw=0.8, zorder=4)); ax.add_patch(Rectangle((rx + 0.08, 3.55), 0.3, 0.9, fc=col, ec=INK, lw=0.8, zorder=4)); ax.text(rx, 3.1, nm, fontsize=8.5, ha='center', color=INK, zorder=5)
    ax.text(16.95, 2.3, 'receptors: gate proteins in the membrane;\nthe transmitter opens them and ions enter', fontsize=8.3, ha='center', color=INK)
    ax.text(12.5, 4.7, 'cleft\n20–40 nm', fontsize=8.5, color=MUTED, ha='left', va='center')
    save(fig, 'fig_neuron.png')

# ---------------- Fig. 3. The cortical microcircuit ----------------
def fig_circuit():
    fig, ax = canvas(11, 5.2, 22, 10.4)
    for px in (2.4, 5.4): ax.add_patch(Polygon([[px - 0.9, 3.6], [px + 0.9, 3.6], [px, 5.5]], closed=True, fc=LB, ec=BLUE, lw=2, zorder=3)); ax.plot([px, px], [5.5, 7.4], color=BLUE, lw=2.5, zorder=2); ax.plot([px, px], [3.6, 2.0], color=BLUE, lw=1.6, zorder=2)
    ax.text(3.9, 1.6, 'pyramidal (excitatory) cells:\nabout 80 % of cortical neurons,\ntransmitter: glutamate', fontsize=8.8, ha='center', va='top', color=INK)
    cx, cy = 10.2, 5.6; ax.add_patch(Circle((cx, cy), 1.5, fc='none', ec=AQUA, lw=2, ls=(0, (2, 2)), zorder=2)); ax.add_patch(Circle((cx, cy), 1.0, fc=LO, ec=ORANGE, lw=2, zorder=3)); ax.text(cx, cy, 'PV', fontsize=12, ha='center', va='center', color=INK, fontweight='bold', zorder=4)
    ax.text(cx, 9.55, 'inhibitory PV cell:\ntransmitter: GABA', fontsize=8.8, ha='center', va='top', color=INK); ax.text(cx, cy - 1.75, 'dashed: the net around the cell\n(protective shell)', fontsize=8.3, color=AQUA, ha='center', va='top')
    arr(ax, (3.3, 4.3), (9.15, 5.2), BLUE, rad=-0.1, lw=2.2); arr(ax, (6.3, 4.2), (9.2, 4.95), BLUE, rad=0.08, lw=2.2); ax.text(7.75, 2.75, 'excitation: glutamate\nopens AMPA and NMDA', fontsize=8.3, color=BLUE, ha='center', va='top')
    arr(ax, (9.25, 6.1), (2.95, 4.9), ORANGE, inhibit=True, rad=0.3, lw=2.2); arr(ax, (9.35, 6.35), (5.95, 4.95), ORANGE, inhibit=True, rad=0.25, lw=2.2); ax.text(4.2, 8.55, 'inhibition (GABA) onto the cell body,\ndelay no more than 2 ms', fontsize=8.3, color=ORANGE, ha='center')
    for yy, txt, fc, ec in ((8.7, 'parvalbumin (PV): binds calcium and removes\nit quickly after every impulse', LO, ORANGE), (6.75, 'the enzyme GAD67: makes GABA;\nless GAD67 means weaker inhibition', LO, ORANGE), (4.8, 'NMDA receptor: its activity maintains production\nof PV, GAD67 and the protective enzymes', LY, YELLOW)):
        box(ax, 17.3, yy, 9.0, 1.5, txt, fc=fc, ec=ec, fs=8.3); ax.plot([11.3, 12.8], [cy + (yy - 6.75) * 0.25, yy], color=MUTED, lw=0.8, ls=':')
    t = np.linspace(0, 1, 400); x0, y0, wdt = 14.6, 1.75, 6.9
    ax.plot(x0 + wdt * t, y0 + 0.55 * np.maximum(np.sin(2 * np.pi * 4 * t), 0) ** 3 + 0.8, color=BLUE, lw=1.6); ax.plot(x0 + wdt * t, y0 + 0.55 * np.maximum(np.sin(2 * np.pi * 4 * t - 0.9), 0) ** 3, color=ORANGE, lw=1.6)
    ax.text(x0 - 0.2, y0 + 1.0, 'pyramidal', fontsize=8, color=BLUE, ha='right'); ax.text(x0 - 0.2, y0 + 0.2, 'PV cells', fontsize=8, color=ORANGE, ha='right')
    ax.text(17.3, 1.3, 'gamma rhythm: excitation and inhibition\nalternate 30–80 times a second', fontsize=8.3, ha='center', va='top', color=INK)
    save(fig, 'fig_circuit.png')

# ---------------- Fig. 4. Peroxide and defence ----------------
def fig_redox():
    fig, ax = canvas(11, 6.3, 22, 12.6)
    box(ax, 3.0, 10.9, 5.0, 1.5, 'mitochondria\n(energy production)', fc=PALE, fs=9); box(ax, 3.0, 8.5, 5.0, 1.5, 'NADPH oxidase (Nox2),\nswitched on by the IL-6 signal', fc=PALE, fs=8.8)
    box(ax, 8.8, 9.7, 3.0, 1.7, 'hydrogen\nperoxide\nH₂O₂', fc=LY, ec=YELLOW, fs=10, bold=True); arr(ax, (5.5, 10.9), (7.3, 10.1), MUTED); arr(ax, (5.5, 8.5), (7.3, 9.3), MUTED)
    ax.text(3.0, 12.05, 'sources', fontsize=9, color=MUTED, ha='center', style='italic'); ax.text(16.8, 12.05, 'clearance routes', fontsize=9, color=MUTED, ha='center', style='italic')
    box(ax, 16.8, 10.9, 9.4, 1.3, 'catalase: decomposes peroxide unaided', fc=LA, ec=AQUA, fs=8.8); arr(ax, (10.3, 10.2), (12.1, 10.9), AQUA)
    box(ax, 16.8, 8.9, 9.4, 1.6, 'glutathione peroxidase: spends glutathione (GSH),\nwhich another enzyme then restores', fc=LA, ec=AQUA, fs=8.5); arr(ax, (10.3, 9.5), (12.1, 9.0), AQUA)
    ax.add_patch(FancyBboxPatch((5.6, 0.4), 16.0, 7.1, boxstyle='round,pad=0,rounding_size=0.3', fc=SURF, ec=MUTED, lw=1, ls='--', zorder=0)); ax.text(15.6, 7.05, 'peroxiredoxin: the principal peroxide scavenger', fontsize=9.5, ha='center', color=INK, fontweight='bold')
    box(ax, 8.4, 4.4, 3.4, 1.2, 'working\n(Prx–SH)', fc=LB, ec=BLUE, fs=9); box(ax, 13.7, 4.4, 3.6, 1.2, 'oxidised\n(Prx–SOH)', fc=LB, ec=BLUE, fs=9); box(ax, 19.0, 4.4, 3.4, 1.2, 'cross-linked\n(Prx–S–S)', fc=LB, ec=BLUE, fs=9)
    arr(ax, (10.1, 4.4), (11.9, 4.4), BLUE); ax.text(11.0, 3.75, '+ H₂O₂', fontsize=8.5, color=BLUE, ha='center'); arr(ax, (15.5, 4.4), (17.3, 4.4), BLUE)
    arr(ax, (19.0, 5.0), (8.4, 5.0), BLUE, rad=0.22); ax.text(13.7, 5.55, 'return by thioredoxin: seconds', fontsize=8.5, color=BLUE, ha='center')
    box(ax, 13.7, 1.6, 4.8, 1.3, 'hyperoxidised, switched off\n(Prx–SO₂H)', fc=LO, ec=ORANGE, fs=9); arr(ax, (13.7, 3.8), (13.7, 2.25), ORANGE); ax.text(13.95, 3.0, '+ a second H₂O₂', fontsize=8.5, color=ORANGE, ha='left')
    arr(ax, (11.3, 1.6), (8.4, 3.8), AQUA, rad=-0.25); ax.text(8.5, 1.35, 'repair by sulfiredoxin:\nminutes to hours, needs glutathione', fontsize=8.3, color=AQUA, ha='center', va='top')
    arr(ax, (8.8, 8.8), (10.9, 4.6), YELLOW, rad=0.08, ls=(0, (3, 2)))
    ax.text(19.0, 1.7, 'the more peroxide,\nthe more enzyme is\nswitched off and the\nworse the clearance:\nthe loop feeds itself', fontsize=8.3, color=INK, ha='center', va='center')
    save(fig, 'fig_redox.png')

# ---------------- Fig. 5. Vicious circles and the developmental window ----------------
def fig_loops():
    fig, ax = canvas(11, 6.8, 22, 13.6)
    box(ax, 4.2, 10.6, 6.4, 1.6, 'the NMDA receptor on the PV cell\nworks weakly', fc=LY, ec=YELLOW, fs=9.3); box(ax, 17.3, 10.6, 6.2, 1.6, 'oxidative stress\n(excess peroxide)', fc=LO, ec=ORANGE, fs=9.3)
    box(ax, 10.75, 12.7, 8.6, 1.2, 'fewer protective enzymes and less glutathione', fc=LA, ec=AQUA, fs=9); arr(ax, (4.6, 11.4), (6.4, 12.5), MUTED); arr(ax, (15.1, 12.5), (16.9, 11.4), MUTED)
    arr(ax, (14.2, 10.6), (7.4, 10.6), ORANGE); ax.text(10.8, 10.0, 'peroxide reversibly\nsuppresses the receptor', fontsize=8.5, color=ORANGE, ha='center', va='top')
    box(ax, 17.3, 7.2, 6.2, 1.9, 'the PV cell weakens:\nless PV and GAD67, and the net\naround the cell suffers', fc=LO, ec=ORANGE, fs=9); arr(ax, (17.3, 9.8), (17.3, 8.15), MUTED)
    box(ax, 4.2, 7.2, 6.4, 1.6, 'disinhibition: pyramidal\ncells fire more strongly', fc=LB, ec=BLUE, fs=9); arr(ax, (14.2, 7.2), (7.4, 7.2), MUTED); ax.text(10.8, 7.5, 'on one hypothesis (GAD67)', fontsize=8, color=MUTED, ha='center', style='italic')
    box(ax, 10.75, 4.7, 7.0, 1.4, 'IL-6 signal → NADPH oxidase →\nyet more peroxide', fc=PALE, fs=9); arr(ax, (4.2, 6.4), (7.2, 4.9), MUTED, rad=0.2)
    ax.plot([14.25, 21.4, 21.4], [4.7, 4.7, 10.6], color=MUTED, lw=1.8, solid_capstyle='round', solid_joinstyle='round'); arr(ax, (21.4, 10.6), (20.4, 10.6), MUTED)
    ax.text(17.3, 5.45, 'every loop feeds itself', fontsize=9, color=INK, ha='center', style='italic')
    y = 1.9; ax.plot([1.0, 21.0], [y, y], color=INK, lw=1.5); ax.add_patch(Rectangle((9.2, y - 0.35), 5.4, 0.7, fc=LO, ec=ORANGE, lw=1.2, zorder=2)); ax.text(11.9, y, 'window of vulnerability', fontsize=9, ha='center', va='center', color=INK, zorder=3)
    for xx, lab in ((1.0, 'birth'), (5.2, 'childhood'), (9.2, 'adolescence'), (14.6, 'young adulthood'), (17.6, 'first episode\n(usually at 18–30)')):
        ax.plot([xx, xx], [y - 0.2, y + 0.2], color=INK, lw=1.2); ax.text(xx, y - 0.55, lab, fontsize=8.5, ha='center', va='top', color=INK)
    ax.text(11.9, y + 0.85, 'PV cells and the nets around them are still maturing; until then the cell is less well protected', fontsize=8.5, ha='center', color=MUTED); ax.add_patch(Circle((17.6, y), 0.18, fc=YELLOW, ec=INK, zorder=4))
    save(fig, 'fig_loops.png')

# ---------------- Fig. 6. The dopamine pathway ----------------
def fig_dopamine():
    fig, ax = canvas(11, 4.3, 22, 8.6)
    xs = [2.6, 7.4, 12.2, 17.0]; labs = ['ventral\nhippocampus', 'nucleus\naccumbens', 'ventral\npallidum', 'ventral tegmental\narea (VTA):\ndopamine cells']
    for x, l, fc, ec in zip(xs, labs, (LB, LB, LO, LY), (BLUE, BLUE, ORANGE, YELLOW)): box(ax, x, 5.6, 4.0, 2.0, l, fc=fc, ec=ec, fs=9)
    arr(ax, (4.6, 5.6), (5.4, 5.6), BLUE, lw=2.4); arr(ax, (9.4, 5.6), (10.2, 5.6), ORANGE, inhibit=True, lw=2.4); arr(ax, (14.2, 5.6), (15.0, 5.6), ORANGE, inhibit=True, lw=2.4)
    ax.text(5.0, 6.9, 'excites', fontsize=8.5, color=BLUE, ha='center'); ax.text(9.8, 6.9, 'inhibits', fontsize=8.5, color=ORANGE, ha='center'); ax.text(14.6, 6.9, 'inhibits', fontsize=8.5, color=ORANGE, ha='center')
    box(ax, 17.0, 1.7, 4.6, 1.6, 'striatum:\ndopamine release', fc=LY, ec=YELLOW, fs=9); arr(ax, (17.0, 4.6), (17.0, 2.5), YELLOW, lw=2.4)
    box(ax, 2.9, 1.7, 5.4, 1.6, 'PV cells weakened:\nthe hippocampus\nis over-excited', fc=PALE, fs=8.8); arr(ax, (2.6, 2.5), (2.6, 4.6), MUTED)
    ax.text(10.15, 2.75, 'two inhibitions in series give disinhibition:\nthe more active the hippocampus, the more dopamine\ncells are ready to respond to a signal (normally about\nhalf; in the rat model, almost all)', fontsize=8.5, color=INK, ha='center', va='center')
    ax.text(20.1, 1.7, 'antipsychotics\nblock the D2\nreceptors here', fontsize=8.5, color=MUTED, ha='left', va='center', style='italic')
    save(fig, 'fig_dopamine.png')

# ---------------- Fig. 7. Biology and the variables of the model ----------------
def fig_model():
    fig, ax = canvas(11, 5.6, 22, 11.2)
    K, S, Dsh, dash = AQUA, BLUE, ORANGE, (0, (4, 2))   # derived / link from papers / assumption
    box(ax, 3.4, 8.2, 5.0, 1.9, 'ν — the load:\nperoxide influx', fc=PALE, fs=9.8); box(ax, 11.0, 8.2, 4.0, 1.9, 'u\nperoxide', fc=LY, ec=YELLOW, fs=11.5, bold=True); box(ax, 18.6, 8.2, 5.4, 1.9, 'N — availability of\nthe NMDA receptor', fc=LB, ec=BLUE, fs=9.8)
    box(ax, 11.0, 3.0, 5.6, 2.1, 'x — fraction of working\nperoxiredoxin\n(minutes to hours)', fc=LA, ec=AQUA, fs=9.8); box(ax, 18.6, 3.0, 5.4, 2.1, 'c — defence: repair\nand clearance\n(hours to days)', fc=LO, ec=ORANGE, fs=9.8)
    box(ax, 3.4, 3.0, 5.0, 1.9, 'random fluctuations\nand the daily rhythm\nof the load', fc='white', ec=MUTED, fs=9.3, ls=dash)
    arr(ax, (5.95, 8.2), (8.95, 8.2), K); ax.text(7.45, 8.55, 'influx', fontsize=8.8, color=K, ha='center')
    arr(ax, (3.4, 4.0), (3.4, 7.2), Dsh, ls=dash)
    arr(ax, (10.3, 4.1), (10.3, 7.2), K, inhibit=True); ax.text(10.05, 5.65, 'x removes\nperoxide', fontsize=8.8, color=K, ha='right', va='center')
    arr(ax, (11.7, 7.2), (11.7, 4.1), K, inhibit=True); ax.text(11.95, 5.5, 'excess\nswitches x off', fontsize=8.8, color=K, ha='left', va='center')
    arr(ax, (13.05, 8.55), (15.85, 8.55), S, inhibit=True); ax.text(14.45, 8.9, 'suppresses', fontsize=8.8, color=S, ha='center')
    arr(ax, (18.6, 7.2), (18.6, 4.1), S); ax.text(18.85, 5.65, 'receptor activity\nmaintains the\ndefence (slowly)', fontsize=8.8, color=S, ha='left', va='center')
    arr(ax, (15.85, 3.0), (13.85, 3.0), K); ax.text(14.85, 3.35, 'repairs', fontsize=8.8, color=K, ha='center')
    arr(ax, (13.05, 7.5), (16.5, 4.15), Dsh, ls=dash); ax.text(16.45, 6.45, 'stress switches\non defence', fontsize=8.3, color=Dsh, ha='left', va='center')
    hs = [plt.Line2D([0], [0], color=K, lw=2), plt.Line2D([0], [0], color=S, lw=2), plt.Line2D([0], [0], color=Dsh, lw=2, ls=dash)]
    ax.legend(hs, ['derived from the measured kinetics of the enzyme', 'the link is described in papers; the functional form is an assumption', 'assumed form; the link itself is described for other cells or tissues'], frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.08), ncol=1, fontsize=9)
    save(fig, 'fig_model.png')

# ---------------- Fig. The whole chain ----------------
def fig_overview():
    fig, ax = canvas(11, 5.9, 22, 11.8)
    xs = [2.0, 5.6, 9.2, 12.8, 16.4, 20.0]; y = 6.6; w, h = 3.2, 3.0
    items = [('Weak\nNMDA receptor\n(genes and\nenvironment)', LB, BLUE), ('Fewer\nprotective\nenzymes', LA, AQUA), ('Oxidants\naccumulate', LY, YELLOW), ('The PV cell\nfalls silent:\nless PV\nand GAD67', LO, ORANGE), ('Inhibition\nweakens:\nthe cortex is\nover-excited', LB, BLUE), ('Dopamine\namplifies\nrandom\nsignals', LY, YELLOW)]
    for x, (t, fc, ec) in zip(xs, items): box(ax, x, y, w, h, t, fc=fc, ec=ec, fs=9.0)
    for a, b in zip(xs[:-1], xs[1:]): arr(ax, (a + w / 2, y), (b - w / 2, y), MUTED)
    box(ax, 20.0, 2.2, 3.1, 1.5, 'psychosis', fc='white', ec=INK, fs=11, bold=True); arr(ax, (20.0, y - h / 2), (20.0, 2.95), MUTED)
    arr(ax, (9.2, y + h / 2), (2.0, y + h / 2), ORANGE, inhibit=True, rad=0.32); ax.text(5.6, 10.45, 'oxidants suppress the receptor: the first circle', fontsize=9, color='#c9501f', ha='center')
    arr(ax, (16.4, y - h / 2), (9.2, y - h / 2), ORANGE, rad=-0.32); ax.text(12.8, 2.85, 'over-excitation breeds new oxidants:\nthe second circle', fontsize=9, color='#c9501f', ha='center', va='center')
    for x0, x1, t in [(0.45, 10.75, 'inside one cell: minutes, hours, days'), (11.25, 17.95, 'a network of cells: milliseconds'), (18.45, 21.55, 'symptoms')]:
        ax.plot([x0, x1], [0.75, 0.75], color=MUTED, lw=1.2); ax.plot([x0, x0], [0.75, 1.0], color=MUTED, lw=1.2); ax.plot([x1, x1], [0.75, 1.0], color=MUTED, lw=1.2); ax.text((x0 + x1) / 2, 0.3, t, fontsize=8.8, color=MUTED, ha='center', va='center')
    save(fig, 'fig_overview.png')

# ---------------- Fig. The switch ----------------
def fig_switch():
    eps, mu = 0.02, 1.0; u = np.logspace(-2.5, 2.5, 20000); x = 1 / (1 + mu * u * u / (1 + u)); nu = eps * u + x * u / (1 + u); d = 1 - x; neg = np.where(np.gradient(nu, u) < 0)[0]; i1, i2 = neg[0], neg[-1] + 1; hi, lo = nu[i1], nu[i2]
    dup = d[i2 + np.argmin(np.abs(nu[i2:] - hi))]; ddn = d[np.argmin(np.abs(nu[:i1] - lo))]; W = hi - lo
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.2)); a = ax[0]
    a.axvspan(lo, hi, color='#eeeeec', lw=0, zorder=0); a.plot(nu[:i1], d[:i1], color=AQUA, lw=2.8); a.plot(nu[i1:i2], d[i1:i2], color=MUTED, lw=1.6, ls=(0, (4, 3))); a.plot(nu[i2:], d[i2:], color=ORANGE, lw=2.8)
    a.set_xlim(lo - 1.5 * W, hi + 1.5 * W); a.set_ylim(-0.03, 1.12); a.set_xticks([]); a.set_yticks([0, 0.5, 1]); a.set_xlabel('load: peroxide influx →'); a.set_ylabel('switched-off fraction of the protective enzyme')
    a.annotate('', xy=(hi, dup - 0.025), xytext=(hi, d[i1] + 0.025), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.3)); a.annotate('', xy=(lo, ddn + 0.025), xytext=(lo, d[i2] - 0.025), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.3))
    a.text(lo - 1.35 * W, 0.27, 'defence working', color='#12875c', fontsize=9.8, fontweight='bold'); a.text(hi + 0.60 * W, 0.86, 'defence switched off', color='#c9501f', fontsize=9.4, fontweight='bold', ha='center')
    a.text((lo + hi) / 2, 0.52, 'unstable\nbranch', color=MUTED, fontsize=8.6, ha='center', va='center', bbox=dict(fc='#eeeeec', ec='none', pad=1.0))
    a.text(hi + 0.07 * W, 0.62, 'collapse', fontsize=9.3, color=INK, va='center'); a.text(lo - 0.07 * W, 0.45, 'return', fontsize=9.3, color=INK, ha='right', va='center'); a.text((lo + hi) / 2, 1.07, 'window of two states', fontsize=9.3, color=MUTED, ha='center', va='center')
    a.set_title('One and the same load, two possible states', fontsize=10.5, loc='left')
    b = ax[1]; z = np.linspace(-1.55, 1.6, 600); V = z ** 4 / 4 - z ** 2 / 2 + 0.12 * z; b.plot(z, V, color=INK, lw=2); kl = np.argmin(np.where(z < 0, V, 9)); kr = np.argmin(np.where(z > 0, V, 9))
    b.plot([z[kl]], [V[kl] + 0.035], 'o', ms=12, color=AQUA, mec='white', zorder=5); b.plot([z[kr]], [V[kr] + 0.035], 'o', ms=12, mfc='white', mec=ORANGE, mew=2, zorder=5)
    b.annotate('', xy=(0.3, 0.12), xytext=(-0.8, -0.05), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.3, connectionstyle='arc3,rad=-0.45', ls=(0, (3, 2))))
    b.text(z[kl], V[kl] - 0.07, 'defence\nworking', ha='center', va='top', fontsize=9.8, color='#12875c', fontweight='bold'); b.text(z[kr], V[kr] - 0.07, 'defence\nswitched off', ha='center', va='top', fontsize=9.8, color='#c9501f', fontweight='bold'); b.text(-0.25, 0.36, 'random pushes can carry\nthe cell over the barrier', ha='center', va='center', fontsize=9.2, color=MUTED)
    b.set_ylim(-0.72, 0.55); b.set_xticks([]); b.set_yticks([]); b.set_title('The same thing as a landscape with two wells', fontsize=10.5, loc='left'); b.set_xlabel('state of the cell')
    for q in ax:
        for sp in ('top', 'right'): q.spines[sp].set_visible(False)
    save(fig, 'fig_switch.png')

# ---------------- Fig. The day ----------------
def fig_day():
    fig, ax = plt.subplots(figsize=(11, 4.9)); rows = ['animal active\n(mouse and rat are nocturnal)', 'marker of oxidative damage in\nPV cells of the frontal cortex (rat)', 'nets around PV cells\n(rat)', 'glutathione in the midbrain\n(mouse)', 'oxidation state of the tissue in the\nmaster clock (slices, rodents)']
    n = len(rows); ax.axvspan(12, 24, color='#e4e4e1', lw=0, zorder=0); ax.set_xlim(-0.6, 24.6); ax.set_ylim(-0.6, n + 0.1)
    for i, r in enumerate(rows): ax.text(-1.0, n - 1 - i, r, ha='right', va='center', fontsize=9.2, color=INK, linespacing=1.15); ax.plot([0, 24], [n - 1 - i] * 2, color='#d9d8d4', lw=0.8, zorder=1)
    up = dict(marker='^', ms=12, color=ORANGE, mec='white', ls='none', zorder=4); dn = dict(marker='v', ms=12, color=BLUE, mec='white', ls='none', zorder=4)
    ax.plot([12, 24], [n - 1, n - 1], color=MUTED, lw=6, solid_capstyle='butt', zorder=3)
    ax.plot([0, 24], [n - 2, n - 2], **up); ax.plot([12], [n - 2], **dn); ax.text(12.5, n - 2 + 0.28, '47 % below the maximum', fontsize=8.6, color=MUTED, va='center'); ax.text(18, n - 2 - 0.28, 'at 18 h: 27 % lower', fontsize=8.6, color=MUTED, ha='center', va='center')
    ax.plot([12, 24], [n - 3, n - 3], color=ORANGE, lw=3, alpha=0.45, solid_capstyle='butt', zorder=2); ax.plot([18], [n - 3], **up); ax.text(18.5, n - 3 + 0.28, '40 % brighter than at 0 h', fontsize=8.6, color=MUTED, va='center'); ax.text(12.3, n - 3 - 0.28, 'at 12 h: 27 % brighter', fontsize=8.6, color=MUTED, va='center')
    ax.plot([20], [n - 4], **up); ax.plot([5], [n - 4], **dn); ax.text(19.5, n - 4 + 0.28, '1.2 times above the minimum', fontsize=8.6, color=MUTED, va='center', ha='right')
    ax.plot([14], [n - 5], **up); ax.plot([6.5], [n - 5], **dn); ax.text(14.5, n - 5 + 0.28, 'oxidised at the start of the night', fontsize=8.6, color=MUTED, va='center'); ax.text(7.0, n - 5 + 0.28, 'reduced during the day', fontsize=8.6, color=MUTED, va='center')
    ax.set_xticks(range(0, 25, 6)); ax.set_xticklabels(['0\nlights on', '6', '12\nlights off', '18', '24']); ax.set_yticks([]); ax.set_xlabel('time from lights on, hours')
    ax.text(6, n - 0.3, 'light: the animal sleeps', ha='center', fontsize=9.5, color=MUTED); ax.text(18, n - 0.3, 'dark: the animal is awake', ha='center', fontsize=9.5, color=MUTED)
    for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
    hs = [plt.Line2D([0], [0], **up), plt.Line2D([0], [0], **dn)]; ax.legend(hs, ['daily maximum', 'daily minimum'], frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=9.2)
    save(fig, 'fig_day.png')

if __name__ == '__main__':
    for f in (fig_overview, fig_times, fig_neuron, fig_circuit, fig_redox, fig_switch, fig_loops, fig_day, fig_dopamine, fig_model): f()
    print('ok')
