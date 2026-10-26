# -*- coding: utf-8 -*-
"""Рисунок лестницы захвата по файлу, который пишет tongues.py.

    python plot_tongues.py tongues.csv                       два окна: всё и самое плотное место
    python plot_tongues.py tongues.csv --zoom 4.770 4.790    окно крупного плана задать руками
    python plot_tongues.py tongues.csv --out lestnica.png --minw 0          показать все захваты

ПОЧЕМУ НЕ ПРОСТО plot(kd, rho). Точки после измельчения распределены крайне неравномерно: в ровных
местах шаг 0,006 мВ, а у границ — 5e-6. Обычная ломаная рисует это как вертикальные стены и съедает
узкие ступени. Поэтому каждая ступень рисуется ОТРЕЗКОМ от её левого края до правого, а точки без
захвата — серыми крапинами. Так видно и ширину ступени, и то, что между ними.
"""
import argparse, csv
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, MUTED, PALE = '#2a78d6', '#eb6834', '#1baf7a', '#52514e', '#f3f3f1'
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'savefig.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.22})


def plural(n, f1, f2, f5):
    """Русское склонение числительного: 1 ступень, 2 ступени, 5 ступеней."""
    n = abs(int(n)); d100 = n % 100; d10 = n % 10
    if 11 <= d100 <= 14: return f5
    return f1 if d10 == 1 else (f2 if 2 <= d10 <= 4 else f5)


def load(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            kd = float(r['kd']); rho = float(r['rho']) if r['rho'] not in ('', 'nan') else np.nan
            lock = (int(r['n']), int(r['m'])) if r['n'] and r['m'] else None
            tail = [float(x) for x in r['tail'].split()] if r['tail'].strip() else []
            rows.append(dict(kd=kd, rho=rho, lock=lock, pat=int(r['pattern']),
                             n_sleep=int(r['n_sleep']), tail=tail))
    return sorted(rows, key=lambda z: z['kd'])


def steps(rows):
    """Склеить подряд идущие точки с одним и тем же захватом в ступени."""
    out = []
    for z in rows:
        if z['lock'] and out and out[-1]['lock'] == z['lock'] and out[-1]['i2'] + 1 == z['i']:
            out[-1]['hi'] = z['kd']; out[-1]['i2'] = z['i']; out[-1]['n'] += 1
        elif z['lock']:
            out.append(dict(lock=z['lock'], lo=z['kd'], hi=z['kd'], rho=z['rho'], n=1,
                            tail=z['tail'], pat=z['pat'], i2=z['i']))
    return out


def draw(ax, rows, st, kmin, kmax, minw, label_all):
    sub = [z for z in rows if kmin <= z['kd'] <= kmax]
    free = [z for z in sub if z['lock'] is None and np.isfinite(z['rho'])]
    dead = [z for z in sub if not np.isfinite(z['rho'])]
    if free:
        ax.plot([z['kd'] for z in free], [z['rho'] for z in free], '.', color=MUTED, ms=3.5,
                alpha=0.8, label='захвата нет (узор не повторяется)')
    wide = [s for s in st if s['hi'] >= kmin and s['lo'] <= kmax]
    for s in wide:
        w = s['hi'] - s['lo']
        ax.plot([max(s['lo'], kmin), min(s['hi'], kmax)], [s['rho']] * 2,
                '-', color=(ORANGE if s['lock'][1] > 1 else BLUE), lw=(3.4 if s['lock'][1] > 1 else 2.6),
                solid_capstyle='butt')
        if w < (kmax - kmin) * 0.002:                      # совсем узкую ступень иначе не видно
            ax.plot([0.5 * (s['lo'] + s['hi'])], [s['rho']], 'o',
                    color=(ORANGE if s['lock'][1] > 1 else BLUE), ms=4)
        if label_all or w >= (kmax - kmin) * 0.02:
            ax.annotate('%d:%d' % s['lock'], (0.5 * (s['lo'] + s['hi']), s['rho']),
                        xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8.5,
                        color=(ORANGE if s['lock'][1] > 1 else MUTED),
                        fontweight=('bold' if s['lock'][1] > 1 else 'normal'))
    if dead:
        d0 = min(z['kd'] for z in dead)
        if d0 <= kmax:
            ax.axvspan(max(d0, kmin), kmax, color=PALE)
            ax.text(max(d0, kmin) + (kmax - kmin) * 0.01, ax.get_ylim()[0], ' сна нет вовсе',
                    color=MUTED, fontsize=9, va='bottom')
    ax.set_xlim(kmin, kmax)
    ax.set_xlabel('торможение сонного ядра повреждением $kD$, мВ')
    ax.set_ylabel('число вращения: суток внешней силы на цикл')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv'); p.add_argument('--out', default='pics/tongues.png')
    p.add_argument('--zoom', nargs=2, type=float, default=None)
    p.add_argument('--minw', type=float, default=0.0, help='не рисовать ступени уже этого, мВ')
    p.add_argument('--label-all', action='store_true')
    a = p.parse_args()
    rows = load(a.csv)
    for i, z in enumerate(rows): z['i'] = i
    st = [s for s in steps(rows) if (s['hi'] - s['lo']) >= a.minw or s['lock'][1] > 1]
    kmin = min(z['kd'] for z in rows); kmax = max(z['kd'] for z in rows)
    # окно крупного плана: там, где больше всего разных захватов на единицу длины
    if a.zoom: z0, z1 = a.zoom
    else:
        cnt = {}
        for s in st: cnt[round(0.5 * (s['lo'] + s['hi']), 2)] = cnt.get(round(0.5 * (s['lo'] + s['hi']), 2), 0) + 1
        c = max(cnt, key=cnt.get) if cnt else 0.5 * (kmin + kmax)
        z0, z1 = c - 0.012, c + 0.012

    fig, ax = plt.subplots(2, 1, figsize=(12.4, 7.4))
    draw(ax[0], rows, st, kmin, kmax, a.minw, a.label_all)
    ax[0].set_title('Лестница захвата целиком: %d %s на %.3f мВ'
                    % (len(st), plural(len(st), 'ступень', 'ступени', 'ступеней'), kmax - kmin),
                    fontsize=11, loc='left')
    ax[0].axvspan(z0, z1, color='#fbe2d8', alpha=0.7, zorder=0)
    draw(ax[1], rows, st, z0, z1, 0.0, True)
    nz = sum(1 for s in st if s['lo'] < z1 and s['hi'] > z0)
    na = sum(1 for s in st if s['lo'] < z1 and s['hi'] > z0 and s['lock'][1] > 1)
    ax[1].set_title('Крупный план: %.3f–%.3f мВ, %d %s, из них с чередованием длин %d'
                    % (z0, z1, nz, plural(nz, 'ступень', 'ступени', 'ступеней'), na),
                    fontsize=11, loc='left')
    ax[1].legend(frameon=False, fontsize=8.5, loc='upper left')
    # врезка: узор самой длинной ступени с чередованием
    alt = [s for s in st if s['lock'][1] > 1 and s['tail']]
    if alt:
        s = max(alt, key=lambda y: (y['hi'] - y['lo']))
        ins = ax[1].inset_axes([0.60, 0.08, 0.38, 0.34])
        seq = (s['tail'] * 4)[:min(3 * len(s['tail']), 12)]
        ins.bar(range(len(seq)), seq, color=[ORANGE if v > 36 else BLUE for v in seq], width=0.75)
        ins.axhline(24, color=MUTED, lw=0.9, ls=':')
        ins.set_xticks([]); ins.set_yticks([24, 48]); ins.tick_params(labelsize=8)
        ins.set_ylim(0, max(seq) * 1.25); ins.grid(False)
        ins.set_title('узор %d:%d, ч: %s' % (s['lock'][0], s['lock'][1],
                      ', '.join('%.0f' % v for v in s['tail'])), fontsize=8, loc='left', color=ORANGE)
    fig.tight_layout(pad=1.1)
    fig.savefig(a.out, dpi=170, bbox_inches='tight', pad_inches=0.08)
    print('готово: %s' % a.out)
    print('всего %d %s, из них с чередованием длин циклов %d'
          % (len(st), plural(len(st), 'ступень', 'ступени', 'ступеней'),
             sum(1 for s in st if s['lock'][1] > 1)))
    print('%-8s %-12s %-12s %-10s %s' % ('захват', 'от, мВ', 'до, мВ', 'ширина', 'узор, ч'))
    for s in sorted(st, key=lambda y: y['lo']):
        print('%-8s %-12.6f %-12.6f %-10.2e %s' % ('%d:%d' % s['lock'], s['lo'], s['hi'],
              s['hi'] - s['lo'], ', '.join('%.1f' % v for v in s['tail'][:6])))


if __name__ == '__main__':
    main()
