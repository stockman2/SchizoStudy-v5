# -*- coding: utf-8 -*-
"""Рисунки к брошюре v4: медленная защита, закон длительности, каскад удвоений, сетка уязвимости."""
import sys, math
sys.path.insert(0, '/home/claude/redox')
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, SURF, PALE = '#0b0b0b', '#52514e', '#fcfcfb', '#f3f3f1'
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'savefig.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.22})
def save(fig, name): fig.savefig(name, dpi=170, bbox_inches='tight', pad_inches=0.08); plt.close(fig)

TAU_M, TAU_S = 12., 63.                      # часы
fig, ax = plt.subplots(2, 2, figsize=(11.6, 7.6))

# ---- (а) двухступенчатая защита: отклик на ступень нагрузки ----
a = ax[0, 0]
t = np.linspace(0, 240, 2000)
m = 1 - np.exp(-t / TAU_M)
s = 1 - (TAU_S * np.exp(-t / TAU_S) - TAU_M * np.exp(-t / TAU_M)) / (TAU_S - TAU_M)
a.plot(t / 24, m, color=BLUE, lw=2.4, label='сигнал индукции, 12 ч')
a.plot(t / 24, s, color=ORANGE, lw=2.6, label='белок защиты, 63 ч')
a.axhline(0.5, color=MUTED, lw=0.9, ls=':')
i50 = np.argmax(s >= 0.5); a.plot([t[i50] / 24], [0.5], 'o', color=ORANGE, ms=6)
a.annotate('половина — через %.0f ч' % t[i50], (t[i50] / 24, 0.5), xytext=(3.2, 0.33),
           color=MUTED, fontsize=9, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
a.set_xlabel('сутки после включения нагрузки'); a.set_ylabel('доля стационарного уровня')
a.set_title('а. Две ступени защиты: сигнал быстрый, белок медленный', fontsize=10, loc='left')
a.legend(frameon=False, fontsize=9, loc='lower right')

# ---- (б) закон длительности: аффинный против насыщающихся ----
b = ax[0, 1]
tt = np.array([12., 24., 63., 88.]) / 24.; obs = 1.28 + 0.967 * tt          # измеренная прямая
hold_t = 150. / 24.; hold_obs = 6.629                                        # точка вне подгонки
x = np.linspace(0.3, 7.0, 300)
b.plot(x, 1.28 + 0.967 * x, color=ORANGE, lw=2.6, label='аффинный: 1,28 + 0,97·$\\tau$')
T_, t0 = 6.0, 1.2
b.plot(x, T_ * x / (x + t0), color=BLUE, lw=1.8, ls=(0, (5, 3)), label='насыщающийся: $T x/(x+t_0)$')
b.plot(x, T_ * (1 - np.exp(-x / t0)), color=AQUA, lw=1.8, ls=(0, (2, 2)), label='насыщающийся: $T(1-e^{-x/t_0})$')
b.plot(tt, obs, 'o', color=INK, ms=6, label='точки подгонки (память < 100 ч)')
b.plot([hold_t], [hold_obs], 's', color=ORANGE, ms=9, mfc='white', mew=2,
       label='проверка вне подгонки: 150 ч')
b.axhline(2.0, color=MUTED, lw=1.2, ls=':')
b.text(0.35, 2.12, 'клиническая цель: 2 суток', color=MUTED, fontsize=9)
b.set_xlabel('память защиты $\\tau_{\\rm б}$, суток'); b.set_ylabel('длительность эпизода, суток')
b.set_title('б. Насыщения нет: защита эпизод удлиняет', fontsize=10, loc='left')
b.legend(frameon=False, fontsize=8, loc='upper left')

# ---- (в) лестница числа вращения и чередование длин ----
c = ax[1, 0]
from px.rules.f20 import _scan
KD = np.concatenate([np.linspace(4.70, 4.80, 41), np.linspace(4.80, 5.22, 85)])
rows = _scan(KD, 1600, 3 * 24. / 1600, nd=120, drop=0.70)
kd = np.array([z['kd'] for z in rows])
rho = np.array([z['rho'] if z['rho'] == z['rho'] else np.nan for z in rows])
lock = [z['lock'] for z in rows]
c.step(kd, rho, where='mid', color=BLUE, lw=2.0)
c.plot(kd[np.isfinite(rho)], rho[np.isfinite(rho)], 'o', color=BLUE, ms=2.6)
seen = {}
for k_, r_, f_ in zip(kd, rho, lock):
    if f_ and f_ not in seen: seen[f_] = (k_, r_)
for f_, (k_, r_) in seen.items():
    lab = '%d:%d' % f_
    c.annotate(lab, (k_, r_), xytext=(k_ + (0.012 if f_[1] == 1 else 0.0), r_ + (0.12 if f_[1] > 1 else 0.10)),
               color=(ORANGE if f_[1] > 1 else MUTED), fontsize=9,
               fontweight=('bold' if f_[1] > 1 else 'normal'))
dead = kd[~np.isfinite(rho)]
if len(dead):
    c.axvspan(dead.min(), kd.max(), color=PALE)
    c.text(dead.min() + 0.004, 2.55, 'сна нет\nвовсе', color=MUTED, fontsize=9)
c.set_xlim(4.70, kd.max()); c.set_ylim(0.8, 4.6)
c.set_xlabel('торможение сонного ядра повреждением $kD$, мВ')
c.set_ylabel('число вращения: суток внешней силы на один цикл')
c.set_title('в. Лестница захвата: ритм не ломается, а перескакивает', fontsize=10, loc='left')
# врезка: сам узор чередования в состоянии 4:3
alt = [z for z in rows if z['lock'] and z['pat'] > 1]
if alt:
    z = max(alt, key=lambda y: y['pat'])
    ins = c.inset_axes([0.42, 0.05, 0.55, 0.27])
    seq = (z['cyc_tail'][-z['pat']:] * 4)[:9]
    ins.bar(range(len(seq)), seq, color=[ORANGE if v > 36 else BLUE for v in seq], width=0.75)
    ins.axhline(24, color=MUTED, lw=0.9, ls=':')
    ins.set_xticks([]); ins.set_yticks([24, 48]); ins.set_yticklabels(['24', '48'], fontsize=8)
    ins.set_ylim(0, 58); ins.tick_params(labelsize=8)
    ins.set_title('узор %d:%d при $kD$ = %.3f мВ: %s ч' % (z['lock'][0], z['lock'][1], z['kd'],
                  ', '.join('%.0f' % v for v in z['cyc_tail'][-z['pat']:])), fontsize=8, loc='left', color=ORANGE)
    ins.grid(False)

# ---- (г) сетка уязвимости: края закреплены, разброс ползёт ----
d = ax[1, 1]
from px.core import vuln
JS = np.array([8, 16, 24, 32, 48, 64, 96, 128, 256, 512])
sd = np.array([np.std(np.log(vuln(int(J), 0.10, 1.0)[0])) for J in JS])
mn = np.array([vuln(int(J), 0.10, 1.0)[0].min() for J in JS])
mx = np.array([vuln(int(J), 0.10, 1.0)[0].max() for J in JS])
d.plot(JS, sd, 'o-', color=ORANGE, lw=2.4, ms=5, label='реализованный разброс std(ln v)')
d.axhline(0.0910, color=MUTED, lw=1.0, ls=':')
d.text(9, 0.0923, 'предел при $J\\to\\infty$', color=MUTED, fontsize=9)
d.axhline(0.10, color=BLUE, lw=1.2, ls='--')
d.text(9, 0.1012, 'заданная $\\sigma_v = 0{,}10$', color=BLUE, fontsize=9)
d.text(28, 0.1135, 'края сетки закреплены и почти не~ползут:\nнаименьший множитель %.4f при $J=8$\nи %.4f при $J=512$'
       .replace('~', ' ') % (mn[0], mn[-1]), color=AQUA, fontsize=8.5, va='top')
for J in (16, 32, 64):
    i = int(np.where(JS == J)[0][0]); d.annotate('J = %d' % J, (JS[i], sd[i]), xytext=(JS[i] * 1.05, sd[i] + 0.0035),
                                                 color=ORANGE, fontsize=9)
d.set_xscale('log'); d.set_xticks([8, 16, 32, 64, 128, 512]); d.set_xticklabels(['8', '16', '32', '64', '128', '512'])
d.set_xlabel('число клеток в популяции $J$'); d.set_ylabel('разброс множителей, std(ln v)')
d.set_title('г. Третий путь: $J$ входит как эффективный разброс', fontsize=10, loc='left')
d.legend(frameon=False, fontsize=8.5, loc='upper right')

fig.tight_layout(pad=1.2)
save(fig, 'pics/fig_slow.png')
print('pics/fig_slow.png готов; половина белка через %.0f ч; разброс J=16/32/64: %.4f %.4f %.4f'
      % (t[i50], sd[1], sd[3], sd[5]))
