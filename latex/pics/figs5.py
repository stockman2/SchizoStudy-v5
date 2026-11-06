# -*- coding: utf-8 -*-
"""Рисунки к брошюре v5: асимметричная защита, рваность нагрузки, три фазы, обрыв кэша.

Данные берутся из прогона 17.10 (2400 наборов, полоса 1255, 123 блока, 3 ч 05 мин) — числа
вписаны прямо здесь, чтобы рисунок собирался без прогона и его можно было сверить глазами
с отчётом."""
import json, math, sys
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, SURF, PALE = '#0b0b0b', '#52514e', '#fcfcfb', '#f3f3f1'
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'savefig.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': 0.22})
def save(fig, name): fig.savefig(name, dpi=170, bbox_inches='tight', pad_inches=0.08); plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(11.6, 7.8))

# ---------- (а) Ф23: асимметричная защита против выкладки w(r) ----------
a = ax[0, 0]
r23 = [0.33, 0.50, 1.00, 2.00, 3.00]
w23 = [0.4409, 0.5444, 0.7050, 0.8270, 0.8777]
lv23 = [0.057507, 0.064384, 0.079146, 0.095768, 0.109747]
se23 = [0.007472, 0.007700, 0.008036, 0.008309, 0.008389]
sl = (lv23[-1] - lv23[0]) / (w23[-1] - w23[0]); b0 = lv23[0] - sl * w23[0]
ww = np.linspace(0.40, 0.92, 50)
a.plot(ww, b0 + sl * ww, color=MUTED, lw=1.3, ls='--', label='выкладка: уровень линеен по w(r)')
a.errorbar(w23, lv23, yerr=se23, fmt='o', color=ORANGE, ms=7, lw=1.4, capsize=3,
           label='измеренный уровень защиты')
for r_, w_, l_ in zip(r23, w23, lv23):
    a.annotate('r = %.2f' % r_, (w_, l_), xytext=(4, -11), textcoords='offset points',
               fontsize=8, color=MUTED)
for i in (0, 4):
    a.plot([w23[i]], [lv23[i]], 'o', mfc='none', mec=INK, ms=12, mew=1.2)
a.set_xlabel('предсказанное положение w(r) = f·r/(f·r + 1 − f)')
a.set_ylabel('свой здоровый уровень защиты')
a.set_title('а. Асимметричная защита: порядок совпал точно (ρ = +1, p = 0,008)', fontsize=10, loc='left')
a.legend(frameon=False, fontsize=9, loc='upper left')
a.text(0.63, 0.062, 'в кружках — две точки привязки размаха;\nтри средние предсказаны',
       fontsize=8, color=MUTED)

# ---------- (б) Ф24: какая величина годна, а какая насыщается ----------
b = ax[0, 1]
ts = np.array([0.2, 0.5, 1.0, 2.625, 5.0, 10.0])
shev = np.array([0.5992, 0.6231, 0.6263, 0.6271, 0.6135, 0.5880])
rate = np.array([0.05501, 0.05220, 0.04742, 0.04061, 0.03620, 0.03280])
b.semilogx(ts, rate / rate[0], 'o-', color=ORANGE, lw=2.4, ms=7, label='частота срывов (годная)')
b.semilogx(ts, shev / shev[0], 's--', color=BLUE, lw=1.8, ms=6,
           label='доля повреждавшихся (насыщается)')
b.axvspan(0.5, 2.625, color=YELLOW, alpha=0.16, lw=0)
b.text(1.15, 0.79, 'между постоянными\nзащиты: 12 ч и 63 ч', fontsize=8, color=MUTED, ha='center')
b.axhline(1.0, color=MUTED, lw=0.8, ls=':')
b.set_xlabel('время корреляции нагрузки, сутки (среднее и дисперсия закреплены)')
b.set_ylabel('в долях значения при 0,2 суток')
b.set_title('б. Рваность нагрузки: ответ зависит от того, ЧТО измерять', fontsize=10, loc='left')
b.legend(frameon=False, fontsize=9, loc='lower left')
b.set_xticks([0.2, 0.5, 1, 2.625, 5, 10]); b.set_xticklabels(['0,2', '0,5', '1', '2,6', '5', '10'])

# ---------- (в) Ф25: полоса спасённых, два перебора против одной геометрии ----------
c = ax[1, 0]
Dk = [0.0999, 0.1089, 0.1248, 0.1437, 0.2525, 0.2883, 0.3441]
wk = [0.013540, 0.010500, 0.007324, 0.004712, 0.003218, 0.002533, 0.002134]
sk = [0.000670, 0.000558, 0.000421, 0.000298, 0.000220, 0.000190, 0.000170]
Da = [0.0643, 0.0759, 0.0999, 0.1619]
wa = [0.009544, 0.010481, 0.013540, 0.019109]
sa = [0.000487, 0.000555, 0.000670, 0.000879]
c.errorbar(Dk, wk, yerr=sk, fmt='o-', color=ORANGE, lw=2.2, ms=7, capsize=3,
           label='долю меняет связь со сном  (ρ = −1)')
c.errorbar(Da, wa, yerr=sa, fmt='s-', color=BLUE, lw=2.2, ms=7, capsize=3,
           label='долю меняет размах нагрузки  (ρ = +1)')
c.plot([0.0999], [0.013540], 'o', mfc='none', mec=INK, ms=14, mew=1.4)
c.annotate('общая точка', (0.0999, 0.01354), xytext=(0.135, 0.0162), fontsize=8, color=MUTED,
           arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
c.set_xlabel('доля повреждённых клеток D')
c.set_ylabel('ширина полосы спасённых')
c.set_title('в. Три фазы: ширина полосы — не функция D, знаки противоположны', fontsize=10, loc='left')
c.legend(frameon=False, fontsize=9, loc='upper left')

# ---------- (г) обрыв кэша: откуда взялось ускорение ----------
d = ax[1, 1]
rows = np.array([2000, 5250, 12000])
numpy_ms = np.array([14.87, 77.33, 175.55]) / rows * 1000
kern_ms = np.array([12.61, 31.20, 71.26]) / rows * 1000
d.plot(rows, numpy_ms, 'o-', color=BLUE, lw=2.4, ms=8, label='по одной операции (numpy)')
d.plot(rows, kern_ms, 's-', color=ORANGE, lw=2.4, ms=8, label='слитое ядро')
d.axvspan(2000, 5250, color=PALE, lw=0)
d.annotate('рабочий набор перестал\nвлезать в кэш', (3600, 2.3), fontsize=8, color=MUTED, ha='center')
for x_, y_ in zip(rows, numpy_ms):
    d.annotate('%.1f' % y_, (x_, y_), xytext=(0, 7), textcoords='offset points',
               fontsize=8, color=BLUE, ha='center')
for x_, y_ in zip(rows, kern_ms):
    d.annotate('%.1f' % y_, (x_, y_), xytext=(0, -14), textcoords='offset points',
               fontsize=8, color=ORANGE, ha='center')
d.set_xlabel('строк в процессе')
d.set_ylabel('стоимость строки, мс на шаг')
d.set_ylim(0, 17)
d.set_title('г. Почему ускорение в 3,3 раза взялось именно из слияния', fontsize=10, loc='left')
d.legend(frameon=False, fontsize=9, loc='center right')

fig.tight_layout(pad=1.1)
save(fig, 'pics/fig_v5.png')
print('pics/fig_v5.png готов')
