# -*- coding: utf-8 -*-
"""
mass_core6.py — шаг А и шаг Б вместе: популяция PV-клеток внутри нейронной массы E–PV,
а расписание сна заменено переключателем бодрствования Филлипса–Робинсона.

ЧТО ИЗМЕНЕНО ПРОТИВ mass_core5.py
  (1) ПЕРЕДЕЛАН САМ СЧЁТ, уравнения не тронуты. Прежде перебор был последовательным: девять значений k —
      девять прогонов по циклу. Теперь все точки перебора сложены в ОДНУ матрицу строк и считаются одним
      циклом, а сам батч разрезан по процессам. Шум разыгрывается на длину ОДНОГО блока и повторяется на
      все блоки, поэтому точки перебора строго парные, а ответ не зависит ни от числа блоков, ни от числа
      процессов. Это доказывается в разделе 2а (расхождение должно быть РОВНО нулевым) и проверено ещё и
      снаружи: все числа results.json на пробном прогоне совпали с mass_core5.py до последнего знака.
      Свидетельство: на двух ядрах время счёта упало в 2,5 раза. Ключ --proc=N задаёт число процессов,
      --proc=1 отключает их вовсе.
  (2) Добавлено правило Ф12(в) — ПАРНЫЙ КОНТРОЛЬ к рефрактерному периоду. Прошлый прогон нельзя было
      истолковать: медленный шум нагрузки имеет то же время корреляции, что и предполагаемая память
      защиты, и действует в противоположную сторону. Контроль: тот же набор, то же зерно, тот же шум,
      но вторая ступень защиты лишена длинной памяти. Стационар при этом снова не меняется.

ЗАЧЕМ ЭТО СДЕЛАНО
  В sleep_core2.py и в diag_gradual.py повреждение d вело себя двоично: одна клетка — двухустойчивый
  переключатель, и промежуточных значений почти не было (в середине шкалы d проводило 0,9–1,3 % времени).
  Поэтому «медленная переменная» между здоровьем и срывом в той модели отсутствовала в принципе, и правило Ф4(а)
  было непроверяемо не по недостатку данных, а по устройству модели.
  Считать популяцию НЕЗАВИСИМЫХ клеток было бы подгонкой под нужный ответ: средняя доля повреждённых
  стала бы плавной просто от усреднения, без всякого механизма. Поэтому клетки здесь связаны через то,
  через что они связаны в коре: через активность сети. Меньше работающих PV-клеток — слабее торможение —
  выше возбуждение — выше нагрузка на оставшиеся PV-клетки. Круг замыкается на самом деле, а не на словах.

ТРИ СЛОЯ И ЧТО ИХ СВЯЗЫВАЕТ
  (1) Бодрствование — модель Филлипса–Робинсона (Phillips & Robinson 2007; Phillips, Robinson 2008;
      набор параметров «new nominal», он же Skeldon и соавт. 2014, табл. 1, столбец PR):
          Q(V) = Qmax/(1 + exp(-(V - theta)/sigma))
          tau_v*dVv/dt = nu_vm*Q(Vm) + D_v - Vv       (сонное ядро, VLPO)
          tau_m*dVm/dt = A_m + nu_mv*Q(Vv) - Vm       (моноаминовые ядра)
          chi*dH/dt    = mu*Q(Vm) - H                 (сомногены, по смыслу — аденозин и прочее)
          D_v = B0 + nu_vc*cos(2*pi*(t - alpha)/24) + AG*G        (сомногены G вместо H)
          dG/dt = (1 - G)/18,2 в бодрствовании,  dG/dt = -G/4,2 во сне   (Daan, Beersma, Borbely 1984)
      Бодрствование — это Q(Vm) > 1 с^-1. Переключение происходит само, на складках этой системы:
      никаких порогов H и L, которые были в sleep_core.py, здесь нет, и потолка H_CAP тоже нет —
      полки на прежних графиках создавал именно он.
  (2) Кора — нейронная масса из двух групп, возбуждающей E и тормозной PV (Вильсон–Коуэн):
          E' = [-E + F(c1*E - c2*g*I + Pe)]/tau_e,    I' = [-I + F(c3*E - c4*g*I)]/tau_i
      Веса ВЫХОДА PV-клеток умножены на g = 1 - gdep*D, где D — ДОЛЯ повреждённых PV-клеток.
      Так сделано в Dey Sarkar & Ermentrout 2025: множитель стоит на всех выходных весах PV, а не на одном.
      Величина gdep = 0,4 взята из Volman и соавт. 2011, где торможение опускали до 60 % от исходного.
      Внешний приток к E зависит от бодрствования: Pe = Pe0 + Pe1*Q(Vm)/Q_wake. Это и есть восходящая
      активация коры моноаминовыми ядрами, то есть та же переменная, что переключает сон.
      Оговорка, важная и не замаскированная: tau_e и tau_i — десятки миллисекунд, то есть в тысячи раз
      быстрее всего остального, поэтому кора здесь взята в стационарном состоянии (адиабатическое исключение).
      Гамма-ритм при этом отсутствует НАМЕРЕННО: нужна средняя частота разрядов, от неё зависит обмен.
      Набор весов выбран так, чтобы стационарное состояние было единственным и устойчивым (проверка в разделе 0);
      режим с колебаниями — отдельная задача, здесь он не используется.
  (3) Клетка — то самое редокс-ядро без изменений:
          nu = eps*u + x*u/(1 + u),  eps = e0*c^gam;  N = N0/(1 + u/KN)
          x' = c^m*(1 - x) - mu0*x*u^2/(1 + u);       c' = [N + beta*u/(KB + u) - c]/T
      Но теперь это не одна клетка, а J клеток с РАЗНОЙ уязвимостью: нагрузка на клетку j равна L*v_j,
      где log v_j — равновероятные квантили нормального распределения с размахом sigma_v, mean(v) = 1.
      Основание для перехода к доле: Koulakov, Raghavachari, Kepecs, Lisman 2002 — среднее по популяции
      двухустойчивых единиц с РАЗБРОСАННЫМИ порогами плавно (ступенчато) и сохраняет гистерезис.
      Основание для выбора именно доли, а не среднего повреждения: Tomita и соавт. 2025 — собственная
      разнородность популяции не сворачивается в одно уравнение среднего поля.
  СВЯЗИ МЕЖДУ СЛОЯМИ, все четыре:
      Q(Vm) -> Pe -> E ;   E -> L (нагрузка) ;   L*v_j -> x_j -> D ;   D -> g -> E   и   D -> VLPO -> сон.

НАГРУЗКА
      tau_L*dL/dt = nu0*(1 + A_H*(G - G_ср))*(E/E_неповр)*exp(eta) - L,  tau_L = 0,1*T
  E0 — средняя активность за свободный цикл при D = 0, считается в разделе 0. Поэтому средняя нагрузка
  равна nu0 сама, без подгонки. Суточный член несут сомногены G, а не активность коры: активность в этой
  модели почти прямоугольна (у Филлипса–Робинсона приток моноаминов за время бодрствования не меняется),
  и нагрузка на ней выходила ступенькой, а не пилой. Вклад повреждения отделён делением на E_неповр —
  активность при том же бодрствовании, но при g = 1, иначе бодрствование считалось бы дважды.
  A_H НЕ подбирается под результат: он выбран так, чтобы размах нагрузки
  между бодрствованием и сном вышел 0,20 — это число взято из литературы раньше и независимо
  (потребление глюкозы в коре мыши -16 % во сне; CMRO2 у человека -21 % при засыпании; 8-oxo-dG в PV-клетках
  +25 % после 6 ч нарушенного сна). Суточный член больше не задан руками: он получается из переключателя.
  eta — шум Орнштейна–Уленбека на логарифме нагрузки: сигма 0,08, время корреляции 0,3*T.

СВЯЗЬ «ПОВРЕЖДЕНИЕ МЕШАЕТ ЗАСНУТЬ» (прежнее правило Ф3, но вставлено по-другому)
      D_v = B0 + nu_vc*cos(2*pi*t/24) + AG*G - k*D
  То есть повреждение подтормаживает сонное ядро напрямую, а не поднимает выдуманный порог. Поэтому
  удлинение бодрствования выходит из складок переключателя, а не из упора в потолок: при k от 0 до 7 мВ
  бодрствование растёт плавно с 15,5 до 21,7 ч, при k около 7,5 мВ начинает пропускаться ночь (цикл 48 ч),
  при 8 мВ — две (72 ч), при 9 мВ сон не наступает вовсе. Эта лестница посчитана в разделе 0 заранее.
  Сон считается одинаково восстановительным у здоровых и у больных: chi и mu не меняются. Это допущение.

ЕДИНИЦЫ
  Время клетки — в T (время восстановления Prx). Сутки = r*T, r из [2, 4]. Час = сутки/24. Все постоянные
  переключателя — в часах (tau_v и tau_m заданы в секундах и делятся на 3600: это известная ловушка,
  в статьях tau в секундах, а chi в часах).

ПРАВИЛА, ЗАФИКСИРОВАННЫЕ ДО СЧЁТА — те же, что раньше, без изменений формулировок
  Ф2 (суточный член).
    (а) Срывы скапливаются во второй половине бодрствования: доля срывов позже середины бодрствования > 0,5.
        Прежде получено 0,733.
    (б) Есть порог по размаху суточной нагрузки A*: доля повреждавшихся растёт не плавно, а с переломом.
        ПРЕЖДЕ НЕ ВЫПОЛНЕНО: рост был плавный, 0,073 -> 0,166 при A от 0 до 0,4, всего в 2,3 раза.
        Отрицательный результат сохранён намеренно и проверяется заново: суточный размах — модулятор риска,
        а не ворота. Здесь роль A играет множитель при A_H.
    (в) Сон редко кончает эпизод: доля эпизодов, кончающихся во сне, меньше 0,3.
        ПРЕЖДЕ НЕ ВЫПОЛНЕНО: получено 0,500.
  Ф3 (сон как следствие), перебор k:
    (а) Средняя длительность эпизода при k = k_база не меньше чем в 1,5 раза больше, чем при k = 0. Парно, одно зерно.
    (б) Доля наборов с бодрствованием дольше двух суток больше нуля и растёт с k.
    (в) Доля эпизодов короче суток ниже 0,50.
  Ф4 (сон портится раньше срыва):
    (а) Средняя длина бодрствования растёт с повреждением уже при D < 0,5: в ячейке 0,4–0,5 не меньше
        чем на 10 % больше, чем в ячейке 0,0–0,1. Меньше 20 бодрствований в подпороговых ячейках — непроверяемо.
    (б) Длина бодрствования превышает здоровый уровень не меньше чем за сутки до первого срыва и не больше
        чем за восемь. Здоровый уровень — ячейка D от 0 до 0,1, а не средний по набору.
    Основание ожиданий: Reeve и соавт. 2015; Meyer и соавт. 2022. При построении модели не использовались.
    Честность: прошлый «успех» Ф4(б) был отбором, а не опережением (длина бодрствования перед срывом была
    ровной: 0,681 -> 0,680 за 15 суток), а Ф4(а) был непроверяем из-за двоичности d. Обе части — предсказания.
  Ф5 (новое, записано до счёта; это проверка самого шага А, а не биологии).
    Доля повреждённых D действительно градуальна: времени в середине шкалы (D от 0,3 до 0,7) не меньше 5 %,
    то есть хотя бы в 4 раза больше, чем давало одноклеточное d (0,9–1,3 %). Если нет — связь через сеть
    не создаёт промежуточных состояний, и шаг А не решает ту задачу, для которой делался.

ЧТО НЕ СДЕЛАНО И ЗНАЧИТ НЕ ПРОВЕРЯЕТСЯ
  Гамма-ритма нет (кора стационарна). Аденозин не выделен: H — сомногены вообще, место для вставки
  аденозина — производство mu*Q(Vm). Серотонин не выделен: моноаминовая группа у Филлипса–Робинсона слита
  (TMN + LC + дорсальное ядро шва + VTA); разделять её умеют только модели семейства Diniz Behn & Booth.
  Prx-SO2/3 в коре по времени суток не измеряли никогда — это стоящая оговорка ко всему суточному члену.
  Разброс времени сна от ночи к ночи (SD около 1,1 ч) ещё не добавлен.

ЧИСЛЕННАЯ СХЕМА
  Шаг — экспоненциальный Эйлер: на шаге перекись заморожена, а x, c, L, H и шум двигаются точными формулами.
  Пара Vv, Vm в тысячу раз быстрее шага, поэтому считается подшагами с затуханием: итерация стартует
  с прежнего значения и потому остаётся на той же ветви, пока ветвь не исчезнет, — гистерезис сохраняется.
  Кора — затухающая итерация к стационару (NWC шагов), невязка печатается в разделе 0.
  Шаг свой у каждого набора: dt = r*T/M, то есть M шагов на сутки; в часах шаг один и тот же, 24/M.

Запуск:  python mass_core6.py               обычный вариант
         python mass_core6.py --full        больше наборов, суток и точек перебора
         python mass_core6.py --smoke       несколько наборов, только проверить, что скрипт доходит до конца
         можно добавить --sets=300 --days=60 --m=2000 --j=32   чтобы задать размер счёта вручную
         --proc=N  сколько процессов на батч (по умолчанию — все ядра); --proc=1 — считать в одном
Нужны numpy, scipy, matplotlib. Результат: папка mass_out/ — report.md (в конце раздел для машинного
чтения), results.json, fig1…fig8.
"""
import os, sys, json, time, math
import multiprocessing as mp
import numpy as np
from scipy.integrate import solve_ivp

T0 = time.time(); FULL = '--full' in sys.argv; SMOKE = '--smoke' in sys.argv; OUT = 'mass_out'
def stage(m): print('[%6.1f с] %s' % (time.time() - T0, m), flush=True)
def argint(nm, dflt):
    for a in sys.argv:
        if a.startswith('--%s=' % nm): return int(a.split('=')[1])
    return dflt
UG = np.logspace(-4, 4, 641)
NAMES = ['e0', 'mu0', 'm', 'gam', 'KN', 'N0', 'beta', 'KB', 'T']
RANGES = dict(e0=(0.005, 0.1, 'log'), mu0=(0.01, 10, 'log'), m=(1, 2, 'lin'), gam=(0, 1, 'lin'), KN=(0.1, 100, 'log'), N0=(0.3, 1, 'lin'), beta=(0, 1, 'lin'), KB=(1, 100, 'log'), T=(20, 80, 'log'))
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']; INK = '#52514e'
# --- Филлипс–Робинсон, набор «new nominal» (Skeldon и соавт. 2014, табл. 1, столбец PR) ---
QMAX, THETA, SGM = 100., 10., 3.                 # с^-1, мВ, мВ
TAUV = TAUM = 10. / 3600.                        # часы (в статьях 10 с — ловушка с единицами)
NVM, NMV, A_M = -2.1, -1.8, 1.3                  # мВ*с, мВ*с, мВ
NVC, ALPHA = -2.9, 0.0                           # амплитуда суточного члена, мВ; его фаза, часы
TR, TD = 18.2, 4.2                               # постоянные времени сомногенов: подъём и спад, часы (Daan, Beersma, Borbely 1984)
B0, AG = -1.253, 6.0                             # постоянная часть притока к сонному ядру и вес сомногенов, мВ
# --- Кора: Вильсон–Коуэн, E и PV; режим с единственным устойчивым стационаром (проверяется в разделе 0) ---
C1, C2, C3, C4 = 3.0, 6.0, 8.0, 2.0              # E->E, PV->E, E->PV, PV->PV
AE_S, THE, AI_S, THI = 1.0, 2.0, 1.0, 4.0        # крутизна и порог сигмоид
PE0, PE1 = 1.6, 0.8                              # приток к E: во сне и добавка при полном бодрствовании
GDEP = 0.4                                       # g = 1 - GDEP*D; при D = 1 торможение падает до 60 % (Volman 2011)
NWC = 3; LAM = 0.6                               # шагов итерации коры и затухание
NSUB = 4                                         # подшагов на пару Vv, Vm
TAU_M, TAU_S = 12. / 24., 63. / 24.              # защита в две ступени: сигнал индукции и белок, сутки (из сырых данных Sci Rep 2022)
BFMAX = 3.0                                      # наибольшая сила индукции beta/N0; измеренная индукция белка — 2,85 раза
SR_AMP, SR_RATE, TAU_SR = 1.5, 0.10, 2.0        # серотониновый всплеск: во сколько раз поднимает нагрузку, сколько раз в сутки, постоянная спада в часах
SR_CEIL = 2.0                                    # фармакологический потолок: активность пирамидных клеток вдвое при 5-HT2A (срезы ПФК)
SLPN, TAUSL = 0.24, 2.0                          # шум притока к сонному ядру: амплитуда в мВ и время корреляции в часах
USE_DM = True                                    # связь через среднее повреждение (True) или через долю повреждённых (False)
TSLOW = 5.0                                      # время корреляции медленной составляющей шума нагрузки, сутки
DSWING = 0.20                                    # размах нагрузки от впадины к пику за сутки (литература)
DTHR = 0.5                                       # эпизод: доля повреждённых выше этого
A_PR = 2.0                                       # примирование: beta умножается на (1 + A_PR*p), то есть ПОТОЛОК ВТРОЕ (измерено: до x3)
J0V, VFIX = 32, 1.0                              # сетка уязвимости: закреплённый диапазон вероятностей, привязанный к J = 32
P0V = 0.5 / J0V                                  # крайний квантиль сетки; при J = J0V новая сетка В ТОЧНОСТИ равна прежней

Q = lambda V: QMAX / (1. + np.exp(-(V - THETA) / SGM))
F = lambda z, a, th: 1. / (1. + np.exp(-a * (z - th)))
Fp = lambda z, a, th: a * np.exp(-a * (z - th)) / (1. + np.exp(-a * (z - th))) ** 2

# ---------- Ядро клетки: перенесено из sleep_core2.py дословно ----------
def curve(P, u=UG, a=1.0, frozen=False):
    c = P['N0'] * a + 0 * u if frozen else P['N0'] * a / (1 + u / P['KN']) + P['beta'] * u / (P['KB'] + u)
    cm = c ** P['m']; x = cm / (cm + P['mu0'] * u * u / (1 + u)); return P['e0'] * c ** P['gam'] * u + x * u / (1 + u), x, c

def jac(P, u, x, c, a=1.0):
    eps = P['e0'] * c ** P['gam']; Gu = eps + x / (1 + u) ** 2; ux = -(u / (1 + u)) / Gu; uc = -(P['gam'] * P['e0'] * c ** (P['gam'] - 1) * u) / Gu
    h = u * u / (1 + u); hp = u * (2 + u) / (1 + u) ** 2; q = -P['N0'] * a / (P['KN'] * (1 + u / P['KN']) ** 2) + P['beta'] * P['KB'] / (P['KB'] + u) ** 2
    f1x = -c ** P['m'] - P['mu0'] * h - P['mu0'] * x * hp * ux; f1c = P['m'] * c ** (P['m'] - 1) * (1 - x) - P['mu0'] * x * hp * uc; f2x = q * ux / P['T']; f2c = (q * uc - 1) / P['T']
    return f1x + f2c, f1x * f2c - f1c * f2x

def rhs(t, y, p, nu, a=1.0):
    x, c = y[0], max(y[1], 1e-9); eps = p['e0'] * c ** p['gam']; b = eps + x - nu; s = math.sqrt(b * b + 4 * eps * nu); u = 2 * nu / (b + s) if b > 0 else (s - b) / (2 * eps)
    return [c ** p['m'] * (1 - x) - p['mu0'] * x * u * u / (1 + u), (p['N0'] * a / (1 + u / p['KN']) + p['beta'] * u / (p['KB'] + u) - c) / p['T']]

def window(nu):
    i = np.where(np.diff(nu) < 0)[0]
    if not len(i): return None
    return float(nu[i[0]:].min()), float(nu[i[0]])

def sample(K, rng):
    """beta теперь задаётся не сама по себе, а кратностью к N0: beta = N0 * U(0, BFMAX).
    Причина — измерение: индукция белка сульфиредоксина достигает 2,85 раза, а прежняя выборка
    beta из [0, 1] при N0 из [0,3; 1] такой кратности почти не допускала."""
    S = {}
    for k, (lo, hi, kind) in RANGES.items(): S[k] = (10 ** rng.uniform(np.log10(lo), np.log10(hi), K) if kind == 'log' else rng.uniform(lo, hi, K))[:, None]
    S['bfac'] = rng.uniform(0, BFMAX, K)[:, None]; S['beta'] = S['N0'] * S['bfac']
    return S

def states_at(p, nuv):
    nuc, xc, cc = curve(p); tr, dt_ = jac(p, UG, xc, cc); a0 = nuc[:-1] - nuv; a1 = nuc[1:] - nuv; cr = np.where(a0 * a1 < 0)[0]; out = []
    for i in cr:
        w = a0[i] / (a0[i] - a1[i])
        if (dt_[i] + w * (dt_[i + 1] - dt_[i])) > 0 and (tr[i] + w * (tr[i + 1] - tr[i])) < 0: out.append(float(xc[i] + w * (xc[i + 1] - xc[i])))
    return sorted(out)

# ---------- Слой 1: переключатель бодрствования ----------
def pr_folds():
    """Складки переключателя: D_v, при которых исчезает ветвь. Опубликованные значения 1,45 и 2,46 мВ."""
    Vv = np.linspace(-40, 40, 400001); Dv = Vv - NVM * Q(A_M + NMV * Q(Vv))
    d1 = np.diff(Dv); s = np.where(d1[:-1] * d1[1:] < 0)[0]
    return sorted(float(Dv[i + 1]) for i in s)

def pr_free(kd=0.0, M=1600, nd=60, drop=0.45, slpn=0.0, K=1, seed=5):
    """Свободный цикл переключателя без клетки. При slpn > 0 к притоку добавлен шум, и считается K независимых
    копий сразу, чтобы разброс длительностей был измерен на тысячах циклов, а не на трёх сотнях."""
    dt = 24. / M; ds = dt / NSUB; av = 1 - math.exp(-ds / TAUV); am = 1 - math.exp(-ds / TAUM)
    er = math.exp(-dt / TR); ed = math.exp(-dt / TD); ev = math.exp(-dt / TAUSL); sdz = slpn * math.sqrt(1 - ev * ev)
    rng = np.random.default_rng(seed)
    vv = np.full(K, -5.); vm = np.full(K, A_M); G = np.full(K, 0.3); z = np.zeros(K)
    aw = np.ones(K, bool); st = np.zeros(K); wk, sl, gs, qms = [], [], [], []; seg, segs = [], []
    for i in range(int(nd * M)):
        t = i * dt; Dv = B0 + NVC * math.cos(2 * math.pi * (t - ALPHA) / 24.) + AG * G - kd + z
        for _ in range(NSUB):
            vv += (NVM * Q(vm) + Dv - vv) * av; vm += (A_M + NMV * Q(vv) - vm) * am
        qm = Q(vm); a2 = qm > 1.; G = np.where(a2, 1 + (G - 1) * er, G * ed)
        if sdz > 0: z = z * ev + sdz * rng.standard_normal(K)
        if i > int(nd * M * drop):
            gs.append(G[0]); qms.append(qm.max())
            if a2[0]: seg.append(G[0])
            f = aw & ~a2; w2 = (~aw) & a2
            if f.any():
                wk.append(t - st[f]); st[f] = t
                if f[0] and len(seg) > 50: segs.append(np.array(seg))
                if f[0]: seg = []
            if w2.any():
                sl.append(t - st[w2]); st[w2] = t
                if w2[0]: seg = [G[0]]
        else: st[:] = t; seg = []
        aw = a2
    wk = np.concatenate(wk) if wk else np.array([]); sl = np.concatenate(sl) if sl else np.array([])
    if len(wk) < 3 or len(sl) < 3: return (float('nan'),) * 9
    w = float(np.median(wk)); s_ = float(np.median(sl)); dec = float('nan')
    if segs:
        gg = segs[len(segs) // 2]; n = max(len(gg) // 4, 1); r1 = gg[n] - gg[0]; r2 = gg[-1] - gg[-1 - n]
        if r1 > 0: dec = 100. * (1 - r2 / r1)
    return w, s_, w + s_, float(min(gs)), float(max(gs)), float(max(qms)), dec, float(wk.std()), float(sl.std())

def wc_fix(g, Pe, E=None, I=None, n=4000, lam=0.25):
    """Стационар коры прямой итерацией (для разделa 0: эталон, с которым сверяется быстрая итерация в счёте)."""
    E = np.full_like(np.asarray(g, float), 0.2) if E is None else np.array(E, float); I = np.full_like(E, 0.2) if I is None else np.array(I, float)
    for _ in range(n):
        E = E + lam * (F(C1 * E - C2 * g * I + Pe, AE_S, THE) - E); I = I + lam * (F(C3 * E - C4 * g * I, AI_S, THI) - I)
    ze = C1 * E - C2 * g * I + Pe; zi = C3 * E - C4 * g * I
    ev = []
    for e_, i_, g_, z1, z2 in np.broadcast(E, I, g, ze, zi):
        J = np.array([[-1 + C1 * Fp(z1, AE_S, THE), -C2 * g_ * Fp(z1, AE_S, THE)], [C3 * Fp(z2, AI_S, THI), -1 - C4 * g_ * Fp(z2, AI_S, THI)]])
        ev.append(float(np.linalg.eigvals(J).real.max()))
    return E, I, np.array(ev)

def calibrate(M=1600, nd=60):
    """E0 (средняя активность при D = 0), Q_бодр, средний и размах сомногенов за свободный цикл
    и A_H — вес сомногенов в нагрузке, выбранный по размаху DSWING от впадины к пику."""
    dt = 24. / M; ds = dt / NSUB; av = 1 - math.exp(-ds / TAUV); am = 1 - math.exp(-ds / TAUM)
    er = math.exp(-dt / TR); ed = math.exp(-dt / TD)
    _, _, _, g0, g1, qw, _, _, _ = pr_free(M=M, nd=nd)
    vv, vm, G, E, I = -5., A_M, 0.3, 0.3, 0.2; Es, Gs, Ew, Esl = [], [], [], []
    for i in range(int(nd * M)):
        t = i * dt; Dv = B0 + NVC * math.cos(2 * math.pi * (t - ALPHA) / 24.) + AG * G
        for _ in range(NSUB):
            vv += (NVM * Q(vm) + Dv - vv) * av; vm += (A_M + NMV * Q(vv) - vm) * am
        qm = Q(vm); aw = qm > 1.; G = 1 + (G - 1) * er if aw else G * ed
        Pe = PE0 + PE1 * min(qm / qw, 1.0)
        for _ in range(NWC):
            E += LAM * (F(C1 * E - C2 * I + Pe, AE_S, THE) - E); I += LAM * (F(C3 * E - C4 * I, AI_S, THI) - I)
        if i > int(nd * M * 0.45):
            Es.append(E); Gs.append(G); (Ew if aw else Esl).append(E)
    E0 = float(np.mean(Es)); Gbar = float(np.mean(Gs)); A_H = DSWING / max(g1 - g0, 1e-9)
    return E0, qw, Gbar, g0, g1, A_H, float(np.mean(Ew)), float(np.mean(Esl))

# ---------- Счёт ----------
NDB = 10; NLB = 8; RING = 10; NHD = 20          # ячейки: по повреждению, по суткам до срыва, память бодрствований, гистограмма D

def vuln(J, sigv, fixr=VFIX):
    """Множители нагрузки на клетку, mean = 1 ровно, при любой сетке и любом J.
    Две сетки вероятностей, и выбор между ними — главный вопрос раздела 13:
      fixr = 0  «равновероятные квантили»: p_j = (j + 0,5)/J. Диапазон ПОЛЗЁТ с J: чем больше клеток,
                тем более уязвима самая уязвимая. Тогда J — параметр модели, а не разрешение счёта.
      fixr = 1  «закреплённый диапазон, края»: p_j равномерно от P0V до 1 − P0V, крайние точки закреплены.
                Диапазон от J не зависит вовсе, J — только мелкость разбиения. При J = J0V = 32 эта сетка
                СОВПАДАЕТ с прежней до последнего знака, поэтому прежние прогоны при J = 32 в силе.
      fixr = 2  «закреплённый диапазон, середины»: p_j = P0V + (1 − 2*P0V)*(j + 0,5)/J. По смыслу выглядит
                естественнее (обрезанная логнормаль, выбранная равновероятно), но опора при этом всё равно
                ползёт с J, и сходится оно хуже. Остаётся как диагностика, в счёте не используется.
    sigv и fixr — скаляры или массивы по строкам; возвращается (число строк, J)."""
    sv = np.atleast_1d(np.asarray(sigv, float)); fr = np.atleast_1d(np.asarray(fixr, float))
    if sv.size < fr.size: sv = np.repeat(sv, fr.size // sv.size)
    if fr.size < sv.size: fr = np.repeat(fr, sv.size // fr.size)
    if J == 1: return np.ones((len(sv), 1))
    from scipy.stats import norm
    G = np.stack([(np.arange(J) + 0.5) / J,
                  P0V + (1 - 2 * P0V) * np.arange(J) / max(J - 1, 1),
                  P0V + (1 - 2 * P0V) * (np.arange(J) + 0.5) / J])
    p = G[np.rint(fr).astype(int)]
    v = np.exp(sv[:, None] * norm.ppf(p)); return v / v.mean(1, keepdims=True)

def simulate(P, nu0, thr, xhi, xlo, kd, aH, sigv, E0, qw, Gbar, sigma, tauc_rel, ndays, M, J, rng, warm=10, sigs=0.0, tslow=TSLOW, slpn=SLPN, usedm=USE_DM, sramp=1.0, srate=0.0, taus=TAU_S, vfix=VFIX, apr=0.0, taup=1.0, nbase=None):
    """Все наборы сразу. Длительности в сутках (сутки = r*T). Возвращает сводку по наборам и кривые,
    накопленные ОТДЕЛЬНО ПО КАЖДОМУ НАБОРУ, чтобы потом можно было нормировать на его собственный здоровый уровень."""
    K = len(nu0); nbase = K if nbase is None else int(nbase); NB = K // nbase
    assert NB * nbase == K, 'длина блока не делит число строк'
    ar = np.arange(K)
    def rowv(a):                                                                       # скаляр -> на все строки; длина NB -> по блокам; длина K -> как есть
        a = np.atleast_1d(np.asarray(a, float))
        return np.repeat(a, K // a.size) if a.size in (1, NB) and a.size != K else a.astype(float).copy()
    tl = (lambda a: a) if NB == 1 else (lambda a: np.tile(a, NB))                       # один и тот же шум во всех блоках: перебор строго парный
    kd = rowv(kd); aH = rowv(aH); sramp = rowv(sramp); srate = rowv(srate); udm = rowv(np.asarray(usedm, float))
    sigs = rowv(sigs); tslow = rowv(tslow); slpn = rowv(slpn); tauc_rel = rowv(tauc_rel); sigma = rowv(sigma); taus = rowv(taus); vfix = rowv(vfix); apr = rowv(apr); taup = rowv(taup)
    rgl = list(rng) if isinstance(rng, (list, tuple)) else None                          # список потоков: своё зерно на блок (перебор по зёрнам)
    fastn = (lambda: np.concatenate([g.standard_normal(nbase) for g in rgl])) if rgl else (lambda: tl(rng.standard_normal(nbase)))
    day = P['r'] * P['T']; dt = day / M; dth = 24. / M; ds = dth / NSUB
    v = vuln(J, rowv(sigv), vfix); tauL = 0.1 * P['T']
    ea = np.exp(-dt / (tauc_rel * P['T'])); sdn = sigma * np.sqrt(1 - ea * ea)
    eas = np.exp(-1.0 / (tslow * M)); sds = sigs * np.sqrt(1 - eas * eas)              # медленная составляющая: время корреляции tslow суток
    rngs = np.random.default_rng(77); etas = np.zeros(K); shift = 0.5 * sigs * sigs    # отдельный поток, один и тот же при любом sigs: перебор парный
    evz = math.exp(-dth / TAUSL); sdz = slpn * np.sqrt(1 - evz * evz); zz = np.zeros(K)  # шум притока к сонному ядру: зашумляет и засыпание, и подъём
    rngz = np.random.default_rng(99)
    esr = math.exp(-dth / TAU_SR); psr = srate * dth / 24.; sr = np.zeros(K)            # серотониновые всплески: пуассоновский поток, односторонний
    rngr = np.random.default_rng(1234); last_sr = np.full(K, -1e9); n_sr = np.zeros(K, int); nb_trig = np.zeros(K, int)
    ec = np.exp(-dt / P['T'])[:, None]; eL = np.exp(-dt / tauL)
    av = 1 - math.exp(-ds / TAUV); am = 1 - math.exp(-ds / TAUM)
    er = math.exp(-dth / TR); ed = math.exp(-dth / TD)
    x = np.ones((K, J)); cb = np.repeat(P['N0'][:, None], J, 1)                       # ступень 0: защита от рецептора
    mR = np.zeros((K, J)); sP = np.zeros((K, J))                                      # ступень 1: сигнал индукции; ступень 2: белок
    em = np.exp(-dt / (TAU_M * day))[:, None]; es = np.exp(-dt / (taus * day))[:, None]
    epr = np.exp(-dt / (taup * day))[:, None]; prm = np.zeros((K, J)); apr2 = apr[:, None]   # примирование: след недавнего окисления на том же сигнале, что движет индукцию
    c = cb.copy()                                                                     # полная защита: от рецептора плюс наработанный белок
    eta = np.zeros(K); L = nu0.copy()
    Vv = np.full(K, -5.); Vm = np.full(K, A_M); G = np.full(K, 0.3)
    E = np.full(K, 0.3); I = np.full(K, 0.2); Er = np.full(K, 0.3); Ir = np.full(K, 0.2)
    xhi2 = xhi[:, None]; xlo2 = xlo[:, None]; thr2 = thr[:, None]
    dam = np.zeros(K, bool); nb = np.zeros(K, int); nb_aw = np.zeros(K, int); nb_late = np.zeros(K, int)
    epst = np.zeros(K, int); epsum = np.zeros(K); epn = np.zeros(K, int); epshort = np.zeros(K, int)
    epmin = np.full(K, np.inf)                                                        # самый короткий эпизод у набора
    dam_m = np.zeros(K, bool); epst_m = np.zeros(K, int); epsum_m = np.zeros(K); epn_m = np.zeros(K, int); epshort_m = np.zeros(K, int)  # то же по НЕПРЕРЫВНОЙ переменной: без квантования по 1/J
    ep_sleep = np.zeros(K, int); sl_in_ep = np.zeros(K, int); everdam = np.zeros(K, bool)
    wkst = np.zeros(K, int); wksum = np.zeros(K); wkn = np.zeros(K, int); wkmax = np.zeros(K); wklong = np.zeros(K, int); wk15 = np.zeros(K, int)
    taw = np.zeros(K); tdam = np.zeros(K); cycn = np.zeros(K, int); cyc1 = np.zeros(K); cyc2 = np.zeros(K)
    Dsum = np.zeros(K); Esum = np.zeros(K); Lsum = np.zeros(K); Dmsum = np.zeros(K); Gsum = np.zeros(K)
    prbar = np.zeros(K); prdam = np.zeros(K)
    bmax = np.zeros(K)
    dbin_s = np.zeros((K, NDB)); dbin_n = np.zeros((K, NDB))          # кривая «длина бодрствования против повреждения», по каждому набору
    lb_s = np.zeros((K, NLB)); lb_n = np.zeros((K, NLB))              # то же против срока до срыва
    hb_s = np.zeros(K); hb_n = np.zeros(K)                            # собственный здоровый уровень набора (повреждение ниже 0,1)
    hD = np.zeros((K, NHD)); hDm = np.zeros((K, NHD))                 # построчно: иначе блоки перебора смешались бы
    ring_l = np.zeros((K, RING)); ring_t = np.full((K, RING), -1.0); ptr = np.zeros(K, int); lastdam = np.full(K, -1e9); nlead = np.zeros(K, int)
    Dmin = np.ones(K); Dmax = np.zeros(K)
    NGB = 10; gaps = np.zeros((K, NGB)); gap_sum = np.zeros(K); gap_sq = np.zeros(K); gap_n = np.zeros(K); last_end = np.full(K, -1.0)
    NSB = 6; sA_s = np.zeros((K, NSB)); sA_n = np.zeros((K, NSB)); sHs = np.zeros(K); sHn = np.zeros(K)
    wake_half = 15.49 / 2.                                            # половина свободного бодрствования, часы: для Ф2(а)
    prev_aw = np.ones(K, bool); nst = int((warm + ndays) * M); w0 = int(warm * M)
    for i in range(nst):
        dn = np.clip((xhi2 - x) / np.maximum(xhi2 - xlo2, 1e-9), 0, 1)
        D = (x < thr2).mean(1); Dm = dn.mean(1)
        bmax = np.maximum(bmax, D)
        Dcpl = udm * Dm + (1. - udm) * D                                               # через что идёт связь: среднее повреждение или доля
        t = i * dth; Dv = B0 + NVC * math.cos(2 * math.pi * (t - ALPHA) / 24.) + AG * G - kd * Dcpl + zz
        for _ in range(NSUB):
            Vv += (NVM * Q(Vm) + Dv - Vv) * av; Vm += (A_M + NMV * Q(Vv) - Vm) * am
        qm = Q(Vm); awake = qm > 1.; G = np.where(awake, 1 + (G - 1) * er, G * ed)
        g = 1. - GDEP * Dcpl; Pe = PE0 + PE1 * np.minimum(qm / qw, 1.0)
        for _ in range(NWC):
            E += LAM * (F(C1 * E - C2 * g * I + Pe, AE_S, THE) - E); I += LAM * (F(C3 * E - C4 * g * I, AI_S, THI) - I)
            Er += LAM * (F(C1 * Er - C2 * Ir + Pe, AE_S, THE) - Er); Ir += LAM * (F(C3 * Er - C4 * Ir, AI_S, THI) - Ir)
        if np.any(sdz > 0): zz = zz * evz + sdz * tl(rngz.standard_normal(nbase))
        # нагрузка: суточный член несут сомногены, вклад повреждения — отношение активности к неповреждённой при том же бодрствовании
        if np.any(psr > 0):
            hit = tl(rngr.random(nbase)) < psr
            if hit.any(): sr[hit] = 1.0; last_sr[hit] = i / float(M); n_sr[hit] += 1
            sr *= esr
        Linf = nu0 * (1 + aH * (G - Gbar)) * (E / np.maximum(Er, 1e-9)) * (1 + (sramp - 1) * sr) * np.exp(eta + etas - shift); L = Linf + (L - Linf) * eL
        nu = np.maximum(L[:, None] * v, 1e-12)
        eps = P['e0'][:, None] * c ** P['gam'][:, None]; b = eps + x - nu; s_ = np.sqrt(b * b + 4 * eps * nu)
        u = np.where(b > 0, 2 * nu / (b + s_), (s_ - b) / (2 * np.maximum(eps, 1e-30)))
        aa = c ** P['m'][:, None]; bb = aa + P['mu0'][:, None] * u * u / (1 + u); xinf = aa / bb; x = xinf + (x - xinf) * np.exp(-bb * dt[:, None])
        # защита в две ступени: индукция -> сигнал (12 ч) -> белок (63 ч). Стационар тот же, что был,
        # поэтому окно, границы и весь раздел 1в не меняются: меняется только ход во времени.
        drv = u / (P['KB'][:, None] + u); prm = drv + (prm - drv) * epr                     # 0 <= p <= 1, постоянная taup
        minf = P['beta'][:, None] * (1 + apr2 * prm) * drv; mR = minf + (mR - minf) * em      # усиление индукции примированием
        sP = mR + (sP - mR) * es
        cbi = P['N0'][:, None] / (1 + u / P['KN'][:, None]); cb = cbi + (cb - cbi) * ec
        c = np.maximum(cb + sP, 1e-9)
        eta = eta * ea + sdn * fastn(); etas = etas * eas + sds * tl(rngs.standard_normal(nbase))
        fall_ = prev_aw & ~awake; wake_ = (~prev_aw) & awake
        if fall_.any():
            idf = np.where(fall_)[0]; d_ = (i - wkst[idf]) / float(M)
            if i >= w0:
                wkn[idf] += 1; wksum[idf] += d_; wkmax[idf] = np.maximum(wkmax[idf], d_); wklong[idf] += (d_ > 2.0); wk15[idf] += (d_ > 1.5)
                jb = np.clip((bmax[idf] * NDB).astype(int), 0, NDB - 1)
                dbin_s[idf, jb] += d_; dbin_n[idf, jb] += 1.0
                hm = bmax[idf] < 0.1; hb_s[idf[hm]] += d_[hm]; hb_n[idf[hm]] += 1.0
                sl_in_ep[idf[dam[idf]]] += 1                                    # ночь, случившаяся во время эпизода
                ring_l[idf, ptr[idf]] = d_; ring_t[idf, ptr[idf]] = i / float(M); ptr[idf] = (ptr[idf] + 1) % RING
            bmax[idf] = 0.0
        if wake_.any():
            wkst[wake_] = i
            if i >= w0:
                cycn[wake_] += 1; fst = cycn[wake_] == 1
                cyc1[wake_] = np.where(fst, i / float(M), cyc1[wake_]); cyc2[wake_] = i / float(M)
        prev_aw = awake.copy()
        nd_ = D > DTHR; down = nd_ & ~dam; up = (~nd_) & dam
        qual = (i / float(M) - lastdam) >= 3.0
        if down.any(): epst[down] = i
        if i >= w0:
            if down.any():
                nb[down] += 1; everdam[down] = True; aw_ = down & awake; nb_aw[aw_] += 1
                nb_late[aw_ & (((i - wkst) / float(M) * 24.) > wake_half)] += 1
                nb_trig[down & ((i / float(M) - last_sr) < 0.5)] += 1                  # срыв в пределах 12 часов после всплеска
                fb = np.where(down & qual)[0]
                if len(fb):
                    tb = i / float(M); ld = lastdam[fb]
                    for jj in range(RING):
                        lead = tb - ring_t[fb, jj]; ok = (ring_t[fb, jj] >= 0) & (lead > 0) & (lead < NLB) & (ring_t[fb, jj] > ld)
                        if ok.any():
                            jl = lead[ok].astype(int); lb_s[fb[ok], jl] += ring_l[fb[ok], jj]; lb_n[fb[ok], jl] += 1.0
                    nlead[fb] += 1
            if up.any():
                dur_ = (i - epst[up]) / float(M); epsum[up] += dur_; epn[up] += 1; epshort[up] += (dur_ < 1.0); ep_sleep[up & ~awake] += 1
                epmin[up] = np.minimum(epmin[up], dur_)
                last_end[up] = i / float(M)
            if down.any():                                                            # промежуток между концом прошлого эпизода и началом этого
                gd = np.where(down & (last_end >= 0))[0]
                if len(gd):
                    gg = i / float(M) - last_end[gd]; gap_sum[gd] += gg; gap_sq[gd] += gg * gg; gap_n[gd] += 1
                    np.add.at(gaps, (gd, np.clip(gg.astype(int), 0, NGB - 1)), 1.0)
            sm = sP.mean(1)                                                           # средний по клеткам уровень белка-защиты
            hm = D < 0.1; sHs[hm] += sm[hm]; sHn[hm] += 1                             # его здоровый уровень
            af = (~nd_) & (last_end >= 0)                                             # время, прошедшее с конца последнего эпизода
            if af.any():
                ia = np.where(af)[0]; jb = np.clip((i / float(M) - last_end[ia]).astype(int), 0, NSB - 1)
                np.add.at(sA_s, (ia, jb), sm[ia]); np.add.at(sA_n, (ia, jb), 1.0)
            Dmin = np.minimum(Dmin, D); Dmax = np.maximum(Dmax, D)
            pm_ = prm.mean(1); prbar += pm_ * dt; prdam += pm_ * nd_ * dt
            taw += awake * dt; tdam += nd_ * dt; Dsum += D * dt; Dmsum += Dm * dt; Esum += E * dt; Lsum += L * dt; Gsum += G * dt
            hD[ar, np.clip((D * NHD).astype(int), 0, NHD - 1)] += 1.0; hDm[ar, np.clip((Dm * NHD).astype(int), 0, NHD - 1)] += 1.0
        nm_ = Dm > DTHR; dwn_m = nm_ & ~dam_m; up_m = (~nm_) & dam_m        # эпизод по среднему повреждению: порог не квантован, поэтому годится для сверки по J
        if dwn_m.any(): epst_m[dwn_m] = i
        if i >= w0 and up_m.any():
            dm_ = (i - epst_m[up_m]) / float(M); epsum_m[up_m] += dm_; epn_m[up_m] += 1; epshort_m[up_m] += (dm_ < 1.0)
        dam_m = nm_
        lastdam = np.where(nd_, i / float(M), lastdam); dam = nd_
    tot = ndays * day
    prdam = prdam / np.maximum(tdam, 1e-12)                                                 # уровень примирования ИМЕННО во время эпизода
    per = np.where(cycn > 1, (cyc2 - cyc1) / np.maximum(cycn - 1, 1) * 24., np.nan)
    return dict(nb=nb, nb_aw=nb_aw, nb_late=nb_late, epn=epn, epsum=epsum, epshort=epshort, epn_m=epn_m, epsum_m=epsum_m, epshort_m=epshort_m, epmin=epmin, ep_sleep=ep_sleep, sl_in_ep=sl_in_ep,
                everdam=everdam, open_ep=dam, n_sr=n_sr, nb_trig=nb_trig, wkn=wkn, wksum=wksum, wkmax=wkmax, wklong=wklong, wk15=wk15,
                taw=taw / tot, tdam=tdam / tot, prbar=prbar / tot, prdam=prdam, per=per, nday=nb / float(ndays),
                Dbar=Dsum / tot, Dmbar=Dmsum / tot, Gbar_=Gsum / tot, Ebar=Esum / tot, Lbar=Lsum / tot,
                Dmin=Dmin, Dmax=Dmax, nlead=nlead, gaps=gaps, gap_sum=gap_sum, gap_sq=gap_sq, gap_n=gap_n,
                s_after=(sA_s, sA_n), s_health=(sHs, sHn), NGB=NGB, NSB=NSB, nbase=nbase, NB=NB, dbin=(dbin_s, dbin_n), lbin=(lb_s, lb_n), hbase=(hb_s, hb_n), hist=(hD, hDm))

ENSK = ('hist', 'gaps', 's_after')      # накопители на весь блок целиком: их по блоку надо СКЛАДЫВАТЬ, а не резать

def split(rr):
    """Разрезать результат батча на блоки перебора. Каждый блок — такой же словарь, как у одиночного прогона,
    поэтому всё, что читает результат, не меняется вовсе."""
    nbase, NB = rr['nbase'], rr['NB']; out = []
    for b in range(NB):
        sl = slice(b * nbase, (b + 1) * nbase); d = {}
        for k, z in rr.items():
            if k in ('NGB', 'NSB', 'nbase', 'NB'): d[k] = z
            elif k in ENSK: d[k] = tuple(np.asarray(y)[sl].sum(0) for y in z) if isinstance(z, tuple) else np.asarray(z)[sl].sum(0)
            elif isinstance(z, tuple): d[k] = tuple(np.asarray(y)[sl] for y in z)
            else: d[k] = np.asarray(z)[sl]
        d['nbase'] = nbase; d['NB'] = 1; out.append(d)
    return out

def _job(a):
    """Один кусок батча в отдельном процессе. Шум берётся ПО ЗЕРНУ, а не передаётся готовым объектом,
    и разыгрывается на длину одного блока: поэтому ответ не зависит от того, на сколько процессов разрезан батч."""
    P, n0, th, xh, xl, kw, nbase, sd = a
    rng = [np.random.default_rng(z) for z in sd] if isinstance(sd, (list, tuple)) else np.random.default_rng(sd)
    return split(simulate(P, n0, th, xh, xl, rng=rng, nbase=nbase, **kw))

def trace(p, nu0, thr, xhi, xlo, kd, aH, sigv, E0, qw, Gbar, sigma, tauc_rel, ndays, M, J, rng, warm=6, sigs=0.0):
    """Одна траектория для рисунка: время (сутки), бодрствование, сомногены, активность, нагрузка, доля повреждённых, среднее повреждение."""
    day = p['r'] * p['T']; dt = day / M; dth = 24. / M; ds = dth / NSUB; v = vuln(J, sigv)[0]; tauL = 0.1 * p['T']
    ea = math.exp(-dt / (tauc_rel * p['T'])); sdn = sigma * math.sqrt(1 - ea * ea); ec = math.exp(-dt / p['T']); eL = math.exp(-dt / tauL)
    av = 1 - math.exp(-ds / TAUV); am = 1 - math.exp(-ds / TAUM); er = math.exp(-dth / TR); ed = math.exp(-dth / TD)
    evz = math.exp(-dth / TAUSL); sdz = SLPN * math.sqrt(1 - evz * evz); zz = 0.0; rngz = np.random.default_rng(99)
    eas = math.exp(-1.0 / (TSLOW * M)); sds = sigs * math.sqrt(1 - eas * eas); rngs = np.random.default_rng(77); etas = 0.0; shift = 0.5 * sigs * sigs
    x = np.ones(J); cb = np.full(J, p['N0']); mR = np.zeros(J); sP = np.zeros(J); c = np.full(J, p['N0'])
    em = math.exp(-dt / (TAU_M * day)); es = math.exp(-dt / (TAU_S * day)); eta = 0.0; L = nu0
    Vv, Vm, G, E, I, Er, Ir = -5., A_M, 0.3, 0.3, 0.2, 0.3, 0.2; out = []
    for i in range(int((warm + ndays) * M)):
        dn = np.clip((xhi - x) / max(xhi - xlo, 1e-9), 0, 1); D = float((x < thr).mean()); Dm = float(dn.mean())
        t = i * dth; Dv = B0 + NVC * math.cos(2 * math.pi * (t - ALPHA) / 24.) + AG * G - kd * (Dm if USE_DM else D) + zz
        zz = zz * evz + sdz * rngz.standard_normal(); etas = etas * eas + sds * rngs.standard_normal()
        for _ in range(NSUB):
            Vv += (NVM * Q(Vm) + Dv - Vv) * av; Vm += (A_M + NMV * Q(Vv) - Vm) * am
        qm = Q(Vm); aw = qm > 1.; G = 1 + (G - 1) * er if aw else G * ed
        g = 1. - GDEP * (Dm if USE_DM else D); Pe = PE0 + PE1 * min(qm / qw, 1.0)
        for _ in range(NWC):
            E += LAM * (F(C1 * E - C2 * g * I + Pe, AE_S, THE) - E); I += LAM * (F(C3 * E - C4 * g * I, AI_S, THI) - I)
            Er += LAM * (F(C1 * Er - C2 * Ir + Pe, AE_S, THE) - Er); Ir += LAM * (F(C3 * Er - C4 * Ir, AI_S, THI) - Ir)
        Linf = nu0 * (1 + aH * (G - Gbar)) * (E / max(Er, 1e-9)) * math.exp(eta + etas - shift); L = Linf + (L - Linf) * eL
        nu = np.maximum(L * v, 1e-12); eps = p['e0'] * c ** p['gam']; b = eps + x - nu; s_ = np.sqrt(b * b + 4 * eps * nu)
        u = np.where(b > 0, 2 * nu / (b + s_), (s_ - b) / (2 * np.maximum(eps, 1e-30)))
        aa = c ** p['m']; bb = aa + p['mu0'] * u * u / (1 + u); xi = aa / bb; x = xi + (x - xi) * np.exp(-bb * dt)
        minf = p['beta'] * u / (p['KB'] + u); mR = minf + (mR - minf) * em; sP = mR + (sP - mR) * es
        cbi = p['N0'] / (1 + u / p['KN']); cb = cbi + (cb - cbi) * ec; c = np.maximum(cb + sP, 1e-9)
        eta = eta * ea + sdn * rng.standard_normal()
        if i >= warm * M and i % 10 == 0: out.append((i / float(M) - warm, 1.0 if aw else 0.0, G, E, L / nu0, D, float(sP.mean())))
    return np.array(out)

# ---------- Основной расчёт ----------
def main():
    K, NSET, ND, M, J = (3000, 10, 10, 600, 8) if SMOKE else ((200000, 300, 60, 1600, 32) if FULL else (40000, 200, 40, 1200, 24))
    NSET = argint('sets', NSET); ND = argint('days', ND); M = argint('m', M); J = argint('j', J)
    KSCAN = [0.0, 2.0, 4.5] if SMOKE else ([0., 1., 2., 3., 4., 4.25, 4.5, 4.75, 5.] if FULL else [0., 1., 2., 3., 4., 4.5, 5.])
    QSCAN = [0.1, 0.3] if SMOKE else ([0.05, 0.1, 0.15, 0.2, 0.3] if FULL else [0.05, 0.1, 0.2])
    ASCAN = [0.0, 1.0] if SMOKE else ([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0] if FULL else [0.0, 0.5, 1.0, 2.0])
    WARM, WARM2 = (4, 3) if SMOKE else (10, 6)                                           # разгон: полные прогоны и переборы
    VSCAN = [0.10] if SMOKE else ([0.05, 0.10, 0.15, 0.25, 0.40] if FULL else [0.05, 0.10, 0.20])
    SSCAN = [0.0, 0.12] if SMOKE else ([0.0, 0.04, 0.08, 0.12, 0.16] if FULL else [0.0, 0.08, 0.16])
    SIGS_BASE = 0.0                                                                      # РАБОЧЕЕ значение медленного шума теперь НОЛЬ (см. раздел 4ж)
    SIGS_CTL = 0.08                                                                      # прежнее рабочее значение остаётся ПАРНЫМ КОНТРОЛЕМ
    if SIGS_CTL not in SSCAN: SIGS_CTL = min((v for v in SSCAN if v > 0), key=lambda v: abs(v - SIGS_CTL))
    WARMS = (6, 16)[0 if SMOKE else 1]                                                    # разгон для медленного шума: не меньше трёх времён корреляции
    NREP = 2 if SMOKE else 3
    SIG, TAUC, QPOS, KBASE, SIGV, SEED = 0.08, 0.3, 0.10, 2.0, 0.10, 4242
    RSCAN = [0.05, 0.2] if SMOKE else ([0.02, 0.05, 0.1, 0.2, 0.5] if FULL else [0.05, 0.1, 0.3])     # всплесков в сутки
    ASCAN_SR = [1.5] if SMOKE else ([1.2, 1.4, 1.6, 1.8, 2.0] if FULL else [1.3, 1.6, 2.0])           # во сколько раз всплеск поднимает нагрузку
    nrun_full = len(KSCAN) + 2 * (NREP - 1) + len(SSCAN); nrun_half = len(ASCAN) + len(VSCAN) + len(QSCAN) + len(RSCAN) + len(ASCAN_SR) + 1
    est = 1.77e-6 * M / 2000. * 1600 * (nrun_full * (ND + WARM) + nrun_half * (ND // 2 + WARM2)) * 1000
    print('Перебор идёт батчами: все точки сложены в одну матрицу строк и считаются одним циклом.', flush=True)
    print('Оценка времени счёта по старой, последовательной схеме: около %.0f мин (%d полных прогонов и %d коротких). Батчами ожидается в несколько раз быстрее; настоящее время печатается по каждому батчу. Параметры размера можно задать ключами --sets= --days= --m= --j=.' % (est / 60., nrun_full, nrun_half), flush=True)
    NPROC = argint('proc', 0) or (os.cpu_count() or 1)                                   # сколько процессов на батч; --proc=1 отключает
    print('Процессов на батч: до %d.' % NPROC, flush=True)
    os.makedirs(OUT, exist_ok=True); rng = np.random.default_rng(83); R = {}
    L = ['# Нейронная масса E–PV, популяция PV-клеток и переключатель бодрствования Филлипса–Робинсона\n',
         'Режим: %s. Выборка %d наборов параметров; в расчёт берётся до %d наборов с двумя состояниями; %d клеток в популяции; %d суток на набор; %d шагов на сутки.\n' % ('пробный' if SMOKE else ('полный' if FULL else 'обычный'), K, NSET, J, ND, M),
         'Шаг А: PV-клетки стали популяцией с разбросом уязвимости, и торможение в коре зависит от ДОЛИ повреждённых. Шаг Б: расписание сна заменено переключателем Филлипса–Робинсона, порогов H и L и потолка H_CAP больше нет.\n',
         'Редокс-ядро не менялось. Связь «повреждение мешает заснуть» вставлена как торможение сонного ядра: D_v = B0 + nu_vc*cos − k*D плюс вес сомногенов.\n',
         'Против mass_core.py изменено семь вещей: сомногены получили несимметричные постоянные времени Даана (18,2 ч вверх, 4,2 ч вниз) вместо одной 45 ч; суточный член нагрузки перенесён с активности коры на сомногены; вклад активности взят прямо пропорциональным (A_E = 1) и отделён от бодрствования делением на неповреждённую активность при том же состоянии; k_база = %.1f мВ; главные числа считаются по эпизодической подгруппе; Ф4 печатается ещё и с нормировкой на собственный здоровый уровень набора; Ф2(в) меряется двумя способами.\n' % 2.0]

    # ---------- 0. Проверки до всякого счёта ----------
    stage('проверки')
    fl = pr_folds(); wH, sH, pH, g0_, g1_, qw0, dec_, _, _ = pr_free()
    wN, sN, pN, _, _, _, _, sdw, sds_ = pr_free(slpn=SLPN, K=(8 if SMOKE else 48), nd=(60 if SMOKE else 300), M=800)
    E0, qw, Gbar, g0_, g1_, A_H, ew, es = calibrate()
    gg = np.array([1.0, 0.9, 0.8, 0.7, 0.6]); Ew_, Iw_, ev_w = wc_fix(gg, PE0 + PE1); Es_, Is_, ev_s = wc_fix(gg, PE0)
    Eh, Ih, evh = wc_fix(gg, PE0 + PE1, E=np.full(5, 0.95), I=np.full(5, 0.95))
    uniq = float(np.abs(Ew_ - Eh).max())
    # быстрая итерация коры против эталона
    Ef, If_ = np.full(5, 0.3), np.full(5, 0.2)
    for _ in range(NWC * 40):
        Ef += LAM * (F(C1 * Ef - C2 * gg * If_ + PE0 + PE1, AE_S, THE) - Ef); If_ += LAM * (F(C3 * Ef - C4 * gg * If_, AI_S, THI) - If_)
    wcres = float(np.abs(Ef - Ew_).max())
    # экспоненциальный Эйлер против точного решателя на одной клетке при постоянной нагрузке
    pk = {'e0': 0.02, 'mu0': 0.3, 'm': 2.0, 'gam': 0.5, 'KN': 3.0, 'N0': 0.9, 'beta': 0.1, 'KB': 10.0, 'T': 50.0}
    wk_ = window(curve(pk)[0]); nuM = math.sqrt(wk_[0] * wk_[1]); st = states_at(pk, nuM)
    xx, cc_ = 1.0, pk['N0']; dtq = 3 * pk['T'] / M
    for i in range(M):
        e_ = pk['e0'] * cc_ ** pk['gam']; b_ = e_ + xx - nuM; s_ = math.sqrt(b_ * b_ + 4 * e_ * nuM); uu = 2 * nuM / (b_ + s_) if b_ > 0 else (s_ - b_) / (2 * e_)
        a_ = cc_ ** pk['m']; bq = a_ + pk['mu0'] * uu * uu / (1 + uu); xi = a_ / bq; xx = xi + (xx - xi) * math.exp(-bq * dtq)
        ci = pk['N0'] / (1 + uu / pk['KN']) + pk['beta'] * uu / (pk['KB'] + uu); cc_ = ci + (cc_ - ci) * math.exp(-dtq / pk['T'])
    sol = solve_ivp(lambda t, y: rhs(t, y, pk, nuM), [0, M * dtq], [1.0, pk['N0']], method='LSODA', rtol=1e-10, atol=1e-13)
    dx = abs(xx - sol.y[0, -1]); dc = abs(cc_ - sol.y[1, -1])
    vv_ = vuln(J, SIGV)
    L += ['## 0. Проверки\n', '### Переключатель бодрствования: сверка с опубликованными числами\n', '| величина | получено здесь | опубликовано | источник |\n|---|---|---|---|',
          '| складки переключателя D_v, мВ | %s | 1,45 и 2,46 | Phillips, Robinson 2008; Skeldon и соавт. 2014 |' % (' и '.join('%.3f' % f for f in fl)),
          '| бодрствование, ч | %.2f | 15,5 | то же |' % wH, '| сон, ч | %.2f | 8,5 | то же |' % sH, '| период, ч | %.2f | 24,0 | то же |' % pH,
          '| размах сомногенов G | %.3f – %.3f | 0,17 – 0,67 (Даан: давление сна S) | Daan, Beersma, Borbely 1984 |' % (g0_, g1_),
          '| замедление подъёма сомногенов за бодрствование | %.0f %% | — | следствие постоянной 18,2 ч |' % dec_,
          '| с шумом сна: бодрствование, ч | %.2f ± %.2f (SD) | — | — |' % (wN, sdw),
          '| с шумом сна: сон, ч | %.2f ± %.2f (SD) | SD 1,29 ч актиграфия, 1,12 ч ЭЭГ, 0,54 ч самоотчёт | Messman и соавт. 2022 (26 121 ночь); Gao, Scullin 2024 |' % (sN, sds_),
          '\nСкладки и постоянные переключателя взяты из статьи и не настраивались. Сомногены теперь живут по Даану: подъём с постоянной %.1f ч, спад с %.1f ч, то есть спад втрое с лишним быстрее подъёма. У Филлипса–Робинсона на оба направления стояла одна постоянная 45 ч, и пила выходила почти прямой — подъём замедлялся всего на треть, спад был таким же медленным, как подъём. Это и поправлено.\n' % (TR, TD),
          'Подогнано здесь ровно одно число: постоянная часть притока B0 = %.3f мВ, по длительности бодрствования 15,5 ч. Вес сомногенов AG = %.1f мВ ни на что не влияет в пределах от 4 до 7 мВ (B0 его полностью компенсирует, проверено отдельно). Размах G не подгонялся и вышел %.3f против %.3f у Даана — это независимое совпадение.\n' % (B0, AG, g1_ - g0_, 0.50),
          '### Кора E–PV: стационар единственный и устойчивый\n', '| g = 1 − 0,4·D | E в бодрствовании | PV в бодрствовании | наибольшая действительная часть собственного числа | E во сне |\n|---|---|---|---|---|']
    for i_ in range(5): L.append('| %.2f | %.3f | %.3f | %+.3f | %.3f |' % (gg[i_], Ew_[i_], Iw_[i_], ev_w[i_], Es_[i_]))
    L += ['\n- запуск из далёкой точки (E = I = 0,95) даёт то же решение, расхождение %.2e — стационар единственный' % uniq,
          '- быстрая итерация в счёте (%d шага, затухание %.2f) против эталона: расхождение %.2e' % (NWC, LAM, wcres),
          '- все собственные числа отрицательны, то есть колебаний нет: кора взята в стационарном режиме НАМЕРЕННО, гамма-ритма в этой версии нет',
          '- при падении торможения с g = 1 до g = 0,6 активность E растёт на %.1f %% — это и есть передача «меньше PV → больше возбуждения»' % (100 * (Ew_[-1] / Ew_[0] - 1)),
          '\n### Нагрузка: два члена и ни одного подогнанного\n',
          '- суточный член несут сомногены: нагрузка умножается на (1 + A_H*(G − среднее G)), A_H = %.3f выбран так, чтобы размах от впадины к пику вышел %.2f (число из литературы: потребление глюкозы в коре мыши −16 %% во сне, CMRO2 у человека −21 %% при засыпании, 8-oxo-dG в PV-клетках +25 %% после нарушенного сна)' % (A_H, DSWING),
          '- вклад повреждения несёт активность коры: нагрузка умножается на E/E_неповр, то есть на отношение к активности при том же бодрствовании, но без повреждения. Показатель 1: нагрузка прямо пропорциональна частоте разрядов. Свободного числа здесь нет вовсе',
          '- деление на E_неповр нужно, чтобы бодрствование не считалось дважды: суточный размах уже учтён сомногенами',
          '- средняя активность за свободный цикл E0 = %.4f; в бодрствовании %.4f, во сне %.4f (теперь это не источник суточного размаха, а только опорный уровень)' % (E0, ew, es),
          '- при полном повреждении (D = 1) нагрузка поднимается на %.1f %% — следствие gdep = %.2f, не подбиралось' % (100 * (Ew_[-1] / Ew_[0] - 1), GDEP),
          '\n### Разброс уязвимости и разрешение по доле\n',
          '- множители нагрузки на клетку при sigma_v = %.2f: от %.3f до %.3f, среднее %.3f' % (SIGV, vv_.min(), vv_.max(), vv_.mean()),
          '- доля повреждённых D меняется шагами по 1/%d = %.3f; меньше этого модель различать не может' % (J, 1. / J),
          '\n### Что даёт связь «повреждение мешает заснуть»: лестница пропущенных ночей\n',
          '| k, мВ | бодрствование при D = 1, ч | сон, ч | период, ч |\n|---|---|---|---|']
    kstair = []
    for kk in KSCAN:
        w_, s2, p2, _, _, _, _, _, _ = pr_free(kd=kk, nd=40); kstair.append(dict(k=kk, wake_h=w_, sleep_h=s2, per_h=p2))
        L.append('| %.1f | %s | %s | %s |' % (kk, ('%.2f' % w_) if w_ == w_ else 'сна нет', ('%.2f' % s2) if s2 == s2 else '—', ('%.1f' % p2) if p2 == p2 else '—'))
    L += ['\n**Шум сна.** К притоку к сонному ядру добавлен шум Орнштейна–Уленбека, амплитуда %.2f мВ, время корреляции %.1f ч. Он зашумляет И засыпание, И подъём — это один источник на оба перехода, а не два независимых. Амплитуда выбрана так, чтобы разброс длительности сна вышел около 0,7 ч. Литературные числа (1,12–1,29 ч) включают режим дня — будильник, выходные, — которого в модели нет вовсе, поэтому взята нижняя часть диапазона, а не середина; это моё решение, и оно перебирается в разделе 7.\n' % (SLPN, TAUSL),
          '\nПеребор доведён до обрыва намеренно: в прогоне с k до 4 мВ правило Ф3(б) выходило нулевым при ВСЕХ k просто потому, что бодрствование не дотягивало до суток (22,5 ч при k = 4), а полоса с пропуском ночи лежит между 4 и 5 мВ.',
          '\nЛестница ровная и кончается обрывом: бодрствование плавно удлиняется, а дальше сон не наступает вовсе. Потолка H_CAP, который создавал полки в sleep_core2.py, здесь нет — это складки самого переключателя. С постоянными Даана связь действует втрое сильнее, чем с одной постоянной 45 ч (прежде 15,5 → 18,8 ч при k = 4; теперь столько набирается уже при k = 1).\n',
          '### Шаг по времени\n', '- экспоненциальный Эйлер против точного решателя на одной клетке при постоянной нагрузке: расхождение по x %.2e, по c %.2e' % (dx, dc),
          '- проверочный набор: окно от %.3f до %.3f, устойчивых состояний в его середине %d (должно быть 2)' % (wk_[0], wk_[1], len(st))]
    R['checks'] = dict(pr_folds=fl, pr_folds_published=[1.45, 2.46], wake_h=wH, sleep_h=sH, period_h=pH, Q_wake=qw,
                       G_range=[g0_, g1_], G_range_daan=[0.17, 0.67], G_mean=Gbar, rise_decel_pct=dec_, tau_rise_h=TR, tau_decay_h=TD, B0=B0, AG=AG,
                       E0=E0, E_wake=ew, E_sleep=es, A_H=A_H, wc_E_wake=[float(z) for z in Ew_], wc_E_sleep=[float(z) for z in Es_],
                       wc_eig_max=float(max(ev_w.max(), ev_s.max())), wc_unique_gap=uniq, wc_iter_resid=wcres,
                       E_gain_full_damage=float(Ew_[-1] / Ew_[0] - 1), load_gain_full_damage=float(Ew_[-1] / Ew_[0] - 1),
                       D_resolution=1. / J, v_min=float(vv_.min()), v_max=float(vv_.max()), k_stair=kstair,
                       dx_vs_solver=float(dx), dc_vs_solver=float(dc), n_states_check=len(st))
    if not (1.40 < fl[0] < 1.50 and 2.40 < fl[1] < 2.52): L.append('\n**ВНИМАНИЕ: складки не совпали с опубликованными. Дальше считать нельзя, набор параметров переключателя неверен.**')

    # ---------- 1. Отбор наборов ----------
    stage('выборка %d наборов' % K); S = sample(K, rng); sel = []
    for k in range(K):
        p = {n: float(S[n][k, 0]) for n in NAMES}; w = window(curve(p)[0])
        if not w or w[1] / w[0] <= 1.05: continue
        stt = states_at(p, math.sqrt(w[0] * w[1]))
        if len(stt) < 2: continue
        sel.append(dict(k=k, lo=w[0], hi=w[1], xlo=stt[0], xhi=stt[-1], r=float(rng.uniform(2, 4))))
        if len(sel) >= NSET: break
    nsel = len(sel)
    L += ['\n## 1. Наборы\n', '- взято в расчёт наборов с двумя состояниями: %d' % nsel]
    if nsel < 3:
        L.append('\nНаборов слишком мало.'); open(os.path.join(OUT, 'report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n'); print('\n'.join(L)); return
    P = {n: np.array([float(S[n][d['k'], 0]) for d in sel]) for n in NAMES}; P['r'] = np.array([d['r'] for d in sel])
    lo = np.array([d['lo'] for d in sel]); hi = np.array([d['hi'] for d in sel])
    xlo = np.array([d['xlo'] for d in sel]); xhi = np.array([d['xhi'] for d in sel]); thr = 0.5 * (xlo + xhi)
    q = lambda v, f='%.3f': (f + ' (10–90 %%: ' + f + ' – ' + f + ')') % tuple(np.nanpercentile(v, [50, 10, 90]))
    L += ['- отношение границ окна: ' + q(hi / lo, '%.2f'), '- T: ' + q(P['T'], '%.0f'), '- период внутренних часов в единицах T: ' + q(P['r'], '%.2f'),
          '\nПорог срыва у набора один для всех клеток: середина между устойчивыми ветвями при опорной нагрузке. Клетки различаются не порогом, а нагрузкой L*v_j.']

    # ---------- 1б. Запас до верхней границы: условие, записанное до счёта ----------
    gain_fb = float(Ew_[-1] / Ew_[0]); day_pk = 1 + A_H * (g1_ - Gbar); noi = math.exp(SIG); need = gain_fb * day_pk * noi
    hr = (hi / lo) ** (1 - QPOS)
    L += ['\n## 1б. Хватает ли запаса, чтобы эпизод мог кончиться\n',
          'Это посчитано до всякого счёта и ничего не подгоняет. Над средней нагрузкой должны уместиться сразу три множителя; если запас до верхней границы окна меньше их произведения, то после первого же срыва обратная связь сама дотолкает нагрузку до верхней границы, и эпизод кончиться НЕ СМОЖЕТ — получится не эпизод, а необратимый переход.\n',
          '| что должно уместиться | множитель |\n|---|---|',
          '| обратная связь при полном повреждении (E/E_неповр при D = 1) | %.3f |' % gain_fb,
          '| суточный пик нагрузки | %.3f |' % day_pk, '| шум, одно сигма | %.3f |' % noi,
          '| **вместе** | **%.3f** |' % need,
          '\n| q | запас (hi/lo)^(1−q): медиана | доля наборов, которым запаса хватает |\n|---|---|---|']
    for qq in sorted(set(QSCAN + [QPOS, 0.75])):
        h_ = (hi / lo) ** (1 - qq); L.append('| %.2f%s | %.3f | %.3f |' % (qq, ' (рабочее)' if abs(qq - QPOS) < 1e-9 else '', float(np.median(h_)), float((h_ > need).mean())))
    L += ['\nРабочее положение q перенесено с 0,75 на %.2f именно по этому условию. При 0,75 запаса хватало лишь %.3f наборов, и прошлый прогон это показал прямо: 58 %% наборов с постоянным дефицитом, 55–63 %% эпизодов не кончились за 60 суток, длительность 8–17 суток при ожидаемых часах-сутках. Условие записано как требование к устройству расчёта, а не подобрано под ответ.\n' % (QPOS, float(((hi / lo) ** 0.25 > need).mean())),
          '- наборов, которым запаса хватает при рабочем q: %d из %d (%.3f)' % (int((hr > need).sum()), nsel, float((hr > need).mean()))]
    R['headroom'] = dict(gain_feedback=gain_fb, day_peak=day_pk, noise=noi, need=need, q_work=QPOS,
                         share_ok=float((hr > need).mean()), median_headroom=float(np.median(hr)),
                         by_q={('%.2f' % qq): float((((hi / lo) ** (1 - qq)) > need).mean()) for qq in sorted(set(QSCAN + [QPOS, 0.75]))})

    # ---------- 1в. Закон сохранения разрывов: почему эпизод длинный ----------
    sig_tot = math.sqrt(SIG ** 2 + SIGS_BASE ** 2); bud = 2 * SIG + SIGS_BASE
    gap_up = np.log((hi / lo) ** (1 - QPOS) / day_pk); gap_dn = np.log((hi / lo) ** QPOS * gain_fb * (2 - day_pk))
    up_cap = math.log(SR_AMP * day_pk)                                               # разрыв вверх перекрывает всплеск, а не шум
    Wstar = math.exp((up_cap + bud)) * day_pk / ((2 - day_pk) * gain_fb)
    band = (gap_up <= up_cap) & (gap_dn <= bud)
    L += ['\n## 1в. Почему эпизод получается длинным: закон сохранения разрывов\n',
          'Эпизод начинается, когда нагрузка выбросом вверх перешагивает верхнюю границу окна, и кончается, когда выбросом вниз уходит под нижнюю. Разрывы, которые надо перекрыть:\n',
          '- вверх: ln[ (hi/lo)^(1−q) / суточный пик ]\n- вниз: ln[ (hi/lo)^q · обратная связь · суточная впадина ]\n',
          '**Их сумма не зависит от q вовсе:** ln[ (hi/lo) · %.3f · %.3f / %.3f ] = ln(%.3f · hi/lo). Положение нагрузки в окне только перераспределяет разрывы между «трудно сломаться» и «трудно вернуться», но не уменьшает их сумму. Поэтому длину эпизода нельзя исправить выбором q — это закон сохранения, а не настройка.\n' % (gain_fb, 2 - day_pk, day_pk, gain_fb * (2 - day_pk) / day_pk),
          'Бюджет на ВОЗВРАТ: два быстрых сигма плюс смещение от медленной составляющей = %.3f в логарифме (×%.2f). Бюджет на СРЫВ — это уже не шум, а серотониновый всплеск вместе с суточным пиком: %.3f в логарифме (×%.2f). В этом и состоит слом симметрии: два разрыва меряются разными распределениями.\n' % (bud, math.exp(bud), up_cap, math.exp(up_cap)),
          '**W\\* = %.2f.** У набора с окном шире этого всплеска не хватит на срыв, либо шума не хватит на возврат. Это предсказание модели: шире окно — хронический дефицит, уже — обратимые эпизоды.\n' % Wstar,
          'Честная оговорка: W\\* зависит от двух моих величин — силы медленного шума и амплитуды всплеска. При sigma_м от 0,08 до 0,16 он сдвигается заметно. «Структурный» он только в том смысле, что выведен из уравнений, а не из исхода; параметрически свободным он не является.\n',
          'И вторая: в прошлом прогоне полоса отбиралась по условию «оба перехода укладываются в шум», и это оказалось ошибкой. У отобранных наборов разрыв вверх был 0,9 сигмы, вниз 1,7 — то есть бистабильность слабее шума, и система не имела эпизодов, а дребезжала между ветвями: повреждались 96 % наборов, в среднем повреждено 58 % клеток в любой момент. Здесь условие другое: вверх перекрывает ВСПЛЕСК, вниз — шум.\n',
          '| величина | значение |\n|---|---|',
          '| отношение границ окна: медиана по всем наборам | %.2f |' % float(np.median(hi / lo)),
          '| предельная ширина W\\* | %.2f |' % Wstar,
          '| наборов с окном уже W\\* (эпизодическая полоса) | %d из %d (%.3f) |' % (int(band.sum()), nsel, float(band.mean())),
          '| их окно: медиана | %.2f |' % (float(np.median((hi / lo)[band])) if band.any() else float('nan')),
          '| окно остальных: медиана | %.2f |' % (float(np.median((hi / lo)[~band])) if (~band).any() else float('nan')),
          '\nДальше вердикты выносятся ПО ЭТОЙ ПОЛОСЕ, а числа по всему ансамблю печатаются рядом. Отбор сделан по условию, выведенному из постоянных самой модели и посчитанному до счёта, а не по тому, у кого правила прошли. Исключённые наборы — не «другая болезнь»: модель говорит, что это постоянный дефицит без эпизодов, и это отдельное утверждение, которое тоже надо проверять.',
          '\nРабочее q = %.2f выбрано как середина: оно делит сумму разрывов пополам и даёт наибольшую долю наборов в полосе.' % QPOS]
    R['budget'] = dict(sigma_fast=SIG, sigma_slow=SIGS_BASE, sigma_total=sig_tot, budget=bud, W_star=Wstar,
                       gain_feedback=gain_fb, day_peak=day_pk, band_share=float(band.mean()), band_n=int(band.sum()),
                       W_median_all=float(np.median(hi / lo)), W_median_band=float(np.median((hi / lo)[band])) if band.any() else None,
                       gap_up_median=float(np.median(gap_up)), gap_dn_median=float(np.median(gap_dn)))

    def meas(rr, mask=None):
        mk = np.ones(len(rr['nb']), bool) if mask is None else mask
        rr = {kk: (vv[mk] if isinstance(vv, np.ndarray) and vv.shape[:1] == mk.shape else vv) for kk, vv in rr.items()}
        ep = rr['epn'] > 0; ev = rr['everdam'] | rr['open_ep']; started = rr['nb'] + rr['open_ep'] * 1.0
        dur = np.where(ep, rr['epsum'] / np.maximum(rr['epn'], 1), np.nan)
        epm_ = rr['epn_m'] > 0; durm = np.where(epm_, rr['epsum_m'] / np.maximum(rr['epn_m'], 1), np.nan)
        return dict(n_sets=int(mk.sum()), share_ever=float(ev.mean()), share_break=float((rr['nb'] > 0).mean()),
                    dur=float(np.nanmedian(dur)) if ep.any() else float('nan'),
                    dur_m=float(np.nanmedian(durm)) if epm_.any() else float('nan'),
                    short_m=float(np.nanmedian(np.where(epm_, rr['epshort_m'] / np.maximum(rr['epn_m'], 1), np.nan))) if epm_.any() else float('nan'),
                    n_ep_m=int(rr['epn_m'].sum()),
                    dur_all=float(np.nanmedian(np.where(ev, rr['tdam'] * ND / np.maximum(started, 1), np.nan))) if (started > 0).any() else float('nan'),
                    short=float(np.nanmedian(np.where(ep, rr['epshort'] / np.maximum(rr['epn'], 1), np.nan))) if ep.any() else float('nan'),
                    short_pool=float(rr['epshort'].sum() / max(rr['epn'].sum(), 1)),      # по ВСЕМ эпизодам: медиана по наборам обнуляется, едва короткие есть меньше чем у половины
                    ep_min=float(np.min(rr['epmin'][np.isfinite(rr['epmin'])])) if np.isfinite(rr['epmin']).any() else float('nan'),
                    ep_min_med=float(np.median(rr['epmin'][np.isfinite(rr['epmin'])])) if np.isfinite(rr['epmin']).any() else float('nan'),
                    sleep_end=float(rr['ep_sleep'].sum() / max(rr['epn'].sum(), 1)),
                    sleep_kills=float(rr['ep_sleep'].sum() / max(rr['sl_in_ep'].sum(), 1)),
                    late=float(rr['nb_late'].sum() / max(rr['nb_aw'].sum(), 1)),
                    trig=float(rr['nb_trig'].sum() / max(rr['nb'].sum(), 1)), n_sr=float(rr['n_sr'].mean()),
                    open_ep=float(rr['open_ep'].mean()), stuck=float((rr['wklong'] > 0).mean()), stuck15=float((rr['wk15'] > 0).mean()),
                    wkmax=float(np.median(rr['wkmax'])), wkmean=float(np.median(np.where(rr['wkn'] > 0, rr['wksum'] / np.maximum(rr['wkn'], 1), np.nan))),
                    per=float(np.nanmedian(rr['per'])), taw=float(np.median(rr['taw'])),
                    tdam_ev=float(np.median(rr['tdam'][ev])) if ev.any() else 0.0,
                    nday=float(np.median(rr['nday'][rr['nb'] > 0])) if (rr['nb'] > 0).any() else 0.0,
                    Dbar=float(np.median(rr['Dbar'])), Dmbar=float(np.median(rr['Dmbar'])), Ebar=float(np.median(rr['Ebar'])), Lbar=float(np.median(rr['Lbar'])),
                    gap_mean=float(np.nansum(rr['gap_sum']) / max(np.nansum(rr['gap_n']), 1)), n_gap=int(rr['gap_n'].sum()),
                    Dmin=float(np.median(rr['Dmin'])), Dmax=float(np.median(rr['Dmax'])), chronic=float((rr['Dmin'] > 0.1).mean()), swings=float((rr['Dmax'] - rr['Dmin'] > 0.2).mean()))

    # ---------- 2. Парный перебор по силе связи ----------
    num = lo * (hi / lo) ** QPOS; rows = []; rows_b = []; keep = {}
    NRUN = [0, 0.0]
    def runb(nu0s, NBx, nm='', **kw):
        """Один батч: NBx точек перебора сложены в одну матрицу строк и считаются одним циклом,
        а сам батч разрезан по процессам. Шум разыгрывается на длину ОДНОГО блока и повторяется,
        поэтому блоки строго парные, а ответ не зависит ни от числа блоков, ни от числа процессов."""
        n0 = np.tile(nu0s, NBx) if len(nu0s) == nsel else nu0s
        assert len(n0) == nsel * NBx, 'длина nu0 не отвечает числу блоков'
        sd = kw.pop('seed')
        # Переборные ключи задаются массивом длины NBx (значение на блок). РАСКЛАДЫВАЕМ их здесь же
        # на полную длину строк: тогда внутри simulate угадывать уже нечего, и случай «число блоков
        # совпало с числом наборов» перестаёт быть двусмысленным.
        per = {k: np.repeat(np.asarray(kw.pop(k), float), nsel) for k in list(kw)
               if isinstance(kw[k], np.ndarray) and kw[k].size == NBx}
        npr = max(1, min(NPROC, NBx)); jobs = []
        for ix in np.array_split(np.arange(NBx), npr):
            nb_ = len(ix); rws = (ix[:, None] * nsel + np.arange(nsel)).ravel()
            kk = dict(kw); kk.update({k: v[rws] for k, v in per.items()})
            jobs.append(({k: np.tile(z, nb_) for k, z in P.items()}, n0[rws], np.tile(thr, nb_), np.tile(xhi, nb_), np.tile(xlo, nb_),
                         kk, nsel, [sd[i] for i in ix] if isinstance(sd, (list, tuple)) else sd))
        t0 = time.time(); stage('батч %s: %d точек перебора по %d наборов, процессов %d' % (nm, NBx, nsel, len(jobs)))
        if len(jobs) == 1: out = _job(jobs[0])
        else:
            try:
                with mp.Pool(len(jobs)) as pool: out = [b for part in pool.map(_job, jobs) for b in part]
            except Exception as e:                                                      # не вышло с процессами — считаем в одном, ответ тот же
                stage('  процессы недоступны (%s), считаю в одном' % type(e).__name__); out = [b for j in jobs for b in _job(j)]
        el = time.time() - t0; NRUN[0] += NBx; NRUN[1] += el
        stage('  готово за %.0f с (%.1f с на точку)' % (el, el / NBx))
        assert len(out) == NBx; return out
    # ---------- 2а. Проверка самого батча: ответ не должен зависеть от того, как сложены блоки ----------
    stage('проверка батча: блок в одиночку и тот же блок внутри батча')
    chk_kw = dict(aH=A_H, sigv=SIGV, E0=E0, qw=qw, Gbar=Gbar, sigma=SIG, tauc_rel=TAUC, ndays=2, M=M, J=J, warm=1, sigs=SIGS_BASE)
    b_one = runb(num, 1, 'сверка, один блок', kd=0.0, seed=SEED, **chk_kw)[0]
    b_tri = runb(num, 3, 'сверка, три блока', kd=np.array([0.0, KBASE, 0.0]), seed=SEED, **chk_kw)
    def dmax(a_, b_):
        o = 0.0
        for kk in ('nb', 'epn', 'epsum', 'taw', 'tdam', 'Dbar', 'Dmbar', 'wksum', 'wkmax', 'Dmin', 'Dmax', 'gap_sum', 'gap_sq', 'epmin', 'prbar', 'prdam'):
            o = max(o, float(np.nanmax(np.abs(np.asarray(a_[kk], float) - np.asarray(b_[kk], float)))) if len(a_[kk]) else 0.0)
        for kk in ('hist', 's_after'):
            for y1, y2 in zip(a_[kk], b_[kk]): o = max(o, float(np.abs(y1 - y2).max()))
        o = max(o, float(np.abs(a_['gaps'] - b_['gaps']).max())); return o
    e_in = dmax(b_one, b_tri[0]); e_pair = dmax(b_tri[0], b_tri[2]); e_kd = dmax(b_tri[0], b_tri[1])
    v_batch = bool(e_in == 0.0 and e_pair == 0.0 and e_kd > 0.0)
    L += ['\n## 2а. Проверка самого батча (переделка счёта, записано до счёта)\n',
          'Перебор теперь не последовательный: все точки перебора сложены в ОДНУ матрицу строк и считаются одним циклом. Выигрыш — не в арифметике, а в накладных расходах: вместо девяти проходов по циклу делается один, а длина строки роли почти не играет. Поэтому надо доказать, что ответ от этой переделки не изменился ни в одном знаке.\n',
          'Шум разыгрывается на длину ОДНОГО блока и повторяется на все блоки, поэтому точки перебора строго парные — это одна и та же реализация шума при разных параметрах, как и было задумано.\n',
          '| что сверяется | наибольшее расхождение | ожидание до счёта | вывод |\n|---|---|---|---|',
          '| блок сам по себе против того же блока внутри батча из трёх | %.3e | ровно нуль |  %s |' % (e_in, 'выполнено' if e_in == 0.0 else 'НЕ ВЫПОЛНЕНО'),
          '| два одинаковых блока в одном батче (1-й и 3-й) | %.3e | ровно нуль | %s |' % (e_pair, 'выполнено' if e_pair == 0.0 else 'НЕ ВЫПОЛНЕНО'),
          '| блок с k = 0 против блока с k = %.1f | %.3e | строго больше нуля, иначе параметр не доходит | %s |' % (KBASE, e_kd, 'выполнено' if e_kd > 0 else 'НЕ ВЫПОЛНЕНО'),
          '\nСверка идёт по двенадцати величинам по наборам (срывы, эпизоды, доли времени, бодрствование, пол и потолок D, промежутки) и по трём накопителям-гистограммам. Последняя строка — контроль наоборот: если бы расхождение было нулевым и там, значит параметр перебора до счёта просто не доходит и первые две строки ничего не значат.']
    R['batch_check'] = dict(err_in_batch=e_in, err_paired=e_pair, err_param=e_kd, verdict=v_batch)
    if not v_batch: L.append('\n**Внимание: переделка счёта не сошлась сама с собой. Все числа ниже сомнительны.**')

    nK = len(KSCAN)
    bk = runb(num, nK + 3, 'k, связь через долю, контроль без памяти и контроль со старым шумом',
              kd=np.array(KSCAN + [KBASE, 0.0, KBASE]), aH=A_H, sigv=SIGV, E0=E0, qw=qw, Gbar=Gbar,
              sigma=SIG, tauc_rel=TAUC, ndays=ND, M=M, J=J, seed=SEED, warm=max(WARM, WARMS),
              sigs=np.array([SIGS_BASE] * (nK + 2) + [SIGS_CTL]),                        # последний блок — ПАРНЫЙ КОНТРОЛЬ: прежний медленный шум
              usedm=np.array([1.0] * nK + [0.0, 1.0, 1.0]),                              # блок nK — связь через долю повреждённых, для 4г
              taus=np.array([TAU_S] * (nK + 1) + [TAU_M, TAU_S]))                        # блок nK+1 — защита без длинной памяти: контроль для Ф12
    rr_D = bk[nK]; rr_ns = bk[nK + 1]; rr_sn = bk[nK + 2]
    for ks, rr in zip(KSCAN, bk[:nK]):
        m = meas(rr); m['k'] = ks; rows.append(m); keep[ks] = rr; rows_b.append(meas(rr, band))
        stage('  k = %.1f: длительность %.2f сут; незакрытых %.3f; застряли дольше 1,5 сут %.3f; средняя доля повреждённых %.3f' % (ks, m['dur'], m['open_ep'], m['stuck15'], m['Dbar']))
    L += ['\n## 2. Что даёт связь «повреждение мешает заснуть» (одно зерно на весь перебор)\n',
          '| k, мВ | доля повреждавшихся | со срывом во время записи | длительность эпизода, сут | то же с незакрытыми | незакрытых | эпизодов короче суток: по наборам / по всем | застряли >2 сут | >1,5 сут | бодрствование: среднее / наибольшее, сут | период, ч | доля времени бодрствования | доля повреждённых D: пол / в среднем / потолок | наборов с хроническим дефицитом (пол > 0,1) | наборов, где D колеблется больше 0,2 | активность E | нагрузка |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r_ in rows: L.append('| %.1f | %.3f | %.3f | %.2f | %.2f | %.3f | %.3f / %.3f | %.3f | %.3f | %.2f / %.2f | %.1f | %.3f | %.2f / %.2f / %.2f | %.3f | %.3f | %.3f | %.3f |' % (
        r_['k'], r_['share_ever'], r_['share_break'], r_['dur'], r_['dur_all'], r_['open_ep'], r_['short'], r_['short_pool'], r_['stuck'], r_['stuck15'], r_['wkmean'], r_['wkmax'], r_['per'], r_['taw'],
        r_['Dmin'], r_['Dbar'], r_['Dmax'], r_['chronic'], r_['swings'], r_['Ebar'], r_['Lbar']))
    L.append('\nСтолбец «пол» важнее, чем кажется: если пол D выше нуля, часть PV-клеток повреждена ПОСТОЯННО, и это уже не эпизод, а хронический дефицит, на который эпизоды накладываются сверху. Это согласуется с тем, что PV-мРНК у больных снижена постоянно (−36,7 % в слое 4, Dienel и соавт. 2023), а психозы при этом приступообразны. Но для статистики эпизодов важны наборы из столбца «D колеблется больше 0,2»: только у них есть что считать.')

    # ---------- 2б. Эпизодическая подгруппа ----------
    epi = band                                                                           # полоса из раздела 1в: отбор структурный, а не по исходу
    chron = keep[0.0]['Dmin'] > 0.1                                                      # наборы с постоянным дефицитом
    L += ['\n## 2б. Главные числа по эпизодической подгруппе\n',
          'Подгруппа — это полоса из раздела 1в: наборы, у которых окно уже предельной ширины W\\*. Отбор сделан ДО счёта по постоянным модели, а не по тому, у кого получились эпизоды. Прежде подгруппа определялась по исходу (колебания D больше 0,2), и это было круговым рассуждением.\n',
          '- наборов в полосе (окно уже W\\*): %d из %d (%.3f)' % (int(epi.sum()), nsel, float(epi.mean())),
          '- наборов с постоянным дефицитом (пол D выше 0,1): %d (%.3f)' % (int(chron.sum()), float(chron.mean())),
          '\n| k, мВ | длительность эпизода, сут | эпизодов короче суток | застряли >1,5 сут | срывов во второй половине бодрствования | ночей, оборвавших эпизод | доля повреждённых D: пол / среднее / потолок |\n|---|---|---|---|---|---|---|']
    epi_rows = []
    for ks in KSCAN:
        mi = meas(keep[ks], epi); mi['k'] = ks; epi_rows.append(mi)
        L.append('| %.1f | %.2f | %.3f | %.3f | %.3f | %.3f | %.2f / %.2f / %.2f |' % (ks, mi['dur'], mi['short'], mi['stuck15'], mi['late'], mi['sleep_kills'], mi['Dmin'], mi['Dbar'], mi['Dmax']))
    R['episodic'] = dict(share=float(epi.mean()), n=int(epi.sum()), share_chronic=float(chron.mean()), rows=epi_rows)

    # ---------- 3. Правило Ф3 ----------
    sds = [SEED + 1000 * r_ for r_ in range(1, NREP)]; pairs = []
    if sds:                                                                              # каждому зерну — два блока, k = 0 и k = KBASE, с одним и тем же потоком
        bs = runb(num, 2 * len(sds), 'зёрна', kd=np.array([0.0, KBASE] * len(sds)), aH=A_H, sigv=SIGV, E0=E0, qw=qw, Gbar=Gbar,
                  sigma=SIG, tauc_rel=TAUC, ndays=ND, M=M, J=J, warm=WARM, sigs=SIGS_BASE,
                  seed=[sd for sd in sds for _ in (0, 1)])
    for rep in range(NREP):
        sd = SEED if rep == 0 else sds[rep - 1]
        if rep == 0: d0, db = rows_b[KSCAN.index(0.0)]['dur'], rows_b[KSCAN.index(KBASE)]['dur']
        else: d0 = meas(bs[2 * (rep - 1)], band)['dur']; db = meas(bs[2 * (rep - 1) + 1], band)['dur']
        pairs.append(dict(seed=sd, dur0=d0, durb=db, ratio=db / d0 if d0 > 0 else float('nan')))
        stage('  зерно %d: %.2f -> %.2f, отношение %.2f' % (sd, d0, db, pairs[-1]['ratio']))
    rts = [p_['ratio'] for p_ in pairs]; rmed = float(np.median(rts)); va = bool(rmed >= 1.5)
    st_seq = [r_['stuck'] for r_ in rows_b]; vb = bool(all(st_seq[i + 1] >= st_seq[i] - 1e-12 for i in range(len(st_seq) - 1)) and max(st_seq[1:]) > 0)
    vc = bool(rows_b[KSCAN.index(KBASE)]['short'] < 0.50)
    L += ['\n## 3. Правило Ф3 (вердикты — по полосе из раздела 1в)\n', '| зерно | длительность при k = 0 | при k = %.1f | отношение |\n|---|---|---|---|' % KBASE]
    for p_ in pairs: L.append('| %d | %.2f | %.2f | %.2f |' % (p_['seed'], p_['dur0'], p_['durb'], p_['ratio']))
    L += ['\n| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|',
          '| а | отношение %.2f (по зёрнам: %s) | не меньше 1,5 | %s |' % (rmed, ', '.join('%.2f' % r_ for r_ in rts), 'выполнено' if va else 'не выполнено'),
          '| б | %s | растёт с k | %s |' % (' → '.join('%.3f' % s_ for s_ in st_seq), 'выполнено' if vb else 'не выполнено'),
          '| в | %.3f | ниже 0,50 | %s |' % (rows_b[KSCAN.index(KBASE)]['short'], 'выполнено' if vc else 'не выполнено'),
          '\nВ прежней модели (sleep_core2.py) по части (а) выходило 1,67. Там связь поднимала выдуманный порог засыпания и упиралась в потолок; здесь она тормозит сонное ядро, и потолка нет.']
    R['F3'] = dict(a=dict(ratio_median=rmed, ratios=rts, thresh=1.5, verdict=va), b=dict(stuck_by_k=dict(zip(map(str, KSCAN), st_seq)), verdict=vb),
                   c=dict(short_share=rows_b[KSCAN.index(KBASE)]['short'], verdict=vc), pairs=pairs, prev_model_ratio=1.67)
    R['kscan_band'] = rows_b
    R['kscan'] = rows

    # ---------- 4. Правило Ф4 ----------
    rb_ = keep[KBASE]; ds2, dn2 = [z[band] for z in rb_['dbin']]; ls2, ln2 = [z[band] for z in rb_['lbin']]; hbs, hbn = [z[band] for z in rb_['hbase']]
    ds_, dnn, ls_, ln = ds2.sum(0), dn2.sum(0), ls2.sum(0), ln2.sum(0)                   # сложенные по всем наборам: мера, записанная до счёта
    dcur = np.where(dnn > 0, ds_ / np.maximum(dnn, 1), np.nan); lcur = np.where(ln > 0, ls_ / np.maximum(ln, 1), np.nan)
    base_bout = float(dcur[0]) if dnn[0] > 0 else float('nan')
    own = np.where(hbn >= 3, hbs / np.maximum(hbn, 1), np.nan)                           # собственный здоровый уровень набора
    okn = hbn >= 3
    dn_rel = np.where((dn2 > 0) & okn[:, None], ds2 / np.maximum(dn2, 1) / np.maximum(own[:, None], 1e-9), np.nan)
    ln_rel = np.where((ln2 > 0) & okn[:, None], ls2 / np.maximum(ln2, 1) / np.maximum(own[:, None], 1e-9), np.nan)
    with np.errstate(invalid='ignore'):
        dcur_rel = np.nanmean(dn_rel, 0); lcur_rel = np.nanmean(ln_rel, 0)
        dcnt_rel = (~np.isnan(dn_rel)).sum(0); lcnt_rel = (~np.isnan(ln_rel)).sum(0)
    jlast = max([j for j in range(1, 5) if dnn[j] >= 10], default=-1)                    # последняя подпороговая ячейка, где есть не меньше 10 бодрствований
    nsub_ = float(dnn[1:5].sum()); enough = nsub_ >= 20 and jlast >= 2 and dnn[0] >= 10
    g4 = dcur[jlast] / dcur[0] if (enough and jlast > 0) else float('nan')
    v4a = bool(enough and g4 == g4 and g4 >= 1.10)
    lead_ok = [j for j in range(NLB) if ln[j] >= 5 and lcur[j] > base_bout]
    v4b = bool(lead_ok and min(lead_ok) <= 7)
    L += ['\n## 4. Правило Ф4: портится ли сон раньше срыва (k = %.1f мВ)\n' % KBASE,
          '### (а) Длина бодрствования против доли повреждённых за это бодрствование\n',
          'Левая пара столбцов — мера, записанная до счёта: все бодрствования всех наборов в общей куче. Правая — то же, но каждое бодрствование поделено на СОБСТВЕННЫЙ здоровый уровень своего набора и усреднено по наборам. Вердикт выносится по левой; правая нужна, чтобы увидеть, не создан ли рост межнаборным сдвигом (у наборов, которые вообще срываются, бодрствование длиннее само по себе).\n',
          '| доля повреждённых D | длина бодрствования, сут | число бодрствований | то же в долях своего здорового уровня | наборов |\n|---|---|---|---|---|']
    for j in range(NDB): L.append('| %.1f–%.1f%s | %s | %d | %s | %d |' % (j / 10., (j + 1) / 10., ' (порог эпизода)' if j == 4 else '', ('%.3f' % dcur[j]) if dnn[j] > 0 else '—', int(dnn[j]), ('%.3f' % dcur_rel[j]) if dcnt_rel[j] > 0 else '—', int(dcnt_rel[j])))
    L += ['\n- бодрствований в подпороговых ячейках 0,1–0,5: %d; вердикт берётся по ячейке %s (последней заполненной), потому что доля повреждённых идёт шагами по 1/%d и крайняя ячейка может оказаться пустой структурно' % (int(nsub_), ('%.1f–%.1f' % (jlast / 10., (jlast + 1) / 10.)) if jlast > 0 else 'нет подходящей', J),
          '\n**Ф4(а):** %s Вывод: %s.' % (('подпороговый рост в %.2f раза (последняя заполненная подпороговая ячейка %.1f–%.1f против 0,0–0,1; порог 1,10).' % (g4, max(jlast, 0) / 10., (max(jlast, 0) + 1) / 10.)) if enough else 'подпороговых бодрствований слишком мало (%d), правило непроверяемо: нужно не меньше 20 и хотя бы 10 в проверяемой ячейке.' % int(nsub_), 'выполнено' if v4a else ('непроверяемо' if not enough else 'не выполнено')),
          '\n### (б) Длина бодрствования против срока до первого срыва\n', 'Здоровый уровень (D ниже 0,1): %.3f сут. Учитываются только бодрствования внутри здорового промежутка перед срывом, иначе кривую рисует сам отбор.\n' % base_bout,
          '| суток до срыва | длина бодрствования, сут | число бодрствований | то же в долях своего здорового уровня | наборов |\n|---|---|---|---|---|']
    for j in range(NLB): L.append('| %d–%d | %s | %d | %s | %d |' % (j, j + 1, ('%.3f' % lcur[j]) if ln[j] > 0 else '—', int(ln[j]), ('%.3f' % lcur_rel[j]) if lcnt_rel[j] > 0 else '—', int(lcnt_rel[j])))
    rel_up = [j for j in range(NLB) if lcnt_rel[j] >= 5 and lcur_rel[j] > 1.0]
    L += ['\n**Ф4(б):** бодрствование выше здорового уровня за %s суток до срыва. Вывод: %s.' % (('—' if not lead_ok else '%d–%d' % (min(lead_ok), max(lead_ok) + 1)), 'выполнено' if v4b else 'не выполнено'),
          'В долях собственного здорового уровня превышение держится в ячейках %s. Если правый столбец ровный и близок к 1, то «опережение» в левом — межнаборный сдвиг, а не предвестник.' % ('—' if not rel_up else '%d–%d' % (min(rel_up), max(rel_up) + 1)),
          '\nВ прогоне при q = 0,75 это правило прошло ВХОЛОСТУЮ: левая кривая была мёртво ровной (0,725 / 0,716 / 0,715 / 0,689 / 0,710 / 0,712 / 0,708 / 0,711) и целиком лежала выше здорового уровня, то есть срабатывал сдвиг, а не наклон, а правый столбец был пуст целиком. Если и здесь правый столбец окажется плоским или пустым, вердикт «выполнено» засчитывать нельзя.',
          '\nМера изменена против sleep_core2.py: учитывается КАЖДЫЙ срыв, которому предшествовали хотя бы трое здоровых суток, а не только первый за прогон. Первых срывов было слишком мало, потому что почти все наборы срываются ещё на разгоне. Модель от этого не менялась.\nВ прежней модели эта кривая была ровной (0,681 → 0,680 за 15 суток): предвестника сна не было вовсе. Если он появился здесь, то появился он от связи через сеть, а не от измерения.']
    R['F4'] = dict(a=dict(d_curve=[None if dnn[j] == 0 else float(dcur[j]) for j in range(NDB)], d_counts=[int(v) for v in dnn], gain=float(g4), thresh=1.10, verdict=v4a, enough=bool(enough), n_sub=int(nsub_), bin_used=int(jlast)),
                   b=dict(lead_curve=[None if ln[j] == 0 else float(lcur[j]) for j in range(NLB)], lead_counts=[int(v) for v in ln], base_bout=base_bout, lead_days=lead_ok, verdict=v4b,
                          lead_curve_rel=[None if lcnt_rel[j] == 0 else float(lcur_rel[j]) for j in range(NLB)], lead_sets_rel=[int(v) for v in lcnt_rel], rel_days=rel_up),
                   a_rel=dict(curve=[None if dcnt_rel[j] == 0 else float(dcur_rel[j]) for j in range(NDB)], sets=[int(v) for v in dcnt_rel]),
                   prev_model_flat=[0.681, 0.680])

    # ---------- 4в. Медленный шум нагрузки и правило Ф6 ----------
    stage('медленный шум нагрузки: перебор силы')
    L += ['\n## 4в. Медленный шум нагрузки: есть ли предвестник сна\n',
          'Почему это понадобилось. В прогоне без медленного шума правый столбец Ф4(б) — длина бодрствования в долях собственного здорового уровня набора — совпал до тринадцатого знака в семи ячейках из восьми (1,1003 / 1,0994 / 1,0994 / …). Это не слабый сигнал, а его полное отсутствие, и причина арифметическая: время корреляции шума нагрузки равно 0,3·T, а сутки — это r·T при r около 3, то есть шум коррелирован 0,10 суток, всего 2,4 часа. В модели не было НИ ОДНОЙ величины, меняющейся на масштабе от одних до восьми суток, и предвестник такой длины сделать было не из чего.\n',
          'Что добавлено: вторая составляющая шума на логарифме нагрузки со временем корреляции %.0f суток. Смысл — изо дня в день меняются активность, стресс, питание, то есть всё, что входит в нагрузку, и меняется на масштабе дней. Амплитуду анкерить нечем, поэтому это МОЁ ДОПУЩЕНИЕ, и она перебирается, а не назначается. Среднее сохранено: множитель exp(eta_м − sigma_м²/2), так что при любой силе средняя нагрузка остаётся прежней и перебор не смешивается со сдвигом среднего.\n' % TSLOW,
          'Медленный шум разыгрывается отдельным потоком с постоянным зерном, поэтому строки перебора ПАРНЫЕ: это одна и та же реализация, умноженная на разную амплитуду. Строка sigma_м = 0 воспроизводит прежний прогон в точности — быстрый поток случайных чисел не тронут.\n',
          'Разгон поднят до %d суток: три времени корреляции медленной составляющей.\n' % WARMS,
          '| sigma_м | длина бодрствования в долях своего уровня: 7–8 сут до срыва → 0–1 сут | наборов в крайних ячейках | доля повреждавшихся | длительность эпизода, сут | эпизодов короче суток | застряли >1,5 сут |\n|---|---|---|---|---|---|---|']
    srows = []; scurves = {}
    PR6 = [(ss, TAU_S) for ss in SSCAN] + [(0.0, TAU_M)]                                 # последний блок: шума нет И длинной памяти защиты нет — привязка предвестника
    bss = runb(num, len(PR6), 'медленный шум и привязка предвестника', kd=KBASE, aH=A_H, sigv=SIGV, E0=E0, qw=qw, Gbar=Gbar,
               sigma=SIG, tauc_rel=TAUC, ndays=ND, M=M, J=J, seed=SEED, warm=WARMS,
               sigs=np.array([z[0] for z in PR6]), taus=np.array([z[1] for z in PR6]))
    def curve6(rr):
        ls3, ln3 = rr['lbin']; hs3, hn3 = rr['hbase']
        own3 = np.where(hn3 >= 3, hs3 / np.maximum(hn3, 1), np.nan); ok3 = hn3 >= 3
        rel3 = np.where((ln3 > 0) & ok3[:, None], ls3 / np.maximum(ln3, 1) / np.maximum(own3[:, None], 1e-9), np.nan)
        with np.errstate(invalid='ignore'):
            return np.nanmean(rel3, 0), (~np.isnan(rel3)).sum(0)
    cur_nm, cnt_nm = curve6(bss[-1])                                                     # предвестник без длинной памяти защиты
    for ss, rr in zip(SSCAN, bss[:len(SSCAN)]):
        m = meas(rr); ls3, ln3 = rr['lbin']; hs3, hn3 = rr['hbase']
        own3 = np.where(hn3 >= 3, hs3 / np.maximum(hn3, 1), np.nan); ok3 = hn3 >= 3
        rel3 = np.where((ln3 > 0) & ok3[:, None], ls3 / np.maximum(ln3, 1) / np.maximum(own3[:, None], 1e-9), np.nan)
        with np.errstate(invalid='ignore'):
            cur3 = np.nanmean(rel3, 0); cnt3 = (~np.isnan(rel3)).sum(0)
        scurves[ss] = (cur3, cnt3); m['sigs'] = ss; srows.append(m)
        L.append('| %.2f | %s → %s | %d / %d | %.3f | %.2f | %.3f | %.3f |' % (ss,
                 ('%.4f' % cur3[7]) if cnt3[7] > 0 else '—', ('%.4f' % cur3[0]) if cnt3[0] > 0 else '—',
                 int(cnt3[7]), int(cnt3[0]), m['share_ever'], m['dur'], m['short'], m['stuck15']))
        stage('  sigma_м = %.2f: последние сутки %s' % (ss, ('%.4f' % cur3[0]) if cnt3[0] > 0 else '—'))
    cb, nb3 = scurves[SIGS_BASE]; c0, n0 = scurves[SIGS_CTL]                                 # опорная — при НУЛЕВОМ шуме; рядом печатается прежняя рабочая 0,08
    L += ['\n### Кривая предвестника при опорной силе sigma_м = %.2f\n' % SIGS_BASE,
          '| суток до срыва | в долях своего здорового уровня | наборов | то же при sigma_м = 0 (контроль) |\n|---|---|---|---|']
    for j in range(NLB): L.append('| %d–%d | %s | %d | %s |' % (j, j + 1, ('%.4f' % cb[j]) if nb3[j] > 0 else '—', int(nb3[j]), ('%.4f' % c0[j]) if n0[j] > 0 else '—'))
    from scipy.stats import rankdata as _rd
    fin = np.where(nb3 > 0)[0]
    rho = float(np.corrcoef(_rd(fin), _rd(cb[fin]))[0, 1]) if len(fin) >= 4 else float('nan')
    fin0 = np.where(n0 > 0)[0]
    rho0 = float(np.corrcoef(_rd(fin0), _rd(c0[fin0]))[0, 1]) if len(fin0) >= 4 else float('nan')
    v6a = bool(nb3[0] > 0 and cb[0] >= 1.05); v6b = bool(rho == rho and rho <= -0.5)
    ctrl_ok = bool(n0[0] >= 10)                                                          # контроль осмыслен только при десяти наборах и больше
    v6c = ('непроверяемо' if not ctrl_ok else bool(not (c0[0] >= 1.05 and rho0 == rho0 and rho0 <= -0.5)))
    rho_all = {}
    for ss in SSCAN:
        cu_, cn_ = scurves[ss]; f_ = np.where(cn_ > 0)[0]
        rho_all[ss] = (float(np.corrcoef(_rd(f_), _rd(cu_[f_]))[0, 1]) if len(f_) >= 4 and np.ptp(cu_[f_]) > 0 else float('nan'), int(cn_[0]))
    strong = [ss for ss in SSCAN if rho_all[ss][1] >= 20]
    v6g = bool(strong and all(rho_all[ss][0] == rho_all[ss][0] and rho_all[ss][0] <= -0.5 for ss in strong))
    plateau = float(np.nanmean(np.where(nb3[1:] > 0, cb[1:], np.nan)))
    L += ['\n**Правило Ф6** (записано до счёта): длина бодрствования в долях собственного здорового уровня растёт монотонно по мере приближения к срыву и в последние сутки превышает здоровый уровень не меньше чем на 5 %.\n',
          '| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|',
          '| а: последние сутки | %s | не меньше 1,05 | %s |' % (('%.4f' % cb[0]) if nb3[0] > 0 else 'ячейка пуста', 'выполнено' if v6a else 'не выполнено'),
          '| б: наклон | ранговая корреляция %s | не выше −0,50 | %s |' % (('%+.2f' % rho) if rho == rho else '—', 'выполнено' if v6b else 'не выполнено'),
          '| в: это не артефакт измерения | при sigma_м = 0: последние сутки %s, корреляция %s | та же проверка при нулевом медленном шуме НЕ проходит | %s |' % (
              ('%.4f' % c0[0]) if n0[0] > 0 else '—', ('%+.2f' % rho0) if rho0 == rho0 else '—', 'выполнено' if v6c else 'не выполнено'),
          '| г: наклон держится при всех силах шума | %s | каждая сила с 20+ наборами проходит порог −0,50 | %s |' % (
              ', '.join('%.2f: %s (%d наб.)' % (ss, ('%+.2f' % rho_all[ss][0]) if rho_all[ss][0] == rho_all[ss][0] else 'не опр.', rho_all[ss][1]) for ss in SSCAN), 'выполнено' if v6g else 'не выполнено'),
          '\n| плато на 1–8 сутках | последние сутки | последние сутки выше плато на |\n|---|---|---|',
          '| %.4f | %.4f | %.1f %% |' % (plateau, cb[0], 100 * (cb[0] / plateau - 1)),
          '\nПлато и последние сутки печатаются отдельно намеренно. В прошлом прогоне кривая была плоской на 1–8 сутках и поднималась только в последние — это однодневный эффект, а не предвестник, и так его выдать за предвестник нельзя.',
          '\nЧасть (в) — главная. Если предвестник появляется и при нулевом медленном шуме, значит его сделало измерение, а не механизм. Прошлый прогон прошёл её на технической ошибке: при нулевом шуме кривую дал ОДИН набор, корреляция оказалась неопределённой, и это было засчитано как «контроль не проходит». Теперь контроль требует не меньше десяти наборов, иначе он непроверяем.',
          '\nЧасть (г) закрывает вторую дыру: прежде вердикт выносился только при объявленной опорной силе шума, и он держался на строке с наименьшим числом наборов, тогда как две строки с лучшей статистикой ему противоречили.',
          '\nЧего эта правка НЕ проверяет: направление причинности. Медленный шум поднимает нагрузку, нагрузка поднимает подпороговое повреждение, повреждение удлиняет бодрствование — то есть здесь плохой сон является МАРКЕРОМ повреждения. Обратное направление (длинное бодрствование само подкручивает нагрузку и приближает срыв) появится только с шумом времени засыпания, которого здесь ещё нет.']
    # ---------- 4в-2. Правило Ф18: откуда предвестник, и почему медленный шум выведен из рабочего набора ----------
    fnm = np.where(cnt_nm > 0)[0]
    rho_nm = float(np.corrcoef(_rd(fnm), _rd(cur_nm[fnm]))[0, 1]) if len(fnm) >= 4 and np.ptp(cur_nm[fnm]) > 0 else float('nan')
    v18a = bool(nb3[0] > 0 and cb[0] >= 1.05 and rho == rho and rho <= -0.5)
    nm_ok = bool(cnt_nm[0] >= 10)
    v18b = ('непроверяемо' if not nm_ok else bool(not (cur_nm[0] >= 1.05 and rho_nm == rho_nm and rho_nm <= -0.5)))
    L += ['\n## 4в-2. Правило Ф18: чем на самом деле сделан предвестник\n',
          '**Что изменено в этом прогоне и почему.** Рабочая сила медленного шума нагрузки переведена с 0,08 на НОЛЬ. Основание — прошлый прогон: часть Ф6(в) не выполнилась, то есть при выключенном медленном шуме предвестник вышел НЕ ХУЖЕ (последние сутки 1,0705 против 1,0539, корреляция −0,64 против −0,86). Медленный шум вводился под обоснование «в модели нет ни одной величины, меняющейся на масштабе от одних до восьми суток». Тогда это было правдой. Это перестало быть правдой в mass_core5, когда появилась вторая ступень защиты с постоянной %.0f ч = %.1f суток. То есть шум решал задачу, которую позже решила защита, и остался без обоснования. Вдобавок он не безвреден: по Ф15 он вдвое срезает измеряемый глухой период (t_r/g 0,49 против 0,23).\n' % (TAU_S * 24, TAU_S),
          'Прежнее рабочее значение 0,08 оставлено в прогоне ПАРНЫМ КОНТРОЛЕМ — отдельным блоком при k = %.1f, то же зерно, тот же шум быстрой составляющей. Так видно размер сдвига от самой правки.\n' % KBASE,
          '**Оговорка, которую нельзя терять:** бюджет шума в разделе 1б считается как 2·sigma_быстр + sigma_медл, поэтому при нулевом медленном шуме он сузился с %.2f до %.2f, а вместе с ним изменилась и ПОЛОСА отбора. Значит все числа по полосе в этом прогоне не сравнимы напрямую с прошлым прогоном — сравнивать надо только внутри прогона, с парным контролем.\n' % (2 * SIG + SIGS_CTL, 2 * SIG),
          '\n### Правило Ф18, записано до счёта\n',
          '(а) Предвестник не нуждается в медленном шуме: при sigma_м = 0 последние сутки не ниже 1,05 и корреляция не выше −0,50. **Это подтверждение уже увиденного, а не предсказание** — в прошлом прогоне контроль дал ровно это. Отмечаю прямо, чтобы потом не выдать за новое.\n',
          '(б) ГЛАВНАЯ ЧАСТЬ: если предвестник делает медленная переменная защиты, то при sigma_м = 0 И отнятой длинной памяти (вторая ступень поставлена на %.0f ч вместо %.0f) он должен ПРОПАСТЬ — последние сутки ниже 1,05 ИЛИ корреляция выше −0,50.\n' % (TAU_M * 24, TAU_S * 24),
          '(в) Если (б) не выполнено, то предвестник не от шума и не от памяти защиты. Тогда вывод записан заранее: в модели есть третья медленная величина, которую я не вижу, и следующий шаг — искать её, а не достраивать имеющиеся.\n',
          '| суток до срыва | шума нет, память %.0f ч (рабочее) | шума нет, памяти нет (%.0f ч) | шум 0,08, память %.0f ч (прежнее рабочее) |\n|---|---|---|---|' % (TAU_S * 24, TAU_M * 24, TAU_S * 24)]
    for j in range(NLB):
        L.append('| %d–%d | %s | %s | %s |' % (j, j + 1,
                 ('%.4f' % cb[j]) if nb3[j] > 0 else '—', ('%.4f' % cur_nm[j]) if cnt_nm[j] > 0 else '—', ('%.4f' % c0[j]) if n0[j] > 0 else '—'))
    L += ['\n| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|',
          '| а: предвестник без медленного шума есть | последние сутки %s, корреляция %s | не ниже 1,05 и не выше −0,50 | %s |' % (
              ('%.4f' % cb[0]) if nb3[0] > 0 else '—', ('%+.2f' % rho) if rho == rho else '—', 'выполнено' if v18a else 'не выполнено'),
          '| б: без длинной памяти защиты предвестник пропадает | последние сутки %s, корреляция %s, наборов %d | ниже 1,05 ИЛИ выше −0,50 | %s |' % (
              ('%.4f' % cur_nm[0]) if cnt_nm[0] > 0 else '—', ('%+.2f' % rho_nm) if rho_nm == rho_nm else '—', int(cnt_nm[0]),
              ('выполнено' if v18b is True else ('не выполнено' if v18b is False else v18b))),
          '\nЕсли (а) и (б) выполнены — предвестник приписан: его делает медленная переменная защиты, и медленный шум нагрузки можно убрать из модели совсем, а не только из рабочего набора. Это заодно объясняет, почему предвестник появился ровно в том прогоне, где появилась двухступенчатая защита.']
    R['F18'] = dict(a=dict(last_day=float(cb[0]) if nb3[0] > 0 else None, rank_corr=rho, verdict=v18a,
                           note='подтверждение прошлого прогона, не предсказание'),
                    b=dict(last_day=float(cur_nm[0]) if cnt_nm[0] > 0 else None, rank_corr=rho_nm, sets=int(cnt_nm[0]),
                           tau_s_control_h=TAU_M * 24, verdict=v18b),
                    curve_no_memory=[None if cnt_nm[j] == 0 else float(cur_nm[j]) for j in range(NLB)],
                    sigs_working=SIGS_BASE, sigs_control=SIGS_CTL, budget_before=2 * SIG + SIGS_CTL, budget_now=2 * SIG)

    R['F6'] = dict(a=dict(last_day=float(cb[0]) if nb3[0] > 0 else None, thresh=1.05, verdict=v6a),
                   b=dict(rank_corr=rho, thresh=-0.5, verdict=v6b),
                   g=dict(rank_corr_by_sigma={('%.2f' % ss): rho_all[ss][0] for ss in SSCAN}, sets_by_sigma={('%.2f' % ss): rho_all[ss][1] for ss in SSCAN}, verdict=v6g),
                   plateau=plateau, last_over_plateau=float(cb[0] / plateau - 1),
                   c=dict(control_last_day=float(c0[0]) if n0[0] > 0 else None, control_rank_corr=rho0, control_sets=int(n0[0]), verdict=v6c),
                   sigs_base=SIGS_BASE, tau_slow_days=TSLOW,
                   curves={('%.2f' % ss): dict(curve=[None if scurves[ss][1][j] == 0 else float(scurves[ss][0][j]) for j in range(NLB)],
                                               sets=[int(z) for z in scurves[ss][1]]) for ss in SSCAN},
                   rows=srows)

    # ---------- 4г. Правило Ф7: длина эпизода, и сравнение двух связей ----------
    mD = meas(rr_D, band)                                                                # посчитан тем же батчем, что и раздел 2: то же зерно, тот же шум
    m0 = rows_b[KSCAN.index(0.0)]; mB = rows_b[KSCAN.index(KBASE)]; m0 = rows_b[KSCAN.index(0.0)]; mB = rows_b[KSCAN.index(KBASE)]
    v7a = bool(m0['dur'] == m0['dur'] and m0['dur'] <= 2.0); v7b = bool(m0['short'] == m0['short'] and m0['short'] >= 0.30)
    L += ['\n## 4г. Правило Ф7: длина эпизода\n',
          'Записано до счёта. По наблюдениям эпизод длится от часов до нескольких суток. В прогоне при q = 0,5 выходило 4,6 суток при k = 0 и 13,4 при k = 2, а эпизодов короче суток не было вовсе ни при каком шуме. Проверяем на полосе, при k = 0, чтобы связь со сном не удлиняла эпизод сама.\n',
          '| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|',
          '| а: средняя длительность | %.2f сут | не больше 2 суток | %s |' % (m0['dur'], 'выполнено' if v7a else 'не выполнено'),
          '| б: доля эпизодов короче суток | медиана по наборам %.3f; по ВСЕМ эпизодам %.3f | не меньше 0,30 | %s |' % (m0['short'], m0['short_pool'], 'выполнено' if v7b else 'не выполнено'),
          '\n**Оговорка к части (б), найдена 14.10 по замечанию пользователя.** Вердикт выносится по МЕДИАНЕ ПО НАБОРАМ, и эта статистика обнуляется, едва короткие эпизоды есть меньше чем у половины наборов, — даже если коротких эпизодов в прогоне много. Рядом поэтому печатается доля по ВСЕМ эпизодам, и самый короткий эпизод: %s сут минимум, %s сут медиана по наборам. Вердикт по правилу НЕ пересматривается (менять его после счёта нельзя), но читать надо оба числа.' % (
              ('%.2f' % m0['ep_min']) if m0['ep_min'] == m0['ep_min'] else '—', ('%.2f' % m0['ep_min_med']) if m0['ep_min_med'] == m0['ep_min_med'] else '—'),
          '\n### Через что вести связь: среднее повреждение или доля повреждённых\n',
          'Одно и то же зерно, k = %.1f, оба прогона на полосе. Доля равна нулю ровно до тех пор, пока первая клетка не перешагнёт порог, поэтому у подпорогового повреждения через неё нет пути к сну вовсе. Среднее повреждение порога не имеет.\n' % KBASE,
          '| связь идёт через | длительность эпизода, сут | эпизодов короче суток | доля повреждавшихся | застряли >1,5 сут | среднее повреждение | доля повреждённых |\n|---|---|---|---|---|---|---|',
          '| среднее повреждение | %.2f | %.3f | %.3f | %.3f | %.3f | %.3f |' % (mB['dur'], mB['short'], mB['share_ever'], mB['stuck15'], mB['Dmbar'], mB['Dbar']),
          '| долю повреждённых (как было) | %.2f | %.3f | %.3f | %.3f | %.3f | %.3f |' % (mD['dur'], mD['short'], mD['share_ever'], mD['stuck15'], mD['Dmbar'], mD['Dbar'])]
    R['F7'] = dict(a=dict(dur_k0=m0['dur'], thresh=2.0, verdict=v7a), b=dict(short_k0=m0['short'], thresh=0.30, verdict=v7b),
                   coupling=dict(mean_damage={kk: vv for kk, vv in mB.items() if kk != 'dur_arr'}, fraction={kk: vv for kk, vv in mD.items() if kk != 'dur_arr'}))

    # ---------- 4д. Серотониновый триггер и правило Ф8 ----------
    stage('серотониновый триггер')
    L += ['\n## 4д. Серотониновый триггер: правило Ф8\n',
          'Зачем. Закон сохранения разрывов (раздел 1в) запрещает редкие эпизоды с быстрым возвратом, ПОКА срыв и возврат берутся из одного распределения шума. Сумма двух разрывов у медианного набора — около девяти сигм, а картина требует трёх сигм вверх и полутора вниз. Девять в четыре с половиной не влезают ни при каком q.\n',
          'Выход один: сделать срыв событием ИНОГО рода. Всплеск серотонина односторонен — он поднимает нагрузку и никогда её не опускает, — поэтому разрыв вверх перекрывается им, а разрыв вниз остаётся на обычном шуме. Два разрыва меряются разными распределениями, и закон перестаёт связывать.\n',
          'Откуда числа. Активация 5-HT2A поднимает частоту разрядов пирамидных клеток V слоя префронтальной коры примерно ВДВОЕ (псилоцин 10 мкМ, срезы; то же у NBOH-2C-CN 200 нМ; эффект снимается антагонистом M100907 и ингибитором Gq). Это фармакологический максимум, а не эндогенный всплеск, поэтому ×2 взято ПОТОЛКОМ, а величина эндогенного всплеска перебирается. Длительность: острое действие псилоцибина 6–7 ч, ЛСД 10–11 ч, начало через 0,4–0,8 ч. Отсюда постоянная спада %.0f ч, то есть событие живёт около шести часов.\n' % TAU_SR,
          '**Оговорка, которую нельзя терять:** переход от «частота разрядов в срезе выросла вдвое» к «нагрузка на клетку выросла вдвое» НЕ установлен. Базовая частота в срезе — это спонтанная активность, а не рабочий уровень коры. Поэтому ×2 — верхняя граница шкалы, и все промежуточные значения перебираются.\n',
          '**Рабочее положение нагрузки перенесено на q = %.2f.** При нём шуму нужен выброс 7,3 сигмы, чтобы сорвать клетку самому, — то есть сам по себе он этого не делает практически никогда; а на возврат нужно 1,8 сигмы, то есть часы. Срывы становятся событийными, возврат — обычным.\n' % QPOS,
          '\n### Правило Ф8, записано до счёта\n',
          '(а) Срывы вызываются всплесками: доля срывов, случившихся в пределах 12 часов после всплеска, не меньше 0,80.\n(б) Длительность эпизода не больше 2 суток.\n(в) Доля эпизодов короче суток не меньше 0,30.\n(г) Отклик на дозу: частота срывов растёт вместе с частотой всплесков.\n',
          '\n### Перебор частоты всплесков (амплитуда %.1f)\n' % SR_AMP,
          '| всплесков в сутки | всплесков за прогон на набор | доля повреждавшихся | срывов в сутки | доля срывов после всплеска | длительность эпизода, сут | эпизодов короче суток |\n|---|---|---|---|---|---|---|']
    nR = len(RSCAN); nA_ = len(ASCAN_SR)
    bsr = runb(num, nR + nA_ + 1, 'серотонин', kd=KBASE, aH=A_H, sigv=SIGV, E0=E0, qw=qw, Gbar=Gbar, sigma=SIG, tauc_rel=TAUC,
               ndays=max(ND // 2, 5), M=M, J=J, seed=SEED, warm=WARM2, sigs=SIGS_BASE,
               sramp=np.array([SR_AMP] * nR + list(ASCAN_SR) + [1.0]),
               srate=np.array(list(RSCAN) + [SR_RATE] * nA_ + [0.0]))
    rrows = []
    for rt, rr in zip(RSCAN, bsr[:nR]):
        m = meas(rr, band); m['rate'] = rt; rrows.append(m)
        L.append('| %.2f | %.1f | %.3f | %.3f | %.3f | %.2f | %.3f |' % (rt, m['n_sr'], m['share_ever'], m['nday'], m['trig'], m['dur'], m['short']))
        stage('  всплесков %.2f/сут: доля повреждавшихся %.3f, срывов после всплеска %.3f' % (rt, m['share_ever'], m['trig']))
    L += ['\n### Перебор амплитуды всплеска (частота %.2f в сутки)\n' % SR_RATE,
          '| нагрузка умножается на | доля повреждавшихся | срывов в сутки | доля срывов после всплеска | длительность эпизода, сут | эпизодов короче суток |\n|---|---|---|---|---|---|']
    arows_sr = []
    for am, rr in zip(ASCAN_SR, bsr[nR:nR + nA_]):
        m = meas(rr, band); m['amp'] = am; arows_sr.append(m)
        L.append('| %.1f%s | %.3f | %.3f | %.3f | %.2f | %.3f |' % (am, ' (потолок)' if am >= SR_CEIL else '', m['share_ever'], m['nday'], m['trig'], m['dur'], m['short']))
    rr0 = bsr[-1]; m0sr = meas(rr0, band)
    mb = [r_ for r_ in rrows if abs(r_['rate'] - SR_RATE) < 1e-9]; mb = mb[0] if mb else rrows[len(rrows) // 2]
    v8a = bool(mb['trig'] >= 0.80); v8b = bool(mb['dur'] == mb['dur'] and mb['dur'] <= 2.0); v8c = bool(mb['short'] == mb['short'] and mb['short'] >= 0.30)
    nd_seq = [r_['nday'] for r_ in rrows]; v8d = bool(all(nd_seq[i + 1] >= nd_seq[i] - 1e-12 for i in range(len(nd_seq) - 1)) and nd_seq[-1] > nd_seq[0])
    L += ['\n### Контроль: всплесков нет вовсе\n',
          '| | доля повреждавшихся | срывов в сутки | длительность эпизода, сут | эпизодов короче суток |\n|---|---|---|---|---|',
          '| без всплесков | %.3f | %.3f | %.2f | %.3f |' % (m0sr['share_ever'], m0sr['nday'], m0sr['dur'], m0sr['short']),
          '| со всплесками (%.2f/сут, ×%.1f) | %.3f | %.3f | %.2f | %.3f |' % (SR_RATE, SR_AMP, mb['share_ever'], mb['nday'], mb['dur'], mb['short']),
          '\n| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|',
          '| а: срывы после всплеска | %.3f | не меньше 0,80 | %s |' % (mb['trig'], 'выполнено' if v8a else 'не выполнено'),
          '| б: длительность эпизода | %.2f сут | не больше 2 | %s |' % (mb['dur'], 'выполнено' if v8b else 'не выполнено'),
          '| в: эпизодов короче суток | медиана по наборам %.3f; по ВСЕМ эпизодам %.3f; самый короткий %s сут | не меньше 0,30 | %s |' % (
              mb['short'], mb['short_pool'], ('%.2f' % mb['ep_min']) if mb['ep_min'] == mb['ep_min'] else '—', 'выполнено' if v8c else 'не выполнено'),
          '| г: отклик на дозу | %s | растёт с частотой всплесков | %s |' % (' → '.join('%.3f' % z for z in nd_seq), 'выполнено' if v8d else 'не выполнено'),
          '\n**Если (б) и (в) не выполнены и здесь — дорога плохая, и это главный результат прогона.** Значит односторонний триггер не спасает, и дело не в том, чем вызван срыв, а в самом устройстве «двухустойчивая клетка плюс нагрузка». Тогда следующий шаг — индукция сульфиредоксина, то есть переменная защита, двигающая саму границу окна.',
          '\nЧего этот раздел НЕ делает: всплеск действует только на нагрузку через кору. Второй путь — серотонин как часть восходящей активации, мешающая заснуть, — не реализован, потому что у Филлипса–Робинсона ядра шва слиты с остальными моноаминовыми в одну группу и порознь не разделяются. Это задача для модели семейства Diniz Behn и Booth.']
    R['F8'] = dict(a=dict(trig_share=mb['trig'], thresh=0.80, verdict=v8a), b=dict(dur=mb['dur'], thresh=2.0, verdict=v8b),
                   c=dict(short=mb['short'], short_pool=mb['short_pool'], ep_min=mb['ep_min'], thresh=0.30, verdict=v8c), d=dict(nday_by_rate=nd_seq, verdict=v8d),
                   amp=SR_AMP, rate=SR_RATE, ceiling=SR_CEIL, tau_h=TAU_SR,
                   rate_scan=[{kk: vv for kk, vv in r_.items() if kk != 'dur_arr'} for r_ in rrows],
                   amp_scan=[{kk: vv for kk, vv in r_.items() if kk != 'dur_arr'} for r_ in arows_sr],
                   control={kk: vv for kk, vv in m0sr.items() if kk != 'dur_arr'})

    # ---------- 4е. Правила Ф11 и Ф12: двухступенчатая защита ----------
    stage('двухступенчатая защита: сила индукции и рефрактерный период')
    bf = np.array([float(S['bfac'][d['k'], 0]) for d in sel])                            # кратность индукции beta/N0 у каждого набора
    rb5 = keep[0.0]                                                                      # связь со сном выключена: смотрим чистое действие защиты
    from scipy.stats import rankdata as _rd2
    epm = np.where(rb5['epn'] > 0, rb5['epsum'] / np.maximum(rb5['epn'], 1), np.nan)
    okb = band & (rb5['epn'] > 0)
    c11a = float(np.corrcoef(_rd2(bf[okb]), _rd2(epm[okb]))[0, 1]) if okb.sum() >= 10 else float('nan')
    hi_b = okb & (bf >= 2.0); lo_b = okb & (bf < 1.0)
    dur_hi = float(np.nanmedian(epm[hi_b])) if hi_b.any() else float('nan')
    dur_lo = float(np.nanmedian(epm[lo_b])) if lo_b.any() else float('nan')
    sh = np.where(rb5['epn'] > 0, rb5['epshort'] / np.maximum(rb5['epn'], 1), np.nan)
    short_hi = float(np.nanmedian(sh[hi_b])) if hi_b.any() else float('nan')
    v11a = bool(c11a == c11a and c11a <= -0.40); v11b = bool(dur_hi == dur_hi and dur_hi <= 2.0 and short_hi >= 0.30)
    flick = band & (rb5['tdam'] > 0.5) & ((rb5['Dmax'] - rb5['Dmin']) > 0.2)
    v11d = bool(float(flick.mean()) <= 0.25)
    L += ['\n## 4е. Защита в две ступени: правила Ф11 и Ф12\n',
          'Что изменено в уравнениях. Прежде защита была одна и подчинялась одной постоянной времени T (около 8 часов) и на подъём, и на спад. Теперь ступеней две, и обе постоянные вынуты из сырых измерений, а не придуманы:\n',
          '- сигнал индукции (мРНК сульфиредоксина): постоянная %.0f ч, полураспад около 8 ч;\n- сам белок: постоянная %.0f ч, полураспад около 44 ч.\n' % (TAU_M * 24, TAU_S * 24),
          'Источник — сырые измерения из работы о динамике пути Nrf2 в клетках печени: 876 точек по сульфиредоксину, 17 условий, 44 часа наблюдения. Считал по разности «доза минус контроль», потому что у наблюдаемой неизвестен сдвиг, а разность его убирает. Контроль там сам даёт горб, и без его вычитания спад белка завышается почти целиком за счёт этого горба.\n',
          'Стационарное состояние при этом НЕ изменилось: при постоянной нагрузке защита приходит туда же, куда приходила. Поэтому окно, его границы, полоса и весь раздел 1в остались прежними — изменился только ход во времени. Это и позволяет сравнивать с прошлыми прогонами.\n',
          'Выборка силы индукции тоже изменена, и тоже по измерению: прежде beta бралась из [0, 1] при N0 из [0,3; 1], и кратность beta/N0 почти не достигала измеренных 2,85. Теперь beta = N0 * U(0; %.1f), то есть кратность берётся прямо и покрывает измеренную.\n' % BFMAX,
          '\n### Правило Ф11: закрывает ли индукция эпизод\n',
          '| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|',
          '| а: длительность падает с силой индукции | ранговая связь %s | не слабее −0,40 | %s |' % (('%+.2f' % c11a) if c11a == c11a else '—', 'выполнено' if v11a else 'не выполнено'),
          '| б: при измеренной силе (кратность ≥ 2) эпизод короткий | %.2f сут, коротких по наборам %.3f (по всем эпизодам %.3f) | не больше 2 сут и не меньше 0,30 | %s |' % (
              dur_hi, short_hi, float(rb5['epshort'][hi_b].sum() / max(rb5['epn'][hi_b].sum(), 1)) if hi_b.any() else float('nan'), 'выполнено' if v11b else 'не выполнено'),
          '| г: без вырождения в дребезг | %.3f наборов полосы | не больше 0,25 | %s |' % (float(flick.mean()), 'выполнено' if v11d else 'не выполнено'),
          '\nДля сравнения: у наборов со слабой индукцией (кратность меньше 1) длительность %.2f суток.' % dur_lo,
          '\n| кратность индукции beta/N0 | наборов | длительность эпизода, сут | эпизодов короче суток |\n|---|---|---|---|']
    bfb = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]; brows = []
    for a_, b_ in bfb:
        mm = okb & (bf >= a_) & (bf < b_); n_ = int(mm.sum())
        d_ = float(np.nanmedian(epm[mm])) if n_ else float('nan'); s2_ = float(np.nanmedian(sh[mm])) if n_ else float('nan')
        brows.append(dict(lo=a_, hi=b_, n=n_, dur=d_, short=s2_))
        L.append('| %.1f–%.1f | %d | %s | %s |' % (a_, b_, n_, ('%.2f' % d_) if n_ else '—', ('%.3f' % s2_) if n_ else '—'))
    # ---- Ф12: рефрактерный период ----
    gp = rb5['gaps']; NGB = rb5['NGB']; gtot = gp.sum()
    gmean = float(np.nansum(rb5['gap_sum']) / max(np.nansum(rb5['gap_n']), 1))
    obs2 = float(gp[:2].sum() / max(gtot, 1))                                            # доля промежутков короче 2 суток
    exp2 = float(1 - math.exp(-2.0 / max(gmean, 1e-9)))                                  # столько дала бы случайность без памяти
    v12a = bool(gtot >= 50 and exp2 > 0 and obs2 / exp2 <= 0.5)
    sAs, sAn = rb5['s_after']; sHs, sHn = rb5['s_health']; NSB = rb5['NSB']
    sh0 = float(np.nansum(sHs) / max(np.nansum(sHn), 1))
    scur = np.where(sAn > 0, sAs / np.maximum(sAn, 1), np.nan) / max(sh0, 1e-9)
    above = [j for j in range(NSB) if sAn[j] > 0 and scur[j] >= 1.5]
    v12b = bool(above and max(above) >= 1)
    gpc = rr_ns['gaps']; gtc = gpc.sum()                                                 # контроль: та же реализация шума, но вторая ступень без длинной памяти
    gmc = float(np.nansum(rr_ns['gap_sum']) / max(np.nansum(rr_ns['gap_n']), 1))
    obsc = float(gpc[:2].sum() / max(gtc, 1)); expc = float(1 - math.exp(-2.0 / max(gmc, 1e-9)))
    rat_f = obs2 / max(exp2, 1e-9); rat_c = obsc / max(expc, 1e-9)
    v12c = bool(gtot >= 50 and gtc >= 50 and rat_c >= 1.5 * rat_f and (rat_c - rat_f) >= 0.15)   # и в разах, и в абсолютной величине: при отношении около нуля одного отношения мало
    L += ['\n### Правило Ф12: есть ли рефрактерный период\n',
          'Если белок-защита держится двое суток после эпизода, то сразу после него клетка защищена лучше обычного, и следующий эпизод должен наступать реже, чем дала бы простая случайность.\n',
          '**Почему добавлена часть (в).** В прошлом прогоне части (а) и (б) нельзя было истолковать: медленный шум нагрузки имеет время корреляции %.0f суток, то есть ровно тот же масштаб, что и предполагаемый рефрактерный период, и действует В ПРОТИВОПОЛОЖНУЮ сторону — он СБИВАЕТ эпизоды в кучу, а защита должна их РАЗДВИГАТЬ. По одному прогону эти два действия не разделяются вовсе. Поэтому добавлен парный контроль: тот же набор, то же зерно, тот же шум — но вторая ступень защиты лишена длинной памяти (её постоянная поставлена равной постоянной первой, %.0f ч вместо %.0f ч). Стационарное состояние при этом снова не меняется, меняется только длина памяти. Если провал в распределении промежутков даёт защита, он должен ЗАПОЛНИТЬСЯ в контроле; если провал даёт шум, контроль покажет то же самое.\n' % (TSLOW, TAU_M * 24, TAU_S * 24),
          '| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|',
          '| а: провал в начале распределения промежутков | %.3f против %.3f у случайности, отношение %.2f | отношение не больше 0,50 | %s |' % (obs2, exp2, obs2 / max(exp2, 1e-9), 'выполнено' if v12a else 'не выполнено'),
          '| б: защита остаётся повышенной не меньше полутора суток | %s | уровень не ниже 1,5 от здорового | %s |' % (('до %d–%d суток' % (max(above), max(above) + 1)) if above else 'не держится', 'выполнено' if v12b else 'не выполнено'),
          '| в: провал пропадает, если убрать длинную память (парный контроль) | отношение %.2f в модели против %.2f в контроле | в контроле не меньше чем в полтора раза больше И не меньше чем на 0,15 выше | %s |' % (rat_f, rat_c, 'выполнено' if v12c else 'не выполнено'),
          '\n- контроль: промежутков %d, средний %.2f суток, доля коротких %.3f (в модели %d, %.2f суток, %.3f)' % (int(gtc), gmc, obsc, int(gtot), gmean, obs2),
          '\nЧасть (в) — главная из трёх: только она отделяет защиту от медленного шума. Части (а) и (б) без неё остаются описанием, а не проверкой.',
          '\n- всего промежутков между эпизодами: %d; средний промежуток %.2f суток\n' % (int(gtot), gmean),
          '| суток между эпизодами | 0–1 | 1–2 | 2–3 | 3–4 | 4–5 | 5–6 | 6–7 | 7–8 | 8–9 | 9+ |\n|---|---|---|---|---|---|---|---|---|---|---|',
          '| доля | ' + ' | '.join('%.3f' % (gp[j] / max(gtot, 1)) for j in range(NGB)) + ' |',
          '| случайность без памяти | ' + ' | '.join('%.3f' % (math.exp(-j / max(gmean, 1e-9)) - math.exp(-(j + 1) / max(gmean, 1e-9))) for j in range(NGB)) + ' |',
          '\n| суток после эпизода | ' + ' | '.join('%d–%d' % (j, j + 1) for j in range(NSB)) + ' |\n|' + '---|' * (NSB + 1),
          '| защита, в долях здорового уровня | ' + ' | '.join(('%.2f' % scur[j]) if sAn[j] > 0 else '—' for j in range(NSB)) + ' |']
    R['F11'] = dict(a=dict(corr=c11a, thresh=-0.40, verdict=v11a), b=dict(dur_hi=dur_hi, short_hi=short_hi, dur_lo=dur_lo, verdict=v11b),
                    d=dict(flicker=float(flick.mean()), verdict=v11d), by_bfac=brows, tau_m_h=TAU_M * 24, tau_s_h=TAU_S * 24, bfmax=BFMAX)
    R['F12'] = dict(a=dict(obs_short=obs2, exp_short=exp2, ratio=rat_f, n_gaps=int(gtot), gap_mean=gmean, verdict=v12a),
                    b=dict(curve=[None if sAn[j] == 0 else float(scur[j]) for j in range(NSB)], verdict=v12b),
                    c=dict(ratio_model=rat_f, ratio_control=rat_c, obs_control=obsc, exp_control=expc, n_gaps_control=int(gtc),
                           gap_mean_control=gmc, tau_s_control_h=TAU_M * 24, thresh='контроль >= 1,5 * модель', verdict=v12c),
                    gap_hist=[float(z) for z in gp], gap_hist_control=[float(z) for z in gpc])

    # ---------- 5. Правило Ф5: градуальна ли доля повреждённых ----------
    hD, hDm = rb_['hist']; hD = hD / max(hD.sum(), 1); hDm = hDm / max(hDm.sum(), 1)
    mid = slice(int(0.3 * NHD), int(0.7 * NHD)); midD = float(hD[mid].sum()); midDm = float(hDm[mid].sum())
    v5 = bool(midD >= 0.05)
    L += ['\n## 5. Правило Ф5: стала ли переменная повреждения градуальной\n',
          '| величина | доля времени в середине шкалы (0,3–0,7) | во сколько раз больше одноклеточного d |\n|---|---|---|',
          '| доля повреждённых D | %.4f | %.1f |' % (midD, midD / 0.011), '| среднее повреждение | %.4f | %.1f |' % (midDm, midDm / 0.011),
          '\nУ одной клетки в diag_gradual.py выходило 0,009–0,013. Порог правила: не меньше 0,05, то есть хотя бы вчетверо больше.',
          '\nЧто уже известно и что надо держать в уме при чтении этой цифры: в прогоне при q = 0,75 она шла по разбросу уязвимости почти один в один — 0,017 / 0,056 / 0,133 / 0,346 / 0,726 при sigma_v 0,05 / 0,10 / 0,15 / 0,25 / 0,40, а при самом узком разбросе градуальность почти исчезала (0,017 против 0,011 у одной клетки). Значит плавность даёт РАЗБРОС уязвимости, а не связь через сеть. Koulakov и соавт. 2002 ровно это и утверждают; надежда, что круг E → L → D → g → E сам создаст промежуточные состояния, пока не подтвердилась. Перебор по sigma_v в разделе 7 — главная проверка этого вывода.',
          '\n**Ф5:** получено %.4f. Вывод: %s.' % (midD, 'выполнено' if v5 else 'не выполнено'),
          '\n| ячейка D | доля времени | ячейка | доля времени |\n|---|---|---|---|']
    for j in range(NHD // 2): L.append('| %.2f–%.2f | %.4f | %.2f–%.2f | %.4f |' % (j / float(NHD), (j + 1) / float(NHD), hD[j], (j + NHD // 2) / float(NHD), (j + NHD // 2 + 1) / float(NHD), hD[j + NHD // 2]))
    R['F5'] = dict(mid_share_D=midD, mid_share_Dmean=midDm, single_cell_ref=0.011, thresh=0.05, verdict=v5, hist_D=[float(z) for z in hD], hist_Dmean=[float(z) for z in hDm])

    # ---------- 6. Правило Ф2 ----------
    nAs = len(ASCAN); nVs = len(VSCAN); nQs = len(QSCAN)
    bh = runb(np.concatenate([np.tile(num, nAs + nVs)] + [lo * (hi / lo) ** qp for qp in QSCAN]), nAs + nVs + nQs, 'размах, разброс, посадка',
              kd=np.array([0.0] * nAs + [KBASE] * (nVs + nQs)), aH=np.array([A_H * af for af in ASCAN] + [A_H] * (nVs + nQs)),
              sigv=np.array([SIGV] * nAs + list(VSCAN) + [SIGV] * nQs), E0=E0, qw=qw, Gbar=Gbar, sigma=SIG, tauc_rel=TAUC,
              ndays=max(ND // 2, 5), M=M, J=J, seed=SEED, warm=WARM2)
    arows = [];
    for af, rr in zip(ASCAN, bh[:nAs]):
        m = meas(rr); m['a_fac'] = af; m['swing'] = DSWING * af; arows.append(m)
    sh = [r_['share_ever'] for r_ in arows]
    gro = max(sh) / max(min(sh), 1e-9); r0 = rows[KSCAN.index(0.0)]
    v2a = bool(r0['late'] > 0.5); v2b = bool(gro >= 4.0 and max(np.diff(sh)) > 2.5 * np.median(np.abs(np.diff(sh)))); v2c = bool(r0['sleep_end'] < 0.3)
    L += ['\n## 6. Правило Ф2: суточный член\n',
          '| часть | получено | прежде | ожидание до счёта | вывод |\n|---|---|---|---|---|',
          '| а: срывы во второй половине бодрствования | %.3f | 0,733 | больше 0,5 | %s |' % (r0['late'], 'выполнено' if v2a else 'не выполнено'),
          '| б: есть порог по размаху A* | рост в %.1f раза при размахе от 0 до %.2f | в 2,3 раза, плавно | перелом, а не плавный рост | %s |' % (gro, DSWING * max(ASCAN), 'выполнено' if v2b else 'не выполнено'),
          '| в: сон редко кончает эпизод | %.3f | 0,500 | меньше 0,3 | %s |' % (r0['sleep_end'], 'выполнено' if v2c else 'не выполнено'),
          '\nЧасть (в) измерена ещё и вторым способом, без вердикта: доля НОЧЕЙ, случившихся во время эпизода, которые его оборвали, — %.3f. Записанная до счёта мера отвечает на вопрос «куда приходятся концы эпизодов», вторая — на вопрос «обрывает ли ночь эпизод». Это разные величины, и вердикт выносится только по первой.' % r0['sleep_kills'],
          '\n### Перебор размаха суточной нагрузки (измерено при k = 0)\n', '| множитель при A_H | размах нагрузки | доля повреждавшихся | срывов во второй половине бодрствования | доля времени в повреждении | эпизодов короче суток | ночей, оборвавших эпизод |\n|---|---|---|---|---|---|---|']
    for r_ in arows: L.append('| %.2f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |' % (r_['a_fac'], r_['swing'], r_['share_ever'], r_['late'], r_['tdam_ev'], r_['short'], r_['sleep_kills']))
    L += ['\nЧасть (б) не выполнялась в прежней модели, и отрицательный результат сохранён намеренно: суточный размах выглядит модулятором риска, а не воротами. Сверка нужна именно потому, что ответ заранее известен и неудобен.']
    R['F2'] = dict(a=dict(late_share=r0['late'], prev=0.733, verdict=v2a), b=dict(shares=sh, growth=float(gro), a_scan=ASCAN, prev_growth=2.3, verdict=v2b),
                   c=dict(sleep_end=r0['sleep_end'], sleep_kills=r0['sleep_kills'], prev=0.500, verdict=v2c), rows=arows)

    # ---------- 7. Переборы без вердикта ----------
    vrows = []
    for sv, rr in zip(VSCAN, bh[nAs:nAs + nVs]):
        m = meas(rr); m['sigv'] = sv; hh = rr['hist'][0]; hh = hh / max(hh.sum(), 1); m['mid'] = float(hh[mid].sum()); vrows.append(m)
    L += ['\n## 7. Переборы без вердикта\n', '### Разброс уязвимости клеток sigma_v (единственный новый параметр шага А)\n',
          '| sigma_v | доля времени D в середине шкалы | доля повреждавшихся | длительность эпизода, сут | застряли >1,5 сут | средняя доля повреждённых |\n|---|---|---|---|---|---|']
    for r_ in vrows: L.append('| %.2f | %.4f | %.3f | %.2f | %.3f | %.3f |' % (r_['sigv'], r_['mid'], r_['share_ever'], r_['dur'], r_['stuck15'], r_['Dbar']))
    L.append('\nЭто главная чувствительность всей переделки: если выводы держатся только при одном значении sigma_v, значит они держатся на разбросе, а не на механизме.')
    qrows = []
    for qp, rr in zip(QSCAN, bh[nAs + nVs:]):
        m = meas(rr); m['q'] = qp; qrows.append(m)
    L += ['\n### Насколько глубоко посажена средняя нагрузка (медианы по наборам со срывами)\n',
          '| q | доля повреждавшихся | время в повреждении | срывов в сутки | застряли >1,5 сут | эпизодов короче суток | средняя доля повреждённых |\n|---|---|---|---|---|---|---|']
    for r_ in qrows: L.append('| %.2f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |' % (r_['q'], r_['share_ever'], r_['tdam_ev'], r_['nday'], r_['stuck15'], r_['short'], r_['Dbar']))
    R['vscan'] = vrows; R['qscan'] = qrows

    # ---------- 7в. Правило Ф13: зависит ли ответ от числа клеток ----------
    def cnv(Jx, kind, s_=None, nq=999):
        """Насколько распределение множителей при данном J отличается от своего предела: по квантилям."""
        s_ = SIGV if s_ is None else s_; ref = vuln(2048, s_, kind)[0]; a_ = vuln(Jx, s_, kind)[0]
        qq = np.linspace(0.001, 0.999, nq); return float(np.abs(np.quantile(a_, qq) - np.quantile(ref, qq)).max())
    JS13 = [16, J0V] if SMOKE else ([16, J0V, 64, 128] if FULL else [16, J0V, 64])
    j13 = []
    for Jx in JS13:
        bj = runb(num, 2, 'число клеток J = %d, две сетки' % Jx, kd=0.0, aH=A_H, sigv=SIGV, E0=E0, qw=qw, Gbar=Gbar,
                  sigma=SIG, tauc_rel=TAUC, ndays=max(ND // 2, 5), M=M, J=Jx, seed=SEED, warm=WARM2, sigs=SIGS_BASE,
                  vfix=np.array([1.0, 0.0]))                                             # блок 0 — закреплённый диапазон, блок 1 — равновероятные квантили
        for gi, gn in ((0, 'закреплённый диапазон'), (1, 'равновероятные квантили')):
            mm = meas(bj[gi], band); hh = bj[gi]['hist'][0]; hh = hh / max(hh.sum(), 1); vv13 = vuln(Jx, SIGV, 1.0 - gi)[0]
            j13.append(dict(J=Jx, grid=gn, share_ever=mm['share_ever'], dur=mm['dur'], dur_m=mm['dur_m'], short_m=mm['short_m'],
                            mid=float(hh[mid].sum()), n_ep_m=mm['n_ep_m'], thr_eff=(int(Jx * DTHR) + 1) / float(Jx),
                            v_min=float(vv13.min()), v_max=float(vv13.max())))
    KK13 = (('share_ever', 'доля повреждавшихся'), ('dur_m', 'длительность эпизода по непрерывной переменной, сут'), ('mid', 'Ф5: доля времени в середине шкалы'))
    def spr(rws, k_):
        vs = [r_[k_] for r_ in rws if r_[k_] == r_[k_]]
        return float((max(vs) - min(vs)) / max(abs(np.mean(vs)), 1e-12)) if len(vs) >= 2 else float('nan')
    nw13 = [r_ for r_ in j13 if r_['grid'].startswith('закр')]; od13 = [r_ for r_ in j13 if not r_['grid'].startswith('закр')]
    sp_n = {k_: spr(nw13, k_) for k_, _ in KK13}; sp_o = {k_: spr(od13, k_) for k_, _ in KK13}
    v13a = bool(all(sp_n[k_] == sp_n[k_] and sp_n[k_] <= 0.10 for k_, _ in KK13))
    v13b = bool(any(sp_o[k_] == sp_o[k_] and sp_o[k_] > 0.10 for k_, _ in KK13))
    j32 = [r_ for r_ in j13 if r_['J'] == J0V]
    e13 = float(max(abs(j32[0][k_] - j32[1][k_]) for k_, _ in KK13)) if len(j32) == 2 else float('nan')
    v13c = bool(len(j32) == 2 and e13 == 0.0)
    L += ['\n## 7в. Правило Ф13: зависит ли ответ от числа клеток в популяции\n',
          'Зачем это понадобилось. Число клеток J до сих пор выбиралось по удобству счёта — 32 в одних прогонах, 64 в других, 24 по умолчанию, — и НИ ОДИН параметр модели с ним не связан. Но J входил в модель не как разрешение, а как параметр, и в двух местах сразу.\n',
          '**Место первое: хвост распределения уязвимости.** Множители нагрузки брались равновероятными квантилями p_j = (j + 0,5)/J, поэтому крайний квантиль ползёт с J: чем больше клеток, тем более уязвима самая уязвимая. При sigma_v = %.2f самый уязвимый множитель идёт 0,854 / 0,826 / 0,802 / 0,781 / 0,763 при J = 8 / 16 / 32 / 64 / 128. Переход с 32 на 64 сдвигает первую ломающуюся клетку на 0,33 сигмы быстрого шума (при sigma_v = 0,20 — на 0,66 сигмы). На фоне закона сохранения разрывов, где всё меряется единицами сигм, это не мелочь.\n' % SIGV,
          '**Место второе: порог эпизода квантован.** Эпизод — это D выше %.2f, а D принимает только значения j/J, поэтому настоящий порог равен (целая часть J/2 + 1)/J: 0,625 при J = 8, 0,531 при J = 32, 0,516 при J = 64. Само определение эпизода ползёт с J. Поэтому здесь ЕЩЁ ОДНА мера длительности — по среднему повреждению, где порог не квантован вовсе; вердикт выносится по ней.\n' % DTHR,
          'Что сделано. Сетка вероятностей закреплена: p_j равномерно от %.4f до %.4f при любом J. Крайний квантиль привязан к J = %d, поэтому при J = %d новая сетка совпадает с прежней ДО ПОСЛЕДНЕГО ЗНАКА — это часть (в) правила, — и все прежние прогоны при J = 32 остаются в силе без пересчёта. Теперь J — только мелкость разбиения, и это утверждение проверяемо.\n' % (P0V, 1 - P0V, J0V, J0V),
          '**Чего мы НЕ знаем и о чём честно говорим.** Закреплённый диапазон — это прочтение «популяция покрывает фиксированный разброс уязвимости». Второе прочтение — «популяция есть J клеток, честно вытянутых из логнормали», и тогда более крайняя клетка при большем J не артефакт, а правда, а J надо анкерить по измерению. Прямого измерения «сколько PV-клеток делят одну нагрузку» нет. Косвенно: плотность PV-клеток в бочонковой коре мыши 8427 ± 1405 на мм³ в слое IV и около 8500–8960 в слое V (Frontiers in Neuroanatomy 2019, по бочонковым колонкам и септам). При колонке диаметром 300 мкм и толщине слоя IV 200 мкм это около 120 PV-клеток в слое IV и порядка 400 на всю колонку — НО геометрия здесь МОЁ ДОПУЩЕНИЕ, а не измерение, и это мышь, а не префронтальная кора человека. Для человека метаанализ девяти работ даёт только размеры эффекта (g = −0,27), абсолютных плотностей там нет. Вывод для модели: биологическое J — это скорее сотни, а не десятки, то есть все наши прогоны стоят в области, где хвост заужен; при J порядка 400 квантование порога тоже пропадает само (0,5025).\n',
          '\n### Чем обоснован выбор сетки: сходимость, посчитанная без всякого счёта\n',
          'Выбор между построениями сделан по измеримому признаку, а не по вкусу. Признак: насколько распределение множителей при данном J отличается от своего же предела при J, уходящем в бесконечность (наибольшее расхождение по квантилям, эталон J = 2048). Это чистая арифметика сетки, модель тут не участвует вовсе.\n',
          '| J | равновероятные квантили (как было) | закреплённый диапазон, края (взято) | закреплённый диапазон, середины |\n|---|---|---|---|'] + [
          '| %d | %.5f | %.5f | %.5f |' % (Jx, cnv(Jx, 0), cnv(Jx, 1), cnv(Jx, 2)) for Jx in (8, 16, 24, J0V, 64, 128)] + [
          '\nЗакрепление по КРАЯМ сходится быстрее всего: от J = 16 к J = 128 расхождение падает в %.0f раз, тогда как у прежней сетки — всего в %.1f раза. Закрепление по СЕРЕДИНАМ по смыслу выглядит естественнее, но хуже: его опора всё равно ползёт (самый уязвимый множитель %.4f при J = 32 против %.4f при J = 128, то есть на %.1f процента), и вдобавок оно разошлось бы с прежними прогонами при J = 32 на %.1f процента. Края дают сдвиг опоры %.2f процента и совпадение с прежними прогонами ровно нулевое.\n' % (
              cnv(16, 1) / max(cnv(128, 1), 1e-12), cnv(16, 0) / max(cnv(128, 0), 1e-12),
              float(vuln(J0V, SIGV, 2)[0].min()), float(vuln(128, SIGV, 2)[0].min()),
              100 * abs(vuln(128, SIGV, 2)[0].min() / vuln(J0V, SIGV, 2)[0].min() - 1),
              100 * abs(vuln(J0V, SIGV, 2)[0].min() / vuln(J0V, SIGV, 0)[0].min() - 1),
              100 * abs(vuln(128, SIGV, 1)[0].min() / vuln(J0V, SIGV, 1)[0].min() - 1)),
          'Выбор НЕ подгонка под желаемый ответ: признак посчитан до всякого счёта и не зависит ни от одного параметра модели кроме sigma_v. Он же объясняет, почему при J = 8 закрепление по краям выглядело бы хуже прежней сетки: восемь точек далеки от любого предела. Поэтому J = 8 в перебор не берётся.\n',
          '\n### Правило Ф13, записано до счёта\n',
          '(а) На закреплённой сетке три величины — доля повреждавшихся, длительность эпизода по непрерывной переменной и Ф5 — меняются по всему перебору J не больше чем на 10 %% от своего среднего.\n(б) На прежней сетке хотя бы одна из этих трёх меняется больше чем на 10 %%. Это контроль наоборот: без него часть (а) ничего не значит, потому что могла бы пройти от нечувствительности меры.\n(в) При J = %d обе сетки дают расхождение РОВНО нуль.\n' % J0V,
          '| сетка | J | множители от / до | настоящий порог эпизода | доля повреждавшихся | длительность по D, сут | длительность по среднему повреждению, сут | коротких | эпизодов в счёте | Ф5: середина шкалы |\n|---|---|---|---|---|---|---|---|---|---|']
    for r_ in j13:
        L.append('| %s | %d | %.3f / %.3f | %.4f | %.3f | %s | %s | %s | %d | %.4f |' % (
            r_['grid'], r_['J'], r_['v_min'], r_['v_max'], r_['thr_eff'], r_['share_ever'],
            ('%.2f' % r_['dur']) if r_['dur'] == r_['dur'] else '—', ('%.2f' % r_['dur_m']) if r_['dur_m'] == r_['dur_m'] else '—',
            ('%.3f' % r_['short_m']) if r_['short_m'] == r_['short_m'] else '—', r_['n_ep_m'], r_['mid']))
    L += ['\n| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|',
          '| а: на закреплённой сетке ответ от J не зависит | %s | каждая не больше 0,10 | %s |' % (
              '; '.join('%s %.3f' % (nm, sp_n[k_]) for k_, nm in KK13), 'выполнено' if v13a else 'не выполнено'),
          '| б: на прежней сетке зависит (контроль наоборот) | %s | хотя бы одна больше 0,10 | %s |' % (
              '; '.join('%s %.3f' % (nm, sp_o[k_]) for k_, nm in KK13), 'выполнено' if v13b else 'не выполнено'),
          '| в: при J = %d сетки совпадают | расхождение %.3e | ровно нуль | %s |' % (J0V, e13, 'выполнено' if v13c else 'НЕ ВЫПОЛНЕНО'),
          '\nЧто означает каждый исход, записано до счёта. Если не выполнено (а) — J влез в модель третьим путём, которого я не вижу, и его надо искать, а не замазывать. Если не выполнено (в) — ошибка в самой переанкеровке, и тогда вся таблица недействительна. Если не выполнено (б), а эпизодов в каждой строке не меньше тридцати, то вывод обратный тому, с которого я начинал, и его надо принять как есть: ответ модели к числу клеток нечувствителен, значит смешение J = 32 и J = 64 в прежних прогонах было безвредным, и я поднял ложную тревогу. Если же (б) не выполнено при малом числе эпизодов, то тест просто нечувствителен, и (а) надо перепроверять на других величинах, а не объявлять выполненным.',
          '\nПочему вердикт по длительности выносится по НЕПРЕРЫВНОЙ переменной, а не по доле повреждённых: порог по доле квантован шагом 1/J, и он сам ползёт с J (столбец «настоящий порог эпизода»). Мерить J-независимость мерой, которая от J зависит по построению, бессмысленно. Столбец по доле оставлен рядом для сверки с прежними прогонами.',
          '\nОтдельная ось, НАМЕРЕННО не тронутая в этом прогоне: квантование порога эпизода. Менять сетку уязвимости и определение эпизода в одном сравнении — это ровно та ошибка, которую мы допустили в mass_core5.py, когда вместе с динамикой защиты поменялась и выборка силы индукции.']
    R['F13'] = dict(rows=j13, spread_fixed=sp_n, spread_old=sp_o, thresh=0.10, J_anchor=J0V, p0=P0V,
                    convergence={str(Jx): dict(old=cnv(Jx, 0), fixed_edges=cnv(Jx, 1), fixed_mid=cnv(Jx, 2)) for Jx in (8, 16, 24, J0V, 64, 128)},
                    a=dict(verdict=v13a), b=dict(verdict=v13b), c=dict(err=e13, verdict=v13c),
                    lit=dict(pv_density_L4_mouse_per_mm3=8426.59, pv_density_L4_sd=1405.35,
                             est_pv_per_barrel_column='120 в слое IV, около 400 на колонку — геометрия моё допущение',
                             source='Frontiers in Neuroanatomy 2019, бочонковая кора мыши'))

    # ---------- помощник к Ф15: CV^2 промежутков ВНУТРИ набора ----------
    NGMIN = 5                                                                            # меньше пяти промежутков на набор не берём: выборочная CV^2 слишком смещена
    rng15 = np.random.default_rng(555)
    def cv2_sets(rr, mask, nrep=400):
        """Медиана по наборам от CV^2 их собственных промежутков, и её нулевое значение при ТОМ ЖЕ наборе n.
        Вывод (см. analiz_velichin_do_progona.txt): при глухом периоде t_r и пуассоновском риске
        CV^2 = (1 - t_r/g)^2, а без глухого периода CV^2 = 1 РОВНО при любой частоте. Величина
        безразмерна и считается ПО НАБОРУ, поэтому смешение наборов в неё не попадает — в отличие
        от доли коротких промежутков, которая равна 1 + CV^2(частот) и физики не видит вовсе."""
        n_ = rr['gap_n'][mask]; ok = n_ >= NGMIN
        if ok.sum() < 5: return float('nan'), float('nan'), int(ok.sum())
        n_ = n_[ok]; sm = rr['gap_sum'][mask][ok]; sq = rr['gap_sq'][mask][ok]
        mu = sm / n_; var = np.maximum((sq - n_ * mu * mu) / np.maximum(n_ - 1, 1), 0.0)
        obs = float(np.median(var / np.maximum(mu * mu, 1e-12)))
        nul = []                                                                         # нуль при нашем же наборе n: экспоненциальные промежутки
        for _ in range(nrep):
            v = []
            for nn in n_.astype(int):
                g = rng15.exponential(1.0, nn); v.append(g.var(ddof=1) / max(g.mean() ** 2, 1e-12))
            nul.append(np.median(v))
        return obs, float(np.median(nul)), int(ok.sum())

    # ---------- 7г. Правило Ф14: есть ли у защиты собственная память ----------
    TS14 = [TAU_M, 24. / 24, 63. / 24, 88. / 24, 150. / 24]                              # сутки: контроль, три значения из сырого опыта и 150 ч — точка для ПРОВЕРКИ предсказания (разд. 7д)
    SG14 = [0.0, SIGS_BASE]
    pr14 = [(sg, ts_) for sg in SG14 for ts_ in TS14]
    b14 = runb(num, len(pr14), 'память защиты', kd=0.0, aH=A_H, sigv=SIGV, E0=E0, qw=qw, Gbar=Gbar, sigma=SIG, tauc_rel=TAUC,
               ndays=ND, M=M, J=J, seed=SEED, warm=max(WARM, WARMS),
               sigs=np.array([z[0] for z in pr14]), taus=np.array([z[1] for z in pr14]))
    r14 = []
    for (sg, ts_), rr in zip(pr14, b14):
        gp_ = rr['gaps']; gt_ = float(gp_.sum())
        gm_ = float(np.nansum(rr['gap_sum']) / max(np.nansum(rr['gap_n']), 1))
        ob_ = float(gp_[:2].sum() / max(gt_, 1)); ex_ = float(1 - math.exp(-2.0 / max(gm_, 1e-9)))
        epm_ = np.where(rr['epn'] > 0, rr['epsum'] / np.maximum(rr['epn'], 1), np.nan)
        hb_ = band & (rr['epn'] > 0) & (bf >= 2.0)
        cv2m, cv2n, nn_ = cv2_sets(rr, band)
        r14.append(dict(sigs=sg, tau_s_h=ts_ * 24, n_gap=int(gt_), gap_mean=gm_, obs_short=ob_, exp_short=ex_,
                        cv2=cv2m, cv2_null=cv2n, n_sets_cv=int(nn_),
                        ratio=ob_ / max(ex_, 1e-9), dur_hi=float(np.nanmedian(epm_[hb_])) if hb_.any() else float('nan'),
                        n_hi=int(hb_.sum())))
    g14 = lambda sg: [z for z in r14 if abs(z['sigs'] - sg) < 1e-12]
    q0 = g14(0.0); qn = g14(SIGS_BASE)
    pick = lambda rws, th: next((z for z in rws if abs(z['tau_s_h'] - th) < 0.6), None)
    m0 = pick(q0, 63.); c0 = pick(q0, TAU_M * 24); mn = pick(qn, 63.); cn = pick(qn, TAU_M * 24)
    ok14 = all(z is not None and z['n_gap'] >= 50 for z in (m0, c0, mn, cn))
    v14a = bool(ok14 and m0['ratio'] <= 0.70 and c0['ratio'] >= 0.90)
    v14b = bool(ok14 and abs(mn['ratio'] - cn['ratio']) <= 0.15)
    mono = lambda rws: all(rws[i + 1]['gap_mean'] >= rws[i]['gap_mean'] - 1e-12 for i in range(len(rws) - 1))
    v14c = bool(ok14 and mono(sorted(q0, key=lambda z: z['tau_s_h'])) and mono(sorted(qn, key=lambda z: z['tau_s_h']))
                and m0['gap_mean'] >= 1.30 * c0['gap_mean'] and mn['gap_mean'] >= 1.30 * cn['gap_mean'])
    L += ['\n## 7г. Правило Ф14: есть ли у защиты собственная память\n',
          'Зачем. Раздел 4е показал две вещи. Защита РАБОТАЕТ: убрали её длинную память — эпизодов стало на %.0f %% больше. Но рефрактерного периода она не создаёт: избыток коротких промежутков (отношение 1,21 к случайности) в модели и в контроле ОДИН И ТОТ ЖЕ, то есть он весь от медленного шума нагрузки, а не от защиты. Медленный шум коррелирован 5 суток и сбивает эпизоды в кучу; защита должна их раздвигать. Пока оба действуют вместе, второе под первым не видно.\n' % 53.,
          'Поэтому проверка одна: **выключить медленный шум.** Если у защиты есть собственная память, при sigma_м = 0 провал в начале распределения промежутков обязан появиться, и обязан пропасть в контроле с короткой памятью. Если он не появится и там — значит памяти у защиты нет вовсе, а весь её вклад сводится к одному множителю скорости.\n',
          '**И сразу поправка к тому, что я писал раньше про постоянные времени.** Я перемерил сырые данные заново и нашёл у себя две ошибки. Пик белка не в 32 часах, а в 18 (при DEM 10 и 100). А постоянная спада НЕ ОПРЕДЕЛЕНА этими данными: она зависит от того, как убирать неизвестный сдвиг наблюдаемой.\n',
          '| как убран сдвиг наблюдаемой | пик белка | постоянная спада | полураспад |\n|---|---|---|---|',
          '| вычесть контроль того же часа | 18 ч | 24,4 ч | 16,9 ч |',
          '| вычесть только уровень в начале | 18 ч | 49,8 ч | 34,5 ч |',
          '| не вычитать ничего | 18 ч | 87,6 ч | 60,7 ч |',
          '\nНаша модель стоит на %.0f ч. Это ВНУТРИ интервала, но ближе к медленному краю, и пинается оно ничем. Контроль там сам даёт индукцию (его сигнал растёт с 2,03 до 3,46), поэтому вычитание контроля скорее перевычитает, а невычитание недовычитает; какой край правильный — из этих данных не следует. Раз раздел 4е показал, что именно эта постоянная задаёт масштаб промежутков, её нельзя держать назначенной: **она перебирается здесь вместе с шумом.**\n' % (TAU_S * 24),
          'Первая ступень, наоборот, определена хорошо: мРНК спадает с постоянной 9,6 / 9,0 / 8,1 ч при DEM 100 / 316 / 561 (полураспад 6,6 / 6,2 / 5,6 ч). В модели стоит %.0f ч — чуть медленнее измеренного, того же порядка.\n' % (TAU_M * 24),
          '\n### Правило Ф14, записано до счёта\n',
          '(а) При sigma_м = 0 отношение «коротких промежутков к случайности» у модели (память %.0f ч) не выше 0,70, а у контроля (память %.0f ч) не ниже 0,90.\n' % (63., TAU_M * 24),
          '(б) При sigma_м = %.2f эти два отношения расходятся не больше чем на 0,15 — то есть шум действительно маскирует память, как я утверждаю.\n' % SIGS_BASE,
          '(в) Средний промежуток растёт с длиной памяти при обоих sigma_м, и от %.0f ч до 63 ч — не меньше чем на 30 %%. Это ровно то число, которое я в прошлый раз забыл записать заранее и потому не имел права поставить себе в заслугу.\n' % (TAU_M * 24),
          '(г) Без вердикта: длительность эпизода при измеренной силе индукции против длины памяти.\n',
          '| sigma_м | память защиты, ч | промежутков | средний промежуток, сут | коротких | у случайности | отношение | длительность при кратности >= 2, сут | наборов |\n|---|---|---|---|---|---|---|---|---|']
    for z in r14:
        L.append('| %.2f | %.0f%s | %d | %.2f | %.3f | %.3f | %.2f | %s | %d |' % (
            z['sigs'], z['tau_s_h'], ' (контроль)' if abs(z['tau_s_h'] - TAU_M * 24) < 0.6 else '', z['n_gap'], z['gap_mean'],
            z['obs_short'], z['exp_short'], z['ratio'],
            ('%.2f' % z['dur_hi']) if z['dur_hi'] == z['dur_hi'] else '—', z['n_hi']))
    L += ['\n| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|']
    if ok14:
        L += ['| а: при sigma_м = 0 провал появляется | модель %.2f, контроль %.2f | модель не выше 0,70, контроль не ниже 0,90 | %s |' % (m0['ratio'], c0['ratio'], 'выполнено' if v14a else 'не выполнено'),
              '| б: при sigma_м = %.2f шум маскирует память | %.2f против %.2f, разница %.2f | разница не больше 0,15 | %s |' % (SIGS_BASE, mn['ratio'], cn['ratio'], abs(mn['ratio'] - cn['ratio']), 'выполнено' if v14b else 'не выполнено'),
              '| в: промежуток растёт с памятью | при sigma_м = 0: %.2f -> %.2f сут (в %.2f раза); при %.2f: %.2f -> %.2f (в %.2f раза) | монотонно и не меньше чем в 1,3 раза | %s |' % (
                  c0['gap_mean'], m0['gap_mean'], m0['gap_mean'] / max(c0['gap_mean'], 1e-9), SIGS_BASE, cn['gap_mean'], mn['gap_mean'], mn['gap_mean'] / max(cn['gap_mean'], 1e-9), 'выполнено' if v14c else 'не выполнено')]
    else:
        L.append('| все части | промежутков меньше 50 хотя бы в одной нужной строке | не меньше 50 | непроверяемо |')
    L += ['\nЧто означает каждый исход, записано до счёта. Выполнено (а) и (б) — у защиты есть собственная память, она просто спрятана под медленным шумом, и сульфиредоксин остаётся в модели как механизм памяти. Не выполнено (а) при выполненном (в) — памяти нет, защита работает только как множитель скорости, и тогда искать источник рефрактерности надо в другом месте, а не чинить этот. Не выполнено (в) — защита не влияет на частоту эпизодов вовсе, и тогда вывод раздела 4е про 53 % был артефактом одного зерна.',
          '\n**Оговорка, которая остаётся в силе.** Сигнал индукции в модели — это прямая функция перекиси, beta*u/(KB + u). В действительности путь идёт через Keap1 и Nrf2, и ровно этот путь в источнике и измерялся (в наборе есть Keap1, p62, Nrf2 как отдельные вещества). Наша одна экспонента с постоянной %.0f ч сворачивает всю эту цепочку в один шаг. Это МОЁ упрощение, а не измеренная связь.',
          '\n**Второе расхождение с данными, которое я не чиню в этом прогоне.** При высокой дозе белок не выходит на полку вовсе: при DEM 561 он растёт до самого конца опыта (кратность 7,0 за 44 ч и всё ещё вверх), при DEM 316 пик в 38 ч, а при DEM 10 и 100 — в 18 ч. У нас же белок при постоянной нагрузке всегда приходит к своему стационару. Наша выборка силы индукции beta = N0*U(0; %.1f) покрывает кратности до 3, то есть две нижние дозы (x2,10 и x2,66), но не x4,16 и не x7,03. Это следующий кандидат на проверку, и менять его ЗДЕСЬ нельзя: выборка и динамика в одном прогоне — та самая ошибка mass_core5.py.' % BFMAX]
    R['F14'] = dict(rows=r14, a=dict(verdict=v14a), b=dict(verdict=v14b), c=dict(verdict=v14c), checkable=ok14,
                    lit=dict(mrna_tau_h=[9.6, 9.0, 8.1], mrna_half_h=[6.6, 6.2, 5.6], protein_peak_h=18.,
                             protein_tau_h_by_baseline=dict(subtract_control=24.4, subtract_t0=49.8, raw=87.6),
                             induction_fold=dict(DEM10=2.10, DEM100=2.66, DEM316=4.16, DEM561=7.03),
                             model_tau_m_h=TAU_M * 24, model_tau_s_h=TAU_S * 24, bfmax=BFMAX,
                             note='постоянная спада белка не определена данными; пик в 18 ч, а не в 32 — поправка к прежнему разбору'))

    # ---------- 7д. Правило Ф15: рефрактерность, измеренная годной статистикой ----------
    stage('Ф15: CV^2 промежутков по наборам, с нулём при нашем же n')
    L += ['\n## 7д. Правило Ф15: рефрактерность, измеренная годной статистикой\n',
          'Почему прежнее измерение выброшено. Доля коротких промежутков, поделённая на «случайность без памяти», разбирается аналитически до конца. Наборы имеют разные частоты эпизодов, за равное время наблюдения набор даёт число промежутков, пропорциональное своей частоте, поэтому объединённая выборка смещена к быстрым наборам. Отсюда\n',
          '    R = E[l(1 − exp(−lT))] / ( E[l] (1 − exp(−T E[l])) ),   и при малом пороге   R → 1 + CV²(частот).\n',
          '**То есть эта величина измеряет разброс частот по наборам и ничего больше.** Наблюдённое R = 1,21 при среднем промежутке 5,01 суток отвечает разбросу частот CV = 0,63 — обычной разнородности нашего ансамбля. Вот почему R стояло на 1,07–1,33 во ВСЕХ восьми условиях прошлого прогона, включая полностью выключенный медленный шум. Правила Ф12(а), Ф12(в), Ф14(а) и Ф14(б) мерили этот артефакт, а не модель. Вердикты по ним НЕ ПЕРЕСМАТРИВАЮТСЯ (менять вердикт после счёта нельзя), но и доводом ни в одну сторону они не являются.\n',
          'Годная замена выводится так же коротко. Пусть после эпизода есть глухой период t_r, а дальше пуассоновский риск: промежуток = t_r + Exp(l), среднее g = t_r + 1/l, дисперсия 1/l², откуда\n',
          '    CV² = (1 − t_r/g)²,   то есть   t_r/g = 1 − CV.\n',
          'Без глухого периода CV² = 1 РОВНО, при любой частоте. Величина безразмерна, считается ПО КАЖДОМУ набору и потому смешения наборов не содержит. Выборочная CV² при малом числе промежутков занижена, но это смещение считается заранее розыгрышем при ТОМ ЖЕ наборе n, а не подгоняется.\n',
          '\n### Правило Ф15, записано до счёта\n',
          '(а) Рефрактерность есть, если медиана CV² по наборам ниже нулевого значения, посчитанного при нашем же n, не меньше чем на 0,10. Тогда глубина читается прямо: t_r/g = 1 − CV.\n(б) Контроль на отсутствие ошибки: та же статистика на разыгранных экспоненциальных промежутках с тем же n должна дать нулевое значение с точностью 0,03.\n(в) Контроль на мощность: на разыгранных промежутках с вшитым глухим периодом t_r = 0,25·g статистика должна дать не выше 0,50. По расчёту ожидается около 0,42 против нуля 0,77, то есть разделение полное.\n(г) Берутся только наборы, у которых не меньше %d промежутков.\n' % NGMIN,
          '| sigma_м | память защиты, ч | наборов в счёте | CV² наблюдённая | CV² под нулём (при том же n) | разность | t_r/g, если читать как глухой период |\n|---|---|---|---|---|---|---|']
    for z in r14:
        dlt = z['cv2_null'] - z['cv2'] if (z['cv2'] == z['cv2'] and z['cv2_null'] == z['cv2_null']) else float('nan')
        trg = 1 - math.sqrt(max(z['cv2'], 0.0)) if z['cv2'] == z['cv2'] else float('nan')
        L.append('| %.2f | %.0f%s | %d | %s | %s | %s | %s |' % (z['sigs'], z['tau_s_h'], ' (контроль)' if abs(z['tau_s_h'] - TAU_M * 24) < 0.6 else '',
                 z['n_sets_cv'], ('%.3f' % z['cv2']) if z['cv2'] == z['cv2'] else '—', ('%.3f' % z['cv2_null']) if z['cv2_null'] == z['cv2_null'] else '—',
                 ('%+.3f' % -dlt) if dlt == dlt else '—', ('%.2f' % trg) if trg == trg else '—'))
    zb = [z for z in r14 if abs(z['sigs'] - SIGS_BASE) < 1e-12 and abs(z['tau_s_h'] - 63.) < 0.6]
    zb = zb[0] if zb else None
    # Контроли (б) и (в) идут через ТУ ЖЕ функцию cv2_sets, что и настоящее измерение:
    # иначе проверялась бы не она, а её копия, и ошибка в ней осталась бы незамеченной.
    def fake_gaps(ng, nper, gmean=4.0, trfrac=0.0):
        tr = trfrac * gmean; sc = gmean - tr
        sm = np.zeros(ng); sq = np.zeros(ng)
        for i in range(ng):
            g = tr + rng15.exponential(sc, nper); sm[i] = g.sum(); sq[i] = float((g * g).sum())
        return dict(gap_n=np.full(ng, float(nper)), gap_sum=sm, gap_sq=sq)
    NQ, NPER, NR15 = 60, 8, 200; mk1 = np.ones(NQ, bool)
    # Сравниваем МЕДИАНУ ПО ПОВТОРАМ с медианой нулевого распределения, а не одну реализацию с медианой:
    # у одной реализации при 60 наборах по 8 промежутков разброс 5-95 % равен 0,68-0,90, то есть ±0,11,
    # и допуск 0,03 для неё был бы заведомо невыполним. Это та же ошибка «порог без шума величины».
    _, nul_b, _ = cv2_sets(fake_gaps(NQ, NPER, 4.0, 0.0), mk1, nrep=400)
    ctl_b = float(np.median([cv2_sets(fake_gaps(NQ, NPER, 4.0, 0.0), mk1, nrep=1)[0] for _ in range(NR15)]))
    ctl_c = float(np.median([cv2_sets(fake_gaps(NQ, NPER, 4.0, 0.25), mk1, nrep=1)[0] for _ in range(NR15)]))
    v15a = bool(zb is not None and zb['cv2'] == zb['cv2'] and (zb['cv2_null'] - zb['cv2']) >= 0.10)
    v15b = bool(ctl_b == ctl_b and nul_b == nul_b and abs(ctl_b - nul_b) <= 0.03)
    v15c = bool(ctl_c == ctl_c and ctl_c <= 0.50)
    chk15 = bool(zb is not None and zb['cv2'] == zb['cv2'])
    L += ['\n| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|',
          '| а: рефрактерность есть | %s | ниже нуля не меньше чем на 0,10 | %s |' % (
              ('CV² = %.3f против нуля %.3f, разность %+.3f' % (zb['cv2'], zb['cv2_null'], zb['cv2'] - zb['cv2_null'])) if chk15
              else 'наборов с %d и больше промежутками не нашлось' % NGMIN, 'выполнено' if v15a else ('не выполнено' if chk15 else 'непроверяемо')),
          '| б: статистика без ошибки | та же функция на экспоненциальных промежутках даёт %.3f (медиана по %d повторам) против своего нуля %.3f | расходятся не больше чем на 0,03 | %s |' % (
              ctl_b, NR15, nul_b, 'выполнено' if v15b else 'НЕ ВЫПОЛНЕНО'),
          '| в: статистика обладает мощностью | с вшитым t_r = 0,25·g даёт %.3f (медиана по %d повторам) | не выше 0,50 (расчёт обещал около 0,42) | %s |' % (
              ctl_c, NR15, 'выполнено' if v15c else 'НЕ ВЫПОЛНЕНО'),
          '\nЧасти (б) и (в) идут через ту же функцию, что и настоящее измерение, а не через её копию: иначе проверялась бы копия, а ошибка в рабочем пути осталась бы незамеченной. Если не выполнено (б) или (в) — ошибка в самом измерении, и строка (а) ничего не значит. Если (б) и (в) выполнены, а (а) нет — у защиты нет глухого периода, и это уже вывод о модели, а не об измерении: тогда рефрактерность надо искать в примировании индукции, а не в запасе белка.']
    R['F15'] = dict(rows=[dict(sigs=z['sigs'], tau_s_h=z['tau_s_h'], cv2=z['cv2'], cv2_null=z['cv2_null'], n_sets=z['n_sets_cv']) for z in r14],
                    a=dict(verdict=v15a), b=dict(ctl_exp=ctl_b, verdict=v15b), c=dict(ctl_refr=ctl_c, verdict=v15c), n_gap_min=NGMIN,
                    theory=dict(R_null='1 + CV^2(rates)', cv2_null='1 exactly', readout='t_r/g = 1 - CV', cv_rates_implied=0.63))

    # ---------- 7е. Правило Ф16: закон длительности, проверка вне подгонки ----------
    stage('Ф16: закон длительности, подгонка по четырём точкам и проверка в пятой')
    L += ['\n## 7е. Правило Ф16: закон длительности эпизода, проверенный вне подгонки\n',
          'Форма выведена, а не подобрана. Внутри эпизода нагрузка стоит выше верха окна на Delta (в логарифме), а защита поднимает верх окна как delta(t) = delta_макс·(1 − exp(−t/tau_б)). Эпизод кончается, когда верх окна догонит нагрузку, либо когда нагрузка сама упадёт:\n',
          '    длительность = min( tau_б · ln(1/(1 − Delta/delta_макс)),  время экскурсии нагрузки ).\n',
          'То есть линейно по tau_б, пока защита успевает, и полка на длительности собственной экскурсии нагрузки, когда уже не успевает. Полка обязана совпасть с тем, что мерило правило Ф10 (длительность = время нагрузки выше верха окна, корреляция 0,94) — так два правила сшиваются в одно утверждение.\n',
          '**Как проверяется.** Две постоянные подгоняются по ЧЕТЫРЁМ точкам (память 12, 24, 63, 88 ч), а проверяется ПРЕДСКАЗАНИЕ в пятой (150 ч), которая в подгонку не входит. Три формы-кандидата расходятся там на 1,2 суток, то есть на девять стандартных ошибок медианы (SE = 3,4 %, восстановлена из перебора по зёрнам в разделе 3), поэтому точка различающая. Точку 36 ч я в прогон НЕ взял: там формы расходятся всего на 0,4 суток, и блок был бы потрачен зря.\n',
          '\n### Правило Ф16, записано до счёта\n',
          '(а) Длительность при памяти 150 ч совпадает с предсказанием лучшей формы в пределах 2 SE, то есть 6,7 %.\n(б) Жёсткая форма min(k·tau, T) отвергается: её предсказание отличается от наблюдённого больше чем на 2 SE. Эта форма означала бы, что Delta/delta_макс у всех наборов одна и та же.\n(в) Полка формы совпадает с медианным временем экскурсии нагрузки из Ф10 в пределах 20 %.\n']
    from scipy.optimize import curve_fit as _cf
    FORMS = (('min(k*t, T): одна Delta на всех', lambda t, k, T: np.minimum(k * t, T)),
             ('T*t/(t+t0): Delta разбросана', lambda t, T, t0: T * t / (t + t0)),
             ('T*(1-exp(-t/t0)): плавное насыщение', lambda t, T, t0: T * (1 - np.exp(-t / t0))))
    f16 = []
    for sg in SG14:
        rws = sorted([z for z in r14 if abs(z['sigs'] - sg) < 1e-12], key=lambda z: z['tau_s_h'])
        tt = np.array([z['tau_s_h'] / 24. for z in rws]); dd = np.array([z['dur_hi'] for z in rws])
        fitm = tt < 100. / 24.                                                            # подгонка только по точкам до 100 ч
        if np.sum(np.isfinite(dd[fitm])) < 4: continue
        hold = ~fitm
        for nm, fn in FORMS:
            try:
                pp, _ = _cf(fn, tt[fitm], dd[fitm], p0=[3., 2.], maxfev=40000)
                res = dd[fitm] - fn(tt[fitm], *pp); rms = float(np.sqrt((res * res).mean()))
                pr = float(fn(tt[hold][0], *pp)) if hold.any() else float('nan')
                ob = float(dd[hold][0]) if hold.any() else float('nan')
                f16.append(dict(sigs=sg, form=nm, rms_fit=rms, pred_150=pr, obs_150=ob,
                                err_rel=abs(pr - ob) / max(ob, 1e-9) if ob == ob else float('nan'),
                                pars=[float(x) for x in pp]))
            except Exception:
                f16.append(dict(sigs=sg, form=nm, rms_fit=float('nan'), pred_150=float('nan'), obs_150=float('nan'), err_rel=float('nan'), pars=[]))
    L += ['| sigma_м | форма | RMS подгонки по 4 точкам, сут | предсказание при 150 ч | наблюдено | промах |\n|---|---|---|---|---|---|']
    for z in f16:
        L.append('| %.2f | %s | %s | %s | %s | %s |' % (z['sigs'], z['form'],
                 ('%.3f' % z['rms_fit']) if z['rms_fit'] == z['rms_fit'] else '—',
                 ('%.2f' % z['pred_150']) if z['pred_150'] == z['pred_150'] else '—',
                 ('%.2f' % z['obs_150']) if z['obs_150'] == z['obs_150'] else '—',
                 ('%.1f %%' % (100 * z['err_rel'])) if z['err_rel'] == z['err_rel'] else '—'))
    gd16 = [z for z in f16 if z['err_rel'] == z['err_rel']]
    best16 = min(gd16, key=lambda z: z['rms_fit']) if gd16 else None
    hard16 = [z for z in gd16 if z['form'].startswith('min(')]
    v16a = bool(best16 is not None and best16['err_rel'] <= 0.067)
    v16b = bool(hard16 and all(z['err_rel'] > 0.067 for z in hard16))
    L += ['\n| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|']
    if not best16: L.append('| а и б | подгонка не вышла: в части строк длительность не определена | — | непроверяемо |')
    if best16:
        L += ['| а: предсказание лучшей формы подтверждается | лучшая «%s», промах %.1f %% | не больше 6,7 %% (2 SE) | %s |' % (
                  best16['form'], 100 * best16['err_rel'], 'выполнено' if v16a else 'не выполнено'),
              '| б: жёсткая форма отвергается | промах %s | больше 6,7 %% | %s |' % (
                  ', '.join('%.1f %%' % (100 * z['err_rel']) for z in hard16) if hard16 else '—', 'выполнено' if v16b else 'не выполнено'),
              '\nЕсли подтвердится лучшая форма, то закон длительности у нас есть в замкнутом виде, и из него уже СЛЕДУЕТ, что пассивная защита цели Ф11(б) не достигает ни при одном значении внутри измеренного интервала 24–88 ч: закон даёт 2,55 суток при 24 ч и 2,98 при 24 ч с медленным шумом, против требуемых 2. Это и есть количественный довод за примирование индукции, полученный без всякого прогона — прогон его только подтверждает или ломает.']
    R['F16'] = dict(rows=f16, a=dict(verdict=v16a), b=dict(verdict=v16b), se_median=0.034,
                    note='подгонка по памяти 12/24/63/88 ч, проверка в 150 ч; точка 36 ч не бралась — формы там не различаются')

    # ---------- 7ж. Правило Ф19: примирование индукции в полной модели ----------
    stage('Ф19: примирование индукции, парные критерии уровня 0,05')
    from scipy.stats import binom as _bn
    TPR = [12. / 24, 24. / 24]
    b19 = runb(num, 1 + len(TPR), 'примирование', kd=0.0, aH=A_H, sigv=SIGV, E0=E0, qw=qw, Gbar=Gbar,
               sigma=SIG, tauc_rel=TAUC, ndays=ND, M=M, J=J, seed=SEED, warm=max(WARM, WARMS), sigs=SIGS_BASE,
               apr=np.array([0.0] + [A_PR] * len(TPR)), taup=np.array([1.0] + TPR))
    def durset(rr): return np.where(rr['epn'] > 0, rr['epsum'] / np.maximum(rr['epn'], 1), np.nan)
    def shrset(rr): return np.where(rr['epn'] > 0, rr['epshort'] / np.maximum(rr['epn'], 1), np.nan)
    def sign_test(a_, b_, mask, alpha=0.05):
        """Парный знаковый критерий. Под нулевой гипотезой число наборов, где стало короче,
        k ~ Bin(n, 1/2): E[k] = n/2, Var[k] = n/4. Критическое значение — ФОРМУЛА, не число,
        потому что n известно только после прогона. Распределение длительностей не используется."""
        ok = mask & np.isfinite(a_) & np.isfinite(b_) & (a_ != b_)
        n = int(ok.sum())
        if n < 10: return dict(n=n, k=0, crit=float('nan'), p=float('nan'), verdict='непроверяемо')
        k = int((b_[ok] < a_[ok]).sum()); mu = n / 2.; sd = math.sqrt(n / 4.)
        return dict(n=n, k=k, mu=mu, sd=sd, crit=float(mu + 1.645 * sd), z=float((k - mu) / sd),
                    p=float(_bn.sf(k - 1, n, 0.5)), verdict=bool(_bn.sf(k - 1, n, 0.5) <= alpha))
    def med_upper(v, alpha=0.05):
        """Односторонняя верхняя граница медианы по порядковой статистике: наименьшее r, при котором
        P(Bin(n,1/2) <= r-1) >= 1-alpha; граница — r-я по возрастанию. Распределение не нужно.
        Покрытие проверено розыгрышем на экспоненциальном, логнормальном и смеси: промах 0,038-0,052."""
        x = np.sort(np.asarray(v, float)); x = x[np.isfinite(x)]; n = len(x)
        if n < 10: return float('nan'), -1, n
        for r in range(1, n + 1):
            if _bn.cdf(r - 1, n, 0.5) >= 1 - alpha: return float(x[r - 1]), r, n
        return float(x[-1]), n, n
    hi19 = band & (bf >= 2.0)                                                             # измеренная сила индукции, полоса
    d_ctl = durset(b19[0]); s_ctl = shrset(b19[0])
    r19 = []
    for tpv, rr in zip(TPR, b19[1:]):
        d_on = durset(rr); s_on = shrset(rr)
        st_d = sign_test(d_ctl, d_on, hi19); st_s = sign_test(s_ctl, s_on, hi19)
        ub, rr_, nn_ = med_upper(d_on[hi19])
        ok_ = hi19 & (rr['epn'] > 0)
        r19.append(dict(tau_p_h=tpv * 24, dur=float(np.nanmedian(d_on[hi19])), dur_ub=ub, ub_rank=rr_, ub_n=nn_,
                        short_pool=float(rr['epshort'][ok_].sum() / max(rr['epn'][ok_].sum(), 1)),
                        pr_dam=float(np.median(rr['prdam'][ok_])) if ok_.any() else float('nan'),
                        sign_dur=st_d, sign_short=st_s, n_ep=int(rr['epn'][ok_].sum())))
    ok0 = hi19 & (b19[0]['epn'] > 0)
    ub0, r0_, n0_ = med_upper(d_ctl[hi19])
    base19 = dict(dur=float(np.nanmedian(d_ctl[hi19])), dur_ub=ub0,
                  short_pool=float(b19[0]['epshort'][ok0].sum() / max(b19[0]['epn'][ok0].sum(), 1)),
                  n_ep=int(b19[0]['epn'][ok0].sum()), n_sets=int(ok0.sum()))
    # (г) калибровка критерия: случайно переставляем метки «опыт/контроль» по наборам
    rngp = np.random.default_rng(321); dd = (r19[0]['sign_dur'], )
    a_, b_ = d_ctl, durset(b19[1]); okp = hi19 & np.isfinite(a_) & np.isfinite(b_) & (a_ != b_)
    dif = (b_[okp] - a_[okp]); fires = 0
    for _ in range(2000):
        sg = rngp.choice([-1.0, 1.0], len(dif)); kk = int((dif * sg < 0).sum()); nn = len(dif)
        fires += (_bn.sf(kk - 1, nn, 0.5) <= 0.05)
    cal19 = fires / 2000.
    fin19 = [z for z in r19 if z['dur'] == z['dur']]
    best19 = min(fin19, key=lambda z: z['dur']) if fin19 else r19[0]
    chk19 = bool(fin19 and best19['sign_dur']['n'] >= 10 and len(dif) >= 10)              # меньше десяти наборов — проверять нечего
    v19a = (bool(best19['sign_dur']['verdict'] is True) if chk19 else 'непроверяемо')
    v19b = (bool(best19['dur_ub'] == best19['dur_ub'] and best19['dur_ub'] <= 2.0) if chk19 else 'непроверяемо')
    v19c = (bool(best19['sign_short']['verdict'] is True) if chk19 else 'непроверяемо')
    v19g = (bool(0.02 <= cal19 <= 0.09) if chk19 else 'непроверяемо')
    vt = lambda v: 'выполнено' if v is True else ('не выполнено' if v is False else v)
    L += ['\n## 7ж. Правило Ф19: примирование индукции в полной модели\n',
          '**Что изменено.** Усиление индукции теперь зависит от истории: beta умножается на (1 + %.1f·p), где p — след недавнего окисления, p\' = [u/(K_B+u) − p]/tau_p. Движущая сила взята ТА ЖЕ, что у индукции, потому что примирование в источнике Nrf2-зависимо, то есть стоит на том же сигнале; это экономит параметр. Потолок усиления ВТРОЕ — измерен (Bischoff и соавт. 2019: максимум отклика сульфиредоксина на второе воздействие до трёх раз выше). Постоянная памяти tau_p ничем не закреплена (в опыте проверен один интервал, 24 ч), поэтому перебирается: 12 и 24 ч.\n' % A_PR,
          '**Ускорение первой ступени НЕ добавлено намеренно.** На одной клетке (prime_cell.py) оно дало от −1 до −5 %% против −52 %% у усиления, и причина записана до счёта: первая ступень с постоянной %.0f ч к началу эпизода уже уравновешена, а скорость всей цепочки задаёт вторая. Добавлять второй механизм в тот же прогон значило бы смешать два изменения.\n' % (TAU_M * 24),
          'Парный контроль — блок с выключенным примированием, то же зерно, тот же шум, те же наборы. Связь со сном выключена (k = 0), как и в Ф11, чтобы мерить чистое действие защиты.\n',
          '\n### Правило Ф19, записано до счёта\n',
          'Все проверки — статистические, уровня 0,05, и критические значения заданы ФОРМУЛОЙ, а не числом, потому что n известно только после прогона.\n',
          '(а) Примирование укорачивает эпизод. Парный знаковый критерий по наборам: k — число наборов, где с примированием короче, под нулевой гипотезой k ~ Bin(n, 1/2), то есть E[k] = n/2 и Var[k] = n/4. Вердикт по точному биномиальному p, порог 0,05. Распределение длительностей не используется вовсе.\n',
          '(б) Длительность доходит до клинической цели: односторонняя верхняя 95-процентная граница медианы по порядковой статистике не выше 2,0 суток. Граница распределения не предполагает; её покрытие проверено розыгрышем на трёх разных распределениях — промах 0,038–0,052 при номинальных 0,05.\n',
          '(в) Короткие эпизоды появляются: тот же знаковый критерий на доле эпизодов короче суток внутри набора, уровень 0,05.\n',
          '(г) КАЛИБРОВКА КРИТЕРИЯ: если метки «с примированием / без» переставить по наборам случайно, критерий (а) должен срабатывать примерно в 5 %% случаев. Допуск 0,02–0,09. Если нет — критерий ловит не примирование, и части (а)–(в) недействительны.\n',
          '\n| | длительность, сут | верхняя граница медианы (95 %%) | коротких по всем эпизодам | эпизодов | примирование p во время эпизода |\n|---|---|---|---|---|---|',
          '| контроль: примирования нет | %.2f | %.2f | %.3f | %d | — |' % (base19['dur'], base19['dur_ub'], base19['short_pool'], base19['n_ep'])]
    for z in r19:
        L.append('| примирование, tau_p = %.0f ч | %.2f | %.2f | %.3f | %d | %.3f |' % (
            z['tau_p_h'], z['dur'], z['dur_ub'], z['short_pool'], z['n_ep'], z['pr_dam']))
    L += ['\n| часть | получено | ожидание до счёта | вывод |\n|---|---|---|---|',
          '| а: укорачивает (знаковый критерий) | k = %d из n = %d; под нулём E = %.1f, SD = %.2f; критическое %.1f; p = %.1e | p не больше 0,05 | %s |' % (
              best19['sign_dur']['k'], best19['sign_dur']['n'], best19['sign_dur'].get('mu', float('nan')),
              best19['sign_dur'].get('sd', float('nan')), best19['sign_dur'].get('crit', float('nan')),
              best19['sign_dur']['p'], vt(v19a)),
          '| б: доходит до 2,0 суток | медиана %.2f, верхняя граница %.2f сут (ранг %d из %d) | граница не выше 2,0 | %s |' % (
              best19['dur'], best19['dur_ub'], best19['ub_rank'], best19['ub_n'], vt(v19b)),
          '| в: короткие эпизоды появляются | k = %d из n = %d; p = %.1e; доля по всем эпизодам %.3f против %.3f в контроле | p не больше 0,05 | %s |' % (
              best19['sign_short']['k'], best19['sign_short']['n'], best19['sign_short']['p'],
              best19['short_pool'], base19['short_pool'], vt(v19c)),
          '| г: калибровка критерия | при случайной перестановке меток срабатывает в %.3f случаев | от 0,02 до 0,09 | %s |' % (
              cal19, vt(v19g) if v19g is not False else 'НЕ ВЫПОЛНЕНО'),
          '\nЧасть (г) читается первой: если критерий срабатывает на перемешанных метках, строки (а)–(в) ничего не значат. База для сравнения — %.2f суток при доле коротких %.3f, по %d наборам полосы с кратностью индукции не ниже 2 (прогон 14.10, Ф11).' % (
              base19['dur'], base19['short_pool'], base19['n_sets'])]
    R['F19'] = dict(base=base19, rows=r19, calib_false_alarm=cal19, A_PR=A_PR, tau_p_scan_h=[t * 24 for t in TPR],
                    a=dict(verdict=v19a), b=dict(thresh_days=2.0, verdict=v19b), c=dict(verdict=v19c), g=dict(verdict=v19g),
                    method=dict(sign_test='Bin(n,1/2), E=n/2, Var=n/4, точный биномиальный p',
                                median_bound='порядковая статистика, покрытие проверено розыгрышем 0,038-0,052'))

    # ---------- 8. Связь с параметрами ----------
    from scipy.stats import rankdata
    L += ['\n## 8. С чем связано застревание: ранговая корреляция с наибольшим бодрствованием\n', '| величина | корреляция |\n|---|---|']
    cors = {}
    for qn in NAMES + ['r']:
        cv = float(np.corrcoef(rankdata(P[qn]), rankdata(rb_['wkmax']))[0, 1]); cors[qn] = cv; L.append('| %s | %+.2f |' % (qn, cv))
    for nm, vvv in (('запас до верхней границы, log', np.log(hi / num)), ('доля времени в повреждении', rb_['tdam']), ('средняя доля повреждённых D', rb_['Dbar'])):
        cv = float(np.corrcoef(rankdata(vvv), rankdata(rb_['wkmax']))[0, 1]); cors[nm] = cv; L.append('| %s | %+.2f |' % (nm, cv))
    R['corr'] = cors

    # ---------- 9. Для машинного чтения ----------
    R['setup'] = dict(mode='smoke' if SMOKE else ('full' if FULL else 'default'), n_sets=nsel, n_sampled=K, n_cells=J, ndays=ND, steps_per_day=M, seed=SEED, n_rep=NREP,
                      sigma=SIG, tauc_rel=TAUC, q_pos=QPOS, k_scan=KSCAN, k_base=KBASE, q_scan=QSCAN, a_scan=ASCAN, v_scan=VSCAN, sigma_v=SIGV,
                      D_thresh=DTHR, g_dep=GDEP, swing_target=DSWING, tau_m_h=TAU_M * 24, tau_s_h=TAU_S * 24, bfmax=BFMAX, sr_amp=SR_AMP, sr_rate=SR_RATE, sr_tau_h=TAU_SR, sr_ceiling=SR_CEIL, r_scan=RSCAN, a_scan_sr=ASCAN_SR, sleep_noise=SLPN, tau_sleep_noise_h=TAUSL, use_mean_damage=USE_DM, s_scan=SSCAN, sigs_base=SIGS_BASE, tau_slow_days=TSLOW, warm_slow=WARMS, wc=dict(c1=C1, c2=C2, c3=C3, c4=C4, a_e=AE_S, th_e=THE, a_i=AI_S, th_i=THI, Pe0=PE0, Pe1=PE1),
                      pr=dict(Qmax=QMAX, theta=THETA, sigma=SGM, tau_v_s=10., tau_m_s=10., nu_vm=NVM, nu_mv=NMV, A_m=A_M, nu_vc=NVC, B0=B0, AG=AG, tau_rise_h=TR, tau_decay_h=TD),
                      T_range=list(RANGES['T'][:2]))
    R['verdicts'] = dict(F2a=v2a, F2b=v2b, F2c=v2c, F3a=va, F3b=vb, F3c=vc, F4a=('непроверяемо' if not enough else v4a), F4b=v4b, F5=v5,
                         F6a=v6a, F6b=v6b, F6c=v6c, F6g=v6g, F7a=v7a, F7b=v7b, F8a=v8a, F8b=v8b, F8c=v8c, F8d=v8d,
                         F11a=v11a, F11b=v11b, F11d=v11d, F12a=v12a, F12b=v12b, F12c=v12c, F13a=v13a, F13b=v13b, F13c=v13c, F14a=v14a, F14b=v14b, F14c=v14c, F15a=v15a, F15b=v15b, F15c=v15c, F16a=v16a, F16b=v16b, F18a=v18a, F18b=v18b, F19a=v19a, F19b=v19b, F19c=v19c, F19g=v19g, batch_ok=v_batch)
    R['time_s'] = round(time.time() - T0, 1)
    def clean(o):
        if isinstance(o, dict): return {k2: clean(v2) for k2, v2 in o.items() if k2 != 'dur_arr'}
        if isinstance(o, (list, tuple)): return [clean(v2) for v2 in o]
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, float) and (o != o or abs(o) == float('inf')): return None
        return o
    R = clean(R)
    L += ['\n## 9. Для машинного чтения\n', '```json', json.dumps(R, ensure_ascii=False, indent=1), '```']
    L.append('\nВремя счёта: %.0f с.' % (time.time() - T0)); text = '\n'.join(L) + '\n'
    open(os.path.join(OUT, 'report.md'), 'w', encoding='utf-8').write(text); json.dump(R, open(os.path.join(OUT, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print('\n' + text)

    # ---------- Рисунки ----------
    stage('рисунки'); import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'font.size': 9})
    sv_ = lambda f, n: (f.savefig(os.path.join(OUT, n), dpi=110), plt.close(f))
    cen = (np.arange(NDB) + 0.5) / NDB
    fig, a = plt.subplots(figsize=(8.5, 4.8)); ok = dnn > 0
    a.plot(cen[ok], dcur[ok], 'o-', color=COL[0], lw=2, label='все бодрствования в общей куче'); a.axvline(0.5, color=COL[1], lw=2, label='порог эпизода')
    okr = dcnt_rel > 0; a2_ = a.twinx(); a2_.plot(cen[okr], dcur_rel[okr], 's--', color=COL[2], lw=1.8); a2_.axhline(1.0, color=COL[2], ls=':', lw=1)
    a2_.set_ylabel('в долях своего здорового уровня', color=COL[2]); a2_.grid(False)
    a.axvspan(0, 0.5, color='#e4e4e1', lw=0, zorder=0, label='ниже порога'); a.axhline(base_bout, color=INK, ls='--', lw=1, label='здоровый уровень')
    a.set_xlabel('наибольшая доля повреждённых D за это бодрствование'); a.set_ylabel('длина бодрствования, сутки')
    a.set_title('Ф4(а): удлиняется ли бодрствование ещё до порога', fontsize=10, loc='left'); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); sv_(fig, 'fig1_bout_vs_damage.png')

    fig, a = plt.subplots(figsize=(8.5, 4.8)); okl = ln > 0; xs = np.arange(NLB) + 0.5
    a.bar(xs[okl], lcur[okl], width=0.85, color=COL[0], edgecolor='#fcfcfb', label='все бодрствования в общей куче'); a.axhline(base_bout, color=COL[1], lw=2, label='здоровый уровень')
    okr2 = lcnt_rel > 0; a3_ = a.twinx(); a3_.plot(xs[okr2], lcur_rel[okr2], 's--', color=COL[2], lw=1.8); a3_.axhline(1.0, color=COL[2], ls=':', lw=1)
    a3_.set_ylabel('в долях своего здорового уровня', color=COL[2]); a3_.grid(False)
    a.set_xlabel('суток до первого срыва'); a.set_ylabel('длина бодрствования, сутки'); a.invert_xaxis()
    a.set_title('Ф4(б): опережение', fontsize=10, loc='left'); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); sv_(fig, 'fig2_lead.png')

    ks_ = [r_['k'] for r_ in rows]
    fig, a = plt.subplots(figsize=(8.5, 4.8))
    a.plot(ks_, [r_['dur'] for r_ in rows], 'o-', color=COL[0], lw=2, label='только закончившиеся эпизоды')
    a.plot(ks_, [r_['dur_all'] for r_ in rows], 's--', color=COL[2], lw=2, label='с учётом незакрытых')
    a.axhline(rows[0]['dur'] * 1.5, color=COL[1], ls='--', lw=1.2, label='в 1,5 раза выше уровня k = 0')
    for kk in kstair:
        if kk['per_h'] == kk['per_h'] and kk['per_h'] > 30: a.axvline(kk['k'], color=INK, ls=':', lw=1); break
    a.set_yscale('log'); a.set_xlabel('сила связи k, мВ'); a.set_ylabel('длительность эпизода, сутки')
    a.set_title('Длительность эпизода: парное сравнение (пунктир — где начинает пропускаться ночь)', fontsize=10, loc='left'); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); sv_(fig, 'fig3_duration.png')

    fig, a = plt.subplots(figsize=(8.5, 4.8))
    a.plot(ks_, [r_['stuck15'] for r_ in rows], '^-', color=COL[2], lw=2, label='бодрствование дольше 1,5 суток')
    a.plot(ks_, [r_['stuck'] for r_ in rows], 'o-', color=COL[1], lw=2, label='дольше 2 суток')
    a.plot(ks_, [r_['short'] for r_ in rows], 's-', color=COL[0], lw=2, label='эпизоды короче суток')
    a.plot(ks_, [r_['Dbar'] for r_ in rows], 'd-', color=COL[3], lw=2, label='средняя доля повреждённых D')
    a.set_xlabel('сила связи k, мВ'); a.set_ylabel('доля'); a.set_title('Застревание и возврат; потолка больше нет', fontsize=10, loc='left'); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); sv_(fig, 'fig4_stuck.png')

    j = int(np.argmax(rb_['wkmax'])); pj = {n: float(P[n][j]) for n in NAMES}; pj['r'] = float(P['r'][j]); nd_tr = min(14, ND)
    fig, ax = plt.subplots(4, 2, figsize=(13, 9.5), sharex='col', sharey='row')
    for col, ks in enumerate([0.0, KBASE]):
        tr_ = trace(pj, num[j], thr[j], xhi[j], xlo[j], ks, A_H, SIGV, E0, qw, Gbar, SIG, TAUC, nd_tr, M, J, np.random.default_rng(11))
        t_, aw_, H_, E_, L_, D_, Dm_ = [tr_[:, z] for z in range(7)]
        for row in range(4):
            sl = np.where(aw_ < 0.5)[0]
            if len(sl):
                for bb in np.split(sl, np.where(np.diff(sl) > 1)[0] + 1):
                    if len(bb): ax[row, col].axvspan(t_[bb[0]], t_[bb[-1]], color='#e4e4e1', lw=0, zorder=0)
        ax[0, col].plot(t_, H_, color=COL[0], lw=1.4); ax[0, col].set_title('k = %.1f мВ' % ks, fontsize=10, loc='left')
        ax[1, col].plot(t_, E_, color=COL[2], lw=1.4)
        ax[2, col].plot(t_, L_, color=COL[3], lw=1.4)
        ax[3, col].plot(t_, D_, color=COL[1], lw=1.6, label='доля повреждённых D'); ax[3, col].plot(t_, Dm_ / max(np.nanmax(Dm_), 1e-9), color=INK, lw=1, ls='--', label='защита (белок), в долях пика')
        ax[3, col].axhline(DTHR, color=INK, ls=':', lw=1); ax[3, col].set_xlabel('время, сутки')
    ax[0, 0].set_ylabel('сомногены G'); ax[1, 0].set_ylabel('активность E'); ax[2, 0].set_ylabel('нагрузка / средняя'); ax[3, 0].set_ylabel('повреждение')
    ax[3, 0].legend(frameon=False, fontsize=8)
    fig.suptitle('Серым — сон. Слева связи «повреждение мешает заснуть» нет, справа есть', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); sv_(fig, 'fig5_traces.png')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    Vvg = np.linspace(-25, 20, 20001); Dvg = Vvg - NVM * Q(A_M + NMV * Q(Vvg))
    ax[0].plot(Dvg, Q(A_M + NMV * Q(Vvg)), color=COL[0], lw=2)
    for f_ in fl: ax[0].axvline(f_, color=COL[1], ls='--', lw=1.2)
    ax[0].set_xlim(-2, 6); ax[0].set_xlabel('приток к сонному ядру D_v, мВ'); ax[0].set_ylabel('активность моноаминов Q(Vm), с$^{-1}$')
    ax[0].set_title('Складки переключателя: %s (опубликовано 1,45 и 2,46)' % ' и '.join('%.3f' % f_ for f_ in fl), fontsize=10, loc='left')
    ks2 = [kk['k'] for kk in kstair]; wk2 = [kk['wake_h'] if kk['wake_h'] == kk['wake_h'] else np.nan for kk in kstair]
    ax[1].plot(ks2, wk2, 'o-', color=COL[0], lw=2); ax[1].axhline(15.48, color=INK, ls='--', lw=1, label='норма 15,5 ч')
    ax[1].set_xlabel('сила связи k, мВ'); ax[1].set_ylabel('бодрствование при D = 1, ч'); ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title('Лестница пропущенных ночей (разрыв = сон не наступает)', fontsize=10, loc='left')
    fig.tight_layout(); sv_(fig, 'fig6_switch_checks.png')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4)); Dg = np.linspace(0, 1, 101); gq = 1 - GDEP * Dg
    Ewg, _, _ = wc_fix(gq, PE0 + PE1); Esg, _, _ = wc_fix(gq, PE0)
    ax[0].plot(Dg, Ewg, color=COL[0], lw=2, label='бодрствование'); ax[0].plot(Dg, Esg, color=COL[2], lw=2, label='сон')
    ax[0].set_xlabel('доля повреждённых PV-клеток D'); ax[0].set_ylabel('активность E'); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title('Отклик коры на утрату PV: это и есть обратная связь', fontsize=10, loc='left')
    Erw, _, _ = wc_fix(np.ones_like(Dg), PE0 + PE1); Ers, _, _ = wc_fix(np.ones_like(Dg), PE0)
    ax[1].plot(Dg, Ewg / Erw, color=COL[0], lw=2, label='бодрствование'); ax[1].plot(Dg, Esg / Ers, color=COL[2], lw=2, label='сон')
    ax[1].axhline(1.0, color=INK, ls='--', lw=1); ax[1].set_xlabel('доля повреждённых PV-клеток D'); ax[1].set_ylabel('нагрузка / средняя')
    ax[1].set_title('Вклад повреждения в нагрузку: E / E при том же бодрствовании без повреждения', fontsize=10, loc='left'); ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); sv_(fig, 'fig7_dose_response.png')

    fig, a = plt.subplots(figsize=(8.5, 4.8)); xs6 = np.arange(NLB) + 0.5
    for i6, ss in enumerate(SSCAN):
        cu, cn = scurves[ss]; ok6 = cn > 0
        if ok6.any(): a.plot(xs6[ok6], cu[ok6], 'o-' if abs(ss - SIGS_BASE) < 1e-9 else 's--', lw=2.2 if abs(ss - SIGS_BASE) < 1e-9 else 1.3,
                             color=COL[i6 % len(COL)] if ss > 0 else INK, label='sigma_м = %.2f%s' % (ss, ' (опорная)' if abs(ss - SIGS_BASE) < 1e-9 else (' — контроль' if ss == 0 else '')))
    a.axhline(1.0, color=INK, ls=':', lw=1); a.axhline(1.05, color=COL[1], ls='--', lw=1.2, label='порог правила Ф6')
    a.set_xlabel('суток до срыва'); a.set_ylabel('длина бодрствования в долях своего здорового уровня'); a.invert_xaxis()
    a.set_title('Ф6: предвестник сна и его зависимость от медленного шума', fontsize=10, loc='left'); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); sv_(fig, 'fig9_prodrome.png')

    fig, a = plt.subplots(figsize=(8.5, 4.8)); ce = (np.arange(NHD) + 0.5) / NHD
    a.bar(ce - 0.012, hD, width=0.024, color=COL[0], label='доля повреждённых D (шаг А)')
    a.bar(ce + 0.012, hDm, width=0.024, color=COL[3], label='среднее повреждение')
    a.axhspan(0, 0.011, color=COL[1], alpha=0.25, lw=0, label='уровень одноклеточного d в середине шкалы')
    a.axvspan(0.3, 0.7, color='#e4e4e1', lw=0, zorder=0); a.set_yscale('log')
    a.set_xlabel('значение переменной повреждения'); a.set_ylabel('доля времени')
    a.set_title('Ф5: градуальна ли переменная повреждения (серое — середина шкалы)', fontsize=10, loc='left'); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); sv_(fig, 'fig8_graded.png')
    stage('готово: папка %s' % OUT)

if __name__ == '__main__':
    main()
