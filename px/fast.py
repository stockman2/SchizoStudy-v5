# -*- coding: utf-8 -*-
"""Слитое ядро поклеточного редокс-блока (numba) и simulate поверх него.

ЗАЧЕМ. Замеры 16.10: счёт упирается не в арифметику, а в полосу памяти. Каждая элементарная
операция numpy — это отдельный проход по массиву K×J, и шаг делает таких проходов около сорока
пяти. На прогоне пользователя (16 ядер, 42 блока, 2000 наборов) это дало около 19 ГБ/с — потолок
двухканальной памяти при шестнадцати потоках, то есть добавлять процессы бессмысленно.

ЧТО ДЕЛАЕТ ЭТОТ ФАЙЛ. Весь поклеточный блок считается ОДНИМ циклом: все промежуточные величины
(nu, eps, b, s_, u, aa, bb, xinf, drv, minf, cbi) живут в регистрах и в память не попадают вовсе.
Остаётся прочитать шесть массивов состояния и параметры и записать шесть обратно: около двадцати
проходов вместо сорока пяти. Переключатель сна, кора, шум и накопители остаются как были — они
дают 15 % времени, и трогать их незачем.

ПОРЯДОК ДЕЙСТВИЙ СОХРАНЁН ДОСЛОВНО. Арифметика внутри ядра выписана ровно в том же порядке, что в
px/model.py, включая np.maximum(..., 1e-12), деление на 2*max(eps,1e-30) и выбор ветви по знаку b.
Это не вопрос вкуса: другой порядок дал бы другой последний разряд, и побитовая сверка перестала
бы что-либо значить.

ЧЕМ ЭТО ПРОВЕРЯЕТСЯ. tests/test_fast.py гоняет simulate и simulate_fast на одних строках и одном
зерне и требует: все СЧЁТНЫЕ величины совпадают точно, непрерывные — до 1e-12. Пока сверка не
пройдена, быстрый путь не включается нигде.

ГЕНЕРАТОР СЛУЧАЙНЫХ ЧИСЕЛ НЕ ТРОНУТ. Он работает с массивами длины K (это дёшево) и в том же
порядке, что раньше, поэтому воспроизводимость и парность блоков сохраняются полностью.
"""
import math, os
import numpy as np

# Один поток numba на процесс, и выставить это НАДО до импорта numba — позже он уже не слушает.
# Почему именно один: блоки считаются шестнадцатью процессами, и если каждый поднимет по
# шестнадцать потоков, получится 256 потоков на 16 ядер. Кто хочет иначе — ставит переменную сам,
# setdefault её не перебьёт.
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
# Слой потоков — workqueue, а не OpenMP. При одном потоке выбор ничего не меняет по скорости, зато
# OpenMP делает родительский процесс непригодным для fork: стоит хоть раз вызвать ядро, и
# последующий mp.Pool убивает программу с 'fork() called from a process already using GNU OpenMP'.
# Поймано сверкой вердиктов 16.10 на Linux; на Windows (spawn) этого не видно вовсе.
os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')
from .consts import *
from .core import vuln, NDB, NLB, RING, NHD
from . import model as M

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:                                    # numba не установлена — быстрый путь просто выключен
    HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco
    prange = range


@njit(cache=True, parallel=True, fastmath=False, nogil=True)
def cell_step(x, c, cb, mR, sP, prm, L, v, e0, gam, mm, mu0, KB, beta, KN, N0,
              dt, em, es, esd, ec, epr, apr2, thr, xhi, xlo, D, Dm, R, nc, a_n, endn, enup):
    """Один шаг поклеточного блока для всех строк и клеток. Меняет x, c, cb, mR, sP, prm на месте
    и записывает в D и Dm долю повреждённых и среднее повреждение по НОВОМУ x."""
    K, J = x.shape
    for i in prange(K):
        e0i = e0[i]; gami = gam[i]; mi = mm[i]; mu0i = mu0[i]; KBi = KB[i]
        bei = beta[i]; KNi = KN[i]; N0i = N0[i]; dti = dt[i]
        emi = em[i]; esi = es[i]; esdi = esd[i]; eci = ec[i]; epri = epr[i]; apri = apr2[i]
        Li = L[i]; thri = thr[i]; den = xhi[i] - xlo[i]
        ani = a_n[i]; edni = endn[i]; eupi = enup[i]
        if den < 1e-9: den = 1e-9
        nd = 0.0; sdn = 0.0
        # ПОЛОСА СПАСЁННЫХ ПРАЙМИНГОМ. Клетки упорядочены по уязвимости (v растёт с j), поэтому при
        # монотонной динамике повреждённые были бы ровным хвостом {j >= j*}. Медленная защита этот
        # порядок переворачивает: клетка уязвимее рухнувшей соседки может остаться целой, если
        # раньше напраймировалась. Считаем, сколько ЦЕЛЫХ клеток лежит ВЫШЕ первой повреждённой —
        # это и есть ширина полосы. Замер 16.10 на мелкой сетке: около 8,5 % популяции.
        first = -1; nres = 0.0
        for j in range(J):
            nu = Li * v[i, j]
            if nu < 1e-12: nu = 1e-12
            xi = x[i, j]; ci = c[i, j]
            eps = e0i * ci ** gami
            b = eps + xi - nu
            s_ = math.sqrt(b * b + 4.0 * eps * nu)
            if b > 0.0:
                u = 2.0 * nu / (b + s_)
            else:
                ee = eps if eps > 1e-30 else 1e-30
                u = (s_ - b) / (2.0 * ee)
            aa = ci ** mi
            bb = aa + mu0i * u * u / (1.0 + u)
            xinf = aa / bb
            xn = xinf + (xi - xinf) * math.exp(-bb * dti)
            drv = u / (KBi + u)
            pn = drv + (prm[i, j] - drv) * epri
            minf = bei * (1.0 + apri * pn) * drv
            mn = minf + (mR[i, j] - minf) * emi
            # Асимметричная защита: подъём и спад со своими постоянными. Сравнение то же, что в
            # numpy (np.where(mR > sP, es, esd)): mn — уже НОВОЕ mR, sP[i, j] — ещё СТАРОЕ sP.
            # Поле при этом непрерывно: в точке mn = sP множитель умножается на нуль, поэтому
            # перескок ветки от последнего разряда безвреден.
            sn = mn + (sP[i, j] - mn) * (esi if mn > sP[i, j] else esdi)
            # Ёмкость защиты КЛЕТКИ: пока она повреждена, тянется к N0*(1 - a_n), иначе обратно к
            # N0, и с разными постоянными — теряет быстро, набирает медленно. При a_n = 0 цель
            # всегда N0, nc остаётся на N0 точно, и выражение тождественно прежнему.
            ntgt = N0i * (1.0 - ani) if xn < thri else N0i
            ncn = ntgt + (nc[i, j] - ntgt) * (edni if ntgt < nc[i, j] else eupi)
            nc[i, j] = ncn
            cbi_ = ncn / (1.0 + u / KNi)
            cbn = cbi_ + (cb[i, j] - cbi_) * eci
            cn = cbn + sn
            if cn < 1e-9: cn = 1e-9
            x[i, j] = xn; prm[i, j] = pn; mR[i, j] = mn; sP[i, j] = sn
            cb[i, j] = cbn; c[i, j] = cn
            if xn < thri:
                nd += 1.0
                if first < 0: first = j
            elif first >= 0:
                nres += 1.0
            d_ = (xhi[i] - xn) / den
            if d_ < 0.0: d_ = 0.0
            elif d_ > 1.0: d_ = 1.0
            sdn += d_
        D[i] = nd / J
        Dm[i] = sdn / J
        R[i] = nres / J


def available():
    """Можно ли пользоваться быстрым путём: есть numba и ядро компилируется."""
    if not HAVE_NUMBA: return False
    try:
        z = np.ones((2, 2)); o = np.ones(2); D = np.zeros(2); Dm = np.zeros(2)
        cell_step(z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), o, z,
                  o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o * 2, o * 0, D, Dm, o * 0,
                  z.copy(), o * 0, o, o)
        return True
    except Exception:
        return False


def enabled():
    """Включён ли быстрый путь. REDOX_FAST=0 выключает его принудительно — этим пользуется сверка
    вердиктов, которая обязана прогнать оба пути на одной машине и одном конфиге.

    ЯДРО ЗДЕСЬ НЕ ВЫЗЫВАЕТСЯ, и это не экономия, а необходимость: эту проверку делает родительский
    процесс перед ветвлением, а первый же вызов ядра поднимает в нём пул потоков и делает fork
    смертельным. Компилируемость проверяется в отдельном процессе — см. warm_subprocess."""
    if os.environ.get('REDOX_FAST', '1') == '0': return False
    return HAVE_NUMBA


def warm_subprocess():
    """Скомпилировать ядро в ОТДЕЛЬНОМ процессе, до того как родитель разветвится.

    Зачем отдельный процесс, а не просто вызов здесь. Первый вызов ядра поднимает пул потоков
    numba, после чего fork из этого процесса запрещён (на OpenMP — мгновенная смерть, на workqueue
    — неопределённое поведение). Родитель обязан остаться чистым, поэтому компиляция уезжает в
    одноразовый интерпретатор: он компилирует, кладёт результат в дисковый кэш (cache=True) и
    умирает.

    Зачем вообще прогревать. Компиляция стоит десятки секунд. Без прогрева шестнадцать рабочих
    процессов стартуют на пустом кэше и компилируют наперегонки, дерясь за файл кэша; на Windows,
    где процессы запускаются через spawn, это умножается на шестнадцать.

    Возвращает (успех, секунд). При неуспехе вызывающий обязан выставить REDOX_FAST=0, иначе дети
    пойдут на ядро, которое не компилируется."""
    import subprocess, sys, time
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t = time.time()
    code = ('import sys; sys.path.insert(0, %r)\n'
            'from px import fast\n'
            'sys.exit(0 if fast.available() else 1)\n' % root)
    try:
        r = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, timeout=600)
        return r.returncode == 0, time.time() - t
    except Exception:
        return False, time.time() - t


# ====================================================================================
# simulate_fast: ТОТ ЖЕ код, что в px/model.py, с заменой одного блока на ядро
# ====================================================================================
# Он не переписан руками, а СОБРАН ИЗ ИСХОДНИКА model.py подстановкой текста. Так сделано
# намеренно: при ручной копии через неделю разойдётся какая-нибудь мелочь, и сверка перестанет
# что-либо доказывать. Каждая подстановка проверяется на единственность — если model.py изменится
# так, что кусок не найден или найден дважды, импорт падает с понятным сообщением, а не молча
# собирает не то.
import inspect, re

_SRC = inspect.getsource(M.simulate)

_CELL_OLD = """        nu = np.maximum(L[:, None] * v, 1e-12)
        eps = P['e0'][:, None] * c ** P['gam'][:, None]; b = eps + x - nu; s_ = np.sqrt(b * b + 4 * eps * nu)
        u = np.where(b > 0, 2 * nu / (b + s_), (s_ - b) / (2 * np.maximum(eps, 1e-30)))
        aa = c ** P['m'][:, None]; bb = aa + P['mu0'][:, None] * u * u / (1 + u); xinf = aa / bb; x = xinf + (x - xinf) * np.exp(-bb * dt[:, None])"""

_CELL_NEW = """        _cell(x, c, cb, mR, sP, prm, L, v, _p_e0, _p_gam, _p_m, _p_mu0, _p_KB, _p_beta,
              _p_KN, _p_N0, _dt1, _em1, _es1, _esd1, _ec1, _epr1, _apr1, thr, xhi, xlo, _D, _Dm,
              _R, nc, _a_n1, _endn1, _enup1)"""

_TAIL_OLD = """        drv = u / (P['KB'][:, None] + u); prm = drv + (prm - drv) * epr                     # 0 <= p <= 1, постоянная taup
        minf = P['beta'][:, None] * (1 + apr2 * prm) * drv; mR = minf + (mR - minf) * em      # усиление индукции примированием
        # Ступень 2. При симметричной защите множитель один и выражение буквально прежнее.
        # При асимметричной подъём и спад идут со своими постоянными: наработка белка и его
        # распад — разные процессы, и равенство их времён было допущением, а не выводом.
        sP = mR + (sP - mR) * (np.where(mR > sP, es, esd) if asym else es)
        ntgt = np.where(x < thr2, N0c * (1 - a_n2), N0c)
        nc = ntgt + (nc - ntgt) * np.where(ntgt < nc, e_ndn, e_nup)
        cbi = nc / (1 + u / P['KN'][:, None]); cb = cbi + (cb - cbi) * ec
        c = np.maximum(cb + sP, 1e-9)"""

_HEAD_OLD = """        dn = np.clip((xhi2 - x) / np.maximum(xhi2 - xlo2, 1e-9), 0, 1)
        D = (x < thr2).mean(1); Dm = dn.mean(1)
        # ПОЛОСА СПАСЁННЫХ ПРАЙМИНГОМ: сколько ЦЕЛЫХ клеток лежит выше первой повреждённой.
        # Клетки упорядочены по уязвимости, поэтому при монотонной динамике повреждённые были бы
        # ровным хвостом и полоса была бы пуста. Медленная защита порядок переворачивает: клетка
        # уязвимее рухнувшей соседки остаётся целой, если раньше напраймировалась. Замер 16.10 на
        # мелкой сетке дал около 8,5 % популяции.
        _bel = x < thr2
        _fi = np.argmax(_bel, 1); _any = _bel.any(1)
        Rb = np.where(_any, ((_ARJ[None, :] > _fi[:, None]) & (~_bel)).sum(1), 0) / float(J)"""
# Копии, а не ссылки: ядро перезаписывает _D и _Dm ВНУТРИ ТОГО ЖЕ ШАГА, и без копирования
# накопители получили бы долю повреждённых со следующего шага. Ошибка найдена сверкой: taw, Lbar и
# Ebar совпадали побитово, а Dbar и tdam расходились как 1/M — верный признак сдвига на шаг.
# Копия массива длины K на шаг стоит пренебрежимо мало против работы по K*J клеткам.
_HEAD_NEW = """        D = _D.copy(); Dm = _Dm.copy(); Rb = _R.copy()"""

_INIT = """    _p_e0 = np.ascontiguousarray(P['e0'], float); _p_gam = np.ascontiguousarray(P['gam'], float)
    _p_m = np.ascontiguousarray(P['m'], float); _p_mu0 = np.ascontiguousarray(P['mu0'], float)
    _p_KB = np.ascontiguousarray(P['KB'], float); _p_beta = np.ascontiguousarray(P['beta'], float)
    _p_KN = np.ascontiguousarray(P['KN'], float)
    # ЁМКОСТЬ, А НЕ ПАРАМЕТР НАБОРА. Цель релаксации nc — это N0c, то есть N0 ПОСЛЕ
    # статического снижения n0_mul, а не сырое P['N0']. Ядро брало сырое, поэтому при
    # n0_mul < 1 ёмкость сползала обратно к неуменьшенной: эталон давал ровно 0,85 N0,
    # ядро 0,88886 (найдено 21.10 на фальсификаторе Ф28). Внутри ядра N0 больше нигде
    # не нужен — в защите его заменил nc, — поэтому подмена здесь и исчерпывает правку.
    _p_N0 = np.ascontiguousarray(N0c[:, 0], float)
    _dt1 = np.ascontiguousarray(dt, float); _em1 = np.ascontiguousarray(em[:, 0], float)
    _es1 = np.ascontiguousarray(es[:, 0], float); _ec1 = np.ascontiguousarray(ec[:, 0], float)
    _esd1 = np.ascontiguousarray(esd[:, 0], float)
    _a_n1 = np.ascontiguousarray(a_n2[:, 0], float)
    _endn1 = np.ascontiguousarray(e_ndn[:, 0], float); _enup1 = np.ascontiguousarray(e_nup[:, 0], float)
    nc = np.ascontiguousarray(nc)
    _epr1 = np.ascontiguousarray(epr[:, 0], float); _apr1 = np.ascontiguousarray(apr2[:, 0], float)
    x = np.ascontiguousarray(x); c = np.ascontiguousarray(c); cb = np.ascontiguousarray(cb)
    mR = np.ascontiguousarray(mR); sP = np.ascontiguousarray(sP); prm = np.ascontiguousarray(prm)
    v = np.ascontiguousarray(v, float)
    _dn0 = np.clip((xhi2 - x) / np.maximum(xhi2 - xlo2, 1e-9), 0, 1)
    _D = np.ascontiguousarray((x < thr2).mean(1)); _Dm = np.ascontiguousarray(_dn0.mean(1))
    _bel0 = x < thr2; _fi0 = np.argmax(_bel0, 1)
    _R = np.ascontiguousarray(np.where(_bel0.any(1),
                                       ((_ARJ[None, :] > _fi0[:, None]) & (~_bel0)).sum(1), 0) / float(J))
    prev_aw = np.ones(K, bool)"""


def _build():
    s = _SRC
    for old in (_CELL_OLD, _TAIL_OLD, _HEAD_OLD):
        assert s.count(old) == 1, 'px/model.py изменился: кусок не найден или найден дважды:\n%s' % old[:90]
    s = s.replace(_CELL_OLD, _CELL_NEW).replace(_TAIL_OLD, '').replace(_HEAD_OLD, _HEAD_NEW)
    assert s.count('    prev_aw = np.ones(K, bool)') == 1
    s = s.replace('    prev_aw = np.ones(K, bool)', _INIT)
    s = s.replace('def simulate(', 'def simulate_fast(', 1)
    return s


_NS = dict(vars(M)); _NS['_cell'] = cell_step
exec(compile(_build(), '<px.fast simulate_fast>', 'exec'), _NS)
simulate_fast = _NS['simulate_fast']
