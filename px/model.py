# -*- coding: utf-8 -*-
"""Сборка слоёв и шаг по времени: simulate считает ВСЕ наборы и ВСЕ блоки перебора сразу."""
# ПЕРЕНЕСЕНО ИЗ mass_core6.py ДОСЛОВНО, автоматически, не перенабрано руками.
# Любая правка здесь обязана проходить регрессионный тест tests/test_regression.py,
# который требует ПОБИТОВОГО совпадения с эталонным датасетом в ref/.
import math
import numpy as np
from .consts import *
from .core import vuln, NDB, NLB, RING, NHD, NEPR
from .cortex import wc_fix

def simulate(P, nu0, thr, xhi, xlo, kd, aH, sigv, E0, qw, Gbar, sigma, tauc_rel, ndays, M, J, rng, warm=10, sigs=0.0, tslow=TSLOW, slpn=SLPN, usedm=USE_DM, sramp=1.0, srate=0.0, taus=TAU_S, vfix=VFIX, apr=0.0, taup=1.0, itp=1.0, tausd=None, hiw=None, a_da=0.0, tau_d=7.0, da_src=0.0, nu_mul=1.0, nu_mul2=None, t_mul=-1.0,
             a_n=0.0, taun_dn=1.0, taun_up=30.0, n0_mul=1.0, nbase=None):
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
    itp = rowv(itp)                                                                    # 1 — интерполировать момент пересечения порога, 0 — прежний счёт по шагам
    sigs = rowv(sigs); tslow = rowv(tslow); slpn = rowv(slpn); tauc_rel = rowv(tauc_rel); sigma = rowv(sigma); taus = rowv(taus); vfix = rowv(vfix); apr = rowv(apr); taup = rowv(taup)
    # Асимметричная защита: постоянная СПАДА отдельно от подъёма. tausd = None (или равное taus)
    # воспроизводит прежнюю модель В ТОЧНОСТИ — проверено побитово, поэтому прежние вердикты в силе.
    # tausd: постоянная СПАДА. None или любое неположительное значение означает «та же, что на
    # подъёме», и тогда выражение для sP буквально прежнее — проверено побитово. Неположительное
    # значение нужно потому, что спецификация блока хранит числа, а не None.
    # ДОФАМИНОВАЯ ПЕТЛЯ (версия 5.1), по умолчанию ВЫКЛЮЧЕНА: при a_da = 0 переменная считается,
    # но в нагрузку не входит, и модель воспроизводится побитово.
    # Зачем она добавлена. Единственное измеренное место, которое нынешняя модель закрыть не может:
    # рабочая область не достаёт до лестницы захвата. При k = 5 мВ средняя доля повреждённых 0,344,
    # а первая ступень стоит на k*D = 4,77 мВ, то есть нужно D около 0,95 УСТОЙЧИВО. Ни одна
    # переменная модели так не умеет: самая долгая память — 63 часа.
    # Что добавляется. Один скаляр d — тонус, которым движет СОБСТВЕННОЕ растормаживание модели:
    #       d_инф = E/Er - 1   (у здоровой клетки ровно нуль),   постоянная tau_d в сутках,
    # и обратно в нагрузку множителем (1 + a_da*d). Два параметра, и оба неизмеримы по литературе:
    # направление подтверждено (катаболизм дофамина даёт перекись), величина и время — нет.
    # Поэтому правило Ф26 спрашивает не «какое значение», а «есть ли вообще качественная разница
    # с разомкнутой петлёй» — структурный вопрос, не зависящий от выбора чисел.
    # da_src: ЧЕМ движется тонус. 0 — локальным растормаживанием (E/Er - 1); 1 — самой долей
    # повреждённых D. Второе вернее петле Грейса: она идёт от повреждения через гиппокамп и VTA,
    # мимо локальной активности коры, которой у нас всё равно нет отдельным компартментом.
    # Разница громадна по величине: растормаживание даёт около 4 % превышения, доля повреждённых —
    # десятки процентов, то есть при одном и том же усилении петля сильнее на порядок.
    # nu_mul: ПОСТОЯННЫЙ множитель нагрузки. Нужен не модели, а правилу Ф26: чтобы спросить,
    # даёт ли дофаминовая петля что-нибудь сверх простого подъёма нагрузки, нужен статический
    # эквивалент той же величины. Положение в окне (q) для этого не годится — всё окно по нагрузке
    # шириной в проценты, а петля поднимает её в разы.
    a_da = rowv(a_da); tau_d = rowv(tau_d); da_src = rowv(da_src); nu_mul = rowv(nu_mul)
    # ДВУХФАЗНАЯ НАГРУЗКА — нужна для настоящей проверки на гистерезис. Без неё «петля удерживает
    # сломанное состояние» проверить нельзя вовсе: все прогоны стартуют из здорового, и при низкой
    # нагрузке наборы просто не ломаются — это «не поймало», а не «отпустило». С t_mul >= 0
    # множитель нагрузки меняется на nu_mul2 начиная с этих суток (считая от конца разгона), то
    # есть систему сначала ломают, потом нагрузку снимают и смотрят, вернулась ли она.
    nu_mul2 = rowv(-1.0 if nu_mul2 is None else nu_mul2); t_mul = rowv(t_mul)
    # ПОТЕРЯ ЁМКОСТИ ЗАЩИТЫ — петля, действующая на ВОССТАНОВЛЕНИЕ, а не на нагрузку.
    # Зачем она. Нагрузочная петля (a_da) убила эпизоды при любой силе, и причина структурная:
    # нагрузка — величина НА НАБОР, одна на все 32 клетки, поэтому петля двигает популяцию целиком
    # и набор уезжает в D = 1. Эта петля действует ПОКЛЕТОЧНО: повреждённая клетка теряет свою
    # ёмкость, соседняя нет, популяция остаётся разнородной, и эпизодам есть где быть.
    # Устройство: пока клетка повреждена, её N0 тянется к N0*(1 - a_n); когда восстановилась — назад
    # к N0, но с ДРУГОЙ постоянной. Теряет быстро, набирает медленно — это и есть защёлка на
    # восстановлении. Асимметричная релаксация у нас уже проверена на Ф23.
    # При a_n = 0 выражение тождественно прежнему: цель всегда N0, и nc остаётся на N0 точно.
    # n0_mul: ПОСТОЯННОЕ снижение ёмкости. Нужно не модели, а правилу: чтобы спросить, даёт ли
    # петля что-нибудь сверх простого снижения ёмкости, нужен статический эквивалент той же
    # средней величины. Ровно та же роль, что у nu_mul для нагрузочной петли.
    a_n = rowv(a_n); taun_dn = rowv(taun_dn); taun_up = rowv(taun_up); n0_mul = rowv(n0_mul)
    N0c = P['N0'][:, None] * n0_mul[:, None]            # ёмкость набора после статического снижения
    nu_mul2 = np.where(nu_mul2 > 0, nu_mul2, nu_mul)
    # Накопители ПО ФАЗАМ: доля повреждённых до и после смены множителя.
    d_ph1 = np.zeros(K); t_ph1 = np.zeros(K); d_ph2 = np.zeros(K); t_ph2 = np.zeros(K)
    tausd = rowv(-1.0 if tausd is None else tausd)
    tausd = np.where(tausd > 0, tausd, taus)
    asym = bool(np.any(np.abs(tausd - taus) > 0))
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
    esd = np.exp(-dt / (tausd * day))[:, None]                  # множитель спада второй ступени
    e_ndn = np.exp(-dt / (np.maximum(taun_dn, 1e-9) * day))[:, None]   # потеря ёмкости
    e_nup = np.exp(-dt / (np.maximum(taun_up, 1e-9) * day))[:, None]   # её возврат
    a_n2 = a_n[:, None]
    epr = np.exp(-dt / (taup * day))[:, None]; prm = np.zeros((K, J)); apr2 = apr[:, None]   # примирование: след недавнего окисления на том же сигнале, что движет индукцию
    nc = np.repeat(N0c, J, 1)                   # ёмкость защиты КАЖДОЙ клетки; стартует с N0
    ncbar = np.zeros(K)                         # её среднее по времени и клеткам, для отчёта
    c = cb.copy()                                                                     # полная защита: от рецептора плюс наработанный белок
    eta = np.zeros(K); L = nu0.copy()
    dda = np.zeros(K); ed_da = np.exp(-dt / (np.maximum(tau_d, 1e-9) * day))   # тонус и его множитель
    dsum = np.zeros(K)                                                          # средний тонус за прогон
    Vv = np.full(K, -5.); Vm = np.full(K, A_M); G = np.full(K, 0.3)
    E = np.full(K, 0.3); I = np.full(K, 0.2); Er = np.full(K, 0.3); Ir = np.full(K, 0.2)
    xhi2 = xhi[:, None]; xlo2 = xlo[:, None]; thr2 = thr[:, None]; _ARJ = np.arange(J)
    rbar = np.zeros(K)                        # полоса спасённых, усреднённая по времени
    dam = np.zeros(K, bool); nb = np.zeros(K, int); nb_aw = np.zeros(K, int); nb_late = np.zeros(K, int)
    epst = np.zeros(K)       # момент начала эпизода В ШАГАХ, дробный: при itp = 1 это i-1+f, при itp = 0 ровно i
    epsum = np.zeros(K); epn = np.zeros(K, int); epshort = np.zeros(K, int)
    epmin = np.full(K, np.inf)                                                        # самый короткий эпизод у набора
    # ЗАПИСИ ОТДЕЛЬНЫХ ЭПИЗОДОВ. Раньше их писал ВТОРОЙ интегратор в px/rules/f10.py, со своей
    # самосверкой против simulate. 17.10 эта сверка отказала: интерполяция момента пересечения
    # появилась здесь и не появилась там, расхождение вышло ровно в один шаг на каждую границу, и
    # все пять вердиктов Ф10 стали непроверяемыми. Дубликат интегратора будет расходиться при
    # КАЖДОЙ правке модели, поэтому он убран, а записи пишет авторитет.
    # Ранговые связи из сумм не восстановить, поэтому хранятся сами записи, но с потолком NEPR на
    # набор; переполнение считается отдельно и печатается, чтобы молчаливой потери не было.
    # Столбцы: длительность, хвост, пик повреждения, время над верхом окна, НАЧАЛО (сутки).
    # Начало хранится, чтобы правило могло отбросить эпизоды, НАЧАВШИЕСЯ в разгоне: запись идёт
    # по концу эпизода, и эпизод, начавшийся до разгона, иначе попал бы в счёт.
    ep_rec = np.zeros((K, NEPR, 5)); ep_rn = np.zeros(K, int); ep_ov = np.zeros(K, int)
    # ПОДЁННЫЕ НАКОПИТЕЛИ ПОТОКА СОБЫТИЙ, без потолка. Записи ep_rec ограничены NEPR на набор,
    # и замер 23.10 на 400 сутках дал переполнение 15–20 %: переполняли «дребезжащие» наборы у
    # границы, у которых эпизодов десятки. Исключать их — значит отбирать по самой измеряемой
    # величине и выбрасывать целый режим поведения, а не шум. Здесь потолка нет: on_day — число
    # срывов, начавшихся в сутки k; risk_day — время ПОД РИСКОМ в эти сутки, то есть вне
    # эпизода. Поправка на время под риском обязательна: поглощение само выедает пул, и поток
    # событий падал бы со временем даже при неизменном риске. Память 2*ndays*8 байт на набор.
    nday_i = max(int(ndays), 1)
    on_day = np.zeros((K, nday_i)); risk_day = np.zeros((K, nday_i))
    Dpk = np.zeros(K); tabv = np.zeros(K); tlast_hi = np.zeros(K)
    hiw2 = None if hiw is None else rowv(hiw)
    dam_m = np.zeros(K, bool); epst_m = np.zeros(K); epsum_m = np.zeros(K); epn_m = np.zeros(K, int); epshort_m = np.zeros(K, int)  # то же по НЕПРЕРЫВНОЙ переменной: без квантования по 1/J
    ep_sleep = np.zeros(K, int); sl_in_ep = np.zeros(K, int); everdam = np.zeros(K, bool)
    wkst = np.zeros(K, int); wksum = np.zeros(K); wkn = np.zeros(K, int); wkmax = np.zeros(K); wklong = np.zeros(K, int); wk15 = np.zeros(K, int)
    taw = np.zeros(K); tdam = np.zeros(K); cycn = np.zeros(K, int); cyc1 = np.zeros(K); cyc2 = np.zeros(K)
    Dsum = np.zeros(K); Esum = np.zeros(K); Lsum = np.zeros(K); Dmsum = np.zeros(K); Gsum = np.zeros(K)
    prbar = np.zeros(K); prdam = np.zeros(K)
    bmax = np.zeros(K)
    dbin_s = np.zeros((K, NDB)); dbin_n = np.zeros((K, NDB))          # кривая «длина бодрствования против повреждения», по каждому набору
    # То же, но ячейка выбирается по D НА НАЧАЛЕ бодрствования, а не по максимуму за него.
    # Зачем. Максимум за бодрствование частично ВЫЗВАН его длиной: чем дольше тянется, тем больше
    # успевает накопиться. Метка ячейки тогда зависит от измеряемой величины — обратная
    # причинность, и длинные бодрствования систематически уезжают в ячейки с завышенной D.
    # Именно отсюда наблюдаемый прирост Ф4 ниже предсказания переключателя (1.085 против 1.114).
    # D на начале бодрствования причинно предшествует его длине и от неё не зависит вовсе.
    dbs_s = np.zeros((K, NDB)); dbs_n = np.zeros((K, NDB)); bst = np.zeros(K)
    hs_s = np.zeros(K); hs_n = np.zeros(K)                            # свой здоровый уровень при той же разметке
    lb_s = np.zeros((K, NLB)); lb_n = np.zeros((K, NLB))              # то же против срока до срыва
    hb_s = np.zeros(K); hb_n = np.zeros(K)                            # собственный здоровый уровень набора (повреждение ниже 0,1)
    hD = np.zeros((K, NHD)); hDm = np.zeros((K, NHD))                 # построчно: иначе блоки перебора смешались бы
    ring_l = np.zeros((K, RING)); ring_t = np.full((K, RING), -1.0); ptr = np.zeros(K, int); lastdam = np.full(K, -1e9); nlead = np.zeros(K, int)
    Dmin = np.ones(K); Dmax = np.zeros(K)
    NGB = 10; gaps = np.zeros((K, NGB)); gap_sum = np.zeros(K); gap_sq = np.zeros(K); gap_n = np.zeros(K); last_end = np.full(K, -1.0)
    NSB = 6; sA_s = np.zeros((K, NSB)); sA_n = np.zeros((K, NSB)); sHs = np.zeros(K); sHn = np.zeros(K)
    wake_half = 15.49 / 2.                                            # половина свободного бодрствования, часы: для Ф2(а)
    prev_aw = np.ones(K, bool); nst = int((warm + ndays) * M); w0 = int(warm * M)
    # Прошлые значения порогующих величин: нужны, чтобы линейно восстановить момент пересечения
    # ВНУТРИ шага. Без этого длительность эпизода квантуется шагом, и доля эпизодов короче суток
    # (short_pool) скачет на десятки процентов — замерено 16.10 на сходимости по M.
    Dpv = np.zeros(K); Dmpv = np.zeros(K)
    for i in range(nst):
        dn = np.clip((xhi2 - x) / np.maximum(xhi2 - xlo2, 1e-9), 0, 1)
        D = (x < thr2).mean(1); Dm = dn.mean(1)
        # ПОЛОСА СПАСЁННЫХ ПРАЙМИНГОМ: сколько ЦЕЛЫХ клеток лежит выше первой повреждённой.
        # Клетки упорядочены по уязвимости, поэтому при монотонной динамике повреждённые были бы
        # ровным хвостом и полоса была бы пуста. Медленная защита порядок переворачивает: клетка
        # уязвимее рухнувшей соседки остаётся целой, если раньше напраймировалась. Замер 16.10 на
        # мелкой сетке дал около 8,5 % популяции.
        _bel = x < thr2
        _fi = np.argmax(_bel, 1); _any = _bel.any(1)
        Rb = np.where(_any, ((_ARJ[None, :] > _fi[:, None]) & (~_bel)).sum(1), 0) / float(J)
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
        # Тонус идёт за растормаживанием со своей постоянной. Считается ВСЕГДА (это дёшево), а в
        # нагрузку входит только при a_da > 0 — тогда разомкнутая и замкнутая петли различаются
        # ровно одним множителем и сравнимы построчно.
        dinf = (1.0 - da_src) * (E / np.maximum(Er, 1e-9) - 1.0) + da_src * D
        dda = dinf + (dda - dinf) * ed_da
        tday = i / float(M) - warm                                  # сутки от конца разгона
        mul_now = np.where((t_mul >= 0) & (tday >= t_mul), nu_mul2, nu_mul)
        Linf = nu0 * mul_now * (1 + a_da * dda) * (1 + aH * (G - Gbar)) * (E / np.maximum(Er, 1e-9)) * (1 + (sramp - 1) * sr) * np.exp(eta + etas - shift); L = Linf + (L - Linf) * eL
        nu = np.maximum(L[:, None] * v, 1e-12)
        eps = P['e0'][:, None] * c ** P['gam'][:, None]; b = eps + x - nu; s_ = np.sqrt(b * b + 4 * eps * nu)
        u = np.where(b > 0, 2 * nu / (b + s_), (s_ - b) / (2 * np.maximum(eps, 1e-30)))
        aa = c ** P['m'][:, None]; bb = aa + P['mu0'][:, None] * u * u / (1 + u); xinf = aa / bb; x = xinf + (x - xinf) * np.exp(-bb * dt[:, None])
        # защита в две ступени: индукция -> сигнал (12 ч) -> белок (63 ч). Стационар тот же, что был,
        # поэтому окно, границы и весь раздел 1в не меняются: меняется только ход во времени.
        drv = u / (P['KB'][:, None] + u); prm = drv + (prm - drv) * epr                     # 0 <= p <= 1, постоянная taup
        minf = P['beta'][:, None] * (1 + apr2 * prm) * drv; mR = minf + (mR - minf) * em      # усиление индукции примированием
        # Ступень 2. При симметричной защите множитель один и выражение буквально прежнее.
        # При асимметричной подъём и спад идут со своими постоянными: наработка белка и его
        # распад — разные процессы, и равенство их времён было допущением, а не выводом.
        sP = mR + (sP - mR) * (np.where(mR > sP, es, esd) if asym else es)
        ntgt = np.where(x < thr2, N0c * (1 - a_n2), N0c)
        nc = ntgt + (nc - ntgt) * np.where(ntgt < nc, e_ndn, e_nup)
        cbi = nc / (1 + u / P['KN'][:, None]); cb = cbi + (cb - cbi) * ec
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
                js = np.clip((bst[idf] * NDB).astype(int), 0, NDB - 1)   # разметка по началу бодрствования
                dbs_s[idf, js] += d_; dbs_n[idf, js] += 1.0
                hm2 = bst[idf] < 0.1; hs_s[idf[hm2]] += d_[hm2]; hs_n[idf[hm2]] += 1.0
                sl_in_ep[idf[dam[idf]]] += 1                                    # ночь, случившаяся во время эпизода
                ring_l[idf, ptr[idf]] = d_; ring_t[idf, ptr[idf]] = i / float(M); ptr[idf] = (ptr[idf] + 1) % RING
            bmax[idf] = 0.0
        if wake_.any():
            wkst[wake_] = i; bst[wake_] = D[wake_]            # D на начале бодрствования: метка, не зависящая от его длины
            if i >= w0:
                cycn[wake_] += 1; fst = cycn[wake_] == 1
                cyc1[wake_] = np.where(fst, i / float(M), cyc1[wake_]); cyc2[wake_] = i / float(M)
        prev_aw = awake.copy()
        hiL = np.zeros(K, bool) if hiw2 is None else (L > hiw2)   # нагрузка выше верха окна набора
        nd_ = D > DTHR; down = nd_ & ~dam; up = (~nd_) & dam
        if 0.0 <= tday < nday_i:                      # накопители: срывы и время вне эпизода
            kday = int(tday)
            on_day[down, kday] += 1.0
            risk_day[~dam, kday] += 1.0 / float(M)    # шаг равен 1/M суток у всех наборов
        # Доля шага ДО пересечения: линейно по D между прошлым и текущим значением. При itp = 0
        # момент остаётся прежним (ровно i), поэтому старый счёт воспроизводится в точности.
        dn_d = np.maximum(D - Dpv, 1e-300); fdn = np.clip((DTHR - Dpv) / dn_d, 0.0, 1.0)
        dn_u = np.maximum(Dpv - D, 1e-300); fup = np.clip((Dpv - DTHR) / dn_u, 0.0, 1.0)
        tdn = i - itp * (1.0 - fdn); tup = i - itp * (1.0 - fup)      # момент пересечения в шагах
        qual = (i / float(M) - lastdam) >= 3.0
        if down.any():
            epst[down] = tdn[down]; Dpk[down] = 0.0; tabv[down] = 0.0; tlast_hi[down] = tdn[down]
        Dpk = np.where(nd_, np.maximum(Dpk, D), Dpk)
        tabv += (nd_ & hiL) / float(M)
        tlast_hi = np.where(nd_ & hiL, float(i), tlast_hi)
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
                dur_ = (tup[up] - epst[up]) / float(M); epsum[up] += dur_; epn[up] += 1; epshort[up] += (dur_ < 1.0); ep_sleep[up & ~awake] += 1
                epmin[up] = np.minimum(epmin[up], dur_)
                last_end[up] = tup[up] / float(M)
                iu = np.where(up)[0]                                  # записи эпизодов, по порядку
                tl_ = (tup[iu] - tlast_hi[iu]) / float(M)             # хвост: от последнего «над окном» до конца
                fit = ep_rn[iu] < NEPR
                if fit.any():
                    jj = iu[fit]; pos = ep_rn[jj]
                    ep_rec[jj, pos, 0] = dur_[fit]; ep_rec[jj, pos, 1] = tl_[fit]
                    ep_rec[jj, pos, 2] = Dpk[jj];   ep_rec[jj, pos, 3] = tabv[jj]
                    ep_rec[jj, pos, 4] = epst[jj] / float(M)
                    ep_rn[jj] += 1
                if (~fit).any(): ep_ov[iu[~fit]] += 1
            if down.any():                                                            # промежуток между концом прошлого эпизода и началом этого
                gd = np.where(down & (last_end >= 0))[0]
                if len(gd):
                    gg = tdn[gd] / float(M) - last_end[gd]; gap_sum[gd] += gg; gap_sq[gd] += gg * gg; gap_n[gd] += 1
                    np.add.at(gaps, (gd, np.clip(gg.astype(int), 0, NGB - 1)), 1.0)
            sm = sP.mean(1)                                                           # средний по клеткам уровень белка-защиты
            hm = D < 0.1; sHs[hm] += sm[hm]; sHn[hm] += 1                             # его здоровый уровень
            af = (~nd_) & (last_end >= 0)                                             # время, прошедшее с конца последнего эпизода
            if af.any():
                ia = np.where(af)[0]; jb = np.clip((i / float(M) - last_end[ia]).astype(int), 0, NSB - 1)
                np.add.at(sA_s, (ia, jb), sm[ia]); np.add.at(sA_n, (ia, jb), 1.0)
            Dmin = np.minimum(Dmin, D); Dmax = np.maximum(Dmax, D)
            # Доля шага, проведённая ЗА порогом. Прежде бралось значение на конце шага (0 или 1),
            # что даёт смещение порядка dt; поправка на пересечение убирает первый порядок.
            frd = nd_ + itp * (up * fup - down * fdn)
            pm_ = prm.mean(1); prbar += pm_ * dt; prdam += pm_ * frd * dt
            rbar += Rb * dt; dsum += dda * dt; ncbar += nc.mean(1) * dt
            ph2 = (t_mul >= 0) & (tday >= t_mul)
            d_ph1 += np.where(ph2, 0.0, D) * dt; t_ph1 += np.where(ph2, 0.0, dt)
            d_ph2 += np.where(ph2, D, 0.0) * dt; t_ph2 += np.where(ph2, dt, 0.0)
            taw += awake * dt; tdam += frd * dt; Dsum += D * dt; Dmsum += Dm * dt; Esum += E * dt; Lsum += L * dt; Gsum += G * dt
            hD[ar, np.clip((D * NHD).astype(int), 0, NHD - 1)] += 1.0; hDm[ar, np.clip((Dm * NHD).astype(int), 0, NHD - 1)] += 1.0
        nm_ = Dm > DTHR; dwn_m = nm_ & ~dam_m; up_m = (~nm_) & dam_m        # эпизод по среднему повреждению: порог не квантован, поэтому годится для сверки по J
        # Здесь интерполяция точна по построению: Dm непрерывна, порог не квантован.
        mn_d = np.maximum(Dm - Dmpv, 1e-300); mdn = np.clip((DTHR - Dmpv) / mn_d, 0.0, 1.0)
        mn_u = np.maximum(Dmpv - Dm, 1e-300); mup = np.clip((Dmpv - DTHR) / mn_u, 0.0, 1.0)
        if dwn_m.any(): epst_m[dwn_m] = (i - itp * (1.0 - mdn))[dwn_m]
        if i >= w0 and up_m.any():
            tm_ = (i - itp * (1.0 - mup))[up_m]
            dm_ = (tm_ - epst_m[up_m]) / float(M); epsum_m[up_m] += dm_; epn_m[up_m] += 1; epshort_m[up_m] += (dm_ < 1.0)
        dam_m = nm_
        lastdam = np.where(nd_, i / float(M), lastdam); dam = nd_
        Dpv = D; Dmpv = Dm                                                  # для интерполяции на следующем шаге
    tot = ndays * day
    prdam = prdam / np.maximum(tdam, 1e-12)                                                 # уровень примирования ИМЕННО во время эпизода
    per = np.where(cycn > 1, (cyc2 - cyc1) / np.maximum(cycn - 1, 1) * 24., np.nan)
    return dict(nb=nb, nb_aw=nb_aw, nb_late=nb_late, epn=epn, epsum=epsum, epshort=epshort, epn_m=epn_m, epsum_m=epsum_m, epshort_m=epshort_m, epmin=epmin, ep_sleep=ep_sleep, sl_in_ep=sl_in_ep,
                everdam=everdam, open_ep=dam, n_sr=n_sr, nb_trig=nb_trig, wkn=wkn, wksum=wksum, wkmax=wkmax, wklong=wklong, wk15=wk15,
                taw=taw / tot, tdam=tdam / tot, prbar=prbar / tot, prdam=prdam, per=per, nday=nb / float(ndays),
                Dbar=Dsum / tot, Dmbar=Dmsum / tot, Gbar_=Gsum / tot, Ebar=Esum / tot, Lbar=Lsum / tot,
                Dmin=Dmin, Dmax=Dmax, nlead=nlead, gaps=gaps, gap_sum=gap_sum, gap_sq=gap_sq, gap_n=gap_n,
                s_after=(sA_s, sA_n), s_health=(sHs, sHn), NGB=NGB, NSB=NSB, nbase=nbase, NB=NB, rbar=rbar / tot, dbar=dsum / tot, ncbar=ncbar / tot,
                D_ph1=d_ph1 / np.maximum(t_ph1, 1e-12), D_ph2=d_ph2 / np.maximum(t_ph2, 1e-12), ep_rec=ep_rec, ep_rn=ep_rn, ep_ov=ep_ov, on_day=on_day, risk_day=risk_day, dbin=(dbin_s, dbin_n), dbstart=(dbs_s, dbs_n), lbin=(lb_s, lb_n), hbase=(hb_s, hb_n), hstart=(hs_s, hs_n), hist=(hD, hDm))


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
    return split(_sim()(P, n0, th, xh, xl, rng=rng, nbase=nbase, **kw))


_SIM = None


def _sim():
    """Какой simulate звать: слитое ядро, если numba на месте и REDOX_FAST не выключен, иначе этот.

    Импорт ОТЛОЖЕННЫЙ и не может быть иным: px/fast.py читает исходник simulate через
    inspect.getsource, то есть импортирует этот модуль. Импорт на уровне файла дал бы цикл.
    Результат запоминается, чтобы развилка считалась один раз на процесс, а не на каждый блок."""
    global _SIM
    if _SIM is None:
        try:
            from . import fast
            _SIM = fast.simulate_fast if fast.enabled() else simulate
        except Exception:
            _SIM = simulate
    return _SIM

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
