# -*- coding: utf-8 -*-
"""Статистики и критерии. Каждый критерий здесь обязан знать про себя три вещи:
нулевое значение, свой шум при нашем n и свою мощность. Это правило добыто дорого —
четырьмя промахами (Ф9, Ф13, Ф14, доля коротких промежутков и доля коротких эпизодов),
и оно записано в заметке analiz_velichin_do_progona.txt.

Отдельно: у знакового критерия НАПРАВЛЕНИЕ задаётся обязательным аргументом без значения
по умолчанию. Причина — ошибка 14.10 в правиле Ф19(в): для длительности «меньше» значит
«лучше», а для доли коротких эпизодов «лучше» значит «больше», и одна и та же функция,
применённая к обоим, дала p = 0,99 там, где правильное направление давало 0,038.
Теперь перепутать направление молча нельзя: аргумент обязателен, а self_test() проверяет
оба направления на синтетике с известным ответом.
"""
import math
import numpy as np
from scipy.stats import binom, norm


def sign_test(a, b, mask, direction, alpha=0.05):
    """Парный знаковый критерий: a — контроль, b — опыт, по наборам.
    direction = 'decrease' — ждём, что b МЕНЬШЕ a (длительность эпизода);
    direction = 'increase' — ждём, что b БОЛЬШЕ a (доля коротких эпизодов).
    Под нулевой гипотезой k ~ Bin(n, 1/2): E[k] = n/2, Var[k] = n/4. Вердикт по точному
    биномиальному p. Распределение самой величины не используется вовсе, поэтому её
    дисперсию знать не нужно. Связки (a == b) исключаются, как и положено."""
    assert direction in ('decrease', 'increase'), 'направление критерия обязательно'
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.asarray(mask, bool) & np.isfinite(a) & np.isfinite(b) & (a != b)
    n = int(ok.sum())
    if n < 10:
        return dict(n=n, k=0, p=float('nan'), direction=direction, verdict='непроверяемо')
    k = int((b[ok] < a[ok]).sum() if direction == 'decrease' else (b[ok] > a[ok]).sum())
    mu = n / 2.; sd = math.sqrt(n / 4.); p = float(binom.sf(k - 1, n, 0.5))
    return dict(n=n, k=k, mu=float(mu), sd=float(sd), crit=float(mu + 1.645 * sd),
                z=float((k - mu) / sd), p=p, direction=direction, verdict=bool(p <= alpha))


def med_upper(v, alpha=0.05):
    """Односторонняя верхняя граница медианы по порядковой статистике: наименьшее r, при
    котором P(Bin(n,1/2) <= r-1) >= 1-alpha; граница — r-я по возрастанию. Распределение
    величины не предполагается. Покрытие проверено розыгрышем на экспоненциальном,
    логнормальном и смеси двух при n = 30/60/120: промах 0,038-0,052 при номинальных 0,05."""
    x = np.sort(np.asarray(v, float)); x = x[np.isfinite(x)]; n = len(x)
    if n < 10: return dict(bound=float('nan'), rank=-1, n=n)
    for r in range(1, n + 1):
        if binom.cdf(r - 1, n, 0.5) >= 1 - alpha:
            return dict(bound=float(x[r - 1]), rank=r, n=n)
    return dict(bound=float(x[-1]), rank=n, n=n)


def cv2_gaps(gap_n, gap_sum, gap_sq, mask, nmin=5, nrep=400, seed=555):
    """Медиана по наборам от CV^2 ИХ СОБСТВЕННЫХ промежутков, и её нулевое значение при том
    же наборе n. Вывод: при глухом периоде t_r и пуассоновском риске промежуток = t_r + Exp,
    откуда CV^2 = (1 - t_r/g)^2, то есть t_r/g = 1 - CV; без глухого периода CV^2 = 1 РОВНО
    при любой частоте. Величина безразмерна и считается ПО НАБОРУ, поэтому разнородность
    ансамбля в неё не попадает — в отличие от доли коротких промежутков, которая равна
    1 + CV^2(частот) и физики не видит вовсе."""
    rng = np.random.default_rng(seed)
    n_ = np.asarray(gap_n, float)[mask]; ok = n_ >= nmin
    if ok.sum() < 5: return dict(cv2=float('nan'), null=float('nan'), n_sets=int(ok.sum()), tr_over_g=float('nan'))
    n_ = n_[ok]; sm = np.asarray(gap_sum, float)[mask][ok]; sq = np.asarray(gap_sq, float)[mask][ok]
    mu = sm / n_; var = np.maximum((sq - n_ * mu * mu) / np.maximum(n_ - 1, 1), 0.0)
    obs = float(np.median(var / np.maximum(mu * mu, 1e-12)))
    nul = []
    for _ in range(nrep):
        v = [(lambda g: g.var(ddof=1) / max(g.mean() ** 2, 1e-12))(rng.exponential(1.0, int(nn))) for nn in n_]
        nul.append(np.median(v))
    return dict(cv2=obs, null=float(np.median(nul)), n_sets=int(ok.sum()),
                tr_over_g=float(1 - math.sqrt(max(obs, 0.0))))


def boot_se(v, nrep=2000, seed=11):
    """Относительная стандартная ошибка медианы, загрузкой по наборам. Нужна, чтобы порог
    ставился от шума величины, а не от круглого числа."""
    x = np.asarray(v, float); x = x[np.isfinite(x)]
    if len(x) < 5: return float('nan')
    rng = np.random.default_rng(seed)
    m = float(np.median(x))
    bs = [float(np.median(rng.choice(x, len(x)))) for _ in range(nrep)]
    return float(np.std(bs) / max(abs(m), 1e-12))


def spread(vals):
    """Размах последовательности в долях её среднего: для проверок на независимость от параметра."""
    v = [x for x in vals if x == x]
    return float((max(v) - min(v)) / max(abs(np.mean(v)), 1e-12)) if len(v) >= 2 else float('nan')


def self_test():
    """Проверка самих критериев на синтетике с ИЗВЕСТНЫМ ответом. Гоняется на каждом прогоне:
    если критерий сломан, прогон обязан об этом сказать до всяких вердиктов."""
    rng = np.random.default_rng(7); out = {}
    n = 60; a = rng.exponential(4.0, n); b = a * 0.5                                   # опыт вдвое меньше
    mk = np.ones(n, bool)
    d = sign_test(a, b, mk, 'decrease'); i = sign_test(a, b, mk, 'increase')
    out['sign_decrease_p'] = d['p']; out['sign_increase_p'] = i['p']
    out['sign_direction_ok'] = bool(d['p'] < 1e-10 and i['p'] > 0.99)                  # оба направления различимы
    perm = 0
    for _ in range(2000):
        sg = rng.choice([-1.0, 1.0], n); dif = (b - a) * sg
        perm += (sign_test(np.zeros(n), dif, mk, 'decrease')['p'] <= 0.05)
    out['sign_false_alarm'] = perm / 2000.                                             # должно быть около 0,05
    out['sign_calibrated'] = bool(out['sign_false_alarm'] <= 0.09)   # односторонне: консервативность критерия не дефект
    cov = 0
    for _ in range(1500):
        x = rng.exponential(4.0, 60)
        cov += (float(np.log(2) * 4.0) > med_upper(x)['bound'])
    out['med_bound_miss'] = cov / 1500.
    out['med_bound_ok'] = bool(out['med_bound_miss'] <= 0.09)
    nn = np.full(60, 8.0)
    g = 0.25 * 4 + rng.exponential(3.0, (60, 8))                                       # вшит глухой период t_r/g = 0,25
    r = cv2_gaps(nn, g.sum(1), (g * g).sum(1), np.ones(60, bool), nrep=60)
    out['cv2_with_refractory'] = r['cv2']; out['cv2_power_ok'] = bool(r['cv2'] <= 0.50)
    out['all_ok'] = bool(out['sign_direction_ok'] and out['sign_calibrated'] and out['med_bound_ok'] and out['cv2_power_ok'])
    return out


# ====================================================================================
# МОЩНОСТЬ И ВЕРДИКТ «НЕПРОВЕРЯЕМО»
# ====================================================================================
# Решение пользователя 14.10: если у части правила не хватает мощности, ставится
# «непроверяемо» СИММЕТРИЧНО — и там, где она провалилась бы, и там, где прошла бы.
# Основание: «прошло» на тесте без мощности так же бессмысленно, как «не прошло».
#
# Критерий один на все правила: часть проверяема, если расстояние от НУЛЕВОГО значения
# до ПОРОГА не меньше минимального различимого эффекта
#       MDE = (z_0,95 + z_0,80) * SE = 2,486 * SE
# при односторонней альфе 0,05 и мощности 0,80. Иначе тест не способен отличить порог
# от нуля при нашем n, и вердикт не выносится. Рядом печатается n, которое нужно.

Z95, Z80 = 1.6448536269514722, 0.8416212335729143
MDE_K = Z95 + Z80            # 2.4864748605243866


def mde(se): return MDE_K * float(se)


# Допуск РАВНОСИЛЬНОСТИ: когда правило утверждает «наблюдение СОГЛАСУЕТСЯ с предсказанием»,
# допуск обязан быть ШИРЕ минимального различимого эффекта, иначе утверждение неопровержимо
# в принципе: при допуске 2 SE зазор 2 SE всегда меньше MDE = 2,4865 SE, и вердикт всегда
# «непроверяемо». Поэтому допуск равносильности — 3 SE, и это записано здесь, а не подобрано
# в правиле.
EQ_K = 3.0
def eq_tol(se): return EQ_K * float(se)


def need_n(n, se, gap):
    """Сколько наборов нужно, чтобы зазор gap стал различимым при той же дисперсии."""
    if not (se == se and gap == gap) or gap == 0: return float('nan')
    return float(n * (mde(se) / abs(gap)) ** 2)


def adjudicate(value, thresh, se, direction, null=None, n=None, alpha=0.05, lim=None):
    """Единый вердикт с проверкой мощности.
    direction = 'le' — гипотеза «величина не больше порога»; 'ge' — «не меньше порога».
    null — значение под нулевой гипотезой; если не задано, берётся порог (тогда проверяется
    только достоверность отклонения, а зазор считается от самого порога, что строже).
    Возвращает словарь с verdict: True / False / 'непроверяемо'."""
    assert direction in ('le', 'ge')
    v = float(value); th = float(thresh)
    nul = th if null is None else float(null)
    gap = abs(th - nul) if null is not None else abs(v - th)
    m = mde(se) if se == se else float('nan')
    weak = not (m == m) or (gap < m)
    raw = (v <= th) if direction == 'le' else (v >= th)
    out = dict(value=v, thresh=th, null=nul, se=(float(se) if se == se else float('nan')),
               mde=m, gap=gap, raw_verdict=bool(raw), n=n,
               n_required=need_n(n, se, gap) if n else float('nan'))
    out['verdict'] = ('непроверяемо' if weak else bool(raw))
    # Порог, вышедший за область достижимых значений величины (доля вне [0; 1], длительность
    # меньше нуля), делает правило невыполнимым ПО АРИФМЕТИКЕ, а не по природе. Это тоже
    # «непроверяемо», и причина печатается отдельно.
    if lim is not None and not (float(lim[0]) <= th <= float(lim[1])):
        out['verdict'] = 'непроверяемо'; out['reason'] = 'порог вне достижимого диапазона %s' % (lim,)
    elif weak:
        out['reason'] = 'зазор %.4g меньше MDE %.4g' % (gap, m) if m == m else 'SE не определена'
    return out


def sign_power(n, p_alt=0.75, alpha=0.05):
    """Мощность знакового критерия при n парах против альтернативы p_alt. Точно, по биному."""
    if n < 5: return 0.0
    k = int(binom.isf(alpha, n, 0.5)) + 1          # наименьшее k, отклоняющее нуль
    return float(binom.sf(k - 1, n, p_alt))


def sign_adjudicate(res, p_alt=0.75, alpha=0.05, min_power=0.80):
    """Обёртка к sign_test: при недостаточной мощности вердикт становится «непроверяемо»."""
    n = int(res.get('n', 0)); pw = sign_power(n, p_alt, alpha)
    nn = n
    while nn < 4000 and sign_power(nn, p_alt, alpha) < min_power: nn += 1
    r = dict(res); r['power'] = pw; r['p_alt'] = p_alt; r['n_required'] = nn
    if pw < min_power: r['verdict'] = 'непроверяемо'
    return r


def slope_test(x, y, sy, direction, null=0.0, alpha=0.05):
    """Наклон взвешенной прямой по точкам перебора и ЕГО СОБСТВЕННАЯ ошибка.
    Зачем он нужен. Монотонность по переборам из 3-7 точек проверялась знаковым критерием по
    парам соседних точек, и у него не было мощности: при двух изменившихся парах нужно 23.
    Но у каждой точки перебора есть СВОЯ ошибка (биномиальная у доли, пуассоновская у частоты),
    и наклон взвешенной прямой использует их все сразу:
          b = sum w_i (x_i - <x>)(y_i - <y>) / sum w_i (x_i - <x>)^2,   w_i = 1/sigma_i^2,
          SE(b) = 1 / sqrt(sum w_i (x_i - <x>)^2).
    При 2000 наборов на точку SE наклона мала, и монотонность становится проверяемой тремя точками.
    direction: 'increase' — ждём b > null, 'decrease' — b < null."""
    assert direction in ('increase', 'decrease')
    x = np.asarray(x, float); y = np.asarray(y, float); sy = np.asarray(sy, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sy) & (sy > 0)
    n = int(ok.sum())
    if n < 3 or np.ptp(x[ok]) == 0:
        return dict(slope=float('nan'), se=float('nan'), mde=float('nan'), gap=float('nan'),
                    value=float('nan'), thresh=float('nan'), null=float(null), n=n,
                    n_required=float('nan'), direction=direction, verdict='непроверяемо',
                    reason='точек меньше трёх или перебор вырожден')
    # пол на ошибку: точка с нулевой ошибкой получила бы бесконечный вес и задавила бы все
    # остальные. Пол берётся как сотая доля наибольшей ошибки в переборе.
    sy_ = np.maximum(sy[ok], 0.01 * np.nanmax(sy[ok]))
    x_, y_, w = x[ok], y[ok], 1.0 / sy_ ** 2
    xm = np.sum(w * x_) / np.sum(w); ym = np.sum(w * y_) / np.sum(w)
    sxx = np.sum(w * (x_ - xm) ** 2)
    b = float(np.sum(w * (x_ - xm) * (y_ - ym)) / sxx); se = float(1.0 / math.sqrt(sxx))
    out = adjudicate(b if direction == 'increase' else -b,
                     (null + mde(se)) if direction == 'increase' else -(null - mde(se)),
                     se, 'ge', null=null if direction == 'increase' else -null, n=n)
    out.update(slope=b, se=se, n=n, direction=direction, x=[float(v) for v in x_],
               y=[float(v) for v in y_])
    return out


def trend_test(succ, ntr, scores, direction='increase', alpha=0.05):
    """Критерий тренда Кохрана–Армитеджа: растёт ли ДОЛЯ по упорядоченным точкам перебора.
    Зачем он вместо наклона взвешенной прямой (найдено 16.10 на Ф3(б)). Доля застрявших идёт
    0, 0, 0, 0, 0, 0,026, 0,499 — это не прямая, а клюшка, и наклон прямой по таким точкам
    занижен настолько, что не доходит до собственного MDE: вердикт выходит «не выполнено» при
    тридцати сигмах разницы между краями перебора. Кохран–Армитедж для этого и придуман:
          z = sum n_i (x_i - <x>) (p_i - p_общ) / sqrt( p_общ (1 - p_общ) * sum n_i (x_i - <x>)^2 ),
    где дисперсия берётся по ОБЩЕЙ доле под нулевой гипотезой. Нулевые ячейки тут не дают нулевой
    дисперсии (она общая), поэтому ловушка бесконечного веса не возникает вовсе."""
    assert direction in ('increase', 'decrease')
    k = np.asarray(succ, float); n = np.asarray(ntr, float); x = np.asarray(scores, float)
    ok = np.isfinite(k) & np.isfinite(n) & np.isfinite(x) & (n > 0)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return dict(z=float('nan'), p=float('nan'), n=int(ok.sum()), verdict='непроверяемо',
                    reason='точек меньше трёх или перебор вырожден')
    k, n, x = k[ok], n[ok], x[ok]
    N = n.sum(); p0 = k.sum() / N
    xb = float(np.sum(n * x) / N)
    num = float(np.sum(n * (x - xb) * (k / n - p0)))
    den = math.sqrt(max(p0 * (1 - p0) * float(np.sum(n * (x - xb) ** 2)), 1e-300))
    z = num / den
    if direction == 'decrease': z = -z
    pv = float(norm.sf(z))
    return dict(z=float(z), p=pv, n=int(len(k)), p_pooled=float(p0), direction=direction,
                shares=[float(v) for v in (k / n)], scores=[float(v) for v in x],
                verdict=bool(pv <= alpha))


def rho_se(n): return float('nan') if n is None or n < 4 else 1.0 / math.sqrt(n - 1)


def prop_se(p, n, jeffreys=True):
    """Ошибка доли. ВАЖНО про нулевую долю (ловушка найдена 16.10 по прогону Ф3(б)).
    При p = 0 обычная формула даёт SE = 0, то есть ВЕС такой точки во взвешенной прямой равен
    бесконечности: пять нулевых точек перебора задавили две ненулевые, наклон вышел ровно 0,0
    с ошибкой ровно 0,0, и вердикт «не выполнено» был получен из арифметики, а не из модели.
    Лечение — поправка Джеффриса: вместо самой доли в ОШИБКУ подставляется (k + 1/2)/(n + 1),
    то есть нулевая доля ведёт себя как «меньше одного события», а не как «событий не бывает».
    Само значение доли при этом не меняется — поправка входит только в ошибку."""
    if n is None or n <= 0: return float('nan')
    p = min(max(float(p), 0.0), 1.0)
    pe = (p * n + 0.5) / (n + 1.0) if jeffreys else p
    return math.sqrt(max(pe * (1 - pe), 1e-12) / n)


def power_selftest():
    """Проверка самой машинерии мощности на известных ответах."""
    o = {}
    o['mde_k'] = MDE_K
    o['mde_k_ok'] = bool(abs(MDE_K - 2.4865) < 1e-3)
    o['sign_power_30'] = sign_power(30, 0.75); o['sign_power_10'] = sign_power(10, 0.75)
    o['sign_power_mono'] = bool(sign_power(10, 0.75) < sign_power(30, 0.75) < sign_power(80, 0.75))
    o['rho_n_for_040'] = float((MDE_K / 0.40) ** 2 + 1)       # сколько наборов нужно для порога 0,40
    a = adjudicate(0.30, 0.40, 0.25, 'ge', null=0.0, n=20)     # заведомо слабый тест
    b = adjudicate(0.30, 0.40, 0.02, 'ge', null=0.0, n=400)    # заведомо сильный
    o['weak_is_unverifiable'] = bool(a['verdict'] == 'непроверяемо')
    o['strong_is_decided'] = bool(b['verdict'] is False)
    o['all_ok'] = bool(o['mde_k_ok'] and o['sign_power_mono'] and o['weak_is_unverifiable'] and o['strong_is_decided'])
    return o


def perm_rank_p(x, y, direction='increase'):
    """ТОЧНЫЙ перестановочный критерий согласия двух порядков. Для малого числа точек.

    Зачем он вместо «ранговая связь равна ровно +1». Требование ровного порядка падает от одной
    инверсии, а одна инверсия на пяти точках случается и при верной модели — это та же ошибка,
    что «строгая монотонность» (найдена 14.10 на Ф14(в)). Здесь нулевая гипотеза — случайная
    расстановка, статистика — ранговая связь, и p считается перебором ВСЕХ перестановок, без
    приближений и без предположений о распределении.

    При n точках перестановок n!: до 8 точек это не дороже мгновения (40320). Больше 8 —
    возвращается nan, и вызывающий обязан взять другой критерий.

    Возвращает (ранговая связь, p односторонний, число точек)."""
    import itertools, math
    from scipy.stats import rankdata
    x = np.asarray(x, float); y = np.asarray(y, float)
    f = np.isfinite(x) & np.isfinite(y)
    x, y = x[f], y[f]; n = len(x)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0: return float('nan'), float('nan'), n
    if n > 8:
        # Перебор всех перестановок дороже разумного (9! = 362880 и дальше хуже), поэтому с девяти
        # точек берётся обычный критерий Спирмена. Он асимптотический, но при n >= 9 его уровень
        # уже близок к точному, а главное — он ДАЁТ p, тогда как возврат nan делал часть правила
        # непроверяемой из-за одной лишней точки перебора (поймано 21.10 на Ф27).
        from scipy.stats import spearmanr
        r = spearmanr(x, y)
        rho_ = float(r.statistic); p2 = float(r.pvalue)
        p1 = p2 / 2.0 if (rho_ > 0) == (direction == 'increase') else 1.0 - p2 / 2.0
        return rho_, p1, n
    rx = rankdata(x); ry = rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    sgn = 1.0 if direction == 'increase' else -1.0
    hit = 0; tot = 0
    for perm in itertools.permutations(range(n)):
        r = float(np.corrcoef(rx, ry[list(perm)])[0, 1]); tot += 1
        if sgn * r >= sgn * rho - 1e-12: hit += 1
    return rho, hit / float(tot), n


def haz_trend(starts, durs, nep, ovf, t0, t1, K=4, direction='increase'):
    """ХОД РИСКА ВО ВРЕМЕНИ: точный условный критерий тренда, порог из матожидания и дисперсии.

    ЗАЧЕМ. Прежняя мера прогрессирования — разность средних долей повреждённых между половинами
    прогона — опиралась на ложную посылку, что при разомкнутой петле прибавки нет. При постоянной
    нагрузке переключатель бистабилен, сломавшийся набор не возвращается, поэтому доля сломанных
    есть монотонно неубывающая функция времени, и её прибавка никогда не ноль. Порог «не больше
    MDE» при этом падает как 1/sqrt(n), а снос не падает вовсе: контроль проходил ЛИШЬ по малости
    выборки и при 2400 наборах упал бы наверняка. Здесь посылка другая и верная: при постоянных
    условиях риск постоянен ВО ВРЕМЕНИ, и проверяется именно это.

    УСТРОЙСТВО. Окно [t0, t1) делится на K равных частей. Для набора i берутся n_i начал эпизодов,
    попавших в окно, и A_ik — время ПОД РИСКОМ в части k, то есть длина части минус время, занятое
    эпизодами. Время под риском обязательно: поглощение само выедает пул, и без поправки на него
    поток событий падал бы со временем даже при постоянном риске.

    НУЛЕВАЯ ГИПОТЕЗА, точно. При постоянном во времени риске h_i распределение начал по частям,
    УСЛОВНО по их числу n_i, полиномиально с p_ik = A_ik / A_i. Отсюда для центрированных оценок
    x_k матожидание и дисперсия выписываются без единого неизвестного параметра:

        E[T_i] = n_i * mu_i,                 mu_i = sum_k x_k p_ik
        Var[T_i] = n_i * (sum_k x_k^2 p_ik - mu_i^2),      T_i = sum_k x_k N_ik

    Суммирование ВНУТРИ НАБОРА и только потом по наборам — обязательно: риски h_i различаются между
    наборами на порядки, и объединённый полиномиальный расчёт подменял бы ход во времени разбросом
    между наборами (та же ошибка, что ловилась 21.10 на сравнении двух средних вместо парного).

    ИНВАРИАНТНОСТЬ К СМЕЩЕНИЮ СЧЁТА. Счётные величины при M = 1600 смещены примерно на 2 %. Это
    смещение общий множитель для всех частей окна, а критерий смотрит на РАСПРЕДЕЛЕНИЕ событий по
    частям, не на их число, поэтому общий множитель сокращается тождественно.

    ПЕРЕПОЛНЕНИЕ. Записей не больше NEPR на набор; у набора с ovf > 0 поздние эпизоды потеряны, и
    потеряны именно у самых активных наборов. Такой набор исключается целиком, а доля исключённых
    возвращается: отбор, зависящий от измеряемой величины, — ровно та ловушка, что ловилась на Ф27.

    Возвращает (Z, N, nsets, frac_excluded).
    """
    starts = np.asarray(starts, float); durs = np.asarray(durs, float)
    nep = np.asarray(nep, int); ovf = np.asarray(ovf, int)
    w = (t1 - t0) / float(K)
    xs = np.arange(K, dtype=float) - 0.5 * (K - 1.0)          # центрированные оценки, sum x = 0
    keep = ovf <= 0
    T = 0.0; ET = 0.0; VT = 0.0; N = 0; used = 0
    for i in range(len(nep)):
        if not keep[i]:
            continue
        m = int(nep[i])
        st = starts[i, :m]; du = durs[i, :m]
        # время под риском в каждой части: длина минус пересечение с эпизодами
        A = np.full(K, w, float)
        for s_, d_ in zip(st, du):
            e_ = s_ + max(d_, 0.0)
            for k in range(K):
                a = t0 + k * w; b = a + w
                ov = min(e_, b) - max(s_, a)
                if ov > 0: A[k] -= ov
        A = np.maximum(A, 0.0)
        if A.sum() <= 0: continue
        sel = (st >= t0) & (st < t1)
        n_i = int(sel.sum())
        used += 1
        if n_i == 0: continue
        kk = np.minimum(((st[sel] - t0) / w).astype(int), K - 1)
        Nk = np.bincount(kk, minlength=K).astype(float)
        p = A / A.sum()
        mu = float((xs * p).sum())
        T += float((xs * Nk).sum()); ET += n_i * mu
        VT += n_i * (float((xs * xs * p).sum()) - mu * mu); N += n_i
    if N < 20 or VT <= 0:
        return float('nan'), N, used, float(1.0 - keep.mean())
    Z = (T - ET) / math.sqrt(VT)
    return (Z if direction == 'increase' else -Z), N, used, float(1.0 - keep.mean())


def haz_trend_day(on_day, risk_day, t0, t1, K=4, direction='increase'):
    """ХОД РИСКА ВО ВРЕМЕНИ по подённым накопителям. Замена haz_trend: без потолка записей и,
    значит, без отбора, зависящего от измеряемой величины (переполнение 15–20 % на 400 сутках).

    Окно [t0, t1) делится на K частей. Для набора i берутся N_ik срывов и A_ik времени под риском.
    При риске h_i, постоянном ВО ВРЕМЕНИ, распределение срывов по частям, условно по их числу n_i,
    полиномиально с p_ik = A_ik / A_i, откуда без неизвестных параметров:

        T_i = sum_k x_k N_ik,  E[T_i] = n_i*mu_i,  Var[T_i] = n_i*(sum_k x_k^2 p_ik - mu_i^2).

    Суммирование внутри набора и только потом по наборам: риски наборов различаются на порядки.

    ВНИМАНИЕ, нулевая гипотеза «риск постоянен» в этой модели ЛОЖНА, и это измерено: 23.10 при
    разомкнутой петле вышло Z = -2.63, риск падает. Причина структурная — внутри набора самые
    уязвимые клетки срываются первыми и остаются сломанными, поэтому под риском остаются всё менее
    уязвимые. Поправка на время под риском считает ВРЕМЯ, но не состав того, кто под риском.
    Поэтому Z здесь годится только для СРАВНЕНИЯ ПЛЕЧ, а не как критерий против нуля.
    """
    on = np.asarray(on_day, float); ar = np.asarray(risk_day, float)
    if on.ndim != 2 or on.shape != ar.shape:
        return float('nan'), 0, 0.0
    ndy = on.shape[1]
    i0 = int(math.ceil(max(t0, 0.0))); i1 = int(min(t1, ndy))
    if i1 - i0 < K:
        return float('nan'), 0, 0.0
    ed = np.linspace(i0, i1, K + 1).astype(int)
    N = np.stack([on[:, ed[k]:ed[k + 1]].sum(1) for k in range(K)], 1)
    A = np.stack([ar[:, ed[k]:ed[k + 1]].sum(1) for k in range(K)], 1)
    xs = np.arange(K, dtype=float) - 0.5 * (K - 1.0)
    At = A.sum(1); good = At > 0
    p = np.zeros_like(A); p[good] = A[good] / At[good, None]
    ni = N.sum(1); ni = np.where(good, ni, 0.0)
    mu = (xs * p).sum(1)
    T = float((xs * N[good]).sum()) if good.any() else 0.0
    ET = float((ni * mu).sum())
    VT = float((ni * ((xs * xs * p).sum(1) - mu * mu)).sum())
    Ntot = int(ni.sum())
    rate = (float(N[good].sum()) / float(At[good].sum())) if good.any() and At[good].sum() > 0 else float('nan')
    if Ntot < 20 or VT <= 0:
        return float('nan'), Ntot, rate
    Z = (T - ET) / math.sqrt(VT)
    return (Z if direction == 'increase' else -Z), Ntot, rate
