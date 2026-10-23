# -*- coding: utf-8 -*-
"""АНАЛИТИЧЕСКИЕ ОЖИДАНИЯ ДО СЧЁТА. Ни одна функция здесь не видит результатов simulate:
всё считается из отобранных наборов, слоёв сна и коры и постоянных модели. Поэтому числа,
которые она выдаёт, можно ставить порогами правил, не впадая в подгонку под уже увиденное.

ЕДИНАЯ МОДЕЛЬ ОПАСНОСТИ, из которой выведены почти все ожидания этого файла.
Логарифм нагрузки у набора стоит на m0 = ln(lo*(hi/lo)^q), барьер (верх окна) — на ln(hi).
Обозначим зазор в логарифме
      G = ln(hi) - m0 = (1 - q)*ln(hi/lo),
а суточный подъём нагрузки — ell(t) с размахом L = ln(day_pk). Сомногены растут всё
бодрствование и спадают во сне, поэтому ell(t) берётся кусочно-линейной: вверх за долю
бодрствования w, вниз за остаток суток. Быстрый шум — гауссов с сигмой sigma_tot.
Срыв есть выброс шума за барьер, и его частота при больших зазорах идёт как
      h(t) = exp(-d(t)^2/2),   d(t) = (G - ell(t))/sigma.
Предэкспонента меняется медленно и во ВСЕХ отношениях ниже сокращается, поэтому её тут нет.
Это тот же закон сохранения разрывов, что в разделе 1в, только проинтегрированный по суткам.

Что эта модель НЕ знает: гистерезиса (повреждённая клетка держится, пока не восстановится x),
индукции защиты и связи со сном. Поэтому там, где гистерезис может только добавить (Ф5),
ожидание помечено как НИЖНЯЯ ГРАНИЦА, а не как точка.
"""
import math
import numpy as np
from scipy.stats import norm
from .consts import SR_AMP, SR_RATE, TAU_SR, DSWING
from .sleep import pr_free

NT = 480                     # точек на сутки в численных интегралах: шаг 3 минуты


def _profile(w, L, nt=NT):
    """Суточный профиль подъёма нагрузки ell(t) и признак бодрствования. t в долях суток."""
    t = (np.arange(nt) + 0.5) / nt
    aw = t <= w
    ell = np.where(aw, L * t / max(w, 1e-9), L * (1 - (t - w) / max(1 - w, 1e-9)))
    return t, aw, ell


def _h(G, ell, sig):
    """Частота срывов с точностью до постоянной: exp(-d^2/2) при d = (G - ell)/sigma."""
    d = (np.asarray(G)[..., None] - ell) / sig
    return np.exp(-0.5 * d * d)


def late_share(S, afac=1.0):
    """Ф2(а). Доля срывов во ВТОРОЙ половине бодрствования.
    Если бы срывы были равномерны по бодрствованию, вышло бы ровно 0,5 — это нулевое значение
    правила. В модели опасности нагрузка к концу бодрствования выше на L, то есть барьер ближе
    на L/sigma сигм, и частота срывов растёт по экспоненте квадрата. Отсюда ожидание
          late = int_{w/2}^{w} h dt / int_0^{w} h dt.
    Считается по каждому набору полосы и возвращается медианой: вердикт правила тоже по полосе."""
    G = np.asarray(S['gap_up']) + np.log(S['day_peak']); sig = float(S['sigma_total'])
    w = float(S['wake_h'] / S['period_h']); L = afac * math.log(S['day_peak'])
    t, aw, ell = _profile(w, L)
    h = _h(G, ell, sig)
    num = h[:, aw & (t > w / 2)].sum(1); den = h[:, aw].sum(1)
    v = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
    return v


def swing_curve(S, afacs):
    """Ф2(б). Отклик частоты срывов на размах суточной нагрузки, БЕЗ всяких ворот: просто
    интеграл опасности по суткам при размахе a*L. Возвращает H(a)/H(1) по каждому набору.
    Ворот в этой модели нет по построению, поэтому если наблюдённая кривая доли повреждавшихся
    ложится на эту, то порога по размаху нет, и прежний отрицательный результат подтверждён
    количественно, а не на глаз."""
    G = np.asarray(S['gap_up']) + np.log(S['day_peak']); sig = float(S['sigma_total'])
    w = float(S['wake_h'] / S['period_h']); L0 = math.log(S['day_peak'])
    out = {}
    for a in afacs:
        _, _, ell = _profile(w, a * L0)
        out[a] = _h(G, ell, sig).mean(1)
    return out


def share_ever_pred(S, afacs, ndays, anchor_share, anchor_a=1.0):
    """Ф2(б), продолжение. Доля наборов, повреждавшихся хоть раз за прогон, как функция размаха.
    Предэкспонента опасности неизвестна, поэтому она берётся ОДНИМ числом из наблюдения при
    рабочем размахе (anchor_share) — это одно-параметрическая привязка, и так и записано.
    Дальше кривая предсказана целиком: share(a) = <1 - exp(-lam*H_i(a)/H_i(anchor))>."""
    H = swing_curve(S, list(afacs) + [anchor_a]); Ha = np.maximum(H[anchor_a], 1e-300)
    lam = -math.log(max(1 - min(max(anchor_share, 1e-6), 1 - 1e-6), 1e-6))
    return {a: float(np.mean(1 - np.exp(-lam * H[a] / Ha))) for a in afacs}


def mid_share(S, sigv, up_only=False):
    """Ф5. Доля времени, которую доля повреждённых D проводит в середине шкалы (0,3-0,7),
    если градуальность целиком делает РАЗБРОС уязвимости, а гистерезиса нет вовсе.
    Выкладка. Клетка с множителем v ломается при ln(nu*v) > ln(hi). Множители — exp(sigma_v*z),
    z гауссов, поэтому D = 1 - Phi(z*), где sigma_v*z* = ln(hi/nu). Значит
          0,3 < D < 0,7  <=>  ln(hi/nu) в (sigma_v*z_0,3 ; sigma_v*z_0,7) = (-c ; +c),  c = 0,5244*sigma_v,
    то есть логарифм нагрузки должен попасть в полосу ШИРИНОЙ 2c вокруг барьера. Доля времени
    в полосе — это гауссова вероятность при текущем суточном подъёме, усреднённая по суткам:
          mid = < Phi((ell - G + c)/sigma) - Phi((ell - G - c)/sigma) >_t.
    Поправка на гистерезис, и почему она НЕ знаковая (исправлено 14.10, после того как
    наблюдение 0,0398 оказалось НИЖЕ квазистатической оценки 0,139, то есть «нижняя граница»
    была неверна). Гистерезис действует в две стороны: он задерживает D в середине на обратном
    ходу нагрузки — это добавляет время, — но он же не пускает D обратно через середину, пока x
    не восстановился, и тогда обратного прохода нет вовсе — это отнимает время. Если считать, что
    средним счётом пропадает ровно обратный проход, остаётся половина: up_only = True. Поэтому
    ожидание даётся ВИЛКОЙ (половина; квазистатика), а не точкой, и правило проверяется по вилке."""
    G = np.asarray(S['gap_up']) + np.log(S['day_peak']); sig = float(S['sigma_total'])
    w = float(S['wake_h'] / S['period_h']); L = math.log(S['day_peak'])
    c = float(sigv) * (norm.ppf(0.7) - norm.ppf(0.3)) / 2.
    _, _, ell = _profile(w, L)
    z = (ell - np.asarray(G)[:, None]) / sig
    v = (norm.cdf(z + c / sig) - norm.cdf(z - c / sig)).mean(1)
    return 0.5 * v if up_only else v


def trig_share(S, amp=SR_AMP, rate=SR_RATE, tau_h=TAU_SR):
    """Ф8(а). Доля срывов, вызванных серотониновым всплеском.
    Нулевое значение (всплеск ни при чём, совпадение по времени): 1 - exp(-rate*0,5 сут) при окне
    12 ч. Ожидание механизма: всплеск поднимает нагрузку в amp раз, то есть приближает барьер на
    B = ln(amp)/sigma сигм на время порядка 2*tau. Доля срывов, добавленных всплесками,
          trig = R_всп / (R_всп + R_0),
    где R_0 = <h> по суткам, а R_всп = rate * int по окну [h(ell + B) - h(ell)] со случайной фазой
    всплеска в сутках. Главное, что эта величина видит: НАБОРЫ ПОЛОСЫ отобраны условием
    gap_up <= ln(SR_AMP*day_pk), то есть у многих из них суточного пика хватает и БЕЗ всплеска —
    у таких trig близко к нулю по построению отбора, и общая доля обязана выйти много ниже
    записанного в правиле 0,80. Взвешивание наборов равное: сколько срывов даст каждый набор,
    эта модель не знает, и это её прямая оговорка."""
    G = np.asarray(S['gap_up']) + np.log(S['day_peak']); sig = float(S['sigma_total'])
    w = float(S['wake_h'] / S['period_h']); L = math.log(S['day_peak'])
    t, aw, ell = _profile(w, L)
    B = math.log(amp); win = 2 * tau_h / 24.                      # всплеск живёт около двух постоянных спада
    h0 = _h(G, ell, sig)                                          # (наборов, точек)
    hs = _h(G, ell + B, sig)
    R0 = h0.mean(1)                                               # срывов в сутки без всплесков, с точностью до постоянной
    # фаза всплеска равномерна по суткам, поэтому прибавка усреднена по всем положениям окна:
    # среднее по фазе от интеграла по окну длины win равно win * <прибавка> по суткам.
    Rs = rate * win * (hs - h0).mean(1)
    pred = Rs / np.maximum(Rs + R0, 1e-300)
    null = 1 - math.exp(-rate * 0.5)
    need = np.asarray(S['gap_up']) > math.log(S['day_peak'])      # суточного пика не хватает без всплеска
    return dict(pred=pred, null=float(null), need_spike=float(np.mean(need[S['band']])),
                boost_sigma=float(B / sig), window_days=float(win))


def rate_response(S, rates, amp=SR_AMP, tau_h=TAU_SR):
    """Ф8(г). Во сколько раз частота срывов при частоте всплесков rate больше, чем при наименьшей
    из перебора. Предэкспонента опасности сокращается в отношении, поэтому отношения предсказаны
    БЕЗ всякой привязки: R(rate) = R_0 + rate*окно*<прибавка>, и отношения R(rate)/R(rate_мин)
    зависят только от зазоров наборов, размаха суток и амплитуды всплеска."""
    G = np.asarray(S['gap_up']) + np.log(S['day_peak']); sig = float(S['sigma_total'])
    w = float(S['wake_h'] / S['period_h']); L = math.log(S['day_peak'])
    t, aw, ell = _profile(w, L); win = 2 * tau_h / 24.
    R0 = _h(G, ell, sig).mean(1); dH = (_h(G, ell + math.log(amp), sig) - _h(G, ell, sig)).mean(1)
    band = S['band']
    tot = {r: float(np.mean((R0 + r * win * dH)[band])) for r in rates}
    r0 = min(rates)
    return {r: tot[r] / max(tot[r0], 1e-300) for r in rates}


_WAKE = {}


def wake_h(kD, M=1600, nd=20):
    """Длина бодрствования при постоянном торможении сонного ядра k*D, часы. Детерминированный
    переключатель, никакого счёта клеток: это и есть лестница из раздела 0, посчитанная точечно."""
    key = (round(float(kD), 6), M, nd)
    if key not in _WAKE:
        _WAKE[key] = float(pr_free(kd=float(kD), M=M, nd=nd)[0])
    return _WAKE[key]


def bout_gain(k, d_lo, d_hi, M=1600, nd=20):
    """Ф4(а) и Ф6(а). Во сколько раз длиннее бодрствование при повреждении d_hi, чем при d_lo.
    Повреждение входит в приток к сонному ядру ровно как -k*D, поэтому ответ берётся из самого
    переключателя и НЕ требует ни одного прогона клеток. Это ожидание механизма: если наблюдение
    его сильно превосходит, работает не торможение ядра, а что-то ещё."""
    w0 = wake_h(k * d_lo, M, nd); w1 = wake_h(k * d_hi, M, nd)
    return dict(ratio=float(w1 / w0) if w0 > 0 else float('nan'), wake_lo_h=w0, wake_hi_h=w1,
                k=float(k), d_lo=float(d_lo), d_hi=float(d_hi))


def episode_dur_affine(taus_days, a=1.63, b=0.73):
    """Ф7(а) и Ф8(б). Длительность эпизода по измеренному в Ф16 закону: dur = a + b*tau_защиты,
    сутки (RMS подгонки 0,22 сут на пяти точках перебора памяти). Закон АФФИННЫЙ, а не
    насыщающийся, поэтому пассивная защита не доводит эпизод до двух суток нигде внутри
    измеренного диапазона 24-88 ч — отсюда и ожидание для Ф7(а)."""
    return float(a + b * float(taus_days))


def summary(S, cfg, afacs=(0.0, 0.5, 1.0, 2.0)):
    """Все ожидания одним словарём: печатается в отчёт ДО вердиктов."""
    sigv = cfg.d['model']['sigv']; k = cfg.d['model']['kd']; band = S['band']
    lt = late_share(S); md = mid_share(S, sigv); tg = trig_share(S)
    w = float(S['wake_h'] / S['period_h'])
    return dict(late_share=float(np.nanmedian(lt[band])), late_share_all=float(np.nanmedian(lt)),
                sleep_end_null=float(1 - w), wake_frac=w,
                mid_share_quasi=float(np.mean(md[band])), mid_share_half=float(0.5 * np.mean(md[band])),
                mid_single_cell=0.011,
                trig_share=float(np.mean(tg['pred'][band])), trig_null=tg['null'],
                trig_need_spike=tg['need_spike'], spike_boost_sigma=tg['boost_sigma'],
                dur_affine=episode_dur_affine(cfg.d['model']['taus']),
                gap_sigma=float(np.median((np.asarray(S['gap_up'])[band] + math.log(S['day_peak'])) / S['sigma_total'])),
                day_rise_sigma=float(math.log(S['day_peak']) / S['sigma_total']))
