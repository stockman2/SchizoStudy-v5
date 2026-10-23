# -*- coding: utf-8 -*-
"""Переключатель бодрствования Филлипса–Робинсона: складки, свободный цикл, калибровка."""
# ПЕРЕНЕСЕНО ИЗ mass_core6.py ДОСЛОВНО, автоматически, не перенабрано руками.
# Любая правка здесь обязана проходить регрессионный тест tests/test_regression.py,
# который требует ПОБИТОВОГО совпадения с эталонным датасетом в ref/.
import math
import numpy as np
from .consts import *
from .cortex import wc_fix

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
