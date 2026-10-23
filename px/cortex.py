# -*- coding: utf-8 -*-
"""Кора: Вильсон–Коуэн, пара E и PV, стационарное решение затухающей итерацией."""
# ПЕРЕНЕСЕНО ИЗ mass_core6.py ДОСЛОВНО, автоматически, не перенабрано руками.
# Любая правка здесь обязана проходить регрессионный тест tests/test_regression.py,
# который требует ПОБИТОВОГО совпадения с эталонным датасетом в ref/.
import numpy as np
from .consts import *

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
