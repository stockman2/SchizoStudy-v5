# -*- coding: utf-8 -*-
"""Правила. Каждое — отдельный файл с двумя вещами: needs(cfg) -> список блоков
и run(ctx) -> (словарь для машинного чтения, список строк отчёта).
Никакого слоя абстрактного правила с наследованием здесь нет намеренно: иначе через неделю
мы будем отлаживать фреймворк вместо модели."""
import importlib, pkgutil

def available():
    return sorted(m.name for m in pkgutil.iter_modules(__path__) if m.name.startswith('f'))

def load(name):
    return importlib.import_module('.' + name, __name__)
