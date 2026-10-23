# -*- coding: utf-8 -*-
"""Отчёт прогона и ЖУРНАЛ ВЕРДИКТОВ.

Журнал нужен потому, что прогоны теперь выборочные и каждый отчёт частичный. Он хранит по
каждому правилу: последнее посчитанное значение, вердикт, дату, хеши блоков, которыми он
получен, и пометки. Благодаря этому «отрицательные результаты сохраняются» работает само,
а не моей памятью, и видно, какой вердикт на каком коде и на каких блоках получен.

Вердикты ПОСЛЕ СЧЁТА НЕ ПЕРЕСМАТРИВАЮТСЯ, даже если правило оказалось негодным: негодность
отмечается в поле note рядом. Исправление кода под УЖЕ ЗАПИСАННОЕ правило — не пересмотр.
"""
import datetime, json, os
import numpy as np


def _plain(o):
    if isinstance(o, dict): return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_plain(v) for v in o]
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return _plain(o.tolist())
    return o


class Ledger:
    def __init__(self, path=None):
        path = path or os.environ.get('REDOX_LEDGER', 'ledger.json')
        self.path = path
        self.d = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else dict(rules={}, runs=[])

    def record(self, rule, part, value, verdict, blocks, note=''):
        r = self.d['rules'].setdefault(rule, {})
        r[part] = dict(value=_plain(value), verdict=_plain(verdict), blocks=list(blocks),
                       date=datetime.date.today().isoformat(), note=note)

    def run(self, info): self.d['runs'].append(_plain(info)); self.d['runs'] = self.d['runs'][-200:]

    def save(self):
        json.dump(self.d, open(self.path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def table(self):
        L = ['| правило | часть | значение | вердикт | дата | блоки | пометка |', '|---|---|---|---|---|---|---|']
        for rule in sorted(self.d['rules']):
            for part in sorted(self.d['rules'][rule]):
                z = self.d['rules'][rule][part]
                v = z['value']
                vs = ('%.4g' % v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
                L.append('| %s | %s | %s | %s | %s | %s | %s |' % (
                    rule, part, vs[:40], z['verdict'], z['date'], ', '.join(z['blocks'][:3]), z['note'][:60]))
        return '\n'.join(L)


class Report:
    def __init__(self, out, title):
        self.out = out; os.makedirs(out, exist_ok=True)
        self.L = ['# %s\n' % title]; self.R = {}

    def md(self, *lines): self.L += list(lines)

    def json(self, key, val): self.R[key] = _plain(val)

    def save(self, extra=''):
        if extra: self.L.append(extra)
        self.L += ['\n## Для машинного чтения\n', '```json',
                   json.dumps(self.R, ensure_ascii=False, indent=1, default=float), '```']
        open(os.path.join(self.out, 'report.md'), 'w', encoding='utf-8').write('\n'.join(self.L) + '\n')
        json.dump(self.R, open(os.path.join(self.out, 'results.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1, default=float)
