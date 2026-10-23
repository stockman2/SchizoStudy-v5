# -*- coding: utf-8 -*-
"""Блоки, их склейка и счёт. Главная мысль пайплайна: правило НЕ считает блоки само,
а ОБЪЯВЛЯЕТ, какие ему нужны. Сборщик берёт объединение запросов выбранных правил,
склеивает одинаковые блоки (их много: Ф11, Ф12, Ф15 и Ф19 все просят k = 0 без шума),
группирует по суткам/шагам/J в батчи и считает каждый блок ровно ОДИН раз.
Выбрали два правила — посчитались только их блоки.

Хеш блока печатается рядом с числами правила. Это важнее, чем кажется: два правила,
просящие «тот же» блок, обязаны получить буквально один и тот же блок, иначе парные
сравнения между правилами развалятся молча.

Распараллеливание — ПО БЛОКАМ, не по строкам. Разрезать строки нельзя без потери
побитовой воспроизводимости: шум разыгрывается на длину блока и тиражируется.
"""
import hashlib, json, math, os, time
import multiprocessing as mp
import numpy as np
from .consts import TAU_S, TSLOW, SLPN, USE_DM, VFIX
from .model import _job

# ---------------------------------------------------------------------------------
# КЭШ РЕЗУЛЬТАТОВ БЛОКОВ. Счёт — единственное дорогое место пайплайна (3,9 часа на 40
# блоков при 2000 наборах), и он ПОЛНОСТЬЮ детерминирован: тот же блок при тех же
# наборах даёт побитово тот же результат. Значит его можно посчитать один раз.
#
# Ключ кэша включает ВСЁ, от чего результат зависит:
#   - разрешённые значения всех построчных полей блока (уже с подставленными слоями),
#   - поля группы: суток, шагов, J, разгон,
#   - зерно блока и положение нагрузки q,
#   - хеш выборки наборов (K, nsel, зерно выборки) и сами начальные нагрузки,
#   - величины слоёв, входящие в счёт: E0, Q_бодр, средние сомногены,
#   - ХЕШ ИСХОДНОГО КОДА МОДЕЛИ (consts, core, cortex, sleep, model).
# Последнее — главная защита: правка модели сама обесценивает кэш, и старые блоки
# никогда не подмешаются к новым. Если нужно обойти кэш, есть ключ --nocache.
# fast.py ВХОДИТ В ХЕШ, и это не перестраховка. 21.10 ядро целилось в сырое P['N0']
# вместо уменьшенной ёмкости N0c: эталон давал 0,85 N0, ядро 0,88886. Правка ядра кэш
# НЕ обесценивала, то есть прогон подмешал бы блоки, посчитанные ядром, которого уже нет.
# Тождественность ядра эталону — это утверждение, которое мы проверяем, а не гарантия;
# кэш не умеет её проверять, поэтому он обязан жить ровно столько, сколько живёт ядро.
CODE_FILES = ('consts.py', 'core.py', 'cortex.py', 'sleep.py', 'model.py', 'fast.py')


def code_version():
    h = hashlib.sha1()
    here = os.path.dirname(os.path.abspath(__file__))
    for f in CODE_FILES:
        with open(os.path.join(here, f), 'rb') as fh: h.update(fh.read())
    return h.hexdigest()[:12]


def _save_block(path, r):
    """Результат блока — словарь из массивов, кортежей массивов и чисел. Кортежи
    раскладываются на части с пометкой в имени, числа кладутся как массивы нулевой
    размерности; при чтении всё собирается обратно."""
    flat = {}
    for k, v in r.items():
        if isinstance(v, tuple):
            flat['T%d|%s' % (len(v), k)] = np.asarray(len(v))
            for i, y in enumerate(v): flat['t%d|%s' % (i, k)] = np.asarray(y)
        else:
            flat['v|%s' % k] = np.asarray(v)
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **flat)
    os.replace(tmp, path)                      # запись атомарна: прерванный прогон не оставит битый файл


def _load_block(path):
    z = np.load(path, allow_pickle=False)
    out = {}; tup = {}
    for name in z.files:
        kind, k = name.split('|', 1)
        if kind == 'v':
            a = z[name]; out[k] = a if a.ndim else (int(a) if a.dtype.kind in 'iu' else float(a))
        elif kind.startswith('T'): tup.setdefault(k, int(kind[1:]))
        else: out.setdefault('__t__' + k, {})[int(kind[1:])] = z[name]
    for k, n in tup.items():
        d = out.pop('__t__' + k)
        out[k] = tuple(d[i] for i in range(n))
    return out

# поля, задающие группу: блоки с разными значениями в один батч не кладутся
GROUP = ('ndays', 'M', 'J', 'warm')
# построчные поля: значение на блок, раскладывается на строки внутри батча
ROW = ('kd', 'aH', 'sigv', 'sigma', 'tauc_rel', 'sigs', 'tslow', 'slpn', 'usedm',
       'sramp', 'srate', 'taus', 'vfix', 'apr', 'taup', 'tausd', 'itp', 'a_da', 'tau_d', 'da_src', 'nu_mul', 'nu_mul2', 't_mul', 'a_n', 'taun_dn', 'taun_up', 'n0_mul')
FROM_SETS = dict(aH='A_H', sigv='sigv', sigma='sigma', tauc_rel='tauc_rel')   # поля, чьё None берётся из посчитанных слоёв
DEFAULTS = dict(kd=0.0, aH=None, sigv=None, sigma=None, tauc_rel=None, sigs=0.0, tslow=TSLOW,
                slpn=SLPN, usedm=1.0 if USE_DM else 0.0, sramp=1.0, srate=0.0, taus=TAU_S,
                vfix=VFIX, apr=0.0, taup=1.0,
                tausd=-1.0,          # <= 0 — спад с той же постоянной, что подъём (прежняя модель)
                itp=1.0,             # 1 — интерполировать момент пересечения порога
                a_da=0.0,            # сила дофаминовой петли; 0 — петля РАЗОМКНУТА (прежняя модель)
                tau_d=7.0,           # постоянная тонуса, сутки
                da_src=0.0,          # чем движется тонус: 0 — растормаживание, 1 — доля повреждённых
                nu_mul=1.0,          # постоянный множитель нагрузки: статический эквивалент для Ф26
                nu_mul2=-1.0,        # множитель во ВТОРОЙ фазе; <= 0 — фазы нет
                t_mul=-1.0,          # сутки смены множителя; < 0 — фазы нет
                a_n=0.0,             # глубина потери ёмкости защиты; 0 — петли нет (прежняя модель)
                taun_dn=1.0,         # постоянная ПОТЕРИ ёмкости, сутки
                taun_up=30.0,        # постоянная её ВОЗВРАТА, сутки (медленнее — это и есть защёлка)
                n0_mul=1.0,          # постоянное снижение ёмкости: статический эквивалент для Ф28
                q=None, seed=None)


class Block:
    """Спецификация блока. Всё, что влияет на счёт, входит в хеш; что не влияет — нет."""

    def __init__(self, tag='', **kw):
        bad = set(kw) - set(GROUP) - set(ROW) - {'q', 'seed'}
        assert not bad, 'неизвестные поля блока: %s' % sorted(bad)
        for k_, v_ in list(kw.items()):                 # множитель 1,0 — это и есть «из слоёв»:
            if isinstance(v_, (list, tuple)) and len(v_) == 2 and float(v_[1]) == 1.0: kw[k_] = None

        self.tag = tag; self.kw = dict(kw)

    def filled(self, cfg):
        """Подставить значения по умолчанию из конфига там, где блок их не задал."""
        d = dict(DEFAULTS)
        for k, v in cfg.block_defaults().items(): d[k] = v
        d.update(self.kw)
        return d

    def key(self, cfg):
        d = self.filled(cfg)
        return hashlib.sha1(json.dumps({k: d[k] for k in sorted(d)}, sort_keys=True,
                                       default=float).encode()).hexdigest()[:10]

    def group(self, cfg):
        d = self.filled(cfg)
        return tuple(d[k] for k in GROUP)

    def __repr__(self): return 'Block(%s, %s)' % (self.tag, self.kw)


class Runner:
    """Собирает запросы, склеивает, считает, отдаёт результаты по хешу блока."""

    def __init__(self, sets, cfg, nproc=1, log=print, cache=True):
        self.S = sets; self.cfg = cfg; self.nproc = max(1, int(nproc)); self.log = log
        self.want = {}            # хеш -> Block
        self.res = {}             # хеш -> результат simulate для этого блока
        self.n_blocks = 0; self.t_calc = 0.0
        self.n_hit = 0; self.cache_dir = None; self.code_ver = code_version()
        if cache:
            self.cache_dir = os.path.join(cfg.d['cache']['dir'], 'blocks')
            os.makedirs(self.cache_dir, exist_ok=True)

    def _ckey(self, d, g, n0_block, row_vals):
        """Ключ кэша блока: всё, от чего зависит результат, включая версию кода модели."""
        S = self.S
        spec = dict(code=self.code_ver, sets=self.cfg.key(), nsel=int(S['nsel']),
                    group=[float(x) for x in g], seed=int(d['seed']), q=float(d['q']),
                    row={k: float(v) for k, v in row_vals.items()},
                    E0=float(S['E0']), qw=float(S['qw']), Gbar=float(S['Gbar']),
                    n0=hashlib.sha1(np.ascontiguousarray(n0_block, dtype=np.float64).tobytes()).hexdigest()[:16])
        return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]

    def _cpath(self, ck): return os.path.join(self.cache_dir, 'b_%s.npz' % ck)

    def request(self, block):
        k = block.key(self.cfg)
        self.want.setdefault(k, block)
        return k

    def run_all(self):
        if not self.want: return
        groups = {}
        for k, b in self.want.items(): groups.setdefault(b.group(self.cfg), []).append((k, b))
        for g, items in sorted(groups.items(), key=lambda z: str(z[0])):
            self._run_group(g, items)

    def _resolve(self, d):
        """Построчные поля блока с подставленными слоями: ровно то, что уйдёт в simulate."""
        S = self.S; out = {}
        for f in ROW:
            v = d[f]
            out[f] = (S[FROM_SETS[f]] if v is None else
                      (S[v[0]] * float(v[1]) if isinstance(v, (list, tuple)) else v))
        return out

    def _run_group(self, g, items):
        S = self.S; nsel = S['nsel']
        ndays, M, J, warm = g
        # ---- кэш: что уже посчитано раньше, то не считаем вовсе
        keep = []
        for k, b in items:
            d = b.filled(self.cfg)
            n0b = S['lo'] * (S['hi'] / S['lo']) ** d['q']
            ck = self._ckey(d, g, n0b, self._resolve(d)) if self.cache_dir else None
            if ck is not None and os.path.exists(self._cpath(ck)):
                try:
                    self.res[k] = _load_block(self._cpath(ck)); self.n_hit += 1; continue
                except Exception as e:
                    self.log('  кэш блока %s не читается (%s), считаю заново' % (k, type(e).__name__))
            keep.append((k, b, ck))
        if keep and self.n_hit:
            self.log('  из кэша взято блоков: %d, считать надо %d' % (self.n_hit, len(keep)))
        if not keep:
            self.log('батч: все %d блоков взяты из кэша (суток %d, шагов %d, J %d)' % (len(items), ndays, M, J))
            return
        cks = [z[2] for z in keep]; items = [(z[0], z[1]) for z in keep]; NB = len(items)
        dd = [b.filled(self.cfg) for _, b in items]
        kw = {}
        for f in ROW:
            # None — значит «взять из слоёв»; ('A_H', 0.5) — «взять из слоёв и умножить»
            # (нужно переборам, которые задаются множителем при посчитанной величине, как Ф2(б)).
            v = [(S[FROM_SETS[f]] if d[f] is None else
                  (S[d[f][0]] * float(d[f][1]) if isinstance(d[f], (list, tuple)) else d[f])) for d in dd]
            kw[f] = np.repeat(np.asarray(v, float), nsel) if len(set(map(float, v))) > 1 else float(v[0])
        n0 = np.concatenate([(S['lo'] * (S['hi'] / S['lo']) ** d['q']) for d in dd])
        seeds = [int(d['seed']) for d in dd]
        args = dict(E0=S['E0'], qw=S['qw'], Gbar=S['Gbar'], hiw=S['hi'])
        t0 = time.time()
        self.log('батч: %d блоков по %d наборов, суток %d, шагов %d, J %d, разгон %d, процессов %d'
                 % (NB, nsel, ndays, M, J, warm, min(self.nproc, NB)))
        npr = max(1, min(self.nproc, NB)); jobs = []
        for ix in np.array_split(np.arange(NB), npr):
            nb_ = len(ix); rws = (ix[:, None] * nsel + np.arange(nsel)).ravel()
            kk = {f: (v[rws] if isinstance(v, np.ndarray) else v) for f, v in kw.items()}
            kk.update(E0=args['E0'], qw=args['qw'], Gbar=args['Gbar'], ndays=ndays, M=M, J=J,
                      warm=warm, hiw=np.tile(args['hiw'], nb_))   # верх окна набора: нужен Ф10
            jobs.append(({k_: np.tile(v, nb_) for k_, v in S['P'].items()}, n0[rws],
                         np.tile(S['thr'], nb_), np.tile(S['xhi'], nb_), np.tile(S['xlo'], nb_),
                         kk, nsel, [seeds[i] for i in ix]))
        # Прогрев ядра numba ОДИН раз, в родительском процессе и до ветвления. Подробности —
        # в докстроке px.fast.warmup: без этого шестнадцать процессов компилируют наперегонки.
        if not getattr(self, '_warm', False):
            self._warm = True
            from . import fast
            if fast.enabled():
                ok, el = fast.warm_subprocess()
                if not ok: os.environ['REDOX_FAST'] = '0'   # иначе дети пойдут на ядро, которого нет
                self.log('  слитое ядро numba: %s, потоков %s'
                         % ('прогрето за %.0f с' % el if ok else 'НЕ ПРОГРЕЛОСЬ, иду на numpy',
                            os.environ.get('NUMBA_NUM_THREADS', '?')))
            else:
                self.log('  слитое ядро выключено, иду на numpy')
        if len(jobs) == 1: out = _job(jobs[0])
        else:
            try:
                with mp.Pool(len(jobs)) as pool: out = [z for part in pool.map(_job, jobs) for z in part]
            except Exception as e:
                self.log('  процессы недоступны (%s), считаю в одном' % type(e).__name__)
                out = [z for j in jobs for z in _job(j)]
        assert len(out) == NB
        el = time.time() - t0; self.t_calc += el; self.n_blocks += NB
        self.log('  готово за %.0f с (%.1f с на блок)' % (el, el / NB))
        for (k, _), ck, r in zip(items, cks, out):
            self.res[k] = r
            if ck is not None:
                try: _save_block(self._cpath(ck), r)
                except Exception as e: self.log('  не удалось положить блок %s в кэш (%s)' % (k, type(e).__name__))

    def get(self, key): return self.res[key]
