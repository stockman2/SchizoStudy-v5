# Редокс-модель PV-интернейронов: как устроен проект и как его гонять

> **English version of this description:** [see below](#english-description-of-the-project).

## Запуск

```bash
cd ~/redox
python run.py --list                       # какие правила есть
python run.py --rules all --dry            # СКОЛЬКО БЛОКОВ И СКОЛЬКО ВРЕМЕНИ, без счёта
python run.py --rules f19                  # посчитать только Ф19
python run.py --rules f02,f06,f18          # несколько правил
python run.py --rules all --sets 300       # все правила, но на прежней выборке
python run.py --ledger                     # журнал вердиктов, накопленный за все прогоны
python tests/test_regression.py            # побитовая сверка с mass_core6.py
```

Считаются ТОЛЬКО блоки, нужные выбранным правилам. Одинаковые блоки разных правил склеиваются и
считаются один раз: на `--rules all` из 78 запрошенных блоков считается 40.

**Сначала `--dry`.** При `nsel = 2000` прогон «все правила» идёт около трёх часов, а не минут:
время растёт прямо пропорционально числу наборов, числу суток и числу клеток J.

## Файлы

| файл | что внутри |
|---|---|
| `run.py` | точка входа: разбор ключей, самопроверка критериев, сборка блоков, отчёт, журнал |
| `config/base.toml` | все параметры прогона. Менять надо ЗДЕСЬ, а не в коде |
| `px/consts.py` `core.py` `cortex.py` `sleep.py` `model.py` | модель. Перенесены из `mass_core6.py` ДОСЛОВНО; правка допустима только вместе с регрессионным тестом |
| `px/conf.py` | чтение конфига, выборка наборов с кэшем, слои сна и коры, закон разрывов и полоса |
| `px/batch.py` | блоки, их склейка по хешу, батчи, распараллеливание по блокам |
| `px/stats.py` | критерии: знаковый, граница медианы, CV² промежутков, загрузка, МОЩНОСТЬ и вердикт «непроверяемо» |
| `px/predict.py` | АНАЛИТИЧЕСКИЕ ожидания до счёта: одна модель опасности на все правила |
| `px/report.py` | отчёт прогона и журнал вердиктов (`ledger.json`) |
| `px/rules/_blocks.py` | общие наборы блоков. Один и тот же блок разные правила обязаны просить ЧЕРЕЗ ОДНУ функцию |
| `px/rules/_meas.py` | сводка по блоку (`meas`) и кривые предвестника |
| `px/rules/fNN.py` | правило: `needs(cfg)` — какие блоки нужны, `run(ctx)` — числа и отчёт |
| `tests/test_regression.py` | четыре сверки с монолитом, в том числе ПОБИТОВАЯ |

### Правила со своим счётом (блоков не просят, в смету `--dry` не входят)

| правило | что считает | цена |
|---|---|---|
| `f10` | регистратор КАЖДОГО эпизода: хвост, пик повреждения, время над границей. Копия шага сверяется с `px.model.simulate` побитово на тех же строках | ~7 мин при `M = 1600`, свой кэш `cache/f10_*.npz` |
| `f20` | каскад удвоений периода детерминированного переключателя, без клеток и редокса | секунды—минуты, кэш не нужен |
| `mass_core6.py` | монолит. Заморожен как эталон, в счёте не участвует |

## Как добавить правило

1. `px/rules/f20.py` с двумя функциями: `needs(cfg)` возвращает список `Block`, `run(ctx)` —
   `(словарь, список строк отчёта)`.
2. Блоки просить через `_blocks.py`. Нужен новый — добавить туда функцию, а не собирать на месте:
   иначе два правила молча получат разные блоки и парные сравнения между ними развалятся.
3. Каждую часть проводить через `stats.adjudicate` или `stats.sign_adjudicate`: они сами ставят
   «непроверяемо», когда мощности не хватает.
4. `ctx.verdict(...)` на каждую часть — иначе её не будет в журнале.

## Три правила, добытые дорого

1. **У критерия должно быть нулевое значение, свой шум и своя мощность.** Порог, отстоящий от
   нулевого значения меньше чем на MDE = 2,4865·SE, ничего не различает: вердикт — «непроверяемо»,
   и СИММЕТРИЧНО, то есть и там, где правило прошло бы.
2. **Направление знакового критерия задаётся обязательным аргументом.** 14.10 в Ф19(в) оно было
   перепутано молча: p = 0,989 вместо 0,038.
3. **Медиана по наборам от доли редких событий вырождена**: обнуляется, едва события есть меньше
   чем у половины наборов. Рядом всегда печатается доля по всем событиям.

## Что нельзя делать

- Менять вердикт после счёта. Негодность правила пишется в `note` рядом, вердикт остаётся.
  Исправление кода под УЖЕ ЗАПИСАННОЕ правило — не пересмотр.
- Резать батч по строкам между процессами: шум разыгрывается на длину блока, побитовая
  воспроизводимость сломается. Распараллеливание — только по блокам.
- Править `px/model.py`, `core.py`, `cortex.py`, `sleep.py`, `consts.py` без `tests/test_regression.py`.
  Отдельно помнить: их хеш входит в ключ кэша блоков, поэтому любая правка там обесценивает весь
  кэш — это правильно (старые блоки не подмешаются), но это часы счёта, и делать это надо осознанно,
  одним прогоном. Поправка сетки уязвимости из Ф22(в) — ровно такой случай.

---

## English description of the project

**What this is.** A mechanistic ODE model of oxidative damage to parvalbumin (PV)
interneurons, run as a pre-registered *in silico* refutation programme: every
hypothesis is a numbered rule, thresholds come from analytic expectations computed
before any simulation, and verdicts live in an append-only ledger. The science is
described in `latex/report.tex` (English) and `latex/report_ru.tex` (Russian); the
per-term provenance of the core equations is in `latex/core_derivation.tex`.

### Run

```bash
cd ~/redox
python run.py --list                       # available rules
python run.py --rules all --dry            # block count and time estimate, no compute
python run.py --rules f19                  # one rule; f02,f06,f18: several
python run.py --rules all --sets 300       # all rules on the previous sample
python run.py --ledger                     # accumulated verdict journal
python tests/test_regression.py            # bitwise check vs mass_core6.py
```

Only the blocks required by the chosen rules are computed. Identical blocks of
different rules are hash-deduplicated and computed once: `--rules all` requests 78
blocks, 40 are unique. **Run `--dry` first.** At `nsel = 2000` a full "all rules" run
takes about three hours: time grows linearly in the number of sets, days, and cells J.

### Files

| file | contents |
|---|---|
| `run.py` | entry point: CLI args, criteria self-check, block assembly, report, ledger |
| `config/base.toml` | all run parameters. Change them HERE, not in code |
| `px/consts.py` `core.py` `cortex.py` `sleep.py` `model.py` | the model, moved from `mass_core6.py` VERBATIM; edits only together with the regression test |
| `px/conf.py` | TOML config, cached set sampling, sleep/cortex layers, gap law, band |
| `px/batch.py` | blocks, hash dedup, per-block parallelism, block cache (keyed by model-source hash) |
| `px/stats.py` | criteria: sign test, median bound, CV² of gaps, load, POWER, "unfalsifiable" verdict |
| `px/predict.py` | analytic expectations before any compute: one risk model for all rules |
| `px/report.py` | run report and append-only `ledger.json` |
| `px/rules/_blocks.py` | shared block constructors; different rules must request the same block through one function |
| `px/rules/_meas.py` | per-block summary (`meas`) and precursor curves |
| `px/rules/fNN.py` | one rule per file: `needs(cfg)` — required blocks, `run(ctx)` — numbers and report |
| `tests/test_regression.py` | four checks vs the monolith, including a bitwise one |

### Rules with their own compute (request no blocks, absent from `--dry` estimates)

| rule | what it computes | cost |
|---|---|---|
| `f10` | recorder of EVERY episode: tail, damage peak, time above the window. Step copy verified bitwise vs `px.model.simulate` | ~7 min at `M = 1600`, own cache `cache/f10_*.npz` |
| `f20` | period-doubling cascade of the deterministic switch, no cells, no redox | seconds–minutes, no cache |
| `mass_core6.py` | the frozen monolith: reference only, not used in compute | |

### How to add a rule

1. `px/rules/f20.py` with two functions: `needs(cfg)` returns a list of `Block`s,
   `run(ctx)` returns `(dict, list of report lines)`.
2. Request blocks through `_blocks.py`. If a new block is needed, add a function there
   rather than assembling it in place: otherwise two rules silently get different
   blocks and pairwise comparisons between them fall apart.
3. Pass every part through `stats.adjudicate` or `stats.sign_adjudicate`: they return
   "unfalsifiable" on their own when power is insufficient.
4. `ctx.verdict(...)` for every part — otherwise it will not appear in the ledger.

### Pipeline discipline

- Only requested blocks are computed; identical blocks are deduplicated by hash
  (78 requested → 40 unique).
- Block results are cached under a key containing the model source hash: editing the
  model invalidates the cache by design.
- Parallelisation is per block only: noise is drawn for one block length and repeated
  over scan points, so blocks are strictly paired.
- Verdicts are never revised: a defect is annotated beside the verdict and the
  question is re-asked as a new rule on a fresh seed.

### Verification

- `tests/test_regression.py` requires set sampling, band arithmetic, and `simulate`
  to match the frozen monolith `mass_core6.py` bitwise on pinned configurations;
  cross-machine comparison uses a tolerance of 1e-12 with a machine fingerprint
  (measured worst 3e-15).
- The fused numba kernel is assembled from the reference source by textual
  substitution with uniqueness checks; it is verified on all 67 output fields over
  nine pinned configurations and by diffing all verdicts (3.3× wall-clock gain).

### Three rules learned the hard way

1. **A criterion needs a null value, its own noise, and its own power.** A threshold
   closer to the null value than the MDE = 2.4865·SE distinguishes nothing: the
   verdict is "unfalsifiable", SYMMETRICALLY — also where the rule would have passed.
2. **The direction of the sign test is a mandatory argument.** On 14.10, in F19(c) it
   was silently reversed: p = 0.989 instead of 0.038.
3. **The median over sets of a rare-event share is degenerate:** it collapses to zero
   as soon as fewer than half of the sets have any events. The share over all events
   is always printed next to it.

### What must not be done

- Changing a verdict after the fact. A rule's unfitness goes into a `note` beside the
  verdict; the verdict stays. Fixing code for an ALREADY RECORDED rule is not a
  review.
- Slicing a batch by rows across processes: the noise is drawn per block length, so
  bitwise reproducibility would break. Parallelise by blocks only.
- Editing `px/model.py`, `core.py`, `cortex.py`, `sleep.py`, `consts.py` without
  `tests/test_regression.py`. Remember that their hash enters the block-cache key, so
  any edit there invalidates the whole cache — correct (old blocks never mix in), but
  it costs hours of compute and should be done deliberately, in one run. Fixing the
  vulnerability grid from F22(c) is exactly such a case.
