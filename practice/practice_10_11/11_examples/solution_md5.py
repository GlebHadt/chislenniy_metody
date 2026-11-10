# Задача 1. Сумма массива: цикл vs a.sum() на n = 10^5, 10^6, 10^7

print('#1')

import time
import numpy as np

rng = np.random.default_rng(42)


def loop_sum(arr):
    total = 0.0
    for x in arr:
        total += x
    return total


def numpy_sum(arr):
    return arr.sum()


def measure(fn, x, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(x)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


print(f"{'n':>10} | {'цикл, мс':>10} | {'NumPy, мс':>10} | {'ускорение':>10} | совпадают")
print("-" * 62)
for n in (10**5, 10**6, 10**7):
    a = rng.normal(size=n)
    r_loop,  t_loop  = measure(loop_sum,  a)
    r_numpy, t_numpy = measure(numpy_sum, a)
    print(f"{n:>10} | {t_loop*1000:10.2f} | {t_numpy*1000:10.3f} | "
          f"{t_loop / t_numpy:9.1f}× | {np.isclose(r_loop, r_numpy)}")

# Читаемость: цикл — 4 строки (переменная-аккумулятор, for, +=, return),
# NumPy — одно выражение a.sum(), которое прямо говорит «сумма».

# ── Выводы ────────────────────────────────────────────────
# Пример запуска: n=1e5 → ~240×, n=1e6 → ~120×, n=1e7 → ~95×. Разница >10× есть уже
# при n = 1e5 (и даже раньше): цикл платит за каждый элемент интерпретатором и созданием
# объекта-скаляра, а a.sum() — один вызов C-кода с SIMD. На малых n ускорение выше, т.к.
# массив целиком в кеше процессора; на 1e7 (80 МБ) NumPy упирается в пропускную
# способность памяти, и отношение немного снижается. Читаемость: a.sum() однозначно лучше.

print('#2')

# Задача 2. Среднее и стандартное отклонение
import math
import time
import numpy as np

rng = np.random.default_rng(42)
a = rng.normal(size=1_000_000)


def loop_mean_std(arr):
    n = len(arr)
    s = 0.0
    s2 = 0.0
    for x in arr:
        s += x
        s2 += x * x
    mean = s / n
    var = s2 / n - mean * mean          # формула «в лоб»: E[x^2] - (E[x])^2
    return mean, math.sqrt(var)


def numpy_mean_std(arr):
    return arr.mean(), arr.std()


def measure(fn, x, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(x)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


r_loop,  t_loop  = measure(loop_mean_std,  a)
r_numpy, t_numpy = measure(numpy_mean_std, a)

print(f"цикл:   {t_loop*1000:10.2f} мс   mean = {r_loop[0]:.15f}  std = {r_loop[1]:.15f}")
print(f"NumPy:  {t_numpy*1000:10.2f} мс   mean = {r_numpy[0]:.15f}  std = {r_numpy[1]:.15f}")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"|Δmean| = {abs(r_loop[0] - r_numpy[0]):.2e},  |Δstd| = {abs(r_loop[1] - r_numpy[1]):.2e}")
print(f"совпадают до 1e-12: {np.allclose(r_loop, r_numpy, rtol=0, atol=1e-12)}")

# Демонстрация неустойчивости формулы E[x^2] - E[x]^2:
# сдвигаем данные на 1e8 — std не должно измениться (≈ 1), но формула «в лоб» ломается
b = a + 1e8
print("\nданные со сдвигом 1e8 (истинное std ≈ 1):")
n = len(b)
s = s2 = 0.0
for x in b:
    s += x
    s2 += x * x
var_naive = s2 / n - (s / n) ** 2
print(f"цикл (E[x²]-E[x]²): var = {var_naive:.6f}  ← должна быть ≈ 1, "
      f"{'отрицательная → sqrt невозможен' if var_naive < 0 else 'потеря точности'}")
print(f"NumPy  a.std():     std = {b.std():.6f}")

# ── Выводы ────────────────────────────────────────────────
# NumPy быстрее ~50×, результаты совпадают до 1e-12 (|Δstd| ~ 1e-14). Но формула «в лоб»
# E[x²] − E[x]² вычитает два близких больших числа (катастрофическое сокращение): при сдвиге
# данных на 1e8 дисперсия получается отрицательной. NumPy std считает в два прохода —
# сначала среднее, потом сумма (x − mean)², — а суммирование ведёт попарно (pairwise),
# поэтому ошибка округления растёт как O(log n), а не O(n). Читаемость: 2 вызова против 10 строк.

print('#3')

# Задача 3. Поэлементное преобразование y = sin(x) + x^2
import math
import sys
import time
import numpy as np

rng = np.random.default_rng(42)
a = rng.normal(size=1_000_000)


def loop_transform(arr):
    result = []
    for x in arr:
        result.append(math.sin(x) + x * x)
    return result


def numpy_transform(arr):
    return np.sin(arr) + arr**2


def measure(fn, x, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(x)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


r_loop,  t_loop  = measure(loop_transform,  a)
r_numpy, t_numpy = measure(numpy_transform, a)

print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")

# ── Память ────────────────────────────────────────────────
# список: сам список (массив указателей по 8 байт) + отдельный объект float (24 байта) на элемент
list_bytes = sys.getsizeof(r_loop) + sum(sys.getsizeof(v) for v in r_loop)
arr_bytes = r_numpy.nbytes
print(f"\nпамять результата: список ≈ {list_bytes/2**20:.1f} МБ, ndarray = {arr_bytes/2**20:.1f} МБ "
      f"(в {list_bytes/arr_bytes:.1f} раза меньше)")
# В np.sin(a) + a**2 создаётся 2 временных массива (np.sin(a) и a**2) + 1 итоговый:
# пиковая память ≈ 3 × 8 МБ. Можно сократить до одного буфера:
y = np.sin(a)
y += a * a          # временный a*a всё ещё есть; полностью без временных — np.square(a, out=tmp)
print(f"in-place вариант совпадает: {np.allclose(y, r_numpy)}")

# ── Выводы ────────────────────────────────────────────────
# Ускорение ~10× — меньше, чем у суммы, т.к. sin — дорогая операция и сама по себе,
# доля накладных расходов интерпретатора ниже. Список из 1e6 float занимает ~38 МБ
# (8 байт указатель + 24 байта объект float на элемент), ndarray — 8 МБ. Выражение
# np.sin(a) + a**2 создаёт 2 временных массива (np.sin(a), a**2) + 1 итоговый: пик ~24 МБ.
# Код NumPy — точная запись формулы, поэтому он короче и читается как математика.

print('#4')

# Задача 4. Условная замена (clipping): < 0 → 0, > 1 → 1
import time
import numpy as np

rng = np.random.default_rng(42)
a = rng.normal(size=1_000_000)


def loop_clip(arr):
    result = []
    for x in arr:
        if x < 0:
            result.append(0.0)
        elif x > 1:
            result.append(1.0)
        else:
            result.append(x)
    return result


def numpy_clip(arr):
    return np.clip(arr, 0, 1)


def measure(fn, x, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(x)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


r_loop,  t_loop  = measure(loop_clip,  a)
r_numpy, t_numpy = measure(numpy_clip, a)

print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")
changed = np.sum((a < 0) | (a > 1))
print(f"изменено элементов: {changed} из {a.size} ({changed / a.size:.1%})")
# Читаемость: цикл — 1 for + 2 if (if/elif) + else, 9 строк;
# NumPy — 0 циклов, 0 if, 1 строка.

# ── Выводы ────────────────────────────────────────────────
# np.clip быстрее ~70× и изменил ~66% элементов (для N(0,1): P(x<0)+P(x>1) ≈ 0.5+0.16).
# Цикл: 1 for + if/elif/else и 9 строк; NumPy: 0 циклов, 0 if, 1 строка.
# Ветвления в цикле ещё и плохо предсказываются процессором на случайных данных,
# а clip выполняется без ветвлений (min/max) — отсюда большое ускорение.

print('#5')

# Задача 5. Сумма по строкам матрицы (n, 10)
import time
import numpy as np

rng = np.random.default_rng(42)
m = rng.normal(size=(100_000, 10))


def loop_sum_rows(mat):
    result = [0.0] * mat.shape[0]
    for i in range(mat.shape[0]):
        s = 0.0
        for j in range(mat.shape[1]):
            s += mat[i, j]
        result[i] = s
    return result


def numpy_sum_rows(mat):
    return mat.sum(axis=1)


def measure(fn, x, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(x)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


r_loop,  t_loop  = measure(loop_sum_rows,  m)
r_numpy, t_numpy = measure(numpy_sum_rows, m)

print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")
print("SLOC тела функции: цикл = 7, NumPy = 1")

# ── Выводы ────────────────────────────────────────────────
# Ускорение ~55×, SLOC 7 против 1. m.sum(axis=1) читается как намерение — «сумма по
# строкам», а цикл — как механика: два индекса, аккумулятор, предвыделенный список,
# и надо проверить, что i и j не перепутаны. Поддерживать NumPy-вариант проще: меньше
# мест для ошибки, а смена оси — замена одного числа.

print('#6')

# Задача 6. Среднее по столбцам и нормировка (центрирование)
import time
import numpy as np

rng = np.random.default_rng(42)
m = rng.normal(loc=5.0, size=(100_000, 10))


def loop_center(mat):
    n, k = mat.shape
    means = [0.0] * k
    for j in range(k):
        s = 0.0
        for i in range(n):
            s += mat[i, j]
        means[j] = s / n
    result = [[0.0] * k for _ in range(n)]
    for i in range(n):
        for j in range(k):
            result[i][j] = mat[i, j] - means[j]
    return result


def numpy_center(mat):
    return mat - mat.mean(axis=0)       # broadcasting: (n,10) - (10,)


def measure(fn, x, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(x)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


r_loop,  t_loop  = measure(loop_center,  m)
r_numpy, t_numpy = measure(numpy_center, m)

print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")
print(f"макс |среднее столбца| после нормировки: {np.abs(r_numpy.mean(axis=0)).max():.2e}")
print("SLOC тела функции: цикл = 12 (4 цикла, индексы i, j), NumPy = 1 (0 индексов)")

# ── Выводы ────────────────────────────────────────────────
# Ускорение ~60×, средние после нормировки ~1e-13 ≈ 0 (проверка пройдена).
# Цикл: 12 строк, 4 цикла, индексы [i, j] и [i][j]; NumPy: 1 строка без индексов —
# broadcasting сам «растягивает» вектор средних (10,) на все n строк.
# Риск цикла — перепутать порядок осей при подсчёте среднего; в NumPy это axis=0.

print('#7')

# Задача 7. Поиск максимума и его индекса (+ поведение с NaN)
import time
import numpy as np

rng = np.random.default_rng(42)
a = rng.normal(size=1_000_000)


def loop_max(arr):
    max_val = arr[0]
    max_idx = 0
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_val, max_idx


def numpy_max(arr):
    return arr.max(), arr.argmax()


def measure(fn, x, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(x)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


r_loop,  t_loop  = measure(loop_max,  a)
r_numpy, t_numpy = measure(numpy_max, a)

print(f"цикл:   {t_loop*1000:10.2f} мс   max = {r_loop[0]:.6f}, idx = {r_loop[1]}")
print(f"NumPy:  {t_numpy*1000:10.2f} мс   max = {r_numpy[0]:.6f}, idx = {r_numpy[1]}")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {r_loop[0] == r_numpy[0] and r_loop[1] == r_numpy[1]}")

# ── Данные с NaN ──────────────────────────────────────────
b = a.copy()
b[0] = np.nan          # NaN в начале: любое сравнение x > nan даёт False
b[500] = np.nan
print("\nданные с NaN:")
print(f"цикл:        {loop_max(b)}   ← застрял на NaN, настоящий максимум не найден")
print(f"a.max():     {b.max()}      ← NaN «заражает» результат")
print(f"np.nanmax:   {np.nanmax(b):.6f}, np.nanargmax: {np.nanargmax(b)}  ← правильно")

# ── Выводы ────────────────────────────────────────────────
# max/argmax быстрее ~100×. С NaN цикл молча ломается: если NaN первый, любое
# сравнение x > nan — False, и «максимумом» остаётся NaN с индексом 0 (а если NaN в
# середине, он просто пропускается — поведение зависит от позиции, что ещё хуже).
# a.max() честно возвращает nan, а np.nanmax/np.nanargmax дают правильный ответ.
# Пропуски в реальных данных (сбой датчика, пустая ячейка) — норма, поэтому nan-функции —
# это защита от тихих ошибок, а не украшение.

print('#8')

# Задача 8. Фильтрация по порогу: x > 0
import sys
import time
import numpy as np

rng = np.random.default_rng(42)
a = rng.normal(size=1_000_000)


def loop_filter(arr):
    return [x for x in arr if x > 0]


def numpy_filter(arr):
    return arr[arr > 0]


def measure(fn, x, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(x)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


r_loop,  t_loop  = measure(loop_filter,  a)
r_numpy, t_numpy = measure(numpy_filter, a)

print(f"цикл:   {t_loop*1000:10.2f} мс   тип: {type(r_loop).__name__}, элементов: {len(r_loop)}")
print(f"NumPy:  {t_numpy*1000:10.2f} мс   тип: {type(r_numpy).__name__}, элементов: {r_numpy.size}")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")

# ── Память ────────────────────────────────────────────────
# элементы списка — это np.float64-скаляры (32 байта), у обычного float — 24 байта
list_bytes = sys.getsizeof(r_loop) + sum(sys.getsizeof(v) for v in r_loop)
list_py_floats = [float(x) for x in r_loop]
list_py_bytes = sys.getsizeof(list_py_floats) + sum(sys.getsizeof(v) for v in list_py_floats)
arr_bytes = r_numpy.nbytes
print(f"\nсписок np.float64: {list_bytes/2**20:6.1f} МБ  ({list_bytes/len(r_loop):.0f} байт/элемент)")
print(f"список float:      {list_py_bytes/2**20:6.1f} МБ  ({list_py_bytes/len(r_loop):.0f} байт/элемент)")
print(f"ndarray float64:   {arr_bytes/2**20:6.1f} МБ  ({arr_bytes/r_numpy.size:.0f} байт/элемент)")
print(f"ndarray компактнее в {list_py_bytes/arr_bytes:.1f}–{list_bytes/arr_bytes:.1f} раза")

# ── Выводы ────────────────────────────────────────────────
# Маска a[a > 0] быстрее ~10× (меньше, чем у агрегаций: результат заранее неизвестного
# размера, нужен проход для маски и копирование). Результат цикла — list, где каждый
# элемент отдельный объект (32–40 байт на элемент с указателем), а ndarray хранит 8 байт
# на элемент подряд — в 4–5 раз компактнее, и с ним дальше можно работать векторно.

print('#9')

# Задача 9. Замена по маске in-place: отрицательные → модуль
import time
import numpy as np

rng = np.random.default_rng(42)
a = rng.normal(size=1_000_000)


def loop_abs_inplace(arr):
    for i in range(len(arr)):
        if arr[i] < 0:
            arr[i] = -arr[i]
    return arr


def numpy_abs_inplace(arr):
    arr[arr < 0] *= -1          # альтернатива: np.abs(arr, out=arr)
    return arr


# Функции меняют массив, поэтому перед каждым прогоном даём свежую копию
# (копирование — вне замера времени).
def measure(fn, x, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        data = x.copy()
        t0 = time.perf_counter()
        result = fn(data)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


r_loop,  t_loop  = measure(loop_abs_inplace,  a)
r_numpy, t_numpy = measure(numpy_abs_inplace, a)

print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")
print(f"совпадают с np.abs(a): {np.allclose(r_numpy, np.abs(a))}")

# ── Проверка, что новый массив не создаётся ───────────────
for name, fn in (("цикл", loop_abs_inplace), ("NumPy", numpy_abs_inplace)):
    data = a.copy()
    out = fn(data)
    print(f"{name}: тот же объект = {out is data}, общая память = {np.shares_memory(out, data)}")

data = a.copy()
out = np.abs(data, out=data)
print(f"np.abs(a, out=a): тот же объект = {out is data}")

# ── Выводы ────────────────────────────────────────────────
# NumPy-вариант быстрее ~9× (a[a<0] *= -1 строит маску и делает выборку/запись по ней;
# np.abs(a, out=a) ещё быстрее — один проход). Оба варианта возвращают тот же объект,
# новый массив под результат не создаётся (у маски есть только временный bool-массив, 1 байт
# на элемент). Для больших данных это важно: массив на 8 ГБ при 12 ГБ RAM нельзя
# скопировать, а изменить на месте — можно.

print('#10')

# Задача 10. Комбинированная маска: -1 < x < 1
import time
import numpy as np

rng = np.random.default_rng(42)
a = rng.normal(size=1_000_000)


def loop_between(arr):
    return [x for x in arr if -1 < x < 1]


def numpy_between(arr):
    return arr[(arr > -1) & (arr < 1)]


def measure(fn, x, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(x)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


r_loop,  t_loop  = measure(loop_between,  a)
r_numpy, t_numpy = measure(numpy_between, a)

print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")
print(f"доля элементов в (-1, 1): {r_numpy.size / a.size:.1%}  (теория для N(0,1): 68.3%)")

# ── Почему нельзя без скобок ──────────────────────────────
# a > -1 & a < 1  разбирается как  a > (-1 & a) < 1, т.к. & приоритетнее сравнений,
# а цепочка сравнений превращается в (a > (-1 & a)) and ((-1 & a) < 1).
print("\nпопытки без скобок:")
for expr in ("a > -1 & a < 1", "-1 < a < 1", "(a > -1) and (a < 1)"):
    try:
        eval(expr)
        print(f"  {expr:24} → сработало (?)")
    except Exception as e:
        print(f"  {expr:24} → {type(e).__name__}: {str(e)[:70]}")

# ── Выводы ────────────────────────────────────────────────
# Ускорение ~16×, отобрано 68.3% элементов — совпадает с правилом «одной сигмы».
# В Python «-1 < x < 1» — самая читаемая запись, но для массивов она не работает:
# цепочка сравнений раскрывается в and, которому нужен один bool, а массив его не даёт.
# Без скобок «a > -1 & a < 1» ломается из-за приоритета: & связывает сильнее, чем > и <,
# получается a > (-1 & a) < 1, а побитовое & для float не определено → TypeError.
# Поэтому в NumPy каждое условие обязательно берётся в скобки и соединяется & / |.

print('#11')

# Задача 11. Матрица евклидовых расстояний между точками X (n,3) и Y (m,3)
import math
import time
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 3))
Y = rng.normal(size=(1000, 3))


def loop_dist(XY):
    Xl, Yl = XY[0].tolist(), XY[1].tolist()
    D = [[0.0] * len(Yl) for _ in range(len(Xl))]
    for i in range(len(Xl)):
        xi = Xl[i]
        for j in range(len(Yl)):
            yj = Yl[j]
            D[i][j] = math.sqrt((xi[0] - yj[0])**2 + (xi[1] - yj[1])**2 + (xi[2] - yj[2])**2)
    return D


def numpy_dist(XY):
    X, Y = XY
    return np.sqrt(((X[:, None, :] - Y[None, :, :])**2).sum(axis=2))


def numpy_dist_matmul(XY):
    # ||x - y||² = ||x||² + ||y||² - 2·x·y  → временные массивы только (n, m)
    X, Y = XY
    sq = (X**2).sum(axis=1)[:, None] + (Y**2).sum(axis=1)[None, :] - 2 * X @ Y.T
    return np.sqrt(np.maximum(sq, 0))          # max(…, 0) — защита от -1e-16


def measure(fn, x, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(x)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


r_loop,   t_loop   = measure(loop_dist,         (X, Y))
r_numpy,  t_numpy  = measure(numpy_dist,        (X, Y))
r_matmul, t_matmul = measure(numpy_dist_matmul, (X, Y))

print(f"цикл:              {t_loop*1000:10.2f} мс")
print(f"NumPy broadcasting:{t_numpy*1000:10.2f} мс   ускорение: {t_loop / t_numpy:.1f}×")
print(f"NumPy через X@Y.T: {t_matmul*1000:10.2f} мс   ускорение: {t_loop / t_matmul:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)} / {np.allclose(r_loop, r_matmul)}")

try:
    from scipy.spatial.distance import cdist
    r_cdist, t_cdist = measure(lambda XY: cdist(*XY), (X, Y))
    print(f"scipy cdist:       {t_cdist*1000:10.2f} мс   ускорение: {t_loop / t_cdist:.1f}×, "
          f"совпадает: {np.allclose(r_loop, r_cdist)}")
except ImportError:
    print("scipy не установлен — cdist пропущен")

# ── Память ────────────────────────────────────────────────
n, m = X.shape[0], Y.shape[0]
tmp_3d = n * m * 3 * 8
tmp_2d = n * m * 8
print(f"\nвременный массив X[:,None,:]-Y[None,:,:] формы ({n}, {m}, 3): {tmp_3d/2**20:.1f} МБ "
      f"(и ещё столько же на **2)")
print(f"вариант X@Y.T: временные массивы формы ({n}, {m}): {tmp_2d/2**20:.1f} МБ каждый")

# ── Выводы ────────────────────────────────────────────────
# Broadcasting быстрее цикла ~6×, но создаёт временный массив (1000, 1000, 3) ≈ 23 МБ,
# и ещё столько же на возведение в квадрат; при n = m = 10^4 это было бы ~2.2 ГБ.
# Формула ||x||² + ||y||² − 2·X@Y.T работает только с массивами (n, m) (~7.6 МБ) и использует
# BLAS — она и быстрее (~50×), и экономнее; минус — возможные крошечные отрицательные
# значения из-за округления, их обрезаем np.maximum(…, 0). scipy cdist — быстрее всех (~100×)
# и без больших временных массивов. Компромисс: «красивый» broadcasting удобен для небольших
# данных, а для больших нужна формула через матричное произведение или готовая функция.
