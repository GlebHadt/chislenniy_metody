"""
Задача 1. Все корни полинома f(x) = x³ − 6x² + 11x − 6 на отрезке [0, 4].
Запуск: python3 task1.py  →  журналы в консоль и task1_output.txt, график task1_plot.png
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bisection(f, a, b, eps=1e-9, max_iter=200):
    """Метод половинного деления. Возвращает (корень, журнал, статус)."""
    fa, fb = f(a), f(b)
    if fa == 0:
        return a, [], "f_zero"
    if fb == 0:
        return b, [], "f_zero"
    if fa * fb > 0:
        raise ValueError(f"Нет смены знака на [{a}, {b}]: f(a)={fa}, f(b)={fb}")
    journal = []
    for k in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        length = b - a
        journal.append({"iter": k, "a": a, "b": b, "c": c, "f(c)": fc, "length": length})
        if fc == 0:
            return c, journal, "f_zero"
        if length < eps:
            return c, journal, "ok"
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return c, journal, "max_iter"


def print_journal(journal, title):
    print(f"\n=== {title} ===")
    print(f"{'k':>3} | {'a':>14} | {'b':>14} | {'c':>14} | {'f(c)':>11} | {'b-a':>9}")
    print("-" * 82)
    for r in journal:
        print(f"{r['iter']:>3} | {r['a']:>14.10f} | {r['b']:>14.10f} | "
              f"{r['c']:>14.10f} | {r['f(c)']:>11.3e} | {r['length']:>9.2e}")


def f(x):
    return x**3 - 6*x**2 + 11*x - 6


def main():
    eps = 1e-9

    # ── 1. Табуляция и график ─────────────────────────────────
    print("Табуляция f(x) на [0, 4] с шагом 0.1 (показаны точки смены знака):")
    xs = np.round(np.arange(0, 4.0001, 0.1), 10)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        if f(x0) * f(x1) <= 0:
            print(f"  f({x0:.1f}) = {f(x0):+.4f},  f({x1:.1f}) = {f(x1):+.4f}")
    # Интервалы со сменой знака, выбранные по графику.
    # Длины разные (1.5; 0.9; 1.5), и ни в одной середине корень не лежит точно.
    intervals = [(0.0, 1.5), (1.6, 2.5), (2.5, 4.0)]
    for a, b in intervals:
        print(f"  [{a}, {b}]: f(a) = {f(a):+.4f}, f(b) = {f(b):+.4f} → смена знака")

    xx = np.linspace(0, 4, 400)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xx, f(xx), lw=2, label="f(x) = x³ − 6x² + 11x − 6")
    ax.axhline(0, color="k", lw=0.8)
    for (a, b), col in zip(intervals, ["#e4572e", "#29a19c", "#7768ae"]):
        ax.axvspan(a, b, color=col, alpha=0.13)
        ax.text((a + b) / 2, 5.3, f"[{a}, {b}]", ha="center", color=col)
    ax.plot([1, 2, 3], [0, 0, 0], "ko", ms=6, label="корни")
    ax.set_ylim(-6.5, 6.5)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.set_title("Задача 1: график f(x) и интервалы, содержащие корни")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig("task1_plot.png", dpi=150); plt.close(fig)

    # ── 2–3. Бисекция и три журнала ───────────────────────────
    exact = [1.0, 2.0, 3.0]   # 5. f(x) = (x − 1)(x − 2)(x − 3)
    rows = []
    for (a, b), x_star in zip(intervals, exact):
        root, j, st = bisection(f, a, b, eps=eps)
        print_journal(j, f"Корень на [{a}, {b}], ε = 1e-9")
        print(f"корень = {root:.12f}, f(корень) = {f(root):.3e}, "
              f"статус: {st}, итераций: {len(j)}")
        d_abs = abs(root - x_star)
        n_th = math.log2((b - a) / eps)
        rows.append((a, b, x_star, root, len(j), n_th, d_abs, d_abs / abs(x_star)))

    # ── 4–5. Сводная таблица ──────────────────────────────────
    print("\nАналитически: x³ − 6x² + 11x − 6 = (x − 1)(x − 2)(x − 3) → x* = 1, 2, 3")
    print("\nСводная таблица (Δ = |x − x*|, δ = Δ / |x*|):")
    print(f"{'интервал':>11} | {'Δ0':>4} | {'x*':>2} | {'корень':>15} | {'n':>3} | "
          f"{'log2(Δ0/ε)':>10} | {'Δ':>9} | {'δ':>9}")
    print("-" * 86)
    for a, b, x_star, r, n, n_th, da, dr in rows:
        print(f"[{a}, {b}] | {b - a:>4.1f} | {x_star:>2.0f} | {r:>15.12f} | {n:>3} | "
              f"{n_th:>10.2f} | {da:>9.2e} | {dr:>9.2e}")

    # Пояснение к расхождению с оценкой на 1–2 итерации
    print("\nПроверка формулы: длина на итерации k равна Δ0 / 2^(k−1); остановка, когда она < ε,")
    print("→ n = floor(log2(Δ0/ε)) + 2:",
          ", ".join(f"{math.floor(r[5]) + 2}" for r in rows), "— совпадает с фактом.")

    # Отдельно: что будет на «естественном» интервале [1.5, 2.5]
    root, j, st = bisection(f, 1.5, 2.5, eps=eps)
    print(f"\nЗамечание: на [1.5, 2.5] середина c = 2 — точный корень, "
          f"метод останавливается за {len(j)} итерацию (статус {st}).")


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, t): [x.write(t) for x in self.s]
    def flush(self): [x.flush() for x in self.s]


if __name__ == "__main__":
    with open("task1_output.txt", "w", encoding="utf-8") as out:
        sys.stdout = Tee(sys.__stdout__, out)
        try:
            main()
        finally:
            sys.stdout = sys.__stdout__