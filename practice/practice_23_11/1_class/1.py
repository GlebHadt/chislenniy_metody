import math
import numpy as np
import matplotlib.pyplot as plt


def f(x: float) -> float:
    return x**3 - 6 * x**2 + 11 * x - 6


def bisection(func, a: float, b: float, eps: float = 1e-9):
    if func(a) * func(b) >= 0:
        raise ValueError("На концах отрезка функция должна иметь разные знаки.")

    log = []
    step = 0
    while (b - a) > eps:
        step += 1
        c = 0.5 * (a + b)
        fc = func(c)
        log.append((step, a, b, c, fc, b - a))

        if fc == 0.0:
            break
        elif func(a) * fc < 0:
            b = c
        else:
            a = c

    root_approx = 0.5 * (a + b)
    return root_approx, step, log


# 1. График функции на [0, 4]
x_vals = np.linspace(0, 4, 400)
plt.figure(figsize=(8, 4.5))
plt.plot(x_vals, f(x_vals), label=r"$f(x) = x^3 - 6x^2 + 11x - 6$")
plt.axhline(0, color="black", linewidth=1)
plt.scatter([1, 2, 3], [0, 0, 0], color="red", zorder=5, label="Корни: 1, 2, 3")
plt.title("График полинома f(x) на отрезке [0, 4]")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("polynomial_roots.png", dpi=150)
plt.show()

# 2-5. Поиск корней, печать журналов и сводной таблицы
tasks = [
    (1.0, 0.4, 1.4),
    (2.0, 1.7, 2.2),
    (3.0, 2.3, 3.5),
]
eps = 1e-9
summary = []

for idx, (exact_root, a0, b0) in enumerate(tasks, start=1):
    delta_0 = b0 - a0
    approx, n_iter, journal = bisection(f, a0, b0, eps)
    abs_err = abs(approx - exact_root)
    rel_err = abs_err / abs(exact_root)
    theor_iter = math.log2(delta_0 / eps)

    print(f"\n=== Журнал для корня x_{idx} = {exact_root} (отрезок [{a0}, {b0}], Δ0 = {delta_0}) ===")
    print(f"{'Итер.':>5} | {'a':>16} | {'b':>16} | {'c':>16} | {'f(c)':>14} | {'b - a':>12}")
    print("-" * 92)
    for step, a, b, c, fc, width in journal:
        print(f"{step:5d} | {a:16.11f} | {b:16.11f} | {c:16.11f} | {fc:14.6e} | {width:12.6e}")

    summary.append((exact_root, approx, n_iter, theor_iter, abs_err, rel_err))

print("\n=== Итоговая таблица: корень -> число итераций -> Δ -> δ ===")
print(f"{'Корень':>8} | {'Численно x*':>16} | {'N факт':>6} | {'log2(Δ0/ε)':>11} | {'Δ (абс.)':>12} | {'δ (отн.)':>12}")
print("-" * 78)
for exact, approx, n_iter, theor, d_abs, d_rel in summary:
    print(f"{exact:8.1f} | {approx:16.12f} | {n_iter:6d} | {theor:11.4f} | {d_abs:12.4e} | {d_rel:12.4e}")
