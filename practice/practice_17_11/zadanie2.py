import timeit
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Задача 2: Три способа создать список квадратов
# ============================================================

# --- Три фрагмента кода ---
methods = {
    "Цикл с append":      "out = []\nfor i in range(N):\n    out.append(i*i)",
    "List comprehension": "out = [i*i for i in range(N)]",
    "map + lambda":       "out = list(map(lambda i: i*i, range(N)))",
}

N_values = [100_000, 1_000_000]

results_time = {name: [] for name in methods}
results_mem = {name: [] for name in methods}

print("Запуск бенчмарков...\n")
for N in N_values:
    print(f"Размер N={N}:")
    for name, stmt in methods.items():
        # --- Замер времени ---
        times = timeit.repeat(stmt, globals={"N": N}, number=1, repeat=5)
        best = min(times) * 1000  # мс

        # --- Замер памяти ---
        tracemalloc.start()
        exec(stmt, {"N": N})
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1024 / 1024  # МБ

        results_time[name].append(best)
        results_mem[name].append(peak_mb)
        print(f"  {name:20s}  время={best:8.2f} мс  память={peak_mb:6.2f} МБ")
    print()

# ============================================================
# ВИЗУАЛИЗАЦИЯ: два графика рядом
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- Левый график: время (линии с маркерами) ---
for name in methods:
    ax1.plot(N_values, results_time[name], "o-", label=name, linewidth=2)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xticks(N_values)
ax1.set_xticklabels([f"N={N}" for N in N_values])
ax1.set_xlabel("Размер списка (N)", fontsize=12)
ax1.set_ylabel("Время (мс, лог. шкала)", fontsize=12)
ax1.set_title("Время выполнения", fontsize=14, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, which="both", alpha=0.2)

# --- Правый график: память (группированные столбцы) ---
x = np.arange(len(N_values))
width = 0.25
for i, name in enumerate(methods):
    ax2.bar(x + (i - 1) * width, results_mem[name], width, label=name, alpha=0.85)
ax2.set_xticks(list(x))
ax2.set_xticklabels([f"N={N}" for N in N_values])
ax2.set_xlabel("Размер списка (N)", fontsize=12)
ax2.set_ylabel("Пиковая память (МБ)", fontsize=12)
ax2.set_title("Потребление памяти", fontsize=14, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, axis="y", alpha=0.3)  # только горизонтальная сетка

plt.tight_layout()
fig.savefig("squares_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("График сохранён: squares_comparison.png")

# ============================================================
# Вывод
# ============================================================
print("\n" + "=" * 60)
print("ВЫВОД:")
print("=" * 60)
print("""
а) Память: все три способа требуют примерно одинаково,
   потому что результат один и тот же — список из N чисел.
   Промежуточные объекты (вызовы append, вызовы lambda)
   живут недолго и почти не влияют на пик.

б) Время: list comprehension быстрее всех — элемент добавляется
   специальной быстрой инструкцией, без поиска и вызова метода
   append на каждом шаге.

в) map + lambda медленнее: на каждый элемент вызывается
   Python-функция (lambda), а это дорогая операция.

Лучший способ — list comprehension: та же память,
но меньше времени и самый читаемый код.
""")
