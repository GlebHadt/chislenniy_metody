import timeit
import matplotlib.pyplot as plt

# ============================================================
# Задача 1: x in list vs x in set
# ============================================================

sizes = [100, 1000, 10000]
number = 1000   # 1000 проверок в одной серии
repeat = 5      # 5 серий, берём лучшую
target = -1     # числа нет в коллекции — худший случай для list

time_list = []
time_set = []

print("Запуск бенчмарков...\n")
for n in sizes:
    # --- Подготовка данных ---
    data_list = list(range(n))
    data_set = set(range(n))

    # --- Замер: список ---
    times1 = timeit.repeat("target in data_list",
                           globals={"target": target, "data_list": data_list},
                           number=number, repeat=repeat)

    # --- Замер: множество ---
    times2 = timeit.repeat("target in data_set",
                           globals={"target": target, "data_set": data_set},
                           number=number, repeat=repeat)

    # Лучший замер -> время одной проверки в микросекундах
    best1 = min(times1) / number * 1_000_000
    best2 = min(times2) / number * 1_000_000

    time_list.append(best1)
    time_set.append(best2)

    print(f"Размер n={n}:")
    print(f"  list: {best1:8.4f} мкс на проверку")
    print(f"  set:  {best2:8.4f} мкс на проверку")
    print(f"  set быстрее в {best1 / best2:.1f} раз\n")

# ============================================================
# ВИЗУАЛИЗАЦИЯ С MATPLOTLIB
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(sizes, time_list, "ro-", label="x in list (O(n))", linewidth=2)
ax.plot(sizes, time_set, "bs-", label="x in set (O(1))", linewidth=2)

# Логарифм по Y — видно, что list растёт, а set стоит на месте
ax.set_yscale("log")

ax.set_xlabel("Размер коллекции (n)", fontsize=12)
ax.set_ylabel("Время одной проверки (мкс, лог. шкала)", fontsize=12)
ax.set_title("Проверка вхождения: list vs set", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.2)  # сетка и для мелких делений

plt.tight_layout()
fig.savefig("membership_benchmark.png", dpi=150, bbox_inches="tight")
plt.show()

print("График сохранён: membership_benchmark.png")
