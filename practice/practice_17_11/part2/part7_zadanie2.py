import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Задача 2: Композиция преобразований и обратное преобразование
# ============================================================

# ── 1. Исходный набор точек: 200 случайных точек ───────────
rng = np.random.default_rng(42)
points = rng.uniform(-2, 2, size=(200, 2))   # квадрат [-2, 2] × [-2, 2]

# Однородные координаты: добавляем третью координату 1 → форма (200, 3)
points_h = np.hstack([points, np.ones((len(points), 1))])


# ── 2. Матрицы преобразований 3×3 ──────────────────────────
def scale_matrix(sx, sy):
    return np.array([[sx, 0.0, 0.0],
                     [0.0, sy, 0.0],
                     [0.0, 0.0, 1.0]])


def rotate_matrix(deg):
    theta = np.deg2rad(deg)
    return np.array([[np.cos(theta), -np.sin(theta), 0.0],
                     [np.sin(theta),  np.cos(theta), 0.0],
                     [0.0,            0.0,           1.0]])


def translate_matrix(tx, ty):
    return np.array([[1.0, 0.0, tx],
                     [0.0, 1.0, ty],
                     [0.0, 0.0, 1.0]])


S = scale_matrix(1.5, 0.5)
R = rotate_matrix(60)
T = translate_matrix(1, -1)

# Композиция: сначала масштаб, потом поворот, потом перенос
# Порядок: (T @ R @ S) — применяется справа налево
combined = T @ R @ S


# ── 3. Прямое преобразование ───────────────────────────────
# Точки записаны строками → умножаем на M.T
transformed_h = points_h @ combined.T
transformed = transformed_h[:, :2]   # отбрасываем третью координату


# ── 4. Обратное преобразование (в обратном порядке) ────────
S_inv = scale_matrix(1 / 1.5, 1 / 0.5)
R_inv = rotate_matrix(-60)
T_inv = translate_matrix(-1, 1)

# Сначала отменяем перенос, потом поворот, потом масштаб
inverse = S_inv @ R_inv @ T_inv

restored = (transformed_h @ inverse.T)[:, :2]

ok = np.allclose(restored, points, atol=1e-12)
print("Восстановленные == исходные (atol=1e-12):", ok)
print("Максимальная ошибка:", np.max(np.abs(restored - points)))

# Проверка: inverse совпадает с обратной матрицей numpy
print("inverse == np.linalg.inv(combined):", np.allclose(inverse, np.linalg.inv(combined)))


# ── 5. Что будет при прямом порядке (ошибка) ───────────────
wrong_inverse = T_inv @ R_inv @ S_inv   # сначала масштаб, потом поворот, потом перенос
wrong = (transformed_h @ wrong_inverse.T)[:, :2]
print("\nНеправильный порядок → совпадают?", np.allclose(wrong, points, atol=1e-12))
print("Максимальная ошибка:", round(np.max(np.abs(wrong - points)), 3))


# ── 6. Визуализация ────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Левый график: правильный порядок ---
ax1.scatter(points[:, 0], points[:, 1], s=40, color='gray',
            alpha=0.5, label='исходные')
ax1.scatter(transformed[:, 0], transformed[:, 1], s=15, color='crimson',
            label='преобразованные')
ax1.scatter(restored[:, 0], restored[:, 1], s=8, color='steelblue',
            marker='x', label='восстановленные')
ax1.set_title('Исходные / преобразованные / восстановленные',
              fontsize=12, fontweight='bold')

# --- Правый график: неправильный порядок ---
ax2.scatter(points[:, 0], points[:, 1], s=40, color='gray',
            alpha=0.5, label='исходные')
ax2.scatter(wrong[:, 0], wrong[:, 1], s=15, color='darkorange',
            label='«восстановленные» в прямом порядке')
ax2.set_title('Ошибка: обратные шаги в прямом порядке',
              fontsize=12, fontweight='bold')

for ax in (ax1, ax2):
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)

plt.tight_layout()
fig.savefig("task2_inverse.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 7. Ответ на вопрос ─────────────────────────────────────
print("""
Почему обратные преобразования применяют в обратном порядке?

1. Прямая композиция: M = T · R · S (сначала S, потом R, потом T).
   Обратная матрица: M⁻¹ = (T · R · S)⁻¹ = S⁻¹ · R⁻¹ · T⁻¹.
   То есть сначала отменяется последний шаг (перенос), а в конце первый (масштаб).

2. Аналогия: сначала надели носки, потом ботинки. Снимать
   приходится в обратном порядке: сначала ботинки, потом носки.

3. Если применить обратные шаги в прямом порядке (T⁻¹ · R⁻¹ · S⁻¹),
   точки не вернутся на место: матрицы не коммутируют (A·B ≠ B·A).
   Например, обратный масштаб применится к уже повёрнутому облаку,
   то есть растянет его не вдоль тех осей. Правый график это
   показывает: облако искажено и сдвинуто.
""")
