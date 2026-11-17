import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Задача 1: Поворот и отражение правильного шестиугольника
# ============================================================

# ── 1. Исходный набор точек: шестиугольник ─────────────────
# Вершины: x_k = cos(2πk/6), y_k = sin(2πk/6), k = 0..5
k = np.arange(6)
hexagon = np.column_stack([np.cos(2 * np.pi * k / 6),
                           np.sin(2 * np.pi * k / 6)])
hexagon = np.vstack([hexagon, hexagon[0]])   # замыкаем контур
print("Форма массива точек:", hexagon.shape)  # (7, 2)


# ── 2. Матрицы преобразований ──────────────────────────────
theta = np.deg2rad(45)   # поворот на 45 градусов

rotate = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)],
])

reflect_yx = np.array([   # отражение относительно прямой y = x
    [0.0, 1.0],
    [1.0, 0.0],
])


# ── 3. Применяем преобразования ────────────────────────────
# Точки записаны строками → умножаем на T.T
rotated   = hexagon @ rotate.T
reflected = hexagon @ reflect_yx.T


# ── 4. Визуализация ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))

ax.plot(hexagon[:, 0], hexagon[:, 1], 'o-', color='gray',
        label='исходный', lw=2, markersize=7, alpha=0.6)
ax.plot(rotated[:, 0], rotated[:, 1], 'o-', color='steelblue',
        label='поворот на 45°', lw=2, markersize=7)
ax.plot(reflected[:, 0], reflected[:, 1], 'o-', color='crimson',
        label='отражение относительно y = x', lw=2, markersize=7)

# Отмечаем вершину k=0 у каждой фигуры — видно, куда она «уехала»
for pts, color in [(hexagon, 'gray'), (rotated, 'steelblue'), (reflected, 'crimson')]:
    ax.plot(pts[0, 0], pts[0, 1], '*', color=color, markersize=16)

# Прямая y = x — ось отражения
t = np.linspace(-1.3, 1.3, 2)
ax.plot(t, t, '--', color='black', lw=0.8, alpha=0.5, label='прямая y = x')

ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=9)
ax.set_title('Шестиугольник: поворот и отражение (★ — вершина k=0)',
             fontsize=13, fontweight='bold')

plt.tight_layout()
fig.savefig("task1_hexagon.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 5. Ответ на вопрос ─────────────────────────────────────
rotate_90 = np.array([[0.0, -1.0],
                      [1.0,  0.0]])

print("\nПоворот на 90°:\n", rotate_90)
print("Отражение относительно y = x:\n", reflect_yx)
print("Определитель поворота:  ", round(np.linalg.det(rotate_90), 3))
print("Определитель отражения: ", round(np.linalg.det(reflect_yx), 3))

print("""
Почему отражённый шестиугольник выглядит «повёрнутым»?

1. Матрицы поворота на 90° ( 0 -1 ; 1 0 ) и отражения относительно
   y = x ( 0 1 ; 1 0 ) отличаются только знаком одного элемента.
   Отражение относительно y = x = сначала отражение по оси X,
   потом поворот на 90°.

2. Правильный шестиугольник симметричен относительно оси X,
   поэтому первый шаг (отражение по X) форму не меняет — остаётся
   только поворот на 90°. А поворот на 90° для шестиугольника
   выглядит так же, как поворот на 30° (90° = 30° + 60°),
   вершины встают на углы 30°, 90°, 150°, ...

3. Отличие всё же есть: у отражения определитель -1, у поворота +1.
   Отражение меняет порядок обхода вершин: исходные идут против
   часовой стрелки, отражённые — по часовой. Это видно по звёздочке
   (вершина k=0). Поворот так никогда не делает.
""")
