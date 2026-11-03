import math


def relative_error(approx, exact):
    abs_err = abs(approx - exact)
    rel_err = abs_err / exact
    return rel_err

# 1. Кейс с Пи
exact1 = math.pi
approx1 = 3.14
print("Кейс 1:")
print("Абсолютная:", abs(approx1 - exact1))
print("Относительная:", relative_error(approx1, exact1))

# 2. Кейс с 0.1 + 0.2
exact2 = 0.3
approx2 = 0.1 + 0.2
print("\nКейс 2:")
print("Абсолютная:", abs(approx2 - exact2))
print("Относительная:", relative_error(approx2, exact2))

# 3. Кейс с миллионом

exact3 = math.fsum([0.1] * 1000000)
approx3 = sum([0.1] * 1000000)
print("\nКейс 3:")
print("Абсолютная:", abs(approx3 - exact3))
print("Относительная:", relative_error(approx3, exact3))
