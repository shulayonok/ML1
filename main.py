"""
Линейная регрессия

y = kx + b
y = w0 + w1x1 + w2x2 + ...
y = 5 при x1 = 4 x2 = 3 x3 = ...

y = w0 + w1f1(x) + w2f2(x) + w3f3(x) + w4f4(x)
w1 w2 w3 w4

y = W * X
W = (w0, w1, ...)
X = (1, f1(x), ...)

y1, y2, ..., yn
X1, X2, ..., Xn
Y = (y1, y2, ..., yn)

Y = F * W
F =
(
(1, f1(x1), f2(x1), ...)
(1, f1(x2), f2(x2), ...)
...
)

x1 - значения в ходе первого эксперимента
x2 - ... в ходе второго эксп.

x1 -> y1
x2 -> y2

xg -> yg (ранее неизвестная зависимость) - предсказание по старым данным

E = y ~ t (расхождение между действ. выходом и желаемым)
E = (1/2) * sum((t - F * W)^2)
E' (w0, w1, w2, ...)
grad(E) - вектор, где каждая компонента - частная производная
grad(E) = 0

Ниже - все маленькие буквы - векторы
t - вектор желаемых выходов
0 = -t * F + w * (trans(F) * F)
w = invers(transp(F) * F) * transp(F) * t

pinv(F) - псевдообратная матрица
np.linlg.pinv(F)
np.linalg.solve - решает уравнение такоего вида: F * w = t

t1 ~ x1
t1 = w0 + w1 * x1 + w2 * x1^2 + ...
"""
import numpy as np
import matplotlib.pyplot as plt


def regression(x, degree, t):
    design_matrix = np.ones([degree + 1, len(x)])
    for i in range(degree + 1):
        for j in range(len(x)):
            design_matrix[i][j] = np.power(x[j], i)
    F = np.array(design_matrix).T
    if degree + 1 <= 40:
        w = np.linalg.inv(F.T @ F) @ F.T @ t
    else:
        w = np.linalg.pinv(F) @ t
    y = F @ w
    # error
    return y


def regression_plot(x, t, z, degree):
    plt.scatter(x, t, c="blue")
    plt.plot(x, z, c="black")
    y = regression(x, degree, t)
    plt.plot(x, y, c="red")
    plt.show()


def err(x, t):
    Error = []
    for i in range(1, 101):
        F = np.ones((x.shape[0], i + 1))
        for j in range(F.shape[1]):
            F[:, j] = x.T ** j
        # Fw = t
        # w = F* t
        # w = np.linalg.inv(F.T @ F) @ F.T @ t
        w = np.linalg.pinv(F) @ t
        y = F @ w
        Error.append((1 / 2) * np.sum((t - y) ** 2))
    m = np.array(range(1, 101))
    plt.plot(m, Error)
    plt.show()


# Входные данные
N = 1000
x = np.linspace(0, 1, N)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

# degree=1
regression_plot(x, t, z, 1)

# degree=8
regression_plot(x, t, z, 8)

# degree=100
regression_plot(x, t, z, 100)

# degree=180
regression_plot(x, t, z, 180)

# Зависимость ошибки от степени полинома
err(x, t)
