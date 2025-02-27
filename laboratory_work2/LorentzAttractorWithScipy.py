import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('macosx')

# Определяем систему уравнений аттрактора Лоренца
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)  # dx/dt = sigma * (y - x)
    dydt = x * (rho - z) - y  # dy/dt = x * (rho - z) - y
    dzdt = x * y - beta * z  # dz/dt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def animate_attractor(sol):
    # Создаем фигуру для анимации
    fig_anim = plt.figure()
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    ax_anim.set_title("Анимация аттрактора Лоренца")
    ax_anim.set_xlabel("X")
    ax_anim.set_ylabel("Y")
    ax_anim.set_zlabel("Z")

    # Определяем пределы осей на основе решения
    ax_anim.set_xlim(np.min(sol.y[0]), np.max(sol.y[0]))
    ax_anim.set_ylim(np.min(sol.y[1]), np.max(sol.y[1]))
    ax_anim.set_zlim(np.min(sol.y[2]), np.max(sol.y[2]))

    # Создаем пустой график (линию)
    line, = ax_anim.plot([], [], [], lw=2, color='b')

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    def update(frame):
        # Обновляем данные линии до текущего кадра
        line.set_data(sol.y[0][:frame], sol.y[1][:frame])
        line.set_3d_properties(sol.y[2][:frame])
        return line,

    # Ускоряем анимацию (интервал 10 мс)
    ani = animation.FuncAnimation(fig_anim, update, frames=len(sol.t),
                                  init_func=init, interval=10, blit=True)
    plt.show()


def main():
    # Ввод параметров от пользователя
    sigma = float(input("Введите значение sigma (например, 10): ") or "10")
    rho = float(input("Введите значение rho (например, 28): ") or "28")
    beta = float(input("Введите значение beta (например, 8/3): ") or str(8 / 3))

    x0 = float(input("Введите начальное значение x (например, 1): ") or "1")
    y0 = float(input("Введите начальное значение y (например, 1): ") or "1")
    z0 = float(input("Введите начальное значение z (например, 1): ") or "1")

    t_max = float(input("Введите время моделирования (например, 40): ") or "40")
    dt = float(input("Введите шаг по времени (например, 0.01): ") or "0.01")

    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    init_state = [x0, y0, z0]

    # Численное интегрирование системы
    sol = solve_ivp(lorenz, t_span, init_state, args=(sigma, rho, beta), t_eval=t_eval)

    # Визуализация статических графиков
    fig = plt.figure(figsize=(12, 8))

    # 3D-график аттрактора Лоренца
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.5)
    ax1.set_title("Аттрактор Лоренца")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # График x(t)
    ax2 = fig.add_subplot(222)
    ax2.plot(sol.t, sol.y[0], 'r')
    ax2.set_title("X(t)")
    ax2.set_xlabel("Время")
    ax2.set_ylabel("X")

    # График y(t)
    ax3 = fig.add_subplot(223)
    ax3.plot(sol.t, sol.y[1], 'g')
    ax3.set_title("Y(t)")
    ax3.set_xlabel("Время")
    ax3.set_ylabel("Y")

    # График z(t)
    ax4 = fig.add_subplot(224)
    ax4.plot(sol.t, sol.y[2], 'b')
    ax4.set_title("Z(t)")
    ax4.set_xlabel("Время")
    ax4.set_ylabel("Z")

    plt.tight_layout()
    plt.show()

    # Запрос на запуск анимации
    animate_choice = input("Показать анимацию аттрактора? (y/n): ")
    if animate_choice.lower() == 'y':
        animate_attractor(sol)


if __name__ == "__main__":
    main()

# Введите значение sigma (например, 10): 10
# Введите значение rho (например, 28): 28
# Введите значение beta (например, 8/3): 2.66666666667
# Введите начальное значение x (например, 1): 1
# Введите начальное значение y (например, 1): 1
# Введите начальное значение z (например, 1): 1
# Введите время моделирования (например, 40): 40
# Введите шаг по времени (например, 0.01): 0.01


# Введите значение sigma: 10
# Введите значение rho: 40
# Введите значение beta: 2.66666666667
# Введите начальное значение x: 0.1
# Введите начальное значение y: 0
# Введите начальное значение z: 0
# Введите время моделирования: 50
# Введите шаг по времени: 0.005


# Введите значение sigma: 10
# Введите значение rho: 5
# Введите значение beta: 2.66666666667
# Введите начальное значение x: 1
# Введите начальное значение y: 1
# Введите начальное значение z: 1
# Введите время моделирования: 20
# Введите шаг по времени: 0.01