import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.use('macosx')

def simulate_trajectory(v0, angle_deg, wind_vector, mass, radius, Cd, dt=0.001):
    """
    Симуляция траектории снаряда с учетом сопротивления воздуха и ветра (2D).
    Параметр wind_vector — кортеж (wind_x, wind_y).
    Возвращает массивы времени T, координат X, Y, горизонтальной скорости VX,
    вертикальной скорости VY и относительной скорости VREL.
    """
    g = 9.81      # ускорение свободного падения
    rho = 1.29    # плотность воздуха
    A = np.pi * radius ** 2  # площадь поперечного сечения снаряда

    angle_rad = np.deg2rad(angle_deg)
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    x = 0.0
    y = 0.0

    T, X, Y, VX, VY, VREL = [], [], [], [], [], []
    t = 0.0

    while y >= 0:
        T.append(t)
        X.append(x)
        Y.append(y)
        VX.append(vx)
        VY.append(vy)

        # Вычисляем относительную скорость с учетом ветрового вектора
        vx_rel = vx - wind_vector[0]
        vy_rel = vy - wind_vector[1]
        v_rel = np.sqrt(vx_rel ** 2 + vy_rel ** 2)
        VREL.append(v_rel)

        # Сила сопротивления (квадратичный закон)
        if v_rel != 0:
            Fd = 0.5 * rho * Cd * A * (v_rel ** 2)
            Fdx = -Fd * (vx_rel / v_rel)
            Fdy = -Fd * (vy_rel / v_rel)
        else:
            Fdx = 0
            Fdy = 0

        # Ускорения
        ax = Fdx / mass
        ay = -g + Fdy / mass

        # Обновление скоростей и координат
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        t += dt

    return np.array(T), np.array(X), np.array(Y), np.array(VX), np.array(VY), np.array(VREL)


def animate_trajectory(T, X, Y, VREL):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, X.max() * 1.1)
    ax.set_ylim(0, Y.max() * 1.1)
    ax.set_title("Анимация полёта снаряда")
    ax.set_xlabel("X (м)")
    ax.set_ylabel("Y (м)")
    ax.grid(True)

    line, = ax.plot([], [], 'r-', lw=2, label='Траектория')
    point, = ax.plot([], [], 'bo', markersize=8, label='Снаряд')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black')
    speed_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, color='black')
    ax.legend()

    def init():
        line.set_data([], [])
        point.set_data([], [])
        time_text.set_text('')
        speed_text.set_text('')
        return line, point, time_text, speed_text

    def update(frame):
        line.set_data(X[:frame], Y[:frame])
        point.set_data([X[frame]], [Y[frame]])
        time_text.set_text(f"t = {T[frame]:.2f} с")
        speed_text.set_text(f"v = {VREL[frame]:.2f} м/с")
        return line, point, time_text, speed_text

    ani = animation.FuncAnimation(fig, update, frames=len(T), init_func=init,
                                  interval=1, blit=True, repeat=False)
    plt.show()


def main():
    """
    Главная функция: запрашивает параметры, рассчитывает траекторию, строит
    статические графики и запускает анимацию (по выбору пользователя).
    """
    print("=== Расчёт траектории полёта с учётом сопротивления и ветра (2D) ===")
    try:
        v0 = float(input("Начальная скорость (м/с): "))
        angle_deg = float(input("Угол броска (градусы): "))
        print("Введите компоненты ветрового вектора:")
        wind_x = float(input("  Горизонтальная составляющая (м/с): "))
        wind_y = float(input("  Вертикальная составляющая (м/с): "))
        wind_vector = (wind_x, wind_y)
        mass = float(input("Масса снаряда (кг): "))
        radius = float(input("Радиус снаряда (м): "))
        Cd = float(input("Коэффициент лобового сопротивления (обычно 0.1..1.0): "))
    except ValueError:
        print("Ошибка ввода! Проверьте правильность введённых значений.")
        return

    # Расчёт траектории
    T, X, Y, VX, VY, VREL = simulate_trajectory(v0, angle_deg, wind_vector, mass, radius, Cd)

    # Построение дополнительных графиков
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # 1. Траектория полёта
    axs[0].plot(X, Y, 'r-', label='Траектория')
    axs[0].set_title("Траектория полёта")
    axs[0].set_xlabel("X (м)")
    axs[0].set_ylabel("Y (м)")
    axs[0].grid(True)
    axs[0].legend()

    # 2. Горизонтальная и вертикальная скорости по времени
    axs[1].plot(T, VX, label='Vx (горизонтальная)', color='blue')
    axs[1].plot(T, VY, label='Vy (вертикальная)', color='orange')
    axs[1].set_title("Скорости по времени")
    axs[1].set_xlabel("Время (с)")
    axs[1].set_ylabel("Скорость (м/с)")
    axs[1].grid(True)
    axs[1].legend()

    # 3. Относительная скорость по времени
    axs[2].plot(T, VREL, label='Относительная скорость', color='green')
    axs[2].set_title("Относительная скорость по времени")
    axs[2].set_xlabel("Время (с)")
    axs[2].set_ylabel("Скорость (м/с)")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    # Запрос анимации
    answer = input("Показать анимацию полёта снаряда? (y/n): ")
    if answer.lower() == 'y':
        animate_trajectory(T, X, Y, VREL)


if __name__ == "__main__":
    main()
