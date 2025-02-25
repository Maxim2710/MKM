import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.use('macosx')


def simulate_trajectory(v0, angle_deg, wind_vector, mass, radius, Cd, omega, dt=0.001):
    """
    Симуляция траектории снаряда с учетом сопротивления воздуха, ветра и эффекта Магнуса (2D).
    Если задать:
      Cd = 0, wind_vector = (0,0), omega = 0,
    то модель сводится к классической параболической траектории:
      x(t) = v0*cos(theta)*t
      y(t) = v0*sin(theta)*t - 0.5*g*t^2
    """
    g = 9.81  # ускорение свободного падения (м/с²)
    rho = 1.29  # плотность воздуха (кг/м³)
    A = np.pi * radius ** 2  # площадь поперечного сечения

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

        # Относительная скорость (с учетом ветра)
        vx_rel = vx - wind_vector[0]
        vy_rel = vy - wind_vector[1]
        v_rel = np.sqrt(vx_rel ** 2 + vy_rel ** 2)
        VREL.append(v_rel)

        # Сила сопротивления воздуха (Drag)
        if v_rel != 0:
            Fd = 0.5 * rho * Cd * A * v_rel ** 2
            Fdx = -Fd * (vx_rel / v_rel)
            Fdy = -Fd * (vy_rel / v_rel)
        else:
            Fdx, Fdy = 0, 0

        # Эффект Магнуса
        if v_rel != 0:
            F_Mx = 0.5 * rho * A * radius * omega * vy_rel
            F_My = -0.5 * rho * A * radius * omega * vx_rel
        else:
            F_Mx, F_My = 0, 0

        # Итоговые ускорения
        ax = (Fdx + F_Mx) / mass
        ay = -9.81 + (Fdy + F_My) / mass

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


def analytical_test(v0, angle_deg, wind_vector, mass, radius, Cd, omega):
    """
    Проводит аналитическую проверку для случая без сопротивления, ветра и эффекта Магнуса.
    Строит графики численной симуляции и аналитического решения, а также выводит ключевые параметры.
    """
    # Численное решение
    T, X, Y, VX, VY, VREL = simulate_trajectory(v0, angle_deg, wind_vector, mass, radius, Cd, omega)

    g = 9.81
    angle_rad = np.deg2rad(angle_deg)

    # Аналитическое решение
    t_flight = 2 * v0 * np.sin(angle_rad) / g  # время полёта
    t_analytic = np.linspace(0, t_flight, len(T))
    x_analytic = v0 * np.cos(angle_rad) * t_analytic
    y_analytic = v0 * np.sin(angle_rad) * t_analytic - 0.5 * g * t_analytic ** 2

    # Ключевые параметры
    flight_time_numerical = T[-1]
    range_numerical = X[-1]
    max_height_numerical = max(Y)

    range_analytic = v0 ** 2 * np.sin(2 * angle_rad) / g
    max_height_analytic = (v0 * np.sin(angle_rad)) ** 2 / (2 * g)

    # Построение графиков
    plt.figure(figsize=(8, 5))
    plt.plot(X, Y, label="Численная симуляция", lw=2)
    plt.plot(x_analytic, y_analytic, 'r--', label="Аналитическое решение")
    plt.title(f"Сравнение для v0 = {v0} м/с, угол = {angle_deg}°")
    plt.xlabel("X (м)")
    plt.ylabel("Y (м)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Результаты для v0 = {v0} м/с, угол = {angle_deg}°:")
    print(f"  Численное время полёта: {flight_time_numerical:.3f} с, аналитическое: {t_flight:.3f} с")
    print(f"  Численная дальность: {range_numerical:.3f} м, аналитическая: {range_analytic:.3f} м")
    print(f"  Численная макс. высота: {max_height_numerical:.3f} м, аналитическая: {max_height_analytic:.3f} м")
    print("-" * 60)

    return T, X, Y, VREL


def simulation_test(v0, angle_deg, wind_vector, mass, radius, Cd, omega):
    """
    Проводит тест для случая с ненулевыми значениями для Cd, ветра и эффекта Магнуса.
    Здесь аналитического решения нет, поэтому выводятся только результаты численной симуляции.
    """
    T, X, Y, VX, VY, VREL = simulate_trajectory(v0, angle_deg, wind_vector, mass, radius, Cd, omega)

    plt.figure(figsize=(8, 5))
    plt.plot(X, Y, label="Численная симуляция", lw=2)
    plt.title(f"Симуляция: v0 = {v0} м/с, угол = {angle_deg}°, wind = {wind_vector}, Cd = {Cd}, omega = {omega}")
    plt.xlabel("X (м)")
    plt.ylabel("Y (м)")
    plt.legend()
    plt.grid(True)
    plt.show()

    flight_time = T[-1]
    range_sim = X[-1]
    max_height = max(Y)
    print(
        f"Тест с параметрами: v0 = {v0} м/с, угол = {angle_deg}°, wind = {wind_vector}, mass = {mass}, radius = {radius}, Cd = {Cd}, omega = {omega}")
    print(f"  Время полёта: {flight_time:.3f} с, Дальность: {range_sim:.3f} м, Макс. высота: {max_height:.3f} м")
    print("-" * 60)

    return T, X, Y, VREL


def main():
    """
    Главная функция проводит три теста:

    Тест 1 (аналитический):
      - v0 = 50 м/с, угол = 45°
      - wind_vector = (0, 0), Cd = 0, omega = 0

    Тест 2 (аналитический):
      - v0 = 60 м/с, угол = 30°
      - wind_vector = (0, 0), Cd = 0, omega = 0

    Тест 3 (расширенный случай):
      - v0 = 40 м/с, угол = 40°
      - wind_vector = (2, 0) (есть ветер), Cd = 0.2, omega = 20 (есть сопротивление и эффект Магнуса)
    """
    print("Аналитический тест 1: v0 = 50 м/с, угол = 45° (без ветра, сопротивления и эффекта Магнуса)")
    v0_test1 = 50.0
    angle_test1 = 45.0
    wind_vector = (0.0, 0.0)
    mass = 1.0
    radius = 0.05
    Cd = 0.0
    omega = 0.0
    T1, X1, Y1, VREL1 = analytical_test(v0_test1, angle_test1, wind_vector, mass, radius, Cd, omega)

    print("Аналитический тест 2: v0 = 60 м/с, угол = 30° (без ветра, сопротивления и эффекта Магнуса)")
    v0_test2 = 60.0
    angle_test2 = 30.0
    T2, X2, Y2, VREL2 = analytical_test(v0_test2, angle_test2, wind_vector, mass, radius, Cd, omega)

    print("Тест 3: Случай с ветром, сопротивлением и эффектом Магнуса")
    v0_test3 = 40.0
    angle_test3 = 40.0
    wind_vector_test3 = (2.0, 0.0)  # горизонтальный ветер
    mass_test3 = 0.5
    radius_test3 = 0.03
    Cd_test3 = 0.2
    omega_test3 = 20.0
    T3, X3, Y3, VREL3 = simulation_test(v0_test3, angle_test3, wind_vector_test3, mass_test3, radius_test3, Cd_test3,
                                        omega_test3)

    # Опционально: запрашиваем анимацию для каждого теста
    answer = input("Показать анимацию для Теста 1? (y/n): ")
    if answer.lower() == 'y':
        animate_trajectory(T1, X1, Y1, VREL1)

    answer = input("Показать анимацию для Теста 2? (y/n): ")
    if answer.lower() == 'y':
        animate_trajectory(T2, X2, Y2, VREL2)

    answer = input("Показать анимацию для Теста 3? (y/n): ")
    if answer.lower() == 'y':
        animate_trajectory(T3, X3, Y3, VREL3)


if __name__ == "__main__":
    main()
