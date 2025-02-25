import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Для MacOS можно использовать backend 'macosx'
matplotlib.use('macosx')

# ====================================================================
# Итоговые формулы, используемые в симуляции:
#
# 1. Площадь поперечного сечения:
#       A = π * (radius)^2
#
# 2. Относительная скорость (учёт ветра):
#       v_rel = sqrt((vx - wind_x)^2 + (vy - wind_y)^2 + (vz - wind_z)^2)
#
# 3. Сила сопротивления воздуха (Drag):
#       F_d = 0.5 * ρ * C_d * A * (v_rel)^2
#
# 4. Разложение силы сопротивления по осям:
#       F_dx = -F_d * (vx - wind_x) / v_rel
#       F_dy = -F_d * (vy - wind_y) / v_rel
#       F_dz = -F_d * (vz - wind_z) / v_rel
#
# 5. Ускорения:
#       ax = F_dx / mass
#       ay = F_dy / mass
#       az = -g + (F_dz / mass)
#
# g  - ускорение свободного падения (9.81 м/с²)
# ρ  - плотность воздуха (1.29 кг/м³)
# ====================================================================

def simulate_trajectory_3d(v0, elevation_deg, azimuth_deg, wind_vector, mass, radius, Cd, dt=0.001):
    """
    Симуляция траектории снаряда в 3D с учетом сопротивления воздуха и ветра.

    Параметры:
      v0            - начальная скорость (м/с)
      elevation_deg - угол подъёма относительно горизонтали (в градусах)
      azimuth_deg   - азимут относительно оси X (в градусах)
      wind_vector   - кортеж (wind_x, wind_y, wind_z)
      mass          - масса снаряда (кг)
      radius        - радиус снаряда (м)
      Cd            - коэффициент лобового сопротивления
      dt            - шаг по времени (с)

    Возвращает:
      T      - массив времени
      X, Y, Z - массивы координат по осям X, Y, Z
      VX, VY, VZ - массивы компонент скорости
      VREL   - массив относительных скоростей (учёт ветра)
    """
    g = 9.81      # ускорение свободного падения (м/с²)
    rho = 1.29    # плотность воздуха (кг/м³)
    A = np.pi * radius ** 2  # площадь поперечного сечения

    # Перевод углов в радианы
    elevation_rad = np.deg2rad(elevation_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)

    # Начальные компоненты скорости
    vx = v0 * np.cos(elevation_rad) * np.cos(azimuth_rad)
    vy = v0 * np.cos(elevation_rad) * np.sin(azimuth_rad)
    vz = v0 * np.sin(elevation_rad)

    # Начальные координаты
    x = 0.0
    y = 0.0
    z = 0.0

    T, X, Y, Z = [], [], [], []
    VX, VY, VZ, VREL = [], [], [], []
    t = 0.0

    while z >= 0:
        T.append(t)
        X.append(x)
        Y.append(y)
        Z.append(z)
        VX.append(vx)
        VY.append(vy)
        VZ.append(vz)

        # Вычисляем относительную скорость с учетом ветра:
        vx_rel = vx - wind_vector[0]
        vy_rel = vy - wind_vector[1]
        vz_rel = vz - wind_vector[2]
        v_rel = np.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2)
        VREL.append(v_rel)

        # Сила сопротивления (Drag)
        if v_rel != 0:
            Fd = 0.5 * rho * Cd * A * (v_rel ** 2)
            Fdx = -Fd * (vx_rel / v_rel)
            Fdy = -Fd * (vy_rel / v_rel)
            Fdz = -Fd * (vz_rel / v_rel)
        else:
            Fdx = Fdy = Fdz = 0

        # Ускорения
        ax = Fdx / mass
        ay = Fdy / mass
        az = -g + (Fdz / mass)

        # Обновление скоростей
        vx += ax * dt
        vy += ay * dt
        vz += az * dt

        # Обновление координат
        x += vx * dt
        y += vy * dt
        z += vz * dt

        t += dt

    return (np.array(T), np.array(X), np.array(Y), np.array(Z),
            np.array(VX), np.array(VY), np.array(VZ), np.array(VREL))


def animate_trajectory_3d(T, X, Y, Z, VREL):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, X.max() * 1.1)
    ax.set_ylim(0, Y.max() * 1.1)
    ax.set_zlim(0, Z.max() * 1.1)
    ax.set_title("Анимация полёта снаряда (3D)")
    ax.set_xlabel("X (м)")
    ax.set_ylabel("Y (м)")
    ax.set_zlabel("Z (м)")

    line, = ax.plot([], [], [], 'r-', lw=2, label='Траектория')
    point, = ax.plot([], [], [], 'bo', markersize=8, label='Снаряд')
    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
    speed_text = ax.text2D(0.05, 0.90, "", transform=ax.transAxes)
    ax.legend()

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        time_text.set_text("")
        speed_text.set_text("")
        return line, point, time_text, speed_text

    def update(frame):
        # Обновляем линию траектории
        line.set_data(X[:frame], Y[:frame])
        line.set_3d_properties(Z[:frame])
        # Обновляем положение снаряда
        point.set_data([X[frame]], [Y[frame]])
        point.set_3d_properties([Z[frame]])
        time_text.set_text(f"t = {T[frame]:.2f} с")
        speed_text.set_text(f"v = {VREL[frame]:.2f} м/с")
        return line, point, time_text, speed_text

    ani = animation.FuncAnimation(fig, update, frames=len(T), init_func=init,
                                  interval=1, blit=True, repeat=False)
    plt.show()


def analytical_test_3d(v0, elevation_deg, azimuth_deg, mass, radius, dt=0.001):
    """
    Аналитическая проверка для 3D модели при отсутствии сопротивления и ветра.
    Параметры ветра и сопротивления устанавливаются равными нулю:
      wind_vector = (0, 0, 0), Cd = 0
    Аналитическое решение:
      x(t) = v0*cos(elevation)*cos(azimuth)*t
      y(t) = v0*cos(elevation)*sin(azimuth)*t
      z(t) = v0*sin(elevation)*t - 0.5*g*t^2
    """
    g = 9.81
    wind_vector = (0.0, 0.0, 0.0)
    Cd = 0.0

    # Получаем численное решение
    (T, X, Y, Z, VX, VY, VZ, VREL) = simulate_trajectory_3d(v0, elevation_deg, azimuth_deg, wind_vector, mass, radius, Cd, dt)

    # Аналитическое решение
    elevation_rad = np.deg2rad(elevation_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    T_flight = (2 * v0 * np.sin(elevation_rad)) / g  # время полёта
    t_analytic = np.linspace(0, T_flight, len(T))
    x_analytic = v0 * np.cos(elevation_rad) * np.cos(azimuth_rad) * t_analytic
    y_analytic = v0 * np.cos(elevation_rad) * np.sin(azimuth_rad) * t_analytic
    z_analytic = v0 * np.sin(elevation_rad) * t_analytic - 0.5 * g * t_analytic**2

    # Ключевые параметры
    flight_time_numerical = T[-1]
    range_numerical = np.sqrt(X[-1]**2 + Y[-1]**2)
    max_height_numerical = max(Z)
    range_analytic = v0 * np.cos(elevation_rad) * T_flight
    max_height_analytic = (v0 * np.sin(elevation_rad))**2 / (2 * g)

    # Построение графиков
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X, Y, Z, 'r-', lw=2, label='Численная симуляция')
    ax.plot(x_analytic, y_analytic, z_analytic, 'b--', lw=2, label='Аналитическое решение')
    ax.set_title(f"Сравнение 3D траекторий: v0 = {v0} м/с, elev = {elevation_deg}°, azim = {azimuth_deg}°")
    ax.set_xlabel("X (м)")
    ax.set_ylabel("Y (м)")
    ax.set_zlabel("Z (м)")
    ax.legend()
    plt.show()

    print(f"Результаты для v0 = {v0} м/с, elevation = {elevation_deg}°, azim = {azimuth_deg}°:")
    print(f"  Численное время полёта: {flight_time_numerical:.3f} с, аналитическое: {T_flight:.3f} с")
    print(f"  Численная дальность (горизонтальная): {range_numerical:.3f} м, аналитическая: {range_analytic:.3f} м")
    print(f"  Численная макс. высота: {max_height_numerical:.3f} м, аналитическая: {max_height_analytic:.3f} м")
    print("-" * 60)

    return T, X, Y, Z, VX, VY, VZ, VREL


def simulation_test_3d(v0, elevation_deg, azimuth_deg, wind_vector, mass, radius, Cd, dt=0.001):
    """
    Тест для 3D модели с ненулевыми значениями для сопротивления и ветра.
    Здесь аналитического решения нет, поэтому выводятся только результаты численной симуляции.
    """
    (T, X, Y, Z, VX, VY, VZ, VREL) = simulate_trajectory_3d(v0, elevation_deg, azimuth_deg, wind_vector, mass, radius, Cd, dt)

    # Построение 3D графика траектории
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X, Y, Z, 'r-', lw=2, label='Численная симуляция')
    ax.set_title(f"3D траектория: v0 = {v0} м/с, elev = {elevation_deg}°, azim = {azimuth_deg}°, wind = {wind_vector}, Cd = {Cd}")
    ax.set_xlabel("X (м)")
    ax.set_ylabel("Y (м)")
    ax.set_zlabel("Z (м)")
    ax.legend()
    plt.show()

    flight_time = T[-1]
    range_sim = np.sqrt(X[-1]**2 + Y[-1]**2)
    max_height = max(Z)
    print(f"Тест 3 с параметрами: v0 = {v0} м/с, elev = {elevation_deg}°, azim = {azimuth_deg}°, wind = {wind_vector}, mass = {mass}, radius = {radius}, Cd = {Cd}")
    print(f"  Время полёта: {flight_time:.3f} с, Дальность (горизонтальная): {range_sim:.3f} м, Макс. высота: {max_height:.3f} м")
    print("-" * 60)

    return T, X, Y, Z, VX, VY, VZ, VREL


def main():
    print("=== 3D симуляция траектории с учетом сопротивления и ветра ===")

    # Аналитический тест 1: без ветра и сопротивления
    print("Аналитический тест 1:")
    v0_test1 = 50.0
    elevation_test1 = 45.0
    azimuth_test1 = 30.0
    mass_test1 = 1.0
    radius_test1 = 0.05
    T1, X1, Y1, Z1, VX1, VY1, VZ1, VREL1 = analytical_test_3d(v0_test1, elevation_test1, azimuth_test1, mass_test1, radius_test1)

    # Аналитический тест 2: другой набор параметров
    print("Аналитический тест 2:")
    v0_test2 = 40.0
    elevation_test2 = 30.0
    azimuth_test2 = 60.0
    mass_test2 = 1.0
    radius_test2 = 0.05
    T2, X2, Y2, Z2, VX2, VY2, VZ2, VREL2 = analytical_test_3d(v0_test2, elevation_test2, azimuth_test2, mass_test2, radius_test2)

    # Симуляционный тест 3: с ненулевыми Cd и ветром
    print("Симуляционный тест 3:")
    v0_test3 = 80.0
    elevation_test3 = 40.0
    azimuth_test3 = 20.0
    wind_vector_test3 = (2.0, 1.0, 0.0)  # присутствует ветер
    mass_test3 = 0.5
    radius_test3 = 0.03
    Cd_test3 = 0.2
    T3, X3, Y3, Z3, VX3, VY3, VZ3, VREL3 = simulation_test_3d(v0_test3, elevation_test3, azimuth_test3, wind_vector_test3, mass_test3, radius_test3, Cd_test3)

    # Запрос на показ анимаций для всех симуляционных тестов
    answer = input("Показать анимацию для всех симуляционных тестов? (y/n): ")
    if answer.lower() == 'y':
        print("Запуск анимации для Аналитического теста 1...")
        animate_trajectory_3d(T1, X1, Y1, Z1, VREL1)
        print("Запуск анимации для Аналитического теста 2...")
        animate_trajectory_3d(T2, X2, Y2, Z2, VREL2)
        print("Запуск анимации для Симуляционного теста 3...")
        animate_trajectory_3d(T3, X3, Y3, Z3, VREL3)


if __name__ == "__main__":
    main()
