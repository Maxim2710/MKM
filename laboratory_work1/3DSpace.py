import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Для MacOS можно использовать backend 'macosx'
matplotlib.use('macosx')

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
      T    - массив времени
      X, Y, Z - массивы координат по осям X, Y, Z
      VX, VY, VZ - массивы компонент скорости
      VREL - массив относительных скоростей (учёт ветра)
    """
    g = 9.81  # ускорение свободного падения
    rho = 1.29  # плотность воздуха
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

        # Вычисляем относительную скорость с учетом ветра
        vx_rel = vx - wind_vector[0]
        vy_rel = vy - wind_vector[1]
        vz_rel = vz - wind_vector[2]
        v_rel = np.sqrt(vx_rel ** 2 + vy_rel ** 2 + vz_rel ** 2)
        VREL.append(v_rel)

        # Сила сопротивления (квадратичный закон)
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
        az = -g + Fdz / mass  # Гравитация действует по оси Z

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


def main():
    print("=== Расчёт траектории полёта с учётом сопротивления и ветра (3D) ===")
    try:
        v0 = float(input("Начальная скорость (м/с): "))
        elevation_deg = float(input("Угол подъёма (градусы): "))
        azimuth_deg = float(input("Азимут (градусы, от оси X): "))
        print("Введите компоненты ветрового вектора:")
        wind_x = float(input("  Компонента по X (м/с): "))
        wind_y = float(input("  Компонента по Y (м/с): "))
        wind_z = float(input("  Компонента по Z (м/с): "))
        wind_vector = (wind_x, wind_y, wind_z)
        mass = float(input("Масса снаряда (кг): "))
        radius = float(input("Радиус снаряда (м): "))
        Cd = float(input("Коэффициент лобового сопротивления (обычно 0.1..1.0): "))
    except ValueError:
        print("Ошибка ввода! Проверьте правильность введённых значений.")
        return

    # Расчёт траектории
    (T, X, Y, Z, VX, VY, VZ, VREL) = simulate_trajectory_3d(
        v0, elevation_deg, azimuth_deg, wind_vector, mass, radius, Cd)

    # Построение статических графиков
    fig = plt.figure(figsize=(15, 10))

    # 1. 3D траектория полёта
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(X, Y, Z, 'r-', label='Траектория')
    ax1.set_title("3D Траектория полёта")
    ax1.set_xlabel("X (м)")
    ax1.set_ylabel("Y (м)")
    ax1.set_zlabel("Z (м)")
    ax1.legend()

    # 2. Компоненты скорости по времени
    ax2 = fig.add_subplot(222)
    ax2.plot(T, VX, label='Vx (м/с)', color='blue')
    ax2.plot(T, VY, label='Vy (м/с)', color='orange')
    ax2.plot(T, VZ, label='Vz (м/с)', color='green')
    ax2.set_title("Компоненты скорости по времени")
    ax2.set_xlabel("Время (с)")
    ax2.set_ylabel("Скорость (м/с)")
    ax2.legend()
    ax2.grid(True)

    # 3. Относительная скорость
    ax3 = fig.add_subplot(223)
    ax3.plot(T, VREL, label='Относительная скорость (м/с)', color='purple')
    ax3.set_title("Относительная скорость по времени")
    ax3.set_xlabel("Время (с)")
    ax3.set_ylabel("Скорость (м/с)")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    answer = input("Показать анимацию полёта снаряда (3D)? (y/n): ")
    if answer.lower() == 'y':
        animate_trajectory_3d(T, X, Y, Z, VREL)


if __name__ == "__main__":
    main()