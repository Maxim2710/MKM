import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

# Используем стиль, совместимый с текущей версией matplotlib
matplotlib.use('macosx')
plt.style.use('seaborn-v0_8-darkgrid')


# Функция для вычисления потенциала и векторного поля в каждой точке сетки
def compute_field_and_potential(X, Y, charges):
    V = np.zeros_like(X)  # потенциал
    Ex = np.zeros_like(X)  # компонент поля по x
    Ey = np.zeros_like(X)  # компонент поля по y
    epsilon = 1e-9  # маленькое число для избежания деления на ноль

    for charge in charges:
        x0, y0 = charge['pos']
        q = charge['q']
        dx = X - x0
        dy = Y - y0
        r = np.sqrt(dx ** 2 + dy ** 2) + epsilon
        V += q / r
        Ex += q * dx / r ** 3
        Ey += q * dy / r ** 3
    return V, Ex, Ey


def update_plot(charges):
    global cbar
    ax_main.clear()
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)

    V, Ex, Ey = compute_field_and_potential(X, Y, charges)

    # Если потенциал постоянен, задаём небольшой диапазон для корректного отображения
    if np.isclose(np.max(V), np.min(V)):
        levels = np.linspace(np.min(V) - 1, np.max(V) + 1, 100)
    else:
        levels = np.linspace(np.min(V), np.max(V), 100)

    # Заполненные контуры потенциала
    cf = ax_main.contourf(X, Y, V, levels=levels, cmap='coolwarm', alpha=0.8)
    cs = ax_main.contour(X, Y, V, levels=levels, colors='k', linewidths=0.5, alpha=0.6)
    ax_main.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    # Обновление или создание colorbar
    if cbar is None:
        cbar = fig_main.colorbar(cf, cax=cbar_ax)
    else:
        cbar.update_normal(cf)

    # Рисуем векторное поле
    ax_main.streamplot(X, Y, Ex, Ey, color='k', linewidth=1, density=1.5,
                       arrowstyle='->', arrowsize=1.5)

    # Отмечаем заряды и создаём легенду
    pos_label_added = False
    neg_label_added = False
    for charge in charges:
        x0, y0 = charge['pos']
        if charge['q'] > 0:
            if not pos_label_added:
                ax_main.plot(x0, y0, 'ro', markersize=10, label='Положительный заряд')
                pos_label_added = True
            else:
                ax_main.plot(x0, y0, 'ro', markersize=10)
        else:
            if not neg_label_added:
                ax_main.plot(x0, y0, 'bo', markersize=10, label='Отрицательный заряд')
                neg_label_added = True
            else:
                ax_main.plot(x0, y0, 'bo', markersize=10)

    ax_main.set_xlabel('x', fontsize=12)
    ax_main.set_ylabel('y', fontsize=12)
    ax_main.set_title('Линии поля и эквипотенциалы', fontsize=14)
    ax_main.grid(True, linestyle='--', alpha=0.5)

    if pos_label_added or neg_label_added:
        ax_main.legend(loc='upper right', fontsize=10)

    # Добавляем аннотацию с полезными данными
    info_text = (f"Количество зарядов: {len(charges)}\n"
                 f"Макс потенциал: {np.max(V):.2f}\n"
                 f"Мин потенциал: {np.min(V):.2f}")
    ax_main.text(0.02, 0.98, info_text, transform=ax_main.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    fig_main.canvas.draw_idle()


# Глобальные переменные
charges_list = []
cbar = None

# Начальные границы сетки
x_min, x_max = -3, 5
y_min, y_max = -7, 7
x = np.linspace(x_min, x_max, 400)
y = np.linspace(y_min, y_max, 400)
X, Y = np.meshgrid(x, y)


# Функция добавления заряда
def add_charge(event):
    try:
        x0 = float(text_box_x.text)
        y0 = float(text_box_y.text)
        q = float(text_box_q.text)
        charges_list.append({'pos': (x0, y0), 'q': q})
        print(f"Заряд добавлен: ({x0}, {y0}, {q})")
    except Exception as e:
        print("Ошибка при добавлении заряда:", e)


# Функция обновления графика зарядов
def update_plot_button(event):
    if not charges_list:
        print("Нет зарядов для отображения. Добавьте заряд.")
    else:
        update_plot(charges_list)


def clear_charges(event):
    global charges_list, cbar
    charges_list = []
    ax_main.clear()
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)

    if cbar is not None:
        cbar.ax.cla()

    V_dummy = np.zeros_like(X)
    levels = np.linspace(-120, 120, 100)
    cf = ax_main.contourf(X, Y, V_dummy, levels=levels, cmap='coolwarm', alpha=0.8)
    cbar = fig_main.colorbar(cf, cax=cbar_ax)

    fig_main.canvas.draw_idle()
    print("Заряды очищены.")


# Функция обновления параметров сетки
def update_grid(event):
    global x_min, x_max, y_min, y_max, X, Y
    try:
        new_x_min = float(text_box_xmin.text)
        new_x_max = float(text_box_xmax.text)
        new_y_min = float(text_box_ymin.text)
        new_y_max = float(text_box_ymax.text)
        if new_x_min >= new_x_max or new_y_min >= new_y_max:
            print("Неверные значения: x_min должен быть меньше x_max, и y_min меньше y_max.")
            return
        x_min, x_max, y_min, y_max = new_x_min, new_x_max, new_y_min, new_y_max
        x = np.linspace(x_min, x_max, 400)
        y = np.linspace(y_min, y_max, 400)
        X, Y = np.meshgrid(x, y)
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(y_min, y_max)
        update_plot(charges_list)
        print("Сетка обновлена.")
    except Exception as e:
        print("Ошибка при обновлении сетки:", e)


# Создание основной фигуры и осей
fig_main, ax_main = plt.subplots(figsize=(8, 8))
# Увеличиваем нижний отступ для размещения дополнительных элементов управления
plt.subplots_adjust(bottom=0.5, right=0.85)
cbar_ax = fig_main.add_axes([0.88, 0.3, 0.03, 0.6])

# Элементы для управления зарядом
axbox_x = plt.axes([0.1, 0.12, 0.2, 0.05])
text_box_x = TextBox(axbox_x, 'x:', initial="0")

axbox_y = plt.axes([0.32, 0.12, 0.2, 0.05])
text_box_y = TextBox(axbox_y, 'y:', initial="0")

axbox_q = plt.axes([0.54, 0.12, 0.2, 0.05])
text_box_q = TextBox(axbox_q, 'q:', initial="1")

axbutton_add = plt.axes([0.1, 0.05, 0.25, 0.06])
button_add = Button(axbutton_add, 'Добавить заряд')
button_add.on_clicked(add_charge)

axbutton_plot = plt.axes([0.37, 0.05, 0.25, 0.06])
button_plot = Button(axbutton_plot, 'Обновить график')
button_plot.on_clicked(update_plot_button)

axbutton_clear = plt.axes([0.64, 0.05, 0.25, 0.06])
button_clear = Button(axbutton_clear, 'Очистить заряды')
button_clear.on_clicked(clear_charges)

# Элементы для управления границами сетки
axbox_xmin = plt.axes([0.1, 0.35, 0.18, 0.05])
text_box_xmin = TextBox(axbox_xmin, 'x_min:', initial=str(x_min))

axbox_xmax = plt.axes([0.3, 0.35, 0.18, 0.05])
text_box_xmax = TextBox(axbox_xmax, 'x_max:', initial=str(x_max))

axbox_ymin = plt.axes([0.5, 0.35, 0.18, 0.05])
text_box_ymin = TextBox(axbox_ymin, 'y_min:', initial=str(y_min))

axbox_ymax = plt.axes([0.7, 0.35, 0.18, 0.05])
text_box_ymax = TextBox(axbox_ymax, 'y_max:', initial=str(y_max))

axbutton_grid = plt.axes([0.37, 0.27, 0.25, 0.06])
button_grid = Button(axbutton_grid, 'Обновить сетку')
button_grid.on_clicked(update_grid)

plt.show()
