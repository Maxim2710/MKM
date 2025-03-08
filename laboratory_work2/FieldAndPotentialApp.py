import sys
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox, QGroupBox, QListWidget,
    QFileDialog, QSlider, QCheckBox, QComboBox, QSplitter, QMenuBar, QAction
)
from PyQt5.QtCore import Qt, QTimer

# Функция вычисления потенциала и векторного поля
def compute_field_and_potential(X, Y, charges):
    V = np.zeros_like(X)
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(X)
    epsilon = 1e-9  # для избежания деления на ноль
    for charge in charges:
        x0, y0 = charge['pos']
        q = charge['q']
        dx = X - x0
        dy = Y - y0
        r = np.sqrt(dx**2 + dy**2) + epsilon
        V += q / r
        Ex += q * dx / r**3
        Ey += q * dy / r**3
    return V, Ex, Ey

# Класс для холста matplotlib, встроенного в PyQt5
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        # Разрешаем автоматическое расширение холста
        self.setSizePolicy(self.sizePolicy().Expanding, self.sizePolicy().Expanding)

# Основное окно приложения
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Поля зарядов")
        # Параметры по умолчанию
        self.default_x_min, self.default_x_max = -3, 5
        self.default_y_min, self.default_y_max = -7, 7
        self.default_amplitude = 0.5
        self.default_animation_interval = 50
        self.default_colormap = 'coolwarm'

        self.charges = []
        self.t = 0.0  # переменная времени для анимации
        self.amplitude = self.default_amplitude  # амплитуда движения зарядов

        # Дополнительные настройки
        self.show_streamlines = True
        self.show_contours = True
        self.colormap = self.default_colormap
        self.animation_interval = self.default_animation_interval  # мс

        # Настройка области (сетка)
        self.x_min, self.x_max = self.default_x_min, self.default_x_max
        self.y_min, self.y_max = self.default_y_min, self.default_y_max
        self.x = np.linspace(self.x_min, self.x_max, 400)
        self.y = np.linspace(self.y_min, self.y_max, 400)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Создание холста matplotlib и навигационной панели
        self.canvas = MplCanvas(self, width=8, height=8, dpi=100)
        self.canvas.ax.set_xlim(self.x_min, self.x_max)
        self.canvas.ax.set_ylim(self.y_min, self.y_max)
        plt.style.use('seaborn-v0_8-darkgrid')
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Создание оси для colorbar
        self.cbar = None
        self.cbar_ax = self.canvas.fig.add_axes([0.88, 0.3, 0.03, 0.6])

        # Создание панели управления
        controls_widget = self.create_controls_panel()

        # QSplitter для разделения области с графиком и панели управления
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(controls_widget)
        self.splitter.setSizes([400, 400])  # можно настроить начальное соотношение

        # Основной макет: навигационная панель + splitter
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.splitter)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Создаем меню
        self.create_menu_bar()

        self.statusBar().showMessage("Готово")
        self.update_plot()
        self.animation_timer = None
        self.canvas_expanded = False  # состояние, развернут ли график на весь экран

    def create_menu_bar(self):
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        # Меню Файл
        file_menu = menubar.addMenu("Файл")
        save_action = QAction("Сохранить изображение", self)
        save_action.triggered.connect(self.save_image)
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # Меню Справка
        help_menu = menubar.addMenu("Справка")
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def show_about(self):
        QMessageBox.about(self, "О программе",
                          "Приложение для визуализации полей зарядов.\n"
                          "Разработано с использованием PyQt5 и Matplotlib.")

    def create_controls_panel(self):
        # Группа "Управление зарядом"
        charge_group = QGroupBox("Управление зарядом")
        charge_layout = QGridLayout()
        charge_group.setLayout(charge_layout)
        self.x_input = QLineEdit("0")
        self.x_input.setToolTip("Введите x координату заряда")
        self.y_input = QLineEdit("0")
        self.y_input.setToolTip("Введите y координату заряда")
        self.q_input = QLineEdit("1")
        self.q_input.setToolTip("Введите значение заряда (q)")
        charge_layout.addWidget(QLabel("x:"), 0, 0)
        charge_layout.addWidget(self.x_input, 0, 1)
        charge_layout.addWidget(QLabel("y:"), 1, 0)
        charge_layout.addWidget(self.y_input, 1, 1)
        charge_layout.addWidget(QLabel("q:"), 2, 0)
        charge_layout.addWidget(self.q_input, 2, 1)
        self.add_charge_btn = QPushButton("Добавить заряд")
        self.add_charge_btn.setToolTip("Добавить новый заряд")
        self.add_charge_btn.clicked.connect(self.add_charge)
        charge_layout.addWidget(self.add_charge_btn, 3, 0, 1, 2)

        # Список зарядов и кнопка для удаления выбранного
        self.charge_list = QListWidget()
        self.charge_list.setToolTip("Список добавленных зарядов")
        charge_layout.addWidget(QLabel("Список зарядов:"), 4, 0, 1, 2)
        charge_layout.addWidget(self.charge_list, 5, 0, 1, 2)
        self.remove_charge_btn = QPushButton("Удалить выбранный заряд")
        self.remove_charge_btn.setToolTip("Удалить выделенный заряд")
        self.remove_charge_btn.clicked.connect(self.remove_charge)
        charge_layout.addWidget(self.remove_charge_btn, 6, 0, 1, 2)

        # Группа "Настройки сетки"
        grid_group = QGroupBox("Настройки сетки")
        grid_layout = QGridLayout()
        grid_group.setLayout(grid_layout)
        self.xmin_input = QLineEdit(str(self.x_min))
        self.xmin_input.setToolTip("Введите минимальное значение x")
        self.xmax_input = QLineEdit(str(self.x_max))
        self.xmax_input.setToolTip("Введите максимальное значение x")
        self.ymin_input = QLineEdit(str(self.y_min))
        self.ymin_input.setToolTip("Введите минимальное значение y")
        self.ymax_input = QLineEdit(str(self.y_max))
        self.ymax_input.setToolTip("Введите максимальное значение y")
        grid_layout.addWidget(QLabel("x_min:"), 0, 0)
        grid_layout.addWidget(self.xmin_input, 0, 1)
        grid_layout.addWidget(QLabel("x_max:"), 1, 0)
        grid_layout.addWidget(self.xmax_input, 1, 1)
        grid_layout.addWidget(QLabel("y_min:"), 2, 0)
        grid_layout.addWidget(self.ymin_input, 2, 1)
        grid_layout.addWidget(QLabel("y_max:"), 3, 0)
        grid_layout.addWidget(self.ymax_input, 3, 1)
        self.update_grid_btn = QPushButton("Обновить сетку")
        self.update_grid_btn.setToolTip("Обновить параметры сетки")
        self.update_grid_btn.clicked.connect(self.update_grid)
        grid_layout.addWidget(self.update_grid_btn, 4, 0, 1, 2)

        # Группа "Дополнительные настройки"
        extra_group = QGroupBox("Дополнительные настройки")
        extra_layout = QGridLayout()
        extra_group.setLayout(extra_layout)
        self.streamlines_checkbox = QCheckBox("Показать линии поля")
        self.streamlines_checkbox.setChecked(True)
        self.streamlines_checkbox.setToolTip("Включить/выключить отображение линий поля")
        self.streamlines_checkbox.stateChanged.connect(self.toggle_streamlines)
        extra_layout.addWidget(self.streamlines_checkbox, 0, 0, 1, 2)
        self.contours_checkbox = QCheckBox("Показать эквипотенциалы")
        self.contours_checkbox.setChecked(True)
        self.contours_checkbox.setToolTip("Включить/выключить отображение эквипотенциалов")
        self.contours_checkbox.stateChanged.connect(self.toggle_contours)
        extra_layout.addWidget(self.contours_checkbox, 1, 0, 1, 2)
        extra_layout.addWidget(QLabel("Цветовая карта:"), 2, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["coolwarm", "viridis", "plasma", "inferno", "magma"])
        self.colormap_combo.setToolTip("Выберите цветовую карту")
        self.colormap_combo.setCurrentText(self.colormap)
        self.colormap_combo.currentTextChanged.connect(self.change_colormap)
        extra_layout.addWidget(self.colormap_combo, 2, 1)
        extra_layout.addWidget(QLabel("Скорость анимации (мс):"), 3, 0)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(self.animation_interval)
        self.speed_slider.setTickInterval(10)
        self.speed_slider.setToolTip("Регулировка скорости анимации")
        self.speed_slider.valueChanged.connect(self.change_animation_speed)
        extra_layout.addWidget(self.speed_slider, 3, 1)
        # Кнопка сброса настроек
        self.reset_btn = QPushButton("Сброс настроек")
        self.reset_btn.setToolTip("Сбросить все параметры к значениям по умолчанию")
        self.reset_btn.clicked.connect(self.reset_settings)
        extra_layout.addWidget(self.reset_btn, 4, 0, 1, 2)

        # Группа "Анимация"
        animation_group = QGroupBox("Анимация")
        animation_layout = QHBoxLayout()
        animation_group.setLayout(animation_layout)
        self.update_plot_btn = QPushButton("Обновить график")
        self.update_plot_btn.setToolTip("Принудительно обновить график")
        self.update_plot_btn.clicked.connect(self.update_plot)
        self.clear_btn = QPushButton("Очистить заряды")
        self.clear_btn.setToolTip("Удалить все добавленные заряды")
        self.clear_btn.clicked.connect(self.clear_charges)
        self.start_animation_btn = QPushButton("Запустить анимацию")
        self.start_animation_btn.setToolTip("Начать анимацию движения зарядов")
        self.start_animation_btn.clicked.connect(self.start_animation)
        self.stop_animation_btn = QPushButton("Остановить анимацию")
        self.stop_animation_btn.setToolTip("Остановить анимацию")
        self.stop_animation_btn.clicked.connect(self.stop_animation)
        self.toggle_canvas_btn = QPushButton("Максимизировать график")
        self.toggle_canvas_btn.setToolTip("Развернуть график на весь экран")
        self.toggle_canvas_btn.clicked.connect(self.toggle_canvas_expansion)
        animation_layout.addWidget(self.update_plot_btn)
        animation_layout.addWidget(self.clear_btn)
        animation_layout.addWidget(self.start_animation_btn)
        animation_layout.addWidget(self.stop_animation_btn)
        animation_layout.addWidget(self.toggle_canvas_btn)

        # Группа "Настройки анимации" (новая)
        anim_settings_group = QGroupBox("Настройки анимации")
        anim_settings_layout = QHBoxLayout()
        anim_settings_group.setLayout(anim_settings_layout)
        anim_settings_layout.addWidget(QLabel("Амплитуда:"))
        self.amplitude_slider = QSlider(Qt.Horizontal)
        self.amplitude_slider.setMinimum(0)
        self.amplitude_slider.setMaximum(200)  # диапазон 0 до 2.0
        self.amplitude_slider.setValue(int(self.amplitude * 100))
        self.amplitude_slider.setTickInterval(10)
        self.amplitude_slider.setToolTip("Изменить амплитуду движения зарядов")
        self.amplitude_slider.valueChanged.connect(self.change_amplitude)
        anim_settings_layout.addWidget(self.amplitude_slider)

        # Компоновка всех групп в панели управления
        controls_layout = QGridLayout()
        controls_layout.addWidget(charge_group, 0, 0)
        controls_layout.addWidget(grid_group, 0, 1)
        controls_layout.addWidget(extra_group, 1, 0, 1, 2)
        controls_layout.addWidget(animation_group, 2, 0, 1, 2)
        controls_layout.addWidget(anim_settings_group, 3, 0, 1, 2)
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        self.controls_widget = controls_widget  # для переключения режима растяжения
        return controls_widget

    # Сброс настроек к значениям по умолчанию
    def reset_settings(self):
        self.x_min, self.x_max = self.default_x_min, self.default_x_max
        self.y_min, self.y_max = self.default_y_min, self.default_y_max
        self.amplitude = self.default_amplitude
        self.animation_interval = self.default_animation_interval
        self.colormap = self.default_colormap
        # Обновляем элементы управления
        self.xmin_input.setText(str(self.x_min))
        self.xmax_input.setText(str(self.x_max))
        self.ymin_input.setText(str(self.y_min))
        self.ymax_input.setText(str(self.y_max))
        self.speed_slider.setValue(self.animation_interval)
        self.amplitude_slider.setValue(int(self.amplitude * 100))
        self.colormap_combo.setCurrentText(self.colormap)
        self.t = 0.0  # сброс времени анимации
        self.statusBar().showMessage("Параметры сброшены")
        self.update_grid()

    def toggle_canvas_expansion(self):
        if not self.canvas_expanded:
            self.controls_widget.hide()
            self.splitter.setSizes([self.height(), 0])
            self.toggle_canvas_btn.setText("Восстановить панель")
            self.canvas_expanded = True
        else:
            self.controls_widget.show()
            self.splitter.setSizes([400, 400])
            self.toggle_canvas_btn.setText("Максимизировать график")
            self.canvas_expanded = False

    def update_plot(self):
        self.canvas.ax.clear()
        self.canvas.ax.set_xlim(self.x_min, self.x_max)
        self.canvas.ax.set_ylim(self.y_min, self.y_max)
        V, Ex, Ey = compute_field_and_potential(self.X, self.Y, self.charges)
        if np.isclose(np.max(V), np.min(V)):
            levels = np.linspace(np.min(V) - 1, np.max(V) + 1, 100)
        else:
            levels = np.linspace(np.min(V), np.max(V), 100)
        cf = self.canvas.ax.contourf(self.X, self.Y, V, levels=levels, cmap=self.colormap, alpha=0.8)
        if self.show_contours:
            cs = self.canvas.ax.contour(self.X, self.Y, V, levels=levels, colors='k', linewidths=0.5, alpha=0.6)
            self.canvas.ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
        if self.cbar is None:
            self.cbar = self.canvas.fig.colorbar(cf, cax=self.cbar_ax)
        else:
            self.cbar.update_normal(cf)
        if self.show_streamlines:
            self.canvas.ax.streamplot(self.X, self.Y, Ex, Ey, color='k', linewidth=1, density=1.5,
                                        arrowstyle='->', arrowsize=1.5)
        pos_label_added = False
        neg_label_added = False
        for charge in self.charges:
            x0, y0 = charge['pos']
            if charge['q'] > 0:
                if not pos_label_added:
                    self.canvas.ax.plot(x0, y0, 'ro', markersize=10, label='Положительный заряд')
                    pos_label_added = True
                else:
                    self.canvas.ax.plot(x0, y0, 'ro', markersize=10)
            else:
                if not neg_label_added:
                    self.canvas.ax.plot(x0, y0, 'bo', markersize=10, label='Отрицательный заряд')
                    neg_label_added = True
                else:
                    self.canvas.ax.plot(x0, y0, 'bo', markersize=10)
        self.canvas.ax.set_xlabel("x", fontsize=12)
        self.canvas.ax.set_ylabel("y", fontsize=12)
        self.canvas.ax.set_title("Линии поля и эквипотенциалы", fontsize=14)
        self.canvas.ax.grid(True, linestyle='--', alpha=0.5)
        if pos_label_added or neg_label_added:
            self.canvas.ax.legend(loc='upper right', fontsize=10)
        info_text = (
            f"Количество зарядов: {len(self.charges)}\n"
            f"Макс потенциал: {np.max(V):.2f}\n"
            f"Мин потенциал: {np.min(V):.2f}"
        )
        self.canvas.ax.text(0.02, 0.98, info_text, transform=self.canvas.ax.transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        self.canvas.draw_idle()
        self.statusBar().showMessage("График обновлён")
        self.update_charge_list()

    def update_charge_list(self):
        self.charge_list.clear()
        for i, charge in enumerate(self.charges):
            x0, y0 = charge.get('initial_pos', charge['pos'])
            self.charge_list.addItem(f"{i}: (x={x0:.2f}, y={y0:.2f}, q={charge['q']:.2f})")

    def add_charge(self):
        try:
            x0 = float(self.x_input.text())
            y0 = float(self.y_input.text())
            q = float(self.q_input.text())
            self.charges.append({'pos': (x0, y0), 'q': q})
            self.statusBar().showMessage(f"Заряд добавлен: ({x0}, {y0}, {q})")
            self.update_charge_list()
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка при добавлении заряда: {e}")

    def remove_charge(self):
        selected_items = self.charge_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Ошибка", "Не выбран заряд для удаления.")
            return
        for item in selected_items:
            index = int(item.text().split(":")[0])
            if 0 <= index < len(self.charges):
                del self.charges[index]
        self.statusBar().showMessage("Выбранный заряд удалён")
        self.update_plot()

    def clear_charges(self):
        self.charges = []
        self.canvas.ax.clear()
        self.canvas.ax.set_xlim(self.x_min, self.x_max)
        self.canvas.ax.set_ylim(self.y_min, self.y_max)
        V_dummy = np.zeros_like(self.X)
        levels = np.linspace(-120, 120, 100)
        cf = self.canvas.ax.contourf(self.X, self.Y, V_dummy, levels=levels, cmap=self.colormap, alpha=0.8)
        self.cbar = self.canvas.fig.colorbar(cf, cax=self.cbar_ax)
        self.canvas.draw_idle()
        self.statusBar().showMessage("Заряды очищены")
        self.update_charge_list()

    def update_grid(self):
        try:
            new_x_min = float(self.xmin_input.text())
            new_x_max = float(self.xmax_input.text())
            new_y_min = float(self.ymin_input.text())
            new_y_max = float(self.ymax_input.text())
            if new_x_min >= new_x_max or new_y_min >= new_y_max:
                QMessageBox.warning(self, "Ошибка",
                                    "Неверные значения: x_min должен быть меньше x_max, а y_min меньше y_max.")
                return
            self.x_min, self.x_max, self.y_min, self.y_max = new_x_min, new_x_max, new_y_min, new_y_max
            self.x = np.linspace(self.x_min, self.x_max, 400)
            self.y = np.linspace(self.y_min, self.y_max, 400)
            self.X, self.Y = np.meshgrid(self.x, self.y)
            self.canvas.ax.set_xlim(self.x_min, self.x_max)
            self.canvas.ax.set_ylim(self.y_min, self.y_max)
            self.update_plot()
            self.statusBar().showMessage("Сетка обновлена")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка при обновлении сетки: {e}")

    def animate_step(self):
        self.t += 0.1  # шаг по времени
        for charge in self.charges:
            if 'initial_pos' not in charge:
                charge['initial_pos'] = charge['pos']
            x0, y0 = charge['initial_pos']
            new_x = x0 + self.amplitude * np.cos(self.t)
            new_y = y0 + self.amplitude * np.sin(self.t)
            charge['pos'] = (new_x, new_y)
        self.update_plot()

    def start_animation(self):
        if self.animation_timer is None:
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self.animate_step)
            self.animation_timer.start(self.animation_interval)
            self.statusBar().showMessage("Анимация запущена")

    def stop_animation(self):
        if self.animation_timer is not None:
            self.animation_timer.stop()
            self.animation_timer = None
            self.statusBar().showMessage("Анимация остановлена")

    def toggle_streamlines(self, state):
        self.show_streamlines = (state == Qt.Checked)
        self.update_plot()

    def toggle_contours(self, state):
        self.show_contours = (state == Qt.Checked)
        self.update_plot()

    def change_colormap(self, text):
        self.colormap = text
        self.update_plot()

    def change_animation_speed(self, value):
        self.animation_interval = value
        if self.animation_timer is not None:
            self.animation_timer.setInterval(self.animation_interval)

    def change_amplitude(self, value):
        self.amplitude = value / 100.0
        self.statusBar().showMessage(f"Амплитуда: {self.amplitude:.2f}")
        self.update_plot()

    def save_image(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить изображение", "",
                                                  "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
                                                  options=options)
        if filename:
            self.canvas.fig.savefig(filename)
            self.statusBar().showMessage(f"Изображение сохранено: {filename}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
