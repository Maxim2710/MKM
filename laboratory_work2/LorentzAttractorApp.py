import sys
import numpy as np
import matplotlib
# Используем backend для PyQt5
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QMessageBox, QTabWidget,
    QAction, QFileDialog, QComboBox, QCheckBox, QSlider, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Функция, описывающая систему уравнений аттрактора Лоренца
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

# Численное интегрирование (метод RK4)
def integrate_lorenz(sigma, rho, beta, init_state, t_max, dt):
    t = np.arange(0, t_max + dt, dt)
    num_steps = len(t)
    sol = np.empty((3, num_steps))
    sol[:, 0] = init_state
    for i in range(num_steps - 1):
        current_t = t[i]
        y_current = sol[:, i]
        k1 = lorenz(current_t, y_current, sigma, rho, beta)
        k2 = lorenz(current_t + dt / 2, y_current + dt / 2 * k1, sigma, rho, beta)
        k3 = lorenz(current_t + dt / 2, y_current + dt / 2 * k2, sigma, rho, beta)
        k4 = lorenz(current_t + dt, y_current + dt * k3, sigma, rho, beta)
        sol[:, i + 1] = y_current + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, sol

# Класс для создания canvas matplotlib
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

# Окно для увеличенного просмотра графика
class EnlargedPlotWindow(QMainWindow):
    def __init__(self, title, t, sol, plot_type, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.setCentralWidget(self.canvas)
        self.plot_data(t, sol, plot_type)
        self.canvas.draw()

    def plot_data(self, t, sol, plot_type):
        if plot_type == '3D':
            ax = self.canvas.fig.add_subplot(111, projection='3d')
            sc = ax.scatter(sol[0], sol[1], sol[2], c=t, cmap='viridis', s=1)
            ax.set_title("Аттрактор Лоренца (3D)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            self.canvas.fig.colorbar(sc, ax=ax, label="Время")
        elif plot_type == 'X':
            ax = self.canvas.fig.add_subplot(111)
            ax.plot(t, sol[0], 'r')
            ax.set_title("X(t)")
            ax.set_xlabel("Время")
            ax.set_ylabel("X")
        elif plot_type == 'Y':
            ax = self.canvas.fig.add_subplot(111)
            ax.plot(t, sol[1], 'g')
            ax.set_title("Y(t)")
            ax.set_xlabel("Время")
            ax.set_ylabel("Y")
        elif plot_type == 'Z':
            ax = self.canvas.fig.add_subplot(111)
            ax.plot(t, sol[2], 'b')
            ax.set_title("Z(t)")
            ax.set_xlabel("Время")
            ax.set_ylabel("Z")

# Новая вкладка для интерактивного управления параметрами
class InteractiveTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        # Для интерактивного обновления используем более короткий промежуток интегрирования
        self.init_state = np.array([1, 1, 1])
        self.t_max = 20
        self.dt = 0.02
        self.sigma = 10
        self.rho = 28
        self.beta = 8/3

    def initUI(self):
        layout = QVBoxLayout(self)

        # Панель управления параметрами: sigma, rho, beta
        controls_layout = QGridLayout()

        # Sigma
        lbl_sigma = QLabel("Sigma:")
        self.slider_sigma = QSlider(Qt.Horizontal)
        self.slider_sigma.setRange(0, 300)  # значения от 0 до 30 (умноженные на 10)
        self.slider_sigma.setValue(100)     # соответствует 10
        self.slider_sigma.setTickInterval(10)
        self.slider_sigma.valueChanged.connect(self.update_params)
        self.edit_sigma = QLineEdit("10")
        self.edit_sigma.setFixedWidth(50)
        self.edit_sigma.setReadOnly(True)
        controls_layout.addWidget(lbl_sigma, 0, 0)
        controls_layout.addWidget(self.slider_sigma, 0, 1)
        controls_layout.addWidget(self.edit_sigma, 0, 2)

        # Rho
        lbl_rho = QLabel("Rho:")
        self.slider_rho = QSlider(Qt.Horizontal)
        self.slider_rho.setRange(0, 500)   # 0-50
        self.slider_rho.setValue(280)      # 28
        self.slider_rho.setTickInterval(5)
        self.slider_rho.valueChanged.connect(self.update_params)
        self.edit_rho = QLineEdit("28")
        self.edit_rho.setFixedWidth(50)
        self.edit_rho.setReadOnly(True)
        controls_layout.addWidget(lbl_rho, 1, 0)
        controls_layout.addWidget(self.slider_rho, 1, 1)
        controls_layout.addWidget(self.edit_rho, 1, 2)

        # Beta
        lbl_beta = QLabel("Beta:")
        self.slider_beta = QSlider(Qt.Horizontal)
        self.slider_beta.setRange(100, 500)  # 1.00 - 5.00 (делим на 100)
        self.slider_beta.setValue(267)       # ~2.67
        self.slider_beta.setTickInterval(5)
        self.slider_beta.valueChanged.connect(self.update_params)
        self.edit_beta = QLineEdit("2.67")
        self.edit_beta.setFixedWidth(50)
        self.edit_beta.setReadOnly(True)
        controls_layout.addWidget(lbl_beta, 2, 0)
        controls_layout.addWidget(self.slider_beta, 2, 1)
        controls_layout.addWidget(self.edit_beta, 2, 2)

        layout.addLayout(controls_layout)

        # Чекбокс автообновления и кнопка обновления
        update_layout = QHBoxLayout()
        self.auto_update_cb = QCheckBox("Автообновление")
        self.auto_update_cb.setChecked(True)
        update_layout.addWidget(self.auto_update_cb)
        self.btn_update = QPushButton("Обновить")
        self.btn_update.clicked.connect(self.plot_interactive)
        update_layout.addWidget(self.btn_update)
        update_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(update_layout)

        # Canvas для интерактивного 3D-графика
        self.interactive_canvas = MplCanvas(self, width=8, height=6, dpi=100)
        layout.addWidget(self.interactive_canvas)

        # Таймер для автообновления (если включено)
        self.interactive_timer = QTimer(self)
        self.interactive_timer.setInterval(500)
        self.interactive_timer.timeout.connect(self.plot_interactive)
        if self.auto_update_cb.isChecked():
            self.interactive_timer.start()
        self.auto_update_cb.toggled.connect(self.toggle_interactive_timer)

    def toggle_interactive_timer(self, checked):
        if checked:
            self.interactive_timer.start()
        else:
            self.interactive_timer.stop()

    def update_params(self):
        # Обновляем числовые поля от слайдеров
        self.sigma = self.slider_sigma.value() / 10.0
        self.rho = self.slider_rho.value() / 10.0
        self.beta = self.slider_beta.value() / 100.0
        self.edit_sigma.setText(f"{self.sigma:.1f}")
        self.edit_rho.setText(f"{self.rho:.1f}")
        self.edit_beta.setText(f"{self.beta:.2f}")
        if self.auto_update_cb.isChecked():
            self.plot_interactive()

    def plot_interactive(self):
        # Выполняем интегрирование для интерактивного режима
        t, sol = integrate_lorenz(self.sigma, self.rho, self.beta, self.init_state, self.t_max, self.dt)
        self.interactive_canvas.fig.clf()
        ax = self.interactive_canvas.fig.add_subplot(111, projection='3d')
        sc = ax.scatter(sol[0], sol[1], sol[2], c=t, cmap='plasma', s=1)
        ax.set_title("Интерактивный Аттрактор Лоренца")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        self.interactive_canvas.fig.colorbar(sc, ax=ax, label="Время")
        self.interactive_canvas.draw()

# Основное окно приложения с улучшенным интерфейсом
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Аттрактор Лоренца")
        self.setGeometry(100, 100, 1200, 800)
        self.dark_mode = False
        self.initUI()
        self.realtime_timer = QTimer(self)
        self.realtime_timer.setInterval(2000)
        self.realtime_timer.timeout.connect(self.plot_static)

    def initUI(self):
        self.create_menu()
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Группа ввода параметров с подсказками
        params_group = QGroupBox("Параметры модели")
        params_layout = QGridLayout()
        self.sigma_edit = QLineEdit("10")
        self.sigma_edit.setToolTip("Значение sigma, например, 10")
        self.rho_edit = QLineEdit("28")
        self.rho_edit.setToolTip("Значение rho, например, 28")
        self.beta_edit = QLineEdit(str(8 / 3))
        self.beta_edit.setToolTip("Значение beta, например, 8/3")
        self.x0_edit = QLineEdit("1")
        self.x0_edit.setToolTip("Начальное значение x, например, 1")
        self.y0_edit = QLineEdit("1")
        self.y0_edit.setToolTip("Начальное значение y, например, 1")
        self.z0_edit = QLineEdit("1")
        self.z0_edit.setToolTip("Начальное значение z, например, 1")
        self.tmax_edit = QLineEdit("40")
        self.tmax_edit.setToolTip("Время моделирования, например, 40")
        self.dt_edit = QLineEdit("0.01")
        self.dt_edit.setToolTip("Шаг dt, например, 0.01")

        params_layout.addWidget(QLabel("Sigma:"), 0, 0)
        params_layout.addWidget(self.sigma_edit, 0, 1)
        params_layout.addWidget(QLabel("Rho:"), 1, 0)
        params_layout.addWidget(self.rho_edit, 1, 1)
        params_layout.addWidget(QLabel("Beta:"), 2, 0)
        params_layout.addWidget(self.beta_edit, 2, 1)
        params_layout.addWidget(QLabel("Начальное X:"), 3, 0)
        params_layout.addWidget(self.x0_edit, 3, 1)
        params_layout.addWidget(QLabel("Начальное Y:"), 4, 0)
        params_layout.addWidget(self.y0_edit, 4, 1)
        params_layout.addWidget(QLabel("Начальное Z:"), 5, 0)
        params_layout.addWidget(self.z0_edit, 5, 1)
        params_layout.addWidget(QLabel("Время моделирования:"), 6, 0)
        params_layout.addWidget(self.tmax_edit, 6, 1)
        params_layout.addWidget(QLabel("Шаг dt:"), 7, 0)
        params_layout.addWidget(self.dt_edit, 7, 1)
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # Кнопки управления
        buttons_layout = QHBoxLayout()
        self.plot_button = QPushButton("Построить графики")
        self.plot_button.setToolTip("Построить статические графики аттрактора")
        self.anim_button = QPushButton("Запустить анимацию")
        self.anim_button.setToolTip("Запустить анимацию аттрактора")
        self.reset_button = QPushButton("Сброс параметров")
        self.reset_button.setToolTip("Сбросить параметры к значениям по умолчанию")
        self.realtime_checkbox = QCheckBox("Режим реального времени")
        self.realtime_checkbox.setToolTip("Автоматически обновлять графики каждые 2 секунды")
        self.realtime_checkbox.toggled.connect(self.toggle_realtime)
        buttons_layout.addWidget(self.plot_button)
        buttons_layout.addWidget(self.anim_button)
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addWidget(self.realtime_checkbox)
        main_layout.addLayout(buttons_layout)

        # Виджет с вкладками: Статические графики, Анимация и Интерактивное управление
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Вкладка статических графиков
        self.static_tab = QWidget()
        static_layout = QVBoxLayout(self.static_tab)
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Цветовая карта для 3D:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "jet"])
        colormap_layout.addWidget(self.colormap_combo)
        colormap_layout.addStretch()
        static_layout.addLayout(colormap_layout)
        self.static_canvas = MplCanvas(self, width=8, height=6, dpi=100)
        static_layout.addWidget(self.static_canvas)
        self.static_toolbar = NavigationToolbar(self.static_canvas, self)
        static_layout.addWidget(self.static_toolbar)
        enlarge_layout = QHBoxLayout()
        self.enlarge_3d_button = QPushButton("Увеличить 3D-график")
        self.enlarge_x_button = QPushButton("Увеличить X(t)")
        self.enlarge_y_button = QPushButton("Увеличить Y(t)")
        self.enlarge_z_button = QPushButton("Увеличить Z(t)")
        enlarge_layout.addWidget(self.enlarge_3d_button)
        enlarge_layout.addWidget(self.enlarge_x_button)
        enlarge_layout.addWidget(self.enlarge_y_button)
        enlarge_layout.addWidget(self.enlarge_z_button)
        static_layout.addLayout(enlarge_layout)
        self.tabs.addTab(self.static_tab, "Статические графики")

        # Вкладка анимации
        self.anim_tab = QWidget()
        anim_layout = QVBoxLayout(self.anim_tab)
        self.anim_canvas = MplCanvas(self, width=8, height=6, dpi=100)
        anim_layout.addWidget(self.anim_canvas)
        self.anim_toolbar = NavigationToolbar(self.anim_canvas, self)
        anim_layout.addWidget(self.anim_toolbar)
        anim_control_layout = QHBoxLayout()
        self.anim_play_pause_button = QPushButton("Пауза")
        self.anim_play_pause_button.setToolTip("Приостановить/Возобновить анимацию")
        self.anim_play_pause_button.clicked.connect(self.toggle_anim_play_pause)
        self.save_video_button = QPushButton("Сохранить видео")
        self.save_video_button.setToolTip("Сохранить анимацию в видео (требуется FFmpeg)")
        self.save_video_button.clicked.connect(self.save_video)
        anim_control_layout.addWidget(self.anim_play_pause_button)
        anim_control_layout.addWidget(self.save_video_button)
        anim_control_layout.addStretch()
        anim_layout.addLayout(anim_control_layout)
        self.tabs.addTab(self.anim_tab, "Анимация")

        # Новая вкладка для интерактивного управления
        self.interactive_tab = InteractiveTab(self)
        self.tabs.addTab(self.interactive_tab, "Интерактивное управление")

        # Связываем сигналы с обработчиками
        self.plot_button.clicked.connect(self.plot_static)
        self.anim_button.clicked.connect(self.plot_animation)
        self.reset_button.clicked.connect(self.reset_parameters)
        self.enlarge_3d_button.clicked.connect(lambda: self.enlarge_plot('3D'))
        self.enlarge_x_button.clicked.connect(lambda: self.enlarge_plot('X'))
        self.enlarge_y_button.clicked.connect(lambda: self.enlarge_plot('Y'))
        self.enlarge_z_button.clicked.connect(lambda: self.enlarge_plot('Z'))

        self.statusBar().showMessage("Готово")

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Файл")
        save_action = QAction("Сохранить график", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_plot)
        exit_action = QAction("Выход", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        settings_menu = menubar.addMenu("Настройки")
        dark_mode_action = QAction("Переключить темный режим", self)
        dark_mode_action.triggered.connect(self.toggle_dark_mode)
        settings_menu.addAction(dark_mode_action)
        help_menu = menubar.addMenu("Помощь")
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def toggle_dark_mode(self):
        if not self.dark_mode:
            self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: #ffffff; }
            QLabel, QLineEdit, QPushButton, QGroupBox, QTabWidget::pane { background-color: #3c3f41; color: #ffffff; }
            QTabBar::tab { background: #3c3f41; padding: 5px; }
            QTabBar::tab:selected { background: #2b2b2b; }
            QSlider::handle:horizontal { background: #d3d3d3; }
            """)
            self.dark_mode = True
            self.statusBar().showMessage("Темный режим включен")
        else:
            self.setStyleSheet("")
            self.dark_mode = False
            self.statusBar().showMessage("Темный режим выключен")

    def show_about(self):
        QMessageBox.about(self, "О программе",
                          "Аттрактор Лоренца – инновационный GUI интерфейс\n"
                          "Разработано на PyQt5 и Matplotlib.\n"
                          "Добавлена интерактивная панель для динамического изменения параметров.")

    def save_plot(self):
        current_index = self.tabs.currentIndex()
        if current_index == 0:
            filename, _ = QFileDialog.getSaveFileName(self, "Сохранить график", "", "PNG Image (*.png);;All Files (*)")
            if filename:
                self.static_canvas.fig.savefig(filename)
                self.statusBar().showMessage("Статический график сохранен")
        elif current_index == 1:
            filename, _ = QFileDialog.getSaveFileName(self, "Сохранить кадр анимации", "",
                                                      "PNG Image (*.png);;All Files (*)")
            if filename:
                self.anim_canvas.fig.savefig(filename)
                self.statusBar().showMessage("Кадр анимации сохранен")

    def reset_parameters(self):
        self.sigma_edit.setText("10")
        self.rho_edit.setText("28")
        self.beta_edit.setText(str(8 / 3))
        self.x0_edit.setText("1")
        self.y0_edit.setText("1")
        self.z0_edit.setText("1")
        self.tmax_edit.setText("40")
        self.dt_edit.setText("0.01")
        self.statusBar().showMessage("Параметры сброшены")

    def toggle_realtime(self, checked):
        if checked:
            self.realtime_timer.start()
            self.statusBar().showMessage("Режим реального времени включен")
        else:
            self.realtime_timer.stop()
            self.statusBar().showMessage("Режим реального времени выключен")

    def get_parameters(self):
        try:
            sigma = float(self.sigma_edit.text())
            rho = float(self.rho_edit.text())
            beta = float(self.beta_edit.text())
            x0 = float(self.x0_edit.text())
            y0 = float(self.y0_edit.text())
            z0 = float(self.z0_edit.text())
            tmax = float(self.tmax_edit.text())
            dt = float(self.dt_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Введите корректные числовые значения!")
            return None
        return sigma, rho, beta, np.array([x0, y0, z0]), tmax, dt

    def plot_static(self):
        params = self.get_parameters()
        if params is None:
            return
        sigma, rho, beta, init_state, tmax, dt = params
        t, sol = integrate_lorenz(sigma, rho, beta, init_state, tmax, dt)
        self.t = t
        self.sol = sol
        self.static_canvas.fig.clf()
        ax1 = self.static_canvas.fig.add_subplot(221, projection='3d')
        cmap = self.colormap_combo.currentText()
        sc = ax1.scatter(sol[0], sol[1], sol[2], c=t, cmap=cmap, s=1)
        ax1.set_title("Аттрактор Лоренца")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        self.static_canvas.fig.colorbar(sc, ax=ax1, label="Время")
        ax2 = self.static_canvas.fig.add_subplot(222)
        ax2.plot(t, sol[0], 'r')
        ax2.set_title("X(t)")
        ax2.set_xlabel("Время")
        ax2.set_ylabel("X")
        ax3 = self.static_canvas.fig.add_subplot(223)
        ax3.plot(t, sol[1], 'g')
        ax3.set_title("Y(t)")
        ax3.set_xlabel("Время")
        ax3.set_ylabel("Y")
        ax4 = self.static_canvas.fig.add_subplot(224)
        ax4.plot(t, sol[2], 'b')
        ax4.set_title("Z(t)")
        ax4.set_xlabel("Время")
        ax4.set_ylabel("Z")
        self.static_canvas.fig.tight_layout()
        self.static_canvas.draw()
        self.statusBar().showMessage("Статические графики построены")

    def plot_animation(self):
        if not hasattr(self, 't') or not hasattr(self, 'sol'):
            QMessageBox.warning(self, "Внимание", "Сначала постройте статические графики!")
            return
        sol = self.sol
        self.anim_canvas.fig.clf()
        ax = self.anim_canvas.fig.add_subplot(111, projection='3d')
        ax.set_title("Анимация аттрактора Лоренца")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(np.min(sol[0]), np.max(sol[0]))
        ax.set_ylim(np.min(sol[1]), np.max(sol[1]))
        ax.set_zlim(np.min(sol[2]), np.max(sol[2]))
        self.line, = ax.plot([], [], [], lw=2, color='b')

        def init():
            self.line.set_data([], [])
            self.line.set_3d_properties([])
            return self.line,

        def update(frame):
            self.line.set_data(sol[0, :frame], sol[1, :frame])
            self.line.set_3d_properties(sol[2, :frame])
            return self.line,

        self.ani = animation.FuncAnimation(self.anim_canvas.fig, update,
                                           frames=sol.shape[1], init_func=init,
                                           interval=10, blit=True)
        self.anim_canvas.draw()
        self.anim_running = True  # Флаг, указывающий, что анимация запущена
        self.statusBar().showMessage("Анимация запущена")
        self.tabs.setCurrentWidget(self.anim_tab)
        self.anim_play_pause_button.setText("Пауза")

    def toggle_anim_play_pause(self):
        if not hasattr(self, 'ani'):
            return
        if self.anim_running:
            self.ani.event_source.stop()
            self.anim_play_pause_button.setText("Возобновить")
            self.statusBar().showMessage("Анимация приостановлена")
            self.anim_running = False
        else:
            self.ani.event_source.start()
            self.anim_play_pause_button.setText("Пауза")
            self.statusBar().showMessage("Анимация возобновлена")
            self.anim_running = True

    def save_video(self):
        if not hasattr(self, 'ani'):
            QMessageBox.warning(self, "Внимание", "Сначала запустите анимацию!")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить видео", "", "MP4 Video (*.mp4);;All Files (*)")
        if filename:
            try:
                self.ani.save(filename, writer='ffmpeg', dpi=100)
                self.statusBar().showMessage("Видео сохранено")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить видео:\n{e}")

    def enlarge_plot(self, plot_type):
        if not hasattr(self, 't') or not hasattr(self, 'sol'):
            QMessageBox.warning(self, "Внимание", "Сначала постройте графики!")
            return
        title = ""
        if plot_type == '3D':
            title = "Увеличенный 3D-график Аттрактора Лоренца"
        elif plot_type == 'X':
            title = "Увеличенный график X(t)"
        elif plot_type == 'Y':
            title = "Увеличенный график Y(t)"
        elif plot_type == 'Z':
            title = "Увеличенный график Z(t)"
        self.enlarged_window = EnlargedPlotWindow(title, self.t, self.sol, plot_type)
        self.enlarged_window.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
