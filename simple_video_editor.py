import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QCheckBox, QFileDialog, QProgressBar,QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal,QCoreApplication,pyqtSlot
import time

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.frame = None
        self.total_frames = 0

    def open_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError("Error opening video file")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self, frame_index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame

    def apply_gaussian_blur(self, kernel_size):
        if self.frame is not None:
            self.frame = cv2.GaussianBlur(self.frame, (kernel_size, kernel_size), 0)

    def apply_threshold(self, threshold_value):
        if self.frame is not None:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
            self.frame = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def convert_to_grayscale(self):
        if self.frame is not None:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def adjust_brightness_contrast(self, brightness, contrast):
        if self.frame is not None:
            alpha = (contrast + 100) / 100.0
            beta = brightness
            adjusted_frame = cv2.convertScaleAbs(self.frame, alpha=alpha, beta=beta)
            min_val = np.min(adjusted_frame)
            max_val = np.max(adjusted_frame)
            if min_val < 0:
                adjusted_frame = adjusted_frame - min_val
            if max_val > 255:
                adjusted_frame = adjusted_frame * (255 / max_val)
            self.frame = np.clip(adjusted_frame, 0, 255)

    def apply_bilateral_filter(self, d, sigma_color, sigma_space):
        if self.frame is not None:
            self.frame = cv2.bilateralFilter(self.frame, d, sigma_color, sigma_space)

    def apply_sharpening_filter(self, kernel_size):
        if self.frame is not None:
            kernel = np.zeros((kernel_size, kernel_size), np.float32)
            kernel[int((kernel_size - 1) / 2), int((kernel_size - 1) / 2)] = 2.0
            self.frame = cv2.filter2D(self.frame, -1, kernel)

    def apply_gamma_transform(self, gamma):
        if self.frame is not None:
            gamma_corrected = np.power(self.frame / 255.0, gamma) * 255.0
            self.frame = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

    def close_video(self):
        if self.cap is not None:
            self.cap.release()

class VideoSaveThread(QThread):
    progress_updated = pyqtSignal(int)

    def __init__(self, video_processor,video_path,save_path,brightness,contrast,check_status):
        super().__init__()
        self.video_processor = video_processor
        self.file_path = video_path
        self.save_path = save_path
        self.brightness = brightness
        self.contrast = contrast
        self.check_status = check_status


    def run(self):
        video_path = self.file_path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        save_path = self.save_path

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))

        for frame_index in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            self.video_processor.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_processor.adjust_brightness_contrast(self.brightness, self.contrast)
            self.video_processor.apply_gaussian_blur(self.check_status['blur']['kernel_size'])
            if self.check_status['binary']['status']:
                self.video_processor.apply_threshold(self.check_status['binary']['threshold'])
                print('Binary threshold applied')


            adjusted_frame = self.video_processor.frame

            out.write(adjusted_frame)
            progress = int((frame_index + 1) * 100 / total_frames)
            self.progress_updated.emit(progress)
            print(progress)

            # 处理待处理的事件队列，确保主线程有足够时间处理信号和更新进度条
            QCoreApplication.processEvents()

        out.release()
        cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Processor")
        self.video_processor = None
        self.playing = False
        self.current_frame = None
        self.check_status = {'binary':{'status':False,'threshold':0},
                             'gray':{'status':False,'threshold':0},
                             'blur':{'kernel_size':0}}

        # Create widgets
        self.video_label = QLabel()
        self.save_button = QPushButton("Save")
        self.open_button = QPushButton("Open")
        self.play_button = QPushButton("Play/Pause")
        self.blur_slider = QSlider(Qt.Horizontal)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.gray_checkbox = QCheckBox("Grayscale")
        self.Binary_checkbox = QCheckBox("Binary")
        self.brightness_label = QLabel("Brightness:")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.contrast_label = QLabel("Contrast:")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.progress_bar = QProgressBar()
        self.slider = QSlider(Qt.Horizontal)
        self.brightness_vlabel = QLabel("value:0")
        self.contrast_vlabel = QLabel("value:0")
        self.blur_vlabel = QLabel("value:0")
        self.binary_vlabel = QLabel("value:0")
        ##双边滤波和锐化算子
        self.bilateral_label = QLabel("Bilateral:")
        self.bilateral_slider = QSlider(Qt.Horizontal)
        self.bilateral_vlabel = QLabel("value:0")
        self.sharpen_label = QLabel("Sharpen:")
        self.sharpen_slider = QSlider(Qt.Horizontal)
        self.sharpen_vlabel = QLabel("value:0")

        self.gamma_label = QLabel("Gamma:")
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_vlabel = QLabel("value:1.0")

        # Set widget properties
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.blur_slider.setRange(1, 67)
        self.threshold_slider.setRange(0, 255)
        self.brightness_vlabel.setAlignment(Qt.AlignCenter)
        self.contrast_vlabel.setAlignment(Qt.AlignCenter)
        self.brightness_slider.setRange(-100, 100)
        self.contrast_slider.setRange(-100, 100)
        self.progress_bar.setRange(0, 100)
        self.slider.setRange(0, 100)

        self.bilateral_slider.setRange(0, 100)  # 设置双边滤波滑块的长度范围
        self.sharpen_slider.setRange(0, 20)  # 设置锐化算子滑块的长度范围
        self.gamma_slider.setRange(0,25)
        # Create layouts
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.save_button)
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.open_button)
        control_layout.addWidget(QLabel("Blur:"))
        control_layout.addWidget(self.blur_vlabel)
        control_layout.addWidget(self.blur_slider)
        control_layout.addWidget(self.binary_vlabel)
        control_layout.addWidget(self.Binary_checkbox)
        control_layout.addWidget(QLabel("Threshold:"))
        control_layout.addWidget(self.threshold_slider)
        control_layout.addWidget(self.gray_checkbox)
        control_layout.addWidget(self.brightness_vlabel)
        control_layout.addWidget(self.brightness_label)
        control_layout.addWidget(self.brightness_slider)
        control_layout.addWidget(self.contrast_vlabel)
        control_layout.addWidget(self.contrast_label)
        control_layout.addWidget(self.contrast_slider)

        control_layout.addWidget(self.bilateral_vlabel)
        control_layout.addWidget(self.bilateral_label)
        control_layout.addWidget(self.bilateral_slider)
        control_layout.addWidget(self.sharpen_vlabel)
        control_layout.addWidget(self.sharpen_label)
        control_layout.addWidget(self.sharpen_slider)
        control_layout.addWidget(self.gamma_vlabel)
        control_layout.addWidget(self.gamma_label)
        control_layout.addWidget(self.gamma_slider)


        main_layout = QVBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.slider)

        # Create main widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Connect signals to slots
        self.open_button.clicked.connect(self.open_video)
        self.save_button.clicked.connect(self.save_video)
        self.play_button.clicked.connect(self.toggle_play)
        self.blur_slider.valueChanged.connect(self.update_blur)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.gray_checkbox.stateChanged.connect(self.update_grayscale)
        self.Binary_checkbox.stateChanged.connect(self.update_threshold)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        self.slider.valueChanged.connect(self.update_video_position)

        self.bilateral_slider.valueChanged.connect(self.update_bilateral)
        self.sharpen_slider.valueChanged.connect(self.update_sharpen)
        self.gamma_slider.valueChanged.connect(self.update_gamma)

        self.blur_slider.valueChanged.connect(self.update_blur_label)
        self.threshold_slider.valueChanged.connect(self.update_binary_label)
        self.brightness_slider.valueChanged.connect(self.update_brightness_label)
        self.contrast_slider.valueChanged.connect(self.update_contrast_label)
        # Timer for updating frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Connect progress bar signal
        self.progress_bar.valueChanged.connect(self.seek_video)

    def open_video(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, "Open Video", filter="Video Files (*.mp4 *.avi *.mkv)")
        if video_path:
            self.video_processor = VideoProcessor(video_path)
            self.video_processor.open_video()
            self.timer.start(33)
            self.playing = True
            self.play_button.setText("Pause")
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(self.video_processor.total_frames - 1)
            self.slider.setValue(0)
            self.update_frame()

    def update_frame(self):
        if self.video_processor is not None:
            frame_index = self.progress_bar.value()
            self.video_processor.read_frame(frame_index)
            if self.video_processor.frame is not None:
                self.current_frame = self.video_processor.frame.copy()
                self.video_processor.adjust_brightness_contrast(self.brightness_slider.value(), self.contrast_slider.value())

                kernel_size = self.blur_slider.value()
                if kernel_size % 2 == 0:
                    kernel_size += 1
                self.video_processor.apply_gaussian_blur(kernel_size)

                if self.Binary_checkbox.isChecked():
                    threshold_value = self.threshold_slider.value()
                    self.video_processor.apply_threshold(threshold_value)

                if self.gray_checkbox.isChecked():
                    self.video_processor.convert_to_grayscale()

                gamma_value = self.gamma_slider.value() / 10.0  # 根据需要调整范围和步长

                if gamma_value > 0:
                    self.video_processor.apply_gamma_transform(gamma_value)

                sharpen_value = self.sharpen_slider.value()

                bilateral_value = self.bilateral_slider.value()
                sigma_color = 100  # 根据需要调整参数
                sigma_space = 5 # 根据需要调整参数

                if bilateral_value > 0:
                    self.video_processor.apply_bilateral_filter(bilateral_value, sigma_color, sigma_space)

                # 继续处理其他功能的代码

                if sharpen_value > 0:
                    self.video_processor.apply_sharpening_filter(sharpen_value)

                adjusted_frame = self.video_processor.frame
                height, width, _ = adjusted_frame.shape
                adjusted_frame = QImage(adjusted_frame.data, width, height, QImage.Format_RGB888)
                adjusted_frame = QPixmap.fromImage(adjusted_frame)
                self.video_label.setPixmap(adjusted_frame.scaled(self.video_label.size(), Qt.KeepAspectRatio))

    def update_blur(self):
        kernel_size = self.blur_slider.value()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.video_processor.apply_gaussian_blur(kernel_size)
        self.update_frame()

    def update_threshold(self):
        if self.Binary_checkbox.isChecked():
            threshold_value = self.threshold_slider.value()
            self.video_processor.apply_threshold(threshold_value)
        else:
            self.update_frame()

    def update_grayscale(self):
        if self.gray_checkbox.isChecked():
            self.video_processor.convert_to_grayscale()
        else:
            self.update_frame()

    def update_brightness(self):
        brightness = self.brightness_slider.value()
        self.video_processor.adjust_brightness_contrast(brightness, self.contrast_slider.value())
        self.update_frame()

    def update_contrast(self):
        contrast = self.contrast_slider.value()
        self.video_processor.adjust_brightness_contrast(self.brightness_slider.value(), contrast)
        self.update_frame()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.timer.start(33)
            self.play_button.setText("Pause")
        else:
            self.timer.stop()
            self.play_button.setText("Play")

    def seek_video(self, value):
        self.timer.stop()
        self.update_frame()
        self.timer.start(33)

    def update_contrast_label(self,value):
        self.contrast_vlabel.setText(f'value:{value}')
    def update_brightness_label(self,value):
        self.brightness_vlabel.setText(f'value:{value}')
    def update_blur_label(self,value):
        self.blur_vlabel.setText(f'value:{value}')
    def update_binary_label(self,value):
        self.binary_vlabel.setText(f'value:{value}')

    def save_video(self):
        if self.video_processor is not None:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(self, "Save Video", filter="MP4 (*.mp4)")
            self.check_status["binary"]["status"] = self.Binary_checkbox.isChecked()
            self.check_status["binary"]["threshold"] = self.threshold_slider.value()
            self.check_status["blur"]["kernel_size"] = self.blur_slider.value()
            if file_path:
                self.video_save_thread = VideoSaveThread(self.video_processor,self.video_processor.video_path, file_path,self.brightness_slider.value(),self.contrast_slider.value(),self.check_status)
                self.video_save_thread.progress_updated.connect(self.progress_bar.setValue)
                self.video_save_thread.start()

    def update_bilateral_label(self, value):
        self.bilateral_vlabel.setText(f"value:{value}")

    def update_sharpen_label(self, value):
        self.sharpen_vlabel.setText(f"value:{value}")

    @pyqtSlot(int)
    def update_bilateral(self, value):
        self.update_bilateral_label(value)
        self.update_frame()

    @pyqtSlot(int)
    def update_sharpen(self, value):
        self.update_sharpen_label(value)
        self.update_frame()

    @pyqtSlot(int)
    def update_gamma(self, value):
        gamma_value = value / 10.0  # 根据需要调整范围和步长
        self.gamma_vlabel.setText(f"value:{gamma_value}")
        self.update_frame()

    def update_video_position(self, value):
        if self.video_processor is not None:
            frame_index = self.video_processor.total_frames * value // 100
            self.progress_bar.setValue(frame_index)
            self.update_frame()

    def closeEvent(self, event):
        if self.video_processor is not None:
            self.video_processor.close_video()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.open_video()
    window.show()
    sys.exit(app.exec_())
