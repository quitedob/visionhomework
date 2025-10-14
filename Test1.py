# OpenCV图像处理实验 - Tkinter GUI版本 (性能优化版)
# 交互式演示基本的图像处理操作

import cv2  # OpenCV计算机视觉库
import numpy as np  # 数值计算库
import tkinter as tk  # Python标准GUI库
from tkinter import filedialog, ttk, messagebox  # Tkinter子模块
from PIL import Image, ImageTk  # Python图像处理库
import os  # 操作系统接口
import threading  # 多线程支持
import queue  # 线程安全队列
import weakref  # 弱引用缓存
from functools import lru_cache  # LRU缓存装饰器

class ImageProcessingExperiment:
    """
    OpenCV图像处理实验主类

    提供交互式的图像处理实验演示，包括：
    - 图像读取和显示
    - 图像保存格式对比（JPEG质量、PNG压缩）
    - 图像创建和复制
    - RGB通道分离和合并
    - 色彩空间转换
    - 灰度图像分离演示

    支持中英文界面切换，步骤化引导学习
    """
    def __init__(self):
        """
        初始化图像处理实验应用程序

        创建Tkinter主窗口，初始化所有变量和数据结构，
        设置多语言支持，默认使用中文界面
        """
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("OpenCV图像处理实验 (性能优化版)")
        self.root.geometry("1400x900")  # 窗口大小：1400x900像素
        self.root.resizable(True, True)  # 允许调整窗口大小

        # 设置现代化的窗口样式
        self.root.configure(bg='#f5f5f5')

        # 初始化实例变量
        self.original_image = None  # 存储原始图像（OpenCV格式）
        self.current_image_path = ""  # 当前图像文件路径
        self.current_step = 0  # 当前实验步骤（0-7）
        self.total_steps = 7  # 总实验步骤数
        self.current_language = "zh"  # 当前语言，默认为中文

        # 性能优化相关变量
        self.image_cache = weakref.WeakValueDictionary()  # 图像缓存
        self.processing_queue = queue.Queue()  # 处理队列
        self.current_thread = None  # 当前处理线程
        self.is_processing = False  # 处理状态标志
        self.photo_cache = {}  # PIL图像缓存
        self.max_cache_size = 10  # 最大缓存大小
        self.canvas_update_timer = None  # 画布更新计时器

        # 多语言文本资源
        # 支持中文和英文界面切换
        self.text_resources = {
            "zh": {
                "app_title": "OpenCV图像处理实验",
                "current_step": "当前步骤",
                "select_image": "选择图片",
                "previous": "上一步",
                "next": "下一步",
                "language": "语言",
                "image_display": "图像显示",
                "status_ready": "准备开始，请选择一张图片。",
                "status_loaded": "图片已加载: ",
                "status_no_selection": "未选择图片。",
                "steps": {
                    0: {
                        "title": "选择图片",
                        "description": "选择一个图片文件开始实验"
                    },
                    1: {
                        "title": "步骤1: 图像读取和显示",
                        "description": "显示原始图像并了解基本属性"
                    },
                    2: {
                        "title": "步骤2: 图像保存 - 质量对比",
                        "description": "对比JPEG不同质量设置的效果"
                    },
                    3: {
                        "title": "步骤3: 图像保存 - 压缩对比",
                        "description": "对比PNG不同压缩级别效果"
                    },
                    4: {
                        "title": "步骤4: 分离灰度图",
                        "description": "了解对单通道灰度图像使用split()函数的结果"
                    },
                    5: {
                        "title": "步骤5: 图像创建和复制",
                        "description": "创建新图像并了解复制操作"
                    },
                    6: {
                        "title": "步骤6: RGB通道分离",
                        "description": "分离并显示RGB三个颜色通道"
                    },
                    7: {
                        "title": "步骤7: 通道合并与色彩空间",
                        "description": "合并通道并了解BGR与RGB色彩空间差异"
                    }
                }
            },
            "en": {
                "app_title": "OpenCV Image Processing Experiment",
                "current_step": "Current Step",
                "select_image": "Select Image",
                "previous": "Previous",
                "next": "Next",
                "language": "Language",
                "image_display": "Image Display",
                "status_ready": "Ready to start. Please select an image.",
                "status_loaded": "Image loaded: ",
                "status_no_selection": "No image selected.",
                "steps": {
                    0: {
                        "title": "Select Image",
                        "description": "Choose an image file to begin the experiment"
                    },
                    1: {
                        "title": "Step 1: Image Loading & Display",
                        "description": "Display the original image and understand basic properties"
                    },
                    2: {
                        "title": "Step 2: Image Saving - Quality Comparison",
                        "description": "Compare JPEG images with different quality settings"
                    },
                    3: {
                        "title": "Step 3: Image Saving - Compression Comparison",
                        "description": "Compare PNG images with different compression levels"
                    },
                    4: {
                        "title": "Step 4: Grayscale Image Split",
                        "description": "Understand the result of using split() on single-channel grayscale images"
                    },
                    5: {
                        "title": "Step 5: Image Creation & Copying",
                        "description": "Create new images and understand copying operations"
                    },
                    6: {
                        "title": "Step 6: RGB Channel Separation",
                        "description": "Split image into individual color channels (B, G, R)"
                    },
                    7: {
                        "title": "Step 7: Channel Merging & Color Spaces",
                        "description": "Merge channels and understand BGR vs RGB color spaces"
                    }
                }
            }
        }

        # Image labels (keep in English to prevent encoding issues)
        self.image_labels = {
            "step2_title": "JPEG Quality Comparison",
            "step2_label1": "High Quality (default 95)",
            "step2_label2": "Low Quality (default 10)",
            "step3_title": "PNG Compression Comparison",
            "step3_label1": "Low Compression (default 0)",
            "step3_label2": "High Compression (default 9)",
            "step4_title": "Grayscale Image Split",
            "step4_label1": "Color Image",
            "step4_label2": "Grayscale Split Result",
            "step5_title": "Image Creation & Copying",
            "step5_label1": "Original Image",
            "step5_label2": "Black Image (np.zeros)",
            "step6_title": "RGB Channel Separation",
            "step6_label1": "Original Image",
            "step6_label2": "Blue Channel (B)",
            "step7_title": "BGR vs RGB Color Spaces",
            "step7_label1": "BGR Format (OpenCV)",
            "step7_label2": "RGB Format (Matplotlib)"
        }

        self.setup_gui()
        self.create_widgets()
        self.update_language()  # Set initial language

    # 性能优化相关方法
    def clear_cache(self):
        """清理缓存以释放内存"""
        self.image_cache.clear()
        self.photo_cache.clear()

    @lru_cache(maxsize=20)
    def get_resized_image_key(self, width, height, image_hash):
        """生成缓存键"""
        return f"{width}x{height}_{image_hash}"

    def cache_image(self, image, key):
        """缓存图像（弱引用）"""
        if len(self.image_cache) >= self.max_cache_size:
            # 清理一半缓存
            keys_to_remove = list(self.image_cache.keys())[:self.max_cache_size // 2]
            for k in keys_to_remove:
                if k in self.image_cache:
                    del self.image_cache[k]
        self.image_cache[key] = image

    def get_cached_image(self, key):
        """获取缓存图像"""
        return self.image_cache.get(key)

    def process_image_in_background(self, task_func, *args, **kwargs):
        """在后台线程中处理图像以避免GUI阻塞"""
        if self.is_processing:
            self.stop_background_processing()

        self.processing_queue = queue.Queue()
        self.is_processing = True

        def worker():
            try:
                result = task_func(*args, **kwargs)
                self.processing_queue.put(('result', result))
            except Exception as e:
                self.processing_queue.put(('error', str(e)))
            finally:
                self.is_processing = False

        self.current_thread = threading.Thread(target=worker, daemon=True)
        self.current_thread.start()

        # 定期检查处理结果
        self.check_processing_result()

    def check_processing_result(self):
        """检查后台处理结果"""
        try:
            while not self.processing_queue.empty():
                status, result = self.processing_queue.get_nowait()
                if status == 'result':
                    self.on_processing_complete(result)
                elif status == 'error':
                    self.on_processing_error(result)
        except queue.Empty:
            pass

        if self.is_processing:
            self.root.after(50, self.check_processing_result)

    def stop_background_processing(self):
        """停止后台处理"""
        self.is_processing = False
        if self.current_thread and self.current_thread.is_alive():
            # 等待线程结束
            self.current_thread.join(timeout=0.1)

    def on_processing_complete(self, result):
        """处理完成回调"""
        # 子类可重写此方法
        pass

    def on_processing_error(self, error_msg):
        """处理错误回调"""
        messagebox.showerror("处理错误", f"图像处理失败: {error_msg}")

    def schedule_canvas_update(self, update_func, delay=100):
        """延迟执行画布更新以提高性能"""
        if self.canvas_update_timer:
            self.root.after_cancel(self.canvas_update_timer)
        self.canvas_update_timer = self.root.after(delay, update_func)

    def setup_gui(self):
        """
        设置主GUI布局

        配置网格布局，确保窗口在调整大小时各组件能够正确伸缩
        """
        # 创建主框架，带有现代化样式
        self.main_frame = ttk.Frame(self.root, padding="20", style='Modern.TFrame')
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重 - 确保图像区域能够随窗口缩放
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        # 设置图像框架行（row=2）的权重，使其能够随窗口垂直缩放
        self.main_frame.rowconfigure(2, weight=1)

        # 创建现代化样式
        self.create_modern_styles()

    def create_modern_styles(self):
        """
        创建现代化的样式主题

        定义应用程序的视觉样式，包括颜色、字体、间距等，
        创造更加美观和专业的用户界面
        """
        style = ttk.Style()

        # 设置整体主题色彩
        self.primary_color = '#2196F3'  # 主蓝色
        self.secondary_color = '#FFC107'  # 强调黄色
        self.background_color = '#f5f5f5'  # 背景色
        self.surface_color = '#ffffff'  # 表面色
        self.text_color = '#333333'  # 文字色

        # 设置主题色彩供其他方法使用
        primary_color = self.primary_color
        secondary_color = self.secondary_color
        background_color = self.background_color
        surface_color = self.surface_color
        text_color = self.text_color

        # 配置标签框架样式
        style.configure('Modern.TLabelframe', background=surface_color, relief='flat', borderwidth=0)
        style.configure('Modern.TLabelframe.Label', background=surface_color, foreground=primary_color, font=('Arial', 12, 'bold'))

        # 配置按钮样式
        style.configure('Modern.TButton',
                       background=primary_color,
                       foreground='white',
                       font=('Arial', 10),
                       relief='flat',
                       padding=10)
        style.map('Modern.TButton',
                 background=[('active', '#1976D2'), ('pressed', '#1565C0')],
                 relief=[('pressed', 'sunken'), ('!pressed', 'flat')])

        # 配置标签样式
        style.configure('Modern.TLabel', background=surface_color, foreground=text_color, font=('Arial', 10))

        # 配置画布样式
        style.configure('Modern.TFrame', background=background_color)

        # 配置标题样式
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground=primary_color, background=background_color)

        # 配置状态栏样式
        style.configure('Status.TLabel', font=('Arial', 9), foreground='#666666', background=background_color)

    def create_widgets(self):
        """
        创建所有GUI组件

        构建用户界面的各个组件：
        - 标题区域（包含语言切换按钮）
        - 步骤信息显示区域
        - 图像显示画布
        - 导航控制按钮
        - 状态栏
        """
        # 标题和语言切换按钮行
        title_frame = ttk.Frame(self.main_frame, style='Modern.TFrame')
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        title_frame.columnconfigure(0, weight=1)

        title_label = ttk.Label(title_frame, text="OpenCV图像处理实验",
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, sticky=tk.W)

        # 语言切换按钮
        self.lang_button = ttk.Button(title_frame, text="中/EN", width=10, command=self.toggle_language,
                                    style='Modern.TButton')
        self.lang_button.grid(row=0, column=1, padx=(20, 0))

        # 步骤信息框架
        self.step_frame = ttk.LabelFrame(self.main_frame, text="当前步骤", padding="15", style='Modern.TLabelframe')
        self.step_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        self.step_frame.columnconfigure(0, weight=1)

        self.step_title_label = ttk.Label(self.step_frame, text="", style='Modern.TLabel')
        self.step_title_label.grid(row=0, column=0, sticky=tk.W)

        self.step_desc_label = ttk.Label(self.step_frame, text="", wraplength=1000, style='Modern.TLabel')
        self.step_desc_label.grid(row=1, column=0, sticky=tk.W, pady=(8, 0))

        # 图像显示框架
        self.image_frame = ttk.LabelFrame(self.main_frame, text="图像显示", padding="15", style='Modern.TLabelframe')
        self.image_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)

        # 创建用于图像显示的画布
        self.canvas = tk.Canvas(self.image_frame, bg=self.surface_color, height=500, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 滚动条样式
        scrollbar_style = ttk.Style()
        scrollbar_style.configure('Modern.Horizontal.TScrollbar', background=self.surface_color, troughcolor=self.background_color, borderwidth=0, arrowcolor=self.primary_color)
        scrollbar_style.configure('Modern.Vertical.TScrollbar', background=self.surface_color, troughcolor=self.background_color, borderwidth=0, arrowcolor=self.primary_color)

        h_scrollbar = ttk.Scrollbar(self.image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview, style='Modern.Horizontal.TScrollbar')
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        v_scrollbar = ttk.Scrollbar(self.image_frame, orient=tk.VERTICAL, command=self.canvas.yview, style='Modern.Vertical.TScrollbar')
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        # 导航框架
        self.nav_frame = ttk.Frame(self.main_frame, style='Modern.TFrame')
        self.nav_frame.grid(row=3, column=0, pady=(20, 0))

        # 导航按钮
        nav_button_frame = ttk.Frame(self.nav_frame, style='Modern.TFrame')
        nav_button_frame.grid(row=0, column=0, columnspan=4)

        self.prev_button = ttk.Button(nav_button_frame, text="上一步", command=self.previous_step, style='Modern.TButton')
        self.prev_button.grid(row=0, column=0, padx=(0, 15))

        self.select_button = ttk.Button(nav_button_frame, text="选择图片", command=self.select_image, style='Modern.TButton')
        self.select_button.grid(row=0, column=1, padx=(0, 15))

        self.next_button = ttk.Button(nav_button_frame, text="下一步 ■", command=self.next_step, style='Modern.TButton')
        self.next_button.grid(row=0, column=2, padx=(0, 15))

        self.step_label = ttk.Label(nav_button_frame, text="步骤 0/7", style='Modern.TLabel')
        self.step_label.grid(row=0, column=3, padx=(30, 0))

        # 状态栏
        self.status_label = ttk.Label(self.main_frame, text="准备开始，请选择一张图片。",
                                   style='Status.TLabel')
        self.status_label.grid(row=4, column=0, pady=(15, 0))

        # Initialize UI state
        self.update_ui_state()

    def toggle_language(self):
        """
        切换界面语言

        在中文和英文之间切换应用程序界面语言
        中文界面适合中文用户，英文界面避免编码问题
        """
        self.current_language = "en" if self.current_language == "zh" else "zh"
        self.update_language()

    def update_language(self):
        """
        根据当前语言更新所有UI文本

        更新界面上的所有文本元素以匹配当前选择的语言：
        - 窗口标题
        - 按钮文本
        - 标签文本
        - 状态信息
        - 步骤描述
        """
        texts = self.text_resources[self.current_language]

        # Update window title
        self.root.title(texts["app_title"])

        # Update frame titles
        self.step_frame.config(text=texts["current_step"])
        self.image_frame.config(text=texts["image_display"])

        # Update buttons
        self.select_button.config(text=texts["select_image"])
        self.prev_button.config(text=texts["previous"])
        self.next_button.config(text=texts["next"])
        self.lang_button.config(text="中/EN")

        # Update status
        if self.current_image_path:
            self.status_label.config(text=f"{texts['status_loaded']}{os.path.basename(self.current_image_path)}")
        elif self.current_step == 0:
            self.status_label.config(text=texts["status_ready"])
        else:
            self.status_label.config(text=texts["status_no_selection"])

        # Update step info
        self.update_ui_state()

    def select_image(self):
        """
        选择图像文件

        打开文件选择对话框，让用户选择要处理的图像文件
        支持多种常见图像格式
        """
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("TIFF files", "*.tiff"),
                ("GIF files", "*.gif"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = file_path
            self.load_image()
            self.current_step = 1
            self.update_ui_state()
            self.show_current_step()
            self.update_language()  # Update status text with current language
        else:
            self.status_label.config(text=self.text_resources[self.current_language]["status_no_selection"])

    def load_image(self):
        """
        使用OpenCV加载选定的图像（性能优化版）

        从文件路径读取图像，支持多种图像格式。
        如果加载失败，创建一个默认的备用图像并显示警告。
        使用缓存机制避免重复加载相同图像。
        """
        # 检查缓存
        cache_key = self.current_image_path
        cached_image = self.get_cached_image(cache_key)
        if cached_image is not None:
            self.original_image = cached_image.copy()
            return

        try:
            # 优化的图像加载
            self.original_image = cv2.imread(self.current_image_path, cv2.IMREAD_COLOR)
            if self.original_image is None:
                raise Exception("Failed to load image")

            # 缓存原始图像
            self.cache_image(self.original_image, cache_key)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            # Create a default image
            self.original_image = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.putText(self.original_image, 'Failed to load', (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    def update_ui_state(self):
        """Update UI elements based on current step"""
        # Update step label
        self.step_label.config(text=f"Step {self.current_step}/{self.total_steps}")

        # Update navigation buttons
        self.prev_button.config(state=tk.NORMAL if self.current_step > 1 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_step > 0 and self.current_step < self.total_steps else tk.DISABLED)
        self.select_button.config(state=tk.NORMAL if self.current_step == 0 else tk.DISABLED)

        # Update step info
        step_info = self.text_resources[self.current_language]["steps"].get(self.current_step, {})
        if step_info:
            self.step_title_label.config(text=step_info["title"])
            self.step_desc_label.config(text=step_info["description"])
        else:
            self.step_title_label.config(text="")
            self.step_desc_label.config(text="")

    def previous_step(self):
        """Go to previous step"""
        if self.current_step > 1:
            self.current_step -= 1
            self.update_ui_state()
            self.show_current_step()

    def next_step(self):
        """Go to next step"""
        if self.current_step < self.total_steps:
            self.current_step += 1
            self.update_ui_state()
            self.show_current_step()

    def show_current_step(self):
        """Display content for current step"""
        if self.original_image is None:
            return

        # Clear previous content
        self.clear_step_content()

        if self.current_step == 1:
            self.show_step_1()
        elif self.current_step == 2:
            self.show_step_2()
        elif self.current_step == 3:
            self.show_step_3()
        elif self.current_step == 4:
            self.show_step_4()
        elif self.current_step == 5:
            self.show_step_5()
        elif self.current_step == 6:
            self.show_step_6()
        elif self.current_step == 7:
            self.show_step_7()

    def clear_step_content(self):
        """Clear previous step content"""
        # Clear main canvas
        self.canvas.delete("all")

        # Clear step content frame if it exists
        if hasattr(self, 'step_content_frame'):
            self.step_content_frame.destroy()

        # Reset to original canvas layout
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def show_step_1(self):
        """
        显示步骤1：原始图像及其属性（性能优化版）

        在画布上显示原始图像，并标注图像的基本属性信息：
        - 图像尺寸（高x宽x通道数）
        - 像素格式

        使用缓存和优化的图像处理提高性能。
        """
        # 使用缓存优化图像显示
        cache_key = f"step1_{self.current_image_path}"
        cached_photo = self.photo_cache.get(cache_key)

        if cached_photo is None:
            # Convert OpenCV image to PIL format for Tkinter
            img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)

            # 自适应缩放图像，避免黑边
            max_width, max_height = 600, 400
            width, height = pil_image.size
            if width > max_width or height > max_height:
                # 计算缩放比例，保持纵横比
                ratio = min(max_width/width, max_height/height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to Tkinter format and cache
            self.photo = ImageTk.PhotoImage(pil_image)
            self.photo_cache[cache_key] = self.photo
        else:
            self.photo = cached_photo

        # 延迟执行画布更新以提高性能
        def update_canvas():
            # Display on canvas
            self.canvas.update_idletasks()  # 确保获取正确的画布尺寸
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # 如果画布尺寸仍然很小，使用默认值
            if canvas_width < 100:
                canvas_width = 800
            if canvas_height < 100:
                canvas_height = 500

            x = (canvas_width - self.photo.width()) // 2
            y = (canvas_height - self.photo.height()) // 2

            self.canvas.delete("all")  # 清除之前的内容
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)

            # Add text information
            info_text = f"Image Properties:\n"
            info_text += f"Shape: {self.original_image.shape}\n"
            info_text += f"Height: {self.original_image.shape[0]} pixels\n"
            info_text += f"Width: {self.original_image.shape[1]} pixels\n"
            info_text += f"Channels: {self.original_image.shape[2]} (BGR format)\n"
            info_text += f"Data type: {self.original_image.dtype}"

            self.canvas.create_text(20, 20, anchor=tk.NW, text=info_text,
                                   font=("Arial", 10), fill="white")

            # Update canvas scroll region - 确保滑动条正常工作
            self.canvas.update_idletasks()  # 确保布局完成
            bbox = self.canvas.bbox(tk.ALL)
            if bbox:
                self.canvas.config(scrollregion=bbox)
            else:
                # 如果没有内容，使用画布尺寸
                canvas_width = self.canvas.winfo_width() or 800
                canvas_height = self.canvas.winfo_height() or 500
                self.canvas.config(scrollregion=(0, 0, canvas_width, canvas_height))

        self.schedule_canvas_update(update_canvas, delay=50)

    def show_step_2(self):
        """Step 2: JPEG Quality Comparison with adjustable quality"""
        self.show_comparison_with_controls(self.image_labels["step2_title"],
                                        self.process_jpeg_quality_dynamic,
                                        self.image_labels["step2_label1"], self.image_labels["step2_label2"],
                                        "jpeg", 95, 10)

    def show_step_3(self):
        """Step 3: PNG Compression Comparison with adjustable compression"""
        self.show_comparison_with_controls(self.image_labels["step3_title"],
                                        self.process_png_compression_dynamic,
                                        self.image_labels["step3_label1"], self.image_labels["step3_label2"],
                                        "png", 0, 9)

    def show_step_4(self):
        """Step 4: Grayscale Image Split"""
        self.show_comparison(self.image_labels["step4_title"],
                           self.process_grayscale_split(),
                           self.image_labels["step4_label1"], self.image_labels["step4_label2"])

    def show_step_5(self):
        """Step 5: Image Creation and Copying"""
        self.show_comparison(self.image_labels["step5_title"],
                           self.process_image_creation(),
                           self.image_labels["step5_label1"], self.image_labels["step5_label2"])

    def show_step_6(self):
        """Step 6: Channel Separation - Adaptive Display"""
        self.show_channel_separation_adaptive()

    def show_channel_separation_adaptive(self):
        """
        自适应显示通道分离结果

        根据图像的通道数智能地显示分离结果：
        - 单通道灰度图：显示原图和分离结果
        - 3通道彩色图：显示原图和三个通道
        - 4通道图：显示原图和四个通道

        自动调整布局以适应不同数量的通道显示
        """
        if self.original_image is None:
            return

        # 检测图像通道数
        if len(self.original_image.shape) == 2:
            # 单通道灰度图
            channels = 1
            channel_names = ["Grayscale"]
        elif len(self.original_image.shape) == 3:
            channels = self.original_image.shape[2]
            if channels == 3:
                channel_names = ["Blue (B)", "Green (G)", "Red (R)"]
            elif channels == 4:
                channel_names = ["Blue (B)", "Green (G)", "Red (R)", "Alpha (A)"]
            else:
                channel_names = [f"Channel {i}" for i in range(channels)]
        else:
            # 无法处理的图像格式
            self.canvas.delete("all")
            self.canvas.create_text(400, 200, text="Unsupported image format for channel separation", font=("Arial", 12))
            return

        print(f"=== Channel Separation Analysis ===")
        print(f"Image shape: {self.original_image.shape}")
        print(f"Number of channels: {channels}")

        # 获取分离的通道
        separated_channels = self.get_separated_channels()

        # 创建多通道显示布局
        self.display_multi_channel_layout(separated_channels, channel_names)

    def get_separated_channels(self):
        """
        获取图像的所有分离通道

        Returns:
            list: 分离后的通道列表，每个通道都转换为BGR格式便于显示
        """
        if len(self.original_image.shape) == 2:
            # 单通道灰度图
            gray_bgr = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
            return [gray_bgr]
        elif len(self.original_image.shape) == 3:
            channels = self.original_image.shape[2]
            if channels >= 3:
                # 分离BGR通道
                b, g, r = cv2.split(self.original_image)
                result = [
                    cv2.cvtColor(b, cv2.COLOR_GRAY2BGR),  # Blue
                    cv2.cvtColor(g, cv2.COLOR_GRAY2BGR),  # Green
                    cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)   # Red
                ]
                # 如果有第四个通道（Alpha），也添加进来
                if channels >= 4:
                    _, _, _, a = cv2.split(self.original_image)
                    result.append(cv2.cvtColor(a, cv2.COLOR_GRAY2BGR))  # Alpha
                return result
            else:
                # 其他通道数的处理
                return [cv2.cvtColor(self.original_image[:, :, i], cv2.COLOR_GRAY2BGR)
                       for i in range(channels)]
        else:
            return []

    def _draw_channel_layout_deferred(self, channels, channel_names):
        """
        延迟执行的绘图函数，在UI渲染完成后才进行计算和绘制。
        这是解决Tkinter布局时序问题的关键。
        """
        # 清空之前的图像引用，防止内存泄漏
        if hasattr(self, '_photos'):
            self._photos.clear()
        else:
            self._photos = []

        # 此时调用winfo_width/height可以获取到正确的、最终的画布尺寸
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # 包含原图的总显示数量
        total_images = len(channels) + 1  # +1 for original image

        # 决定网格布局
        if total_images <= 2:
            rows, cols = 1, 2
        elif total_images <= 4:
            rows, cols = 2, 2
        elif total_images <= 6:
            rows, cols = 2, 3
        else:  # 最多显示8张图片
            rows, cols = 2, 4

        # 计算每个图片的显示大小
        margin = 30
        spacing = 20
        title_height = 40
        label_height = 30

        available_width = canvas_width - 2 * margin - (cols - 1) * spacing
        available_height = canvas_height - 2 * margin - title_height - (rows - 1) * spacing - rows * label_height

        # 基于可用空间计算图片尺寸，并保持宽高比
        if hasattr(self, 'original_image') and self.original_image is not None:
            orig_h, orig_w = self.original_image.shape[:2]
            if orig_w > 0 and orig_h > 0:
                ratio = orig_w / orig_h

                # 尝试基于宽度计算高度
                img_width = int(available_width / cols)
                img_height = int(img_width / ratio)

                # 如果计算出的高度过高，则基于高度重新计算宽度
                max_img_height = int(available_height / rows)
                if img_height > max_img_height:
                    img_height = max_img_height
                    img_width = int(img_height * ratio)

        # 限制最小尺寸
        if 'img_width' not in locals() or img_width < 80:
            img_width = 150
        if 'img_height' not in locals() or img_height < 80:
            img_height = 150

        # 计算内容区域的总大小
        content_width = cols * img_width + (cols - 1) * spacing
        content_height = title_height + rows * img_height + (rows - 1) * spacing + rows * label_height

        # 计算居中偏移量
        offset_x = max(margin, (canvas_width - content_width) // 2)
        offset_y = max(margin, (canvas_height - content_height) // 2)

        # 设置滚动区域
        scroll_width = max(canvas_width, content_width + 2 * margin)
        scroll_height = max(canvas_height, content_height + 2 * margin)
        self.canvas.config(scrollregion=(0, 0, scroll_width, scroll_height))

        # 创建标题
        title_text = f"Channel Separation ({len(channels)} channels)"
        title_x = scroll_width / 2
        self.canvas.create_text(title_x, offset_y, text=title_text,
                                font=("Arial", 16, "bold"), fill="black", anchor=tk.N)

        # 显示原图
        self.display_single_image_with_label(
            self.original_image, "Original",
            0, 0, img_width, img_height, offset_x, offset_y, spacing
        )

        # 显示分离的通道
        for i, (channel_img, channel_name) in enumerate(zip(channels, channel_names)):
            # +1是因为原图占了第0个位置
            item_index = i + 1
            row = item_index // cols
            col = item_index % cols

            self.display_single_image_with_label(
                channel_img, channel_name,
                row, col, img_width, img_height, offset_x, offset_y, spacing
            )

    def display_multi_channel_layout(self, channels, channel_names):
        """
        调度器函数：清空画布并延迟调用真正的绘图函数。
        """
        if not channels:
            return

        # 清空画布上的所有旧内容
        self.canvas.delete("all")

        # 安排_draw_channel_layout_deferred函数在10毫秒后执行
        # 这给了Tkinter足够的时间来完成UI渲染，确保画布尺寸正确
        self.root.after(10, self._draw_channel_layout_deferred, channels, channel_names)

    def display_single_image_with_label(self, img, label, row, col, img_width, img_height, offset_x, offset_y, spacing):
        """
        显示单个图像及其标签，使用新的偏移参数
        """
        # 计算位置
        title_height = 40  # 标题高度
        label_height = 30  # 标签高度

        x = offset_x + col * (img_width + spacing)
        y = offset_y + title_height + row * (img_height + spacing + label_height)

        # 转换图像格式
        if img is not None:
            # 保持宽高比进行缩放
            h, w = img.shape[:2]
            if w > 0 and h > 0:
                scale = min(img_width / w, img_height / h)
                new_w, new_h = int(w * scale), int(h * scale)

                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)

                # 创建带背景的画布以居中显示
                bg_color = (240, 240, 240)  # 浅灰色背景
                display_img = Image.new('RGB', (img_width, img_height), bg_color)

                paste_x = (img_width - new_w) // 2
                paste_y = (img_height - new_h) // 2
                display_img.paste(pil_img, (paste_x, paste_y))

                photo = ImageTk.PhotoImage(display_img)

                # 显示图像并保存引用
                self.canvas.create_image(x, y, anchor=tk.NW, image=photo)
                self._photos.append(photo)

        # 显示标签
        label_y = y + img_height + 5  # 标签紧跟在图片下方
        self.canvas.create_text(x + img_width // 2, label_y, text=label,
                                font=("Arial", 10, "bold"), fill="#333333", anchor=tk.N)  # 使用N锚点

    def show_step_7(self):
        """Step 7: Channel Merging & Color Spaces"""
        self.show_comparison(self.image_labels["step7_title"],
                           self.process_color_spaces(),
                           self.image_labels["step7_label1"], self.image_labels["step7_label2"])

    def show_comparison(self, title, images, label1, label2):
        """Display two images side by side for comparison (性能优化版)"""
        if len(images) != 2:
            return

        # 检查缓存
        cache_key = f"comparison_{title}_{hash(str(images))}_{label1}_{label2}"
        cached_photo = self.photo_cache.get(cache_key)

        if cached_photo is not None:
            self.photo = cached_photo
            # 直接显示缓存结果
            self.display_cached_comparison()
            return

        # 在后台处理图像以避免阻塞GUI
        self.process_image_in_background(self._process_comparison_images, title, images, label1, label2, cache_key)

    def _process_comparison_images(self, title, images, label1, label2, cache_key):
        """在后台处理对比图像"""
        img1, img2 = images

        # Convert to PIL format
        try:
            if img1 is not None:
                img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                pil_img1 = Image.fromarray(img1_rgb)
            else:
                pil_img1 = Image.new('RGB', (300, 200), color='gray')

            if img2 is not None:
                img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                pil_img2 = Image.fromarray(img2_rgb)
            else:
                pil_img2 = Image.new('RGB', (300, 200), color='gray')

            # 自适应缩放图像，避免黑边
            max_width, max_height = 400, 300

            # 缩放第一张图像
            width, height = pil_img1.size
            if width > max_width or height > max_height:
                ratio = min(max_width/width, max_height/height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                pil_img1 = pil_img1.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 缩放第二张图像
            width, height = pil_img2.size
            if width > max_width or height > max_height:
                ratio = min(max_width/width, max_height/height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                pil_img2 = pil_img2.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Create combined image
            combined_width = pil_img1.width + pil_img2.width + 40  # 40px gap
            combined_height = max(pil_img1.height, pil_img2.height) + 80  # Extra space for labels

            combined_img = Image.new('RGB', (combined_width, combined_height), color='white')

            # Paste images
            y_offset = 40
            combined_img.paste(pil_img1, (20, y_offset))
            combined_img.paste(pil_img2, (pil_img1.width + 40, y_offset))

            # Add labels
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(combined_img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            # Title
            bbox = draw.textbbox((0, 0), title, font=font)
            title_width = bbox[2] - bbox[0]
            draw.text(((combined_width - title_width) // 2, 10), title, fill='black', font=font)

            # Labels
            draw.text((20 + (pil_img1.width - len(label1)*10) // 2, y_offset + pil_img1.height + 10),
                     label1, fill='black', font=font)
            draw.text((pil_img1.width + 40 + (pil_img2.width - len(label2)*10) // 2, y_offset + pil_img2.height + 10),
                     label2, fill='black', font=font)

            return combined_img, cache_key

        except Exception as e:
            return f"Error: {str(e)}", None

    def on_processing_complete(self, result):
        """处理完成回调 - 显示对比图像"""
        if isinstance(result, tuple) and len(result) == 2:
            combined_img, cache_key = result
            if combined_img and not isinstance(combined_img, str):
                # Convert to Tkinter format
                self.photo = ImageTk.PhotoImage(combined_img)

                # 缓存结果
                if cache_key and len(self.photo_cache) < self.max_cache_size:
                    self.photo_cache[cache_key] = self.photo

                # 显示结果
                self.display_cached_comparison()
            else:
                # 错误情况
                self.canvas.create_text(400, 200, text=combined_img, font=("Arial", 12))
        else:
            self.canvas.create_text(400, 200, text=f"Processing error: {result}", font=("Arial", 12))

    def display_cached_comparison(self):
        """显示缓存的对比图像"""
        def update_display():
            canvas_width = self.canvas.winfo_width() or 800
            canvas_height = self.canvas.winfo_height() or 600

            x = max(0, (canvas_width - self.photo.width()) // 2)
            y = max(0, (canvas_height - self.photo.height()) // 2)

            self.canvas.delete("all")
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
            self.canvas.config(scrollregion=(0, 0, max(canvas_width, self.photo.width()),
                                           max(canvas_height, self.photo.height())))

        self.schedule_canvas_update(update_display, delay=30)

    def show_comparison_with_controls(self, title, process_func, label1, label2, control_type, default_val1, default_val2):
        """
        显示带控制的图像对比

        创建左右分栏布局：
        - 左侧：图像对比显示区域
        - 右侧：质量参数控制面板（滑块+标签+重置按钮）

        用于步骤2和步骤3，提供交互式的参数调整功能
        """
        # Create a main frame for this step
        self.step_content_frame = ttk.Frame(self.image_frame)
        self.step_content_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.step_content_frame.columnconfigure(0, weight=3)  # Image area takes more space
        self.step_content_frame.columnconfigure(1, weight=1)  # Controls take less space
        self.step_content_frame.rowconfigure(0, weight=1)

        # Image canvas (left side)
        self.image_canvas = tk.Canvas(self.step_content_frame, bg='lightgray', height=400)
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Control panel (right side)
        control_title = "质量控制" if self.current_language == "zh" else "Quality Controls"
        self.control_frame = ttk.LabelFrame(self.step_content_frame, text=control_title, padding="10")
        self.control_frame.grid(row=0, column=1, sticky=(tk.N, tk.S), padx=(0, 10))
        self.control_frame.columnconfigure(0, weight=1)

        # Quality sliders
        settings_text = f"{control_type.upper()} 设置" if self.current_language == "zh" else f"{control_type.upper()} Settings"
        ttk.Label(self.control_frame, text=settings_text, font=("Arial", 10, "bold")).grid(row=0, column=0, pady=(0, 10))

        # Slider 1
        ttk.Label(self.control_frame, text=f"{label1}:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.slider1_var = tk.IntVar(value=default_val1)
        self.slider1 = ttk.Scale(self.control_frame, from_=0, to=100 if control_type == "jpeg" else 9,
                                orient=tk.HORIZONTAL, variable=self.slider1_var, command=self.update_comparison)
        self.slider1.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        self.slider1_label = ttk.Label(self.control_frame, text=f"Value: {default_val1}")
        self.slider1_label.grid(row=3, column=0, sticky=tk.W, pady=(0, 15))

        # Slider 2
        ttk.Label(self.control_frame, text=f"{label2}:").grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        self.slider2_var = tk.IntVar(value=default_val2)
        self.slider2 = ttk.Scale(self.control_frame, from_=0, to=100 if control_type == "jpeg" else 9,
                                orient=tk.HORIZONTAL, variable=self.slider2_var, command=self.update_comparison)
        self.slider2.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        self.slider2_label = ttk.Label(self.control_frame, text=f"Value: {default_val2}")
        self.slider2_label.grid(row=6, column=0, sticky=tk.W, pady=(0, 15))

        # Reset button
        reset_text = "重置为默认值" if self.current_language == "zh" else "Reset to Default"
        ttk.Button(self.control_frame, text=reset_text,
                  command=lambda: self.reset_sliders(default_val1, default_val2)).grid(row=7, column=0, pady=(10, 0))

        # Store parameters for dynamic updates
        self.current_process_func = process_func
        self.current_title = title
        self.current_label1 = label1
        self.current_label2 = label2
        self.control_type = control_type

        # Store initial canvas size for consistent positioning
        self.canvas_width = self.image_canvas.winfo_width()
        self.canvas_height = self.image_canvas.winfo_height()

        # Initial display
        self.update_comparison()

    def update_comparison(self, event=None):
        """
        当滑块值改变时更新对比显示

        实时响应用户的滑块操作：
        - 更新滑块标签显示当前值
        - 调用处理函数生成新的对比图像
        - 刷新显示画布

        Args:
            event: Tkinter事件对象（滑块回调参数）
        """
        if not hasattr(self, 'current_process_func'):
            return

        # Update slider labels
        self.slider1_label.config(text=f"Value: {self.slider1_var.get()}")
        self.slider2_label.config(text=f"Value: {self.slider2_var.get()}")

        # Process images with current slider values
        try:
            val1, val2 = self.slider1_var.get(), self.slider2_var.get()
            images = self.current_process_func(val1, val2)

            # Display the comparison
            self.display_comparison_on_canvas(self.current_title, images,
                                            self.current_label1, self.current_label2,
                                            center_for_controls=True)
        except Exception as e:
            self.image_canvas.delete("all")
            self.image_canvas.create_text(200, 200, text=f"Error: {str(e)}", font=("Arial", 10))

    def display_comparison_on_canvas(self, title, images, label1, label2, center_for_controls=False):
        """
        在专用画布上显示对比图像

        将两张图像合并显示在同一个画布上，包含标题和标签。
        支持两种显示模式：
        - 普通模式：用于常规步骤
        - 控制模式：用于步骤2和3，优化左侧画布的空间利用

        Args:
            center_for_controls: 是否为带控制的步骤优化显示位置
        """
        if len(images) != 2:
            return

        img1, img2 = images

        # Convert to PIL format
        try:
            if img1 is not None:
                img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                pil_img1 = Image.fromarray(img1_rgb)
            else:
                pil_img1 = Image.new('RGB', (300, 200), color='gray')

            if img2 is not None:
                img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                pil_img2 = Image.fromarray(img2_rgb)
            else:
                pil_img2 = Image.new('RGB', (300, 200), color='gray')

            # 自适应缩放图像，避免黑边
            if center_for_controls:
                # 带控制面板的步骤使用较小图像
                max_width, max_height = 350, 250
            else:
                max_width, max_height = 400, 300

            # 缩放第一张图像
            width, height = pil_img1.size
            if width > max_width or height > max_height:
                ratio = min(max_width/width, max_height/height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                pil_img1 = pil_img1.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 缩放第二张图像
            width, height = pil_img2.size
            if width > max_width or height > max_height:
                ratio = min(max_width/width, max_height/height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                pil_img2 = pil_img2.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Create combined image
            combined_width = pil_img1.width + pil_img2.width + 40
            combined_height = max(pil_img1.height, pil_img2.height) + 80

            combined_img = Image.new('RGB', (combined_width, combined_height), color='white')

            # Paste images
            y_offset = 40
            combined_img.paste(pil_img1, (20, y_offset))
            combined_img.paste(pil_img2, (pil_img1.width + 40, y_offset))

            # Add labels
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(combined_img)
            try:
                if center_for_controls:
                    font = ImageFont.truetype("arial.ttf", 12)  # Smaller font for controls
                else:
                    font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            # Title
            bbox = draw.textbbox((0, 0), title, font=font)
            title_width = bbox[2] - bbox[0]
            draw.text(((combined_width - title_width) // 2, 10), title, fill='black', font=font)

            # Labels
            if center_for_controls:
                # Adjust label positioning for smaller images
                char_width = 6  # Approximate character width for smaller font
            else:
                char_width = 8

            draw.text((20 + (pil_img1.width - len(label1)*char_width) // 2, y_offset + pil_img1.height + 5),
                     label1, fill='black', font=font)
            draw.text((pil_img1.width + 40 + (pil_img2.width - len(label2)*char_width) // 2, y_offset + pil_img2.height + 5),
                     label2, fill='black', font=font)

            # Convert to Tkinter format and display
            self.photo = ImageTk.PhotoImage(combined_img)

            # Clear canvas and display
            self.image_canvas.delete("all")

            # Use stored canvas size for consistent positioning
            # 如果没有存储的尺寸，强制更新并获取真实尺寸
            if not hasattr(self, 'canvas_width') or self.canvas_width < 100:
                self.image_canvas.update_idletasks()
                self.canvas_width = self.image_canvas.winfo_width()
                self.canvas_height = self.image_canvas.winfo_height()

                # 如果仍然很小，使用默认值
                if self.canvas_width < 100:
                    self.canvas_width = 400
                if self.canvas_height < 100:
                    self.canvas_height = 300

            canvas_width = self.canvas_width
            canvas_height = self.canvas_height

            if center_for_controls:
                # Special positioning for steps 2 and 3 with controls
                # Move image to the right to account for control panel and center it better
                x = max(20, (canvas_width - combined_width) // 2 + 30)  # Shift right by 30px
                y = max(10, (canvas_height - combined_height) // 2 - 20)  # Shift up slightly
            else:
                x = max(0, (canvas_width - combined_width) // 2)
                y = max(0, (canvas_height - combined_height) // 2)

            self.image_canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
            self.image_canvas.config(scrollregion=(0, 0, max(canvas_width, combined_width), max(canvas_height, combined_height)))

        except Exception as e:
            self.image_canvas.delete("all")
            self.image_canvas.create_text(200, 200, text=f"Error: {str(e)}", font=("Arial", 10))

    def reset_sliders(self, val1, val2):
        """
        重置滑块到默认值

        将两个滑块都恢复到初始默认值，但不立即更新图像显示。
        用户需要手动拖动滑块才会看到图像变化，这样可以避免位置偏移问题。

        Args:
            val1: 第一个滑块的默认值
            val2: 第二个滑块的默认值
        """
        self.slider1_var.set(val1)
        self.slider2_var.set(val2)
        self.slider1_label.config(text=f"Value: {val1}")
        self.slider2_label.config(text=f"Value: {val2}")
        # 注意：这里不再调用update_comparison()，避免图片位置偏移
        # 用户需要手动拖动滑块来更新显示

    def process_jpeg_quality_dynamic(self, quality1, quality2):
        """
        动态处理JPEG质量对比

        根据指定的质量值生成两张不同质量的JPEG图像，
        用于对比质量参数对图像文件大小和视觉效果的影响

        Args:
            quality1: 第一张图像的质量值 (0-100)
            quality2: 第二张图像的质量值 (0-100)

        Returns:
            tuple: (img1, img2) 两张处理后的图像
        """
        # Save temporary files with specified quality values
        cv2.imwrite('temp_quality1.jpg', self.original_image, [cv2.IMWRITE_JPEG_QUALITY, quality1])
        cv2.imwrite('temp_quality2.jpg', self.original_image, [cv2.IMWRITE_JPEG_QUALITY, quality2])

        # Read back
        img1 = cv2.imread('temp_quality1.jpg')
        img2 = cv2.imread('temp_quality2.jpg')

        # Clean up temp files
        try:
            os.remove('temp_quality1.jpg')
            os.remove('temp_quality2.jpg')
        except:
            pass

        return img1, img2

    def process_png_compression_dynamic(self, compression1, compression2):
        """Process PNG compression comparison with dynamic values"""
        # Save temporary files with specified compression values
        cv2.imwrite('temp_compression1.png', self.original_image, [cv2.IMWRITE_PNG_COMPRESSION, compression1])
        cv2.imwrite('temp_compression2.png', self.original_image, [cv2.IMWRITE_PNG_COMPRESSION, compression2])

        # Read back
        img1 = cv2.imread('temp_compression1.png')
        img2 = cv2.imread('temp_compression2.png')

        # Clean up temp files
        try:
            os.remove('temp_compression1.png')
            os.remove('temp_compression2.png')
        except:
            pass

        return img1, img2

    def process_jpeg_quality(self):
        """Process JPEG quality comparison (legacy method)"""
        return self.process_jpeg_quality_dynamic(95, 10)

    def process_png_compression(self):
        """Process PNG compression comparison (legacy method)"""
        return self.process_png_compression_dynamic(0, 9)

    def process_image_creation(self):
        """Process image creation and copying"""
        # Create black image
        black_image = np.zeros(self.original_image.shape, np.uint8)
        return self.original_image, black_image

    def process_channel_separation(self):
        """Process channel separation"""
        if len(self.original_image.shape) == 3 and self.original_image.shape[2] == 3:
            b, g, r = cv2.split(self.original_image)
            # Convert blue channel to BGR for display
            b_bgr = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
            return self.original_image, b_bgr
        else:
            return self.original_image, None

    def process_grayscale_split(self):
        """
        处理灰度图像分离演示

        将彩色图像转换为灰度图像，然后使用cv2.split()函数进行分离。
        演示单通道图像与多通道图像在split()操作上的行为差异。

        关键教学点：
        - 灰度图像只有一个通道
        - cv2.split()对单通道图像的行为
        - 分离结果的分析和验证
        """
        print("\n=== Grayscale Image Split Demonstration ===")

        # Show original color image info
        print(f"Original color image shape: {self.original_image.shape}")

        # Convert original color image to grayscale
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        print(f"Converted to grayscale - shape: {gray_image.shape}")

        # Demonstrate the key concept: split() on single-channel images
        print("\nDemonstrating cv2.split() on grayscale (single-channel) image:")
        split_result = cv2.split(gray_image)
        print(f"cv2.split(gray_image) returns {len(split_result)} channel(s)")
        print(f"Each channel shape: {[img.shape for img in split_result]}")

        # Show that all channels are identical for grayscale
        if len(split_result) >= 1:
            print(f"All channels are identical: {np.array_equal(split_result[0], gray_image)}")
            print("This demonstrates that split() on grayscale images returns the same image in each channel")

        # For display purposes, convert back to BGR format
        if len(split_result) == 1:
            grayscale_split_bgr = cv2.cvtColor(split_result[0], cv2.COLOR_GRAY2BGR)
        else:
            grayscale_split_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        print("=== End Grayscale Split Demonstration ===\n")
        return self.original_image, grayscale_split_bgr

    def process_color_spaces(self):
        """Process color space conversion"""
        if len(self.original_image.shape) == 3 and self.original_image.shape[2] == 3:
            # BGR to RGB conversion
            rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            # Convert back to BGR for display in OpenCV
            rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            return self.original_image, rgb_bgr
        else:
            return self.original_image, None

    def cleanup(self):
        """清理资源，确保线程安全退出"""
        # 停止后台处理
        self.stop_background_processing()

        # 清理缓存
        self.clear_cache()

        # 取消计时器
        if self.canvas_update_timer:
            self.root.after_cancel(self.canvas_update_timer)

    def run(self):
        """
        启动应用程序

        进入Tkinter主事件循环，开始用户交互界面
        程序将持续运行直到用户关闭窗口
        """
        # 设置关闭事件处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 启动主循环
        self.root.mainloop()

    def on_closing(self):
        """窗口关闭事件处理"""
        self.cleanup()
        self.root.destroy()

# 主程序入口
if __name__ == "__main__":
    try:
        # 创建图像处理实验应用实例
        app = ImageProcessingExperiment()
        # 启动应用程序
        app.run()
    except Exception as e:
        print(f"程序启动失败: {e}")
        import traceback
        traceback.print_exc()