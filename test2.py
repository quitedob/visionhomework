# PyTorch MNIST手写数字识别实验 (高级GUI版本)
# 基于卷积神经网络的MNIST数据集分类实现 - 增强版Tkinter GUI

import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库
import torchvision  # PyTorch计算机视觉库
import matplotlib.pyplot as plt  # 数据可视化库
import tkinter as tk  # Python标准GUI库
from tkinter import ttk, messagebox, filedialog  # Tkinter子模块
from PIL import Image, ImageTk, ImageDraw  # Python图像处理库
import threading  # 多线程支持
import queue  # 线程安全队列
import time  # 时间处理
import matplotlib
matplotlib.use('TkAgg')  # 设置Matplotlib后端
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns  # 增强可视化
import weakref  # 弱引用缓存
from functools import lru_cache  # LRU缓存装饰器

from torchvision import datasets, transforms  # 数据集和变换操作
from torch.autograd import Variable  # 自动求导变量
from torch.utils.data import random_split  # 数据集分割
from tqdm import tqdm  # 进度条库

def setup_data():
    """
    设置MNIST数据集下载和预处理

    配置数据变换、下载数据集并进行数据分割以减少训练时间
    使用torchvision标准方式加载MNIST数据集
    """
    # 数据预处理变换：转换为Tensor并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为PyTorch张量
        transforms.Normalize(mean=[0.5], std=[0.5])  # 数据标准化
    ])

    # 使用torchvision标准方式下载训练数据集
    # download=True 表示如果本地没有就自动下载
    train_dataset = datasets.MNIST(
        root='./data',  # 数据存储路径
        train=True,  # 下载训练集
        download=True,  # 如果本地没有就下载
        transform=transform  # 应用数据变换
    )

    # 使用torchvision标准方式下载测试数据集
    # download=True 表示如果本地没有就自动下载
    test_dataset = datasets.MNIST(
        root='./data',  # 数据存储路径
        train=False,  # 下载测试集
        download=True,  # 如果本地没有就下载
        transform=transform  # 应用数据变换
    )

    # 为了一致性，重命名变量
    data_train = train_dataset
    data_test = test_dataset

    # 数据分割：减少数据集大小以加快训练速度
    data_train, _ = random_split(
        dataset=data_train,  # 原始训练集
        lengths=[1000, 59000],  # 分割为1000张训练，59000张丢弃
        generator=torch.Generator().manual_seed(0)  # 设置随机种子保证可重现
    )

    # 数据分割：减少测试集大小
    data_test, _ = random_split(
        dataset=data_test,  # 原始测试集
        lengths=[1000, 9000],  # 分割为1000张测试，9000张丢弃
        generator=torch.Generator().manual_seed(0)  # 设置随机种子保证可重现
    )

    return data_train, data_test

def create_data_loaders(data_train, data_test):
    """
    创建数据加载器

    将数据集打包成批次，便于模型训练和测试
    """
    # 训练数据加载器
    data_loader_train = torch.utils.data.DataLoader(
        dataset=data_train,  # 训练数据集
        batch_size=4,  # 每个批次4张图片
        shuffle=True  # 随机打乱数据顺序
    )

    # 测试数据加载器
    data_loader_test = torch.utils.data.DataLoader(
        dataset=data_test,  # 测试数据集
        batch_size=4,  # 每个批次4张图片
        shuffle=True  # 随机打乱数据顺序
    )

    return data_loader_train, data_loader_test

def preview_data(data_loader_train):
    """
    数据预览函数

    显示一个批次的训练数据及其标签
    """
    # 获取一个批次的数据
    images, labels = next(iter(data_loader_train))

    # 将批次图像转换为网格显示格式
    img = torchvision.utils.make_grid(images)
    img = img.numpy().transpose(1, 2, 0)  # 调整维度顺序

    # 反标准化以正确显示图像
    std = [0.5]
    mean = [0.5]
    img = img * std + mean

    # 打印图像标签
    print("批次图像标签:", [labels[i].item() for i in range(4)])

    # 显示图像
    plt.imshow(img)
    plt.title("MNIST训练数据预览")
    plt.show()

class MNIST_CNN(torch.nn.Module):
    """
    MNIST手写数字识别卷积神经网络

    采用现代化的CNN架构：两个卷积块（含批标准化）和两个全连接层
    包含批标准化层加速收敛，Dropout防止过拟合，学习率调度器优化训练
    """
    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # 卷积层：特征提取
        self.conv_layers = torch.nn.Sequential(
            # 第一卷积块
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # 1->64通道，步长2减小尺寸
            torch.nn.BatchNorm2d(64),  # 批标准化：加速收敛，提高稳定性
            torch.nn.ReLU(),  # 激活函数

            # 第二卷积块
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64->128通道，步长2减小尺寸
            torch.nn.BatchNorm2d(128),  # 批标准化：加速收敛，提高稳定性
            torch.nn.ReLU(),  # 激活函数
        )

        # 全连接层：分类
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(7 * 7 * 128, 512),  # 7x7x128 -> 512，减小全连接层尺寸
            torch.nn.ReLU(),  # 激活函数
            torch.nn.Dropout(p=0.5),  # Dropout防止过拟合，概率0.5
            torch.nn.Linear(512, 10)  # 512 -> 10个数字类别
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像张量 [batch_size, 1, 28, 28]

        Returns:
            输出分类结果 [batch_size, 10]
        """
        x = self.conv_layers(x)  # 卷积特征提取
        x = x.view(-1, 7 * 7 * 128)  # 展平为全连接层输入
        x = self.classifier(x)  # 分类输出
        return x

class MNIST_Experiment_GUI:
    """
    MNIST手写数字识别实验高级GUI主类

    提供交互式的MNIST分类实验演示，包括：
    - 数据预览和加载（增强可视化）
    - 模型训练进度显示（实时监控）
    - 训练结果可视化（损失曲线、准确率曲线）
    - 模型测试和预测结果展示（详细分析）
    - 数字手写识别功能
    - 模型保存和加载

    支持中英文界面切换，步骤化引导学习，现代化设计
    """
    def __init__(self):
        """
        初始化MNIST实验高级GUI应用程序

        创建Tkinter主窗口，初始化所有变量和数据结构，
        设置多语言支持，默认使用中文界面
        """
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("PyTorch MNIST手写数字识别实验 (高级版)")
        self.root.geometry("1400x900")  # 窗口大小：1400x900像素
        self.root.resizable(True, True)  # 允许调整窗口大小

        # 设置现代化的窗口样式
        self.root.configure(bg='#f5f5f5')

        # 初始化实例变量
        self.data_train = None  # 训练数据集
        self.data_test = None  # 测试数据集
        self.data_loader_train = None  # 训练数据加载器
        self.data_loader_test = None  # 测试数据加载器
        self.model = None  # 神经网络模型
        self.current_step = 0  # 当前实验步骤（0-6）
        self.total_steps = 6  # 总实验步骤数
        self.current_language = "zh"  # 当前语言，默认为中文
        self.is_training = False  # 训练状态标志

        # 高级功能相关变量
        self.training_history = {'loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}  # 训练历史
        self.processing_queue = queue.Queue()  # 处理队列
        self.current_thread = None  # 当前处理线程
        self.figure_cache = weakref.WeakValueDictionary()  # 图形缓存
        self.photo_cache = {}  # PIL图像缓存
        self.max_cache_size = 15  # 最大缓存大小
        self.digit_drawing = False  # 手写数字绘制状态
        self.digit_image = None  # 手写数字图像

        # 多语言文本资源
        self.text_resources = {
            "zh": {
                "app_title": "PyTorch MNIST手写数字识别实验",
                "current_step": "当前步骤",
                "load_data": "加载数据",
                "preview_data": "预览数据",
                "build_model": "搭建模型",
                "train_model": "训练模型",
                "test_model": "测试模型",
                "language": "语言",
                "image_display": "图像显示",
                "status_ready": "准备开始MNIST实验。",
                "status_data_loaded": "MNIST数据集已加载完成！",
                "status_data_preview": "数据预览完成。",
                "status_model_built": "卷积神经网络模型已搭建完成。",
                "status_training": "模型正在训练中...",
                "status_trained": "模型训练完成！",
                "status_testing": "正在测试模型...",
                "status_digit_recognition": "手写数字识别",
                "status_model_saved": "模型已保存",
                "status_model_loaded": "模型已加载",
                "steps": {
                    0: {
                        "title": "步骤1: 加载数据",
                        "description": "下载并加载MNIST手写数字数据集"
                    },
                    1: {
                        "title": "步骤2: 数据预览",
                        "description": "预览训练数据样本和标签分布"
                    },
                    2: {
                        "title": "步骤3: 搭建模型",
                        "description": "构建卷积神经网络模型"
                    },
                    3: {
                        "title": "步骤4: 训练模型",
                        "description": "训练神经网络并实时监控训练过程"
                    },
                    4: {
                        "title": "步骤5: 测试模型",
                        "description": "测试模型性能并可视化预测结果"
                    },
                    5: {
                        "title": "步骤6: 手写识别",
                        "description": "手写数字识别和模型保存/加载"
                    }
                },
                "tabs": {
                    "data": "数据管理",
                    "model": "模型管理",
                    "training": "训练监控",
                    "testing": "模型测试",
                    "drawing": "手写识别"
                },
                "controls": {
                    "save_model": "保存模型",
                    "load_model": "加载模型",
                    "clear_canvas": "清除画布",
                    "predict_digit": "识别数字",
                    "batch_size": "批次大小",
                    "learning_rate": "学习率",
                    "epochs": "训练轮数",
                    "dataset_size": "数据集大小"
                }
            },
            "en": {
                "app_title": "PyTorch MNIST Digit Recognition Experiment",
                "current_step": "Current Step",
                "load_data": "Load Data",
                "preview_data": "Preview Data",
                "build_model": "Build Model",
                "train_model": "Train Model",
                "test_model": "Test Model",
                "language": "Language",
                "image_display": "Image Display",
                "status_ready": "Ready to start MNIST experiment.",
                "status_data_loaded": "MNIST dataset loaded successfully!",
                "status_data_preview": "Data preview completed.",
                "status_model_built": "CNN model built successfully.",
                "status_training": "Model training in progress...",
                "status_trained": "Model training completed!",
                "status_testing": "Testing model performance...",
                "status_digit_recognition": "Handwritten Digit Recognition",
                "status_model_saved": "Model saved successfully",
                "status_model_loaded": "Model loaded successfully",
                "steps": {
                    0: {
                        "title": "Step 1: Load Data",
                        "description": "Download and load MNIST dataset"
                    },
                    1: {
                        "title": "Step 2: Preview Data",
                        "description": "Preview training samples and label distribution"
                    },
                    2: {
                        "title": "Step 3: Build Model",
                        "description": "Build convolutional neural network"
                    },
                    3: {
                        "title": "Step 4: Train Model",
                        "description": "Train the network with real-time monitoring"
                    },
                    4: {
                        "title": "Step 5: Test Model",
                        "description": "Test model and visualize predictions"
                    },
                    5: {
                        "title": "Step 6: Digit Recognition",
                        "description": "Handwritten digit recognition and model save/load"
                    }
                },
                "tabs": {
                    "data": "Data Management",
                    "model": "Model Management",
                    "training": "Training Monitor",
                    "testing": "Model Testing",
                    "drawing": "Digit Drawing"
                },
                "controls": {
                    "save_model": "Save Model",
                    "load_model": "Load Model",
                    "clear_canvas": "Clear Canvas",
                    "predict_digit": "Recognize Digit",
                    "batch_size": "Batch Size",
                    "learning_rate": "Learning Rate",
                    "epochs": "Epochs",
                    "dataset_size": "Dataset Size"
                }
            }
        }

        self.setup_gui()
        self.create_widgets()
        self.update_language()  # Set initial language

    def setup_gui(self):
        """
        设置高级GUI布局（标签页界面）

        创建现代化的标签页界面，提供更好的用户体验
        """
        # 创建主框架，带有现代化样式
        self.main_frame = ttk.Frame(self.root, padding="10", style='Modern.TFrame')
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)

        # 创建现代化样式
        self.create_modern_styles()

        # 创建顶部标题栏
        self.create_header()

        # 创建标签页控制区域
        self.tab_control = ttk.Notebook(self.main_frame, style='Modern.TNotebook')
        self.tab_control.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))

        # 创建各个标签页
        self.create_data_tab()
        self.create_model_tab()
        self.create_training_tab()
        self.create_testing_tab()
        self.create_drawing_tab()

        # 创建底部状态栏
        self.create_status_bar()

    def create_header(self):
        """创建顶部标题栏"""
        header_frame = ttk.Frame(self.main_frame, style='Modern.TFrame')
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.columnconfigure(0, weight=1)

        # 标题
        title_label = ttk.Label(header_frame, text="PyTorch MNIST手写数字识别实验 (高级版)",
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, sticky=tk.W)

        # 语言切换按钮
        self.lang_button = ttk.Button(header_frame, text="中/EN", width=10,
                                    command=self.toggle_language, style='Modern.TButton')
        self.lang_button.grid(row=0, column=1, padx=(20, 0))

        # 步骤信息
        step_text = self.text_resources[self.current_language]["current_step"]
        self.step_frame = ttk.LabelFrame(header_frame, text=step_text, padding="10", style='Modern.TLabelframe')
        self.step_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

        self.step_title_label = ttk.Label(self.step_frame, text="", style='Modern.TLabel')
        self.step_title_label.grid(row=0, column=0, sticky=tk.W)

        self.step_desc_label = ttk.Label(self.step_frame, text="", wraplength=1200, style='Modern.TLabel')
        self.step_desc_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))

    def create_status_bar(self):
        """创建底部状态栏"""
        status_frame = ttk.Frame(self.main_frame, style='Modern.TFrame')
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        # 状态标签
        self.status_label = ttk.Label(status_frame, text="准备开始MNIST实验。",
                                   style='Status.TLabel')
        self.status_label.grid(row=0, column=0, sticky=tk.W)

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var,
                                          maximum=100, length=300)
        self.progress_bar.grid(row=0, column=1, padx=(20, 0))

        # 步骤标签
        self.step_label = ttk.Label(status_frame, text="步骤 0/6", style='Modern.TLabel')
        self.step_label.grid(row=0, column=2, padx=(20, 0))

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

        # 配置标签框架样式
        style.configure('Modern.TLabelframe', background=self.surface_color, relief='flat', borderwidth=0)
        style.configure('Modern.TLabelframe.Label', background=self.surface_color, foreground=self.primary_color, font=('Arial', 12, 'bold'))

        # 配置按钮样式
        style.configure('Modern.TButton',
                       background=self.primary_color,
                       foreground='white',
                       font=('Arial', 10),
                       relief='flat',
                       padding=10)
        style.map('Modern.TButton',
                 background=[('active', '#1976D2'), ('pressed', '#1565C0')],
                 relief=[('pressed', 'sunken'), ('!pressed', 'flat')])

        # 配置标签样式
        style.configure('Modern.TLabel', background=self.surface_color, foreground=self.text_color, font=('Arial', 10))

        # 配置画布样式
        style.configure('Modern.TFrame', background=self.background_color)

        # 配置标题样式
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground=self.primary_color, background=self.background_color)

        # 配置状态栏样式
        style.configure('Status.TLabel', font=('Arial', 9), foreground='#666666', background=self.background_color)

        # 配置标签页样式
        style.configure('Modern.TNotebook', background=self.background_color)
        style.configure('Modern.TNotebook.Tab', background=self.surface_color,
                       foreground=self.primary_color, padding=[12, 8], font=('Arial', 10, 'bold'))
        style.map('Modern.TNotebook.Tab',
                 background=[('selected', self.primary_color)],
                 foreground=[('selected', 'white')])

    def create_data_tab(self):
        """创建数据管理标签页"""
        self.data_tab = ttk.Frame(self.tab_control, style='Modern.TFrame')
        tab_text = self.text_resources[self.current_language]["tabs"]["data"]
        self.tab_control.add(self.data_tab, text=tab_text)

        # 数据管理控制面板
        control_frame = ttk.LabelFrame(self.data_tab, text="数据控制", padding="15", style='Modern.TLabelframe')
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

        # 加载数据按钮
        self.load_data_button = ttk.Button(control_frame, text="加载数据",
                                         command=self.load_data, style='Modern.TButton')
        self.load_data_button.grid(row=0, column=0, padx=(0, 15))

        # 数据预览按钮
        self.preview_button = ttk.Button(control_frame, text="预览数据",
                                       command=self.preview_data, style='Modern.TButton')
        self.preview_button.grid(row=0, column=1, padx=(0, 15))

        # 数据集大小控制
        ttk.Label(control_frame, text="数据集大小:", style='Modern.TLabel').grid(row=0, column=2, padx=(20, 5))
        self.dataset_size_var = tk.IntVar(value=1000)
        dataset_size_spin = ttk.Spinbox(control_frame, from_=100, to=10000, increment=500,
                                       textvariable=self.dataset_size_var, width=10)
        dataset_size_spin.grid(row=0, column=3, padx=(0, 15))

        # 数据可视化区域
        self.data_viz_frame = ttk.LabelFrame(self.data_tab, text="数据可视化", padding="15", style='Modern.TLabelframe')
        self.data_viz_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.data_viz_frame.columnconfigure(0, weight=1)
        self.data_viz_frame.rowconfigure(0, weight=1)

        # 创建matplotlib图形
        self.data_figure = Figure(figsize=(10, 6), dpi=80)
        self.data_canvas = FigureCanvasTkAgg(self.data_figure, self.data_viz_frame)
        self.data_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置数据标签页网格权重
        self.data_tab.columnconfigure(0, weight=1)
        self.data_tab.rowconfigure(1, weight=1)

    def create_model_tab(self):
        """创建模型管理标签页"""
        self.model_tab = ttk.Frame(self.tab_control, style='Modern.TFrame')
        tab_text = self.text_resources[self.current_language]["tabs"]["model"]
        self.tab_control.add(self.model_tab, text=tab_text)

        # 模型控制面板
        control_frame = ttk.LabelFrame(self.model_tab, text="模型控制", padding="15", style='Modern.TLabelframe')
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

        # 搭建模型按钮
        self.build_model_button = ttk.Button(control_frame, text="搭建模型",
                                           command=self.build_model, style='Modern.TButton')
        self.build_model_button.grid(row=0, column=0, padx=(0, 15))

        # 保存/加载模型按钮
        self.save_model_button = ttk.Button(control_frame, text="保存模型",
                                          command=self.save_model, style='Modern.TButton')
        self.save_model_button.grid(row=0, column=1, padx=(0, 15))

        self.load_model_button = ttk.Button(control_frame, text="加载模型",
                                          command=self.load_model, style='Modern.TButton')
        self.load_model_button.grid(row=0, column=2, padx=(0, 15))

        # 模型参数设置
        params_frame = ttk.LabelFrame(self.model_tab, text="模型参数", padding="15", style='Modern.TLabelframe')
        params_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

        # 学习率
        ttk.Label(params_frame, text="学习率:", style='Modern.TLabel').grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.learning_rate_var = tk.DoubleVar(value=0.001)
        lr_spin = ttk.Spinbox(params_frame, from_=0.0001, to=0.1, increment=0.0001,
                             textvariable=self.learning_rate_var, width=10, format="%.4f")
        lr_spin.grid(row=0, column=1, sticky=tk.W, padx=(0, 30))

        # 批次大小
        ttk.Label(params_frame, text="批次大小:", style='Modern.TLabel').grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.batch_size_var = tk.IntVar(value=32)
        batch_spin = ttk.Spinbox(params_frame, from_=8, to=128, increment=8,
                                textvariable=self.batch_size_var, width=10)
        batch_spin.grid(row=0, column=3, sticky=tk.W, padx=(0, 30))

        # 训练轮数
        ttk.Label(params_frame, text="训练轮数:", style='Modern.TLabel').grid(row=0, column=4, sticky=tk.W, padx=(0, 10))
        self.epochs_var = tk.IntVar(value=5)
        epochs_spin = ttk.Spinbox(params_frame, from_=1, to=50, increment=1,
                                textvariable=self.epochs_var, width=10)
        epochs_spin.grid(row=0, column=5, sticky=tk.W)

        # 模型信息显示区域
        self.model_info_frame = ttk.LabelFrame(self.model_tab, text="模型信息", padding="15", style='Modern.TLabelframe')
        self.model_info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.model_info_frame.columnconfigure(0, weight=1)
        self.model_info_frame.rowconfigure(0, weight=1)

        # 模型信息文本框
        self.model_info_text = tk.Text(self.model_info_frame, height=15, wrap=tk.WORD, font=('Consolas', 10))
        model_scrollbar = ttk.Scrollbar(self.model_info_frame, orient=tk.VERTICAL, command=self.model_info_text.yview)
        self.model_info_text.configure(yscrollcommand=model_scrollbar.set)

        self.model_info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        model_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # 配置模型标签页网格权重
        self.model_tab.columnconfigure(0, weight=1)
        self.model_tab.rowconfigure(2, weight=1)

    def create_training_tab(self):
        """创建训练监控标签页"""
        self.training_tab = ttk.Frame(self.tab_control, style='Modern.TFrame')
        tab_text = self.text_resources[self.current_language]["tabs"]["training"]
        self.tab_control.add(self.training_tab, text=tab_text)

        # 训练控制面板
        control_frame = ttk.LabelFrame(self.training_tab, text="训练控制", padding="15", style='Modern.TLabelframe')
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

        # 开始训练按钮
        self.train_button = ttk.Button(control_frame, text="训练模型",
                                     command=self.start_training, style='Modern.TButton')
        self.train_button.grid(row=0, column=0, padx=(0, 15))

        # 停止训练按钮
        self.stop_train_button = ttk.Button(control_frame, text="停止训练",
                                          command=self.stop_training, style='Modern.TButton', state=tk.DISABLED)
        self.stop_train_button.grid(row=0, column=1, padx=(0, 15))

        # 训练状态显示
        self.training_status_label = ttk.Label(control_frame, text="未开始训练", style='Modern.TLabel')
        self.training_status_label.grid(row=0, column=2, padx=(30, 0))

        # 训练监控区域
        self.training_viz_frame = ttk.LabelFrame(self.training_tab, text="训练监控", padding="15", style='Modern.TLabelframe')
        self.training_viz_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.training_viz_frame.columnconfigure(0, weight=1)
        self.training_viz_frame.rowconfigure(0, weight=1)

        # 创建训练监控图形
        self.training_figure = Figure(figsize=(12, 8), dpi=80)
        self.training_canvas = FigureCanvasTkAgg(self.training_figure, self.training_viz_frame)
        self.training_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置训练标签页网格权重
        self.training_tab.columnconfigure(0, weight=1)
        self.training_tab.rowconfigure(1, weight=1)

    def create_testing_tab(self):
        """创建模型测试标签页"""
        self.testing_tab = ttk.Frame(self.tab_control, style='Modern.TFrame')
        tab_text = self.text_resources[self.current_language]["tabs"]["testing"]
        self.tab_control.add(self.testing_tab, text=tab_text)

        # 测试控制面板
        control_frame = ttk.LabelFrame(self.testing_tab, text="测试控制", padding="15", style='Modern.TLabelframe')
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

        # 开始测试按钮
        self.test_button = ttk.Button(control_frame, text="测试模型",
                                    command=self.test_model, style='Modern.TButton')
        self.test_button.grid(row=0, column=0, padx=(0, 15))

        # 批量测试按钮
        self.batch_test_button = ttk.Button(control_frame, text="批量测试",
                                          command=self.batch_test_model, style='Modern.TButton')
        self.batch_test_button.grid(row=0, column=1, padx=(0, 15))

        # 测试结果显示
        self.test_result_label = ttk.Label(control_frame, text="", style='Modern.TLabel')
        self.test_result_label.grid(row=0, column=2, padx=(30, 0))

        # 测试结果可视化区域
        self.testing_viz_frame = ttk.LabelFrame(self.testing_tab, text="测试结果", padding="15", style='Modern.TLabelframe')
        self.testing_viz_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.testing_viz_frame.columnconfigure(0, weight=1)
        self.testing_viz_frame.rowconfigure(0, weight=1)

        # 创建测试结果显示画布
        self.testing_canvas = tk.Canvas(self.testing_viz_frame, bg=self.surface_color, highlightthickness=0)
        self.testing_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置测试标签页网格权重
        self.testing_tab.columnconfigure(0, weight=1)
        self.testing_tab.rowconfigure(1, weight=1)

    def create_drawing_tab(self):
        """创建手写识别标签页"""
        self.drawing_tab = ttk.Frame(self.tab_control, style='Modern.TFrame')
        tab_text = self.text_resources[self.current_language]["tabs"]["drawing"]
        self.tab_control.add(self.drawing_tab, text=tab_text)

        # 绘图控制面板
        control_frame = ttk.LabelFrame(self.drawing_tab, text="绘图控制", padding="15", style='Modern.TLabelframe')
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

        # 识别按钮
        self.predict_button = ttk.Button(control_frame, text="识别数字",
                                       command=self.predict_digit, style='Modern.TButton')
        self.predict_button.grid(row=0, column=0, padx=(0, 15))

        # 清除画布按钮
        self.clear_drawing_button = ttk.Button(control_frame, text="清除画布",
                                             command=self.clear_drawing, style='Modern.TButton')
        self.clear_drawing_button.grid(row=0, column=1, padx=(0, 15))

        # 预测结果显示
        self.prediction_label = ttk.Label(control_frame, text="请绘制一个数字", style='Modern.TLabel', font=('Arial', 12, 'bold'))
        self.prediction_label.grid(row=0, column=2, padx=(30, 0))

        # 手写画布
        self.drawing_frame = ttk.LabelFrame(self.drawing_tab, text="手写区域", padding="15", style='Modern.TLabelframe')
        self.drawing_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.drawing_frame.columnconfigure(0, weight=1)
        self.drawing_frame.rowconfigure(0, weight=1)

        # 创建绘图画布
        self.drawing_canvas = tk.Canvas(self.drawing_frame, width=280, height=280, bg='white', highlightthickness=2)
        self.drawing_canvas.grid(row=0, column=0)

        # 绑定鼠标事件
        self.drawing_canvas.bind('<Button-1>', self.start_drawing)
        self.drawing_canvas.bind('<B1-Motion>', self.draw)
        self.drawing_canvas.bind('<ButtonRelease-1>', self.stop_drawing)

        # 配置手写标签页网格权重
        self.drawing_tab.columnconfigure(0, weight=1)
        self.drawing_tab.rowconfigure(1, weight=1)

    def create_widgets(self):
        """
        创建所有GUI组件（由标签页系统替代）

        现代化标签页界面已在setup_gui中创建，此方法仅初始化UI状态
        """
        # 初始化UI状态
        self.update_ui_state()

    # 高级功能方法
    def clear_cache(self):
        """清理缓存以释放内存"""
        self.figure_cache.clear()
        self.photo_cache.clear()

    def process_in_background(self, task_func, *args, **kwargs):
        """在后台线程中处理任务以避免GUI阻塞"""
        if hasattr(self, 'current_thread') and self.current_thread and self.current_thread.is_alive():
            self.stop_background_processing()

        self.processing_queue = queue.Queue()

        def worker():
            try:
                result = task_func(*args, **kwargs)
                self.processing_queue.put(('result', result))
            except Exception as e:
                self.processing_queue.put(('error', str(e)))

        self.current_thread = threading.Thread(target=worker, daemon=True)
        self.current_thread.start()
        self.check_processing_result()

    def check_processing_result(self):
        """检查后台处理结果"""
        try:
            while not self.processing_queue.empty():
                status, result = self.processing_queue.get_nowait()
                if status == 'result':
                    self.on_processing_complete(result)
                elif status == 'error':
                    messagebox.showerror("错误", f"处理失败: {result}")
        except queue.Empty:
            pass

        if hasattr(self, 'current_thread') and self.current_thread and self.current_thread.is_alive():
            self.root.after(50, self.check_processing_result)

    def on_processing_complete(self, result):
        """处理完成回调（可重写）"""
        pass

    def stop_background_processing(self):
        """停止后台处理"""
        if hasattr(self, 'current_thread') and self.current_thread:
            self.current_thread.join(timeout=0.1)

    # 手写数字识别功能
    def start_drawing(self, event):
        """开始绘制"""
        self.digit_drawing = True
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        """绘制数字"""
        if self.digit_drawing:
            # 绘制粗线条
            self.drawing_canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=12, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE
            )
            self.last_x, self.last_y = event.x, event.y

    def stop_drawing(self, event):
        """停止绘制"""
        self.digit_drawing = False

    def clear_drawing(self):
        """清除画布"""
        self.drawing_canvas.delete("all")
        self.prediction_label.config(text="请绘制一个数字")

    def predict_digit(self):
        """识别手写数字"""
        if self.model is None:
            messagebox.showwarning("警告", "请先训练模型！")
            return

        # 获取画布图像
        try:
            # 创建PIL图像
            img = Image.new('L', (280, 280), 'white')
            draw = ImageDraw.Draw(img)

            # 从tkinter画布获取图像数据
            ps = self.drawing_canvas.postscript(colormode='gray')
            from PIL import ImageTk
            import io
            img_data = Image.open(io.BytesIO(ps.encode('utf-8').encode('latin-1')))

            # 转换为MNIST格式 (28x28)
            img = img_data.convert('L').resize((28, 28), Image.Resampling.LANCZOS)

            # 预处理图像
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = 1 - img_array  # 反转颜色
            img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)

            # 预测
            self.model.eval()
            with torch.no_grad():
                output = self.model(img_tensor)
                _, predicted = torch.max(output, 1)
                confidence = torch.softmax(output, dim=1)[0][predicted[0]].item() * 100

            self.prediction_label.config(
                text=f"预测结果: {predicted[0].item()} (置信度: {confidence:.1f}%)",
                foreground='green' if confidence > 80 else 'orange'
            )

        except Exception as e:
            messagebox.showerror("识别错误", f"手写识别失败: {str(e)}")

    # 模型保存和加载功能
    def save_model(self):
        """保存模型"""
        if self.model is None:
            messagebox.showwarning("警告", "没有可保存的模型！")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
        )

        if file_path:
            try:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'training_history': self.training_history,
                    'model_config': {
                        'learning_rate': self.learning_rate_var.get(),
                        'batch_size': self.batch_size_var.get(),
                        'epochs': self.epochs_var.get()
                    }
                }, file_path)

                self.status_label.config(text=self.text_resources[self.current_language]["status_model_saved"])
                messagebox.showinfo("成功", f"模型已保存到: {file_path}")
            except Exception as e:
                messagebox.showerror("保存失败", f"模型保存失败: {str(e)}")

    def load_model(self):
        """加载模型"""
        file_path = filedialog.askopenfilename(
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
        )

        if file_path:
            try:
                checkpoint = torch.load(file_path)

                # 创建新模型
                self.model = MNIST_CNN()
                self.model.load_state_dict(checkpoint['model_state_dict'])

                # 恢复训练历史
                if 'training_history' in checkpoint:
                    self.training_history = checkpoint['training_history']

                # 恢复模型配置
                if 'model_config' in checkpoint:
                    config = checkpoint['model_config']
                    self.learning_rate_var.set(config.get('learning_rate', 0.001))
                    self.batch_size_var.set(config.get('batch_size', 32))
                    self.epochs_var.set(config.get('epochs', 5))

                self.current_step = max(3, self.current_step)  # 至少到搭建模型步骤
                self.update_ui_state()
                self.status_label.config(text=self.text_resources[self.current_language]["status_model_loaded"])
                messagebox.showinfo("成功", f"模型已从 {file_path} 加载")
            except Exception as e:
                messagebox.showerror("加载失败", f"模型加载失败: {str(e)}")

    # 批量测试功能
    def batch_test_model(self):
        """批量测试模型"""
        if self.model is None:
            messagebox.showwarning("警告", "请先训练模型！")
            return

        self.process_in_background(self._batch_test_worker)

    def _batch_test_worker(self):
        """批量测试工作线程"""
        self.model.eval()
        correct = 0
        total = 0
        class_correct = list(10*[0])
        class_total = list(10*[0])

        with torch.no_grad():
            for data in self.data_loader_test:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                c = (predicted == labels)
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        accuracy = 100 * correct / total
        class_accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                           for i in range(10)]

        return {
            'overall_accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'confusion_matrix': self._create_confusion_matrix()
        }

    def _create_confusion_matrix(self):
        """创建混淆矩阵"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in self.data_loader_test:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())

        from sklearn.metrics import confusion_matrix
        return confusion_matrix(all_labels, all_preds)

    def on_processing_complete(self, result):
        """批量测试结果处理"""
        if isinstance(result, dict) and 'overall_accuracy' in result:
            self.display_batch_test_results(result)

    def display_batch_test_results(self, results):
        """显示批量测试结果"""
        # 切换到测试标签页
        self.tab_control.select(self.testing_tab)

        # 显示总体准确率
        self.test_result_label.config(
            text=f"总体准确率: {results['overall_accuracy']:.2f}%"
        )

        # 清除画布并绘制结果
        self.testing_canvas.delete("all")

        canvas_width = self.testing_canvas.winfo_width() or 1000
        canvas_height = self.testing_canvas.winfo_height() or 600

        # 绘制各类别准确率条形图
        class_accs = results['class_accuracies']
        digits = list(range(10))

        bar_width = 60
        max_height = 300
        x_start = 100
        y_base = canvas_height - 100

        for i, (digit, acc) in enumerate(zip(digits, class_accs)):
            x = x_start + i * (bar_width + 20)
            height = int((acc / 100) * max_height)
            y = y_base - height

            # 绘制条形
            color = '#4CAF50' if acc > 80 else '#FFC107' if acc > 60 else '#F44336'
            self.testing_canvas.create_rectangle(
                x, y, x + bar_width, y_base,
                fill=color, outline='black'
            )

            # 添加标签
            self.testing_canvas.create_text(
                x + bar_width // 2, y_base + 20,
                text=str(digit), font=("Arial", 12, "bold")
            )
            self.testing_canvas.create_text(
                x + bar_width // 2, y - 10,
                text=f"{acc:.1f}%", font=("Arial", 10)
            )

        # 添加标题
        self.testing_canvas.create_text(
            canvas_width // 2, 30,
            text="各类别准确率分布", font=("Arial", 16, "bold")
        )

        # 更新滚动区域
        self.testing_canvas.config(scrollregion=self.testing_canvas.bbox(tk.ALL))

    def toggle_language(self):
        """
        切换界面语言

        在中文和英文之间切换应用程序界面语言
        """
        self.current_language = "en" if self.current_language == "zh" else "zh"
        self.update_language()

    def update_language(self):
        """
        根据当前语言更新所有UI文本
        """
        texts = self.text_resources[self.current_language]

        # Update window title
        self.root.title(texts["app_title"])

        # Update frame titles
        self.step_frame.config(text=texts["current_step"])
        self.image_frame.config(text=texts["image_display"])

        # Update buttons
        self.load_data_button.config(text=texts["load_data"])
        self.preview_button.config(text=texts["preview_data"])
        self.build_model_button.config(text=texts["build_model"])
        self.train_button.config(text=texts["train_model"])
        self.test_button.config(text=texts["test_model"])
        self.lang_button.config(text="中/EN")

        # Update step info
        self.update_ui_state()

    def update_ui_state(self):
        """根据当前步骤更新UI状态"""
        # Update step label
        self.step_label.config(text=f"Step {self.current_step}/{self.total_steps}")

        # Update navigation buttons
        self.load_data_button.config(state=tk.NORMAL if self.current_step == 0 else tk.DISABLED)
        self.preview_button.config(state=tk.NORMAL if self.current_step >= 1 else tk.DISABLED)
        self.build_model_button.config(state=tk.NORMAL if self.current_step >= 1 else tk.DISABLED)
        self.train_button.config(state=tk.NORMAL if self.current_step >= 2 and not self.is_training else tk.DISABLED)
        self.test_button.config(state=tk.NORMAL if self.current_step >= 3 else tk.DISABLED)

        # Update step info
        step_info = self.text_resources[self.current_language]["steps"].get(self.current_step, {})
        if step_info:
            self.step_title_label.config(text=step_info["title"])
            self.step_desc_label.config(text=step_info["description"])
        else:
            self.step_title_label.config(text="")
            self.step_desc_label.config(text="")

    def load_data(self):
        """加载MNIST数据集（增强版）"""
        try:
            self.status_label.config(text="正在加载数据...")
            self.progress_var.set(10)

            # 获取数据集大小设置
            dataset_size = self.dataset_size_var.get()

            # 自定义数据加载函数
            def load_custom_data(size):
                # 数据预处理变换
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])

                # 下载完整数据集
                full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
                full_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

                # 分割为指定大小的数据集
                train_data, _ = random_split(
                    full_train, [size, len(full_train) - size],
                    generator=torch.Generator().manual_seed(42)
                )
                test_size = min(size, len(full_test))
                test_data, _ = random_split(
                    full_test, [test_size, len(full_test) - test_size],
                    generator=torch.Generator().manual_seed(42)
                )

                return train_data, test_data

            self.data_train, self.data_test = load_custom_data(dataset_size)

            # 使用用户设置的批次大小
            batch_size = self.batch_size_var.get()
            self.data_loader_train = torch.utils.data.DataLoader(
                self.data_train, batch_size=batch_size, shuffle=True
            )
            self.data_loader_test = torch.utils.data.DataLoader(
                self.data_test, batch_size=batch_size, shuffle=True
            )

            self.progress_var.set(100)
            self.status_label.config(text=self.text_resources[self.current_language]["status_data_loaded"])
            self.current_step = 1
            self.update_ui_state()

            # 显示数据统计
            self.display_data_statistics()
            messagebox.showinfo("成功", f"MNIST数据集加载完成！\n训练集: {len(self.data_train)} 张\n测试集: {len(self.data_test)} 张")

        except Exception as e:
            self.progress_var.set(0)
            messagebox.showerror("错误", f"数据加载失败: {str(e)}")

    def display_data_statistics(self):
        """显示数据统计信息"""
        if not self.data_train:
            return

        # 切换到数据标签页
        self.tab_control.select(self.data_tab)

        # 清除图形
        self.data_figure.clear()

        # 创建子图
        fig = self.data_figure
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. 数据样本预览
        ax1 = fig.add_subplot(gs[0, :])
        images, labels = next(iter(self.data_loader_train))
        img_grid = torchvision.utils.make_grid(images[:16], nrow=8, normalize=True)
        ax1.imshow(img_grid.permute(1, 2, 0))
        ax1.set_title(f"数据样本预览 (标签: {labels[:16].tolist()})")
        ax1.axis('off')

        # 2. 标签分布
        ax2 = fig.add_subplot(gs[1, 0])
        label_counts = {}
        for _, label in self.data_train:
            label_counts[label.item()] = label_counts.get(label.item(), 0) + 1

        digits = list(label_counts.keys())
        counts = list(label_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(digits)))

        bars = ax2.bar(digits, counts, color=colors)
        ax2.set_title("训练集标签分布")
        ax2.set_xlabel("数字")
        ax2.set_ylabel("数量")
        ax2.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')

        # 3. 数据集信息
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')

        info_text = f"""数据集信息:

训练集大小: {len(self.data_train)}
测试集大小: {len(self.data_test)}
批次大小: {self.batch_size_var.get()}
图像尺寸: 28×28 像素
通道数: 1 (灰度图)
数据范围: [-1, 1] (标准化后)

类别分布均衡性: {'是' if max(counts)/min(counts) < 1.2 else '否'}
最大/最小比例: {max(counts)/min(counts):.2f}"""

        ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        self.data_canvas.draw()

    def preview_data(self):
        """预览数据"""
        if self.data_loader_train is None:
            messagebox.showwarning("警告", "请先加载数据！")
            return

        try:
            self.status_label.config(text=self.text_resources[self.current_language]["status_data_preview"])
            self.display_data_preview()
            self.current_step = 2
            self.update_ui_state()
        except Exception as e:
            messagebox.showerror("错误", f"数据预览失败: {str(e)}")

    def build_model(self):
        """搭建模型"""
        try:
            self.status_label.config(text=self.text_resources[self.current_language]["status_model_built"])
            self.model = MNIST_CNN()
            self.display_model_info()
            self.current_step = 3
            self.update_ui_state()
            messagebox.showinfo("成功", "卷积神经网络模型搭建完成！")
        except Exception as e:
            messagebox.showerror("错误", f"模型搭建失败: {str(e)}")

    def start_training(self):
        """开始训练模型（增强版）"""
        if self.model is None:
            messagebox.showwarning("警告", "请先搭建模型！")
            return

        # 切换到训练监控标签页
        self.tab_control.select(self.training_tab)

        self.is_training = True
        self.training_status_label.config(text="训练中...")
        self.train_button.config(state=tk.DISABLED)
        self.stop_train_button.config(state=tk.NORMAL)
        self.update_ui_state()

        # 在后台线程中训练模型
        training_thread = threading.Thread(target=self.train_model_thread)
        training_thread.daemon = True
        training_thread.start()

    def stop_training(self):
        """停止训练"""
        self.is_training = False
        self.training_status_label.config(text="训练已停止")
        self.train_button.config(state=tk.NORMAL)
        self.stop_train_button.config(state=tk.DISABLED)

    def train_model_thread(self):
        """训练模型的线程函数（增强版）"""
        try:
            self.status_label.config(text=self.text_resources[self.current_language]["status_training"])
            self.progress_var.set(0)

            # 重置训练历史
            self.training_history = {'loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}

            # 获取训练参数
            num_epochs = self.epochs_var.get()
            learning_rate = self.learning_rate_var.get()

            # 增强的训练函数
            self._train_model_with_realtime_monitoring(num_epochs, learning_rate)

            self.progress_var.set(100)
            self.status_label.config(text=self.text_resources[self.current_language]["status_trained"])
            self.is_training = False
            self.current_step = 4
            self.training_status_label.config(text="训练完成")
            self.train_button.config(state=tk.NORMAL)
            self.stop_train_button.config(state=tk.DISABLED)
            self.update_ui_state()
            messagebox.showinfo("成功", "模型训练完成！")

        except Exception as e:
            self.is_training = False
            self.training_status_label.config(text="训练失败")
            self.train_button.config(state=tk.NORMAL)
            self.stop_train_button.config(state=tk.DISABLED)
            self.update_ui_state()
            messagebox.showerror("错误", f"训练失败: {str(e)}")

    def _train_model_with_realtime_monitoring(self, num_epochs, learning_rate):
        """带实时监控的训练函数"""
        # 定义损失函数和优化器
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        total_steps = num_epochs * len(self.data_loader_train)
        current_step = 0

        for epoch in range(num_epochs):
            if not self.is_training:
                break

            running_loss = 0.0
            running_correct = 0

            # 训练阶段
            self.model.train()
            for data in self.data_loader_train:
                if not self.is_training:
                    break

                current_step += 1
                progress = (current_step / total_steps) * 100
                self.root.after(0, self.update_progress, progress)

                X_train, y_train = data

                # 前向传播
                outputs = self.model(X_train)
                _, predicted = torch.max(outputs.data, 1)

                # 计算损失
                loss = criterion(outputs, y_train)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 累计统计
                running_loss += loss.data.item()
                running_correct += torch.sum(predicted == y_train.data).item()

            # 测试阶段
            self.model.eval()
            testing_correct = 0

            with torch.no_grad():
                for data in self.data_loader_test:
                    if not self.is_training:
                        break
                    X_test, y_test = data
                    outputs = self.model(X_test)
                    _, predicted = torch.max(outputs.data, 1)
                    testing_correct += torch.sum(predicted == y_test.data).item()

            # 计算指标
            train_accuracy = 100 * running_correct / len(self.data_loader_train.dataset)
            test_accuracy = 100 * testing_correct / len(self.data_loader_test.dataset)
            avg_loss = running_loss / len(self.data_loader_train.dataset)
            current_lr = optimizer.param_groups[0]['lr']

            # 记录训练历史
            self.training_history['loss'].append(avg_loss)
            self.training_history['train_acc'].append(train_accuracy)
            self.training_history['test_acc'].append(test_accuracy)
            self.training_history['lr'].append(current_lr)

            # 更新训练监控图表
            self.root.after(0, self.update_training_plots)

            print(f"Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%, LR: {current_lr:.6f}")

            scheduler.step()

    def update_progress(self, progress):
        """更新训练进度"""
        self.progress_var.set(progress)

    def update_training_plots(self):
        """更新训练监控图表"""
        if not self.training_history['loss']:
            return

        # 清除图形
        self.training_figure.clear()

        # 创建子图
        fig = self.training_figure
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        epochs = range(1, len(self.training_history['loss']) + 1)

        # 1. 损失曲线
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.training_history['loss'], 'b-', linewidth=2, label='训练损失')
        ax1.set_title('训练损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. 准确率曲线
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.training_history['train_acc'], 'g-', linewidth=2, label='训练准确率')
        ax2.plot(epochs, self.training_history['test_acc'], 'r-', linewidth=2, label='测试准确率')
        ax2.set_title('模型准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. 学习率曲线
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, self.training_history['lr'], 'orange', linewidth=2, label='学习率')
        ax3.set_title('学习率变化')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. 训练信息
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        if self.training_history['loss']:
            current_epoch = len(self.training_history['loss'])
            current_loss = self.training_history['loss'][-1]
            current_train_acc = self.training_history['train_acc'][-1]
            current_test_acc = self.training_history['test_acc'][-1]
            current_lr = self.training_history['lr'][-1]

            info_text = f"""训练进度信息:

当前Epoch: {current_epoch}
当前损失: {current_loss:.4f}
训练准确率: {current_train_acc:.2f}%
测试准确率: {current_test_acc:.2f}%
当前学习率: {current_lr:.6f}

最佳测试准确率: {max(self.training_history['test_acc']):.2f}%
损失趋势: {'下降' if len(self.training_history['loss']) > 1 and self.training_history['loss'][-1] < self.training_history['loss'][-2] else '上升'}"""

            ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        self.training_canvas.draw()

    def test_model(self):
        """测试模型"""
        if self.model is None:
            messagebox.showwarning("警告", "请先训练模型！")
            return

        try:
            self.status_label.config(text=self.text_resources[self.current_language]["status_testing"])
            self.display_test_results()
            self.current_step = 5
            self.update_ui_state()
        except Exception as e:
            messagebox.showerror("错误", f"测试失败: {str(e)}")

    def display_data_preview(self):
        """在画布上显示数据预览"""
        # 获取一个批次的数据
        images, labels = next(iter(self.data_loader_train))

        # 将批次图像转换为网格显示格式
        img = torchvision.utils.make_grid(images, nrow=4)
        img = img.numpy().transpose(1, 2, 0)  # 调整维度顺序

        # 反标准化以正确显示图像
        std = [0.5]
        mean = [0.5]
        img = img * std + mean

        # 转换为PIL图像
        img = (img * 255).astype(np.uint8)
        pil_image = Image.fromarray(img)

        # 调整图像大小以适应画布
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width < 100:
            canvas_width = 800
        if canvas_height < 100:
            canvas_height = 400

        # 保持宽高比缩放
        img_width, img_height = pil_image.size
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 转换为Tkinter格式
        self.photo = ImageTk.PhotoImage(pil_image)

        # 在画布上居中显示
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2

        self.canvas.delete("all")
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)

        # 添加标签信息
        label_text = f"MNIST Data Preview\nLabels: {[labels[i].item() for i in range(min(4, len(labels)))]}"
        self.canvas.create_text(canvas_width//2, y + new_height + 20, text=label_text,
                               font=("Arial", 12), fill="black", anchor=tk.N)

        # 更新画布滚动区域
        self.canvas.config(scrollregion=(0, 0, max(canvas_width, new_width), max(canvas_height, new_height + 40)))

    def display_model_info(self):
        """显示模型信息"""
        self.canvas.delete("all")

        canvas_width = self.canvas.winfo_width() or 800
        canvas_height = self.canvas.winfo_height() or 400

        # 显示模型结构信息
        model_info = "CNN Model Architecture:\n\n"
        model_info += "Conv Layer 1: 1 -> 64 channels (3x3, stride=2)\n"
        model_info += "Conv Layer 2: 64 -> 128 channels (3x3, stride=2)\n"
        model_info += "Fully Connected: 7*7*128 -> 512 -> 10\n"
        model_info += "Dropout: 0.8\n"
        model_info += "Total parameters: ~1.2M"

        self.canvas.create_text(canvas_width//2, canvas_height//2, text=model_info,
                               font=("Arial", 12), fill="black", anchor=tk.CENTER)

    def display_test_results(self):
        """显示测试结果"""
        # 获取一个测试批次
        X_test, y_test = next(iter(self.data_loader_test))
        inputs = X_test

        # 模型预测
        self.model.eval()
        with torch.no_grad():
            pred = self.model(inputs)
            _, predicted = torch.max(pred, 1)

        # 创建结果图像
        img = torchvision.utils.make_grid(X_test, nrow=4)
        img = img.numpy().transpose(1, 2, 0)

        # 反标准化
        std = [0.5]  # 使用与transform一致的单通道参数
        mean = [0.5]
        img = img * std + mean

        # 转换为PIL图像
        img = (img * 255).astype(np.uint8)
        pil_image = Image.fromarray(img)

        # 调整大小
        canvas_width = self.canvas.winfo_width() or 800
        canvas_height = self.canvas.winfo_height() or 400

        img_width, img_height = pil_image.size
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 转换为Tkinter格式
        self.photo = ImageTk.PhotoImage(pil_image)

        # 显示图像
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2

        self.canvas.delete("all")
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)

        # 添加预测结果
        true_labels = [y_test[i].item() for i in range(min(4, len(y_test)))]
        pred_labels = [predicted[i].item() for i in range(min(4, len(predicted)))]

        result_text = f"Test Results:\nTrue: {true_labels}\nPred: {pred_labels}"
        self.canvas.create_text(canvas_width//2, y + new_height + 20, text=result_text,
                               font=("Arial", 12), fill="black", anchor=tk.N)

        # 更新画布滚动区域
        self.canvas.config(scrollregion=(0, 0, max(canvas_width, new_width), max(canvas_height, new_height + 40)))

    def run(self):
        """
        启动应用程序

        进入Tkinter主事件循环
        """
        self.root.mainloop()

def train_model_with_progress(model, data_loader_train, data_loader_test, num_epochs=5, progress_callback=None):
    """
    带进度回调的训练函数
    使用学习率调度器优化训练过程
    """
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，设置初始学习率

    # 添加学习率调度器：每3个epoch降低学习率到原来的0.5倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    total_steps = num_epochs * len(data_loader_train)
    current_step = 0

    # 训练循环
    for epoch in range(num_epochs):
        running_loss = 0.0  # 累计损失
        running_correct = 0  # 累计正确预测数

        # 训练阶段
        model.train()  # 设置为训练模式

        for data in data_loader_train:
            current_step += 1
            progress = (current_step / total_steps) * 100
            if progress_callback:
                progress_callback(progress)

            X_train, y_train = data
            # PyTorch现代版本中Tensor已自动支持梯度计算，无需Variable

            # 前向传播
            outputs = model(X_train)
            _, predicted = torch.max(outputs.data, 1)

            # 计算损失
            loss = criterion(outputs, y_train)

            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 累计统计
            running_loss += loss.data.item()
            running_correct += torch.sum(predicted == y_train.data).item()

        # 测试阶段
        model.eval()  # 设置为评估模式
        testing_correct = 0

        with torch.no_grad():  # 不计算梯度
            for data in data_loader_test:
                X_test, y_test = data
                # PyTorch现代版本中Tensor已自动支持梯度计算，无需Variable

                outputs = model(X_test)
                _, predicted = torch.max(outputs.data, 1)
                testing_correct += torch.sum(predicted == y_test.data).item()

        # 输出训练结果
        train_accuracy = 100 * running_correct / len(data_loader_train.dataset)
        test_accuracy = 100 * testing_correct / len(data_loader_test.dataset)
        avg_loss = running_loss / len(data_loader_train.dataset)
        current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率

        print(f"Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%, LR: {current_lr:.6f}")

        # 每个epoch结束后更新学习率
        scheduler.step()

def train_model(model, data_loader_train, data_loader_test, num_epochs=5):
    """
    训练模型

    使用交叉熵损失和Adam优化器训练CNN模型，并使用学习率调度器优化训练

    Args:
        model: 要训练的模型
        data_loader_train: 训练数据加载器
        data_loader_test: 测试数据加载器
        num_epochs: 训练轮数
    """
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，设置初始学习率

    # 添加学习率调度器：每3个epoch降低学习率到原来的0.5倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # 训练循环
    for epoch in range(num_epochs):
        running_loss = 0.0  # 累计损失
        running_correct = 0  # 累计正确预测数

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # 训练阶段
        model.train()  # 设置为训练模式

        # 使用tqdm创建训练进度条
        train_pbar = tqdm(data_loader_train, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
                          unit="batch", leave=False)

        for data in train_pbar:
            X_train, y_train = data
            # PyTorch现代版本中Tensor已自动支持梯度计算，无需Variable

            # 前向传播
            outputs = model(X_train)
            _, predicted = torch.max(outputs.data, 1)

            # 计算损失
            loss = criterion(outputs, y_train)

            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 累计统计
            running_loss += loss.data.item()
            running_correct += torch.sum(predicted == y_train.data).item()

        # 测试阶段
        model.eval()  # 设置为评估模式
        testing_correct = 0

        with torch.no_grad():  # 不计算梯度
            for data in data_loader_test:
                X_test, y_test = data
                # PyTorch现代版本中Tensor已自动支持梯度计算，无需Variable

                outputs = model(X_test)
                _, predicted = torch.max(outputs.data, 1)
                testing_correct += torch.sum(predicted == y_test.data).item()

        # 输出训练结果
        train_accuracy = 100 * running_correct / len(data_loader_train.dataset)
        test_accuracy = 100 * testing_correct / len(data_loader_test.dataset)
        avg_loss = running_loss / len(data_loader_train.dataset)
        current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率

        print(f"损失: {avg_loss:.4f}, 训练准确率: {train_accuracy:.2f}%, 测试准确率: {test_accuracy:.2f}%, 学习率: {current_lr:.6f}")
        print()

        # 每个epoch结束后更新学习率
        scheduler.step()

def test_model(model, data_loader_test):
    """
    测试模型性能

    随机选择一个测试批次进行预测并可视化结果

    Args:
        model: 训练好的模型
        data_loader_test: 测试数据加载器
    """
    # 获取一个测试批次
    X_test, y_test = next(iter(data_loader_test))
    inputs = X_test

    # 模型预测
    model.eval()
    with torch.no_grad():
        pred = model(inputs)
        _, predicted = torch.max(pred, 1)

    # 输出预测结果
    print("预测标签:", [i.item() for i in predicted.data])
    print("真实标签:", [i.item() for i in y_test])

    # 可视化结果
    img = torchvision.utils.make_grid(X_test)
    img = img.numpy().transpose(1, 2, 0)

    # 反标准化
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img = img * std + mean

    # 显示图像
    plt.imshow(img)
    plt.title("模型预测结果可视化")
    plt.show()

def main():
    """
    主函数：启动GUI版本的MNIST分类实验

    创建并运行Tkinter GUI应用程序
    """
    try:
        # 创建GUI应用程序实例
        app = MNIST_Experiment_GUI()
        # 启动应用程序
        app.run()
    except Exception as e:
        # 如果GUI启动失败，回退到命令行版本
        print(f"GUI启动失败，使用命令行版本: {e}")
        print("=== PyTorch MNIST手写数字识别实验（命令行版本）===")
        print()

        try:
            # 1. 数据准备
            print("1. 准备数据...")
            data_train, data_test = setup_data()
            data_loader_train, data_loader_test = create_data_loaders(data_train, data_test)
            print("数据准备完成！")
            print()

            # 2. 数据预览
            print("2. 数据预览...")
            preview_data(data_loader_train)
            print()

            # 3. 模型搭建
            print("3. 搭建卷积神经网络...")
            model = MNIST_CNN()
            print("模型结构:")
            print(model)
            print()

            # 4. 模型训练
            print("4. 开始训练模型...")
            train_model(model, data_loader_train, data_loader_test, num_epochs=5)
            print()

            # 5. 模型测试
            print("5. 测试模型性能...")
            test_model(model, data_loader_test)
            print()

            print("=== 实验完成！ ===")
        except Exception as e2:
            print(f"命令行版本也运行失败: {e2}")
            import traceback
            traceback.print_exc()

# 程序入口
if __name__ == "__main__":
    main()
